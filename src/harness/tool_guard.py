"""
L3: 工具安全护栏 (Tool Safety Guard)

原理:
    LLM 对工具的选择和参数具有不确定性。ToolGuard 在 Agent 的 _execute_tool
    入口处建立一道确定性的防火墙，对所有工具调用做五项检查，任何一项不通过
    即拒绝执行并返回结构化错误信息给 LLM，让 LLM 有机会自我修正。

架构位置:
    ReActAgent._execute_tool() 的调用入口
    ┌─────────────┐
    │ LLM 决策     │ → tool_call: search_semantic(args={top_k:50, description:"..."})
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ ToolGuard   │ ← 本模块: 五项安全校验
    │ .pre_check()│
    └──────┬──────┘
           │ ✓ 通过
    ┌──────▼──────┐
    │ _execute    │ → 实际工具执行
    │ _tool()     │
    └─────────────┘

五项校验:
    1. 工具名白名单  — 防止 LLM 幻觉出不存在的工具名
    2. 参数 JSON 大小 — 防止 LLM 注入过大的参数体
    3. top_k 范围    — 防止一次性请求过量数据
    4. 调用频率限制  — 防止 ReAct 循环中反复刷同一工具
    5. 用户上下文要求 — 确保敏感工具不暴露给未登录用户

拒绝后的自愈机制:
    LLM 收到的不是异常, 而是结构化的错误 Observation:
    "[拒绝] 工具 search_semantic: top_k 超出范围 [1, 20] (收到: 50)"
    LLM 看到这条消息后可以调整参数重新调用, 或换用其他工具。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ToolPolicy:
    """单个工具的运行时安全策略。

    每个字段的含义:
        max_calls_per_session: 同一 session 内该工具最多被调用的次数。
            防止 ReAct 循环中 LLM 反复调用同一工具无进展。
        max_params_size: 参数 JSON 序列化后的最大字节数。
            防止 LLM 在 description/query 参数中注入超大文本。
        allowed_top_k_range: (min, max) 闭区间。top_k 超出范围时拒绝。
            防止 LLM 在一次调用中请求全量数据。
        require_user_context: 是否需要 user_id。
            标记为 True 的工具在 user_id=None 时直接拒绝。
        timeout_seconds: 工具执行的最长允许时间 (秒)。
    """

    def __init__(
        self,
        max_calls_per_session: int = 10,
        max_params_size: int = 500,
        allowed_top_k_range: tuple = (1, 20),
        require_user_context: bool = False,
        timeout_seconds: float = 5.0,
    ):
        self.max_calls_per_session = max_calls_per_session
        self.max_params_size = max_params_size
        self.allowed_top_k_range = allowed_top_k_range
        self.require_user_context = require_user_context
        self.timeout_seconds = timeout_seconds


class ToolGuard:
    """工具调用安全护栏。

    与 ReActAgent 的关系:
        ReActAgent 在 __init__ 中创建 self.tool_guard = ToolGuard()
        在 _execute_tool 入口处调用 self.tool_guard.pre_check(...)
        在 session reset 时调用 self.tool_guard.reset_session(...)

    线程安全:
        _call_counts 的读写只在单线程的 asyncio event loop 中进行,
        每一个 session 在任一时刻只有一条 astream/invoke 在执行
        (由 WebSocket 连接天然隔离), 因此不需要额外加锁。
    """

    # ── 每个工具的策略定义 ──
    DEFAULT_POLICIES: Dict[str, ToolPolicy] = {
        "get_user_profile": ToolPolicy(
            max_calls_per_session=1,  # 一次查询即可获取完整画像，多余说明 LLM 在乱来
            require_user_context=True,
        ),
        "search_cold_start": ToolPolicy(
            max_calls_per_session=5,
            allowed_top_k_range=(1, 30),
        ),
        "search_by_preference": ToolPolicy(
            max_calls_per_session=10,
            max_params_size=800,
            allowed_top_k_range=(1, 30),
            require_user_context=True,
        ),
        "search_by_filter": ToolPolicy(
            max_calls_per_session=10,
            max_params_size=800,
            allowed_top_k_range=(1, 30),
        ),
        "search_by_mood": ToolPolicy(
            max_calls_per_session=5,
            max_params_size=300,
            allowed_top_k_range=(1, 30),
        ),
        "search_semantic": ToolPolicy(
            max_calls_per_session=10,
            max_params_size=1000,
            allowed_top_k_range=(1, 30),
        ),
    }

    def __init__(self, policies: Optional[Dict[str, ToolPolicy]] = None):
        """
        Args:
            policies: 自定义工具策略, 与 DEFAULT_POLICIES 合并。
                      传入的 key 会覆盖默认值。
        """
        self._policies = dict(self.DEFAULT_POLICIES)
        if policies:
            self._policies.update(policies)

        # {session_id: {tool_name: call_count}}
        self._call_counts: Dict[str, Dict[str, int]] = {}

    # ── 公开 API ──

    def pre_check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session_id: str = "default",
        user_id: Optional[int] = None,
    ) -> Optional[str]:
        """工具调用前的安全校验。

        Returns:
            None  — 所有检查通过, 可以安全执行。
            str   — 拒绝原因。调用方应将其作为 Observation 返回给 LLM,
                     让 LLM 看到拒绝原因并自我修正。

        设计决策:
            返回 None/str 而非 raise Exception, 因为:
            1. 拒绝是一次"教学机会" — LLM 需要看到具体原因来修正行为
            2. Exception 会中断 ReAct 循环, 而返回错误 Observation
               让 LLM 可以在同一轮循环中调整策略
        """
        policy = self._policies.get(tool_name)

        # ── 检查 1: 工具名白名单 ──
        if policy is None:
            return (
                f"未知工具: '{tool_name}'。"
                f"可用工具: {', '.join(sorted(self._policies.keys()))}"
            )

        # ── 检查 2: 参数 JSON 大小 ──
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > policy.max_params_size:
            return (
                f"参数过大 ({len(args_str)} bytes > "
                f"max {policy.max_params_size})。请精简参数内容。"
            )

        # ── 检查 3: top_k 范围 ──
        if "top_k" in args:
            lo, hi = policy.allowed_top_k_range
            top_k = int(args["top_k"])
            if not (lo <= top_k <= hi):
                return (
                    f"top_k={top_k} 超出允许范围 [{lo}, {hi}]。"
                    f"请将 top_k 设置在此范围内。"
                )

        # ── 检查 4: 用户上下文 ──
        if policy.require_user_context and user_id is None:
            return (
                f"工具 '{tool_name}' 需要用户信息, "
                f"但当前请求未提供 user_id。请先让用户登录或换用其他工具。"
            )

        # ── 检查 5: 调用频率 ──
        if session_id not in self._call_counts:
            self._call_counts[session_id] = {}
        session_counts = self._call_counts[session_id]
        current = session_counts.get(tool_name, 0)

        if current >= policy.max_calls_per_session:
            return (
                f"工具 '{tool_name}' 本会话已调用 {current} 次 "
                f"(上限 {policy.max_calls_per_session})。"
                f"请基于已有结果给出最终推荐, 不要再调用此工具。"
            )

        # 全部通过 → 计数
        session_counts[tool_name] = current + 1
        return None

    def reset_session(self, session_id: str) -> None:
        """Reset session 时清空该 session 的工具调用计数。"""
        self._call_counts.pop(session_id, None)
        logger.debug(f"ToolGuard: reset session '{session_id}'")

    def get_stats(self, session_id: str) -> Dict[str, int]:
        """获取某个 session 的工具调用统计 (用于可观测性)。"""
        return dict(self._call_counts.get(session_id, {}))

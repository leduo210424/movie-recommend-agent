"""
L6: 错误恢复层 (Error Recovery)

三个独立组件, 解决不同维度的故障场景:

┌──────────────────────────────────────────────────────────────┐
│                     错误恢复层架构                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  RetryPolicy          CircuitBreaker       GracefulDegrader  │
│  (瞬时故障)            (持续故障)           (完全不可用)        │
│                                                              │
│  问题: 网络抖动        问题: API 连续失败    问题: LLM 彻底挂了  │
│  策略: 指数退避重试    策略: 熔断→冷却→试探  策略: 降级到规则引擎 │
│  恢复: 自动           恢复: 半开→关闭        恢复: 手动或自动探测  │
│                                                              │
│  三者协作流程:                                                │
│                                                              │
│  LLM API 调用                                                │
│      │                                                       │
│      ├── RetryPolicy 包裹 (重试 3 次, 指数退避)               │
│      │     │                                                 │
│      │     ├── 成功 → CircuitBreaker.record_success()         │
│      │     │                                                 │
│      │     └── 3 次全部失败                                    │
│      │           │                                           │
│      │           └── CircuitBreaker.record_failure() × 3      │
│      │                 │                                     │
│      │                 ├── failures < 5 → 异常向上抛          │
│      │                 │                                     │
│      │                 └── failures >= 5 → 断路器熔断          │
│      │                       │                               │
│      │                       └── GracefulDegrader 接管        │
│      │                              │                        │
│      │                              └── 纯规则推荐            │
│      │                                 (不走 LLM)             │
│      │                                                       │
│      └── 30s 后断路器半开 → 尝试 1 次 LLM 调用                │
│              ├── 成功 → 断路器关闭, 恢复正常                   │
│              └── 失败 → 断路器重新打开, 继续降级               │
│                                                              │
└──────────────────────────────────────────────────────────────┘

原理说明:

1. RetryPolicy (重试策略)
   - 问题根因: LLM API 偶尔因网络波动/服务端短暂过载返回 5xx
   - 解决原理: 瞬时故障的黄金解是重试 + 退避。退避递增让服务端有时间恢复。
   - 指数退避公式: delay = base_delay × 2^attempt (1s, 2s, 4s)
   - 注意事项: 重试必须幂等 — LLM generate 操作本身就是幂等的 (无副作用)

2. CircuitBreaker (断路器)
   - 问题根因: 连续失败说明 LLM API 出现了持续性故障 (配额耗尽/服务宕机),
     继续重试只会浪费时间和客户端资源。
   - 解决原理: 三态状态机 (Closed → Open → Half-Open → Closed)
     - Closed (闭合): 正常状态, 请求直接放行
     - Open (断开): 熔断状态, 请求直接拒绝, 不实际调用 LLM
     - Half-Open (半开): 冷却期后, 允许 1 次试探请求
   - 状态转换:
       Closed ── failures >= 5 ──→ Open
       Open   ── 30s 冷却后 ──→ Half-Open (自动)
       Half-Open ── 试探成功 ──→ Closed
       Half-Open ── 试探失败 ──→ Open (重新冷却)
   - 为什么需要 Half-Open?
     直接 Open → Closed 太危险: 如果 API 仍然不可用, 会立即再次熔断,
     造成 "震荡"。Half-Open 用 1 次请求做健康检查, 风险最小化。

3. GracefulDegrader (优雅降级)
   - 问题根因: 断路器熔断后, Agent 无法使用 LLM, 但不能对用户返回 500。
   - 解决原理: 跳过 LLM 推理环节, 用 BasicRecommender 的确定性方法
     直接生成推荐 — 用户体验下降 (没有智能推理), 但系统仍然可用。
   - 降级策略分层:
     - 有 user_id + 用户画像 → 走 recommend() (用户向量匹配)
     - 有 user_id 无画像 → 走 _cold_start_recommend() (贝叶斯热度)
     - 无 user_id → 走 _cold_start_recommend() + 去重
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RetryPolicy
# ═══════════════════════════════════════════════════════════════

class RetryPolicy:
    """指数退避重试策略。

    原理:
        瞬时故障 (网络抖动、API 服务端短暂 503) 通常在几百毫秒到几秒内恢复。
        固定的重试间隔会导致 "惊群效应" — 所有客户端同时重试, 进一步压垮服务端。
        指数退避让重试在时间上分散, 给服务端恢复留出窗口。

    参数选择:
        max_retries=3: 4 次及以上重试的边际收益极低。3 次重试覆盖了
                       99% 的瞬时故障, 更多只是在浪费用户等待时间。
        base_delay=1s: DeepSeek API 的 P99 延迟约 2-3s, 1s 起步确保
                       至少给服务端一个请求周期的时间恢复。
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.total_retries = 0
        self.total_successes = 0

    async def execute(
        self,
        coro_factory: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """用重试策略包裹一个异步协程。

        Args:
            coro_factory: 零参数的可调用对象, 每次调用返回一个新的 awaitable。
                         必须是工厂函数而非协程对象本身, 因为协程只能 await 一次。
            operation_name: 用于日志标识。

        Returns:
            协程的成功返回值。

        Raises:
            最后一次重试的异常 (所有重试耗尽后)。
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):  # 0, 1, 2, 3 (共 4 次尝试)
            try:
                result = await coro_factory()
                self.total_successes += 1
                if attempt > 0:
                    logger.info(
                        f"[Retry] {operation_name} 在第 {attempt} 次重试后成功"
                    )
                return result  # 成功, 直接返回

            except Exception as e:
                last_error = e
                self.total_retries += 1

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(
                        f"[Retry] {operation_name} 失败 "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {str(e)[:120]}. "
                        f"{delay:.1f}s 后重试..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[Retry] {operation_name} 全部 "
                        f"{self.max_retries + 1} 次尝试均失败: "
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )

        # 所有重试耗尽
        raise last_error  # type: ignore[misc]

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_retries": self.total_retries,
            "total_successes": self.total_successes,
        }


# ═══════════════════════════════════════════════════════════════
# CircuitBreaker
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """断路器 — 三态状态机。

    原理:
        参考 Michael Nygard 的 "Release It!" 中提出的断路器模式。
        核心思想: 快速失败 (fail-fast) 优于缓慢超时 (slow-timeout)。

    为什么需要:
        场景: LLM API Key 配额耗尽。每次调用都会等待 30s 后超时。
        无断路器: 用户 A 触发 Agent → 等 30s → 超时。
                   用户 B 触发 Agent → 等 30s → 超时。
                   → 每个请求都在白白等待, 用户体验极差。
        有断路器: 连续 5 次失败 → 熔断。
                   用户 C 触发 Agent → 立即返回降级结果 (不等待)。
                   → 快速失败, 用户体验可接受。

    三态详解:
        CLOSED (闭合/正常):
            - 所有请求正常通过
            - 累计失败计数
            - 失败数达到阈值 → OPEN

        OPEN (断开/熔断):
            - 所有请求立即拒绝, 不实际调用 LLM
            - 冷却计时器运行
            - 冷却时间到 → HALF_OPEN

        HALF_OPEN (半开/试探):
            - 允许有限次试探请求 (默认 1 次)
            - 试探成功 → CLOSED (恢复正常)
            - 试探失败 → OPEN (重新熔断, 重置冷却)
    """

    # 状态常量
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 1,
    ):
        """
        Args:
            name: 断路器标识 (用于日志, 如 "llm_api", "tool_exec")
            failure_threshold: 连续失败多少次后熔断 (默认 5)
            recovery_timeout: 熔断后多少秒进入半开 (默认 30s)
            half_open_max_requests: 半开状态最多放行多少次试探 (默认 1)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._last_state_change: float = time.time()
        self._half_open_requests: int = 0

    # ── 状态查询 ──

    @property
    def state(self) -> str:
        """当前状态。会先检查是否需要从 OPEN 自动转换到 HALF_OPEN。"""
        if self._state == self.OPEN:
            if self._should_attempt_recovery():
                self._transition_to(self.HALF_OPEN)
        return self._state

    @property
    def is_open(self) -> bool:
        """是否处于熔断状态 (请求应被直接拒绝)。"""
        return self.state == self.OPEN

    def _should_attempt_recovery(self) -> bool:
        """检查冷却时间是否已到。"""
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    # ── 状态转换 ──

    def _transition_to(self, new_state: str) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        logger.warning(
            f"[CircuitBreaker:{self.name}] {old_state} → {new_state} "
            f"(failures={self._failure_count}, successes={self._success_count})"
        )

    def record_success(self) -> None:
        """记录一次成功的调用。"""
        self._success_count += 1
        if self._state == self.HALF_OPEN:
            # 半开状态试探成功 → 恢复
            logger.info(
                f"[CircuitBreaker:{self.name}] 试探请求成功, 断路器关闭"
            )
            self._failure_count = 0
            self._half_open_requests = 0
            self._transition_to(self.CLOSED)
        elif self._state == self.CLOSED:
            # 闭合状态下成功 → 重置失败计数
            self._failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """记录一次失败的调用。"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        error_info = f"{type(error).__name__}: {str(error)[:100]}" if error else "unknown"
        logger.warning(
            f"[CircuitBreaker:{self.name}] 调用失败 ({self._failure_count}/"
            f"{self.failure_threshold}): {error_info}"
        )

        if self._state == self.HALF_OPEN:
            # 半开试探失败 → 重新熔断
            self._half_open_requests = 0
            self._transition_to(self.OPEN)

        elif self._state == self.CLOSED and self._failure_count >= self.failure_threshold:
            # 闭合状态失败数达标 → 熔断
            self._transition_to(self.OPEN)

    # ── 执行入口 ──

    def before_call(self) -> Optional[str]:
        """调用前检查。返回 None 表示可以调用, 返回 str 表示拒绝原因。"""
        if self.state == self.OPEN:
            elapsed = time.time() - self._last_failure_time
            remaining = max(0, self.recovery_timeout - elapsed)
            return (
                f"[CircuitBreaker:{self.name}] 断路器已熔断, "
                f"{remaining:.0f}s 后自动恢复"
            )

        if self._state == self.HALF_OPEN:
            if self._half_open_requests >= self.half_open_max_requests:
                return (
                    f"[CircuitBreaker:{self.name}] 半开状态已达试探上限"
                )
            self._half_open_requests += 1

        return None  # 放行

    # ── 统计 ──

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "seconds_in_current_state": time.time() - self._last_state_change,
        }

    def reset(self) -> None:
        """手动重置断路器 (用于测试或运维强制恢复)。"""
        self._state = self.CLOSED
        self._failure_count = 0
        self._half_open_requests = 0
        self._last_state_change = time.time()
        logger.info(f"[CircuitBreaker:{self.name}] 手动重置 → CLOSED")


# ═══════════════════════════════════════════════════════════════
# GracefulDegrader
# ═══════════════════════════════════════════════════════════════

class GracefulDegrader:
    """优雅降级器 — 当 LLM 不可用时回退到纯规则推荐。

    原理:
        Agent 的核心价值来自 LLM 的智能推理 (选工具/理解意图/生成推荐理由)。
        但即使没有 LLM, BasicRecommender 的确定性方法仍能给出可用的推荐 —
        这构成了降级的兜底方案。

    降级层级 (由优到劣):
        Level 0 (正常): LLM 推理 + Tool 调用      → 最佳体验
        Level 1 (降级): 跳过 LLM, 有用户画像时     → 向量匹配推荐
        Level 2 (降级): 跳过 LLM, 无用户画像时     → 贝叶斯热度推荐
        Level 3 (兜底): 纯随机                       → 几乎不用

    与 CircuitBreaker 的协作:
        CircuitBreaker 决定 "是否该降级" (when)
        GracefulDegrader 决定 "降级后做什么" (what)
    """

    def __init__(self, recommender):
        """
        Args:
            recommender: BasicRecommender 实例, 提供确定性推荐方法。
        """
        self._recommender = recommender
        self._degraded_count = 0
        self._normal_count = 0

    def recommend(
        self,
        user_id: Optional[int],
        query: str = "",
        top_k: int = 10,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """降级推荐 — 完全不依赖 LLM。

        直接使用 BasicRecommender 的确定性方法生成推荐。
        不生成推荐解释 (ExplanationEngine 依赖 LLM 的部分被跳过)。

        Returns:
            与 ReActAgent.invoke() 的 results 字段格式兼容的电影列表。
        """
        self._degraded_count += 1
        skip_ids = exclude_ids or set()

        if user_id is not None and self._recommender._get_user_profile(user_id):
            # Level 1 降级: 有用户画像 → 个性化推荐 (无 LLM 查询理解)
            results = self._recommender.recommend(
                user_id=user_id,
                query_text=query,
                top_k=top_k,
                exclude_ids=skip_ids,
            )
            logger.info(
                f"[Degrader] Level 1: 有用户画像的规则推荐 "
                f"(user_id={user_id}, top_k={top_k})"
            )
        else:
            # Level 2 降级: 无用户画像 → 冷启动热度推荐
            results = self._recommender._cold_start_recommend(
                top_k=top_k, exclude_ids=skip_ids
            )
            logger.info(
                f"[Degrader] Level 2: 冷启动热度推荐 (top_k={top_k})"
            )

        # 格式化为与 agent tool_results 兼容的结构
        return [
            {
                "movie_id": r.movie_id,
                "title": r.title,
                "genres": r.genres,
                "score": r.score,
                "user_sim": r.user_sim,
                "rag_sim": r.rag_sim,
                "popularity": r.popularity,
            }
            for r in results
        ]

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._degraded_count + self._normal_count
        return {
            "normal_count": self._normal_count,
            "degraded_count": self._degraded_count,
            "degradation_rate": (
                self._degraded_count / max(total, 1)
            ),
        }

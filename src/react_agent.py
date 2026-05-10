"""
ReAct Agent：LLM 驱动的多步推理 + 工具调用推荐系统。

架构:
    entry → agent (LLM: 推理 + ToolCall) → tools → agent → ... → finalize
                ↑                                  │
                └────────── loop (max 5) ──────────┘
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.basic_recommender import BasicRecommender, RecommendationResult
from src.explanation_engine import ExplanationEngine

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


# ── Tool 定义 ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": (
                "Get user's watch history and preferences. "
                "Only call when the query is vague and you need to understand the user's taste. "
                "Skip if the query already specifies genre, year, or mood constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cold_start",
            "description": "Get popular movies. Use when user has no watch history (cold start).",
            "parameters": {
                "type": "object",
                "properties": {"top_k": {"type": "integer"}},
                "required": ["top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_preference",
            "description": "Personalized recommendations based on user history + semantic query matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["user_id", "query", "top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_filter",
            "description": "Search with genre/year constraints. Use when query has explicit filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query", "top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_mood",
            "description": "Search by mood/atmosphere (e.g., relaxing, thrilling).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["mood", "top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic",
            "description": (
                "Pure semantic search. YOU MUST rewrite the user's request into a short English phrase "
                "made of concrete keywords: genres, themes, emotions, settings, visual style. "
                "Do NOT write a sentence — write comma-separated keywords. "
                "Examples: user says '想看让人想旅行的电影' → description='travel adventure road movie "
                "beautiful scenery exploration wanderlust inspiring journey'. "
                "user says '让人重新思考人生的电影' → description='philosophical drama existential "
                "thought-provoking life changing deep meaning character study'. "
                "Use this for any complex/abstract/emotional request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Comma-separated English keywords: genres themes emotions settings visual_style",
                    },
                    "top_k": {"type": "integer"},
                },
                "required": ["description", "top_k"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个专业的电影推荐顾问 AI Agent。你的目标是根据用户的需求，使用可用的工具为用户找到最合适的电影推荐。

你拥有以下能力：
1. 获取用户的观影历史和偏好
2. 推荐热门/流行电影（冷启动场景）
3. 基于用户偏好的个性化推荐
4. 根据特定条件（类型、年份等）的精确搜索
5. 根据心情/氛围的推荐
6. 语义搜索——你把用户的复杂/抽象需求重写为密集关键词描述，由系统做 embedding 匹配

工作流程：
1. 首先理解用户的查询需求
2. 根据需求选择合适的工具
3. 调用工具获取推荐结果
4. 根据结果进行综合分析和排序
5. 用中文给出有洞察的推荐和解释

推荐关键原则：
- 如果用户 query 含有明确的类型/年份，直接用 search_by_filter
- 如果用户 query 表达心情/氛围，用 search_by_mood
- 如果 query 模糊且用户有历史，用 search_by_preference
- 如果没有用户信息，用 search_cold_start
- 如果 query 表达复杂抽象的场景/感受/愿望（如"看了想旅行"、"让人重新思考人生"），用 search_semantic，并将 query 翻译为逗号分隔的英文关键词（如 "travel, adventure, road movie, beautiful scenery, wanderlust, inspiring"）
- 调用工具时确保参数有效"""


class LLMInterface:
    """LLM 调用抽象接口"""

    def generate(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        raise NotImplementedError


class QwenLLM(LLMInterface):
    """阿里云 DashScope Qwen 实现"""

    def __init__(self, api_key: str, model: str = "qwen-plus"):
        import dashscope
        dashscope.api_key = api_key
        self._model = model

    def generate(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        from dashscope import Generation

        system_text = ""
        conv_msgs = messages
        if messages and messages[0].get("role") == "system":
            system_text = str(messages[0].get("content", ""))
            conv_msgs = messages[1:]

        kwargs = {
            "model": self._model,
            "messages": conv_msgs,
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        if system_text:
            kwargs["system"] = system_text

        response = Generation.call(**kwargs)
        if response.status_code != 200:
            return {"content": "", "tool_calls": None}
        msg = response.output.choices[0].message
        try:
            content = msg["content"] or ""
        except (KeyError, TypeError):
            content = ""
        try:
            tcs = msg["tool_calls"]
        except (KeyError, TypeError):
            tcs = None
        if tcs:
            normalized = []
            for tc in tcs:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        name = func.get("name", "")
                        args_raw = func.get("arguments", {})
                    else:
                        name = getattr(func, "name", "")
                        args_raw = getattr(func, "arguments", "{}")
                else:
                    name = getattr(tc.function, "name", "")
                    args_raw = getattr(tc.function, "arguments", "{}")
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if name:
                    normalized.append({"function": {"name": name, "arguments": args}})
            tcs = normalized
        return {"content": content, "tool_calls": tcs}


class ReActAgent:
    """
    真正的 ReAct Agent：LLM 驱动多步推理 + 工具调用。
    替代原有的规则 Workflow（AgenticRecommender）。
    """

    def __init__(
        self,
        recommender: Optional[BasicRecommender] = None,
        llm: Optional[LLMInterface] = None,
    ):
        self.recommender = recommender or BasicRecommender()
        self.llm = llm
        self.explanation_engine = ExplanationEngine(self.recommender)
        self.graph = self._build_graph()

    # ── Tool 执行（与 llm_agent.py 共享逻辑） ──

    def _execute_tool(self, name: str, args: Dict[str, Any], exclude_ids: Optional[set] = None
                       ) -> Tuple[str, List[Dict[str, Any]]]:
        """执行单个工具，返回 (Observation文本, 结构化电影结果列表)"""
        try:
            if name == "get_user_profile":
                movies = self.recommender.get_user_movies(int(args["user_id"]))
                if not movies:
                    return "新用户，无观影历史", []
                return f"用户看过 {len(movies)} 部电影，包括: {', '.join(movies[:10])}", []

            elif name == "search_cold_start":
                results = self.recommender._cold_start_recommend(
                    top_k=int(args.get("top_k", 5)), exclude_ids=exclude_ids)
                return self._format_movie_list(results, "热门推荐"), self._results_to_dicts(results)

            elif name == "search_by_preference":
                results = self.recommender.recommend(
                    user_id=int(args["user_id"]),
                    query_text=str(args.get("query", "")),
                    top_k=int(args.get("top_k", 5)),
                    exclude_ids=exclude_ids,
                )
                return self._format_movie_list(results, "个性化推荐"), self._results_to_dicts(results)

            elif name == "search_by_filter":
                results = self.recommender.recommend_by_filter(
                    query=str(args.get("query", "")),
                    top_k=int(args.get("top_k", 5)),
                )
                return self._format_movie_list(results, "条件过滤"), self._results_to_dicts(results)

            elif name == "search_by_mood":
                results = self.recommender.recommend_by_mood(
                    mood=str(args.get("mood", "")),
                    top_k=int(args.get("top_k", 5)),
                )
                return self._format_movie_list(results, f"心情推荐: {args.get('mood', '')}"), self._results_to_dicts(results)

            elif name == "search_semantic":
                results = self.recommender.recommend_by_semantic(
                    description=str(args.get("description", "")),
                    top_k=int(args.get("top_k", 5)),
                    exclude_ids=exclude_ids,
                )
                return self._format_movie_list(results, "语义搜索"), self._results_to_dicts(results)

            else:
                return f"未知工具: {name}", []

        except Exception as e:
            return f"工具执行错误: {str(e)}", []

    @staticmethod
    def _results_to_dicts(results: List[RecommendationResult]) -> List[Dict[str, Any]]:
        return [{
            "movie_id": r.movie_id,
            "title": r.title,
            "genres": r.genres,
            "score": r.score,
            "user_sim": r.user_sim,
            "rag_sim": r.rag_sim,
            "popularity": r.popularity,
        } for r in results]

    @staticmethod
    def _format_movie_list(results: List[RecommendationResult], label: str) -> str:
        if not results:
            return f"[{label}] 无结果"
        lines = [f"[{label}] 找到 {len(results)} 部电影:"]
        for i, r in enumerate(results[:5], 1):
            lines.append(f"  {i}. {r.title} (genres: {', '.join(r.genres[:3])}, "
                         f"score: {r.score:.3f})")
        return "\n".join(lines)

    # ── LangGraph 节点 ──

    def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 推理节点：发消息 → 解析 Thought + Action"""
        messages = state.get("messages", [])
        iteration = int(state.get("iteration", 0)) + 1

        if iteration > MAX_ITERATIONS:
            return {**state, "iteration": iteration, "route": "finalize",
                    "agent_thought": "达到最大迭代次数，强制结束"}

        response = self.llm.generate(messages, TOOLS)
        content = response.get("content", "") or ""
        tool_calls = response.get("tool_calls")

        if tool_calls:
            # LLM 决定调用工具
            tc = tool_calls[0] if isinstance(tool_calls[0], dict) else tool_calls[0]
            # 解析 tool_call（兼容 dict/object、JSON string/dict arguments）
            if isinstance(tc, dict):
                func = tc.get("function", {})
                tool_name = func.get("name", "") if isinstance(func, dict) else ""
                args_raw = func.get("arguments", {}) if isinstance(func, dict) else {}
            else:
                tool_name = getattr(tc.function, "name", "")
                args_raw = getattr(tc.function, "arguments", "{}")
            # Qwen 可能返回 JSON 字符串或 dict
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})

            # 将 assistant 的 tool_call 加入对话历史（避免连续两个 user 消息）
            messages.append({
                "role": "assistant",
                "content": content or f"I will use {tool_name} to find movies.",
            })

            return {
                **state,
                "messages": messages,
                "iteration": iteration,
                "agent_thought": content,
                "pending_tool": tool_name,
                "pending_args": args,
                "route": "tools",
            }
        else:
            # LLM 认为推理完成
            return {
                **state,
                "iteration": iteration,
                "agent_thought": content,
                "final_answer": content,
                "route": "finalize",
            }

    def _tools_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行节点"""
        tool_name = state.get("pending_tool", "")
        args = state.get("pending_args", {})
        # LangGraph state 中 list 可安全传递，在此处转回 set
        exclude_raw = state.get("exclude_ids")
        exclude_set = set(exclude_raw) if exclude_raw else None
        observation, movie_dicts = self._execute_tool(tool_name, args, exclude_ids=exclude_set)

        messages = state.get("messages", [])
        messages.append({
            "role": "user",
            "content": f"Observation from {tool_name}: {observation}",
        })

        # 聚合所有工具调用的结果（用于最终输出）
        all_results = list(state.get("tool_results", []))
        all_results.extend(movie_dicts)

        return {
            **state,
            "messages": messages,
            "observation": observation,
            "iteration": int(state.get("iteration", 0)),
            "route": "agent",
            "tool_results": all_results,
        }

    @staticmethod
    def _finalize(state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    # ── Graph 构建 ──

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(dict)

        graph.add_node("entry", lambda s: {**s, "route": "agent"})
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("entry")
        graph.add_edge("entry", "agent")

        graph.add_conditional_edges(
            "agent",
            lambda s: s.get("route", "finalize"),
            {"tools": "tools", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "tools",
            lambda s: s.get("route", "finalize"),
            {"agent": "agent", "finalize": "finalize"},
        )

        graph.add_edge("finalize", END)
        return graph.compile()

    # ── 对外接口 ──

    def invoke(
        self,
        user_id: Optional[int],
        query: str = "",
        top_k: int = 10,
        exclude_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        # 空查询无意图信号 → 直接冷启动，不浪费 API 调用
        if not query.strip():
            results = self.recommender._cold_start_recommend(top_k=top_k, exclude_ids=exclude_ids)
            return {
                "route": "react_agent",
                "decision_reason": "空查询，直接使用热门推荐兜底",
                "results": self._results_to_dicts(results),
            }

        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户ID: {user_id or '无'}\n查询: {query}\n请推荐 {top_k} 部电影。"},
        ]

        exclude_list = list(exclude_ids) if exclude_ids else []

        result = self.graph.invoke({
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "messages": messages,
            "iteration": 0,
            "exclude_ids": exclude_list,
        })

        # 格式化输出
        tool_results = result.get("tool_results", [])
        # 去重（同一部电影可能被多个工具返回）
        seen = set()
        unique_results = []
        for r in tool_results:
            mid = r.get("movie_id")
            if mid and mid not in seen:
                seen.add(mid)
                unique_results.append(r)

        return {
            "route": "react_agent",
            "decision_reason": result.get("agent_thought", ""),
            "observation": result.get("observation", ""),
            "iterations": result.get("iteration", 0),
            "final_answer": result.get("final_answer", ""),
            "results": unique_results[:result.get("top_k", 10)],
        }

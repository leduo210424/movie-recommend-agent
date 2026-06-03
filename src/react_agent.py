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
1. 首先理解用户的查询需求（注意参考对话历史中的上下文和反馈）
2. 根据需求选择合适的工具
3. 调用工具获取推荐结果
4. 根据结果进行综合分析和排序
5. 用中文给出有洞察的推荐和解释

多轮对话与反馈原则：
- 如果对话历史中包含用户对之前推荐的反馈，你必须据此调整当前推荐：
  * 用户 liked 某部电影 → 优先推荐类型/风格相似的电影，在 query 中融入偏好信号
  * 用户 disliked 某部电影 → 避免推荐同类型的电影，调整推荐策略
  * 用户说"换一批"或类似表达 → 使用不同的检索策略或调整参数
- 跟踪对话中的上下文：如果用户问"还有类似的吗"，用上一次推荐结果作为参考
- 如果用户引用之前的推荐（如"第二部不错"），根据对应电影的属性调整后续推荐

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

    def stream_generate(self, messages: List[Dict], tools: List[Dict]):
        """流式生成，逐 chunk yield。返回 (delta_content: str, is_final: bool, tool_calls: list|None)"""
        raise NotImplementedError


class DeepSeekLLM(LLMInterface):
    """DeepSeek 模型 (通过 OpenAI 兼容 API 调用)"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com"):
        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        kwargs = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"content": "", "tool_calls": None}

        msg = response.choices[0].message
        content = msg.content or ""
        tcs = msg.tool_calls

        if tcs:
            normalized = []
            for tc in tcs:
                name = tc.function.name
                args_raw = tc.function.arguments
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if name:
                    normalized.append({"function": {"name": name, "arguments": args}})
            tcs = normalized
        return {"content": content, "tool_calls": tcs}

    @staticmethod
    def _normalize_stream_tool_calls(tool_call_buf: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        if not tool_call_buf:
            return None
        normalized = []
        for idx in sorted(tool_call_buf.keys()):
            buf = tool_call_buf[idx]
            name = buf["name"]
            args_raw = buf["arguments"]
            args = json.loads(args_raw) if args_raw else {}
            if name:
                normalized.append({"function": {"name": name, "arguments": args}})
        return normalized or None

    def stream_generate(self, messages: List[Dict], tools: List[Dict]):
        """同步流式生成 (供 evaluate.py 等非 async 场景使用)。"""
        kwargs = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": True,
        }

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"LLM stream call failed: {e}")
            yield (f"[LLM 调用失败: {e}]", True, None)
            return

        tool_call_buf: Dict[int, Dict[str, Any]] = {}
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    yield (delta.content, False, None)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                        buf = tool_call_buf[idx]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments
        except Exception as e:
            logger.error(f"LLM stream iteration error: {e}")
            yield (f"[流式处理异常: {e}]", True, None)
            return

        yield ("", True, self._normalize_stream_tool_calls(tool_call_buf))

    async def astream_generate(self, messages: List[Dict], tools: List[Dict]):
        """异步流式生成 (供 WebSocket astream 使用，不阻塞事件循环)。"""
        kwargs = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": True,
        }

        try:
            print(f"[ASTREAM_GEN] creating stream with model={self._model}, timeout=60...", flush=True)
            stream = await self._async_client.chat.completions.create(**kwargs)
            print(f"[ASTREAM_GEN] stream created, iterating chunks...", flush=True)
        except Exception as e:
            logger.error(f"LLM async stream call failed: {e}")
            print(f"[ASTREAM_GEN] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            yield (f"[LLM 调用失败: {e}]", True, None)
            return

        tool_call_buf: Dict[int, Dict[str, Any]] = {}
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    yield (delta.content, False, None)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                        buf = tool_call_buf[idx]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments
        except Exception as e:
            logger.error(f"LLM async stream iteration error: {e}")
            yield (f"[流式处理异常: {e}]", True, None)
            return

        yield ("", True, self._normalize_stream_tool_calls(tool_call_buf))


class ReActAgent:
    """
    真正的 ReAct Agent：LLM 驱动多步推理 + 工具调用。
    替代原有的规则 Workflow（AgenticRecommender）。

    支持多轮对话（session）和用户反馈（feedback）。
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
        # session 管理：session_id → {"history": [...], "exclude_ids": set, "liked_genres": [], ...}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # ── Tool 执行 ──

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
            # LLM 可能返回多个 tool_call（并行调用）
            pending_tools = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "") if isinstance(func, dict) else ""
                    args_raw = func.get("arguments", {}) if isinstance(func, dict) else {}
                else:
                    tool_name = getattr(tc.function, "name", "")
                    args_raw = getattr(tc.function, "arguments", "{}")
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if tool_name:
                    pending_tools.append({"name": tool_name, "args": args})

            # 将 assistant 的 tool_calls 加入对话历史
            tool_names = ", ".join(t["name"] for t in pending_tools)
            messages.append({
                "role": "assistant",
                "content": content or f"I will use {tool_names} to find movies.",
            })

            return {
                **state,
                "messages": messages,
                "iteration": iteration,
                "agent_thought": content,
                "pending_tools": pending_tools,
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
        """工具执行节点，支持并行执行多个工具"""
        pending_tools = state.get("pending_tools", [])
        exclude_raw = state.get("exclude_ids")
        exclude_set = set(exclude_raw) if exclude_raw else None

        messages = state.get("messages", [])
        all_results = list(state.get("tool_results", []))
        observations = []

        for tool in pending_tools:
            tool_name = tool["name"]
            args = tool["args"]
            observation, movie_dicts = self._execute_tool(tool_name, args, exclude_ids=exclude_set)
            observations.append(f"Observation from {tool_name}: {observation}")
            all_results.extend(movie_dicts)

        messages.append({
            "role": "user",
            "content": "\n".join(observations),
        })

        return {
            **state,
            "messages": messages,
            "observation": "\n".join(observations),
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

    # ── Session 与反馈管理 ──

    def _compress_history(self, session_id: str):
        """裁剪对话历史，防止上下文过长。保留 system 消息 + 最近 N 轮。"""
        session = self.sessions.get(session_id)
        if not session:
            return
        history = session.get("history", [])
        if len(history) <= 24:
            return
        # 保留 system 消息（第一条）+ 最近 10 轮（每轮约 2 条: user + assistant/tool）
        session["history"] = history[:1] + history[-20:]

    def add_feedback(
        self,
        session_id: str,
        user_id: Optional[int],
        movie_id: int,
        feedback: str,  # "like" | "dislike"
        movie_title: str = "",
        movie_genres: Optional[List[str]] = None,
    ):
        """记录用户对某部推荐电影的反馈，影响后续推荐。"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "exclude_ids": set(),
                "liked_genres": [],
                "last_results": [],
            }
        session = self.sessions[session_id]

        # 记录排除列表
        if feedback == "dislike":
            session["exclude_ids"].add(movie_id)
        elif feedback == "like":
            session["exclude_ids"].discard(movie_id)
            if movie_genres:
                for g in movie_genres:
                    if g not in session["liked_genres"]:
                        session["liked_genres"].append(g)

        # 注入反馈到对话历史
        if feedback == "like":
            feedback_text = (
                f"[用户反馈] 用户喜欢 {movie_title} (movie_id={movie_id})。"
                f"请记住这一偏好，后续推荐时优先考虑类似风格的电影。"
                f"{'类型: ' + ', '.join(movie_genres) if movie_genres else ''}"
            )
        else:
            feedback_text = (
                f"[用户反馈] 用户不喜欢 {movie_title} (movie_id={movie_id})。"
                f"请避免推荐同类型的电影，调整推荐策略。"
                f"{'类型: ' + ', '.join(movie_genres) if movie_genres else ''}"
            )
        session["history"].append({"role": "user", "content": feedback_text})

    def get_exclude_ids(self, session_id: str) -> set:
        """获取某个 session 的累计排除电影 ID。"""
        session = self.sessions.get(session_id)
        if not session:
            return set()
        return session.get("exclude_ids", set())

    # ── 对外接口 ──

    def invoke(
        self,
        user_id: Optional[int],
        query: str = "",
        top_k: int = 10,
        exclude_ids: Optional[set] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        # 初始化 session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "exclude_ids": set(),
                "liked_genres": [],
                "last_results": [],
            }
        session = self.sessions[session_id]

        # 合并外部 exclude_ids 和 session 累积的 exclude_ids
        merged_exclude = set(exclude_ids) if exclude_ids else set()
        merged_exclude |= session.get("exclude_ids", set())

        # 空查询无意图信号 → 直接冷启动，不浪费 API 调用
        if not query.strip():
            results = self.recommender._cold_start_recommend(top_k=top_k, exclude_ids=merged_exclude)
            result_dicts = self._results_to_dicts(results)
            # 更新 session 中的 last_results
            session["last_results"] = result_dicts
            return {
                "route": "react_agent",
                "decision_reason": "空查询，直接使用热门推荐兜底",
                "results": result_dicts,
                "session_id": session_id,
            }

        # 构建消息：首次对话包含 system prompt，后续拼接到历史
        history = session["history"]
        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        # 在 query 中融入 liked_genres 偏好信号
        liked_genres = session.get("liked_genres", [])
        enhanced_query = query
        if liked_genres:
            genre_hint = "、".join(liked_genres[:5])
            enhanced_query = f"{query}（用户偏好类型: {genre_hint}）"

        history.append({
            "role": "user",
            "content": f"用户ID: {user_id or '无'}\n查询: {enhanced_query}\n请推荐 {top_k} 部电影。",
        })

        exclude_list = list(merged_exclude)

        result = self.graph.invoke({
            "user_id": user_id,
            "query": enhanced_query,
            "top_k": top_k,
            "messages": list(history),  # 复制一份传给 graph，避免被 graph 修改
            "iteration": 0,
            "exclude_ids": exclude_list,
        })

        # 把 graph 运行后的最终消息状态写回 session
        # graph 返回的 messages 包含了完整对话（assistant tool_calls + user observations + final answer）
        result_messages = result.get("messages", [])
        if result_messages:
            session["history"] = result_messages
            self._compress_history(session_id)

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

        final_results = unique_results[:result.get("top_k", 10)]
        session["last_results"] = final_results

        # 生成推荐解释
        explanations = []
        if final_results:
            try:
                rec_list = [{
                    "movie_id": r.get("movie_id"),
                    "title": r.get("title"),
                    "user_sim": r.get("user_sim", 0),
                    "rag_sim": r.get("rag_sim", 0),
                    "popularity": r.get("popularity", 0),
                } for r in final_results]
                explanations_obj = self.explanation_engine.explain_batch(user_id, rec_list)
                explanations = [e.to_dict() for e in explanations_obj]
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {e}")

        return {
            "route": "react_agent",
            "decision_reason": result.get("agent_thought", ""),
            "observation": result.get("observation", ""),
            "iterations": result.get("iteration", 0),
            "final_answer": result.get("final_answer", ""),
            "results": final_results,
            "explanations": explanations,
            "session_id": session_id,
        }

    async def astream(
        self,
        user_id: Optional[int],
        query: str = "",
        top_k: int = 10,
        session_id: str = "default",
        cancel_event=None,  # asyncio.Event, set 时中断
    ):
        """流式版 invoke：逐事件 yield，支持 WebSocket 推送和用户中断。"""
        import asyncio

        # ── 初始化（和 invoke() 相同逻辑，复用 _ensure_session） ──
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [], "exclude_ids": set(),
                "liked_genres": [], "last_results": [],
            }
        session = self.sessions[session_id]
        merged_exclude = session.get("exclude_ids", set())

        if not query.strip():
            results = self.recommender._cold_start_recommend(top_k=top_k, exclude_ids=merged_exclude)
            result_dicts = self._results_to_dicts(results)
            session["last_results"] = result_dicts
            yield {"event": "results", "data": result_dicts}
            yield {"event": "done", "data": None}
            return

        history = session["history"]
        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        liked_genres = session.get("liked_genres", [])
        enhanced_query = query
        if liked_genres:
            genre_hint = "、".join(liked_genres[:5])
            enhanced_query = f"{query}（用户偏好类型: {genre_hint}）"

        history.append({
            "role": "user",
            "content": f"用户ID: {user_id or '无'}\n查询: {enhanced_query}\n请推荐 {top_k} 部电影。",
        })

        messages = list(history)
        exclude_list = list(merged_exclude)
        all_results = []

        # ── 手动 ReAct 循环（替代 graph.invoke），每步之间 yield 事件 ──
        for iteration in range(1, MAX_ITERATIONS + 1):
            # 中断检查
            if cancel_event and cancel_event.is_set():
                yield {"event": "cancelled", "data": "用户取消了请求"}
                session["history"] = messages
                self._compress_history(session_id)
                return

            print(f"[ASTREAM] iteration={iteration}, sending thinking event", flush=True)
            yield {"event": "thinking", "data": f"Agent 第 {iteration} 轮推理中..."}

            # ── LLM 流式推理 ──
            accumulated = ""
            final_tool_calls = None
            print(f"[ASTREAM] calling astream_generate...", flush=True)
            async for delta, is_final, tcs in self.llm.astream_generate(messages, TOOLS):
                if cancel_event and cancel_event.is_set():
                    yield {"event": "cancelled", "data": "用户取消了请求"}
                    session["history"] = messages
                    self._compress_history(session_id)
                    return
                if delta:
                    accumulated += delta
                    yield {"event": "token", "data": delta}
                if is_final:
                    final_tool_calls = tcs

            # ── LLM 结束后：判断 route ──
            if final_tool_calls and len(final_tool_calls) > 0:
                # 解析所有 tool_call（支持并行调用）
                parsed_tools = []
                for tc in final_tool_calls:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    t_name = func.get("name", "") if isinstance(func, dict) else ""
                    args_raw = func.get("arguments", {}) if isinstance(func, dict) else {}
                    t_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if t_name:
                        parsed_tools.append({"name": t_name, "args": t_args})

                for tool in parsed_tools:
                    yield {"event": "tool_call", "data": {"tool": tool["name"], "args": tool["args"]}}

                tool_names = ", ".join(t["name"] for t in parsed_tools)
                messages.append({
                    "role": "assistant",
                    "content": accumulated or f"I will use {tool_names}.",
                })

                # ── 执行所有 tools ──
                observations = []
                for tool in parsed_tools:
                    observation, movie_dicts = self._execute_tool(
                        tool["name"], tool["args"], exclude_ids=merged_exclude)
                    all_results.extend(movie_dicts)
                    observations.append(f"Observation from {tool['name']}: {observation}")
                    yield {"event": "observation", "data": {
                        "tool": tool["name"], "summary": observation,
                        "movie_count": len(movie_dicts),
                    }}

                messages.append({
                    "role": "user",
                    "content": "\n".join(observations),
                })
            else:
                # LLM 没有 tool_call，推理完成
                yield {"event": "reasoning_done", "data": accumulated}
                break

        # ── 去重 + 格式化 ──
        seen = set()
        unique_results = []
        for r in all_results:
            mid = r.get("movie_id")
            if mid and mid not in seen:
                seen.add(mid)
                unique_results.append(r)
        final_results = unique_results[:top_k]
        session["last_results"] = final_results
        session["history"] = messages
        self._compress_history(session_id)

        # 生成推荐解释
        explanations = []
        if final_results:
            try:
                rec_list = [{
                    "movie_id": r.get("movie_id"),
                    "title": r.get("title"),
                    "user_sim": r.get("user_sim", 0),
                    "rag_sim": r.get("rag_sim", 0),
                    "popularity": r.get("popularity", 0),
                } for r in final_results]
                print(f"[ASTREAM] generating explanations for {len(rec_list)} movies, user_id={user_id}", flush=True)
                explanations_obj = self.explanation_engine.explain_batch(user_id, rec_list)
                explanations = [e.to_dict() for e in explanations_obj]
                print(f"[ASTREAM] generated {len(explanations)} explanations", flush=True)
            except Exception as e:
                import traceback
                print(f"[ASTREAM] EXPLANATION FAILED: {e}", flush=True)
                traceback.print_exc()
                logger.warning(f"Failed to generate explanations: {e}")

        yield {"event": "results", "data": {
            "movies": final_results,
            "explanations": explanations,
            "route": "react_agent",
            "iterations": iteration,
            "session_id": session_id,
        }}
        yield {"event": "done", "data": None}

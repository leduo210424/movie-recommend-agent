"""
FastAPI wrapper for the ReAct Agent Movie Recommender.
Supports multi-turn conversation (session), user feedback, and WebSocket streaming.
"""
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.react_agent import DeepSeekLLM, ReActAgent
from src.user_memory import UserMemoryStore
from src.basic_recommender import BasicRecommender
from src.chroma_store import ChromaMovieStore

logger = logging.getLogger(__name__)


class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    query: Optional[str] = ""
    top_k: int = 5
    session_id: str = "default"


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    feedback: str  # "like" or "dislike"
    movie_title: str = ""
    movie_genres: Optional[List[str]] = None
    session_id: str = "default"


class MovieCreateRequest(BaseModel):
    """新增电影请求"""
    title: str
    genres: List[str] = []
    release_year: Optional[float] = None
    overview_en: str = ""
    movie_id: Optional[int] = None  # 可选，自动分配


class MovieUpdateRequest(BaseModel):
    """更新电影请求 (所有字段可选, 只更新传入的)"""
    title: Optional[str] = None
    genres: Optional[List[str]] = None
    release_year: Optional[float] = None
    overview_en: Optional[str] = None
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None


app = FastAPI(title="Movie Recommend Agent API", version="0.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(Path(ROOT) / "static")), name="static")

_react_agent: Optional[ReActAgent] = None
_chroma_store: Optional[ChromaMovieStore] = None


def _get_agent() -> ReActAgent:
    global _react_agent
    if _react_agent is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="DEEPSEEK_API_KEY not set."
            )
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        llm = DeepSeekLLM(api_key=api_key, model=model, base_url=base_url)
        _react_agent = ReActAgent(llm=llm)
    return _react_agent


def _get_chroma_store() -> ChromaMovieStore:
    """获取 ChromaDB 存储单例 (延迟初始化)"""
    global _chroma_store
    if _chroma_store is None:
        chroma_dir = os.path.join(ROOT, "data", "chroma")
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        _chroma_store = ChromaMovieStore(
            persist_dir=chroma_dir,
            model_name=model_name,
        )
    return _chroma_store


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/harness/status")
def harness_status():
    """Harness 工程层状态：暴露断路器、降级器、工具调用统计。"""
    try:
        agent = _get_agent()
    except HTTPException:
        return {"status": "agent_not_initialized"}

    return {
        "status": "ok",
        "circuit_breaker": agent.llm_circuit_breaker.stats,
        "degrader": agent.degrader.stats,
        "tool_calls": {
            sid: agent.tool_guard.get_stats(sid)
            for sid in agent.sessions
        } if agent.sessions else {},
    }


@app.post("/harness/circuit-breaker/reset")
def reset_circuit_breaker():
    """手动重置 LLM 断路器 (运维接口)。"""
    agent = _get_agent()
    agent.llm_circuit_breaker.reset()
    return {"status": "ok", "message": "断路器已手动重置为 CLOSED"}


@app.get("/")
def home():
    return FileResponse(str(Path(ROOT) / "static" / "index.html"))


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    try:
        agent = _get_agent()
        out = agent.invoke(
            user_id=payload.user_id,
            query=payload.query or "",
            top_k=payload.top_k,
            session_id=payload.session_id,
        )
        return {
            "route": out.get("route"),
            "decision_reason": out.get("decision_reason"),
            "results": out.get("results", []),
            "session_id": out.get("session_id", payload.session_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    """用户对推荐结果进行反馈，影响后续推荐。"""
    try:
        agent = _get_agent()

        # 获取电影 embedding 用于在线微调用户画像
        movie_embedding = _get_movie_embedding(payload.movie_id)

        # 1. 向 Agent session 注入反馈（影响 LLM 后续推理）
        agent.add_feedback(
            session_id=payload.session_id,
            user_id=payload.user_id,
            movie_id=payload.movie_id,
            feedback=payload.feedback,
            movie_title=payload.movie_title,
            movie_genres=payload.movie_genres,
        )

        # 2. 在线微调用户画像向量
        profile_update = {"status": "skipped", "reason": "无需更新"}
        if movie_embedding is not None:
            try:
                profiles_dir = os.path.join(ROOT, "data", "processed")
                profile_update = UserMemoryStore.update_profile_with_feedback(
                    profiles_dir=profiles_dir,
                    user_id=payload.user_id,
                    movie_embedding=movie_embedding,
                    feedback=payload.feedback,
                    alpha=0.05,
                )
            except Exception as e:
                logger.warning(f"在线微调用户画像失败: {e}")
                profile_update = {"status": "skipped", "reason": str(e)}

        return {
            "status": "ok",
            "message": f"反馈已记录: {payload.feedback} movie_id={payload.movie_id}",
            "profile_update": profile_update,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/reset")
def reset_session(session_id: str = "default"):
    """重置指定 session，开始全新对话。"""
    agent = _get_agent()
    if session_id in agent.sessions:
        del agent.sessions[session_id]
    # Harness L3: 重置工具调用计数
    agent.tool_guard.reset_session(session_id)
    return {"status": "ok", "message": f"Session '{session_id}' 已重置"}


# ═══════════════════════════════════════════════════════════════
# 电影查询 API (公开, 只读)
# ═══════════════════════════════════════════════════════════════

@app.get("/movies")
def list_movies(
    offset: int = 0,
    limit: int = 50,
    genre: Optional[str] = None,
    min_year: Optional[int] = None,
):
    """分页列出电影，支持按类型/年份过滤。"""
    store = _get_chroma_store()
    records = store.list_movies(
        offset=offset, limit=limit,
        genre=genre, min_year=min_year,
    )
    return {
        "total": store.count(),
        "offset": offset,
        "limit": limit,
        "results": [r.to_dict() for r in records],
    }


@app.get("/movies/stats")
def movie_stats():
    """电影存储统计信息"""
    store = _get_chroma_store()
    return store.stats()


@app.get("/movies/search")
def search_movies(
    query: str = "",
    top_k: int = 10,
    genre: Optional[str] = None,
):
    """纯语义搜索 (不依赖用户画像): 多粒度融合检索"""
    store = _get_chroma_store()
    results = store.search(
        query_text=query,
        top_k=top_k,
        genre_filter=genre,
    )
    return {"query": query, "top_k": top_k, "results": results}


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    """获取单部电影"""
    store = _get_chroma_store()
    movie = store.get_movie(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"电影 {movie_id} 不存在")
    return movie.to_dict()


# ═══════════════════════════════════════════════════════════════
# 管理员 API (需要 ADMIN_API_KEY 鉴权)
# ═══════════════════════════════════════════════════════════════

def _verify_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    """管理员鉴权依赖: 从 HTTP Header 读取 X-Admin-Key 并校验。

    用法: 在端点参数中声明 `_admin: None = Depends(_verify_admin)` 即可。
    前端调用时需要加 Header: X-Admin-Key: <your-secret>
    """
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected:
        # 未配置 ADMIN_API_KEY 时拒绝所有管理操作 (fail-secure)
        raise HTTPException(
            status_code=501,
            detail="管理员功能未启用 (ADMIN_API_KEY 未配置)",
        )
    if x_admin_key != expected:
        raise HTTPException(
            status_code=401,
            detail="管理员密钥无效",
        )


@app.post("/admin/movies")
def admin_create_movie(payload: MovieCreateRequest, _admin: None = Depends(_verify_admin)):
    """[管理员] 新增电影 (自动编码 embedding 并写入 ChromaDB)"""
    store = _get_chroma_store()
    movie_dict = {
        "title": payload.title,
        "genres": payload.genres,
        "release_year": payload.release_year,
        "overview_en": payload.overview_en,
    }
    if payload.movie_id is not None:
        movie_dict["movie_id"] = payload.movie_id

    try:
        movie_id = store.add_movie(movie_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"新增电影失败: {e}")

    return {"status": "ok", "movie_id": movie_id, "message": f"电影 '{payload.title}' 已新增"}


@app.put("/admin/movies/{movie_id}")
def admin_update_movie(movie_id: int, payload: MovieUpdateRequest, _admin: None = Depends(_verify_admin)):
    """[管理员] 更新电影 (部分更新: 只修改传入的字段)"""
    store = _get_chroma_store()

    fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="至少需要提供一个要更新的字段")

    ok = store.update_movie(movie_id, **fields)
    if not ok:
        raise HTTPException(status_code=404, detail=f"电影 {movie_id} 不存在")
    return {"status": "ok", "movie_id": movie_id, "updated_fields": list(fields.keys())}


@app.delete("/admin/movies/{movie_id}")
def admin_delete_movie(movie_id: int, _admin: None = Depends(_verify_admin)):
    """[管理员] 删除电影 (从三个 ChromaDB collection 中移除)"""
    store = _get_chroma_store()
    ok = store.delete_movie(movie_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"电影 {movie_id} 不存在")
    return {"status": "ok", "movie_id": movie_id, "message": f"电影 {movie_id} 已删除"}


# ── WebSocket 流式推荐 ──

@app.websocket("/ws/recommend")
async def websocket_recommend(ws: WebSocket):
    """WebSocket 端点：支持流式推荐 + 用户中断 + session 复用。"""
    await ws.accept()
    cancel_event = asyncio.Event()
    cancel_task = None

    async def _listen_for_cancel():
        """后台监听取消消息——在推荐请求发出后才启动。"""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "cancel":
                    cancel_event.set()
                elif msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
        except (WebSocketDisconnect, Exception):
            cancel_event.set()

    try:
        # Step 1: 先收推荐请求（此时没有后台监听，不会冲突）
        raw = await ws.receive_text()
        req = json.loads(raw)
        if req.get("type") != "recommend":
            await ws.send_json({"event": "error", "data": "首条消息必须是 recommend 类型"})
            return

        user_id = req.get("user_id")
        query = req.get("query", "")
        top_k = req.get("top_k", 5)
        session_id = req.get("session_id", "default")

        # 初始化 agent（放在 try 块内，异常能被前端感知）
        try:
            agent = _get_agent()
        except HTTPException as e:
            await ws.send_json({"event": "error", "data": e.detail})
            return

        # Step 2: 推荐请求已收到，现在启动取消监听
        cancel_task = asyncio.create_task(_listen_for_cancel())

        print(f"[WS] starting astream for user={user_id}, query={query[:50]}", flush=True)
        async for evt in agent.astream(
            user_id=user_id,
            query=query,
            top_k=top_k,
            session_id=session_id,
            cancel_event=cancel_event,
        ):
            print(f"[WS] event: {evt.get('event', '?')}", flush=True)
            await ws.send_json(evt)
        print(f"[WS] astream finished", flush=True)
    except WebSocketDisconnect:
        cancel_event.set()
    except Exception as e:
        print(f"[WS ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"event": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
            try:
                await cancel_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await ws.close()
        except Exception:
            pass


# ── 辅助 ──

def _get_movie_embedding(movie_id: int):
    """从 BasicRecommender 中获取指定电影的 embedding。"""
    try:
        recommender = BasicRecommender()
        movie = recommender.movie_lookup.get(movie_id)
        if movie is not None:
            return movie.embedding
    except Exception:
        pass
    return None

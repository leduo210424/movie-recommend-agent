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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


app = FastAPI(title="Movie Recommend Agent API", version="0.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(Path(ROOT) / "static")), name="static")

_react_agent: Optional[ReActAgent] = None


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


@app.get("/health")
def health():
    return {"status": "ok"}


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
    return {"status": "ok", "message": f"Session '{session_id}' 已重置"}


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

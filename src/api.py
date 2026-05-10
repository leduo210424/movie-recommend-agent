"""
FastAPI wrapper for the ReAct Agent Movie Recommender.
Provides a /recommend endpoint powered by QwenLLM + ReAct Agent.
"""
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.react_agent import QwenLLM, ReActAgent


class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    query: Optional[str] = ""
    top_k: int = 5


app = FastAPI(title="Movie Recommend Agent API", version="0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(Path(ROOT) / "static")), name="static")

_react_agent: Optional[ReActAgent] = None


def _get_agent() -> ReActAgent:
    global _react_agent
    if _react_agent is None:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="DASHSCOPE_API_KEY not set."
            )
        model = os.getenv("QWEN_MODEL", "qwen-plus")
        llm = QwenLLM(api_key=api_key, model=model)
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
        )
        return {
            "route": out.get("route"),
            "decision_reason": out.get("decision_reason"),
            "results": out.get("results", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

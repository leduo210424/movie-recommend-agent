from typing import Any, Dict, Optional

from src.basic_recommender import BasicRecommender
from src.explanation_engine import ExplanationEngine


class QwenAgenticRecommender:
    """Qwen LLM Agent wrapper (backward-compatible with original API)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus"):
        try:
            from src.llm_agent import QwenAgent
            self.agent = QwenAgent(api_key=api_key, model=model)
            self.explanation_engine = ExplanationEngine(self.agent.basic_recommender)
        except ImportError as e:
            raise ImportError(f"Failed to import QwenAgent: {str(e)}")

    def invoke(
        self,
        user_id: Optional[int],
        query: str = "",
        top_k: int = 10,
        pool_size: int = 50,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        result = self.agent.recommend(
            user_id=user_id, query=query, top_k=top_k, session_id=session_id
        )
        result["route"] = "qwen_llm_agent"
        result["pool_size"] = pool_size
        return result

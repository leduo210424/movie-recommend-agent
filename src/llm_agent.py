"""
Qwen-based LLM Agent with Tool Calling for movie recommendations.
Uses dashscope SDK to interact with Qwen models.
"""

import os
import json
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import dashscope
from dashscope import Generation

from src.basic_recommender import BasicRecommender, RecommendationResult
from src.explanation_engine import ExplanationEngine

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool invocation"""
    tool_name: str
    status: str  # "success" or "error"
    data: Any
    message: str


class QwenAgent:
    """
    Qwen-based Agent with Tool Calling for intelligent movie recommendations.
    Supports dynamic tool selection and reasoning.
    """
    
    # Tool definitions in JSON Schema format for Qwen
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_user_profile",
                "description": "Get user's watch history and preferences. Only call when the query is vague (e.g., 'recommend something', 'I'm bored') and you need to understand the user's taste. Skip if the query already specifies genre, year, mood, or other explicit constraints — go directly to search_by_filter or search_by_mood instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "description": "User ID to fetch profile for"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_cold_start",
                "description": "Get popular/trending movies for users with no watch history (cold start scenario).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_k": {
                            "type": "integer",
                            "description": "Number of movies to return (default 5)"
                        }
                    },
                    "required": ["top_k"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_by_preference",
                "description": "Get personalized recommendations based on user's watch history and semantic similarity to query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "description": "User ID for personalized search"
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what movies the user wants"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of recommendations to return (default 5)"
                        }
                    },
                    "required": ["user_id", "query", "top_k"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_by_filter",
                "description": "Search movies with explicit filters like genre, year range. Useful for specific requirements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query with genre/year constraints (e.g., 'sci-fi movies from 2000-2010')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5)"
                        }
                    },
                    "required": ["query", "top_k"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_by_mood",
                "description": "Get movies matching a specific mood/atmosphere (e.g., 'relaxing', 'thrilling', 'romantic').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mood": {
                            "type": "string",
                            "description": "Mood or atmosphere (e.g., 'relaxing', 'action-packed', 'thought-provoking')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of movies to return (default 5)"
                        }
                    },
                    "required": ["mood", "top_k"]
                }
            }
        }
    ]
    
    SYSTEM_PROMPT = """你是一个专业的电影推荐顾问 AI Agent。你的目标是根据用户的需求，使用可用的工具为用户找到最合适的电影推荐。

你拥有以下能力：
1. 获取用户的观影历史和偏好
2. 推荐热门/流行电影（冷启动场景）
3. 基于用户偏好的个性化推荐
4. 根据特定条件（类型、年份等）的精确搜索
5. 根据心情/氛围的推荐

你的工作流程：
1. 首先理解用户的查询需求
2. 根据需求选择合适的工具
3. 调用工具获取推荐结果
4. 根据结果进行综合分析和排序
5. 用中文给出有洞察的推荐和解释

推荐的关键原则：
- 多角度考虑（相似度、用户偏好、流行度）
- 给出清晰的推荐理由
- 如果用户信息不足，先获取用户资料
- 根据上下文智能选择推荐策略

在调用工具时，确保参数有效。在返回最终结果前，进行推理分析。"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus"):
        """
        Initialize Qwen Agent.
        
        Args:
            api_key: Aliyun DashScope API key (default from DASHSCOPE_API_KEY env var)
            model: Qwen model to use (qwen-turbo, qwen-plus, qwen-max, etc.)
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not set. Please provide API key or set env var.")
        
        dashscope.api_key = self.api_key
        self.model = model
        self.basic_recommender = BasicRecommender()
        self.explanation_engine = ExplanationEngine(self.basic_recommender)
        
        # Session storage for multi-turn conversation history
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool and return result"""
        try:
            if tool_name == "get_user_profile":
                return self._tool_get_user_profile(tool_input.get("user_id"))
            elif tool_name == "search_cold_start":
                return self._tool_search_cold_start(tool_input.get("top_k", 5))
            elif tool_name == "search_by_preference":
                return self._tool_search_by_preference(
                    tool_input.get("user_id"),
                    tool_input.get("query"),
                    tool_input.get("top_k", 5)
                )
            elif tool_name == "search_by_filter":
                return self._tool_search_by_filter(
                    tool_input.get("query"),
                    tool_input.get("top_k", 5)
                )
            elif tool_name == "search_by_mood":
                return self._tool_search_by_mood(
                    tool_input.get("mood"),
                    tool_input.get("top_k", 5)
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    status="error",
                    data=None,
                    message=f"Unknown tool: {tool_name}"
                )
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                status="error",
                data=None,
                message=f"Error: {str(e)}"
            )
    
    def _tool_get_user_profile(self, user_id: int) -> ToolResult:
        """Get user's watch history and profile"""
        try:
            user_movies = self.basic_recommender.get_user_movies(user_id)
            if not user_movies:
                return ToolResult(
                    tool_name="get_user_profile",
                    status="success",
                    data={"user_id": user_id, "movies_count": 0, "status": "new_user"},
                    message="User is new with no watch history"
                )
            return ToolResult(
                tool_name="get_user_profile",
                status="success",
                data={
                    "user_id": user_id,
                    "movies_count": len(user_movies),
                    "sample_movies": user_movies[:5],
                    "status": "existing_user"
                },
                message=f"User {user_id} has watched {len(user_movies)} movies"
            )
        except Exception as e:
            return ToolResult(
                tool_name="get_user_profile",
                status="error",
                data=None,
                message=f"Failed to get user profile: {str(e)}"
            )
    
    def _tool_search_cold_start(self, top_k: int) -> ToolResult:
        """Get popular movies (cold start recommendation)"""
        try:
            results = self.basic_recommender.cold_start_recommend(top_k)
            return ToolResult(
                tool_name="search_cold_start",
                status="success",
                data={
                    "movies": [asdict(r) for r in results],
                    "count": len(results),
                    "strategy": "Popular movies for new users"
                },
                message=f"Found {len(results)} popular movies"
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_cold_start",
                status="error",
                data=None,
                message=f"Cold start search failed: {str(e)}"
            )
    
    def _tool_search_by_preference(self, user_id: int, query: str, top_k: int) -> ToolResult:
        """Get personalized recommendations based on user + query"""
        try:
            results = self.basic_recommender.recommend(user_id, query, top_k)
            return ToolResult(
                tool_name="search_by_preference",
                status="success",
                data={
                    "movies": [asdict(r) for r in results],
                    "count": len(results),
                    "strategy": "Personalized based on user similarity + semantic search"
                },
                message=f"Found {len(results)} personalized recommendations"
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_by_preference",
                status="error",
                data=None,
                message=f"Personalized search failed: {str(e)}"
            )
    
    def _tool_search_by_filter(self, query: str, top_k: int) -> ToolResult:
        """Search with genre/year filters"""
        try:
            # Use db_filter branch logic from BasicRecommender
            results = self.basic_recommender.recommend_by_filter(query, top_k)
            return ToolResult(
                tool_name="search_by_filter",
                status="success",
                data={
                    "movies": [asdict(r) for r in results],
                    "count": len(results),
                    "strategy": "Filtered search by genre/year/constraints"
                },
                message=f"Found {len(results)} movies matching filters"
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_by_filter",
                status="error",
                data=None,
                message=f"Filter search failed: {str(e)}"
            )
    
    def _tool_search_by_mood(self, mood: str, top_k: int) -> ToolResult:
        """Search by mood/atmosphere"""
        try:
            results = self.basic_recommender.recommend_by_mood(mood, top_k)
            return ToolResult(
                tool_name="search_by_mood",
                status="success",
                data={
                    "movies": [asdict(r) for r in results],
                    "count": len(results),
                    "strategy": f"Movies matching mood: {mood}"
                },
                message=f"Found {len(results)} movies for mood '{mood}'"
            )
        except Exception as e:
            return ToolResult(
                tool_name="search_by_mood",
                status="error",
                data=None,
                message=f"Mood search failed: {str(e)}"
            )
    
    def recommend(
        self, 
        user_id: Optional[int] = None, 
        query: str = "", 
        top_k: int = 5,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate movie recommendations using Qwen Agent with Tool Calling.
        
        Args:
            user_id: Optional user ID for personalized recommendations
            query: User's query/request in natural language
            top_k: Number of recommendations to return
            session_id: Session ID to maintain conversation history
        
        Returns:
            Dict with keys: route, decision_reason, results[], explanations[]
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        history = self.sessions[session_id]
        
        # Build prompt
        user_message = self._build_user_message(user_id, query, top_k)
        
        # Add to conversation history
        history.append({
            "role": "user",
            "content": user_message
        })
        
        # Main Tool Calling Loop
        all_results = []
        decision_reason = ""
        route = "llm_agent"
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call Qwen with tools
            response = self._call_qwen(history)
            
            if response is None:
                break
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # No more tool calls, extract final answer
                decision_reason = response.get("content", "")
                
                # Only add if it's not empty, representing final reasoning
                if decision_reason:
                    history.append({
                        "role": "assistant",
                        "content": decision_reason
                    })
                break
            
            # Execute tools
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("arguments", {})
                
                logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                result = self._execute_tool(tool_name, tool_input)
                tool_results.append(result)
                
                if result.status == "success":
                    if isinstance(result.data, dict) and "movies" in result.data:
                        all_results = result.data["movies"]
            
            # Add assistant response to history (it selected a tool)
            history.append({
                "role": "assistant",
                "content": response.get("content", "") or f"I am searching using tool {','.join([tc.get('name') for tc in tool_calls])}."
            })
            
            # Add tool results to history
            tool_results_text = self._format_tool_results(tool_results)
            history.append({
                "role": "user",
                "content": f"Tool execution results:\n{tool_results_text}\nPlease continue based on these results."
            })
        
        # Limit history length to prevent context explosion
        if len(history) > 20:
            # Keep first message and last 10 messages
            self.sessions[session_id] = history[:1] + history[-10:]
        
        # Convert to API response format
        results = []
        explanations = []
        
        if all_results:
            for movie_data in all_results:
                results.append({
                    "movie_id": movie_data.get("movie_id"),
                    "title": movie_data.get("title"),
                    "genres": movie_data.get("genres", []),
                    "release_year": movie_data.get("release_year"),
                    "score": movie_data.get("score", 0),
                    "components": {
                        "user_sim": movie_data.get("user_sim", 0),
                        "rag_sim": movie_data.get("rag_sim", 0),
                        "popularity": movie_data.get("popularity", 0),
                    },
                    "reasons": []
                })
            
            # Generate explanations
            try:
                rec_list = [
                    {
                        "movie_id": r.get("movie_id"),
                        "title": r.get("title"),
                        "user_sim": r.get("components", {}).get("user_sim", 0),
                        "rag_sim": r.get("components", {}).get("rag_sim", 0),
                        "popularity": r.get("components", {}).get("popularity", 0)
                    }
                    for r in results
                ]
                explanations_obj = self.explanation_engine.explain_batch(user_id, rec_list)
                explanations = [e.to_dict() for e in explanations_obj]
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {str(e)}")
        
        return {
            "route": route,
            "decision_reason": decision_reason,
            "results": results,
            "explanations": explanations
        }
    
    def _build_user_message(self, user_id: Optional[int], query: str, top_k: int) -> str:
        """Build the initial user message for the Agent"""
        parts = []
        
        if user_id is not None:
            parts.append(f"用户ID: {user_id}")
        
        parts.append(f"用户查询: {query}")
        parts.append(f"推荐数量: {top_k}")
        parts.append("\n请根据用户需求，使用合适的工具为用户提供电影推荐。")
        
        return "\n".join(parts)
    
    def _call_qwen(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call Qwen API with tools"""
        try:
            response = Generation.call(
                model=self.model,
                messages=history,
                tools=self.TOOLS,
                system=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.8
            )
            
            if response.status_code != 200:
                error_msg = f"Qwen API error: {response.code} - {response.message}"
                logger.error(error_msg)
                return {"content": f"API Request Failed: {error_msg}", "tool_calls": None}
            
            # Extract response content
            message = response.output.choices[0].message
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', None)
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
        except Exception as e:
            error_msg = f"Qwen API call failed: {str(e)}"
            logger.error(error_msg)
            return {"content": error_msg, "tool_calls": None}
    
    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from Qwen response"""
        tool_calls = []
        
        # Check if response contains tool_calls
        if response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                try:
                    if isinstance(tool_call, dict):
                        func_obj = tool_call.get('function', {})
                        tool_name = func_obj.get('name', '')
                        args_str = func_obj.get('arguments', '{}')
                    else:
                        tool_name = getattr(tool_call.function, 'name', '')
                        args_str = getattr(tool_call.function, 'arguments', '{}')
                    
                    # Parse arguments
                    if isinstance(args_str, str):
                        arguments = json.loads(args_str)
                    else:
                        arguments = args_str
                    
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": arguments
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {str(e)}")
        
        return tool_calls
    
    def _format_tool_results(self, tool_results: List[ToolResult]) -> str:
        """Format tool results for adding back to conversation history"""
        formatted = []
        for result in tool_results:
            status_icon = "✅" if result.status == "success" else "❌"
            formatted.append(f"{status_icon} {result.tool_name}: {result.message}")
            if result.status == "success" and result.data:
                formatted.append(f"  数据: {json.dumps(result.data, ensure_ascii=False, indent=2)[:500]}")
        return "\n".join(formatted)


def get_qwen_agent(api_key: Optional[str] = None, model: str = "qwen-plus") -> QwenAgent:
    """Factory function to get or create QwenAgent singleton"""
    if not hasattr(get_qwen_agent, "_instance"):
        get_qwen_agent._instance = QwenAgent(api_key=api_key, model=model)
    return get_qwen_agent._instance

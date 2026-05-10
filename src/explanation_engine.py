"""
Phase 7: Recommendation Explanation Engine
生成推荐解释，包括特征贡献度、用户历史证据、相似项推荐链
"""
from src.basic_recommender import BasicRecommender
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


@dataclass
class ExplanationScore:
    """特征贡献度分数"""
    user_similarity: float      # 用户相似度贡献
    rag_similarity: float       # RAG语义相似度贡献
    popularity: float           # 热度贡献
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Explanation:
    """完整的推荐解释"""
    movie_id: int
    title: str
    scores: ExplanationScore
    features: List[str]                    # 1-2条核心理由
    evidence: List[Dict]                   # 用户看过的相似电影证据
    similar_titles: List[str]              # 相关推荐链（如果喜欢XX也会喜欢YY）
    summary: str                           # 简洁中文总结
    
    def to_dict(self) -> Dict:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "scores": self.scores.to_dict(),
            "features": self.features,
            "evidence": self.evidence,
            "similar_titles": self.similar_titles,
            "summary": self.summary
        }


class ExplanationEngine:
    """推荐解释引擎"""
    
    def __init__(self, recommender: "BasicRecommender"):
        """
        Args:
            recommender: BasicRecommender 实例（包含电影、用户、嵌入数据）
        """
        self.recommender = recommender
        self.movies = {int(mid): record for mid, record in zip(recommender.movie_ids, recommender.movie_records)}
    
    def explain(self, user_id: Optional[int], movie_id: int, 
                user_sim: float, rag_sim: float, popularity: float,
                weights: Tuple[float, float, float] = (0.25, 0.25, 0.5)) -> Explanation:
        """
        为单个推荐生成解释
        
        Args:
            user_id: 用户ID（可选）
            movie_id: 推荐的电影ID
            user_sim: 用户相似度分数 [0, 1]
            rag_sim: RAG相似度分数 [0, 1]
            popularity: 热度分数 [0, 1]
            weights: (w_user, w_rag, w_popularity)
        
        Returns:
            Explanation对象
        """
        movie = self.recommender.movie_lookup.get(movie_id)
        if not movie:
            title = f"Movie {movie_id}"
            genres = []
        else:
            title = movie.title
            genres = movie.genres
        
        # 1. 特征贡献度分数
        w1, w2, w3 = weights
        scores = ExplanationScore(
            user_similarity=round(user_sim, 3),
            rag_similarity=round(rag_sim, 3),
            popularity=round(popularity, 3)
        )
        
        # 2. 核心理由（1-2条）
        features = self._extract_features(
            user_sim, rag_sim, popularity, weights, genres, user_id, movie_id
        )
        
        # 3. 用户历史证据
        evidence = self._find_evidence(user_id, movie_id, genres) if user_id else []
        
        # 4. 相似项推荐链
        similar_titles = self._find_similar_items(movie_id)
        
        # 5. 简洁总结
        summary = self._generate_summary(title, features, evidence)
        
        return Explanation(
            movie_id=movie_id,
            title=title,
            scores=scores,
            features=features,
            evidence=evidence,
            similar_titles=similar_titles,
            summary=summary
        )
    
    def _extract_features(self, user_sim: float, rag_sim: float, popularity: float,
                         weights: Tuple[float, float, float], 
                         genres: List[str], user_id: Optional[int], movie_id: int) -> List[str]:
        """
        提取1-2条核心理由
        """
        w1, w2, w3 = weights
        features = []
        
        # 按权重排序特征
        candidates = [
            (user_sim * w1, f"与你的品味相似（相似度{user_sim:.1%}）"),
            (rag_sim * w2, f"与你关注的内容相关（相似度{rag_sim:.1%}）"),
            (popularity * w3, f"热门且高分推荐（热度{popularity:.1%}）")
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 选择Top 1-2
        for i in range(min(2, len(candidates))):
            if candidates[i][0] > 0.05:  # 只包含有效贡献
                features.append(candidates[i][1])
        
        # 如果特征太少，补充流派信息
        if len(features) < 1:
            features.append(f"推荐的{','.join(genres[:2]) if genres else '电影'}")
        
        return features[:2]
    
    def _find_evidence(self, user_id: Optional[int], movie_id: int, target_genres: List[str]) -> List[Dict]:
        """
        找用户看过的相似电影作为证据
        """
        evidence = []
        
        if not user_id:
            return evidence
        
        try:
            user_profile = self.recommender._get_user_profile(user_id)
            if not user_profile:
                return evidence
            
            # 获取用户看过的电影
            watched_movies = user_profile.get("positive_movies", [])[:5]
            
            # 对比流派，找共同点
            for movie_id_watched in watched_movies[:3]:
                movie = self.recommender.movie_lookup.get(int(movie_id_watched))
                if not movie:
                    continue
                    
                movie_genres = movie.genres
                movie_title = movie.title
                
                # 计算流派重叠
                overlap = set(target_genres) & set(movie_genres)
                if overlap:
                    evidence.append({
                        "title": movie_title,
                        "reason": f"你看过的{','.join(list(overlap)[:1])}电影"
                    })
        except Exception as e:
            pass
        
        return evidence[:2]
    
    def _find_similar_items(self, movie_id: int, top_k: int = 2) -> List[str]:
        """
        找推荐链（类似电影）
        """
        similar = []
        try:
            movie = self.recommender.movie_lookup.get(movie_id)
            if not movie:
                return similar
                
            target_genres = movie.genres
            
            # 从相同流派中找其他电影
            for mid in self.recommender.movie_order:
                if mid == movie_id:
                    continue
                m = self.recommender.movie_lookup.get(mid)
                if not m:
                    continue
                if set(m.genres) & set(target_genres):
                    similar.append(m.title)
                    if len(similar) >= top_k:
                        break
        except Exception as e:
            pass
        
        return similar[:2]
    
    def _generate_summary(self, title: str, features: List[str], evidence: List[Dict]) -> str:
        """
        生成简洁的中文总结（<100字）
        """
        main_reason = features[0] if features else "根据你的偏好"
        
        summary = f"推荐{title}，{main_reason}。"
        
        if evidence:
            ev_reason = evidence[0].get("reason", "")
            summary += f"{ev_reason}。"
        
        if len(summary) > 100:
            summary = summary[:97] + "..."
        
        return summary
    
    def explain_batch(self, user_id: Optional[int], recommendations: List[Dict], 
                     weights: Tuple[float, float, float] = (0.25, 0.25, 0.5)) -> List[Explanation]:
        """
        批量为推荐列表生成解释
        
        Args:
            user_id: 用户ID（可选）
            recommendations: [{"movie_id": int, "title": str, "user_sim": float, "rag_sim": float, "popularity": float}, ...]
            weights: 权重配置
        
        Returns:
            Explanation列表
        """
        explanations = []
        for rec in recommendations:
            exp = self.explain(
                user_id=user_id,
                movie_id=rec["movie_id"],
                user_sim=rec.get("user_sim", 0.0),
                rag_sim=rec.get("rag_sim", 0.0),
                popularity=rec.get("popularity", 0.0),
                weights=weights
            )
            explanations.append(exp)
        
        return explanations

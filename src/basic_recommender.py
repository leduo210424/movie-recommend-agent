from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.chroma_store import ChromaMovieStore


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USER_SIM_WEIGHT = 0.30
RAG_SIM_WEIGHT = 0.40
POPULARITY_WEIGHT = 0.30
DEFAULT_POOL_SIZE = 50


@dataclass
class MovieCandidate:
    movie_id: int
    title: str
    genres: List[str]
    release_year: Optional[float]
    avg_rating: float
    rating_count: int
    embedding: np.ndarray


@dataclass
class RecommendationResult:
    movie_id: int
    title: str
    genres: List[str]
    release_year: Optional[float]
    user_sim: float
    rag_sim: float
    popularity: float
    score: float
    reasons: List[str]


def load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {file_path}")
        return data
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_movie_stats(full_data_file: str | Path) -> Dict[int, Dict[str, float]]:
    records = load_json_records(full_data_file)
    if not records:
        raise ValueError(f"No records found in {full_data_file}")

    rating_sum: Dict[int, float] = {}
    rating_count: Dict[int, int] = {}
    for record in records:
        movie_id = int(record["movie_id"])
        rating = float(record["rating"])
        rating_sum[movie_id] = rating_sum.get(movie_id, 0.0) + rating
        rating_count[movie_id] = rating_count.get(movie_id, 0) + 1

    stats: Dict[int, Dict[str, float]] = {}
    for movie_id, total_rating in rating_sum.items():
        count = rating_count[movie_id]
        stats[movie_id] = {
            "avg_rating": total_rating / max(count, 1),
            "rating_count": float(count),
        }
    return stats


def build_embedding_text(movie: Dict[str, Any]) -> str:
    title = str(movie.get("title", "")).strip()
    genres = movie.get("genres", [])
    if isinstance(genres, (list, tuple)):
        genre_text = ", ".join(str(item) for item in genres if str(item).strip())
    else:
        genre_text = str(genres)
    release_year = movie.get("release_year")
    year_text = "unknown"
    if release_year is not None and str(release_year) != "nan":
        try:
            year_text = str(int(float(release_year)))
        except (TypeError, ValueError):
            year_text = str(release_year)
    return f"Title: {title}. Genres: {genre_text}. Release year: {year_text}."


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def normalize_to_unit(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    clipped = min(high, max(low, value))
    return (clipped - low) / (high - low)


class BasicRecommender:
    def __init__(
        self,
        movie_file: str | Path = "data/processed/movies.json",
        full_data_file: str | Path = "data/processed/full_data.json",
        movie_embeddings_file: str | Path = "data/processed/movie_embeddings.npy",
        movie_ids_file: str | Path = "data/processed/movie_ids.npy",
        user_profiles_file: str | Path = "data/processed/user_profiles.jsonl",
        user_embeddings_file: str | Path = "data/processed/user_embeddings.npy",
        user_ids_file: str | Path = "data/processed/user_ids.npy",
        movie_index_meta_file: str | Path = "data/processed/movie_index_meta.json",
        plot_index_file: str | Path = "data/processed/movie_plot_index.faiss",
        attr_index_file: str | Path = "data/processed/movie_attr_index.faiss",
        model_name: Optional[str] = None,
        chroma_store: Optional[ChromaMovieStore] = None,
    ) -> None:

        # ── ChromaDB 后端 (新) ──
        self.chroma_store = chroma_store

        # ── 元数据层 (FAISS 和 ChromaDB 共享) ──
        self.movie_records = load_json_records(movie_file)
        if not self.movie_records:
            raise ValueError(f"No movie metadata found in {movie_file}")

        self.movie_stats = load_movie_stats(full_data_file)
        self.movie_lookup: Dict[int, MovieCandidate] = {}
        self.movie_order: List[int] = []

        if chroma_store is not None:
            # ChromaDB 模式: 从 ChromaDB 构建 movie_lookup (不含 embedding)
            for record in self.movie_records:
                movie_id_int = int(record["movie_id"])
                stats = self.movie_stats.get(movie_id_int, {"avg_rating": 0.0, "rating_count": 0.0})
                candidate = MovieCandidate(
                    movie_id=movie_id_int,
                    title=str(record.get("title", "")),
                    genres=self._normalize_genres(record.get("genres")),
                    release_year=self._to_float_or_none(record.get("release_year")),
                    avg_rating=float(stats["avg_rating"]),
                    rating_count=int(stats["rating_count"]),
                    embedding=np.zeros(384, dtype=np.float32),  # dummy, embedding 查询走 ChromaDB
                )
                self.movie_lookup[movie_id_int] = candidate
                self.movie_order.append(movie_id_int)

            # 加载文本模型 (用于 query/profile 编码, 不做电影编码)
            if model_name is not None:
                self.model_name = model_name
            else:
                self.model_name = chroma_store._model_name
            self.text_model = SentenceTransformer(self.model_name)

            # FAISS 索引全部跳过
            self.use_multi_index = True  # ChromaDB 自带多粒度融合
            self.movie_embeddings = np.zeros((1, 384), dtype=np.float32)
            self.movie_ids = np.array([], dtype=np.int32)
            self.all_movie_embeddings = self.movie_embeddings
            self.all_movie_embeddings_norm = self.movie_embeddings
            self.plot_index = None
            self.attr_index = None
            self.movie_index_meta = {}

        else:
            # FAISS 模式 (原逻辑, 保持不变)
            self.movie_embeddings = np.load(movie_embeddings_file).astype(np.float32)
            self.movie_ids = np.load(movie_ids_file).astype(np.int32)
            if len(self.movie_records) != len(self.movie_embeddings) or len(self.movie_ids) != len(self.movie_embeddings):
                raise ValueError("Movie metadata, ids, and embeddings must have the same length.")

            for record, embedding, movie_id in zip(self.movie_records, self.movie_embeddings, self.movie_ids):
                movie_id_int = int(movie_id)
                stats = self.movie_stats.get(movie_id_int, {"avg_rating": 0.0, "rating_count": 0.0})
                candidate = MovieCandidate(
                    movie_id=movie_id_int,
                    title=str(record.get("title", "")),
                    genres=self._normalize_genres(record.get("genres")),
                    release_year=self._to_float_or_none(record.get("release_year")),
                    avg_rating=float(stats["avg_rating"]),
                    rating_count=int(stats["rating_count"]),
                    embedding=np.asarray(embedding, dtype=np.float32),
                )
                self.movie_lookup[movie_id_int] = candidate
                self.movie_order.append(movie_id_int)

            self.movie_index_meta = self._load_json(movie_index_meta_file)
            self.model_name = model_name or str(self.movie_index_meta.get("model", DEFAULT_MODEL))
            self.text_model = SentenceTransformer(self.model_name)

            self.all_movie_embeddings = self.movie_embeddings.astype(np.float32)
            self.all_movie_embeddings_norm = self.all_movie_embeddings / np.clip(
                np.linalg.norm(self.all_movie_embeddings, axis=1, keepdims=True),
                1e-12,
                None,
            )

            # 多粒度索引
            self.plot_index = self._load_faiss_index(plot_index_file)
            self.attr_index = self._load_faiss_index(attr_index_file)
            self.use_multi_index = self.plot_index is not None and self.attr_index is not None

        # Bayesian 平滑热度：避免评分人数少的冷门片 pop=1.0 挤占真正热门电影
        all_ratings = [c.avg_rating for c in self.movie_lookup.values() if c.rating_count > 0]
        all_counts = [c.rating_count for c in self.movie_lookup.values() if c.rating_count > 0]
        self._global_mean_rating = float(np.mean(all_ratings)) if all_ratings else 3.0
        self._global_mean_count = float(np.mean(all_counts)) if all_counts else 10.0
        self._bayesian_pop: Dict[int, float] = {}
        for mid, movie in self.movie_lookup.items():
            C = self._global_mean_count
            m = self._global_mean_rating
            self._bayesian_pop[mid] = (C * m + movie.rating_count * movie.avg_rating) / (C + movie.rating_count)

        self.user_profiles = self._load_user_profiles(user_profiles_file)
        self.user_embeddings = np.load(user_embeddings_file).astype(np.float32)
        self.user_ids = np.load(user_ids_file).astype(np.int32)
        self.user_lookup: Dict[int, np.ndarray] = {
            int(user_id): np.asarray(embedding, dtype=np.float32)
            for user_id, embedding in zip(self.user_ids, self.user_embeddings)
        }

    @staticmethod
    def _load_json(path: str | Path) -> Dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {}
        return json.loads(file_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_faiss_index(path: str | Path) -> Optional[faiss.Index]:
        file_path = Path(path)
        if not file_path.exists():
            return None
        return faiss.read_index(str(file_path))

    @staticmethod
    def _load_user_profiles(path: str | Path) -> Dict[int, Dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Profile file not found: {file_path}")

        profile_index: Dict[int, Dict[str, Any]] = {}
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            profile = json.loads(line)
            profile_index[int(profile["user_id"])] = profile
        return profile_index

    @staticmethod
    def _normalize_genres(genres: Any) -> List[str]:
        if genres is None:
            return []
        if isinstance(genres, (list, tuple)):
            return [str(item) for item in genres if str(item).strip()]
        return [str(genres)]

    @staticmethod
    def _to_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result):
            return None
        return result

    @staticmethod
    def _normalize_rating(rating: float) -> float:
        return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))

    def _build_profile_text_embedding(self, profile_text: str) -> np.ndarray:
        embedding = self.text_model.encode(
            [profile_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return np.asarray(embedding, dtype=np.float32)

    def _build_context_embedding(self, profile_text: str, query_text: Optional[str] = None) -> np.ndarray:
        profile_embedding = self._build_profile_text_embedding(profile_text)
        if not query_text:
            return profile_embedding

        # 中文查询 → 翻译为英文后编码 (英文模型直接编码中文 = 噪声)
        query_for_encode = self._translate_query(query_text)

        query_embedding = self.text_model.encode(
            [query_for_encode],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        mixed = 0.6 * profile_embedding + 0.4 * query_embedding
        norm = float(np.linalg.norm(mixed))
        if norm == 0.0:
            return profile_embedding
        return (mixed / norm).astype(np.float32)

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """检测文本是否包含中文字符"""
        return any('一' <= c <= '鿿' for c in text)

    @classmethod
    def _translate_query(cls, query: str) -> str:
        """将中文查询中的情绪/类型词替换为英文，确保英文模型的编码质量。

        非中文查询原样返回。中文混合查询 (如 '科幻动作') 也尝试翻译类型词。
        """
        if not cls._has_chinese(query):
            return query

        # 中文→英文翻译表 (情绪词 + 类型词)
        translations = {
            # 情绪词 (复用 _MOOD_TO_ENGLISH 的映射)
            "轻松": "lighthearted relaxing feel-good",
            "搞笑": "funny comedy humorous laugh",
            "紧张": "tense suspenseful thrilling",
            "治愈": "healing heartwarming comfort",
            "烧脑": "mind-bending complex puzzle",
            "压抑": "dark bleak depressing",
            "刺激": "exciting thrilling intense",
            "温暖": "warm heartwarming cozy",
            "悲伤": "sad tragic emotional",
            "欢乐": "joyful happy cheerful fun",
            "热血": "inspiring passionate epic",
            "恐怖": "horror scary terrifying",
            "悬疑": "mystery suspense thriller",
            "浪漫": "romantic love story",
            "科幻": "science fiction futuristic",
            # 类型词
            "动作": "action martial arts",
            "喜剧": "comedy funny humorous",
            "剧情": "drama character-driven",
            "爱情": "romance love story",
            "惊悚": "thriller suspense",
            "冒险": "adventure exploration",
            "奇幻": "fantasy magical",
            "动画": "animation animated",
            "纪录片": "documentary factual",
            "犯罪": "crime criminal",
            "战争": "war military",
            # 抽象语义
            "推荐好看的": "popular highly-rated must-watch",
            "让人想旅行": "travel adventure wanderlust road movie exploration journey",
            "电影": "movie",
        }

        result = query
        # 按 key 长度降序替换 (长词优先), 前后加空格防止粘连
        for zh, en in sorted(translations.items(), key=lambda x: -len(x[0])):
            if zh in result:
                result = result.replace(zh, f" {en} ")
        # 清理多余空格
        result = " ".join(result.split())
        return result if result != query else f"{query} movie"  # 最少加个 movie 兜底

    def _get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self.user_profiles.get(int(user_id))

    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        embedding = self.user_lookup.get(int(user_id))
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)

    def _popular_movie_ids(self, limit: int) -> List[int]:
        ordered = sorted(
            self.movie_lookup.values(),
            key=lambda item: self._bayesian_pop.get(item.movie_id, 0),
            reverse=True,
        )
        return [item.movie_id for item in ordered[:limit]]

    def _watched_movie_ids(self, user_id: int) -> set[int]:
        profile = self._get_user_profile(user_id)
        if not profile:
            return set()
        watched = set()
        for item in profile.get("recent_movies", []):
            watched.add(int(item["movie_id"]))
        for item in profile.get("positive_movies", []):
            watched.add(int(item["movie_id"]))
        for item in profile.get("disliked_movies", []):
            watched.add(int(item["movie_id"]))
        return watched

    def _favorite_genre_set(self, user_id: int) -> List[str]:
        profile = self._get_user_profile(user_id)
        if not profile:
            return []
        genres = profile.get("favorite_genres", [])
        ordered = [str(item.get("genre", "")) for item in genres if str(item.get("genre", "")).strip()]
        return ordered

    def _genre_overlap(self, movie: MovieCandidate, favorite_genres: List[str]) -> float:
        if not favorite_genres:
            return 0.0
        movie_genres = set(movie.genres)
        overlap = sum(1 for genre in favorite_genres if genre in movie_genres)
        return overlap / max(len(favorite_genres), 1)

    def _multi_index_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> List[int]:
        """
        跨多粒度索引做加权融合检索。

        weights: (full_index_weight, plot_index_weight, attr_index_weight)
        Returns: merged movie_ids 列表，按融合分数降序排列
        """
        # ── ChromaDB 后端 ──
        if self.chroma_store is not None:
            results = self.chroma_store.search(
                query_text="",           # 有预计算向量时不需文本
                query_embedding=query_embedding,
                top_k=top_k,
                weights=weights,
            )
            return [r["movie_id"] for r in results]

        # ── FAISS 后端 (原逻辑) ──
        q = query_embedding.reshape(1, -1).astype(np.float32)
        candidates: Dict[int, float] = {}

        # full 索引（综合语义 + 属性）
        full_scores = self.all_movie_embeddings_norm @ q[0]
        full_top_k = min(top_k, len(full_scores))
        full_top_indices = np.argpartition(-full_scores, full_top_k - 1)[:full_top_k]
        for idx in full_top_indices:
            mid = int(self.movie_ids[idx])
            candidates[mid] = candidates.get(mid, 0.0) + float(full_scores[idx]) * weights[0]

        # plot 索引（剧情语义）
        if self.plot_index is not None:
            plot_scores, plot_indices = self.plot_index.search(q, top_k)
            for i in range(plot_indices.shape[1]):
                idx = plot_indices[0, i]
                if idx < 0 or idx >= len(self.movie_ids):
                    continue
                mid = int(self.movie_ids[idx])
                candidates[mid] = candidates.get(mid, 0.0) + float(plot_scores[0, i]) * weights[1]

        # attr 索引（属性标签）
        if self.attr_index is not None:
            attr_scores, attr_indices = self.attr_index.search(q, top_k)
            for i in range(attr_indices.shape[1]):
                idx = attr_indices[0, i]
                if idx < 0 or idx >= len(self.movie_ids):
                    continue
                mid = int(self.movie_ids[idx])
                candidates[mid] = candidates.get(mid, 0.0) + float(attr_scores[0, i]) * weights[2]

        merged = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        return [mid for mid, _ in merged]

    def _candidate_pool(self, user_id: int, pool_size: int, context_embedding: Optional[np.ndarray] = None,
                        exclude_ids: Optional[set] = None) -> List[int]:
        profile = self._get_user_profile(user_id)
        watched_ids = exclude_ids if exclude_ids is not None else self._watched_movie_ids(user_id)
        favorite_genres = self._favorite_genre_set(user_id)
        popular_ids = self._popular_movie_ids(pool_size)

        candidate_ids: List[int] = []
        candidate_ids.extend(popular_ids)

        for movie in self.movie_lookup.values():
            if movie.movie_id in watched_ids:
                continue
            if self._genre_overlap(movie, favorite_genres) > 0:
                candidate_ids.append(movie.movie_id)

        if profile is not None:
            user_embedding = self._get_user_embedding(user_id)
            if context_embedding is None:
                context_embedding = self._build_profile_text_embedding(str(profile.get("profile_text", "")))

            if self.use_multi_index:
                # 多粒度融合检索：分别从 3 个索引召回并加权融合
                if user_embedding is not None:
                    candidate_ids.extend(
                        self._multi_index_search(user_embedding, pool_size, weights=(0.5, 0.3, 0.2))
                    )
                candidate_ids.extend(
                    self._multi_index_search(context_embedding, pool_size, weights=(0.4, 0.35, 0.25))
                )
            else:
                # 回退到单索引检索（向后兼容）
                if user_embedding is not None:
                    query = user_embedding.reshape(1, -1)
                    similarities = self.all_movie_embeddings_norm @ (query[0] / max(np.linalg.norm(query[0]), 1e-12))
                    top_indices = np.argsort(-similarities)[:pool_size]
                    candidate_ids.extend(int(self.movie_ids[index]) for index in top_indices)

                text_similarities = self.all_movie_embeddings_norm @ context_embedding
                top_text_indices = np.argsort(-text_similarities)[:pool_size]
                candidate_ids.extend(int(self.movie_ids[index]) for index in top_text_indices)

        filtered = []
        seen = set()
        for movie_id in candidate_ids:
            if movie_id in seen or movie_id in watched_ids:
                continue
            seen.add(movie_id)
            filtered.append(movie_id)
        return filtered[: max(pool_size * 2, pool_size)]

    def recommend(
        self,
        user_id: Optional[int],
        query_text: str = "",
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL_SIZE,
        exclude_ids: Optional[set] = None,
    ) -> List[RecommendationResult]:
        if user_id is None or self._get_user_profile(user_id) is None:
            return self._cold_start_recommend(top_k=top_k, exclude_ids=exclude_ids)

        user_profile = self._get_user_profile(user_id)
        user_embedding = self._get_user_embedding(user_id)
        if user_profile is None or user_embedding is None:
            return self._cold_start_recommend(top_k=top_k, exclude_ids=exclude_ids)

        profile_text = str(user_profile.get("profile_text", ""))
        context_embedding = self._build_context_embedding(profile_text, query_text=query_text or None)
        favorite_genres = self._favorite_genre_set(user_id)
        candidate_ids = self._candidate_pool(user_id, pool_size=pool_size, context_embedding=context_embedding,
                                              exclude_ids=exclude_ids)

        profile_vectors = {
            "user": user_embedding,
            "rag": context_embedding,
        }

        results: List[RecommendationResult] = []
        for movie_id in candidate_ids:
            movie = self.movie_lookup.get(movie_id)
            if movie is None:
                continue

            user_sim = cosine_similarity(profile_vectors["user"], movie.embedding)
            rag_sim = cosine_similarity(profile_vectors["rag"], movie.embedding)
            popularity = normalize_to_unit(self._bayesian_pop.get(movie.movie_id, movie.avg_rating), 1.0, 5.0)
            score = (
                USER_SIM_WEIGHT * user_sim
                + RAG_SIM_WEIGHT * rag_sim
                + POPULARITY_WEIGHT * popularity
            )

            reasons = self._build_reasons(movie, user_id, favorite_genres, user_sim, rag_sim, popularity)
            results.append(
                RecommendationResult(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    release_year=movie.release_year,
                    user_sim=user_sim,
                    rag_sim=rag_sim,
                    popularity=popularity,
                    score=score,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _cold_start_recommend(self, top_k: int, exclude_ids: Optional[set] = None) -> List[RecommendationResult]:
        popular_ids = self._popular_movie_ids(max(top_k * 2, top_k))
        skip = exclude_ids or set()
        results: List[RecommendationResult] = []
        for movie_id in popular_ids:
            if movie_id in skip:
                continue
            movie = self.movie_lookup[movie_id]
            popularity = normalize_to_unit(self._bayesian_pop.get(movie.movie_id, movie.avg_rating), 1.0, 5.0)
            score = POPULARITY_WEIGHT * popularity
            results.append(
                RecommendationResult(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    release_year=movie.release_year,
                    user_sim=0.0,
                    rag_sim=0.0,
                    popularity=popularity,
                    score=score,
                    reasons=["冷启动用户，优先使用高热度电影兜底"],
                )
            )
        return results[:top_k]

    # Public API for LLM Agent Tools
    def get_user_movies(self, user_id: int) -> List[str]:
        """Get user's watch history (public API for agents)"""
        watched_ids = self._watched_movie_ids(user_id)
        movies = []
        for movie_id in watched_ids:
            movie = self.movie_lookup.get(movie_id)
            if movie:
                movies.append(movie.title)
        return movies
    
    def cold_start_recommend(self, top_k: int) -> List[RecommendationResult]:
        """Public wrapper for cold start recommendations"""
        return self._cold_start_recommend(top_k)
    
    def recommend_by_filter(self, query: str, top_k: int = 5) -> List[RecommendationResult]:
        """
        Recommend movies by filtering on genre, year, or keywords.
        Parses query for genre/year patterns and filters accordingly.
        """
        # Extract genre keywords from query
        query_lower = query.lower()
        genre_keywords = {
            # English
            "action": "Action", "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi",
            "comedy": "Comedy", "drama": "Drama", "horror": "Horror",
            "romance": "Romance", "thriller": "Thriller", "adventure": "Adventure",
            "fantasy": "Fantasy", "animation": "Animation", "documentary": "Documentary",
            "mystery": "Mystery", "crime": "Crime", "war": "War",
            "western": "Western", "musical": "Musical", "children": "Children's",
            "sci": "Sci-Fi", "fiction": "Sci-Fi",
            # 中文
            "动作": "Action", "科幻": "Sci-Fi", "喜剧": "Comedy",
            "剧情": "Drama", "恐怖": "Horror", "爱情": "Romance",
            "惊悚": "Thriller", "冒险": "Adventure", "奇幻": "Fantasy",
            "动画": "Animation", "纪录片": "Documentary", "悬疑": "Mystery",
            "犯罪": "Crime", "战争": "War", "西部": "Western",
            "音乐": "Musical", "儿童": "Children's",
        }
        
        target_genres = []
        for keyword, genre in genre_keywords.items():
            if keyword in query_lower and genre not in target_genres:
                target_genres.append(genre)
        
        # Extract year range from query (e.g., "2000-2010", "after 2015")
        import re
        year_pattern = r"(\d{4})"
        years = re.findall(year_pattern, query)
        year_range = None
        if len(years) >= 2:
            year_range = (int(years[0]), int(years[1]))
        elif len(years) == 1:
            year = int(years[0])
            if "after" in query_lower:
                year_range = (year, 2025)
            elif "before" in query_lower:
                year_range = (1900, year)
            else:
                year_range = (year - 5, year + 5)
        
        # Filter movies
        candidates = []
        for movie in self.movie_lookup.values():
            # Check genre match
            if target_genres:
                if not any(g in movie.genres for g in target_genres):
                    continue
            
            # Check year match
            if year_range and movie.release_year:
                if not (year_range[0] <= movie.release_year <= year_range[1]):
                    continue
            
            candidates.append(movie)
        
        # Sort by popularity and return
        candidates.sort(
            key=lambda m: (m.avg_rating, m.rating_count),
            reverse=True
        )
        
        results = []
        for movie in candidates[:top_k]:
            popularity = normalize_to_unit(self._bayesian_pop.get(movie.movie_id, movie.avg_rating), 1.0, 5.0)
            score = POPULARITY_WEIGHT * popularity
            reasons = [f"符合过滤条件: {query}"]
            results.append(
                RecommendationResult(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    release_year=movie.release_year,
                    user_sim=0.0,
                    rag_sim=0.0,
                    popularity=popularity,
                    score=score,
                    reasons=reasons,
                )
            )
        
        return results
    
    # ── 情绪词映射 (中英双语 + 复合词拆分) ──
    _MOOD_TO_GENRES = {
        # 英文
        "relaxing":  ["Comedy", "Animation", "Children's", "Documentary"],
        "exciting":  ["Action", "Adventure", "Thriller", "Sci-Fi"],
        "thrilling": ["Thriller", "Horror", "Mystery", "Crime"],
        "romantic":  ["Romance", "Drama"],
        "sad":       ["Drama"],
        "funny":     ["Comedy"],
        "dark":      ["Horror", "Crime", "Thriller", "Drama"],
        "light":     ["Comedy", "Animation", "Children's", "Musical"],
        "thought-provoking": ["Drama", "Documentary", "Sci-Fi", "Mystery"],
        "action-packed":     ["Action", "Adventure", "Sci-Fi", "Thriller"],
        "adventure":         ["Adventure", "Action", "Fantasy", "Sci-Fi"],
        "fantasy":           ["Fantasy", "Adventure", "Animation"],
        "sci-fi":            ["Sci-Fi", "Adventure"],
        "horror":            ["Horror", "Thriller", "Mystery"],
        "crime":             ["Crime", "Thriller", "Mystery", "Drama"],
        "war":               ["War", "Action", "Drama"],
        "western":           ["Western", "Action", "Adventure"],
        # 中文情绪词 → 映射到英文 key 间接使用
        "轻松":   ["Comedy", "Animation", "Children's"],
        "搞笑":   ["Comedy"],
        "紧张":   ["Thriller", "Action"],
        "治愈":   ["Drama", "Animation", "Children's"],
        "烧脑":   ["Mystery", "Sci-Fi", "Thriller"],
        "压抑":   ["Drama", "Horror"],
        "刺激":   ["Action", "Thriller", "Adventure"],
        "温暖":   ["Drama", "Comedy", "Animation"],
        "悲伤":   ["Drama"],
        "欢乐":   ["Comedy", "Animation", "Musical"],
        "热血":   ["Action", "Adventure", "Sci-Fi"],
        "恐怖":   ["Horror", "Thriller"],
        "悬疑":   ["Mystery", "Thriller"],
        "浪漫":   ["Romance", "Drama"],
        "科幻":   ["Sci-Fi", "Adventure"],
    }

    # ── 中文情绪词 → 英文编码短语（避免英文模型编码中文的噪声）──
    _MOOD_TO_ENGLISH = {
        "轻松": "lighthearted relaxing feel-good",
        "搞笑": "funny comedy humorous laugh",
        "紧张": "tense suspenseful thrilling",
        "治愈": "healing heartwarming comfort wholesome",
        "烧脑": "mind-bending cerebral complex puzzle",
        "压抑": "oppressive dark bleak depressing",
        "刺激": "exciting adrenaline thrilling intense",
        "温暖": "warm heartwarming feel-good cozy",
        "悲伤": "sad tragic emotional tearjerker",
        "欢乐": "joyful happy cheerful fun celebration",
        "热血": "inspiring passionate epic motivational",
        "恐怖": "horror scary terrifying frightening",
        "悬疑": "mystery suspense thriller whodunit",
        "浪漫": "romantic love story heartwarming",
        "科幻": "science fiction futuristic space technology",
    }

    @classmethod
    def _split_compound_mood(cls, mood: str) -> List[str]:
        """拆分复合情绪词: '轻松搞笑' → ['轻松', '搞笑']"""
        # 按已知中文情绪词做贪心匹配拆分
        known = sorted(cls._MOOD_TO_GENRES.keys(), key=len, reverse=True)
        remaining = mood.strip()
        parts = []
        while remaining:
            matched = False
            for kw in known:
                if remaining.startswith(kw):
                    parts.append(kw)
                    remaining = remaining[len(kw):]
                    matched = True
                    break
            if not matched:
                # 跳过未识别的字符
                remaining = remaining[1:]
        return parts

    def recommend_by_mood(self, mood: str, top_k: int = 5) -> List[RecommendationResult]:
        """
        Recommend movies by mood/atmosphere.
        支持中英双语情绪词 + 复合词拆分 + 中文→英文翻译编码。
        """
        mood_clean = mood.strip()

        # 1. 拆分复合情绪词
        parts = self._split_compound_mood(mood_clean)
        if not parts:
            parts = [mood_clean]

        # 2. 合并所有匹配的类型
        target_genres = []
        for part in parts:
            genres = self._MOOD_TO_GENRES.get(part, [])
            if not genres:
                # 中文查不到试试英文 key (lower)
                genres = self._MOOD_TO_GENRES.get(part.lower(), [])
            target_genres.extend(genres)
        target_genres = list(dict.fromkeys(target_genres))  # 去重保序

        # 3. 翻译为英文短语再编码（英文模型处理中文 = 随机噪声）
        english_phrases = []
        for part in parts:
            eng = self._MOOD_TO_ENGLISH.get(part)
            if eng:
                english_phrases.append(eng)
            else:
                english_phrases.append(part)  # 未知词保持原样
        english_query = " ".join(english_phrases)

        # 4. 编码（用英文短语，确保语义空间有效性）
        mood_embedding = self.text_model.encode(
            [english_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)
        
        # Find candidates matching mood genres
        candidates = []
        for movie in self.movie_lookup.values():
            if target_genres:
                if not any(g in movie.genres for g in target_genres):
                    continue
            candidates.append(movie)
        
        # Score by semantic similarity to mood
        results = []
        for movie in candidates:
            rag_sim = cosine_similarity(mood_embedding, movie.embedding)
            popularity = normalize_to_unit(self._bayesian_pop.get(movie.movie_id, movie.avg_rating), 1.0, 5.0)
            score = 0.6 * rag_sim + 0.4 * popularity
            
            results.append(
                RecommendationResult(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    release_year=movie.release_year,
                    user_sim=0.0,
                    rag_sim=rag_sim,
                    popularity=popularity,
                    score=score,
                    reasons=[f"符合心情: {mood}"],
                )
            )
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def recommend_by_semantic(self, description: str, top_k: int = 10,
                               exclude_ids: Optional[set] = None) -> List[RecommendationResult]:
        """纯语义检索：用 LLM 重写后的描述直接做 embedding 搜索，不依赖用户画像或类型过滤。"""
        skip = exclude_ids or set()
        desc_embedding = self.text_model.encode(
            [description],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        # ── ChromaDB 后端 ──
        if self.chroma_store is not None:
            chroma_results = self.chroma_store.search(
                query_text=description,
                query_embedding=desc_embedding,
                top_k=max(top_k * 2, top_k),
            )
            results: List[RecommendationResult] = []
            for r in chroma_results:
                mid = r["movie_id"]
                if mid in skip:
                    continue
                movie = self.movie_lookup.get(mid)
                if movie is None:
                    continue
                rag_sim = r.get("fusion_score", 0.0)
                popularity = normalize_to_unit(self._bayesian_pop.get(mid, movie.avg_rating), 1.0, 5.0)
                score = 0.6 * rag_sim + 0.4 * popularity
                results.append(RecommendationResult(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    release_year=movie.release_year,
                    user_sim=0.0,
                    rag_sim=rag_sim,
                    popularity=popularity,
                    score=score,
                    reasons=[f"语义匹配: {description[:60]}"],
                ))
                if len(results) >= top_k:
                    break
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

        # ── FAISS 后端 (原逻辑) ──
        scores = self.all_movie_embeddings_norm @ desc_embedding
        top_indices = np.argsort(-scores)[: max(top_k * 2, top_k)]

        results: List[RecommendationResult] = []
        for idx in top_indices:
            mid = int(self.movie_ids[idx])
            if mid in skip:
                continue
            movie = self.movie_lookup.get(mid)
            if movie is None:
                continue
            rag_sim = float(scores[idx])
            popularity = normalize_to_unit(self._bayesian_pop.get(mid, movie.avg_rating), 1.0, 5.0)
            score = 0.6 * rag_sim + 0.4 * popularity
            results.append(RecommendationResult(
                movie_id=movie.movie_id,
                title=movie.title,
                genres=movie.genres,
                release_year=movie.release_year,
                user_sim=0.0,
                rag_sim=rag_sim,
                popularity=popularity,
                score=score,
                reasons=[f"语义匹配: {description[:60]}"],
            ))
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _build_reasons(
        self,
        movie: MovieCandidate,
        user_id: int,
        favorite_genres: List[str],
        user_sim: float,
        rag_sim: float,
        popularity: float,
    ) -> List[str]:
        reasons: List[str] = []
        if favorite_genres:
            matched_genres = [genre for genre in favorite_genres if genre in movie.genres]
            if matched_genres:
                reasons.append(f"匹配你的偏好类型: {', '.join(matched_genres[:3])}")
        if user_sim >= 0.55:
            reasons.append(f"与长期用户画像相似度较高 ({user_sim:.3f})")
        if rag_sim >= 0.55:
            reasons.append(f"与语义记忆相似度较高 ({rag_sim:.3f})")
        if popularity >= 0.7:
            reasons.append(f"历史热度较高 (normalized={popularity:.3f})")
        if not reasons:
            reasons.append("综合用户画像、语义记忆和热度后进入候选集")
        return reasons

    def format_results(self, results: List[RecommendationResult]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for item in results:
            payload.append(
                {
                    "movie_id": item.movie_id,
                    "title": item.title,
                    "genres": item.genres,
                    "release_year": item.release_year,
                    "score": round(item.score, 4),
                    "components": {
                        "user_sim": round(item.user_sim, 4),
                        "rag_sim": round(item.rag_sim, 4),
                        "popularity": round(item.popularity, 4),
                    },
                    "reasons": item.reasons,
                }
            )
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase 4 baseline recommender.")
    parser.add_argument("--user-id", type=int, default=None, help="Target user id. If omitted, cold start is used.")
    parser.add_argument("--top-k", type=int, default=10, help="How many recommendations to return.")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="Candidate pool size.")
    parser.add_argument("--output-json", action="store_true", help="Print JSON instead of human-readable output.")
    args = parser.parse_args()

    recommender = BasicRecommender()
    results = recommender.recommend(user_id=args.user_id, top_k=args.top_k, pool_size=args.pool_size)

    if args.output_json:
        print(json.dumps(recommender.format_results(results), ensure_ascii=False, indent=2))
        return

    if args.user_id is None:
        print("Cold start recommendations")
    else:
        print(f"Recommendations for user {args.user_id}")

    for rank, item in enumerate(results, start=1):
        print("=" * 72)
        print(f"{rank}. {item.title} ({item.release_year})")
        print(f"genres: {', '.join(item.genres)}")
        print(f"score: {item.score:.4f} | user_sim: {item.user_sim:.4f} | rag_sim: {item.rag_sim:.4f} | popularity: {item.popularity:.4f}")
        for reason in item.reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()

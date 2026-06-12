"""
ChromaDB 电影向量存储: 替代 FAISS + JSON + numpy 三件套。

架构:
    movies_plot (Collection)  ← 剧情语义向量 (384d)
    movies_attr (Collection)  ← 属性标签向量 (384d)
    movies_full (Collection)  ← 综合向量 (384d)

三个 collection 保持与 FAISS 多粒度索引相同的语义粒度,
通过加权融合检索实现向后兼容。

特性:
    - 原生 CRUD: add / update / delete 即时生效, 持久化到磁盘
    - 元数据过滤: ChromaDB where 子句替代手写 Python 循环
    - HNSW 近似搜索: 数据量增大后性能优于 IndexFlatIP 暴力搜索
    - 零运维: PersistentClient 嵌入式模式, 无独立服务进程
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ChromaDB collection 元数据: 用 cosine 距离 (与 FAISS IndexFlatIP 归一化内积等价)
COLLECTION_METADATA = {"hnsw:space": "cosine"}


@dataclass
class MovieRecord:
    """电影记录 (独立于 BasicRecommender 的 MovieCandidate)"""
    movie_id: int
    title: str
    genres: List[str]
    release_year: Optional[float]
    overview_en: str = ""
    avg_rating: float = 0.0
    rating_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "release_year": self.release_year,
            "overview_en": self.overview_en,
            "avg_rating": self.avg_rating,
            "rating_count": self.rating_count,
        }

    @classmethod
    def from_metadata(cls, movie_id: str, metadata: Dict[str, Any]) -> "MovieRecord":
        genres_str = metadata.get("genres", "")
        genres = [g.strip() for g in genres_str.split("|") if g.strip()] if genres_str else []
        release_year = metadata.get("release_year")
        if release_year is not None:
            try:
                release_year = float(release_year)
            except (TypeError, ValueError):
                release_year = None
        return cls(
            movie_id=int(movie_id),
            title=metadata.get("title", ""),
            genres=genres,
            release_year=release_year,
            overview_en=metadata.get("overview_en", ""),
            avg_rating=float(metadata.get("avg_rating", 0)),
            rating_count=int(metadata.get("rating_count", 0)),
        )


class ChromaMovieStore:
    """基于 ChromaDB 的电影向量存储, 原生支持 CRUD + 语义搜索 + 元数据过滤。

    用法:
        store = ChromaMovieStore(persist_dir="data/chroma", model_name="...")
        store.add_movie(movie)        # 新增 (即时持久化)
        store.update_movie(id, ...)   # 更新
        store.delete_movie(id)        # 删除
        results = store.search("科幻 烧脑")  # 多粒度融合检索
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 延迟加载 SentenceTransformer (只在需要编码时加载)
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name

        # 初始化 ChromaDB 客户端 (嵌入式持久化模式)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # 三个 collection 对应三种语义粒度
        self._plot_col = self._client.get_or_create_collection(
            name="movies_plot",
            metadata=COLLECTION_METADATA,
        )
        self._attr_col = self._client.get_or_create_collection(
            name="movies_attr",
            metadata=COLLECTION_METADATA,
        )
        self._full_col = self._client.get_or_create_collection(
            name="movies_full",
            metadata=COLLECTION_METADATA,
        )

        logger.info(
            "ChromaMovieStore initialized: persist_dir=%s, model=%s, "
            "movies_plot=%d, movies_attr=%d, movies_full=%d",
            self.persist_dir, model_name,
            self._plot_col.count(), self._attr_col.count(), self._full_col.count(),
        )

    # ── 模型懒加载 ──

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading SentenceTransformer: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ── 文本 → 向量编码 ──

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """批量文本编码, 返回 float32 numpy 数组。"""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """单文本编码。"""
        return self.encode([text])[0]

    # ── 多粒度文本切分 (与 FAISS 版本一致的语义粒度) ──

    @staticmethod
    def _build_chunks(movie: Dict[str, Any]) -> Dict[str, str]:
        """将电影元数据拆分为三种语义粒度的文本。

        (逻辑复制自 rag_build_index.py:build_movie_chunks, 保持一致)
        """
        title = str(movie.get("title", "")).strip()
        release_year = movie.get("release_year")
        genres = movie.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split("|") if g.strip()]
        overview = str(movie.get("overview_en", "") or movie.get("overview", "")).strip()

        year_text = "unknown"
        if release_year is not None and str(release_year) != "nan":
            try:
                year_text = str(int(float(release_year)))
            except (TypeError, ValueError):
                year_text = str(release_year)

        genre_text = ", ".join(genres) if genres else "unknown"

        # 粒度 1: 剧情语义
        plot_chunk = f"Title: {title}."
        if overview:
            plot_chunk += f" Overview: {overview}"

        # 粒度 2: 属性标签
        attr_chunk = f"Genres: {genre_text}. Release year: {year_text}."

        # 粒度 3: 综合全文
        full_chunk = f"{plot_chunk} {attr_chunk}"

        return {"plot": plot_chunk, "attr": attr_chunk, "full": full_chunk}

    # ── 元数据序列化 ──

    @staticmethod
    def _movie_to_metadata(movie: Dict[str, Any]) -> Dict[str, Any]:
        """将 movie dict 转换为 ChromaDB metadata 格式 (全字符串值)。"""
        genres = movie.get("genres", [])
        if isinstance(genres, list):
            genres_str = "|".join(str(g) for g in genres)
        else:
            genres_str = str(genres)
        return {
            "title": str(movie.get("title", "")),
            "genres": genres_str,
            "release_year": movie.get("release_year"),
            "overview_en": str(movie.get("overview_en", "")),
            "avg_rating": float(movie.get("avg_rating", 0)),
            "rating_count": int(movie.get("rating_count", 0)),
        }

    # ═══════════════════════════════════════════════════════════
    # CRUD 操作
    # ═══════════════════════════════════════════════════════════

    def add_movie(self, movie: Dict[str, Any]) -> int:
        """新增电影: 编码三个语义粒度 → 写入三个 collection → 即时持久化。

        Args:
            movie: dict with title, genres, release_year, overview_en, etc.

        Returns:
            movie_id (int): 新电影 ID (如果未指定则自动生成)
        """
        # 自动分配 ID
        movie_id = movie.get("movie_id")
        if movie_id is None:
            movie_id = self._next_id()
        movie_id = int(movie_id)
        movie["movie_id"] = movie_id

        mid_str = str(movie_id)

        # 多粒度文本切分 + 编码
        chunks = self._build_chunks(movie)
        plot_emb = self.encode_single(chunks["plot"]).tolist()
        attr_emb = self.encode_single(chunks["attr"]).tolist()
        full_emb = self.encode_single(chunks["full"]).tolist()

        metadata = self._movie_to_metadata(movie)

        # 原子性: 先写三个 collection (ChromaDB 不支持跨 collection 事务,
        # 但单 collection 写入是原子的)
        try:
            self._plot_col.add(ids=[mid_str], embeddings=[plot_emb], metadatas=[metadata])
            self._attr_col.add(ids=[mid_str], embeddings=[attr_emb], metadatas=[metadata])
            self._full_col.add(ids=[mid_str], embeddings=[full_emb], metadatas=[metadata])
        except Exception:
            # 补偿: 如果任何一个写入失败, 回滚已写入的
            logger.exception("add_movie(%d) 写入失败, 尝试回滚", movie_id)
            for col in [self._plot_col, self._attr_col, self._full_col]:
                try:
                    col.delete(ids=[mid_str])
                except Exception:
                    pass
            raise

        logger.info("add_movie: id=%d, title=%s, genres=%s", movie_id, movie["title"], metadata["genres"])
        return movie_id

    def update_movie(self, movie_id: int, **fields) -> bool:
        """更新电影元数据和/或向量。

        Args:
            movie_id: 要更新的电影 ID
            **fields: title, genres, release_year, overview_en, avg_rating, rating_count

        Returns:
            是否成功更新 (movie_id 不存在时返回 False)
        """
        mid_str = str(movie_id)

        # 先检查存在性
        existing = self._full_col.get(ids=[mid_str])
        if not existing["ids"]:
            logger.warning("update_movie(%d): movie_id 不存在", movie_id)
            return False

        old_metadata = (existing["metadatas"] or [{}])[0]
        new_metadata = dict(old_metadata)

        # 合并新字段
        field_mapping = {
            "title": "title",
            "overview_en": "overview_en",
            "avg_rating": "avg_rating",
            "rating_count": "rating_count",
        }
        for key, meta_key in field_mapping.items():
            if key in fields:
                new_metadata[meta_key] = fields[key]

        # genres 特殊处理
        if "genres" in fields:
            genres = fields["genres"]
            if isinstance(genres, list):
                new_metadata["genres"] = "|".join(str(g) for g in genres)
            else:
                new_metadata["genres"] = str(genres)

        if "release_year" in fields:
            new_metadata["release_year"] = fields["release_year"]

        # 判断是否需要重新编码 (改了影响语义的字段)
        semantic_fields = {"title", "genres", "overview_en"}
        need_reencode = bool(semantic_fields & set(fields.keys()))

        if need_reencode:
            # 用合并后的元数据重建 movie dict, 重新编码
            genres_str = new_metadata.get("genres", "")
            genres_list = [g.strip() for g in genres_str.split("|") if g.strip()]
            mock_movie = {
                "title": new_metadata.get("title", ""),
                "genres": genres_list,
                "release_year": new_metadata.get("release_year"),
                "overview_en": new_metadata.get("overview_en", ""),
            }
            chunks = self._build_chunks(mock_movie)
            plot_emb = self.encode_single(chunks["plot"]).tolist()
            attr_emb = self.encode_single(chunks["attr"]).tolist()
            full_emb = self.encode_single(chunks["full"]).tolist()

            # 更新 embeddings + metadata
            self._plot_col.update(ids=[mid_str], embeddings=[plot_emb], metadatas=[new_metadata])
            self._attr_col.update(ids=[mid_str], embeddings=[attr_emb], metadatas=[new_metadata])
            self._full_col.update(ids=[mid_str], embeddings=[full_emb], metadatas=[new_metadata])
        else:
            # 只更新 metadata (评分等业务字段)
            self._plot_col.update(ids=[mid_str], metadatas=[new_metadata])
            self._attr_col.update(ids=[mid_str], metadatas=[new_metadata])
            self._full_col.update(ids=[mid_str], metadatas=[new_metadata])

        logger.info("update_movie(%d): fields=%s, reencoded=%s", movie_id, list(fields.keys()), need_reencode)
        return True

    def delete_movie(self, movie_id: int) -> bool:
        """删除电影: 从三个 collection 中移除。

        Args:
            movie_id: 要删除的电影 ID

        Returns:
            是否成功删除
        """
        mid_str = str(movie_id)

        # 检查存在性
        existing = self._full_col.get(ids=[mid_str])
        if not existing["ids"]:
            logger.warning("delete_movie(%d): movie_id 不存在", movie_id)
            return False

        self._plot_col.delete(ids=[mid_str])
        self._attr_col.delete(ids=[mid_str])
        self._full_col.delete(ids=[mid_str])

        logger.info("delete_movie(%d): 已从三个 collection 中删除", movie_id)
        return True

    def get_movie(self, movie_id: int) -> Optional[MovieRecord]:
        """获取单部电影元数据。"""
        mid_str = str(movie_id)
        result = self._full_col.get(ids=[mid_str], include=["metadatas"])
        if not result["ids"]:
            return None
        return MovieRecord.from_metadata(mid_str, result["metadatas"][0])

    def list_movies(
        self,
        offset: int = 0,
        limit: int = 50,
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
    ) -> List[MovieRecord]:
        """分页列出电影, 支持简单过滤。

        注意: ChromaDB 的 get() 不支持 offset, 所以用 Python 层分页。
        对于 < 10 万条的全量列表, 性能可接受。
        """
        where = None
        if genre:
            where = {"genres": {"$contains": genre}}

        # ChromaDB 不支持 offset, 取全部后用 Python 分页
        result = self._full_col.get(
            where=where,
            include=["metadatas"],
            limit=limit + offset if limit > 0 else None,
        )

        records = []
        for mid, meta in zip(result["ids"], result["metadatas"]):
            if min_year is not None and meta.get("release_year"):
                try:
                    if float(meta["release_year"]) < min_year:
                        continue
                except (TypeError, ValueError):
                    pass
            records.append(MovieRecord.from_metadata(mid, meta))

        return records[offset:offset + limit]

    def count(self) -> int:
        """返回电影总数。"""
        return self._full_col.count()

    # ═══════════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        query_embedding: Optional[np.ndarray] = None,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        genre_filter: Optional[str] = None,
        min_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """多粒度融合检索: 三个 collection 独立查询 → 加权融合 → 去重排序。

        Args:
            query_text: 查询文本 (用于日志和 fallback 编码)
            top_k: 返回数量
            query_embedding: 预计算的查询向量 (None 则自动编码)
            weights: (full_weight, plot_weight, attr_weight) 三者之和应为 1.0
            genre_filter: 可选, 按类型过滤 (如 "Action", "Comedy")
            min_year: 可选, 最低上映年份

        Returns:
            [{"movie_id": int, "title": str, "genres": list, ..., "fusion_score": float}, ...]
        """
        if query_embedding is None:
            query_embedding = self.encode_single(query_text)

        q_emb = query_embedding.astype(np.float32).tolist()

        # 构建 ChromaDB where 子句
        where = self._build_where(genre=genre_filter, min_year=min_year)

        # 三个 collection 查询 + 加权融合
        candidates: Dict[int, float] = {}
        candidate_meta: Dict[int, Dict] = {}

        queries = [
            (self._full_col, weights[0]),
            (self._plot_col, weights[1]),
            (self._attr_col, weights[2]),
        ]

        for col, weight in queries:
            if weight <= 0:
                continue
            try:
                result = col.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where=where,
                    include=["metadatas", "distances"],
                )
            except Exception as e:
                logger.warning("ChromaDB query failed for %s: %s", col.name, e)
                continue

            if not result["ids"] or not result["ids"][0]:
                continue

            for mid, distance, meta in zip(
                result["ids"][0], result["distances"][0], result["metadatas"][0]
            ):
                # ChromaDB cosine distance → similarity (0-1)
                similarity = 1.0 - float(distance)
                mid_int = int(mid)
                candidates[mid_int] = candidates.get(mid_int, 0.0) + similarity * weight
                if mid_int not in candidate_meta:
                    candidate_meta[mid_int] = meta

        # 排序 + 去重
        merged = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        results = []
        for mid, score in merged[:top_k]:
            meta = candidate_meta.get(mid, {})
            rec = MovieRecord.from_metadata(str(mid), meta)
            results.append({
                "movie_id": rec.movie_id,
                "title": rec.title,
                "genres": rec.genres,
                "release_year": rec.release_year,
                "overview_en": rec.overview_en,
                "avg_rating": rec.avg_rating,
                "rating_count": rec.rating_count,
                "fusion_score": round(score, 6),
            })

        return results

    @staticmethod
    def _build_where(
        genre: Optional[str] = None,
        min_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """构建 ChromaDB where 过滤条件。

        注意: ChromaDB where 的数值比较要求值是 int/float 类型,
        但我们的 metadata 中 release_year 是 Optional[float]。
        对于 min_year 过滤, 在 Python 层做后过滤更可靠。
        """
        conditions = []

        if genre:
            # ChromaDB $contains 对分号分隔的字符串进行子串匹配
            conditions.append({"genres": {"$contains": genre}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ═══════════════════════════════════════════════════════════
    # 索引统计与维护
    # ═══════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """返回存储统计信息。"""
        return {
            "persist_dir": str(self.persist_dir),
            "model": self._model_name,
            "movie_count": self._full_col.count(),
            "collections": {
                "movies_plot": self._plot_col.count(),
                "movies_attr": self._attr_col.count(),
                "movies_full": self._full_col.count(),
            },
        }

    # ── 内部辅助 ──

    def _next_id(self) -> int:
        """生成下一个自增 movie_id。"""
        all_ids = self._full_col.get(include=[])
        if not all_ids["ids"]:
            return 1
        max_id = max(int(i) for i in all_ids["ids"])
        return max_id + 1

"""
构建 ChromaDB 向量索引: 从 movies.json 重新编码 → 写入 ChromaDB。

与 rag_build_index.py (FAISS 版) 对应, 这个是 ChromaDB 版。
不依赖 .faiss / .npy 文件, 直接用 SentenceTransformer 重新编码原始电影数据。

用法:
    # 全量构建
    python src/build_chroma_index.py

    # 指定模型和数据路径
    python src/build_chroma_index.py \
        --movies data/processed/movies.json \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --chroma-dir data/chroma

    # 采样验证 (只构建前 20 部)
    python src/build_chroma_index.py --sample 20

构建策略:
    - 读取 movies.json 获取元数据 (title, genres, release_year, overview_en)
    - 用 SentenceTransformer 重新编码 (不依赖 .npy / .faiss, 确保语义空间一致)
    - 批量写入 ChromaDB 的三个 collection
    - 构建后输出统计信息 + 检索验证
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chroma_store import ChromaMovieStore

logger = logging.getLogger(__name__)


def load_records(path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 或 JSON array 格式的电影数据。"""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records.")
        return data

    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_movie_stats(full_data_path: Path) -> Dict[int, Dict[str, float]]:
    """从 full_data.json 加载评分统计 (avg_rating, rating_count)。"""
    if not full_data_path.exists():
        logger.warning("full_data.json not found, using default stats")
        return {}

    records = load_records(full_data_path)
    if not records:
        return {}

    rating_sum: Dict[int, float] = {}
    rating_count: Dict[int, int] = {}
    for record in records:
        movie_id = int(record["movie_id"])
        rating = float(record["rating"])
        rating_sum[movie_id] = rating_sum.get(movie_id, 0.0) + rating
        rating_count[movie_id] = rating_count.get(movie_id, 0) + 1

    stats: Dict[int, Dict[str, float]] = {}
    for movie_id, total in rating_sum.items():
        count = rating_count[movie_id]
        stats[movie_id] = {
            "avg_rating": total / max(count, 1),
            "rating_count": float(count),
        }
    return stats


def build_index(
    movies_path: Path,
    full_data_path: Path,
    chroma_dir: Path,
    model_name: str,
    sample: int = 0,
    batch_size: int = 50,
) -> ChromaMovieStore:
    """构建 ChromaDB 索引: movies.json → ChromaDB (三 Collection 多粒度架构)。

    Args:
        movies_path: movies.json 路径
        full_data_path: full_data.json 路径 (评分统计)
        chroma_dir: ChromaDB 持久化目录
        model_name: SentenceTransformer 模型名
        sample: 仅构建前 N 部 (用于测试)
        batch_size: 每批构建数量

    Returns:
        ChromaMovieStore 实例 (已填充数据)
    """
    # 加载数据
    movies = load_records(movies_path)
    stats = load_movie_stats(full_data_path)

    if sample > 0:
        movies = movies[:sample]

    total = len(movies)
    print(f"加载了 {total} 部电影 (movies.json)")
    print(f"评分统计覆盖 {len(stats)} 部电影 (full_data.json)")
    print(f"使用模型: {model_name}")
    print(f"目标目录: {chroma_dir}")
    print()

    # 创建 ChromaMovieStore (延迟加载模型)
    store = ChromaMovieStore(
        persist_dir=str(chroma_dir),
        model_name=model_name,
    )

    # 评分统计合并到 movie dict
    for movie in movies:
        mid = int(movie["movie_id"])
        if mid in stats:
            movie["avg_rating"] = stats[mid]["avg_rating"]
            movie["rating_count"] = int(stats[mid]["rating_count"])
        else:
            movie["avg_rating"] = 0.0
            movie["rating_count"] = 0

    # 批量构建
    imported = 0
    errors = 0

    for i in range(0, total, batch_size):
        batch = movies[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        for movie in batch:
            try:
                store.add_movie(movie)
                imported += 1
            except Exception as e:
                errors += 1
                logger.error("构建失败 movie_id=%s: %s", movie.get("movie_id"), e)
                if errors <= 5:
                    import traceback
                    traceback.print_exc()

        # 进度输出
        progress = min(i + batch_size, total)
        print(f"\r  进度: {progress}/{total} ({progress*100//total}%)  "
              f"成功={imported} 失败={errors}", end="", flush=True)

    print()
    print()

    # 验证
    print("=" * 60)
    print("  构建完成 — 验证")
    print("=" * 60)
    store_stats = store.stats()
    print(f"  ChromaDB 电影总数: {store_stats['movie_count']}")
    print(f"  期望总数:          {total}")
    print(f"  构建成功:          {imported}")
    print(f"  构建失败:          {errors}")

    if store_stats['movie_count'] != total:
        print(f"  ⚠ 数量不一致! 差值={total - store_stats['movie_count']}")
    else:
        print(f"  ✓ 数量一致")

    # 检索一致性抽样验证
    test_queries = ["science fiction action", "romantic comedy", "horror thriller"]
    print(f"\n检索验证 (与期望的语义一致性对比):")
    for q in test_queries:
        results = store.search(q, top_k=3)
        titles = [r["title"] for r in results]
        print(f"  query='{q}' → top3: {titles}")

    print(f"\nChromaDB 目录: {chroma_dir.absolute()}")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index from movies.json"
    )
    parser.add_argument(
        "--movies",
        default="data/processed/movies.json",
        help="Path to movies.json",
    )
    parser.add_argument(
        "--full-data",
        default="data/processed/full_data.json",
        help="Path to full_data.json (ratings)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Only build first N movies (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Movies per batch",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    build_index(
        movies_path=Path(args.movies),
        full_data_path=Path(args.full_data),
        chroma_dir=Path(args.chroma_dir),
        model_name=args.model,
        sample=args.sample,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

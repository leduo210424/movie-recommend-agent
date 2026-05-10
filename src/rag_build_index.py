import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-small",
]


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL or JSON array records from disk."""
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


def normalize_genres(genres: Any) -> List[str]:
    if genres is None:
        return []
    if isinstance(genres, (list, tuple)):
        return [str(g) for g in genres if str(g).strip()]
    return [str(genres)]


def build_movie_chunks(movie: Dict[str, Any]) -> Dict[str, str]:
    """多粒度文档切分策略：将电影元数据切分为'剧情语义'和'属性标签'等不同粒度的文本片段。"""
    title = str(movie.get("title", "")).strip()
    release_year = movie.get("release_year")
    genres = normalize_genres(movie.get("genres"))
    overview = str(movie.get("overview_en", "") or movie.get("overview", "")).strip()

    year_text = "unknown"
    if release_year is not None and str(release_year) != "nan":
        try:
            year_text = str(int(float(release_year)))
        except (TypeError, ValueError):
            year_text = str(release_year)

    genre_text = ", ".join(genres) if genres else "unknown"
    
    # 粒度1: 剧情与主题语义 (Plot Semantic)
    plot_chunk = f"Title: {title}."
    if overview:
        plot_chunk += f" Overview: {overview}"
        
    # 粒度2: 属性层面的强规范标签 (Attribute Labels)
    attr_chunk = f"Genres: {genre_text}. Release year: {year_text}."
    
    # 综合全文 (保留原有作为兜底或Baseline)
    full_chunk = f"{plot_chunk} {attr_chunk}"

    return {
        "plot": plot_chunk,
        "attr": attr_chunk,
        "full": full_chunk
    }


def load_embedding_model(model_name: str, local_files_only: bool) -> tuple[SentenceTransformer, str]:
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    errors: Dict[str, str] = {}

    for candidate in candidates:
        try:
            print(f"Trying model: {candidate}")
            model = SentenceTransformer(candidate, local_files_only=local_files_only)
            print(f"Using model: {candidate}")
            return model, candidate
        except Exception as exc:  # pragma: no cover - runtime dependency/network issues
            errors[candidate] = str(exc)

    err_lines = [f"- {name}: {msg}" for name, msg in errors.items()]
    joined = "\n".join(err_lines)
    raise RuntimeError(
        "Failed to load any embedding model.\n"
        "You can try one of the following:\n"
        "1) pass --model with a valid local path or model id\n"
        "2) run hf auth login if your model source needs authentication\n"
        "3) pre-download model files and run with --local-files-only\n"
        f"Details:\n{joined}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build movie embeddings and FAISS index.")
    parser.add_argument(
        "--input",
        default="data/processed/movies.json",
        help="Input movie file (JSONL or JSON array).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to store embeddings/index outputs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model only from local cache/files (no network download).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    movies = load_records(input_path)
    if not movies:
        raise ValueError(f"No movie records found in {input_path}")

    # 多粒度切分文本
    chunks_list = [build_movie_chunks(movie) for movie in movies]
    plot_texts = [chunks["plot"] for chunks in chunks_list]
    attr_texts = [chunks["attr"] for chunks in chunks_list]
    full_texts = [chunks["full"] for chunks in chunks_list]

    model, used_model_name = load_embedding_model(args.model, args.local_files_only)
    
    print("Encoding plot semantic embeddings...")
    plot_embeddings = model.encode(
        plot_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    print("Encoding attribute label embeddings...")
    attr_embeddings = model.encode(
        attr_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    print("Encoding full baseline embeddings...")
    full_embeddings = model.encode(
        full_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    dim = full_embeddings.shape[1]
    
    # 建立多个维度的独立FAISS索引
    plot_index = faiss.IndexFlatIP(dim)
    plot_index.add(plot_embeddings)
    
    attr_index = faiss.IndexFlatIP(dim)
    attr_index.add(attr_embeddings)
    
    full_index = faiss.IndexFlatIP(dim)
    full_index.add(full_embeddings)

    movie_ids = np.array([int(movie["movie_id"]) for movie in movies], dtype=np.int32)

    # 存储路径
    embeddings_path = output_dir / "movie_embeddings.npy"  # 保留用于向后兼容
    ids_path = output_dir / "movie_ids.npy"
    full_index_path = output_dir / "movie_index.faiss"     # 原综合索引
    plot_index_path = output_dir / "movie_plot_index.faiss" # 新增剧情语义索引
    attr_index_path = output_dir / "movie_attr_index.faiss" # 新增属性标签索引
    meta_path = output_dir / "movie_index_meta.json"

    np.save(embeddings_path, full_embeddings)
    np.save(ids_path, movie_ids)
    faiss.write_index(full_index, str(full_index_path))
    faiss.write_index(plot_index, str(plot_index_path))
    faiss.write_index(attr_index, str(attr_index_path))

    meta = {
        "model": used_model_name,
        "count": int(len(movies)),
        "dim": int(dim),
        "index_type": "IndexFlatIP",
        "source_file": str(input_path).replace("\\", "/"),
        "multi_granularity_indices": {
            "plot_index": "movie_plot_index.faiss",
            "attr_index": "movie_attr_index.faiss",
            "full_index": "movie_index.faiss"
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Build completed")
    print(f"movies: {len(movies)}")
    print(f"embedding_dim: {dim}")
    print(f"saved full embeddings: {embeddings_path}")
    print(f"saved: {ids_path}")
    print(f"saved indices: {full_index_path}, {plot_index_path}, {attr_index_path}")
    print(f"saved: {meta_path}")


if __name__ == "__main__":
    main()

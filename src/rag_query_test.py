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
        raise FileNotFoundError(f"Movie file not found: {path}")

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


def get_model_name(meta_path: Path, override_model: str) -> str:
    if override_model:
        return override_model
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and "model" in meta:
                return str(meta["model"])
        except json.JSONDecodeError:
            pass
    return DEFAULT_MODEL


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


def run_queries(
    queries: List[str],
    movies: List[Dict[str, Any]],
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int,
) -> None:
    for query in queries:
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = index.search(query_embedding, top_k)

        print("=" * 72)
        print(f"Query: {query}")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(movies):
                continue
            movie = movies[idx]
            title = movie.get("title", "")
            genres = movie.get("genres", [])
            genres_text = ", ".join(genres) if isinstance(genres, list) else str(genres)
            print(f"{rank:>2}. {title} | genres: {genres_text} | score: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval tests against FAISS movie index.")
    parser.add_argument(
        "--movie-file",
        default="data/processed/movies.json",
        help="Movie metadata file (JSONL or JSON array).",
    )
    parser.add_argument(
        "--index-file",
        default="data/processed/movie_index.faiss",
        help="FAISS index file.",
    )
    parser.add_argument(
        "--meta-file",
        default="data/processed/movie_index_meta.json",
        help="Metadata JSON generated at build time.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override embedding model name (optional).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbors to return per query.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["science fiction movie", "romantic comedy", "horror thriller"],
        help="One or more free-text queries.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model only from local cache/files (no network download).",
    )
    args = parser.parse_args()

    movie_path = Path(args.movie_file)
    index_path = Path(args.index_file)
    meta_path = Path(args.meta_file)

    movies = load_records(movie_path)
    if not movies:
        raise ValueError(f"No movies loaded from {movie_path}")

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    model_name = get_model_name(meta_path, args.model)
    model, _ = load_embedding_model(model_name, args.local_files_only)

    index = faiss.read_index(str(index_path))

    top_k = min(args.top_k, len(movies))
    run_queries(args.queries, movies, index, model, top_k)


if __name__ == "__main__":
    main()

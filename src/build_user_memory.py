import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from user_memory import UserMemoryStore


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a list in JSON file.")
        return data

    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long-term user memory profiles.")
    parser.add_argument("--movie-file", default="data/processed/movies.json")
    parser.add_argument("--movie-embeddings", default="data/processed/movie_embeddings.npy")
    parser.add_argument("--movie-ids", default="data/processed/movie_ids.npy")
    parser.add_argument("--user-history-file", default="data/processed/user_history.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--half-life-days", type=float, default=30.0)
    args = parser.parse_args()

    movie_store = UserMemoryStore.from_files(
        movie_file=args.movie_file,
        embedding_file=args.movie_embeddings,
        movie_ids_file=args.movie_ids,
    )
    user_histories = load_jsonl_records(Path(args.user_history_file))
    if not user_histories:
        raise ValueError(f"No user histories found in {args.user_history_file}")

    profiles = []
    skipped_users = []
    for history in user_histories:
        try:
            profiles.append(movie_store.build_profile(history, half_life_days=args.half_life_days))
        except ValueError as exc:
            skipped_users.append({"user_id": history.get("user_id"), "reason": str(exc)})

    if not profiles:
        raise ValueError("No user profiles were built.")

    UserMemoryStore.save_profiles(
        profiles=profiles,
        output_dir=args.output_dir,
        source_file=str(Path(args.user_history_file)).replace("\\", "/"),
        half_life_days=args.half_life_days,
    )

    print("User memory build completed")
    print(f"profiles: {len(profiles)}")
    print(f"skipped: {len(skipped_users)}")
    if skipped_users:
        print("Skipped users:")
        for item in skipped_users[:10]:
            print(f"- user {item['user_id']}: {item['reason']}")


if __name__ == "__main__":
    main()

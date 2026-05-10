from __future__ import annotations

import json
import math
import faiss
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_HALF_LIFE_DAYS = 30.0


@dataclass
class MovieMemory:
    movie_id: int
    title: str
    genres: List[str]
    release_year: Optional[float]
    embedding: np.ndarray


@dataclass
class UserProfile:
    user_id: int
    num_ratings: int
    mean_rating: float
    weighted_rating: float
    last_timestamp: int
    favorite_genres: List[Tuple[str, float]]
    recent_movies: List[Dict[str, Any]]
    positive_movies: List[Dict[str, Any]]
    disliked_movies: List[Dict[str, Any]]
    profile_text: str
    embedding: np.ndarray


class UserMemoryStore:
    def __init__(
        self,
        movie_records: List[Dict[str, Any]],
        movie_embeddings: np.ndarray,
        movie_ids: np.ndarray,
    ) -> None:
        if len(movie_records) != len(movie_embeddings) or len(movie_ids) != len(movie_embeddings):
            raise ValueError("Movie metadata, embeddings, and ids must have the same length.")

        self.movie_lookup: Dict[int, MovieMemory] = {}
        for record, embedding, movie_id in zip(movie_records, movie_embeddings, movie_ids):
            self.movie_lookup[int(movie_id)] = MovieMemory(
                movie_id=int(movie_id),
                title=str(record.get("title", "")),
                genres=self._normalize_genres(record.get("genres")),
                release_year=self._to_float_or_none(record.get("release_year")),
                embedding=np.asarray(embedding, dtype=np.float32),
            )

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
        """Map MovieLens rating scale 1-5 into 0.0-1.0."""
        return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))

    @staticmethod
    def _time_decay(delta_days: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
        if delta_days <= 0:
            return 1.0
        decay_rate = math.log(2.0) / half_life_days
        return math.exp(-decay_rate * delta_days)

    @classmethod
    def from_files(
        cls,
        movie_file: str | Path,
        embedding_file: str | Path,
        movie_ids_file: str | Path,
    ) -> "UserMemoryStore":
        movie_path = Path(movie_file)
        embedding_path = Path(embedding_file)
        ids_path = Path(movie_ids_file)

        movie_records = cls._load_records(movie_path)
        movie_embeddings = np.load(embedding_path)
        movie_ids = np.load(ids_path)
        return cls(movie_records, movie_embeddings, movie_ids)

    @staticmethod
    def _load_records(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text.startswith("["):
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Expected list of records in JSON file.")
            return data
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _movie_features(self, movie_id: int) -> Optional[MovieMemory]:
        return self.movie_lookup.get(int(movie_id))

    def build_profile(
        self,
        user_history: Dict[str, Any],
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        top_k_genres: int = 5,
        top_k_movies: int = 5,
    ) -> UserProfile:
        user_id = int(user_history["user_id"])
        raw_history = list(user_history.get("history", []))
        if not raw_history:
            raise ValueError(f"User {user_id} has empty history.")

        sorted_history = sorted(raw_history, key=lambda item: int(item.get("timestamp", 0)))
        last_timestamp = int(sorted_history[-1].get("timestamp", 0))

        weighted_vectors: List[np.ndarray] = []
        vector_weights: List[float] = []
        genre_scores: Dict[str, float] = {}
        rated_movies: List[Dict[str, Any]] = []

        total_rating = 0.0
        rating_count = 0

        for entry in sorted_history:
            movie_id = int(entry["movie_id"])
            rating = float(entry["rating"])
            timestamp = int(entry["timestamp"])
            movie = self._movie_features(movie_id)
            if movie is None:
                continue

            rating_weight = self._normalize_rating(rating)
            delta_days = max(0.0, (last_timestamp - timestamp) / SECONDS_PER_DAY)
            time_weight = self._time_decay(delta_days, half_life_days=half_life_days)
            combined_weight = max(0.0, rating_weight) * time_weight
            combined_weight = max(combined_weight, 1e-6)

            weighted_vectors.append(movie.embedding.astype(np.float32) * combined_weight)
            vector_weights.append(combined_weight)

            for genre in movie.genres:
                genre_scores[genre] = genre_scores.get(genre, 0.0) + combined_weight

            movie_payload = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres,
                "release_year": movie.release_year,
                "rating": rating,
                "timestamp": timestamp,
                "weight": round(combined_weight, 6),
            }
            rated_movies.append(movie_payload)
            total_rating += rating
            rating_count += 1

        if not weighted_vectors:
            raise ValueError(f"User {user_id} has no mappable movie history.")

        embedding = np.sum(weighted_vectors, axis=0) / max(sum(vector_weights), 1e-6)
        embedding = embedding.astype(np.float32)

        favorite_genres = sorted(genre_scores.items(), key=lambda item: item[1], reverse=True)[:top_k_genres]
        positive_movies = [item for item in rated_movies if float(item["rating"]) >= 4.0]
        disliked_movies = [item for item in rated_movies if float(item["rating"]) <= 2.0]
        recent_movies = list(reversed(rated_movies[-top_k_movies:]))

        summary_parts = [
            f"user {user_id}",
            f"ratings={rating_count}",
            f"mean_rating={total_rating / max(rating_count, 1):.2f}",
        ]
        if favorite_genres:
            summary_parts.append(
                "favorite_genres=" + ", ".join(f"{genre}:{score:.2f}" for genre, score in favorite_genres)
            )
        if positive_movies:
            summary_parts.append(
                "liked_movies=" + ", ".join(movie["title"] for movie in positive_movies[:3])
            )
        profile_text = " | ".join(summary_parts)

        return UserProfile(
            user_id=user_id,
            num_ratings=rating_count,
            mean_rating=total_rating / max(rating_count, 1),
            weighted_rating=float(np.mean([item["rating"] for item in rated_movies])),
            last_timestamp=last_timestamp,
            favorite_genres=favorite_genres,
            recent_movies=recent_movies,
            positive_movies=positive_movies[:top_k_movies],
            disliked_movies=disliked_movies[:top_k_movies],
            profile_text=profile_text,
            embedding=embedding,
        )

    @staticmethod
    def load_user_histories(path: str | Path) -> List[Dict[str, Any]]:
        return UserMemoryStore._load_records(Path(path))

    @staticmethod
    def profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
        return {
            "user_id": profile.user_id,
            "num_ratings": profile.num_ratings,
            "mean_rating": round(profile.mean_rating, 4),
            "weighted_rating": round(profile.weighted_rating, 4),
            "last_timestamp": profile.last_timestamp,
            "favorite_genres": [
                {"genre": genre, "score": round(score, 6)} for genre, score in profile.favorite_genres
            ],
            "recent_movies": profile.recent_movies,
            "positive_movies": profile.positive_movies,
            "disliked_movies": profile.disliked_movies,
            "profile_text": profile.profile_text,
        }

    @staticmethod
    def save_profiles(
        profiles: List[UserProfile],
        output_dir: str | Path,
        source_file: str,
        half_life_days: float,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        profile_dicts = [UserMemoryStore.profile_to_dict(profile) for profile in profiles]
        profile_jsonl = output_path / "user_profiles.jsonl"
        profile_jsonl.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in profile_dicts) + "\n",
            encoding="utf-8",
        )

        embeddings = np.stack([profile.embedding for profile in profiles], axis=0).astype(np.float32)
        user_ids = np.array([profile.user_id for profile in profiles], dtype=np.int32)
        np.save(output_path / "user_embeddings.npy", embeddings)
        np.save(output_path / "user_ids.npy", user_ids)
        
        # 增加构建用户画像索引 (User Profile Index)
        dim = embeddings.shape[1]
        user_index = faiss.IndexFlatIP(dim)
        user_index.add(embeddings)
        faiss.write_index(user_index, str(output_path / "user_index.faiss"))

        meta = {
            "source_file": source_file,
            "count": len(profiles),
            "embedding_dim": int(dim) if len(profiles) else 0,
            "half_life_days": half_life_days,
            "index_type": "IndexFlatIP",
            "index_file": "user_index.faiss"
        }
        (output_path / "user_memory_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load_profile_index(output_dir: str | Path) -> Dict[int, Dict[str, Any]]:
        profile_path = Path(output_dir) / "user_profiles.jsonl"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_path}")

        profile_index: Dict[int, Dict[str, Any]] = {}
        for line in profile_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            profile = json.loads(line)
            profile_index[int(profile["user_id"])] = profile
        return profile_index

    @staticmethod
    def get_user_profile(output_dir: str | Path, user_id: int) -> Dict[str, Any]:
        profiles = UserMemoryStore.load_profile_index(output_dir)
        if user_id not in profiles:
            raise KeyError(f"User {user_id} not found in profile store.")
        return profiles[user_id]

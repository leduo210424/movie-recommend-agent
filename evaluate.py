"""
离线评估脚本：对比 ReAct Agent 与 Popular Baseline 的推荐质量。

指标：Recall@K, NDCG@K
切分策略：按时间戳 leave-last-out（每个用户最后 20% 的评分留作测试集）

用法：
    # 单一 query 评估
    python evaluate.py --query "轻松搞笑" --sample-users 50 --chroma

    # 多 query 综合评估（覆盖全部工具路径）
    python evaluate.py --queries "科幻动作" "轻松搞笑" "推荐好看的电影" "让人想旅行的电影" --sample-users 50 --chroma

    # 仅 Popular baseline
    python evaluate.py --baseline-only
"""

from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_user_holdout(
    ratings: List[Dict[str, Any]],
    test_ratio: float = 0.2,
    min_ratings: int = 10,
    positive_threshold: float = 4.0,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    按时间戳对每个用户做 leave-last-out 切分。

    Returns:
        train_ratings: {user_id: [movie_ids rated in train period]}
        test_holdout:  {user_id: [positive movie_ids held out for test]}
    """
    user_ratings: Dict[int, List[Dict]] = defaultdict(list)
    for r in ratings:
        user_ratings[int(r["user_id"])].append(r)

    train_ratings: Dict[int, List[int]] = {}
    test_holdout: Dict[int, List[int]] = {}

    for uid, user_records in user_ratings.items():
        if len(user_records) < min_ratings:
            continue
        user_records.sort(key=lambda x: int(x.get("timestamp", 0)))
        split_idx = int(len(user_records) * (1 - test_ratio))
        if split_idx < 1:
            continue

        train_part = user_records[:split_idx]
        test_part = user_records[split_idx:]

        train_ratings[uid] = [int(r["movie_id"]) for r in train_part]

        positive = [int(r["movie_id"]) for r in test_part
                     if float(r["rating"]) >= positive_threshold]
        if positive:
            test_holdout[uid] = positive

    return train_ratings, test_holdout


class PopularBaseline:
    """热门推荐基线：用 Bayesian 加权评分（兼顾评分与评分人数），并排除已看电影"""

    def __init__(self, movie_stats: Dict[int, Dict[str, float]]):
        # 全局平均分和平均评分人数，用于 Bayesian smoothing
        all_ratings = [s["avg_rating"] for s in movie_stats.values()]
        all_counts = [s["rating_count"] for s in movie_stats.values()]
        self._global_mean = np.mean(all_ratings) if all_ratings else 3.0
        self._global_count = np.mean(all_counts) if all_counts else 10.0

        # Bayesian 加权：score = (C×m + N×R) / (C+N)
        # C = 全局平均评分人数（先验强度），m = 全局平均分（先验均值）
        scored = []
        for mid, s in movie_stats.items():
            r = s.get("avg_rating", 0)
            n = s.get("rating_count", 0)
            bayesian = ((self._global_count * self._global_mean) + (n * r)) / (self._global_count + n)
            scored.append((mid, bayesian, n))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        self.popular_ids = [mid for mid, _, _ in scored]

    def recommend(self, user_id: int = None, top_k: int = 10,
                  exclude_ids: set = None) -> List[int]:
        exclude = exclude_ids or set()
        result = []
        for mid in self.popular_ids:
            if mid not in exclude:
                result.append(mid)
            if len(result) >= top_k:
                break
        return result


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for i, mid in enumerate(recommended[:k]):
        if mid in rel_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    recommender,
    test_holdout: Dict[int, List[int]],
    train_ratings: Dict[int, List[int]],
    top_k_values: List[int],
    label: str,
) -> Dict[str, float]:
    """计算 Recall@K 和 NDCG@K，对每个用户求平均"""
    del train_ratings
    metrics: Dict[str, List[float]] = {}
    for k in top_k_values:
        metrics[f"Recall@{k}"] = []
        metrics[f"NDCG@{k}"] = []

    evaluated = 0
    errors = []
    total = len(test_holdout)

    for uid, heldout in test_holdout.items():
        try:
            results = recommender.recommend(user_id=uid, top_k=max(top_k_values))
            # 空结果仍计入评估（0 hits），不跳过
            if not results:
                recommended_ids = []
            elif not isinstance(results[0], int):
                recommended_ids = [r.movie_id for r in results]
            else:
                recommended_ids = results
        except Exception as e:
            errors.append(f"User {uid}: {type(e).__name__}: {e}")
            if len(errors) <= 3:
                import traceback
                traceback.print_exc()
            continue

        evaluated += 1
        for k in top_k_values:
            metrics[f"Recall@{k}"].append(recall_at_k(recommended_ids, heldout, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(recommended_ids, heldout, k))

    summarized = {
        f"{label}_{name}": round(np.mean(values), 4) if values else 0.0
        for name, values in metrics.items()
    }
    summarized[f"{label}_users_evaluated"] = evaluated
    summarized[f"{label}_users_total"] = total
    if errors:
        summarized[f"{label}_errors"] = len(errors)
        print(f"  [{label}] {len(errors)} users failed, first error: {errors[0][:120]}")
    return summarized


def print_report(
    popular_metrics: Dict[str, float],
    agent_metrics: Dict[str, float],
    top_k_values: List[int],
):
    print("\n" + "=" * 72)
    print("  离线评估报告：Popular vs ReAct Agent")
    print("=" * 72)

    for k in top_k_values:
        r_key = f"Recall@{k}"
        n_key = f"NDCG@{k}"
        pop_r = popular_metrics.get(f"Popular_{r_key}", 0)
        agent_r = agent_metrics.get(f"Agent_{r_key}", 0)
        pop_n = popular_metrics.get(f"Popular_{n_key}", 0)
        agent_n = agent_metrics.get(f"Agent_{n_key}", 0)

        agent_vs_pop = ((agent_r - pop_r) / pop_r * 100) if pop_r > 0 else float("inf")
        agent_vs_pop_n = ((agent_n - pop_n) / pop_n * 100) if pop_n > 0 else float("inf")

        print(f"\n  Recall@{k}:")
        print(f"    Popular (Bayesian):   {pop_r:.4f}")
        print(f"    ReAct Agent:          {agent_r:.4f}  (+{agent_r - pop_r:+.4f} vs Popular, {agent_vs_pop:+.0f}%)")

        print(f"\n  NDCG@{k}:")
        print(f"    Popular (Bayesian):   {pop_n:.4f}")
        print(f"    ReAct Agent:          {agent_n:.4f}  (+{agent_n - pop_n:+.4f} vs Popular, {agent_vs_pop_n:+.0f}%)")

    print(f"\n  Evaluated users: {agent_metrics.get('Agent_users_evaluated', 0)}"
          f" / {agent_metrics.get('Agent_users_total', 0)}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Evaluate movie recommendation agents")
    parser.add_argument("--baseline-only", action="store_true", help="Run only Popular baseline")
    parser.add_argument("--query", type=str, default="",
                        help="Query text for Agent eval (empty = profile-only)")
    parser.add_argument("--queries", type=str, nargs="+", default=None,
                        help="Multiple queries for comprehensive evaluation. "
                             "覆盖全部工具路径的推荐集: "
                             "'科幻动作' '轻松搞笑' '推荐好看的电影' '让人想旅行的电影' ''")
    parser.add_argument("--sample-users", type=int, default=0,
                        help="Randomly sample N users for evaluation (0 = all users)")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data")
    parser.add_argument("--chroma", action="store_true",
                        help="Use ChromaDB backend instead of FAISS")
    args = parser.parse_args()

    # ── 构建查询列表 ──
    if args.queries is not None:
        queries = args.queries
    else:
        queries = [args.query] if args.query else ["(empty)"]

    top_k_values = [5, 10, 20]

    # 加载评分数据
    full_data_path = Path(args.data_dir) / "full_data.json"
    ratings = load_jsonl(full_data_path)
    print(f"Loaded {len(ratings)} ratings from {full_data_path}")

    # 时间戳切分
    train_ratings, test_holdout = build_user_holdout(ratings)
    print(f"Train users: {len(train_ratings)}, Test users: {len(test_holdout)}")

    # 可选：对测试用户做随机采样以加速开发迭代
    if args.sample_users > 0 and args.sample_users < len(test_holdout):
        rng = random.Random(42)
        sampled_uids = rng.sample(sorted(test_holdout.keys()), k=args.sample_users)
        test_holdout = {uid: test_holdout[uid] for uid in sampled_uids}
        print(f"Sampled {len(test_holdout)} users for evaluation")

    # ——— Baselines ———
    from src.basic_recommender import load_movie_stats, BasicRecommender

    movie_stats = load_movie_stats(full_data_path)
    popular = PopularBaseline(movie_stats)

    # 包装：传入 train_ratings 以排除已看电影（公平对比）
    class BaselineEvaluator:
        def __init__(self, inner, train_ratings):
            self.inner = inner
            self.train_ratings = train_ratings

        def recommend(self, user_id, top_k):
            exclude = set(self.train_ratings.get(user_id, []))
            return self.inner.recommend(user_id=user_id, top_k=top_k, exclude_ids=exclude)

    popular_eval = BaselineEvaluator(popular, train_ratings)

    print("\nEvaluating Popular Baseline (Bayesian)...")
    popular_metrics = evaluate(popular_eval, test_holdout, train_ratings, top_k_values, label="Popular")
    print(f"  Users evaluated: {popular_metrics['Popular_users_evaluated']}"
          f" / {popular_metrics['Popular_users_total']}")

    if args.baseline_only:
        for k in top_k_values:
            print(f"  Popular Recall@{k}: {popular_metrics[f'Popular_Recall@{k}']:.4f}")
            print(f"  Popular NDCG@{k}:  {popular_metrics[f'Popular_NDCG@{k}']:.4f}")
        return

    # ——— Agent (ReAct + DeepSeek) ———
    import os
    from src.react_agent import ReActAgent, DeepSeekLLM

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Required for DeepSeekLLM.")
        return
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    print(f"ReAct Agent: DeepSeekLLM model={model}")

    # ── 向量后端 ──
    if args.chroma:
        from src.chroma_store import ChromaMovieStore
        from src.basic_recommender import BasicRecommender
        chroma_store = ChromaMovieStore(persist_dir="data/chroma")
        recommender = BasicRecommender(chroma_store=chroma_store)
        backend_label = "ChromaDB"
    else:
        recommender = None
        backend_label = "FAISS"
    print(f"Backend: {backend_label}")

    class AgentEvaluator:
        def __init__(self, inner, query, train_ratings, query_tag=""):
            self.inner = inner
            self.query = query
            self.train_ratings = train_ratings
            self._tag = query_tag

        def recommend(self, user_id, top_k):
            exclude = set(self.train_ratings.get(user_id, []))
            result = self.inner.invoke(
                user_id=user_id, query=self.query, top_k=top_k,
                exclude_ids=exclude,
                session_id=f"eval_{self._tag}_u{user_id}",
            )
            # 首 3 用户采样工具调用路径
            if not hasattr(self, "_tool_samples"):
                self._tool_samples = []
            if len(self._tool_samples) < 3:
                self._tool_samples.append({
                    "user": user_id,
                    "iters": result.get("iterations", 0),
                    "decision": (result.get("decision_reason", "") or "")[:120],
                    "obs": (result.get("observation", "") or "")[:120],
                })
            recs = result.get("results", [])
            if not recs:
                return []
            if isinstance(recs[0], dict):
                return [rec.get("movie_id") for rec in recs if rec.get("movie_id")]
            return recs

    llm = DeepSeekLLM(api_key=api_key, model=model)
    react_agent = ReActAgent(llm=llm, recommender=recommender)

    # ── 分类评估: query → 相关类型映射 ──
    QUERY_GENRES = {
        "科幻动作":      ["Sci-Fi", "Action"],
        "轻松搞笑":      ["Comedy"],
        "2020年的悬疑片": ["Mystery", "Thriller"],
        # 未列出的 query 不过滤 (全量测试集)
    }

    def filter_test_holdout(test_holdout, allowed_genres, movie_lookup):
        """过滤测试集: 只保留属于 allowed_genres 的电影。
        用户如果没有相关类型的测试电影则被剔除。"""
        filtered = {}
        for uid, test_ids in test_holdout.items():
            relevant = [mid for mid in test_ids
                        if mid in movie_lookup
                        and any(g in allowed_genres for g in movie_lookup[mid].genres)]
            if relevant:
                filtered[uid] = relevant
        return filtered

    # 构建 movie→genres 索引 (BasicRecommender 的 movie_lookup 与向量后端无关)
    movie_lookup_for_genre = BasicRecommender().movie_lookup


    # ── 多 Query 评估 ──
    agent_recalls = {}   # query_label → {full_recall, filtered_recall, ...}
    for qi, query in enumerate(queries):
        query_label = f'"{query}"' if query else "(empty)"
        allowed_genres = QUERY_GENRES.get(query, None)

        print(f"\n[{qi+1}/{len(queries)}] Evaluating: query={query_label}", end="")
        if allowed_genres:
            filtered_test = filter_test_holdout(
                test_holdout, allowed_genres, movie_lookup_for_genre)
            print(f"  [过滤到 {', '.join(allowed_genres)}: "
                  f"{len(filtered_test)} users, "
                  f"{sum(len(v) for v in filtered_test.values())} test movies]")
        else:
            filtered_test = test_holdout
            print(f"  [全类型, {len(filtered_test)} users]")

        label = f"q{qi}"
        agent_eval = AgentEvaluator(react_agent, query, train_ratings, query_tag=f"q{qi}")
        agent_metrics = evaluate(agent_eval, filtered_test, train_ratings, top_k_values, label=label)
        print(f"  Users: {agent_metrics[label + '_users_evaluated']}"
              f" / {agent_metrics[label + '_users_total']}")

        # 打印工具调用采样
        if hasattr(agent_eval, "_tool_samples"):
            for s in agent_eval._tool_samples:
                print(f"    [trace u={s['user']} iters={s['iters']}] "
                      f"decision={s['decision'][:80]}")

        agent_recalls[query_label] = {
            k.replace(label + "_", ""): v
            for k, v in agent_metrics.items()
            if k.startswith(label + "_")
        }
        agent_recalls[query_label]["_genres"] = allowed_genres

    # ── 综合报告 ──
    print("\n" + "=" * 90)
    print("  综合评估报告：分类评估 (过滤 vs 全量)  ×  Recall@20")
    print("=" * 90)

    pop_full_r20 = popular_metrics.get("Popular_Recall@20", 0)
    pop_full_n20 = popular_metrics.get("Popular_NDCG@20", 0)

    # 也计算 Popular 在各分类下的过滤基线
    print(f"  {'Query':<26s} {'类型过滤':<16s} {'Users':>6s} {'Pop_R@20':>9s} {'Agent_R@20':>10s} {'Agent_N@20':>10s}")
    print(f"  {'-'*26} {'-'*16} {'-'*6} {'-'*9} {'-'*10} {'-'*10}")

    for query_label, metrics in agent_recalls.items():
        agent_r20 = metrics.get("Recall@20", 0)
        agent_n20 = metrics.get("NDCG@20", 0)
        genres = metrics.get("_genres")
        genre_str = ", ".join(genres) if genres else "(全类型)"
        users = metrics.get("users_evaluated", "?")

        # Popular baseline in the same filtered test set
        if genres:
            filtered_test = filter_test_holdout(
                test_holdout, genres, movie_lookup_for_genre)
            pop_filtered = evaluate(
                popular_eval, filtered_test, train_ratings, top_k_values, label="PopF")
            pop_filtered_r20 = pop_filtered.get("PopF_Recall@20", 0)
        else:
            pop_filtered_r20 = pop_full_r20

        print(f"  {query_label:<26s} {genre_str:<16s} {str(users):>6s} "
              f"{pop_filtered_r20:>9.4f} {agent_r20:>10.4f} {agent_n20:>10.4f}")

    print(f"\n  Popular (全量) Recall@20 = {pop_full_r20:.4f}  NDCG@20 = {pop_full_n20:.4f}")
    print(f"  Users: {popular_metrics.get('Popular_users_evaluated', 0)}"
          f" / {popular_metrics.get('Popular_users_total', 0)}")
    print("=" * 90)


if __name__ == "__main__":
    main()

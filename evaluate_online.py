"""
在线评估: LLM-as-Judge 对推荐结果的 query 相关性打分。

原理: 离线评估(Popular baseline)衡量的是"用户是否看过",
      无法衡量"推荐是否符合 query 意图"。
      LLM-as-Judge 直接评估后者。

指标:
    Query-Relevance@K: 前K个推荐中与query相关的比例 (0-1)
    Query-NDCG@K: 相关推荐排在越前面分数越高

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    python evaluate_online.py --query "让人想旅行的电影" --top-k 20
    python evaluate_online.py --all-queries  # 跑全部测试query
"""

import argparse, json, os, sys
sys.path.insert(0, ".")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.react_agent import ReActAgent, DeepSeekLLM
from src.basic_recommender import BasicRecommender


JUDGE_PROMPT = """你是一个电影推荐质量评估员。给定一个用户查询和一份推荐列表，
判断每部推荐电影是否与查询相关。

评分标准:
  2 = 高度相关: 电影的主题/类型/氛围完美匹配查询
  1 = 部分相关: 电影与查询有部分关联
  0 = 不相关: 电影与查询毫无关联

只输出 JSON 数组, 每个元素是 {"title": "...", "score": 0/1/2, "reason": "一句话原因"}。
不要输出其他内容。"""


def llm_judge(llm, query: str, recommendations: list) -> list:
    """用 LLM 对推荐列表逐一打分"""
    movies_text = "\n".join(
        f"{i+1}. {r['title']} ({r.get('release_year','?')}) - genres: {', '.join(r.get('genres',[]))}"
        for i, r in enumerate(recommendations)
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"查询: {query}\n\n推荐列表:\n{movies_text}"},
    ]
    resp = llm.generate(messages, tools=[])
    content = resp.get("content", "")
    try:
        # 提取 JSON
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass
    return []


def query_relevance_at_k(judgments: list, k: int, threshold: int = 1) -> float:
    """Query-Relevance@K: 前K个中得分>=threshold的比例"""
    relevant = sum(1 for j in judgments[:k] if j.get("score", 0) >= threshold)
    return relevant / min(k, len(judgments)) if judgments else 0


def query_ndcg_at_k(judgments: list, k: int) -> float:
    """Query-NDCG@K: 用 0/1/2 作为相关性分数计算 NDCG"""
    import math
    dcg = sum(
        j.get("score", 0) / math.log2(i + 2)
        for i, j in enumerate(judgments[:k])
    )
    ideal = sorted([j.get("score", 0) for j in judgments], reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="让人想旅行的电影")
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--user-id", type=int, default=1)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    llm = DeepSeekLLM(api_key=api_key, model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    agent = ReActAgent(llm=llm)

    queries = [
        # ── 简单类型/情绪 ──
        "科幻动作",
        "轻松搞笑",
        "推荐好看的电影",
        "让人想旅行的电影",
        "烧脑悬疑",
        "治愈温暖",
        # ── 长句/复杂语义 ──
        "请推荐一部关于家族传承与帮派历史演进的黑帮犯罪片",
        "我想要看那种结局完全意想不到、前面所有细节在最后一刻串联起来的悬疑片",
        "有没有那种描述普通人面对巨大灾难时展现出非凡勇气的灾难片",
        "推荐一些关于人工智能觉醒、探讨意识本质的科幻电影",
        "适合失恋后一个人看的、能让人重新振作起来的电影",
        # ── 类型+时间+质量约束 ──
        "90年代的经典动作片，要有汽车追逐和枪战",
        "2000年以后评分最高的科幻惊悚片",
        "像教父一样关于权力与背叛的黑帮史诗",
        # ── 混合情绪 + 场景 ──
        "下雨天窝在家里看的温暖治愈的动画片",
        "和朋友一起看的搞笑无厘头喜剧，不要太长",
        "深夜一个人静静看的、能引发思考的文艺剧情片",
    ] if args.all_queries else [args.query]

    # 分类标签
    QUERY_CATEGORIES = {
        "科幻动作": "简单类型",
        "轻松搞笑": "简单情绪",
        "推荐好看的电影": "模糊查询",
        "让人想旅行的电影": "抽象语义",
        "烧脑悬疑": "简单情绪",
        "治愈温暖": "简单情绪",
        "请推荐一部关于家族传承与帮派历史演进的黑帮犯罪片": "长句语义",
        "我想要看那种结局完全意想不到、前面所有细节在最后一刻串联起来的悬疑片": "长句语义",
        "有没有那种描述普通人面对巨大灾难时展现出非凡勇气的灾难片": "长句语义",
        "推荐一些关于人工智能觉醒、探讨意识本质的科幻电影": "长句语义",
        "适合失恋后一个人看的、能让人重新振作起来的电影": "长句语义",
        "90年代的经典动作片，要有汽车追逐和枪战": "类型+时间+质量",
        "2000年以后评分最高的科幻惊悚片": "类型+时间+质量",
        "像教父一样关于权力与背叛的黑帮史诗": "类型+时间+质量",
        "下雨天窝在家里看的温暖治愈的动画片": "混合情绪+场景",
        "和朋友一起看的搞笑无厘头喜剧，不要太长": "混合情绪+场景",
        "深夜一个人静静看的、能引发思考的文艺剧情片": "混合情绪+场景",
    }

    print(f"  {'Query':<50s} {'Cat':<14s} {'Agent_R':>8s} {'Agent_N':>8s} {'Pop_R':>8s} {'AgentWin':>8s}")
    print(f"  {'-'*50} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    results_by_cat = {}  # category → [(agent_r, pop_r), ...]
    for query in queries:
        # Agent 推荐
        result = agent.invoke(user_id=args.user_id, query=query, top_k=args.top_k,
                              session_id=f"judge_{query}")
        agent_recs = result.get("results", [])[:args.top_k]

        # Popular 基线
        rec = BasicRecommender()
        popular = rec._cold_start_recommend(top_k=args.top_k)
        popular_recs = [{
            "title": r.title, "genres": r.genres,
            "release_year": r.release_year
        } for r in popular]

        # LLM 打分
        agent_judgments = llm_judge(llm, query, agent_recs)
        popular_judgments = llm_judge(llm, query, popular_recs)

        agent_r = query_relevance_at_k(agent_judgments, 20)
        agent_n = query_ndcg_at_k(agent_judgments, 20)
        pop_r = query_relevance_at_k(popular_judgments, 20)
        win = "+" if agent_r > pop_r else ("=" if agent_r == pop_r else "-")

        cat = QUERY_CATEGORIES.get(query, "其他")
        results_by_cat.setdefault(cat, []).append((agent_r, pop_r, agent_n))

        # 截断长 query 显示
        display = query[:46] + "..." if len(query) > 48 else query
        print(f"  {display:<50s} {cat:<14s} {agent_r:>8.4f} {agent_n:>8.4f} {pop_r:>8.4f} {win:>8s}")

    # ── 分类汇总 ──
    print()
    print(f"  {'Category':<16s} {'#Q':>4s} {'Avg Agent_R':>11s} {'Avg Pop_R':>10s} {'Agent Win%':>10s}")
    print(f"  {'-'*16} {'-'*4} {'-'*11} {'-'*10} {'-'*10}")
    for cat, vals in sorted(results_by_cat.items()):
        n = len(vals)
        avg_agent = sum(v[0] for v in vals) / n
        avg_pop = sum(v[1] for v in vals) / n
        wins = sum(1 for v in vals if v[0] > v[1])
        print(f"  {cat:<16s} {n:>4d} {avg_agent:>11.4f} {avg_pop:>10.4f} {wins:>4d}/{n:<4d}")

    # 总计
    all_vals = [v for vals in results_by_cat.values() for v in vals]
    total = len(all_vals)
    total_agent = sum(v[0] for v in all_vals) / total
    total_pop = sum(v[1] for v in all_vals) / total
    total_wins = sum(1 for v in all_vals if v[0] > v[1])
    print(f"  {'─'*16} {'─'*4} {'─'*11} {'─'*10} {'─'*10}")
    print(f"  {'总计':<16s} {total:>4d} {total_agent:>11.4f} {total_pop:>10.4f} {total_wins:>4d}/{total:<4d}")


if __name__ == "__main__":
    main()

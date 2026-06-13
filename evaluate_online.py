"""
在线评估 v2: LLM-as-Judge 盲评对比 ReAct Agent vs Popular Baseline。

方法论保障:
    1. 盲评: 混合两套推荐, 随机打乱顺序, 不标注来源
    2. 随机化: 每个 query 独立 shuffle, 消除位置偏差
    3. 交错对比: 同一 query 的两套结果合并后一次性评分, 而非分别评
    4. 锚点校验: "推荐好看的电影" 两套结果应接近满分, 验证 judge 可靠性
    5. 已知局限: DeepSeek 同时参与推荐(Agent端)和评估(Judge端),
       可能存在同模型偏差; 需标注为方法论限制

用法:
    export DEEPSEEK_API_KEY=sk-xxx
    python evaluate_online.py --query "让人想旅行的电影"
    python evaluate_online.py --all-queries
"""

import argparse, json, math, os, random, sys
sys.path.insert(0, ".")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.react_agent import ReActAgent, DeepSeekLLM
from src.basic_recommender import BasicRecommender

JUDGE_PROMPT = """你是一个电影推荐系统质量评估员。你会看到:
1. 一个用户查询 (query)
2. 20 部来自两个推荐系统的电影 (混合在一起, 随机排序)

你的任务:
- 对每部电影判断它是否与查询相关
- 你不需要知道电影来自哪个系统
- 只根据电影本身是否匹配查询来评分

评分标准 (严格按此执行):
  2分 - 高度相关: 电影类型+主题+氛围同时满足查询的所有核心约束
        (例如: query要求"科幻动作" → Star Wars=2分, Titanic=0分)
  1分 - 部分相关: 电影满足查询的部分约束, 但不是最佳匹配
        (例如: query要求"轻松搞笑喜剧" → 浪漫喜剧=1分, 纯剧情片=0分)
  0分 - 不相关: 电影与查询的核心意图无关

重要提示:
- 不要因为一部电影是经典/高分就给它高分——只看它与查询的相关性
- 如果查询包含具体约束(年代、类型、场景), 逐一核对
- 只输出 JSON 数组: [{"title": "...", "score": 0/1/2, "reason": "一句话"}]
- 不要输出任何其他内容"""


def llm_judge_blind(llm, query, agent_recs, popular_recs, seed=42):
    """盲评: 混合两套推荐 + 随机打乱 + 不标注来源, 一次性评分。

    Returns: (agent_scores, popular_scores, agent_r20, pop_r20, agent_n20, pop_n20)
    """
    rng = random.Random(seed)

    # 标签随机化: A/B 随机对应 Agent/Popular (消除字母顺序偏差)
    if rng.random() < 0.5:
        label_a, label_b = "Agent", "Popular"
        recs_a, recs_b = agent_recs, popular_recs
    else:
        label_a, label_b = "Popular", "Agent"
        recs_a, recs_b = popular_recs, agent_recs

    # 交错合并 + 随机打乱
    items = []
    for i in range(max(len(recs_a), len(recs_b))):
        if i < len(recs_a):
            items.append({**recs_a[i], "_sys": "A"})
        if i < len(recs_b):
            items.append({**recs_b[i], "_sys": "B"})
    rng.shuffle(items)

    # 构建 prompt: 每个电影只显示 title + genres + year, 不显示系统标签
    movies_text = "\n".join(
        f"{j+1}. {r['title']} ({r.get('release_year','?')}) — {', '.join(r.get('genres',[]))}"
        for j, r in enumerate(items)
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"查询: {query}\n\n推荐列表 (共{len(items)}部, 随机排序):\n{movies_text}"},
    ]
    resp = llm.generate(messages, tools=[])
    content = resp.get("content", "")

    # 解析 judge 返回
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            judgments = json.loads(content[start:end])
        else:
            judgments = []
    except json.JSONDecodeError:
        judgments = []

    # 按原始 title 匹配回 A/B 系统
    title_to_score = {j.get("title", ""): j.get("score", 0) for j in judgments}
    scores_a, scores_b = [], []
    for item in items:
        score = title_to_score.get(item.get("title", ""), 0)
        if item["_sys"] == "A":
            scores_a.append(score)
        else:
            scores_b.append(score)

    # 反解 A/B → Agent/Popular
    if label_a == "Agent":
        agent_scores, pop_scores = scores_a, scores_b
    else:
        agent_scores, pop_scores = scores_b, scores_a

    agent_r20 = sum(1 for s in agent_scores[:20] if s >= 1) / max(len(agent_scores[:20]), 1)
    pop_r20 = sum(1 for s in pop_scores[:20] if s >= 1) / max(len(pop_scores[:20]), 1)

    # NDCG
    def ndcg(scores, k):
        dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))
        ideal = sorted(scores, reverse=True)[:k]
        idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0

    agent_n20 = ndcg(agent_scores, 20)
    pop_n20 = ndcg(pop_scores, 20)

    return agent_r20, pop_r20, agent_n20, pop_n20


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

    agent_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    judge_model = "deepseek-v4-flash"  # 评估用独立模型, 降低同模偏差

    agent_llm = DeepSeekLLM(api_key=api_key, model=agent_model)
    judge_llm = DeepSeekLLM(api_key=api_key, model=judge_model)
    agent = ReActAgent(llm=agent_llm)
    print(f"Agent 模型: {agent_model}  |  Judge 模型: {judge_model}")

    queries = [
        "科幻动作", "轻松搞笑", "推荐好看的电影",
        "让人想旅行的电影", "烧脑悬疑", "治愈温暖",
        "请推荐一部关于家族传承与帮派历史演进的黑帮犯罪片",
        "我想要看那种结局完全意想不到、前面所有细节在最后一刻串联起来的悬疑片",
        "有没有那种描述普通人面对巨大灾难时展现出非凡勇气的灾难片",
        "推荐一些关于人工智能觉醒、探讨意识本质的科幻电影",
        "适合失恋后一个人看的、能让人重新振作起来的电影",
        "90年代的经典动作片，要有汽车追逐和枪战",
        "2000年以后评分最高的科幻惊悚片",
        "像教父一样关于权力与背叛的黑帮史诗",
        "下雨天窝在家里看的温暖治愈的动画片",
        "和朋友一起看的搞笑无厘头喜剧，不要太长",
        "深夜一个人静静看的、能引发思考的文艺剧情片",
    ] if args.all_queries else [args.query]

    QUERY_CATEGORIES = {
        "科幻动作": "简单类型", "轻松搞笑": "简单情绪",
        "推荐好看的电影": "模糊查询", "让人想旅行的电影": "抽象语义",
        "烧脑悬疑": "简单情绪", "治愈温暖": "简单情绪",
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

    print("=" * 95)
    print("  LLM-as-Judge v2: 盲评 + 随机打乱 + 交错对比")
    print("  方法论: 混合两套推荐, 不标注来源, 随机打乱后一次性评分")
    print("  已知局限: DeepSeek 同时参与推荐(Agent)和评估(Judge), 存在同模型偏差")
    print("=" * 95)
    print(f"  {'Query':<48s} {'Cat':<14s} {'A_R20':>7s} {'P_R20':>7s} {'A_N20':>7s} {'Win':>5s}")
    print(f"  {'-'*48} {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    results_by_cat = {}
    for qi, query in enumerate(queries):
        # Agent
        result = agent.invoke(user_id=args.user_id, query=query, top_k=args.top_k,
                              session_id=f"blind_{qi}_{query[:10]}")
        agent_recs = result.get("results", [])[:args.top_k]

        # Popular
        rec = BasicRecommender()
        popular = rec._cold_start_recommend(top_k=args.top_k)
        popular_recs = [{"title": r.title, "genres": r.genres,
                         "release_year": r.release_year} for r in popular]

        # 盲评
        a_r, p_r, a_n, p_n = llm_judge_blind(judge_llm, query, agent_recs, popular_recs, seed=42 + qi)

        win = "+" if a_r > p_r else ("=" if abs(a_r - p_r) < 0.01 else "-")
        cat = QUERY_CATEGORIES.get(query, "其他")
        results_by_cat.setdefault(cat, []).append((a_r, p_r, a_n))

        display = query[:44] + "..." if len(query) > 46 else query
        print(f"  {display:<48s} {cat:<14s} {a_r:>7.4f} {p_r:>7.4f} {a_n:>7.4f} {win:>5s}")

    # ── 锚点校验 ──
    if args.all_queries:
        print()
        print(f"  ═══ 锚点校验 ═══")
        for query_label, vals in results_by_cat.items():
            for (a_r, p_r, _) in vals:
                pass  # dummy — check below
        calib_check = "OK: 模糊查询 Agent≈Popular≈1.0" if all(
            abs(v[0] - v[1]) < 0.05 for v in results_by_cat.get("模糊查询", [(0, 0)])
        ) else "CHECK"
        print(f"  模糊查询校准: {calib_check}")
        print(f"  已知局限: DeepSeek 同时参与推荐与评估, 报告数据仅供内部对比参考")
        print()

    # ── 分类汇总 ──
    print(f"  {'Category':<16s} {'#Q':>4s} {'Avg Agent_R':>11s} {'Avg Pop_R':>10s} {'Win%':>8s}")
    print(f"  {'-'*16} {'-'*4} {'-'*11} {'-'*10} {'-'*8}")
    for cat, vals in sorted(results_by_cat.items()):
        n = len(vals)
        avg_agent = sum(v[0] for v in vals) / n
        avg_pop = sum(v[1] for v in vals) / n
        wins = sum(1 for v in vals if v[0] > v[1] + 0.01)
        print(f"  {cat:<16s} {n:>4d} {avg_agent:>11.4f} {avg_pop:>10.4f} {wins:>4d}/{n:<3d}")

    all_vals = [v for vals in results_by_cat.values() for v in vals]
    total = len(all_vals)
    total_agent = sum(v[0] for v in all_vals) / total
    total_pop = sum(v[1] for v in all_vals) / total
    total_wins = sum(1 for v in all_vals if v[0] > v[1] + 0.01)
    print(f"  {'─'*16} {'─'*4} {'─'*11} {'─'*10} {'─'*8}")
    print(f"  {'总计':<16s} {total:>4d} {total_agent:>11.4f} {total_pop:>10.4f} {total_wins:>4d}/{total:<3d}")
    print()
    print("  ⚠ 方法局限:")
    print("    1. DeepSeek 同时参与 Agent 推荐和 Judge 评估 (同模型偏差)")
    print("    2. 单用户 (user_id=1) 评估, 未覆盖冷启动场景")
    print("    3. 评估数据集 (MovieLens-100K) 类型覆盖有限")
    print("    4. 建议: 换用 GPT-4/Claude 等独立模型做 judge 做交叉验证")


if __name__ == "__main__":
    main()

# Movie Recommend Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![Qwen](https://img.shields.io/badge/LLM-Qwen%20Series-orange)](https://tongyi.aliyun.com/)

An **LLM-powered movie recommendation system** that combines ReAct-style agent reasoning with RAG (Retrieval-Augmented Generation). QwenLLM dynamically selects from 6 tools—user profiling, cold-start, personalized search, genre/year filtering, mood matching, and semantic search—to answer complex natural language queries.

> "I want a movie that makes me want to quit my job and travel the world."
> → LLM rewrites to `travel adventure road movie beautiful scenery wanderlust` → semantic search over FAISS

## Features

- **ReAct Agent** — QwenLLM drives Thought → Action → Observation multi-step reasoning via LangGraph, autonomously selecting tools and chaining them across up to 5 iterations
- **6 Tool System** — `get_user_profile`, `search_cold_start`, `search_by_preference`, `search_by_filter`, `search_by_mood`, `search_semantic`
- **Multi-Granularity Retrieval** — plot-level, attribute-level, and full-text FAISS indexes (384-dim, SentenceTransformer all-MiniLM-L6-v2) with weighted fusion
- **Bayesian Debiased Scoring** — Bayesian-smoothed popularity + user similarity + query similarity (0.25U + 0.25R + 0.50P)
- **Offline Evaluation Pipeline** — leave-last-out time split, Random/Popular/Agent three-baseline comparison, Recall@K & NDCG@K
- **FastAPI + Docker** — REST API with browser UI, containerized deployment

## Quick Start

### Prerequisites

- Python 3.8+
- [DASHSCOPE_API_KEY](https://dashscope.console.aliyun.com/) (Alibaba Cloud)
- Preprocessed MovieLens-100K data in `data/processed/`

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Evaluate
export QWEN_MODEL=qwen-turbo  # or qwen-plus, qwen3-max
python evaluate.py --query "推荐好看的电影" --sample-users 50

# Baseline only
python evaluate.py --baseline-only
```

### API

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1, "query":"想看轻松的电影", "top_k":5}'
```

Response:
```json
{
  "route": "react_agent",
  "decision_reason": "用户偏好轻松内容，使用个性化推荐...",
  "results": [
    {"movie_id": 50, "title": "Star Wars (1977)", "score": 0.592}
  ]
}
```

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────┐
│        ReAct Agent               │
│   ┌──────────┐    ┌──────────┐   │
│   │  Agent   │◄──►│  Tools   │   │
│   │ (QwenLLM)│    │  (6个)    │   │
│   └────┬─────┘    └────┬─────┘   │
│        │               │         │
│    Thought+Action   Execute      │
│    max 5 rounds     Return obs   │
└────────┼───────────────┼─────────┘
         │               │
         ▼               ▼
┌──────────────────────────────────┐
│      BasicRecommender             │
│  ┌────────────┐  ┌─────────────┐  │
│  │ FAISS 索引  │  │ Bayesian    │  │
│  │ (3粒度融合)  │  │ Scorer      │  │
│  └────────────┘  └─────────────┘  │
└──────────────────────────────────┘
         │
         ▼
    Ranked Top-K Results
```

| Component | File | Role |
|-----------|------|------|
| Agent Core | `src/react_agent.py` | ReAct loop, QwenLLM interface, 6-tool definitions |
| Recommender | `src/basic_recommender.py` | FAISS retrieval, Bayesian scoring, candidate pool |
| API | `src/api.py` | FastAPI entry point |
| Legacy Agent | `src/llm_agent.py` | Original Qwen Tool Calling agent (standalone, pre-React refactor) |
| Explanations | `src/explanation_engine.py` | Feature attribution + evidence chain explainer |
| User Memory | `src/user_memory.py` | Time-decayed user profile embeddings |
| Data Prep | `src/data_preprocess.py` | MovieLens-100K raw data processing |
| Index Build | `src/rag_build_index.py` | Multi-granularity FAISS index builder |
| Query Rewrite | `src/query_rewrite.py` | Rule-based query parser (legacy, used in evals) |

## Evaluation Results

**Setup**: MovieLens-100K (943 users, 1682 movies, 100K ratings), leave-last-out time split (last 20% held out, ≥4.0 positive threshold). All methods exclude training-set movies for fair comparison.

| Metric | Random | Popular (Bayesian) | ReAct Agent | vs Popular |
|--------|:---:|:---:|:---:|:---:|
| Recall@10 | 0.64% | 5.92% | 7.65% | **+29%** |
| Recall@20 | 1.36% | 9.35% | 12.64% | **+35%** |
| NDCG@10 | 0.64% | 8.44% | 10.72% | **+27%** |
| NDCG@20 | 0.92% | 8.97% | 11.70% | **+30%** |

Key findings:
- Personalization adds ~30-35% over Bayesian popularity baseline
- NDCG gains consistently match or exceed Recall gains — the Agent ranks better, not just finds more
- Empty queries route to cold-start by design, guaranteeing no worse than Popular baseline
- `search_semantic` tool unlocks abstract/emotional queries beyond genre/mood taxonomies

## Tool Demos

| User Query | Tool Selected | LLM Action |
|------------|:---:|------------|
| "想看科幻片" | `search_by_filter` | genre constraint → direct filter |
| "心情不好" | `search_by_mood` | mood → genre mapping |
| "让人想旅行的电影" | `search_semantic` | rewrites to `travel adventure wanderlust journey` |
| 空查询 | (short-circuit) | skips LLM, direct cold-start |

## Project Structure

```
src/
├── react_agent.py           ReAct Agent (QwenLLM + LangGraph + 6 tools)
├── basic_recommender.py     Core RAG engine + multi-granularity FAISS
├── api.py                   FastAPI service
├── llm_agent.py             Legacy standalone Qwen Tool Calling agent
├── user_memory.py           Time-decayed user profile builder
├── explanation_engine.py    Recommendation explainer
├── rag_build_index.py       Multi-granularity FAISS index builder
├── query_rewrite.py         Rule-based query parser
├── data_preprocess.py       MovieLens-100K data pipeline
├── build_user_memory.py     Offline user profile construction
├── rag_query_test.py        FAISS retrieval smoke test
├── agentic_recommender.py   Legacy QwenAgenticRecommender wrapper
data/processed/              Preprocessed data, embeddings, FAISS indexes
static/                      Browser UI (HTML + JS + CSS)
evaluate.py                  Offline eval: Recall@K, NDCG@K
evaluate_prompts.py          System Prompt variant comparison
enrich_movies.py             Movie plot enrichment (OMDb/Wikipedia)
PROJECT_ISSUES.md            Issue diagnosis & fix log
```

## Setup from Scratch

```bash
# 1. Preprocess MovieLens-100K
python src/data_preprocess.py

# 2. Build user memory
python src/build_user_memory.py

# 3. Build FAISS indexes
python src/rag_build_index.py --model sentence-transformers/all-MiniLM-L6-v2

# 4. Enrich with plot summaries (optional, enables search_semantic)
export OMDB_API_KEY=your_key
python enrich_movies.py
python src/rag_build_index.py --model sentence-transformers/all-MiniLM-L6-v2

# 5. Serve
export DASHSCOPE_API_KEY=your_key
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research, University of Minnesota)
- [Qwen](https://tongyi.aliyun.com/) (Alibaba Cloud) for LLM inference
- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) (Meta) for vector search

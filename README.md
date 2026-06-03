# Movie Recommend Agent — 电影推荐智能体

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-purple)](https://langchain-ai.github.io/langgraph/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%2FQwen-orange)](https://www.deepseek.com/)

一个**基于大语言模型的电影推荐系统**，融合了 **ReAct 智能体推理**、**RAG 检索增强生成**与**用户画像协同过滤**。LLM 自主从 6 种工具中选择——用户画像查询、冷启动推荐、个性化搜索、类型/年份过滤、情绪匹配、语义搜索——来回答复杂的自然语言查询。

> *"我想看一部让人想辞职去旅行的电影"*
> → LLM 重写为 `travel adventure road movie beautiful scenery wanderlust` → 在 FAISS 向量库中语义搜索

## 特性

- **ReAct 智能体** — 基于 LangGraph 的 Thought → Action → Observation 多步推理，LLM 自主选择工具并链式调用，最多 5 轮迭代
- **6 工具系统** — `get_user_profile`、`search_cold_start`、`search_by_preference`、`search_by_filter`、`search_by_mood`、`search_semantic`
- **多粒度检索** — 剧情级、属性级、全文级三重 FAISS 索引（384 维，SentenceTransformer all-MiniLM-L6-v2），加权融合召回
- **贝叶斯去偏评分** — 贝叶斯平滑流行度 + 用户相似度 + 查询相似度，权重为 0.25U + 0.25R + 0.50P
- **WebSocket 流式推荐** — 实时推送智能体的思考过程、工具调用、观察结果和最终推荐，支持随时取消
- **反馈驱动在线学习** — 用户点赞/踩实时更新用户画像向量（向量加减法），并影响后续多轮对话的推荐策略
- **多轮会话管理** — 每会话维护完整对话历史、已排除电影、累积偏好类型，支持随时重置
- **离线评估体系** — 按时间 leave-last-out 划分，对比 Random / Popular / Agent 三条基线，输出 Recall@K 与 NDCG@K
- **FastAPI 服务** — REST API + WebSocket，附带浏览器交互界面，支持 Docker 容器化

## 快速开始

### 环境要求

- Python 3.8+
- [DeepSeek API Key](https://platform.deepseek.com/) 或 [DashScope API Key](https://dashscope.console.aliyun.com/)（阿里云通义千问）
- 预处理好的 MovieLens-100K 数据（位于 `data/processed/`）

### 安装

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 启动 API 服务
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 运行离线评估（需先设置环境变量）
export DEEPSEEK_API_KEY=your_key        # DeepSeek API 密钥
export DEEPSEEK_MODEL=deepseek-chat     # 模型名称
python evaluate.py --query "推荐好看的电影" --sample-users 50

# 仅运行基线对比（不调用 LLM）
python evaluate.py --baseline-only

# 对比不同系统提示词效果
python evaluate_prompts.py
```

### API 调用示例

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1, "query":"想看轻松搞笑的电影", "top_k":5}'
```

响应示例：
```json
{
  "route": "react_agent",
  "decision_reason": "用户偏好轻松内容，使用个性化推荐...",
  "results": [
    {"movie_id": 50, "title": "Star Wars (1977)", "score": 0.592}
  ]
}
```

### 提交反馈

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1, "movie_id":50, "feedback":"like", "movie_title":"Star Wars (1977)"}'
```

## 系统架构

```
用户查询
    │
    ▼
┌─────────────────────────────────────────┐
│            ReAct Agent (LangGraph)       │
│  ┌───────────┐          ┌───────────┐   │
│  │   Agent   │◄────────►│   Tools   │   │
│  │ (DeepSeek)│          │  (6 个)    │   │
│  └─────┬─────┘          └─────┬─────┘   │
│        │                      │         │
│   思考 → 行动           执行 → 观察     │
│   最多 5 轮迭代                      │
└────────┼──────────────────────┼─────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│           BasicRecommender               │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  FAISS 索引   │  │  Bayesian Scorer │ │
│  │  (3粒度融合)   │  │  (0.25U+0.25R   │ │
│  │              │  │   +0.50P)        │ │
│  └──────────────┘  └──────────────────┘ │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    UserMemory              Explanation
   (时间衰减画像)           Engine (特征归因)
         │
         ▼
   Top-K 推荐结果
```

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| ReAct 智能体 | [src/react_agent.py](src/react_agent.py) | LangGraph ReAct 循环、LLM 接口、6 工具定义与执行、WebSocket 流式事件 |
| 推荐引擎 | [src/basic_recommender.py](src/basic_recommender.py) | FAISS 多粒度检索、贝叶斯评分、候选池构建、冷启动/过滤/情绪/语义搜索 |
| API 服务 | [src/api.py](src/api.py) | FastAPI 入口，REST + WebSocket，反馈处理与会话管理 |
| 用户记忆 | [src/user_memory.py](src/user_memory.py) | 时间衰减用户画像构建、30 天半衰期、反馈在线微调 |
| 解释引擎 | [src/explanation_engine.py](src/explanation_engine.py) | 特征归因 + 证据链解释生成 |

### 工具矩阵

| 工具名称 | 触发条件 | 功能描述 |
|----------|----------|----------|
| `get_user_profile` | 查询模糊，需了解用户偏好 | 获取用户画像：高分电影、偏好类型、平均评分 |
| `search_cold_start` | 新用户或查询为空 | 仅基于贝叶斯流行度推荐热门电影 |
| `search_by_preference` | 有明确用户画像 | 融合用户画像向量与查询向量进行个性化推荐 |
| `search_by_filter` | 查询含类型/年份关键词 | 解析类型和年份约束，精确过滤 |
| `search_by_mood` | 查询表达情绪 | 中文情绪词映射到电影类型（如"心情不好"→喜剧） |
| `search_semantic` | 抽象/意境类查询 | LLM 将中文查询重写为英文关键词，对剧情嵌入做语义搜索 |

## 评估结果

**实验设置**：MovieLens-100K（943 用户，1682 电影，10 万条评分），按时间 leave-last-out 划分（最后 20% 作为测试集，≥4.0 为正样本）。所有方法均排除训练集中已出现的电影以保证公平对比。

| 指标 | 随机 | 流行度 (贝叶斯) | ReAct Agent | 相对提升 |
|------|:---:|:---:|:---:|:---:|
| Recall@10 | 0.64% | 5.92% | 7.65% | **+29%** |
| Recall@20 | 1.36% | 9.35% | 12.64% | **+35%** |
| NDCG@10 | 0.64% | 8.44% | 10.72% | **+27%** |
| NDCG@20 | 0.92% | 8.97% | 11.70% | **+30%** |

关键发现：
- 个性化推荐相比贝叶斯流行度基线提升约 **30%-35%**
- NDCG 增益与 Recall 增益匹配甚至更高——智能体不仅找到更多好电影，排序也更优
- 空查询自动路由到冷启动，保证不低于流行度基线
- `search_semantic` 工具解锁了超出传统类型/情绪分类的抽象意境查询

## 工具调用演示

| 用户查询 | 选中的工具 | LLM 执行动作 |
|----------|:---:|-------------|
| "想看科幻片" | `search_by_filter` | 解析出 genre=Sci-Fi → 精确过滤 |
| "心情不好" | `search_by_mood` | 情绪"不好"→ 映射到 Comedy |
| "让人想旅行的电影" | `search_semantic` | 重写为 `travel adventure wanderlust journey` |
| 空查询 | （短路绕过 LLM） | 直接返回冷启动热门推荐 |

## 项目结构

```
.
├── src/
│   ├── react_agent.py           # ReAct 智能体 (LangGraph + 6工具 + WebSocket流式)
│   ├── basic_recommender.py     # 核心 RAG 引擎 + 多粒度 FAISS 检索
│   ├── api.py                   # FastAPI 服务 (REST + WebSocket)
│   ├── user_memory.py           # 时间衰减用户画像构建与在线学习
│   ├── explanation_engine.py    # 推荐解释引擎 (特征归因 + 证据链)
│   ├── rag_build_index.py       # 多粒度 FAISS 索引构建
│   ├── query_rewrite.py         # 规则查询解析器
│   ├── data_preprocess.py       # MovieLens-100K 数据预处理
│   ├── build_user_memory.py     # 离线用户画像批量构建
│   └── rag_query_test.py        # FAISS 检索冒烟测试
├── data/processed/              # 预处理数据、嵌入向量、FAISS 索引
├── static/                      # 浏览器 UI (HTML + JS + CSS)
│   ├── index.html               # 单页应用
│   ├── app.js                   # REST + WebSocket 客户端逻辑
│   └── styles.css               # 深色科技风格
├── evaluate.py                  # 离线评估：Recall@K, NDCG@K
├── evaluate_prompts.py          # 系统提示词变体对比实验
├── enrich_movies.py             # 电影剧情丰富 (OMDb/Wikipedia)
└── PROJECT_ISSUES.md            # 问题诊断与修复日志
```

## 从零搭建

```bash
# 1. 预处理 MovieLens-100K 数据集
python src/data_preprocess.py

# 2. 构建用户画像
python src/build_user_memory.py

# 3. 构建 FAISS 索引
python src/rag_build_index.py --model sentence-transformers/all-MiniLM-L6-v2

# 4.（可选）丰富电影剧情摘要，启用 search_semantic 工具
export OMDB_API_KEY=your_key
python enrich_movies.py
python src/rag_build_index.py --model sentence-transformers/all-MiniLM-L6-v2

# 5. 启动服务
export DEEPSEEK_API_KEY=your_key      # DeepSeek API 密钥
export DEEPSEEK_MODEL=deepseek-chat   # 或 deepseek-chat, qwen-turbo, qwen-plus 等
export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1  #（可选）API 地址
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000` 即可使用浏览器界面进行交互。

### 环境变量说明

| 变量名 | 必填 | 说明 |
|--------|:---:|------|
| `DEEPSEEK_API_KEY` | 是 | LLM API 密钥（DeepSeek 或 DashScope） |
| `DEEPSEEK_MODEL` | 否 | 模型名称，默认 `deepseek-chat`，可选 `qwen-turbo`、`qwen-plus`、`qwen3-max` |
| `DEEPSEEK_BASE_URL` | 否 | API 基础地址，默认 DeepSeek 官方地址 |

## 设计亮点

### ReAct 推理循环

智能体不是单次调用 LLM，而是在 LangGraph 构建的状态图中循环推理：观察用户查询 → 思考需要什么信息 → 选择调用哪个工具 → 观察工具返回结果 → 决定是否需要更多信息 → 最多 5 轮后输出最终推荐。这种显式的推理链让推荐过程可解释、可调试。

### 贝叶斯去偏评分

朴素流行度会偏向评分次数少的冷门电影（少量高分就能获得虚高的平均分）。系统采用贝叶斯平滑：`score = (C × m + N × R) / (C + N)`，其中 C 为全局平均评分次数，m 为全局平均评分。这使得只有积累了足够评分量的电影才能获得高流行度分数。

### 时间衰减用户画像

用户品味随时间变化——五年前喜欢的电影可能不再代表当前偏好。用户画像构建使用 30 天半衰期的指数衰减：`weight = exp(-ln(2) × delta_days / 30)`，近期评分比早期评分权重更高。

### 反馈驱动在线学习

用户对推荐结果的点赞/踩会立即产生两个效果：
1. **会话级**：已点赞/踩的电影加入排除列表，偏好类型融入后续查询
2. **画像级**：通过向量加减法微调用户画像嵌入（学习率 α=0.05），点赞的电影向量被加入，踩的被减去

### WebSocket 流式体验

当用户开启流式模式时，可以实时看到智能体的"思考过程"——包括推理文字、工具调用名称和参数、工具观察结果，以及最终结论。这极大地增强了系统的可解释性和用户信任感。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

## 致谢

- [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) — GroupLens Research, University of Minnesota
- [DeepSeek](https://www.deepseek.com/) — 大语言模型推理服务
- [Qwen / 通义千问](https://tongyi.aliyun.com/) — 阿里云大语言模型
- [SentenceTransformers](https://www.sbert.net/) — 文本嵌入模型
- [FAISS](https://github.com/facebookresearch/faiss) — Meta 向量相似度搜索库
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 智能体状态图框架

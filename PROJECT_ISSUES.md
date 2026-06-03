# 项目改进记录：问题排查与解决方案

本文档记录了从 2026 年 5 月项目评估开始以来遇到的所有关键问题、排查过程及解决方案，按发现时间排序。

---

## 1. 评估基线排序逻辑错误

**现象**：Popular Baseline 的 Recall@10 仅 0.08\%，Agent 相对提升超过 1000\%，但数值可疑。

**根因**：`PopularBaseline` 使用 `(avg_rating, rating_count)` tuple 排序，先按平均分排序。一部只有 3 人打了 5.0 分的冷门电影排在 1000 人打了 4.8 分的《肖申克》前面。Baseline 推荐的实际上是评分高但没人看过的冷门片。

**修复**：改用 Bayesian 加权评分：$$score = \frac{C \cdot m + N \cdot R}{C + N}$$，其中 $C$ 为全局平均评分人数（先验强度），$m$ 为全局平均分（先验均值），$R$ 为电影平均分，$N$ 为电影评分人数。修复后 Baseline Recall@20 从 1.27\% 升至 9.35\%。

**文件**：`evaluate.py` — `PopularBaseline.__init__()`

---

## 2. Agent 评分公式中 popularity 分量使用 raw avg\_rating（相同 bug）

**现象**：Agent 推荐列表被冷门高分片（如 Star Kid、Prefontaine）占据，Top-5 在不同用户间高度重叠。Agent 的 Recall 远低于修复后的 Bayesian 热门基线。

**根因**：`BasicRecommender.recommend()` 中 `popularity = normalize_to_unit(movie.avg_rating, 1.0, 5.0)`。一部 3 人评 5.0 分的电影 pop=1.0，与《Titanic》(350 人评 4.2 分) 的 pop=0.8 相比，冷门片在 0.2 的 popularity 权重下获得了满分优势。同时 `_popular_movie_ids()` 也用 `(avg_rating, rating_count)` 排序，候选池本身就包含了大量冷门片。

**修复**：
- `__init__` 中新增 `_bayesian_pop` 字典，初始化时预计算
- `_popular_movie_ids()` 改用 Bayesian 排序
- `recommend()` 等 4 处 `normalize_to_unit(movie.avg_rating, ...)` 替换为 `normalize_to_unit(self._bayesian_pop[...], ...)`
- `_branch_db_filter` 中同样修复

**修复后**：Agent Top-5 变为《Star Wars》《Schindler's List》《Shawshank》等真正热门电影。Recall 从假象的 2.67\% 变为真实的 5.09\%，差距从 -71\% 缩小到 -44\%。

**文件**：`src/basic_recommender.py`

---

## 3. 嵌入空间不匹配（rag\_build\_index.py 覆盖了 movie\_embeddings.npy）

**现象**：修复 popularity 后 Agent 仍然落后于 Popular Baseline 44 个百分点。诊断脚本显示 user\_sim 和 rag\_sim 取值范围极窄（0.55-0.75），缺乏区分力。

**根因**：`rag_build_index.py` 默认模型为 `paraphrase-multilingual-MiniLM-L12-v2`，而 `BasicRecommender` 的默认模型为 `all-MiniLM-L6-v2`。执行 `rag_build_index.py` 构建多粒度索引时：
1. 用新模型重新编码了全部电影 embedding，**覆盖**了 `movie_embeddings.npy`
2. 更新了 `movie_index_meta.json` 中的 model 字段
3. `user_embeddings.npy` 仍保留了旧模型的嵌入

结果：`cosine_similarity(user_vec, movie_vec)` 在两个不同语义空间之间计算，本质上是随机噪音。评分公式中 80\% 权重（0.4User + 0.4RAG）依赖跨空间相似度。

**修复**：重新运行 `rag_build_index.py` 时指定与原用户嵌入一致的模型：
```bash
python src/rag_build_index.py --model sentence-transformers/all-MiniLM-L6-v2
```

**教训**：多个离线构建脚本的默认参数必须一致；`user_memory_meta.json` 应记录模型名以便校验。

---

## 4. 评估数据泄漏——Agent 排除列表包含测试集

**现象**：修复嵌入空间后 Agent 仍落后 44 个百分点。诊断显示 Agent 对 5 个用户的 hit 数均低于 Popular。

**根因**：用户画像（`user_profiles.jsonl`）包含全量评分数据，包括测试集。`BasicRecommender._watched_movie_ids()` 从全量画像中获取已看电影列表，在候选池中排除了测试集电影。而 `PopularBaseline` 的 `exclude_ids` 只排除训练集。Agent 永远无法推荐测试集电影，因为它认为用户"已经看过"。

**修复**：给 `BasicRecommender.recommend()`、`_candidate_pool()`、`_cold_start_recommend()` 增加可选参数 `exclude_ids`，允许调用方覆盖默认的 watched_ids。评估管道传入 `train_ratings` 作为排除列表。同时通过 `AgenticRecommender.invoke()` → `AgentState` → 各分支 → 引擎传递该参数。

**修复后**：Agent Recall@20 从落后 -44\% 变为领先 +35\%（全量 907 用户）。

**文件**：`src/basic_recommender.py`、`evaluate.py`

---

## 5. ReAct Agent 架构改造中的问题

### 5.1 对话历史缺少 assistant 消息

**现象**：QwenLLM + ReAct Agent 首次评估全零（所有用户返回空结果）。

**根因**：`_agent_node` 收到 LLM 的 tool\_calls 后直接路由到 `_tools_node`，没有将 assistant 的 tool\_call 消息加入对话历史。`_tools_node` 追加 Observation 作为 user 消息，导致 Qwen 收到连续两个 user 消息（中间缺 assistant），格式非法，返回空。

**修复**：在 `_agent_node` 路由到 tools 之前，将 assistant 消息追加到 messages：
```python
messages.append({
    "role": "assistant",
    "content": content or f"I will use {tool_name} to find movies.",
})
```

### 5.2 Qwen SDK 兼容性

**现象**：`KeyError: 'tool_calls'` — `getattr(msg, "tool_calls", None)` 在 dashscope SDK 中不生效。

**根因**：dashscope SDK 的 Message 对象覆盖了 `__getattr__` 方法，委托给 `__getitem__`。访问不存在的键时抛 `KeyError` 而非返回 default。

**修复**：改用 `try: msg["tool_calls"] except: None`。

### 5.3 tool\_calls.arguments 是 JSON 字符串

**现象**：`_execute_tool` 报错 `string indices must be integers`。

**根因**：Qwen API 返回的 `tool_calls[n].function.arguments` 是 JSON 字符串（如 `'{"user_id": 1}'`），而代码直接当 dict 使用。

**修复**：在 `_agent_node` 和 `QwenLLM.generate()` 两处增加自动检测和解析：
```python
args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
```

### 5.4 MockLLM 跨用户状态不重置

**现象**：MockLLM 评估只评估了 1/50 个用户。

**根因**：`MockLLM._call_count` 在多次 `invoke()` 之间不重置。User 1 调用 2 次后 `_call_count=2`，User 2 从 2 开始，走到 `_continue_or_finish` 而非 `_first_decision`，直接返回空结果。

**修复**：在 `invoke()` 开头调用 `llm.reset()`，后 MockLLM 被移除。

### 5.5 ReAct 格式 SYSTEM\_PROMPT 与 Function Calling 不兼容

**现象**：QwenLLM + ReAct Agent 评估全零，但无报错。

**根因**：初始 SYSTEM\_PROMPT 采用 Thought/Action/Observation 的 ReAct 文本格式，但 Qwen 的原生 Function Calling 机制期望标准 messages+tool\_calls 交互。LLM 在文本回复中输出 "Thought: ..." 而不触发 tool\_calls。

**修复**：将 SYSTEM\_PROMPT 改为标准 Function Calling 风格。

---

## 6. Qwen API 免费额度耗尽

**现象**：`qwen-plus` 返回 `status: 403`，错误信息 `AllocationQuota.FreeTierOnly`。Agent 静默处理为 `content=""` + `tool_calls=None`，全部走 finalize 无工具调用，返回空结果。

**根因**：之前的多次全量评估和调试测试消耗了 Qwen 免费额度。

**修复**：切换到 `qwen-turbo`（仍有余量），或开通付费。

**教训**：LLM 调用的错误处理不应静默吞掉，应至少记录日志或抛出可识别的异常。

---

## 7. search\_semantic 工具效果问题

### 7.1 qwen-turbo 不擅长关键词翻译

**现象**：LLM 调用了 `search_semantic`，但 description 只是原 query 的中文缩短版（"电影让人渴望旅行，充满冒险和探索的氛围"），未做关键词密集化翻译。

**修复**：重写 tool description，强制要求英文字段分隔关键词，并加入具体示例：
```
"user says '想看让人想旅行的电影' → description='travel adventure road movie 
beautiful scenery exploration wanderlust inspiring journey'"
```

### 7.2 MovieLens-100K 无剧情数据

**现象**：即使 LLM 写出了正确的英文关键词，search\_semantic 结果仍极差（Recall -93\%）。

**根因**：`movies.json` 只含 `movie_id + title + genres + release_year`，无剧情摘要。`rag_build_index.py` 的 `build_movie_chunks()` 读取 `overview` 字段但原数据中该字段永为空。embedding 文本仅约 30 个词，无法承载"wanderlust inspiring journey"级别的语义检索。

**修复**：编写 `enrich_movies.py` 从 OMDb/Wikipedia API 批量拉取电影剧情摘要，注入 `overview_en` 字段。修改 `build_movie_chunks()` 读取 `overview_en`。脚本支持增量保存（每 20 条写盘），防止中断丢失进度。修复 MovieLens 标题格式（", The" 后缀转换）。

---

## 8. 架构取舍记录

| 原方案 | 问题 | 决策 |
|--------|------|------|
| LangGraph 规则工作流 | 本质是 if-else DAG，不是 Agent | 替换为 ReAct Agent (LLM 动态决策) |
| MockLLM + QwenLLM 双模式 | 增加复杂度 | 移除 MockLLM，只保留 QwenLLM |
| LangGraph + Qwen 双 Agent 分流 | Qwen 端已有 Tool Calling，LangGraph 端多余 | 统一为 ReAct Agent，单一入口 |
| 方案 B (\texttt{search\_semantic}) | MovieLens-100K 无剧情数据 | 采集 OMDb/Wikipedia 剧情后激活 |

---

---
## 9. 从 DashScope (Qwen) 迁移到 OpenAI 兼容 API (DeepSeek)

### 9.1 动机

DashScope 免费额度耗尽，无法继续调用 Qwen 系列模型。需要切换到通过 OpenAI 兼容 API 调用 DeepSeek 模型。

### 9.2 变更范围

| 组件 | 变更 |
|------|------|
| `requirements.txt` | `dashscope==1.11.0` → `openai>=1.0.0` |
| `src/react_agent.py` | `QwenLLM` → `DeepSeekLLM`；SDK 从 dashscope 换为 openai |
| `src/api.py` | 环境变量 `DASHSCOPE_API_KEY` → `DEEPSEEK_API_KEY`；`QWEN_MODEL` → `DEEPSEEK_MODEL`；新增 `DEEPSEEK_BASE_URL` |
| `evaluate.py` | 同上环境变量 + 类名更新 |
| `evaluate_prompts.py` | dashscope SDK → openai SDK + DeepSeek |
| （遗留文件已移除） | — |

默认模型：`deepseek-chat` → `deepseek-v4-flash`

### 9.3 SDK 响应格式差异

**现象**：直接替换 SDK 后 `generate()` 取不到 content/tool_calls。

**根因**：两个 SDK 的响应对象结构完全不同：
- DashScope：`response.output.choices[0].message["content"]`（dict 访问）
- OpenAI：`response.choices[0].message.content`（属性访问）
- OpenAI streaming 中 tool_calls 是增量式（多个 chunk 逐渐拼接），而非一次性返回完整列表

**修复**：
- `generate()`：改用 `msg.content`、`msg.tool_calls`，遍历 `tc.function.name/arguments`
- `stream_generate()`：改为累积模式，逐 chunk 将 `tc_delta.function.name/arguments` 拼接到 buffer，最后统一 json.loads + 标准化
- 提取公共方法 `_normalize_stream_tool_calls()`

### 9.4 WebSocket 连接错误（HTTPException 未被捕获）

**现象**：前端点击"生成推荐"后立即显示"WebSocket 连接错误"，服务器日志无任何异常输出。

**根因**：`api.py` 的 WebSocket handler 中 `agent = _get_agent()` 在 `try/except` 块**外面**。当 `DEEPSEEK_API_KEY` 未设置时，`_get_agent()` 抛出 `HTTPException(500, "DEEPSEEK_API_KEY not set.")`，该异常在 WebSocket 已 accept 但未进入 try 块时传播到 FastAPI 层，导致 WebSocket 异常关闭，前端只收到 `ws.onerror` 无详情。

**修复**：将 `_get_agent()` 移到 `try` 块内部，单独捕获 `HTTPException` 并通过 `ws.send_json({"event": "error", ...})` 发回具体错误信息，让前端能显示"API key not set"而非通用连接错误。

### 9.5 异步流式推理卡死（同步 HTTP 阻塞事件循环）

**现象**：设置正确的 API key 后，前端卡在"正在流式推理中..."，服务器日志除了 WebSocket 连接外无任何后续输出，请求永远不返回。

**根因**：`ReActAgent.astream()` 是 `async def` 函数（由 FastAPI WebSocket 调用），但内部调用了 `DeepSeekLLM.stream_generate()`，后者使用同步 `OpenAI` 客户端做 `client.chat.completions.create(stream=True)`。同步 HTTP 请求在 `for chunk in stream` 迭代时阻塞整个 asyncio 事件循环，导致 WebSocket 无法收发任何数据。

**修复**：`DeepSeekLLM` 中同时维护两个客户端：
- `self._client = OpenAI(...)` — 供同步路径使用（`generate()`、`stream_generate()`）
- `self._async_client = AsyncOpenAI(...)` — 供异步路径使用

新增 `astream_generate()` 异步生成器，使用 `await async_client.chat.completions.create(stream=True)` + `async for chunk in stream`。`ReActAgent.astream()` 改为 `async for delta, is_final, tcs in self.llm.astream_generate(...)`。

**文件**：`src/react_agent.py`、`src/api.py`

---

*最后更新：2026-05-26*

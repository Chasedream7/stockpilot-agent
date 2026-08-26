---
title: StockPilotX Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# StockPilotX Agent

面向长线投资者的证据驱动研究助手：根据用户的研究目标，自主选择新闻检索、SEC 披露补检索、情绪量化、风险识别、memo 生成和答案自检，而不是固定执行一条流水线。

> 仅用于信息整理与研究复盘，不构成投资建议。实时数据和离线样例数据会被明确区分。

## 本次改造的重点

旧版本每次都固定执行“目标分类 → FinViz → 情绪 → 风险 → memo”，因此更像串行工作流。本版本保留了可复现的规则兜底，但新增了一个可审计的 Supervisor loop：

```text
PM Agent（embedding 语义理解目标）
          ↓
Supervisor Agent（每轮查看完整工具 manifest，再从当前满足前置条件的工具中选择）
          ├── News Scout：FinViz 新闻与市场快照
          ├── Evidence：SEC EDGAR 披露元数据与原文链接
          ├── Quant：标题情绪量化
          ├── Risk：embedding 风险类别归纳
          ├── Portfolio Copilot：证据约束下的 memo
          └── Critic：完整性、证据与“用户下一句”检查
                    ↓
          需要一手证据时自动补 SEC → 重写 memo → 再检查
```

每轮 prompt 都会列出新闻、SEC、情绪、风险、memo、Critic 的完整 manifest，以及每项的前置条件/当前可用性。首次请求时，LLM 可以在新闻和 SEC 之间自主决定先查什么；非法动作、无 API Key、embedding 服务失败或不合法 JSON 会被记录，并走可解释的关键词规则兜底。这样既支持同义表达，也避免一次模型幻觉造成失控调用。

例如，一般“风险观察”会在 Critic 通过后结束；“财报指引是否改变逻辑”可以先触发 SEC，再补新闻、重写 memo 和二次检查。

## 对话记忆与追问

网页用 `st.session_state.messages` 保存当前浏览器会话的用户/助手消息，用 `st.session_state.agent_memory` 保存上一轮的结构化证据（新闻、情绪、风险、SEC、memo、Critic）。用户追问“详细说说那个监管风险”时，Agent 会先把历史对话和上轮 memo 交给 Critic 解析指代；只有 Critic 认为监管一手来源缺失时才补 SEC。它会复用原有新闻、情绪和风险结果，不会重新抓取或重复完整流水线。

点击 **New research conversation**、刷新页面或更换 ticker 会清空这份浏览器会话记忆。

## 可观测性与任务二素材

每次运行都会在 `traces/<run_id>.jsonl` 写入 append-only trace，也可以在网页中下载。每条事件有：

- `run_id`、UTC `timestamp`、`step`、`agent`、`event`
- `input`、简短的 `decision` justification、`tool_call`、`output`
- `duration_ms` / `elapsed_ms`、LLM `input_tokens` / `output_tokens` / `total_tokens`
- `error`，以及 Critic 对用户下一句话的预测：`accept` / `ask_for_evidence` / `retry`
- 本地 embedding 的模型、输入数量与耗时；**不记录 API key 或向量值**

这使得后续可以从 trace 计算工具路径正确性、回答与证据完整性、用户满意度代理信号、时延和 token 成本；也能逐条回放低分 case。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

可选的 LLM 模式不需要在本机配置 API Key：启动页面后，在侧边栏输入 DeepSeek API Key 即可。密钥只用于当前浏览器会话，不会写入 trace、文件或环境变量。

应用通过 OpenAI Python SDK 的兼容模式调用 `https://api.deepseek.com/chat/completions`。默认文本模型为 `deepseek-v4-flash`，可在侧边栏改为 `deepseek-v4-pro`。DeepSeek 负责 Supervisor、Critic 和 memo 等文本推理。

DeepSeek 官方 API 未提供 embeddings 端点，所以语义路由不再调用其他云端模型：应用首次启动时会下载免费的本地模型 `intfloat/multilingual-e5-small`（MIT、94 种语言、384 维），之后在部署机器上计算中文目标与英文新闻标题的向量。它不需要额外 API Key 或按量付费；无法加载时才回退到关键词规则。

规则基线模式无需 API Key。`STOCKPILOT_CONTACT_EMAIL`（可选）用于标识 SEC EDGAR 请求；也可用 `STOCKPILOT_TRACE_DIR` 改变 trace 的保存目录。

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖两条可复现路径：通用风险问题的短路径，以及 Critic 触发 SEC 补证据、修订 memo 的长路径。

生成 24 条 trace-based 评估基线：

```bash
python evaluation/run_evaluation.py
```

详见 [evaluation/README.md](evaluation/README.md)。

## 工具契约

| 工具 | 输入 | 输出 | 何时调用 |
| --- | --- | --- | --- |
| `collect_news` | ticker, max_headlines | 新闻标题、链接、市场快照 | 还没有近期外部证据时 |
| `fetch_sec_filings` | US ticker | 近期 10-K / 10-Q / 8-K 等披露元数据与原文链接 | 财报、监管、公告问题或 Critic 要求一手证据时 |
| `analyze_sentiment` | 新闻标题 | 每条情绪、聚合情绪 | 有新闻后需量化叙事倾向时 |
| `assess_risk` | 情绪新闻、市场快照、用户目标 | 风险等级、语义相似度、信号与匹配度 | 用 embedding 将标题映射到可解释风险 taxonomy；无 key 时回退关键词 |
| `draft_memo` | 汇总指标、证据、SEC 链接 | 含证据与免责声明的中文 memo | 关键信号已准备好时 |
| `self_check` | 用户问题、memo、已用证据、近期消息 | 完整性/证据分、缺口、是否补检索 | 每个 memo 版本生成后；追问时优先执行以解析指代 |

## 数据源策略

当前已接入：

- **FinViz**：低成本获取近期新闻标题和快照，适合发现候选信号，不应被视作最终事实来源。
- **SEC EDGAR**：美国公司申报的权威一手来源；当前取披露元数据与原文链接，适合财报、风险因素和重大事件核验。

建议按“事实层 → 市场层 → 宏观层”逐步补齐，而不要一次接入大量同质新闻源：

| 优先级 | 数据源 | 用途 | 采用理由 |
| --- | --- | --- | --- |
| P0 | SEC EDGAR companyfacts / filings | 营收、利润、风险因素、8-K 事件 | 可追溯的一手事实，最能降低“标题党”风险 |
| P1 | Polygon / Finnhub / Alpha Vantage（三选一） | 历史 OHLCV、公司新闻、财报日历 | 用于计算收益、波动、事件窗口；选择一个稳定供应商即可 |
| P1 | FRED | 利率、CPI、失业、信用利差等宏观变量 | Macro Agent 的权威、结构化来源 |
| P2 | 公司 IR 页面与 earnings call transcript | 指引、管理层表述、电话会问答 | 对“叙事是否改变”有价值，需保留 URL 与发布时间 |
| P2 | 财经新闻 API（如 NewsAPI / Finnhub） | 多来源新闻去重、正文与发布时间 | 只作发现层，重要结论需回到 P0/P2 核验 |

数据表必须带 `source`、`url`、`published_at`、`retrieved_at` 和 `is_primary_source`。评估时把“主来源覆盖率”作为单独指标，不能让流畅的 memo 掩盖证据缺失。

## 为什么当前不用 LangGraph / FastAPI

题目鼓励手写 loop，且本次目标是先证明“能根据中间结果改变路径、能完整追踪、能评估”。因此当前保留一个小而可读的 Python loop，便于录屏时解释每个状态转移；这不是宣称已经完成 8 个独立的专业 Agent。

当基线评估跑通后，建议将 `run_workflow` 的 state / tools 抽到服务层：FastAPI 提供 `POST /runs`、`GET /runs/{run_id}` 和 trace 下载；再用 LangGraph 显式表达状态、条件边和最大轮数。迁移前后可以用同一批 traces 比较路径正确率、P95 时延和 token 成本，避免“为了上框架而上框架”。

---
title: StockPilot Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# StockPilot Agent

## 一、选题动机

### 1.1 解决什么问题？

长线投资者每天面对海量信息（新闻、财报、监管公告、市场快照），需要在极短时间内判断：**哪些信息是噪音？哪些信号可能影响持仓逻辑？是否需要进一步查阅一手来源（如 SEC 披露）？**


### 1.2 这件事现在有多痛？

| 痛点具体表现 | 说明 | 频次 / 成本 |
| --- | --- | --- |
| **信息分散** | 新闻、财报、监管文件分散在 3–5 个不同平台 | 每次复盘 |
| **路径依赖** | 财报、监管、催化问题应该走不同检索路径，但人工容易“一刀切” | 高频 |
| **证据核查成本高** | 看到一条负面标题，不确定是市场情绪还是基本面恶化，需要额外查原始文件 | 中高频 |
| **复盘无结构化记录** | 看完就忘，下次遇到同类问题从头再来 | 长期 |

### 1.3 为什么是 Agent 而不是脚本？

| 对比维度 | 传统脚本（固定流水线） | Agent（自主决策） |
| --- | --- | --- |
| **路径可变性** | 每次都跑：新闻 → 情绪 → 风险 → memo | 根据问题类型动态选择起点和终点 |
| **上下文感知** | 无法识别“反垄断”和“诉讼”是同类语义 | 通过 embedding 语义路由自动归入“监管/法律”类 |
| **深度控制** | 无法决定“何时需要查一手来源” | Critic 自检后自动触发 SEC 补检索 |
| **多轮对话** | 跑完即止，无法追问 | 保留记忆，支持“上次那个风险点具体是哪条新闻？” |
| **可迭代性** | 改逻辑 = 改代码 | 改 Prompt / 改工具签名即可调整行为 |

> 股票研究本质上是一个“假设 → 检索 → 验证 → 修正假设”的螺旋过程，而非线性流水线。Agent 是匹配这种认知模式的技术载体。

## 二、Agent 拆解

### 2.1 整体架构图（文字版）

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                          用户输入（自然语言）                             │
│                     “TSLA 近期反垄断诉讼风险”                              │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    【PM Agent】—— 语义路由层（Embedding）                 │
│  将用户目标与 Goal Profile 做余弦相似度匹配，输出“分析重点”                │
│  例：相似度 0.72 → 命中 “regulation_review”                               │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    【Supervisor Agent】—— 规划层（LLM）                   │
│  每轮根据当前 State（新闻/情绪/风险/证据状态）和可用工具选择下一步动作     │
│  决策输出：{"action": "fetch_sec_filings", "reason": "..."}               │
│  约束：财报/监管类问题优先 SEC；draft_memo 必须等 risk 就绪                │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       【工具执行层】（Tool Executor）                     │
│  collect_news · fetch_sec_filings · analyze_sentiment · assess_risk         │
│  draft_memo · self_check                                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    【Critic Agent】—— 反思层（LLM）                        │
│  检查 memo：是否回答问题？证据够不够？是否存在矛盾？                        │
│  输出：status + should_retrieve + predicted_user_followup                  │
│  若需补证据 → fetch_sec_filings → 重写 memo → 再次检查                      │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    终止条件命中 → 输出最终 Memo（证据链接 + 免责声明）
```

### 2.2 LLM 决策点（共 4 处）

| 决策点 | 负责 Agent | 输入 | 输出 | 失败兜底 |
| --- | --- | --- | --- | --- |
| ① 目标语义匹配 | PM Agent | 用户 mission + Goal Profile 描述 | 最匹配的 Profile | 关键词规则匹配 |
| ② 下一步动作选择 | Supervisor Agent | 当前 State + 完整工具列表 | `{action, reason}` | `rule_next_action()` |
| ③ Memo 生成 | Portfolio Copilot | 汇总指标 + 证据 + SEC 链接 | 结构化 Markdown Memo | 规则模板 Memo |
| ④ 质量自检与补证据 | Critic Agent | Memo + 已用证据 + 对话上下文 | `status + should_retrieve` | 规则检查 |

### 2.3 上下文 / 记忆

**单次运行记忆（State）**

一个 Python dict 贯穿整个 Agent Loop，累积存储：`news_df`、`scored_news`、`summary`、`risk`、`sec_filings`、`memo`、`memo_version`、`critic` 等。每次工具调用的输入输出都会更新 State，确保后续决策能感知历史。

**多轮对话记忆（Session）**

通过 `st.session_state.messages` 保存用户 / 助手消息，通过 `st.session_state.agent_memory` 保存上一轮的结构化证据。用户追问“详细说说那个监管风险”时，Agent 先执行 `self_check` 识别指代，再基于上一轮 State 增量更新，而不是从头跑全流程。

### 2.4 终止条件（2 种）

| 条件 | 触发逻辑 | 设计意图 |
| --- | --- | --- |
| **正常终止** | `decide_next_action` 返回 `finish`，且 memo 已生成、Critic 已检查当前版本、没有待补的一手证据 | 信息充足且经过验证后结束 |
| **硬终止（安全网）** | `tool_steps >= MAX_AGENT_STEPS`（当前为 10），或关键工具持续失败 | 防止死循环；保留并解释证据缺口 |

## 三、工具清单（核心 6 个）

| name | input schema | output | 什么时候调用 |
| --- | --- | --- | --- |
| `collect_news` | `{ "ticker": "string", "max_headlines": "integer" }` | 新闻标题、URL、市场快照、来源说明 | 没有近期新闻证据，或用户明确要求新闻 / 催化 / 情绪时 |
| `fetch_sec_filings` | `{ "ticker": "US ticker", "max_filings": "integer" }`（默认 8） | 公司名、CIK、10-K / 10-Q / 8-K 元数据和原文链接 | 用户问财报、指引、监管、诉讼、公告，或 Critic 要求一手来源时 |
| `analyze_sentiment` | `{ "headlines": "array<string>" }` | 单条情绪、平均分、正 / 中 / 负面比例 | 新闻已抓取且需要量化市场叙事倾向时 |
| `assess_risk` | `{ "scored_news": "object", "snapshot": "object", "goal_profile": "object" }` | 风险等级、风险分、风险类别、语义相似度和匹配信号 | 新闻和情绪就绪后，将标题映射到风险 taxonomy 时 |
| `draft_memo` | `{ "evidence": "object", "goal": "object", "ticker": "string" }` | 中文结构化研究 memo、证据区和免责声明 | 关键证据已准备好，或补证据后需要重写时 |
| `self_check` | `{ "mission": "string", "memo": "string", "sec_filings": "object", "source_note": "string" }` | 完整性分、证据分、缺口、是否补检索、下一步建议 | 每个 memo 版本生成后；多轮追问时作为第一步 |

## 四、设计决策记录

| 决策点 | 我的选择 | 为什么不选另一种方案 |
| --- | --- | --- |
| **框架** | 手写 Agent Loop（while + tool dispatch） | 便于精确埋点，依赖更少，状态转移和终止条件完全可控 |
| **规划层** | LLM 选动作 + 规则兜底 | 纯 LLM 可能输出非法动作，纯规则又不够灵活 |
| **语义路由** | 免费本地 `intfloat/multilingual-e5-small` | 关键词难处理“反垄断 → 监管/法律”这类同义映射；本地模型不需要额外 embedding API |
| **记忆** | 单次 State + 多轮 Session | 分层设计避免历史对话污染当前路径规划 |
| **终止条件** | LLM 自主判断 + 10 步硬上限 | 兼顾智能性与防死循环能力 |
| **Trace 格式** | JSONL（每行一个 JSON） | 可边跑边写，便于按行过滤、抽样和回放 |

## 五、运行与数据源

```bash
pip install -r requirements.txt
streamlit run app.py
```

页面侧边栏输入 DeepSeek API Key 即可启用 LLM 模式，密钥只在当前会话使用，不写入文件或 trace。当前接入：

- **FinViz**：低成本新闻发现层；
- **SEC EDGAR**：财报、风险因素和重大事件的一手来源。

后续可增加历史行情 / 财报日历、FRED 宏观变量和公司 IR / earnings call transcript。重要数据保留 `source`、`url`、`published_at`、`retrieved_at` 和 `is_primary_source`。

## 六、评估入口

评估设计、28 条 LLM 样本、评分表和分层分析见 [`evaluation/README.md`](evaluation/README.md)。

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

## 一、选题动机

### 1.1 解决什么问题？

长线投资者每天要在新闻、财报、监管公告和市场数据之间做判断：哪些是噪音，哪些信号可能改变持仓逻辑，以及什么时候必须回到 SEC 等一手来源核验。

传统流程是“打开 FinViz → 查 SEC EDGAR → 手动判断情绪 → 写复盘 memo”。它重复、耗时，而且容易漏掉隐含路径：例如新闻提到“反垄断调查”，真正需要的下一步可能是补查 8-K，而不是继续浏览更多标题。

### 1.2 这件事有多痛？

| 痛点 | 具体表现 | 频次 / 成本 |
| --- | --- | --- |
| 信息分散 | 新闻、财报和监管文件分布在多个平台 | 每次复盘 |
| 路径依赖 | 财报、监管、催化问题需要不同检索路径，人工容易一刀切 | 高频 |
| 证据核查昂贵 | 一条负面标题到底是情绪还是基本面变化，常需额外查原始文件 | 10–20 分钟 / 次 |
| 复盘不可复用 | 缺少结构化记录，下次遇到同类问题又从头开始 | 长期 |

### 1.3 为什么是 Agent，而不是脚本？

| 维度 | 固定脚本 | StockPilotX Agent |
| --- | --- | --- |
| 路径 | 每次执行同一条流水线 | 根据 mission 和中间结果动态选工具 |
| 语义 | 依赖关键词，容易漏掉“反垄断”等同义表达 | 本地 embedding 将语义映射到目标 / 风险 profile |
| 证据 | 不知道什么时候该查一手来源 | Critic 发现缺口后触发 SEC 补检索 |
| 多轮交互 | 运行结束即丢失上下文 | 通过 `st.session_state` 复用上一轮证据和 memo |
| 迭代 | 改逻辑需要改代码 | 可单独调整 prompt、工具契约和终止护栏 |

股票研究更接近“假设 → 检索 → 验证 → 修正”的循环，而不是线性流水线，因此需要能根据状态改变路径的 Agent。

## 二、Agent 拆解

### 2.1 整体架构

```text
用户问题 + ticker
        │
        ▼
PM / Goal Router（本地 embedding）
  识别研究目标与优先风险类别
        │
        ▼
Supervisor Loop（LLM）
  查看完整工具 manifest、当前 State、历史 Memory
  选择下一步 action
        │
  ┌─────┼────────┬──────────┬──────────┐
  ▼     ▼        ▼          ▼          ▼
新闻   SEC      情绪       风险       memo
                                  │
                                  ▼
                         Critic / self_check（LLM）
                    通过 → finish
                    有缺口 → 补工具 → 重写 memo → 再检查
```

### 2.2 LLM 决策点

| 决策点 | 负责模块 | 输入 | 输出 | 失败兜底 |
| --- | --- | --- | --- | --- |
| 目标语义匹配 | PM / Goal Router | mission + goal profiles | 目标 profile、优先类别 | 关键词匹配 |
| 下一步动作 | Supervisor | State + 完整工具 manifest | `{action, reason}` | `rule_next_action()` |
| Memo 生成 | Portfolio Copilot | 指标、新闻、风险、SEC 证据 | 结构化 Markdown memo | 规则模板 |
| 质量与补证据 | Critic | memo + 证据 + 对话上下文 | 完整性 / 证据分、缺口、建议动作 | 规则检查 |

LLM 负责灵活性，规则负责安全边界：非法 action、工具失败、无 API Key 或模型输出不合法时，系统会记录 trace 并进入可解释的降级路径。

### 2.3 上下文与记忆

- **单次运行 State**：一个 Python dict 贯穿 Agent Loop，累积 `news_df`、`scored_news`、`summary`、`risk`、`sec_filings`、`memo`、`memo_version` 和 `critic`。
- **多轮 Session**：`st.session_state.messages` 保存用户 / 助手消息；`st.session_state.agent_memory` 保存上一轮结构化证据。追问“详细说说那个监管风险”时，先调用 `self_check` 解析指代，只补当前缺口，不重新抓取整条流水线。

### 2.4 终止条件

1. **正常终止**：`decide_next_action` 返回 `finish`，且 memo 已生成、Critic 已检查当前版本，并且没有待补的一手证据。
2. **安全终止**：`tool_steps >= MAX_AGENT_STEPS`（当前为 10），或关键工具持续失败。此时输出明确的证据缺口，不把不完整结果包装成确定结论。

## 三、工具清单

以下工具覆盖“检索 → 量化 → 生成 → 反思”链路。`State` 中的数据由 Supervisor 按前置条件传入。

### Tool 1：`collect_news`

| 项目 | 定义 |
| --- | --- |
| Agent | News Scout |
| Input schema | `{ "ticker": "string", "max_headlines": "integer" }` |
| Output | 新闻标题、URL、市场快照、来源说明 |
| 调用时机 | 没有近期新闻证据，或用户明确要求新闻 / 催化 / 情绪时。若目标明确指向财报、监管或诉讼，Supervisor 可先选 SEC。 |

### Tool 2：`fetch_sec_filings`

| 项目 | 定义 |
| --- | --- |
| Agent | Evidence |
| Input schema | `{ "ticker": "US ticker", "max_filings": "integer" }`（默认 8） |
| Output | 公司名、CIK、10-K / 10-Q / 8-K 元数据和原文链接 |
| 调用时机 | 用户问财报、业绩、指引、监管、诉讼、公告，或 Critic 输出 `should_retrieve = true` 时。失败会写入 State，保留证据缺口。 |

### Tool 3：`assess_risk`

| 项目 | 定义 |
| --- | --- |
| Agent | Risk |
| Input schema | `{ "scored_news": "object", "snapshot": "object", "goal_profile": "object" }` |
| Output | `level`、0–100 风险分、风险类别、匹配标题和语义相似度 |
| 调用时机 | 新闻已抓取且情绪已量化后。使用本地 embedding 将标题映射到监管 / 业绩 / 竞争 / 宏观等风险类别。 |

### Tool 4：`self_check`

| 项目 | 定义 |
| --- | --- |
| Agent | Critic |
| Input schema | `{ "mission": "string", "memo": "string", "sec_filings": "object", "source_note": "string" }` |
| Output | `status`、`answer_score`、`evidence_score`、`should_retrieve`、缺口、`predicted_user_followup` |
| 调用时机 | 每个 memo 版本生成后立即调用；多轮追问时作为第一步。若缺少一手证据，Supervisor 下一轮补 SEC，再重写 memo 并复检。 |

其他内部工具包括 `analyze_goal`、`analyze_sentiment` 和 `draft_memo`，其契约与状态转移见 `app.py` 和 `evaluation/评估设计.md`。

## 四、设计取舍

- **Loop 而非重框架**：当前使用可读的 Python Agent Loop，便于精确记录每次状态转移；后续可迁移到 LangGraph / FastAPI。
- **LLM + 规则双保险**：纯规则不够灵活，纯 LLM 又可能产生非法动作；两者结合才能兼顾泛化和可控性。
- **本地 embedding**：使用免费的 `intfloat/multilingual-e5-small`，不依赖额外 embedding API；模型不可用时回退到关键词路由。
- **JSONL trace**：每行一个事件，记录 action、工具输入输出摘要、耗时和 token 用量，不记录 API Key 或向量值。

## 五、运行与评估

```bash
pip install -r requirements.txt
streamlit run app.py
```

页面侧边栏输入 DeepSeek API Key 即可启用 LLM 模式，密钥只在当前会话使用。当前接入 FinViz 新闻发现层和 SEC EDGAR 一手披露；后续可增加历史行情 / 财报日历、FRED 宏观变量和公司 IR / earnings call transcript。

评估设计、28 条 LLM 样本、评分表和分层分析见 [`evaluation/README.md`](evaluation/README.md)。

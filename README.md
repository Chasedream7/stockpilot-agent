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

## 1. 选题动机

长线投资者真正缺的不是一条新闻，而是把“新闻、公司披露、市场情绪和个人持仓目标”放在同一条可追溯证据链里。传统脚本通常固定执行“抓新闻 → 算情绪 → 生成摘要”，遇到模糊问题、监管事件或证据缺口时仍会照跑，容易把不完整的信息包装成确定结论。

StockPilotX 面向“这次信息是否改变我的投资假设”这类研究任务。它选择 Agent 而不是单纯脚本，是因为下一步取决于当前中间结果：用户问财报/监管时应优先找 SEC；已有研究的追问应先读取记忆并自检；证据不足时应补检索或明确指出缺口。系统只做信息整理和研究复盘，不构成投资建议。

## 2. Agent 拆解

```text
用户问题 + ticker
        │
        ▼
PM / Goal Router
  识别研究目标、优先风险类别
        │
        ▼
Supervisor Loop
  每轮查看完整工具清单、当前 state 和历史 memory
  由 LLM 选择下一步，而不是固定顺序
        │
  ┌─────┼────────┬──────────┬──────────┐
  ▼     ▼        ▼          ▼          ▼
新闻   SEC      情绪       风险       memo
                                  │
                                  ▼
                              Critic / self_check
                         证据足够 → finish
                         有缺口   → 补工具 → 重写 memo
```

- **LLM 决策点**：首次研究时从完整 manifest 中选择第一个工具；之后根据已完成工具、工具输出和 Critic 缺口选择下一步。
- **工具集**：新闻发现、SEC 一手披露、情绪量化、风险语义分类、memo 生成、自检。
- **上下文 / 记忆**：`st.session_state.messages` 保存对话；`st.session_state.agent_memory` 保存新闻、情绪、风险、SEC、memo 和 Critic 结果。追问“详细说说那个监管风险”时先执行 `self_check`，只补缺失证据，不重跑整条流程。
- **终止条件**：Critic 判定答案和证据达到阈值且无待补检索时 `finish`；达到最大轮数、工具失败或证据仍不足时，输出可解释的缺口，不伪装成完整结论。
- **可观测性**：每一步写入 JSONL trace，记录决策、工具调用、输出摘要、耗时和 token 用量；不记录 API key 或向量值。

## 3. 工具清单

| name | input schema | output | 什么时候调用 |
| --- | --- | --- | --- |
| `collect_news` | `{ "ticker": "string", "max_headlines": "integer" }` | 新闻标题、URL、市场快照、来源时间 | 没有近期市场证据，或用户明确要求新闻/催化/情绪时 |
| `fetch_sec_filings` | `{ "ticker": "US ticker" }` | 10-K、10-Q、8-K 元数据和原文链接 | 用户问财报、指引、监管、公告，或 Critic 要求一手来源时 |
| `analyze_sentiment` | `{ "headlines": "array<string>" }` | 单条情绪、平均分、正/中/负面比例 | 已有新闻，且需要量化市场叙事倾向时 |
| `assess_risk` | `{ "news": "object", "market_snapshot": "object", "goal": "object" }` | 风险等级、风险类别、语义相似度、匹配信号 | 新闻和情绪就绪后，需要把标题映射到风险 taxonomy 时 |
| `draft_memo` | `{ "evidence": "object", "goal": "object", "ticker": "string" }` | 中文结构化研究 memo、证据区和免责声明 | 关键证据已经准备好，或补证据后需要重写时 |
| `self_check` | `{ "question": "string", "memo": "string", "evidence": "object", "messages": "array" }` | 完整性分、证据分、缺口、是否补检索、下一步建议 | 每版 memo 生成后；多轮追问时作为第一步 |

## 4. 运行与数据源

```bash
pip install -r requirements.txt
streamlit run app.py
```

页面侧边栏输入 DeepSeek API Key 即可启用 LLM 模式，密钥只在当前会话使用。文本推理通过 DeepSeek 的 OpenAI 兼容接口完成；语义路由使用免费的本地 `intfloat/multilingual-e5-small`，无法加载时回退到关键词分类。

当前数据源：

- **FinViz**：低成本新闻发现层；
- **SEC EDGAR**：财报、风险因素和重大事件的一手来源。

后续可增加历史行情/财报日历（Polygon、Finnhub 或 Alpha Vantage）、宏观变量（FRED）和公司 IR / earnings call transcript。重要结论应保留 `source`、`url`、`published_at`、`retrieved_at` 和 `is_primary_source`。

## 5. 评估入口

评估设计、28 条 LLM 样本和分层分析见 [`evaluation/README.md`](evaluation/README.md)。评估重点是：路径是否正确、产物是否回答问题并带必要证据、Critic 对用户下一句话的预测是否可靠。

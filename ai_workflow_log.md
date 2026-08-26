# AI Workflow Log

本文记录 StockPilot Agent 从需求澄清到评估验收的协作过程。

## 1. 任务目标与拆解

总目标：把一个固定的股票信息流水线，改造成能够自主选择工具、识别证据缺口、支持多轮追问并且可评估的研究 Agent。

我把任务拆成六个相互独立、可以逐项验收的子任务：

1. **理解用户目标**：把自然语言 mission 映射到 goal profile 和优先风险类别。
2. **动态规划**：给 Supervisor 完整工具 manifest，让 LLM 决定第一步和后续动作。
3. **工具执行**：统一工具输入、输出和前置条件，避免模型调用不存在或不适用的工具。
4. **记忆与追问**：保存 `messages` 和结构化研究 memory；追问时先检查历史证据，只补当前缺口。
5. **可观测性**：每个 loop step 写 JSONL trace，记录决策、工具调用、耗时、token 和错误。
6. **评估与验收**：用 28 个 case 分别评估路径正确性、产物 / 证据质量和用户下一句预测。

拆解原则是：每个子任务都必须有一个可观察信号。例如“支持记忆”不是看代码里有没有 `session_state`，而是检查追问 trace 是否以 `self_check` 开始、是否复用了上轮证据、是否避免重新分析已有新闻。

## 2. 关键 Prompt 设计

### 2.1 Supervisor Prompt

Supervisor 每轮都收到完整的工具列表，而不是只看到“当前缺失的下一步”。Prompt 的核心约束如下：

```text
你是研究工作流 Supervisor。
根据用户 mission、当前 State、历史 Memory 和完整工具 manifest，
选择一个且仅一个下一步 action。

可选 action：
collect_news / fetch_sec_filings / analyze_sentiment /
assess_risk / draft_memo / self_check / finish

要求：
1. 首次研究必须从当前最有信息价值的工具开始；
2. 财报、指引、监管、诉讼问题优先考虑 SEC 一手来源；
3. draft_memo 只能在关键证据准备好后调用；
4. 追问必须先 self_check，不得无理由重跑已有新闻和情绪分析；
5. 只输出 JSON：{"action": "...", "reason": "..."}。
```

`reason` 被写入 trace，用来解释为什么选择这一步，而不是只保留一个动作名。

### 2.2 Critic / self_check Prompt

```text
检查这份 memo 是否真正回答用户问题。
分别判断：
- answer_score：是否回答了用户要求；
- evidence_score：关键结论是否有可追溯依据；
- contradictions：是否存在内部矛盾；
- should_retrieve：是否需要补充检索；
- predicted_user_followup：accept / ask_for_evidence / retry。

如果用户问题要求财报、指引或监管依据，而当前没有成功取得 SEC 证据，
必须标记 needs_more_evidence，并建议补 SEC，而不是直接判定通过。
```

这个 Prompt 的目的，是把“写得像答案”与“证据真的够”分开，避免流畅的文字掩盖来源缺失。

### 2.3 Memo Prompt

Memo 生成要求模型只使用 State 中已有的证据，并固定输出用户目标、观察、关键读数、正负证据、风险审视、后续步骤和免责声明。这样便于人工阅读，也便于评估脚本检查结构完整性。

## 3. AI 协作方式

我把 AI 当作三个不同角色使用：

- **架构讨论者**：帮助比较固定流水线、规则路由和 LLM Supervisor 的边界，先确定状态机和工具契约。
- **实现助手**：按子任务修改 loop、session memory、trace 和评估脚本；每次只处理一个可验收改动。
- **评审者**：模拟 Critic 和阅卷人，专门寻找证据缺口、非法 action、错误终止和无法解释的分数。

每轮协作都要求 AI 先说明“改动影响哪些状态字段和验收信号”，再修改代码。这样可以避免为了让 demo 看起来更智能，随意增加没有输入输出契约的 Agent。

## 4. 踩过的坑与修复

### 坑 1：把“调用过 SEC”误当成“拿到 SEC 证据”

早期逻辑只检查 `sec_attempted`。当 SEC 请求失败时，Critic 仍可能把 memo 判为可接受。

**修复**：改为检查 `has_successful_primary_source`，即 `sec_filings.filings` 是否非空；失败时保留 `needs_more_evidence` 和 `ask_for_evidence`，并在评估中单独统计失败识别率。

### 坑 2：Supervisor 只能在固定缺口里选下一步

如果 Prompt 只告诉模型“现在缺少情绪分析”，它本质上还是规则流水线。

**修复**：每轮传入完整 manifest、工具前置条件和当前 State，让 Supervisor 在新闻、SEC、自检等工具之间自主选择；规则只负责非法动作和安全边界兜底。

### 坑 3：追问重新跑完整流程

用户第二次问“详细说说那个监管风险”时，如果只传新问题，Agent 无法解析“那个”指什么。

**修复**：保存 `st.session_state.messages` 与 `st.session_state.agent_memory`。追问路径固定先 `self_check`，再根据缺口决定是否补 SEC；已有新闻、情绪和风险结果不重复计算。

### 坑 4：LLM 输出不是合法 JSON

模型偶尔会在 JSON 字符串中混入引号、解释文字或不存在的 action。

**修复**：记录原始输出和解析错误；拒绝非法 action；使用 `rule_next_action()` 生成可解释的安全动作，并把 fallback 原因写入 trace。

### 坑 5：评估只有一个总分

单列平均分无法区分“路走错了”和“路走对但证据不足”，也无法解释用户为什么可能不满意。

**修复**：拆成 D1 路径正确性、D2 产物 / 证据质量、D3 下一句预测校准；保留每条 case 的 `score_reason`，并设置证据 guardrail，避免高分维度掩盖证据缺失。

### 坑 6：长批次评估中断后丢失进度

评估进程在最终聚合前中断，导致前面已经完成的 trace 没有出现在结果表里。

**修复**：增加 `--start-index` 续跑能力，并从已完成 trace 恢复 23 条记录，再与 5 条续跑结果合并成 28 条总结果。后续应将 checkpoint 改为每条 case 完成即追加写盘。

## 5. 验收标准

### 功能验收

- 首次研究时，Supervisor 能从完整工具列表选择第一步。
- 财报 / 指引 / 监管问题能触发 SEC；SEC 失败时不会伪装成证据充分。
- 每版 memo 都经过 `self_check`。
- 追问首步为 `self_check`，并复用上一轮 memory。
- 达到 `finish` 或 10 步安全上限时，loop 必须可解释地结束。

### 可观测性验收

每个 run 都有 JSONL trace，并至少包含：

- `run_id`、时间戳、step、agent、event；
- action、reason、tool input / output 摘要；
- duration、elapsed 和 token usage；
- 错误、fallback 和终止原因。

API Key、向量值和其他敏感凭据不得写入 trace。

### 评估验收

评估集包含 24 条首次研究和 4 条多轮追问，覆盖：

- SEC 一手来源成功与失败；
- 模糊 / 极短输入；
- Supervisor 首工具选择；
- Critic 下一句预测；
- 研究记忆和最小补检索。

每条结果必须包含样本 ID、输入、D1 / D2 / D3 分数、总分和文字理由。报告还要能按场景、工具调用次数、延迟和 token 量分层，并对低分样本做 trace case study。

## 6. 最终验收命令

```bash
python -m unittest discover -s tests -v
python evaluation/run_evaluation.py --use-llm --model deepseek-v4-flash
```

验收不只看命令是否退出成功，还要检查：

1. 测试全部通过；
2. 结果 CSV 行数与 case 数一致；
3. 每行存在可解释的 `score_reason`；
4. summary 中包含 Q1–Q5；
5. 低分 case 能通过 `trace_path` 回放；
6. 结果目录不包含 API Key 或未授权上传的原始 trace。

## 7. 当前结论与下一步

当前版本已经证明 Agent 可以动态选工具、进行 Critic 补证据、保留多轮研究记忆，并且具备 trace-based evaluation。下一步最有价值的改进不是继续堆 Agent 数量，而是把每条 case 的 checkpoint 持久化、增加独立 judge / 人工盲审，并用同一批 case 做“改动前 → 改动后”的可重复对比。

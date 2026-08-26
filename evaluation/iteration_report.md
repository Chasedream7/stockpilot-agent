# 改后复评：一手来源成功性护栏（v1 历史记录）

> 此文记录 v1 的 24 条单轮基线及当时使用的满意度代理。当前评估已升级为 v2：使用人工标注的用户下一句做 Critic 校准，并新增多轮记忆 case。请以 [评估设计.md](评估设计.md) 和最新 evaluation_results.csv 为准。

## 发现与假设

第一轮 24 条 rule-fixture 基线中，SEC 补检索失败的 `E06`、`L04`、`A04` 仍被 Critic 预测为 `accept`。原因是 Critic 使用了 `sec_attempted`（是否调用过工具）代替 `has_successful_primary_source`（是否真正拿到一手证据）。这会把“工具成功调用”误当成“用户问题已被解决”。

另一个低分 case `L02`（“反垄断调查”）没有触发 SEC，因为意图词表覆盖了“监管/诉讼”，但遗漏“反垄断/调查”。

## 实施

1. 把一手来源判断改为 `bool(sec_filings["filings"])`，而不是工具调用次数。
2. 为业绩/监管护栏增加 `反垄断`、`调查`、`antitrust`、`probe`、`investigation` 同义表达。
3. SEC 失败后不再反复重写同一份 memo；保留 `needs_more_evidence` 和 `ask_for_evidence`，再以可解释的 `finish` 结束，避免触发最大轮数安全停止。
4. 新增单元测试，验证 SEC 失败时不会把最终 Critic 标为 `pass`。

## 同一批 24 个 fixture case 的复评

| 指标 | 改前 | 改后 | 解释 |
| --- | ---: | ---: | --- |
| 平均总分 | 87.33 | 88.97 | `L02` 走对一手来源路径，抵消了更严格满意度判断 |
| 平均路径分 | 98.54 | 100.00 | `L02` 从漏检 SEC 变为补检索 + 修订 |
| 平均产物质量分 | 83.67 | 84.62 | `L02` 成功补入一手来源 |
| 平均满意度代理 | 100.00 | 93.75 | 并非体验退化，而是把 3 条“失败但被误判接受”的 case 正确标成 `ask_for_evidence` |
| 低分样本数（<60） | 5 | 4 | `L02` 从 59 升至 98.4；SEC 失败与低上下文样本仍应低分 |

详细原始表：

- 改前：`evaluation/results/before_primary_source_fix/evaluation_results.csv`
- 改后：`evaluation/results/evaluation_results.csv`

## 三条 case study

### L02：反直觉的“同义词漏召回”

输入为“反垄断调查对原有 thesis 的影响要怎么复盘？”。改前路径在 Critic 后直接结束，缺失 SEC；改后增加 `fetch_sec_filings → draft_memo → self_check`，总分从 59 到 98.4。它说明最简单的关键词意图路由也需要用边界表达压测，不能只用“监管”这个正向词做 happy path。

### E06：工具失败不是问题解决

输入是“财报和指引有没有改变持仓判断？”，fixture 中 SEC 故意失败。改前虽然一手证据为空，Critic 仍预测用户接受；改后最终为 `needs_more_evidence / ask_for_evidence`，并保留失败原因。总分仍被证据 guardrail 封顶为 59，这是预期结果。

### A03：路径全对但答案仍不可信

输入是“有风险吗，快一点”，只有 3 条离线样例新闻。工具路径和 memo 结构均通过，但 `evidence_score=45`，总分被封顶为 59。这个 case 不应靠“再加一个 Agent”解决；更合理的下一步是追问时间范围、持仓期限或容忍回撤，并允许 Agent 在证据不足时拒绝给出强结论。

## 尚未解决的局限

这些结果是 rule-fixture 基线，不是对线上模型质量的宣称。下一次应固定模型、prompt 和真实数据时间窗，用 `--use-llm` 运行同一 case，再抽取至少 20 条真实 trace。离线 judge 也应与 in-loop Critic 分离，并对低分、边界和高分样本各做人工盲审。

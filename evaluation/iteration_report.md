# LLM 评估报告与迭代闭环

## 执行状态

本批使用 DeepSeek `deepseek-v4-flash`，共 28 个 case。23 条首次研究在第一批完成；A04 与 M01–M04 在续跑批次完成。续跑结果和可审计进度保存在 `evaluation/results/llm_20260826_resume/`。trace 中没有 API key，原始 trace 不提交到仓库。


## 已观察到的信号

续跑批次的 5 条结果显示：Q1 的 SEC 发起率和失败识别率均为 100%；Q5 的首步 `self_check`、证据复用、SEC 补取对齐均为 100%。这支持“多轮状态机没有重跑整条流水线”的工程判断，但样本量仍不足以证明线上稳定性。

第一批 23 条真实 trace 共消耗约 237,466 token，平均每条约 10,325 token；这说明复杂度主要来自 Supervisor、memo 和 Critic 多次调用，而不是工具抓取本身。该 token 结果只用于成本分析，不应与没有 LLM 的规则基线比较。

## Case study

### A04：模糊监管问题 + SEC 失败

输入只有“公司监管风险怎么样？”，但问题又隐含一手来源需求。该 case 用来检查 Agent 是否先处理不确定性，再在 SEC 失败时输出证据缺口。低分并不一定表示路由完全错误，可能是“路径正确但证据不可用”；因此报告必须同时看 D1 和 D2，而不是只看总分。

### M01：追问指代解析

seed 研究完成后，用户追问“详细说说那个监管风险”。合格路径应为 `self_check → fetch_sec_filings → draft_memo → self_check`，并复用 seed 的新闻、情绪和风险结果。该 trace 用来验证“历史上下文解决指代、Critic 决定是否补证据”。

### M04：追问时一手来源仍失败

该 case 将 SEC 设为失败，预期是 `needs_more_evidence / ask_for_evidence`，而不是生成一份语气确定的完整 memo。它是评估过度自信和来源失败护栏的边界样本。

## 迭代建议

1. **降低低上下文误判为 accept 的概率**：第一版离线 24 case 中，模糊/低上下文组有 4 条，Critic 曾将其全部倾向 `accept`。改动：在 Critic prompt 增加“问题信息不足时优先 ask_for_evidence/retry”的明确判定表，并将 D3 的 `retry` 少数类召回单独设为门槛。预期：低上下文 `accept` 误判率从 100% 降到 25% 以下。
2. **减少每 case 的 LLM 成本**：第一批 23 条真实 trace 共约 237,466 token，均值约 10,325 token/case。改动：Supervisor 仅在工具状态变化时重规划；memo 修订时只传递证据差异和 Critic 缺口，不重复发送完整新闻标题。预期：平均 token 降到 7,500 以下，路径分保持不下降。
3. **把全量聚合做成可恢复任务**：本次中断发生在最终写 CSV 前，导致 23 条已完成 trace 没有全量分数表。改动：每完成一条 case 就追加 checkpoint 行，结束时从 checkpoint 重建 summary；保留 `run_manifest.json` 标记模型、prompt、样本和完成状态。预期：中断恢复率达到 100%，不再丢失已完成 case 的评分。
4. **独立 Critic 与人工盲审**：当前 D3 仍是 in-loop Critic 对人工代理标签，不能证明真实满意度。改动：增加独立 judge prompt，并对高/中/低分各抽 5 条盲审。预期：D3 与人工标签的 Cohen κ 达到 0.6 以上后，才将其用于线上回归门槛。

## 闭环状态

本轮已真正落地第 3 条的一半：新增 `--start-index`，并保存了 28 条续跑进度；下一步应把 checkpoint 追加写盘和全量聚合补齐，再用同一批 case 做前后对比。当前报告明确区分“已观测数据”和“待完成的全量汇总”，避免把不完整结果包装成结论。

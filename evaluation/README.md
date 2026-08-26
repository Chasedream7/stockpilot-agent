# StockPilot Agent Evaluation

这套评估回答一个核心问题：优化后的 Agent 是否能用合适的证据，稳定地产出用户真正需要的研究回答。文档按“目标、维度、样本、结果、迭代”组织；不要把旧的 fixture 分数当作线上效果。

## 1. 评估维度

### D1 走没走对路：路径正确性（35%）

**评的是什么**：意图识别是否正确，期望工具集与实际工具路径是否匹配。
**信号来源**：trace 的 `planner_decision`、`tool_completed`、`run_finished`，以及 case 标注的 `expected_first_action` / `expected_sec_fetch`。
**怎么打分**：规则评分；检查首个工作流动作、必要工具、是否误取/漏取 SEC、是否正常结束。
**兜底**：D1 低于 60 时，总分封顶 59，避免路线错误被格式完整的答案掩盖。

### D2 做得好不好：回答与证据质量（40%）

**评的是什么**：memo 是否真正回答用户问题，是否完整、合规，并带有足够且可追溯的依据。
**信号来源**：`draft_memo` 的中间产物、最终 memo、新闻数量/来源、SEC filings、免责声明和 Critic 的证据分。
**怎么打分**：规则结构检查 + 证据底线；生产版本再用独立 LLM-as-judge 和人工盲审抽样。
**兜底**：`evidence_score < 50` 时总分封顶 59；缺少一手来源不能被长 memo 掩盖。

### D3 用户会不会满意：下一步反应校准（25%）

**评的是什么**：如果我是用户，看完回答后下一句更可能是“行”（`accept`）、“补证据”（`ask_for_evidence`）还是“再来一次”（`retry`）。
**信号来源**：Critic 输出的 `predicted_user_followup`，对照 case 作者在不看输出时预先填写的 `expected_user_followup`；上线后再接真实会话反馈。
**怎么打分**：三分类精确匹配，输出准确率和混淆矩阵；这不是把 Critic 自己的预测直接当满意度。
**兜底**：`ask_for_evidence` / `retry` 的错判单独报告，不允许被 `accept` 多数类掩盖。

总分：`0.35 × D1 + 0.40 × D2 + 0.25 × D3`，D1 < 60 或证据分 < 50 时封顶 59。

## 2. 样本与执行

共 28 个 case：24 个首次研究（风险、财报/指引、监管/法律、催化/竞争、模糊输入）+ 4 个多轮追问（M01–M04）。每条 case 都写明覆盖场景、首工具、SEC 行为和预期下一句，样本不是随机用户分布，而是面向风险边界的代表性测试集。

真实 DeepSeek 批次已完成 23 条首轮 + A04 续跑 + M01–M04，结果目录为 `evaluation/results/llm_20260826/` 和 `evaluation/results/llm_20260826_resume/`。续跑进度见 `progress.json`。原始 trace 不入库。

## 3. 运行

```bash
python evaluation/run_evaluation.py --use-llm --model deepseek-v4-flash \
  --output-dir evaluation/results/llm_20260826
```

中断后从指定 case 续跑：

```bash
python evaluation/run_evaluation.py --use-llm --start-index 23 \
  --output-dir evaluation/results/llm_20260826_resume
```

API key 只在当前进程隐藏读取，不写入文件、CSV 或 trace。评估时若本地 embedding 模型不可下载，会快速走风险关键词兜底；Supervisor、memo 和 Critic 仍使用真实 LLM。

## 4. 结果阅读顺序

先看 `evaluation_results.csv` 的样本级分数和 `score_reason`，再看 `evaluation_summary.json` 的分层汇总，最后回放低分 trace。报告至少要按场景、输入类型、工具调用次数、延迟段和 token 段切分；只报总分均值不算完成评估。

旧的 `evaluation/results/before_primary_source_fix/` 是 v1 历史基线，仅用于解释缺陷修复，不参与当前主结论。

## 5. 局限与 Goodhart 风险

fixture case 的覆盖面代表“重要边界”，不代表真实用户比例；Critic 预测也不等于真实满意度。最容易被刷高的是 D2 的格式完整度：模型可以堆免责声明和证据标题，却没有增加事实质量。因此必须同时保留主来源覆盖、trace 回放、独立 judge 和人工盲审。

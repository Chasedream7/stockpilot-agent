# Trace-based evaluation baseline

这是任务二的可复现起点，不把 fixture 结果伪装成线上效果。`cases.json` 包含 24 条人工构造的典型输入，覆盖：

- 6 条明确的综合风险问题；
- 6 条财报/指引问题，其中 1 条故意模拟 SEC 检索失败；
- 4 条监管/法律问题，其中包含“反垄断”同义表达漏召回；
- 4 条催化/竞争问题；
- 4 条模糊或低上下文问题，其中包括少新闻量和一手来源失败。

每条 case 都标注了它试图覆盖的情景及是否期望补一手来源。它们不是用户真实行为样本；提交时应补充真实 demo trace，并在报告中将两类样本拆开分析。

## 运行

```bash
python evaluation/run_evaluation.py
```

默认使用离线 fixture 和规则基线，不需要 API Key，结果写到 `evaluation/results/`；trace 写到 `evaluation/results/traces/`。需要实际检验 LLM planner / Critic 时，在固定模型、prompt 版本和运行时间后再运行：

```bash
python evaluation/run_evaluation.py --use-llm
```

命令会以隐藏输入方式要求 API key，只保留在这次进程内，不读取环境变量或写入文件。两种模式不可直接混合比较；每次 LLM 复评前请另存结果，避免覆盖规则基线。

## 指标定义

| 维度 | 分数 | 信号 | 规则 |
| --- | --- | --- | --- |
| 路径正确性 | `route_score` | trace 中实际工具路径 vs case 标注的最小工具集 | 缺少新闻/量化/风险/memo/Critic 会扣分；财报/监管未补 SEC 扣 35 分；一般问题无故补 SEC 扣 15 分 |
| 产物质量 | `quality_score` | memo、目标聚焦、证据区、免责声明、新闻量与一手来源 | 规则混合分，不判断金融结论是否真实 |
| 证据充分性 | `evidence_score` | 新闻量、是否 live、是否成功取得 SEC 一手来源 | 作为质量分输入，也作为总分 guardrail |
| 满意度代理 | `satisfaction_score` | Critic 的 `predicted_user_followup` | `accept=100`、`ask_for_evidence=50`、`retry=0`；不是用户真实反馈 |

总分为 `35% 路径 + 40% 质量 + 25% 满意度`。如果 `evidence_score < 50`，总分会被封顶为 59，防止一份格式流畅但证据薄弱的 memo 拿到高分。

每行 CSV 都有 `score_reason`、完整工具路径、延迟、token、trace 相对路径，便于按场景、工具次数和低分 case 做切片。

## 已暴露的基线问题

当前规则 fixture 基线的低分样本本身就是有价值的 case study 素材：

1. **L02**：`反垄断调查` 没命中当前的一手来源意图关键词，因此没有补 SEC；说明“关键词判断是否需要一手来源”过窄。
2. **E06 / L04 / A04**：SEC 工具失败后，Critic 因为只看“是否尝试过”而不是“是否成功拿到证据”仍给出 `accept`；这是满意度代理和自检逻辑的共同盲点。
3. **A03**：输入极短且只有 3 条样例新闻，流程路径完全正确，但证据分不足导致总分被封顶；反直觉地说明“走对路”不等于“给得出可靠答案”。

下一轮建议先修第 2 条：将 Critic 的 `should_retrieve` 条件从 `sec_attempted` 改为 `has_successful_primary_source`，并在来源失败时要求降级回复或追问用户。然后用同一批 24 case 复跑比较低分样本数、平均 evidence score、`accept` 的校准误差。

## 局限

- 这套默认分数依赖人工 case 标注和规则，不能替代人工审阅事实正确性。
- in-loop Critic 不能同时作为唯一的离线 judge；LLM 版本应另加独立 judge prompt、盲审样本和人工抽检。
- fixture 故意让 live 新闻不可用，所以不能用该基线报告真实线上时延、模型 token 或市场数据质量。
- Goodhart 风险最大的是“memo 的标题/免责声明完整度”：模型可能为拿分而堆版式，却没有增加可核验事实。因此必须保留主来源覆盖率和人工 case study。

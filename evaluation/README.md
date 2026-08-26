# StockPilot Agent Evaluation v2

这是针对优化后 Agent 的可复现实验，不把离线 fixture 分数伪装成线上效果。评估不是泛泛检查 memo 格式，而是回答五个明确业务问题；完整定义、case 映射与判定口径见 [评估设计.md](评估设计.md)。

## 评估对象与样本

- **24 条首次研究 case**：风险、财报/指引、监管/法律、催化/竞争、模糊/低上下文各类输入。
- **4 条多轮追问 case（M01–M04）**：验证历史记忆、指代解析和“只补缺失证据”。
- 每条 case 都有人工标注的首个工作流动作、是否应新取 SEC、以及预期用户下一句（accept / ask_for_evidence / retry）。

因此，v2 的结果共 28 条；evaluation/results/before_primary_source_fix/ 保留的是此前 24 条 v1 基线，不能与 v2 总分直接横比。

## 运行

~~~bash
python evaluation/run_evaluation.py
~~~

默认使用离线 fixture 和规则基线，不需要 API key。结果写到 evaluation/results/，trace 写到 evaluation/results/traces/。

要实际检验 DeepSeek 的自主 Supervisor 和 Critic，请固定模型、prompt 版本和运行日期后另存结果：

~~~bash
python evaluation/run_evaluation.py --use-llm --model deepseek-v4-flash \
  --output-dir evaluation/results/llm_20260826
~~~

API key 只在本次进程内以隐藏输入方式读取，不读取环境变量或写入文件。规则基线衡量兜底路径；只有 llm_fixture 的 Q4 才能说明 LLM 是否自主选择了第一项工具。

如果运行中断，可用 `--start-index N` 从第 N 条 case 续跑；续跑结果应单独保存，待所有批次完成后再合并汇总。`evaluation/results/llm_20260826_resume/progress.json` 记录了一次已保存的 28 条进度。

## 产物与指标

每行 CSV 除路径、质量、证据、trace、延迟和 token 外，还记录：

- expected_first_action / actual_first_action：首个 Supervisor 工作流动作；
- expected_sec_fetch / actual_sec_fetch / has_successful_primary_source：区分“调用了工具”与“真正得到一手证据”；
- route_evidence_outcome：路径正确/错误 × 证据充分/不足；
- expected_user_followup / predicted_user_followup / followup_prediction_correct：Critic 校准，而非自我打分；
- reused_evidence_without_reanalysis：多轮追问是否重复抓新闻或重跑情绪/风险。

总分是 35% 路径 + 40% 产物质量 + 25% Critic 预测对齐。若 evidence_score < 50，总分封顶为 59，避免证据薄弱的流畅 memo 得高分。原来的 satisfaction_score 已移除：把 Critic 自己的预测直接当作用户满意度是无效的自评。

evaluation_summary.json 的 business_questions 是面向阅卷和复盘的五组答案：Q1 的 SEC 成功率与失败识别率、Q2 的二维矩阵、Q3 的混淆矩阵、Q4 的首工具匹配率、Q5 的记忆复用率。

## 边界

- 离线 fixture 不能代表真实市场数据质量、线上时延或 LLM token 成本。
- expected_user_followup 是 case 作者的盲标代理，Q3 应与真实会话反馈及人工盲审分开报告。
- in-loop Critic 不能兼任唯一的离线 judge；生产评估应增加独立 judge prompt、人工抽检和固定的真实数据时间窗。

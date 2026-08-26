"""Run a reproducible, trace-based evaluation for the optimized StockPilot agent."""

import argparse
import csv
import getpass
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


FOLLOWUP_LABELS = {"accept", "ask_for_evidence", "retry"}
REANALYSIS_ACTIONS = {"collect_news", "analyze_sentiment", "assess_risk"}


def load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        cases = json.load(handle)
    for case in cases:
        if case.get("expected_user_followup") not in FOLLOWUP_LABELS:
            raise ValueError(
                f"{case.get('sample_id', '<unknown>')} requires a valid expected_user_followup"
            )
        if not case.get("expected_first_action"):
            raise ValueError(f"{case['sample_id']} requires expected_first_action")
    return cases


def sec_fixture(ticker):
    return {
        "company_name": f"{ticker} Fixture Company",
        "cik": 1234,
        "source_note": "SEC EDGAR fixture (primary-source metadata)",
        "filings": [
            {
                "form": "10-Q",
                "filed_at": "2026-08-01",
                "report_date": "2026-06-30",
                "items": "2.02",
                "document_url": "https://example.com/filing/10q",
            },
            {
                "form": "8-K",
                "filed_at": "2026-07-15",
                "report_date": "2026-07-15",
                "items": "2.02,9.01",
                "document_url": "https://example.com/filing/8k",
            },
        ],
    }


def load_trace(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trace_metrics(events):
    total_tokens = sum(
        value
        for event in events
        if isinstance(value := (event.get("usage") or {}).get("total_tokens"), int)
    )
    elapsed = [event["elapsed_ms"] for event in events if event.get("elapsed_ms") is not None]
    return {
        "latency_ms": round(max(elapsed), 2) if elapsed else 0,
        "token_total": total_tokens,
        "tool_calls": sum(event.get("event") == "tool_completed" for event in events),
    }


def expected_sec_fetch(case):
    return case.get("expected_sec_fetch", case["expects_primary_source"])


def first_execution_action(actions):
    return next((action for action in actions if action != "analyze_goal"), None)


def has_successful_primary_source(result):
    return bool((result.get("sec_filings") or {}).get("filings"))


def score_route(case, actions):
    is_follow_up = case.get("turn_type") == "follow_up"
    required = {"self_check"}
    if is_follow_up:
        if case.get("expects_memo_revision"):
            required.add("draft_memo")
    else:
        required.update({"collect_news", "analyze_sentiment", "assess_risk", "draft_memo"})

    action_set = set(actions)
    reasons = []
    score = 100
    missing = sorted(required - action_set)
    if missing:
        score -= 18 * len(missing)
        reasons.append("缺少必要步骤：" + ", ".join(missing))
    has_sec_action = "fetch_sec_filings" in action_set
    if expected_sec_fetch(case) and not has_sec_action:
        score -= 35
        reasons.append("该回合需要补 SEC，但没有发起检索")
    if not expected_sec_fetch(case) and has_sec_action:
        score -= 15
        reasons.append("已有证据或一般问题却发生了非必要 SEC 检索")
    if is_follow_up and action_set.intersection(REANALYSIS_ACTIONS):
        score -= 25
        reasons.append("追问重新抓取/重算已有新闻证据，未复用研究记忆")
    actual_first_action = first_execution_action(actions)
    if actual_first_action != case["expected_first_action"]:
        score -= 25
        reasons.append(
            f"首个工作流动作应为 {case['expected_first_action']}，实际为 {actual_first_action}"
        )
    if "finish" not in action_set:
        score -= 15
        reasons.append("没有正常终止")
    return max(0, score), "；".join(reasons or ["工具路径符合该场景的最小证据需求"])


def score_deliverable(case, result):
    memo = result.get("memo", "")
    critic = result.get("critic") or {}
    headline_count = (result.get("summary") or {}).get("total", 0)
    has_sec = has_successful_primary_source(result)
    reasons = []

    answer_score = 0
    if len(memo) >= 250:
        answer_score += 35
    else:
        reasons.append("memo 过短")
    if "不构成投资建议" in memo:
        answer_score += 20
    else:
        reasons.append("缺少投资建议免责声明")
    if any(marker in memo for marker in ("证据", "正面证据", "负面/风险证据", "Risk")):
        answer_score += 25
    else:
        reasons.append("没有显式证据区")
    if result.get("goal_profile", {}).get("label"):
        answer_score += 20
    else:
        reasons.append("没有目标聚焦")

    evidence_score = 30 + min(headline_count * 5, 35)
    if (result.get("source_note") or "").startswith("Live"):
        evidence_score += 15
    else:
        reasons.append("新闻来自离线 fixture，不能作为实时研究结论")
    if case["expects_primary_source"]:
        if has_sec:
            evidence_score += 25
        else:
            evidence_score -= 25
            reasons.append("需要一手披露但本回合结束时仍无可用来源")
    evidence_score = max(0, min(100, evidence_score))

    predicted = critic.get("predicted_user_followup", "retry")
    expected = case["expected_user_followup"]
    alignment_score = 100 if predicted == expected else 0
    if predicted != expected:
        reasons.append(f"Critic 预测 {predicted}，人工标注为 {expected}")
    quality_score = round(answer_score * 0.55 + evidence_score * 0.45)
    return quality_score, evidence_score, alignment_score, "；".join(reasons)


def score_total(route_score, quality_score, evidence_score, alignment_score):
    score = route_score * 0.35 + quality_score * 0.4 + alignment_score * 0.25
    return round(min(score, 59) if evidence_score < 50 else score, 1)


def run_case(case, trace_dir, use_llm, model, api_key=None):
    original_finviz, original_sec = app.fetch_finviz_soup, app.fetch_sec_filings
    original_goal_embedding = app.infer_goal_profile_with_embedding
    original_semantic_risk = app.semantic_risk_findings
    app.fetch_finviz_soup = lambda ticker: (_ for _ in ()).throw(
        RuntimeError("evaluation fixture: live news intentionally unavailable")
    )
    if case["sec_mode"] == "success":
        app.fetch_sec_filings = sec_fixture
    elif case["sec_mode"] == "fail":
        app.fetch_sec_filings = lambda ticker: (_ for _ in ()).throw(
            RuntimeError("evaluation fixture: SEC lookup unavailable")
        )
    else:
        app.fetch_sec_filings = lambda ticker: (_ for _ in ()).throw(
            RuntimeError("evaluation fixture: SEC should not be called for this case")
        )
    if use_llm:
        # The LLM evaluation remains DeepSeek-backed; avoid blocking on an optional
        # Hugging Face download when the local embedding cache/network is unavailable.
        app.infer_goal_profile_with_embedding = lambda mission, trace=None: None
        app.semantic_risk_findings = lambda scored_news, priority_categories, trace=None: (
            (_ for _ in ()).throw(RuntimeError("local embedding disabled for LLM evaluation"))
        )
    try:
        run_args = {
            "max_headlines": case["max_headlines"],
            "allow_fallback_data": True,
            "use_llm": use_llm,
            "llm_model": model,
            "trace_dir": trace_dir,
            "api_key": api_key,
        }
        if case.get("turn_type") != "follow_up":
            return app.run_workflow(case["ticker"], case["mission"], **run_args)

        seed_mission = case["seed_mission"]
        seed = app.run_workflow(case["ticker"], seed_mission, **run_args)
        history = [
            {"role": "user", "content": seed_mission},
            {"role": "assistant", "content": seed["memo"]},
            {"role": "user", "content": case["mission"]},
        ]
        return app.run_workflow(
            case["ticker"],
            case["mission"],
            prior_memory=seed["memory"],
            messages=history,
            **run_args,
        )
    finally:
        app.fetch_finviz_soup = original_finviz
        app.fetch_sec_filings = original_sec
        app.infer_goal_profile_with_embedding = original_goal_embedding
        app.semantic_risk_findings = original_semantic_risk


def percentage(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None


def question_summary(rows, run_mode):
    primary_success = [
        row for row in rows
        if row["expects_primary_source"] and row["expected_sec_fetch"] and row["sec_mode"] == "success"
    ]
    primary_failure = [
        row for row in rows
        if row["expects_primary_source"] and row["expected_sec_fetch"] and row["sec_mode"] == "fail"
    ]
    ambiguous = [row for row in rows if row["scenario"] == "ambiguous_low_context"]
    fresh = [row for row in rows if row["turn_type"] == "fresh"]
    follow_ups = [row for row in rows if row["turn_type"] == "follow_up"]
    matrix = Counter(row["route_evidence_outcome"] for row in ambiguous)
    confusion = {
        expected: {predicted: 0 for predicted in sorted(FOLLOWUP_LABELS)}
        for expected in sorted(FOLLOWUP_LABELS)
    }
    for row in rows:
        confusion[row["expected_user_followup"]][row["predicted_user_followup"]] += 1

    return {
        "Q1_primary_source": {
            "question": "财报、指引、监管类问题中，Agent 是否成功补到 SEC 一手来源；若检索失败，Critic 是否识别证据缺口？",
            "sec_fetch_attempt_rate": percentage(sum(row["actual_sec_fetch"] for row in primary_success), len(primary_success)),
            "successful_primary_source_acquisition_rate": percentage(sum(row["has_successful_primary_source"] for row in primary_success), len(primary_success)),
            "critic_failure_detection_rate": percentage(sum(row["critic_status"] == "needs_more_evidence" and row["predicted_user_followup"] == "ask_for_evidence" for row in primary_failure), len(primary_failure)),
            "success_fixture_case_ids": [row["sample_id"] for row in primary_success],
            "failure_fixture_case_ids": [row["sample_id"] for row in primary_failure],
        },
        "Q2_ambiguous_input": {
            "question": "面对模糊或极短输入，Agent 是路径判断错误，还是路径正确但证据不足？",
            "route_evidence_matrix": dict(sorted(matrix.items())),
            "route_correct_rate": percentage(sum(row["route_correct"] for row in ambiguous), len(ambiguous)),
            "evidence_sufficient_rate": percentage(sum(row["evidence_sufficient"] for row in ambiguous), len(ambiguous)),
            "case_ids": [row["sample_id"] for row in ambiguous],
        },
        "Q3_critic_followup_calibration": {
            "question": "Critic 对用户下一句话（accept / ask_for_evidence / retry）的预测，与人工标注的预期反应有多大偏差？",
            "prediction_accuracy": percentage(sum(row["followup_prediction_correct"] for row in rows), len(rows)),
            "mismatch_count": sum(not row["followup_prediction_correct"] for row in rows),
            "confusion_matrix_expected_by_predicted": confusion,
            "annotation_note": "预期反应是 case 作者的盲标代理，不等同于线上真实用户反馈。",
        },
        "Q4_first_tool_selection": {
            "question": "首次研究时，Supervisor 是否从完整工具列表中为任务选择合适的第一项证据工具？",
            "first_action_alignment_rate": percentage(sum(row["first_action_correct"] for row in fresh), len(fresh)),
            "case_ids": [row["sample_id"] for row in fresh],
            "interpretation_note": (
                "rule_fixture_baseline 只显示规则路径；自主选择能力应查看 llm_fixture 的该指标。"
                if run_mode == "rule_fixture_baseline"
                else "该数值来自固定模型、固定 prompt 的 LLM 运行。"
            ),
        },
        "Q5_followup_memory": {
            "question": "追问时，Agent 是否先用 Critic 检查历史证据、复用已有研究，并且只补当前缺口？",
            "first_self_check_rate": percentage(sum(row["actual_first_action"] == "self_check" for row in follow_ups), len(follow_ups)),
            "evidence_reuse_without_reanalysis_rate": percentage(sum(row["reused_evidence_without_reanalysis"] for row in follow_ups), len(follow_ups)),
            "gap_only_sec_fetch_alignment_rate": percentage(sum(row["actual_sec_fetch"] == row["expected_sec_fetch"] for row in follow_ups), len(follow_ups)),
            "case_ids": [row["sample_id"] for row in follow_ups],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=PROJECT_ROOT / "evaluation" / "cases.json")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--use-llm", action="store_true", help="Run DeepSeek Supervisor and Critic.")
    parser.add_argument("--model", default=app.DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start at this zero-based case index, useful for resuming an interrupted run.",
    )
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    api_key = None
    if args.use_llm:
        api_key = getpass.getpass("DeepSeek API key (not saved): ").strip()
        if not api_key:
            parser.error("--use-llm requires a DeepSeek API key entered at the prompt.")

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    if args.start_index < 0 or args.start_index >= len(cases):
        parser.error("--start-index must point to an existing case")
    cases = cases[args.start_index :]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("No evaluation cases selected.")
    run_mode = "llm_fixture" if args.use_llm else "rule_fixture_baseline"

    rows = []
    for case in cases:
        result = run_case(case, trace_dir, args.use_llm, args.model, api_key=api_key)
        events = load_trace(result["trace_path"])
        metrics = trace_metrics(events)
        actions = [step["action"] for step in result["steps"]]
        route_score, route_reason = score_route(case, actions)
        quality_score, evidence_score, alignment_score, quality_reason = score_deliverable(case, result)
        actual_first_action = first_execution_action(actions)
        reused = not bool(set(actions).intersection(REANALYSIS_ACTIONS))
        predicted = (result.get("critic") or {}).get("predicted_user_followup", "retry")
        rows.append({
            "sample_id": case["sample_id"],
            "scenario": case["scenario"],
            "turn_type": case.get("turn_type", "fresh"),
            "input": case["mission"],
            "coverage_note": case["coverage_note"],
            "run_mode": run_mode,
            "llm_model": args.model if args.use_llm else "",
            "expects_primary_source": case["expects_primary_source"],
            "expected_sec_fetch": expected_sec_fetch(case),
            "sec_mode": case["sec_mode"],
            "expected_first_action": case["expected_first_action"],
            "actual_first_action": actual_first_action,
            "first_action_correct": actual_first_action == case["expected_first_action"],
            "actual_sec_fetch": "fetch_sec_filings" in actions,
            "has_successful_primary_source": has_successful_primary_source(result),
            "reused_evidence_without_reanalysis": reused if case.get("turn_type") == "follow_up" else "",
            "route_evidence_outcome": (
                "route_correct_evidence_sufficient" if route_score == 100 and evidence_score >= 50
                else "route_correct_evidence_insufficient" if route_score == 100
                else "route_incorrect_evidence_sufficient" if evidence_score >= 50
                else "route_incorrect_evidence_insufficient"
            ),
            "route_correct": route_score == 100,
            "evidence_sufficient": evidence_score >= 50,
            "route_score": route_score,
            "quality_score": quality_score,
            "evidence_score": evidence_score,
            "followup_alignment_score": alignment_score,
            "total_score": score_total(route_score, quality_score, evidence_score, alignment_score),
            "critic_status": (result.get("critic") or {}).get("status"),
            "expected_user_followup": case["expected_user_followup"],
            "predicted_user_followup": predicted,
            "followup_prediction_correct": predicted == case["expected_user_followup"],
            "latency_ms": metrics["latency_ms"],
            "token_total": metrics["token_total"],
            "tool_calls": metrics["tool_calls"],
            "trace_path": str(Path(result["trace_path"]).relative_to(output_dir)),
            "score_reason": f"路径：{route_reason}。产物：{quality_reason}",
        })

    output_file = output_dir / "evaluation_results.csv"
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "evaluation_version": "v2_explicit_business_questions",
        "sample_count": len(rows),
        "run_mode": run_mode,
        "llm_model": args.model if args.use_llm else None,
        "mean_total_score": round(sum(row["total_score"] for row in rows) / len(rows), 2),
        "mean_route_score": round(sum(row["route_score"] for row in rows) / len(rows), 2),
        "mean_quality_score": round(sum(row["quality_score"] for row in rows) / len(rows), 2),
        "mean_followup_alignment_score": round(sum(row["followup_alignment_score"] for row in rows) / len(rows), 2),
        "low_score_sample_ids": [row["sample_id"] for row in rows if row["total_score"] < 60],
        "business_questions": question_summary(rows, run_mode),
    }
    summary_file = output_dir / "evaluation_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": str(output_file), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

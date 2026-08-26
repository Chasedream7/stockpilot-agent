"""Run reproducible, trace-based baseline evaluation for StockPilotX.

The default uses local fixtures and rules so it is safe to run without a market-data
or LLM key. Pass --use-llm only after recording the model and prompt version in the
submission; those results are not directly comparable to the fixture baseline.
"""

import argparse
import csv
import getpass
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


def load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


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
    total_tokens = 0
    for event in events:
        value = (event.get("usage") or {}).get("total_tokens")
        if isinstance(value, int):
            total_tokens += value
    elapsed_values = [event.get("elapsed_ms") for event in events if event.get("elapsed_ms") is not None]
    return {
        "latency_ms": round(max(elapsed_values), 2) if elapsed_values else 0,
        "token_total": total_tokens,
        "tool_calls": sum(event.get("event") == "tool_completed" for event in events),
    }


def score_route(case, actions):
    required = {"collect_news", "analyze_sentiment", "assess_risk", "draft_memo", "self_check"}
    action_set = set(actions)
    missing = sorted(required - action_set)
    reasons = []
    score = 100
    if missing:
        score -= 18 * len(missing)
        reasons.append("缺少必要步骤：" + ", ".join(missing))

    has_sec = "fetch_sec_filings" in action_set
    if case["expects_primary_source"] and not has_sec:
        score -= 35
        reasons.append("涉及一手披露的问题没有补 SEC")
    if not case["expects_primary_source"] and has_sec:
        score -= 15
        reasons.append("一般问题发生了非必要的 SEC 检索")
    if "finish" not in action_set:
        score -= 15
        reasons.append("没有正常终止")
    if not reasons:
        reasons.append("工具路径符合该场景的最小证据需求")
    return max(0, score), "；".join(reasons)


def score_deliverable(case, result):
    memo = result.get("memo", "")
    critic = result.get("critic") or {}
    source_note = result.get("source_note", "")
    headline_count = (result.get("summary") or {}).get("total", 0)
    sec = result.get("sec_filings") or {}
    has_sec = bool(sec.get("filings"))

    answer_score = 0
    reasons = []
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
    if source_note.startswith("Live"):
        evidence_score += 15
    else:
        reasons.append("新闻来自离线 fixture，不能作为实时研究结论")
    if case["expects_primary_source"]:
        if has_sec:
            evidence_score += 25
        else:
            evidence_score -= 25
            reasons.append("需要一手披露但检索未成功")
    evidence_score = max(0, min(100, evidence_score))

    quality_score = round(answer_score * 0.55 + evidence_score * 0.45)
    predicted_followup = critic.get("predicted_user_followup", "retry")
    satisfaction_score = {"accept": 100, "ask_for_evidence": 50, "retry": 0}.get(
        predicted_followup, 0
    )
    if critic.get("status") != "pass":
        satisfaction_score = min(satisfaction_score, 50)
        reasons.append("Critic 未通过最终版本")
    if not reasons:
        reasons.append("产物结构、证据量和合规提示满足基线规则")
    return quality_score, evidence_score, satisfaction_score, "；".join(reasons)


def score_total(route_score, quality_score, evidence_score, satisfaction_score):
    score = route_score * 0.35 + quality_score * 0.4 + satisfaction_score * 0.25
    # 证据是底线，不允许用流畅回答掩盖低质量来源。
    if evidence_score < 50:
        score = min(score, 59)
    return round(score, 1)


def run_case(case, trace_dir, use_llm, api_key=None):
    original_finviz = app.fetch_finviz_soup
    original_sec = app.fetch_sec_filings
    app.fetch_finviz_soup = lambda ticker: (_ for _ in ()).throw(
        RuntimeError("evaluation fixture: live news intentionally unavailable")
    )
    if case["sec_mode"] == "success":
        app.fetch_sec_filings = sec_fixture
    elif case["sec_mode"] == "fail":
        app.fetch_sec_filings = lambda ticker: (_ for _ in ()).throw(
            RuntimeError("evaluation fixture: SEC lookup unavailable")
        )
    try:
        return app.run_workflow(
            case["ticker"],
            case["mission"],
            max_headlines=case["max_headlines"],
            allow_fallback_data=True,
            use_llm=use_llm,
            trace_dir=trace_dir,
            api_key=api_key,
        )
    finally:
        app.fetch_finviz_soup = original_finviz
        app.fetch_sec_filings = original_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=PROJECT_ROOT / "evaluation" / "cases.json")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--use-llm", action="store_true", help="Run the agent planner and Critic with the configured LLM.")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    api_key = None
    if args.use_llm:
        api_key = getpass.getpass("OpenAI API key (not saved): ").strip()
        if not api_key:
            parser.error("--use-llm requires an API key entered at the prompt.")

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    if args.max_cases:
        cases = cases[: args.max_cases]

    rows = []
    for case in cases:
        result = run_case(case, trace_dir, args.use_llm, api_key=api_key)
        events = load_trace(result["trace_path"])
        metrics = trace_metrics(events)
        actions = [step["action"] for step in result["steps"]]
        route_score, route_reason = score_route(case, actions)
        quality_score, evidence_score, satisfaction_score, quality_reason = score_deliverable(case, result)
        total_score = score_total(route_score, quality_score, evidence_score, satisfaction_score)
        rows.append(
            {
                "sample_id": case["sample_id"],
                "scenario": case["scenario"],
                "input": case["mission"],
                "coverage_note": case["coverage_note"],
                "run_mode": "llm_fixture" if args.use_llm else "rule_fixture_baseline",
                "expected_primary_source": case["expects_primary_source"],
                "actual_tools": " | ".join(actions),
                "route_score": route_score,
                "quality_score": quality_score,
                "evidence_score": evidence_score,
                "satisfaction_score": satisfaction_score,
                "total_score": total_score,
                "critic_status": (result.get("critic") or {}).get("status"),
                "predicted_user_followup": (result.get("critic") or {}).get("predicted_user_followup"),
                "latency_ms": metrics["latency_ms"],
                "token_total": metrics["token_total"],
                "tool_calls": metrics["tool_calls"],
                "trace_path": str(Path(result["trace_path"]).relative_to(output_dir)),
                "score_reason": f"路径：{route_reason}。产物：{quality_reason}",
            }
        )

    output_file = output_dir / "evaluation_results.csv"
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(rows),
        "run_mode": rows[0]["run_mode"],
        "mean_total_score": round(sum(row["total_score"] for row in rows) / len(rows), 2),
        "mean_route_score": round(sum(row["route_score"] for row in rows) / len(rows), 2),
        "mean_quality_score": round(sum(row["quality_score"] for row in rows) / len(rows), 2),
        "mean_satisfaction_score": round(sum(row["satisfaction_score"] for row in rows) / len(rows), 2),
        "low_score_sample_ids": [row["sample_id"] for row in rows if row["total_score"] < 60],
    }
    summary_file = output_dir / "evaluation_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": str(output_file), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

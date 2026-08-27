import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app
import pandas as pd


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.original_finviz = app.fetch_finviz_soup
        self.original_sec = app.fetch_sec_filings
        app.fetch_finviz_soup = lambda ticker: (_ for _ in ()).throw(
            RuntimeError("test: live source unavailable")
        )
        app.fetch_sec_filings = lambda ticker: {
            "company_name": "Test Company",
            "cik": 1234,
            "source_note": "SEC EDGAR filings (stub)",
            "filings": [
                {
                    "form": "10-Q",
                    "filed_at": "2026-08-01",
                    "report_date": "2026-06-30",
                    "items": "",
                    "document_url": "https://example.com/filing",
                }
            ],
        }

    def tearDown(self):
        app.fetch_finviz_soup = self.original_finviz
        app.fetch_sec_filings = self.original_sec

    def run_case(self, mission):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        result = app.run_workflow(
            "TEST",
            mission,
            max_headlines=8,
            allow_fallback_data=True,
            use_llm=False,
            trace_dir=temp_dir.name,
        )
        return result, Path(result["trace_path"])

    def test_generic_goal_ends_after_one_memo_and_critic_check(self):
        result, trace_path = self.run_case("判断这只股票近期是否有值得记录的风险信号")
        actions = [step["action"] for step in result["steps"]]

        self.assertEqual(
            actions,
            [
                "analyze_goal",
                "collect_news",
                "analyze_sentiment",
                "assess_risk",
                "draft_memo",
                "self_check",
                "finish",
            ],
        )
        self.assertEqual(result["critic"]["status"], "pass")
        self.assertTrue(trace_path.exists())

    def test_primary_source_goal_triggers_retrieval_then_revision(self):
        result, trace_path = self.run_case("复盘最近财报指引是否改变长期持仓逻辑")
        actions = [step["action"] for step in result["steps"]]

        self.assertIn("fetch_sec_filings", actions)
        self.assertEqual(actions.count("draft_memo"), 2)
        self.assertEqual(result["critic"]["status"], "pass")

        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(event["event"] == "planner_decision" for event in events))
        self.assertTrue(any(event["event"] == "tool_completed" for event in events))
        self.assertTrue(all("timestamp" in event and "duration_ms" in event for event in events))

    def test_failed_primary_source_does_not_report_a_passing_critic(self):
        app.fetch_sec_filings = lambda ticker: (_ for _ in ()).throw(
            RuntimeError("test: SEC unavailable")
        )
        result, _ = self.run_case("复盘最近财报指引是否改变长期持仓逻辑")

        self.assertEqual(result["critic"]["status"], "needs_more_evidence")
        self.assertEqual(result["critic"]["predicted_user_followup"], "ask_for_evidence")
        self.assertEqual(result["steps"][-1]["action"], "finish")

    def test_follow_up_reuses_evidence_and_only_fetches_the_missing_primary_source(self):
        first, _ = self.run_case("判断近期新闻是否改变我的长期持仓逻辑")
        history = [
            {"role": "user", "content": first["mission"]},
            {"role": "assistant", "content": first["memo"]},
            {"role": "user", "content": "详细说说那个监管风险"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            follow_up = app.run_workflow(
                "TEST",
                "详细说说那个监管风险",
                max_headlines=8,
                allow_fallback_data=True,
                use_llm=False,
                trace_dir=directory,
                prior_memory=first["memory"],
                messages=history,
            )

        actions = [step["action"] for step in follow_up["steps"]]
        self.assertEqual(
            actions,
            [
                "analyze_goal",
                "self_check",
                "fetch_sec_filings",
                "draft_memo",
                "self_check",
                "finish",
            ],
        )
        self.assertIs(follow_up["memory"]["news_df"], first["memory"]["news_df"])
        self.assertEqual(follow_up["critic"]["status"], "pass")
        self.assertGreater(follow_up["memo"].count("本轮追问回答"), 0)
        self.assertIn("详细说说那个监管风险", follow_up["memo"])
        self.assertNotEqual(follow_up["memo"], first["memo"])
        self.assertTrue(follow_up["goal_changed"])
        self.assertEqual(first["goal_profile"]["key"], "hold_check")
        self.assertEqual(follow_up["goal_profile"]["key"], "regulation_review")

    def test_short_regulatory_follow_up_beats_generic_risk_profile(self):
        profile = app.infer_goal_profile("详细说说那个监管风险")
        self.assertEqual(profile["key"], "regulation_review")

    def test_llm_mode_keeps_explicit_follow_up_topic_even_without_embedding(self):
        profile, tool = app.analyze_goal(
            "请核对 SEC 披露中的监管调查",
            use_llm=True,
            model="unused",
            api_key=None,
        )
        self.assertEqual(profile["key"], "regulation_review")
        self.assertEqual(tool, "High-signal intent router")

    def test_repeated_llm_follow_up_is_wrapped_with_latest_question(self):
        first, _ = self.run_case("判断近期新闻是否改变我的长期持仓逻辑")
        latest_question = "你刚才提到的监管风险，具体是哪一份披露支持？"
        original_call = app.call_llm_text
        self.addCleanup(setattr, app, "call_llm_text", original_call)
        app.call_llm_text = lambda *args, **kwargs: first["memo"]

        memo, _ = app.build_memo(
            "TEST",
            latest_question,
            app.profile_by_key("regulation_review"),
            first["summary"],
            first["risk"],
            first["snapshot"],
            first["scored_news"],
            first["source_note"],
            use_llm=True,
            model="test-model",
            sec_filings=first["sec_filings"],
            previous_memo=first["memo"],
            is_follow_up=True,
        )

        self.assertNotEqual(memo, first["memo"])
        self.assertIn("本轮追问回答", memo)
        self.assertIn(latest_question, memo)


class SemanticRoutingTest(unittest.TestCase):
    def setUp(self):
        self.original_embed_texts = app.embed_texts

    def tearDown(self):
        app.embed_texts = self.original_embed_texts

    def test_goal_profile_uses_embedding_similarity(self):
        profiles = [app.DEFAULT_GOAL_PROFILE] + app.GOAL_PROFILES
        earnings_index = next(
            index for index, profile in enumerate(profiles) if profile["key"] == "earnings_review"
        )

        def fake_embeddings(texts, *args, **kwargs):
            if kwargs["mode"] == "query":
                return [[1.0, 0.0]]
            vectors = [[0.0, 1.0] for _ in profiles]
            vectors[earnings_index] = [1.0, 0.0]
            return vectors

        app.embed_texts = fake_embeddings
        profile = app.infer_goal_profile_with_embedding("利润率和管理层展望出现变化")
        self.assertEqual(profile["key"], "earnings_review")
        self.assertEqual(profile["analysis_source"], "Local embedding semantic router")

    def test_explicit_regulatory_terms_can_change_a_near_tied_embedding_goal(self):
        profiles = [app.DEFAULT_GOAL_PROFILE] + app.GOAL_PROFILES
        hold_index = next(
            index for index, profile in enumerate(profiles) if profile["key"] == "hold_check"
        )
        regulation_index = next(
            index for index, profile in enumerate(profiles) if profile["key"] == "regulation_review"
        )

        def fake_embeddings(texts, *args, **kwargs):
            if kwargs["mode"] == "query":
                return [[1.0, 0.0]]
            vectors = [[0.0, 1.0] for _ in profiles]
            vectors[hold_index] = [1.0, 0.0]
            vectors[regulation_index] = [0.999, 0.045]
            return vectors

        app.embed_texts = fake_embeddings
        profile = app.infer_goal_profile_with_embedding(
            "长期持仓逻辑是否受监管机构调查影响"
        )
        self.assertEqual(profile["key"], "regulation_review")

    def test_risk_detection_uses_embedding_category_when_available(self):
        def fake_embeddings(texts, *args, **kwargs):
            if kwargs["mode"] == "passage":
                return [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2], [0.1, 0.1]]
            return [[0.0, 1.0]]

        app.embed_texts = fake_embeddings
        scored_news = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2026-08-26T10:00:00"),
                    "headline": "Unexpected margin compression worries investors",
                    "sentiment_score": -0.4,
                    "sentiment_label": "Negative",
                }
            ]
        ).set_index("datetime")
        risk = app.detect_risks(
            scored_news,
            snapshot={},
            goal_profile=app.profile_by_key("earnings_review"),
            use_semantic=True,
        )
        self.assertEqual(risk["classifier"], "local embedding semantic classifier")
        self.assertEqual(risk["findings"][0]["category"], "业绩/指引")
        self.assertEqual(risk["findings"][0]["classification_source"], "local embedding semantic classifier")


class DeepSeekSDKContractTest(unittest.TestCase):
    def setUp(self):
        self.original_openai = app.OpenAI
        self.original_client = app.get_deepseek_client
        self.original_embedder = app.get_local_embedding_model

    def tearDown(self):
        app.OpenAI = self.original_openai
        app.get_deepseek_client = self.original_client
        app.get_local_embedding_model = self.original_embedder

    def test_deepseek_client_uses_compatibility_base_url(self):
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return object()

        app.OpenAI = fake_openai
        app.get_deepseek_client("deepseek-test-key")

        self.assertEqual(captured["api_key"], "deepseek-test-key")
        self.assertEqual(captured["base_url"], app.DEEPSEEK_BASE_URL)

    def test_llm_call_uses_chat_completions_not_responses(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{\"action\": \"finish\"}"))],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        app.get_deepseek_client = lambda api_key=None: fake_client
        output = app.call_llm_text("system prompt", "user prompt", "deepseek-v4-flash", api_key="key")

        self.assertEqual(output, '{"action": "finish"}')
        self.assertEqual(captured["model"], "deepseek-v4-flash")
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][1]["role"], "user")
        self.assertEqual(captured["extra_body"]["thinking"]["type"], "disabled")

    def test_local_embedding_uses_e5_query_prefix(self):
        captured = {}

        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                captured["texts"] = texts
                captured.update(kwargs)
                return SimpleNamespace(tolist=lambda: [[0.1, 0.2]])

        app.get_local_embedding_model = lambda: FakeEmbedder()
        vectors = app.embed_texts(["embedding input"], mode="query")

        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(captured["texts"], ["query: embedding input"])
        self.assertTrue(captured["normalize_embeddings"])


class SupervisorPlannerTest(unittest.TestCase):
    def setUp(self):
        self.original_client = app.get_deepseek_client
        self.original_json_call = app.call_llm_json
        app.get_deepseek_client = lambda api_key=None: object()
        app.call_llm_json = lambda *args, **kwargs: {
            "action": "fetch_sec_filings",
            "reason": "财报问题应先取一手披露。",
        }

    def tearDown(self):
        app.get_deepseek_client = self.original_client
        app.call_llm_json = self.original_json_call

    def test_llm_can_choose_sec_as_the_first_tool_from_complete_manifest(self):
        state = {
            "ticker": "TEST",
            "mission": "财报指引有没有改变长期逻辑？",
            "goal_profile": app.DEFAULT_GOAL_PROFILE,
            "source_note": "Evidence has not been collected yet.",
            "snapshot": {},
            "news_df": None,
            "scored_news": None,
            "summary": None,
            "risk": None,
            "sec_filings": None,
            "sec_attempted": False,
            "memo": None,
            "memo_version": 0,
            "reviewed_memo_version": 0,
            "revision_requested": False,
            "critic": None,
            "tool_steps": 0,
            "is_follow_up": False,
            "pending_followup_review": False,
            "messages": [{"role": "user", "content": "财报指引有没有改变长期逻辑？"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = app.TraceLogger(trace_dir=directory)
            decision = app.decide_next_action(
                state, use_llm=True, model="test-model", trace=trace, api_key="test-key"
            )

        self.assertEqual(decision["action"], "fetch_sec_filings")
        planner_event = next(event for event in trace.events if event["event"] == "planner_decision")
        manifest = planner_event["input"]["tool_manifest"]
        self.assertTrue({"collect_news", "fetch_sec_filings", "analyze_sentiment", "assess_risk", "draft_memo", "self_check"}.issubset(manifest))

    def test_rate_limit_falls_back_and_disables_further_llm_calls_for_this_run(self):
        class SyntheticRateLimitError(Exception):
            status_code = 429

        def raise_rate_limit(*args, **kwargs):
            raise SyntheticRateLimitError("synthetic quota failure")

        app.call_llm_json = raise_rate_limit
        state = {
            "ticker": "TEST",
            "mission": "判断这只股票近期是否有值得记录的风险信号",
            "goal_profile": app.DEFAULT_GOAL_PROFILE,
            "source_note": "Evidence has not been collected yet.",
            "snapshot": {},
            "news_df": None,
            "scored_news": None,
            "summary": None,
            "risk": None,
            "sec_filings": None,
            "sec_attempted": False,
            "memo": None,
            "memo_version": 0,
            "reviewed_memo_version": 0,
            "revision_requested": False,
            "critic": None,
            "tool_steps": 0,
            "is_follow_up": False,
            "pending_followup_review": False,
            "messages": [{"role": "user", "content": "判断风险"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = app.TraceLogger(trace_dir=directory)
            decision = app.decide_next_action(
                state, use_llm=True, model="test-model", trace=trace, api_key="test-key"
            )

        self.assertEqual(decision["action"], "collect_news")
        self.assertEqual(decision["decision_source"], "LLM error fallback")
        self.assertIn("429", state["llm_disabled_reason"])
        self.assertEqual(len(state["warnings"]), 1)
        self.assertTrue(any(event["event"] == "llm_runtime_disabled" for event in trace.events))


if __name__ == "__main__":
    unittest.main()

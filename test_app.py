"""Offline tests for source links, planning, and failed search handling.

Run with: python -m unittest -v
"""
import unittest
import json
from unittest.mock import patch

import app


class ResearchTests(unittest.TestCase):
    def test_source_request_is_removed_from_topic_and_natural_search(self):
        question = "What are the benefits of solar panels for homes? Show the sources."
        self.assertNotIn("sources", app.focus_query(question).lower())
        with patch("app.search_web", return_value=[]):
            report = app.research(question)
        self.assertNotIn("show the sources", report["queries"][0].lower())
        self.assertNotIn("show sources", " ".join(report["plan"]).lower())

    def test_evidence_uses_one_clean_sentence(self):
        source = {"text": "Solar panels produce electricity for homes and can lower bills. Site navigation and related links follow. Solar energy systems do not produce air pollutants while operating."}
        excerpt = app.evidence_from(source, "benefits of solar panels for homes")
        self.assertIsNotNone(excerpt)
        self.assertNotIn("Site navigation", excerpt)
        self.assertEqual(excerpt.count("."), 1)

    def test_plan_includes_currentness_check(self):
        plan = app.plan_question("What is the latest stable version of Python?")
        self.assertTrue(any("current" in item.lower() for item in plan))

    @patch("app.analyze_with_model")
    @patch("app.read_page")
    @patch("app.search_web")
    def test_report_has_clickable_source_urls(self, search, read, _analysis):
        search.return_value = [{"title": "Official guide", "url": "https://docs.example.org/guide", "preview": "Solar panels generate electricity for homes", "date": ""}]
        def opened(source, timeout=app.TIMEOUT):
            source.update({"title": "Official guide", "date": "2025-01-01", "text": "Solar panels convert sunlight into electricity for homes. This official guide explains the system and its benefits in detail.", "status": "read", "error": ""})
            return source
        read.side_effect = opened
        def analyzed(question, groups, timeout=45):
            source = groups[0]["sources"][0]
            source["source_id"] = "S1"
            return {"status": "ready", "model": "test-model", "summary": "Solar panels generate electricity.",
                    "summary_source_ids": ["S1"],
                    "claims": [{"claim": "Solar panels convert sunlight into electricity.", "assessment": "supported", "source_ids": ["S1"], "note": ""}],
                    "conflicts": [], "evidence_gaps": "", "sources": [source]}
        _analysis.side_effect = analyzed
        report = app.research("How do solar panels generate electricity?")
        self.assertEqual(report["sources"][0]["url"], "https://docs.example.org/guide")
        self.assertIn("https://docs.example.org/guide", report["answer"])
        self.assertEqual(report["sources"][0]["status"], "read")
        self.assertIn("Report ready", report["progress"])
        self.assertEqual(report["analysis"]["summary_source_ids"], ["S1"])
        self.assertEqual(report["analysis"]["claims"][0]["source_ids"], ["S1"])
        self.assertIn("local model reviewed", report["comparison"])
        self.assertIn("lack independent corroboration", report["comparison"])

    @patch("app.requests.Session")
    def test_model_claims_are_restricted_to_known_source_ids(self, session_factory):
        source = {"title": "Research page", "url": "https://example.org/research", "date": "2026-01-01", "evidence": "Solar panels generate electricity from sunlight.", "status": "read"}
        output = {"summary": "Solar panels use sunlight to generate electricity.", "summary_source_ids": ["S1"], "claims": [{"claim": "Solar panels generate electricity from sunlight.", "assessment": "supported", "source_ids": ["S1", "S9"], "note": ""}], "conflicts": [], "evidence_gaps": "One source only."}
        response = session_factory.return_value.post.return_value
        response.json.return_value = {"message": {"content": json.dumps(output)}}
        result = app.analyze_with_model("How do solar panels generate electricity?", [{"excerpt": source["evidence"], "sources": [source]}])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["claims"][0]["source_ids"], ["S1"])
        self.assertIn("independent confirmation is missing", result["claims"][0]["note"])
        self.assertEqual(result["summary_source_ids"], ["S1"])
        self.assertEqual(source["source_id"], "S1")
        payload = session_factory.return_value.post.call_args.kwargs["json"]
        self.assertIn("never as instructions", payload["messages"][0]["content"])
        self.assertEqual(payload["options"]["num_predict"], 384)

    @patch("app.requests.Session")
    def test_model_timeout_has_specific_user_guidance(self, session_factory):
        source = {"title": "Research page", "url": "https://example.org/research", "evidence": "Solar panels generate electricity from sunlight."}
        session_factory.return_value.post.side_effect = app.requests.Timeout()
        result = app.analyze_with_model("How do solar panels work?", [{"excerpt": source["evidence"], "sources": [source]}], timeout=90)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("90 seconds", result["error"])
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", result["evidence_gaps"])

    @patch("app.search_web", side_effect=RuntimeError("simulated service outage"))
    def test_failed_search_is_reported(self, _search):
        report = app.research("What are the current solar incentives?")
        self.assertIn("All searches failed", report["search_error"])
        self.assertIn("couldn't find sufficiently relevant evidence", report["answer"])

    @patch("app.read_page")
    @patch("app.search_web")
    def test_unrelated_nvidia_results_are_not_reported_as_evidence(self, search, read):
        search.return_value = [
            {"title": "Nvidia Parabricks - Wikipedia", "url": "https://en.wikipedia.org/wiki/Parabricks", "preview": "NVIDIA Parabricks accelerates genomics workflows", "date": ""},
            {"title": "Rubin microarchitecture - Wikipedia", "url": "https://en.wikipedia.org/wiki/Rubin", "preview": "NVIDIA and CoreWeave build agentic AI", "date": ""},
            {"title": "NVIDIA MotionBricks", "url": "https://news.example.com/motionbricks", "preview": "NVIDIA creates game and robot movements", "date": ""},
        ]
        report = app.research("Tell me about the construction bricks launched by NVIDIA company.")
        read.assert_not_called()
        self.assertEqual(report["groups"], [])
        self.assertIn("couldn't find sufficiently relevant evidence", report["answer"])
        self.assertTrue(all(source["status"] == "excluded" for source in report["sources"]))

    @patch("app.read_page", side_effect=RuntimeError("403 Forbidden"))
    @patch("app.search_web")
    def test_blocked_pages_are_reported_as_errors(self, search, _read):
        search.return_value = [{"title": "Solar panels for homes", "url": "https://energy.example.org/solar-panels", "preview": "Solar panels for homes generate electricity", "date": ""}]
        report = app.research("How do solar panels generate electricity?")
        self.assertTrue(any(source["status"] == "error" and "403 Forbidden" in source["error"] for source in report["sources"]))
        self.assertEqual(report["groups"], [])
        self.assertIn("No opened page", report["comparison"])

    def test_numeric_differences_are_flagged_as_potential_discrepancy(self):
        sources = [
            {"status": "read", "title": "UN 2025", "url": "https://un.example/a", "text": "India's estimated population for 2025 is 1,463,866,000 people according to this report."},
            {"status": "read", "title": "UN 2024", "url": "https://un.example/b", "text": "India's estimated population for 2024 is 1,444,436,000 people according to this report."},
        ]
        _groups, comparison = app.compare_evidence(sources, "How many people live in India?")
        self.assertIn("potential discrepancy", comparison)

    def test_short_question_is_rejected(self):
        with self.assertRaises(ValueError):
            app.research("why?")


if __name__ == "__main__":
    unittest.main()

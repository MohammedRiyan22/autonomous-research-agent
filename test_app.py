"""Offline tests for source links, planning, and failed search handling.

Run with: python -m unittest -v
"""
import unittest
import json
import io
from unittest.mock import patch

import app


class ResearchTests(unittest.TestCase):
    @patch("web_interface.threading.Thread")
    @patch("web_interface.run_job")
    def test_research_post_route_queues_job_without_server_error(self, _run_job, thread):
        handler = object.__new__(app.Handler)
        handler.path = "/api/research"
        body = json.dumps({"question": "How does solar power work?"}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        captured = {}

        def capture_response(instance, status, response_body, content_type=""):
            captured.update(status=status, body=response_body, content_type=content_type)

        with patch.object(app.Handler, "send", capture_response):
            handler.do_POST()

        self.assertEqual(captured["status"], 202)
        self.assertIn(b"job_id", captured["body"])
        thread.return_value.start.assert_called_once()

    def test_source_selection_prefers_distinct_domains(self):
        sources = [
            {"url": "https://same.example/a", "title": "Top result"},
            {"url": "https://same.example/b", "title": "Second result"},
            {"url": "https://other.example/c", "title": "Different publisher"},
        ]
        selected = app.select_diverse_sources(sources, limit=2)
        self.assertEqual([item["title"] for item in selected], ["Top result", "Different publisher"])

    def test_source_request_is_removed_from_topic_and_natural_search(self):
        question = "What are the benefits of solar panels for homes? Show the sources."
        self.assertNotIn("sources", app.focus_query(question).lower())
        with patch("research_pipeline.search_web", return_value=[]):
            report = app.research(question)
        self.assertNotIn("show the sources", report["queries"][0].lower())
        self.assertNotIn("show sources", " ".join(report["plan"]).lower())

    def test_which_is_framing_is_removed_from_search_topic_and_plan(self):
        question = "Which is the latest bath soap launched by NVIDIA for cleaning its GPUs?"
        focused = app.focus_query(question).lower()
        plan = " ".join(app.plan_question(question)).lower()
        self.assertNotIn("which is", focused)
        self.assertNotIn("which is", plan)

    def test_evidence_uses_one_clean_sentence(self):
        source = {"text": "Solar panels produce electricity for homes and can lower bills. Site navigation and related links follow. Solar energy systems do not produce air pollutants while operating."}
        excerpt = app.evidence_from(source, "benefits of solar panels for homes")
        self.assertIsNotNone(excerpt)
        self.assertNotIn("Site navigation", excerpt)
        self.assertEqual(excerpt.count("."), 1)

    def test_page_title_context_allows_relevant_passage_with_different_question_framing(self):
        source = {
            "title": "Safest New Cars of 2026, According to IIHS",
            "text": "The vehicles listed below do a good job of protecting both front- and rear-seat occupants in various kinds of crashes, and they feature technology that can prevent collisions with other cars and pedestrians.",
        }
        excerpt = app.evidence_from(source, "latest cars that are safest for travelling")
        self.assertIsNotNone(excerpt)
        self.assertIn("protecting", excerpt)

    def test_freshness_word_in_question_does_not_make_unrelated_sentence_evidence(self):
        source = {
            "title": "Safest Cars for 2026",
            "text": "It is calculated based on the latest research and analysis by our data science team.",
        }
        self.assertIsNone(app.evidence_from(source, "latest cars that are safest for travelling"))

    def test_author_biography_is_not_selected_as_topic_evidence(self):
        source = {
            "title": "Safest New Cars of 2026, According to IIHS",
            "text": "Keith is an auto reporter. He focuses on safety, technology, and the environmental impact of cars. The vehicles listed below do a good job of protecting occupants in crashes and feature technology that can prevent collisions.",
        }
        excerpt = app.evidence_from(source, "latest cars that are safest for travelling")
        self.assertIsNotNone(excerpt)
        self.assertNotIn("He focuses", excerpt)

    def test_model_selected_author_biography_cannot_override_page_passage_filter(self):
        source = {
            "title": "Safest New Cars of 2026, According to IIHS",
            "url": "https://example.org/safest-cars",
            "status": "read",
            "text": "He focuses on safety, technology, and the environmental impact of cars. The vehicles listed below do a good job of protecting occupants in crashes and feature technology that can prevent collisions.",
            "semantic_evidence": "He focuses on safety, technology, and the environmental impact of cars.",
        }
        groups, _comparison = app.compare_evidence([source], "latest cars that are safest for travelling")
        self.assertEqual(len(groups), 1)
        self.assertNotIn("He focuses", groups[0]["excerpt"])

    @patch("sources.requests.get")
    def test_page_reader_uses_page_specific_timeout(self, get):
        response = get.return_value
        response.headers = {"content-type": "text/html; charset=utf-8"}
        response.content = (b"<html><head><title>Safety ratings</title></head><body><main><p>"
                            b"This page explains vehicle safety ratings and crash protection in enough readable detail."
                            b"</p></main></body></html>")
        source = {"title": "Safety ratings", "url": "https://cars.example/safety"}
        app.read_page(source)
        self.assertEqual(get.call_args.kwargs["timeout"], app.PAGE_TIMEOUT)

    @patch("model_review.requests.Session")
    def test_semantic_passage_selection_returns_only_exact_source_text(self, session_factory):
        sentence = "Crash test ratings identify vehicles that provide strong protection for drivers in common collisions."
        source = {"title": "Vehicle safety ratings", "url": "https://safety.example/ratings",
                  "status": "read", "text": sentence}
        response = session_factory.return_value.post.return_value
        response.json.return_value = {"message": {"content": json.dumps({
            "selections": [{"source_key": "U1", "passage_ids": ["U1-1", "invented"]}]})}}
        from model_review import select_semantic_passages
        selected, error = select_semantic_passages(
            "Which cars are safest for travelling?", [source])
        self.assertEqual(error, "")
        self.assertEqual(selected[source["url"]], [sentence])

    def test_near_copied_passages_are_grouped_across_domains(self):
        sentence = "Solar panels provide financial benefits for homes by reducing electricity bills over time."
        sources = [
            {"status": "read", "title": "Publisher A", "url": "https://a.example/story", "text": sentence},
            {"status": "read", "title": "Publisher B", "url": "https://b.example/story", "text": sentence},
        ]
        groups, _comparison = app.compare_evidence(sources, "What are the benefits of solar panels for homes?")
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["sources"]), 2)
        self.assertEqual(groups[0]["similar_wording_domains"], 2)

    def test_plan_includes_currentness_check(self):
        plan = app.plan_question("What is the latest stable version of Python?")
        self.assertTrue(any("current" in item.lower() for item in plan))

    def test_current_population_question_plans_a_freshness_check(self):
        plan = app.plan_question("How many people live in India? Show the dates if sources give different numbers.")
        self.assertTrue(any("most current" in item.lower() for item in plan))

    def test_current_year_population_result_outranks_historical_result(self):
        current_year = app.datetime.now().year
        current = {"title": f"India population estimate {current_year}", "preview": "India population estimate", "date": ""}
        historical = {"title": "India census population in 1901", "preview": "India population 1901", "date": ""}
        self.assertGreater(app.candidate_freshness(current, current_year),
                           app.candidate_freshness(historical, current_year))

    def test_current_population_accepts_recent_reference_year(self):
        question = "How many people live in India?"
        year = app.datetime.now().year
        group = {"excerpt": "Total Population (2025): 1,463,900,000", "sources": [
            {"title": "India population data", "date": ""}]}
        self.assertEqual(app.current_population_groups([group], question, year), [group])

    def test_embedding_similarity_uses_local_vectors(self):
        with patch("evidence._ollama_embedding", side_effect=[(1.0, 0.0), (0.8, 0.6)]):
            self.assertAlmostEqual(app.semantic_similarity("resident count", "population total"), 0.8)

    def test_population_extraction_is_not_hardcoded_to_india(self):
        canada = {"excerpt": "Canada population in 2025 was estimated at 41.5 million people.",
                  "sources": [{"title": "Canada population estimate", "url": "https://example.ca/population"}]}
        estimates = app.extract_population_estimates([canada], "How many people live in Canada?")
        self.assertEqual(len(estimates), 1)
        self.assertAlmostEqual(estimates[0]["value"], 41_500_000)
        self.assertEqual(app.country_from_question("What is the population of Australia?"), "Australia")

    def test_population_table_formats_are_extracted(self):
        unfpa = {"excerpt": "Total Population (2025): 1,463,900,000", "sources": [
            {"title": "India population - UNFPA", "source_id": "S1", "url": "https://unfpa.org/data"}]}
        world_bank = {"excerpt": "Population, total Most recent value (2025 billion) 1.46", "sources": [
            {"title": "India | Data", "source_id": "S2", "url": "https://data.worldbank.org/country/in"}]}
        estimates = app.extract_population_estimates([unfpa, world_bank])
        self.assertEqual([item["reference_year"] for item in estimates], [2025, 2025])
        self.assertAlmostEqual(estimates[0]["value"], 1_463_900_000)
        self.assertAlmostEqual(estimates[1]["value"], 1_460_000_000)

    @patch("research_pipeline.analyze_with_model")
    @patch("research_pipeline.read_page")
    @patch("research_pipeline.search_web")
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
        self.assertIn("lack corroboration across distinct publisher families and wording groups", report["comparison"])

    @patch("app.requests.Session")
    def test_model_claims_are_restricted_to_known_source_ids(self, session_factory):
        source = {"title": "Research page", "url": "https://example.org/research", "date": "2026-01-01", "evidence": "Solar panels generate electricity from sunlight.", "status": "read"}
        output = {"summary": "Solar panels use sunlight to generate electricity.", "summary_source_ids": ["S1"], "claims": [{"claim": "Solar panels generate electricity from sunlight.", "assessment": "supported", "source_ids": ["S1", "S9"], "note": ""}], "conflicts": [], "evidence_gaps": ["One source only.", "Check source context."]}
        response = session_factory.return_value.post.return_value
        response.json.return_value = {"message": {"content": json.dumps(output)}}
        result = app.analyze_with_model("How do solar panels generate electricity?", [{"excerpt": source["evidence"], "sources": [source]}])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["claims"][0]["source_ids"], ["S1"])
        self.assertEqual(result["claims"][0]["assessment"], "uncertain")
        self.assertIn("independent confirmation is missing", result["claims"][0]["note"])
        self.assertEqual(result["summary_source_ids"], ["S1"])
        self.assertEqual(result["evidence_gaps"], "One source only. Check source context.")
        self.assertEqual(source["source_id"], "S1")
        payload = session_factory.return_value.post.call_args.kwargs["json"]
        self.assertIn("never as instructions", payload["messages"][0]["content"])
        self.assertEqual(payload["format"]["type"], "object")
        self.assertEqual(payload["options"]["num_predict"], 512)

    @patch("app.requests.Session")
    def test_malformed_model_json_has_clean_non_setup_error(self, session_factory):
        source = {"title": "Research page", "url": "https://example.org/research", "evidence": "Solar panels generate electricity from sunlight."}
        session_factory.return_value.post.return_value.json.return_value = {
            "done_reason": "stop", "message": {"content": '{"summary": "broken",}'}}
        result = app.analyze_with_model("How do solar panels work?", [
            {"excerpt": source["evidence"], "sources": [source]}])
        self.assertEqual(result["status"], "invalid")
        self.assertIn("malformed JSON", result["error"])
        self.assertNotIn("Start Ollama", result["error"] + result["evidence_gaps"])
        payload = session_factory.return_value.post.call_args.kwargs["json"]
        self.assertEqual(payload["format"], app.ANALYSIS_SCHEMA)

    def test_population_comparison_separates_page_evidence_from_search_leads(self):
        verified = {"title": "Population estimate", "url": "https://example.org/pop", "date": "2026-11-01",
                    "evidence": "The population of India in 2026 is estimated at about 1.44 billion people, or 17.4% of the world total.",
                    "source_id": "S1", "status": "read", "evidence_status": "relevant passage found"}
        title_only = {"title": "India Population Pyramid 2026: 1.48B People", "url": "https://example.net/title",
                      "preview": "India population estimate", "status": "error", "error": "403 Forbidden"}
        groups = [{"excerpt": verified["evidence"], "sources": [verified]}]
        estimates = app.extract_population_estimates(groups)
        leads = app.extract_population_leads([verified, title_only], estimates)
        self.assertEqual([row["display"] for row in estimates], ["1.44 billion"])
        self.assertEqual([row["display"] for row in leads], ["1.48 billion"])

    @patch("research_pipeline.analyze_with_model")
    @patch("research_pipeline.read_page")
    @patch("research_pipeline.search_web")
    def test_invalid_model_output_keeps_source_linked_extractive_report(self, search, read, _analysis):
        search.return_value = [{"title": "Home solar guide", "url": "https://energy.example/solar",
                                "preview": "Solar panels generate electricity for homes", "date": ""}]
        def opened(source, timeout=app.TIMEOUT):
            source.update({"title": "Home solar guide", "date": "2025-01-01",
                           "text": "Solar panels generate electricity for homes from sunlight. This guide explains home solar systems and their basic operation in detail.",
                           "status": "read", "error": ""})
            return source
        read.side_effect = opened
        def invalid(_question, groups, timeout=45):
            for index, source in enumerate(groups[0]["sources"], start=1):
                source["source_id"] = f"S{index}"
            return {"status": "invalid", "error": "Malformed JSON", "claims": [], "sources": []}
        _analysis.side_effect = invalid
        report = app.research("How do solar panels generate electricity?")
        self.assertEqual(report["analysis"]["status"], "invalid")
        self.assertIn("source-reported findings", report["analysis"]["fallback_summary"])
        self.assertEqual(report["analysis"]["fallback_claims"][0]["source_ids"], ["S1"])
        self.assertIn("https://energy.example/solar", report["sources"][0]["url"])

    @patch("research_pipeline.analyze_with_model")
    @patch("research_pipeline.read_page")
    @patch("research_pipeline.search_web")
    def test_current_population_run_does_not_synthesize_historical_census_as_current(self, search, read, analyze):
        search.return_value = [{"title": "Demographics of India", "url": "https://history.example/india",
                                "preview": "India population from the 1901 census", "date": ""}]
        def opened(source, timeout=app.TIMEOUT):
            source.update({"title": "Demographics of India", "text": "India's population in the 1901 census was recorded in historical tables and these figures describe that period only.",
                           "status": "read", "error": ""})
            return source
        read.side_effect = opened
        report = app.research("How many people live in India? Show the dates if sources give different numbers.")
        analyze.assert_not_called()
        self.assertEqual(report["analysis"]["status"], "insufficient")
        self.assertTrue(any("2026" in query for query in report["queries"]))
        self.assertIn("1901", report["analysis"]["summary"])
        self.assertTrue(any("Skipping AI review" in step for step in report["progress"]))

    @patch("research_pipeline.search_web", side_effect=RuntimeError("DDGSException: No results found."))
    def test_empty_search_results_are_not_reported_as_service_errors(self, _search):
        report = app.research("What are the benefits of solar panels?")
        self.assertNotIn("DDGSException", report["search_error"])
        self.assertIn("no candidate pages", report["search_error"].lower())
        self.assertEqual(report["analysis"]["status"], "skipped")
        self.assertFalse(any("Checking claims with" in step for step in report["progress"]))
        self.assertTrue(any("Skipping AI review" in step for step in report["progress"]))

    @patch("app.requests.Session")
    def test_claim_comparison_requires_support_and_contradiction_citations(self, session_factory):
        sources = [
            {"title": "Agency report", "url": "https://agency.gov/report", "evidence": "Panels last about 20 years."},
            {"title": "University study", "url": "https://university.edu/study", "evidence": "Panels last about 30 years."},
        ]
        output = {"summary": "Published sources report different panel lifetimes.", "summary_source_ids": ["S1"],
                  "claims": [{"claim": "One source reports 20 years, while another reports 30 years.", "assessment": "conflicting",
                              "source_ids": ["S1", "S99"], "contradicting_source_ids": ["S2"], "note": "Different reported estimates."}],
                  "conflicts": ["Sources report different lifetimes."], "evidence_gaps": "Check methods."}
        session_factory.return_value.post.return_value.json.return_value = {"message": {"content": json.dumps(output)}}
        result = app.analyze_with_model("How long do panels last?", [
            {"excerpt": sources[0]["evidence"], "sources": [sources[0]]},
            {"excerpt": sources[1]["evidence"], "sources": [sources[1]]},
        ])
        claim = result["claims"][0]
        self.assertEqual(claim["assessment"], "conflicting")
        self.assertEqual(claim["source_ids"], ["S1"])
        self.assertEqual(claim["contradicting_source_ids"], ["S2"])

    @patch("app.requests.Session")
    def test_multi_domain_multi_wording_support_is_marked_as_proxy(self, session_factory):
        source_a = {"title": "Agency", "url": "https://agency.gov/a", "evidence": "Panels convert sunlight into household electricity."}
        source_b = {"title": "University", "url": "https://university.edu/b", "evidence": "Solar modules generate power for residential use."}
        output = {"summary": "Solar panels can generate electricity.", "summary_source_ids": ["S1", "S2"],
                  "claims": [{"claim": "Solar panels generate electricity.", "assessment": "supported",
                              "source_ids": ["S1", "S2"], "contradicting_source_ids": [], "note": ""}],
                  "conflicts": [], "evidence_gaps": ""}
        session_factory.return_value.post.return_value.json.return_value = {"message": {"content": json.dumps(output)}}
        result = app.analyze_with_model("How do solar panels generate electricity?", [
            {"excerpt": source_a["evidence"], "sources": [source_a]},
            {"excerpt": source_b["evidence"], "sources": [source_b]},
        ])
        claim = result["claims"][0]
        self.assertEqual(claim["assessment"], "supported")
        self.assertEqual(claim["corroboration"], "multiple-domain wording-diverse")
        self.assertIn("does not prove editorial independence", claim["note"])

    @patch("app.requests.Session")
    def test_copied_cluster_does_not_count_as_independent_corroboration(self, session_factory):
        passage = "Solar panels provide financial benefits for homes by reducing electricity bills over time."
        source_a = {"title": "Publisher A", "url": "https://a.example/story", "evidence": passage}
        source_b = {"title": "Publisher B", "url": "https://b.example/story", "evidence": passage}
        output = {"summary": "Solar panels can lower electricity bills.", "summary_source_ids": ["S1"],
                  "claims": [{"claim": "Solar panels reduce bills.", "assessment": "supported",
                              "source_ids": ["S1", "S2"], "contradicting_source_ids": [], "note": ""}],
                  "conflicts": [], "evidence_gaps": ""}
        session_factory.return_value.post.return_value.json.return_value = {"message": {"content": json.dumps(output)}}
        result = app.analyze_with_model("What are the benefits of solar panels for homes?", [
            {"excerpt": passage, "sources": [source_a, source_b]},
        ])
        self.assertEqual(result["claims"][0]["assessment"], "uncertain")
        self.assertEqual(result["claims"][0]["corroboration"], "not independently corroborated")
        self.assertIn("may be copied or syndicated", result["claims"][0]["note"])

    @patch("app.requests.Session")
    def test_model_timeout_has_specific_user_guidance(self, session_factory):
        source = {"title": "Research page", "url": "https://example.org/research", "evidence": "Solar panels generate electricity from sunlight."}
        session_factory.return_value.post.side_effect = app.requests.Timeout()
        result = app.analyze_with_model("How do solar panels work?", [{"excerpt": source["evidence"], "sources": [source]}], timeout=90)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("90 seconds", result["error"])
        self.assertIn("OLLAMA_TIMEOUT_SECONDS", result["evidence_gaps"])

    @patch("research_pipeline.search_web", side_effect=RuntimeError("simulated service outage"))
    def test_failed_search_is_reported(self, _search):
        report = app.research("What are the current solar incentives?")
        self.assertIn("All searches failed", report["search_error"])
        self.assertIn("couldn't find sufficiently relevant evidence", report["answer"])

    @patch("research_pipeline.read_page")
    @patch("research_pipeline.search_web")
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

    @patch("research_pipeline.read_page", side_effect=RuntimeError("403 Forbidden"))
    @patch("research_pipeline.search_web")
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
        self.assertIn("different reference years", comparison.lower())
        same_year_sources = [
            {"status": "read", "title": "UN 2025", "url": "https://un.example/a", "text": "India's estimated population for 2025 is 1,463,866,000 people according to this report."},
            {"status": "read", "title": "Agency 2025", "url": "https://agency.example/b", "text": "India population in 2025 was 1,410,000,000 people according to this estimate."},
        ]
        _groups, same_year_comparison = app.compare_evidence(same_year_sources, "How many people live in India?")
        self.assertIn("potential same-year population discrepancy", same_year_comparison.lower())

    def test_short_question_is_rejected(self):
        with self.assertRaises(ValueError):
            app.research("why?")


if __name__ == "__main__":
    unittest.main()

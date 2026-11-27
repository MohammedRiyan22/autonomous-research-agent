"""End-to-end research orchestration, persistence, and background jobs."""
from __future__ import annotations
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone

from evidence import (asks_for_quantity, compare_evidence, domain, extract_population_estimates,
                      extract_population_leads, has_recent_reference, relevance_score, source_class, tokens, country_from_question)
from model_review import analyze_with_model, select_semantic_passages
from planning import (candidate_freshness, current_population_groups, focus_query,
                      is_current_population_question, plan_question, population_target_year)
from settings import (DB_PATH, MAX_CROSSCHECK_PAGES, MAX_CROSSCHECK_SEARCHES, MAX_INITIAL_PAGES,
                      MAX_RUN_SECONDS, MAX_SEARCH_QUERIES, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, PAGE_TIMEOUT,
                      SOURCE_REQUEST, STOP, TIMEOUT)
from sources import domain as source_domain, publisher_key, read_page, search_web, select_diverse_sources

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()

def _crosscheck_query(group: dict) -> str:
    """Build a compact query from an evidence passage and its page title."""
    filler = {"source", "sources", "article", "page", "states", "says", "show", "shows", "reported",
              "reports", "research", "according", "listed", "below", "found", "using", "based", "also"}
    ordered = []
    seen = set()
    for text in [group.get("excerpt", ""), " ".join(s.get("title", "") for s in group.get("sources", []))]:
        for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", text.lower()):
            normalized = next(iter(tokens(raw)), "")
            if not normalized or normalized in STOP or normalized in filler or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(raw)
            if len(ordered) == 8:
                break
        if len(ordered) == 8:
            break
    return "independent source " + " ".join(ordered)

def _crosscheck_candidates(found: list[dict], finding: str, finding_number: int,
                          finding_publishers: set[str], seen_urls: set[str]) -> list[dict]:
    finding_words = tokens(finding)
    candidates = []
    for candidate in found:
        key = candidate["url"].split("#", 1)[0].rstrip("/").lower()
        publisher = publisher_key(candidate["url"])
        if key in seen_urls or publisher in finding_publishers:
            continue
        candidate_words = tokens(candidate.get("title", "") + " " + candidate.get("preview", ""))
        matches = len(finding_words & candidate_words)
        # Search results are only leads. Let the opened page and semantic matcher
        # decide whether a paraphrase supports the finding; a strict title/preview
        # overlap gate would discard valid wording-diverse sources too early.
        minimum = 1
        if matches < minimum:
            continue
        seen_urls.add(key)
        candidate["source_class"], candidate["source_priority"] = source_class(candidate["url"], finding)
        candidate["candidate_relevance"] = round(matches / max(1, len(finding_words)), 2)
        candidate["cross_check_finding"] = finding_number
        candidate["evidence_query"] = finding
        candidates.append(candidate)
    return candidates

def _select_crosscheck_sources(candidates: list[dict], limit: int) -> list[dict]:
    """Allocate cross-check pages across findings before opening a second per claim."""
    by_finding: dict[int, list[dict]] = {}
    for source in candidates:
        by_finding.setdefault(int(source.get("cross_check_finding", 0)), []).append(source)
    for rows in by_finding.values():
        rows.sort(key=lambda row: (row.get("candidate_relevance", 0), row.get("source_priority", 0)), reverse=True)
    selected, seen_publishers = [], set()
    finding_numbers = sorted(by_finding)
    while len(selected) < limit:
        added_this_round = False
        for finding_number in finding_numbers:
            rows = by_finding[finding_number]
            while rows:
                source = rows.pop(0)
                publisher = publisher_key(source.get("url", ""))
                if publisher in seen_publishers:
                    continue
                selected.append(source)
                seen_publishers.add(publisher)
                added_this_round = True
                break
            if len(selected) >= limit:
                break
        if not added_this_round:
            break
    return selected

def research(question: str, progress=lambda _msg: None) -> dict:
    question = question.strip()[:500]
    if len(question) < 5:
        raise ValueError("Enter a research question of at least five characters.")
    plan = plan_question(question)
    started = time.monotonic()
    def seconds_left() -> float:
        return max(0.1, MAX_RUN_SECONDS - (time.monotonic() - started))
    progress_steps = []
    def log_step(message: str) -> None:
        if time.monotonic() - started > MAX_RUN_SECONDS:
            raise TimeoutError(f"Research exceeded the {MAX_RUN_SECONDS}-second run limit")
        progress_steps.append(message)
        progress(message)
    log_step("Created a research plan")
    focused = focus_query(question)
    natural_query = SOURCE_REQUEST.sub(" ", question)
    natural_query = re.sub(r"\s+", " ", natural_query).strip(" .,;:!?-") or focused
    current_population = is_current_population_question(question)
    target_year = population_target_year(question)
    if current_population:
        country = country_from_question(question)
        country_label = country or "the requested country"
        queries = list(dict.fromkeys([
            natural_query,
            f'"{country_label}" population {target_year} United Nations World Population Prospects',
            f'"{country_label}" population {target_year} official statistics',
            f'"{country_label}" population estimate {target_year} independent source',
        ]))[:MAX_SEARCH_QUERIES]
    else:
        queries = list(dict.fromkeys([natural_query, focused, f"official primary sources {focused}", f"independent sources {focused}"]))[:MAX_SEARCH_QUERIES]
    # Current population questions get explicit reference-year searches, including official and independent sources.
    sources, search_error = [], ""
    log_step("Searching the web")
    errors = []
    for query_number, query in enumerate(queries, start=1):
        log_step(f"Searching query {query_number} of {len(queries)}")
        try:
            found = search_web(query, timeout=min(TIMEOUT, seconds_left()))
            sources.extend(found)
            log_step(f"Query {query_number} returned {len(found)} results")
        except Exception as exc:  # public service failures vary by network and provider
            if "no results found" in str(exc).lower():
                log_step(f"Query {query_number} returned no results")
                continue
            errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            log_step(f"Query {query_number} failed: {type(exc).__name__}")
    if errors and len(errors) == len(queries):
        search_error = "All searches failed: " + "; ".join(errors)
    elif errors:
        search_error = "Some searches failed; results from successful queries are shown. " + "; ".join(errors)
    if not sources and not search_error:
        search_error = "The search service returned no candidate pages for these queries. No pages were available to open; try again later or check the search connection."
    log_step(f"Found {len(sources)} candidate sources")
    unique, seen, opened = [], set(), []
    for src in sources:
        key = src["url"].split("#", 1)[0].rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            src["candidate_relevance"] = round(relevance_score(src.get("title", "") + " " + src.get("preview", ""), focused), 2)
            src["source_class"], src["source_priority"] = source_class(src["url"], focused)
            words = tokens(focused)
            min_match = 1 if len(words) <= 1 else 2
            matched = len(words & tokens(src.get("title", "") + " " + src.get("preview", "")))
            if matched >= min_match:
                unique.append(src)
            else:
                src.update({"status": "excluded", "error": "Search-result title and preview did not match enough key terms", "text": ""})
                opened.append(src)
    unique.sort(key=lambda src: (
        candidate_freshness(src, target_year) if current_population else 0,
        src.get("candidate_relevance", 0), src.get("source_priority", 0)
    ), reverse=True)
    selected_sources = select_diverse_sources(unique, MAX_INITIAL_PAGES)
    for n, source in enumerate(selected_sources, start=1):
        log_step(f"Opening source {n} of {len(selected_sources)}")
        try:
            opened.append(read_page(source, timeout=min(PAGE_TIMEOUT, seconds_left())))
        except Exception as exc:
            source.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:220]}", "text": ""})
            opened.append(source)
    log_step("Finding initial evidence passages")
    groups, _initial_comparison = compare_evidence(opened, focused)
    semantic_issue = ""
    unmatched_pages = [source for source in opened
                       if source.get("status") == "read" and not source.get("evidence")]
    if unmatched_pages:
        log_step(f"Using local semantic matching for {len(unmatched_pages)} pages without keyword matches")
        semantic_passages, semantic_issue = select_semantic_passages(
            focused, unmatched_pages, timeout=min(OLLAMA_TIMEOUT_SECONDS, seconds_left()))
        if semantic_issue:
            log_step(f"Semantic passage selection unavailable: {semantic_issue}")
        else:
            for source in unmatched_pages:
                passages = semantic_passages.get(source.get("url", ""), [])
                if passages:
                    source["semantic_evidence"] = passages[0]
                    source["evidence_status"] = "relevant passage found (exact sentence selected semantically)"
            groups, _semantic_comparison = compare_evidence(opened, focused)
            selected_count = sum(bool(source.get("semantic_evidence")) for source in unmatched_pages)
            log_step(f"Semantic matching selected exact passages from {selected_count} additional pages")
    crosscheck_groups = current_population_groups(groups, question, target_year) if current_population else groups
    targeted_errors = []
    targeted_candidates = []
    targeted_queries = []
    seen_candidate_urls = {source.get("url", "").split("#", 1)[0].rstrip("/").lower() for source in opened}
    for finding_number, group in enumerate(crosscheck_groups[:MAX_CROSSCHECK_SEARCHES], start=1):
        finding_publishers = {publisher_key(source["url"]) for source in group["sources"]}
        finding = group["excerpt"]
        cross_query = _crosscheck_query(group)
        if len(tokens(cross_query)) < 3:
            continue
        normalized = cross_query.casefold()
        if normalized in targeted_queries:
            continue
        targeted_queries.append(normalized)
        queries.append(cross_query)
        log_step(f"Searching independent publishers for finding {finding_number} of {min(len(crosscheck_groups), MAX_CROSSCHECK_SEARCHES)}")
        found = []
        attempted_fallback = False
        try:
            found = search_web(cross_query, timeout=min(TIMEOUT, seconds_left()))
        except Exception as exc:
            if "no results found" not in str(exc).lower():
                targeted_errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")
                log_step(f"Cross-check search for finding {finding_number} failed: {type(exc).__name__}")
        if not found:
            compact_query = " ".join(cross_query.split()[:6])
            if compact_query.casefold() != normalized and compact_query.casefold() not in targeted_queries:
                attempted_fallback = True
                targeted_queries.append(compact_query.casefold())
                queries.append(compact_query)
                log_step(f"Trying a shorter independent-source query for finding {finding_number}")
                try:
                    found = search_web(compact_query, timeout=min(TIMEOUT, seconds_left()))
                except Exception as exc:
                    if "no results found" not in str(exc).lower():
                        targeted_errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")
                        log_step(f"Short cross-check search for finding {finding_number} failed: {type(exc).__name__}")
        log_step(f"Cross-check search for finding {finding_number} returned {len(found)} results")
        eligible = _crosscheck_candidates(found, finding, finding_number, finding_publishers, seen_candidate_urls)
        if not eligible and not attempted_fallback:
            compact_query = " ".join(cross_query.split()[:5])
            if compact_query.casefold() not in targeted_queries:
                attempted_fallback = True
                targeted_queries.append(compact_query.casefold())
                queries.append(compact_query)
                log_step(f"No independent publisher matched; trying a broader query for finding {finding_number}")
                try:
                    fallback_results = search_web(compact_query, timeout=min(TIMEOUT, seconds_left()))
                    eligible = _crosscheck_candidates(fallback_results, finding, finding_number,
                                                      finding_publishers, seen_candidate_urls)
                    log_step(f"Broader cross-check search for finding {finding_number} returned {len(fallback_results)} results")
                except Exception as exc:
                    if "no results found" not in str(exc).lower():
                        targeted_errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")
                        log_step(f"Broader cross-check search for finding {finding_number} failed: {type(exc).__name__}")
        targeted_candidates.extend(eligible)

    targeted_candidates.sort(key=lambda source: (
        source.get("candidate_relevance", 0), source.get("source_priority", 0)
    ), reverse=True)
    cross_check_sources = _select_crosscheck_sources(targeted_candidates, MAX_CROSSCHECK_PAGES)
    for n, source in enumerate(cross_check_sources, start=1):
        log_step(f"Opening cross-check source {n} of {len(cross_check_sources)}")
        try:
            opened.append(read_page(source, timeout=min(PAGE_TIMEOUT, seconds_left())))
        except Exception as exc:
            source.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:220]}", "text": ""})
            opened.append(source)
    # Cross-check pages can support the same claim with different wording. Reuse
    # the exact-sentence semantic selector on those pages when lexical extraction
    # misses them; it returns source-copied sentences, never generated evidence.
    crosscheck_unmatched = [source for source in cross_check_sources
                            if source.get("status") == "read"
                            and not source.get("evidence")
                            and not source.get("semantic_evidence")]
    if crosscheck_unmatched:
        log_step(f"Semantically checking {len(crosscheck_unmatched)} cross-check pages for paraphrased evidence")
        semantic_passages, crosscheck_semantic_issue = select_semantic_passages(
            question, crosscheck_unmatched, timeout=min(OLLAMA_TIMEOUT_SECONDS, seconds_left()))
        if crosscheck_semantic_issue:
            log_step(f"Cross-check semantic matching unavailable: {crosscheck_semantic_issue}")
        else:
            for source in crosscheck_unmatched:
                passages = semantic_passages.get(source.get("url", ""), [])
                if passages:
                    source["semantic_evidence"] = passages[0]
                    source["evidence_status"] = "relevant passage found (exact sentence selected semantically)"
            selected_count = sum(bool(source.get("semantic_evidence")) for source in crosscheck_unmatched)
            log_step(f"Semantic matching selected exact passages from {selected_count} cross-check pages")
    if targeted_errors:
        targeted_issue = "Some claim-focused cross-check searches failed: " + "; ".join(targeted_errors)
        search_error = (search_error + " " + targeted_issue).strip()
    log_step("Filtering passages and comparing similar wording")
    groups, comparison = compare_evidence(opened, focused)
    model_groups = current_population_groups(groups, question, target_year) if current_population else groups
    if current_population and not model_groups:
        old_years = sorted({int(year) for group in groups
                            for year in re.findall(r"\b(?:18|19|20)\d{2}\b", group.get("excerpt", ""))
                            if int(year) < target_year - 3})
        has_percentage_only = any(re.search(r"%|\bpercent(?:age)?\b", group.get("excerpt", ""), re.I) for group in groups)
        historical_note = f" The oldest matched reference is {old_years[-1]}." if old_years else ""
        percent_note = " Some passages give a share of world population, not an absolute count." if has_percentage_only else ""
        analysis = {"status": "insufficient", "error": "No recent absolute population figure was found in the opened passages.",
                    "summary": (f"I could not verify India's population for {target_year} from the pages opened." + historical_note + percent_note),
                    "claims": [], "summary_source_ids": [], "conflicts": [],
                    "evidence_gaps": f"No readable passage provided an absolute population estimate for {target_year}.", "sources": []}
        log_step("Skipping AI review: no recent absolute population estimate in the opened passages")
    elif not model_groups:
        if not sources:
            reason = "No candidate pages were returned by the web searches, so there is no source evidence for the model to review."
        elif not any(source.get("status") == "read" for source in opened):
            reason = "No source page could be read, so there is no page evidence for the model to review."
        else:
            reason = "Readable pages were found, but none yielded a relevant evidence passage for review."
        analysis = {"status": "skipped", "error": reason,
                    "summary": "I could not find enough readable, relevant source evidence to answer this question.",
                    "claims": [], "summary_source_ids": [], "conflicts": [],
                    "evidence_gaps": reason, "sources": []}
        log_step("Skipping AI review: no answer-relevant evidence passages are available")
    else:
        log_step(f"Checking claims with {OLLAMA_MODEL} via local Ollama")
        analysis = analyze_with_model(question, model_groups, timeout=min(OLLAMA_TIMEOUT_SECONDS, seconds_left()))
    hide_related_passages = bool(analysis.get("unverified_subject")) or (
        analysis.get("status") == "insufficient" and not asks_for_quantity(question)
    )
    if hide_related_passages:
        subject = analysis.get("unverified_subject", "")
        for source in opened:
            if source.get("status") == "read" and source.get("source_id"):
                reason = (f"does not identify '{subject}' in the claimed context" if subject
                          else "does not directly answer the question")
                source["evidence_status"] = f"not used as evidence: passage {reason}"
                source.pop("evidence", None)
                source.pop("semantic_evidence", None)
        groups = []
        analysis["related_passages_suppressed"] = True
        comparison = "No opened passage directly answered the question, so related search results were excluded from the evidence passages."
        log_step("Hiding related passages that do not answer the question")
    population_estimates = extract_population_estimates(groups, question) if asks_for_quantity(question) and re.search(
        r"\bpopulation(?: of| size)?\b|how many people (?:live|reside) in\b", question, re.I
    ) else []
    population_leads = extract_population_leads(opened, population_estimates, question) if current_population else []
    if analysis.get("status") != "ready":
        fallback_claims = []
        if analysis.get("status") == "insufficient":
            fallback_summary = analysis.get("summary", "The retrieved evidence does not support a reliable answer.")
        elif population_estimates:
            fallback_summary = ("Opened source pages report these population estimates: " +
                                "; ".join(f"{item['display']}" + (f" for {item['reference_year']}" if item["reference_year"] else "")
                                          for item in population_estimates) +
                                ". These are source-reported figures, not a single reconciled or independently verified count.")
            fallback_ids = list(dict.fromkeys(item["source_id"] for item in population_estimates if item.get("source_id")))
            for item in population_estimates:
                if item.get("source_id"):
                    reference = f" for reference year {item['reference_year']}" if item.get("reference_year") else ""
                    fallback_claims.append({"claim": f"A retrieved page reports India's population as {item['display']}{reference}.",
                        "assessment": "source-reported", "source_ids": [item["source_id"]],
                        "contradicting_source_ids": [], "note": "Extracted from the page passage; not independently verified."})
            analysis["fallback_summary_source_ids"] = fallback_ids
        elif groups and analysis.get("status") in {"invalid", "unavailable"}:
            fallback_summary = "The retrieved pages contain the source-reported findings below. They are presented as excerpts because the local model did not complete its review."
            for group in groups[:3]:
                ids = list(dict.fromkeys(source.get("source_id") for source in group.get("sources", []) if source.get("source_id")))
                if ids:
                    fallback_claims.append({"claim": group["excerpt"], "assessment": "source-reported",
                        "source_ids": ids, "contradicting_source_ids": [],
                        "note": "Directly extracted passage; not synthesized or independently confirmed."})
            analysis["fallback_summary_source_ids"] = list(dict.fromkeys(
                sid for claim in fallback_claims for sid in claim["source_ids"]))
        else:
            fallback_summary = "I could not find enough readable, relevant source evidence to answer this question."
        analysis["fallback_summary"] = fallback_summary
        analysis["fallback_claims"] = fallback_claims
    if analysis.get("status") == "ready":
        if analysis.get("conflicts"):
            comparison = "The AI review found a possible conflict between the cited passages. Check the passage context, dates, and methods before treating it as a true disagreement."
        elif "potential discrepancy" in comparison.lower():
            comparison = "The simple number check found differing values. The AI review did not mark a direct conflict; compare the dates, units, and methods before interpreting the difference."
        elif groups:
            claims = analysis.get("claims", [])
            weakly_corroborated = sum(1 for claim in claims if claim.get("corroboration") != "multiple-domain wording-diverse")
            if weakly_corroborated:
                comparison = (f"The local model reviewed the retrieved passages and did not flag a direct conflict. "
                              f"{weakly_corroborated} of {len(claims)} listed claims lack corroboration across distinct publisher families and wording groups in this run. "
                              f"The app ran {len(targeted_queries)} claim-focused search queries across up to {min(len(crosscheck_groups), MAX_CROSSCHECK_SEARCHES)} findings; no search can guarantee source independence. "
                              "Evidence about separate claims should not be treated as cross-confirmation.")
            else:
                comparison = "The local model reviewed the retrieved passages and did not flag a direct conflict. This is a model assessment, not proof of agreement or source independence; check each linked page."
    elif analysis.get("status") == "insufficient":
        if re.search(r"\bpopulation(?: of| size)?\b|how many people (?:live|reside) in\b", question, re.I) and not re.search(r"\b(?:18|19|20)\d{2}\b", question):
            has_share = any(re.search(r"%|\bpercent(?:age)?\b", group.get("excerpt", ""), re.I) for group in groups)
            has_historical_year = any(re.search(r"\b(?:18|19|20)\d{2}\b", group.get("excerpt", "")) for group in groups)
            reasons = []
            if has_historical_year:
                reasons.append("historical census references")
            if has_share:
                reasons.append("percentages of world population")
            context = " and ".join(reasons) if reasons else "related passages"
            comparison = f"No recent absolute population count was found. {context.capitalize()} do not establish India's current headcount."
        else:
            comparison = "The cross-check did not establish the requested claim. Related mentions are not confirmation."
    elif analysis.get("conflicts"):
        comparison = "The text and number checks found a possible discrepancy. The AI review did not complete, so compare the passage context, dates, and methods yourself."
    if len({round(item["value"]) for item in population_estimates}) > 1:
        years = {item["reference_year"] for item in population_estimates if item["reference_year"]}
        year_text = f" for reference year {', '.join(map(str, sorted(years)))}" if years else " with no clear reference year"
        values = ", ".join(dict.fromkeys(item["display"] for item in population_estimates))
        comparison = (f"Retrieved pages report different India population estimates{year_text}: {values}. "
                      "This is a reported difference, not a confirmed contradiction: the pages may use different estimate dates or methods. "
                      "The source-by-source comparison below shows the linked pages and publication dates when available.")
    answer = "\n\n".join(f"{g['excerpt']} " + " ".join(f"[{s['title']}]({s['url']})" for s in g["sources"])
                          for g in groups[:4])
    if not answer:
        answer = "I couldn't find sufficiently relevant evidence for this question. The pages found may mention related words without answering it."
    log_step("Report ready")
    return {"question": question, "plan": plan, "queries": queries, "sources": opened,
            "groups": groups, "answer": answer, "comparison": comparison,
            "population_estimates": population_estimates,
            "population_leads": population_leads,
            "analysis": analysis,
            "progress": progress_steps,
            "semantic_issue": semantic_issue,
            "uncertainty": "Passages are first matched by key terms; for unmatched pages, the local model may select exact sentences by meaning. Selection can still miss context and cannot establish source authority, copied reporting, or factual truth. Check the linked pages before relying on a claim.",
            "search_error": search_error, "created_at": datetime.now(timezone.utc).isoformat()}

def save_run(report: dict) -> None:
    with sqlite3.connect(DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL, report_json TEXT NOT NULL)")
        db.execute("INSERT INTO runs(created_at,question,report_json) VALUES(?,?,?)",
                   (report["created_at"], report["question"], json.dumps(report, ensure_ascii=False)))

def run_job(job_id: str, question: str) -> None:
    def progress(step: str) -> None:
        with JOB_LOCK:
            JOBS[job_id]["steps"].append(step)
    try:
        report = research(question, progress=progress)
        save_run(report)
        with JOB_LOCK:
            JOBS[job_id].update({"status": "complete", "report": report})
    except Exception as exc:
        with JOB_LOCK:
            JOBS[job_id].update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

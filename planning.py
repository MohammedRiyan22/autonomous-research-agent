"""Question cleanup, research planning, and freshness selection."""
from __future__ import annotations
from datetime import datetime
import re

from evidence import country_from_question, extract_population_estimates, has_recent_reference
from settings import GENERIC_QUESTION, SOURCE_REQUEST

def is_current_population_question(question: str) -> bool:
    asks_population = bool(re.search(r"\bpopulation\b|how many people (?:live|reside) in\b", question, re.I))
    asks_specific_year = bool(re.search(r"\b(?:18|19|20)\d{2}\b", question))
    return asks_population and not asks_specific_year

def population_target_year(question: str) -> int:
    match = re.search(r"\b(?:18|19|20)\d{2}\b", question)
    return int(match.group()) if match else datetime.now().year

def candidate_freshness(source: dict, target_year: int) -> int:
    """Prefer sources that appear to discuss the requested period over historical pages."""
    text = f"{source.get('title', '')} {source.get('preview', '')}"
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text)]
    has_count = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:billion|bn|b|million|mn|m)\b|\b\d{1,3}(?:,\d{3}){2,}\b", text, re.I))
    if target_year in years:
        return 5 if has_count else 2
    if any(target_year - 3 <= year < target_year for year in years):
        return 4 if has_count else 1
    if years and max(years) < target_year - 3:
        return -3
    published = re.search(r"\b(?:19|20)\d{2}\b", source.get("date", ""))
    if published:
        year = int(published.group())
        if year >= target_year - 2:
            return 2
        if year < target_year - 3:
            return -2
    return 0

def current_population_groups(groups: list[dict], question: str, current_year: int) -> list[dict]:
    """Keep only absolute population figures tied to the target period for model review."""
    target_year = population_target_year(question)
    accepted = []
    for group in groups:
        if not extract_population_estimates([group], question):
            continue
        excerpt_years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", group.get("excerpt", ""))]
        if target_year != current_year:
            if target_year in excerpt_years:
                accepted.append(group)
            continue
        if target_year in excerpt_years:
            accepted.append(group)
            continue
        # For an implicit "current" question, accept a recent reference year
        # (for example 2025 when running in 2026), while still rejecting old
        # census figures and years that occur only as publication metadata.
        recent_context = {"passage": f"{group.get('excerpt', '')} { ' '.join(s.get('title', '') for s in group.get('sources', [])) }",
                          "date": max((s.get("date", "") for s in group.get("sources", [])), default="")}
        if has_recent_reference(recent_context, current_year):
            accepted.append(group)
    return accepted

def plan_question(question: str) -> list[str]:
    """Turn question intent into visible research subquestions."""
    q = re.sub(r"\s+", " ", question).strip().rstrip("?.! ")
    subject = focus_query(q)
    quoted = re.findall(r"[\"']([^\"']{2,80})[\"']", q)
    topic = f"{subject}" if subject else q
    if quoted:
        topic = f"{topic} (especially {', '.join(quoted)})"
    plan = [f"What entity or topic does the question refer to: {topic}?"]
    plan.append(f"What do primary or official sources say about {topic}?")
    plan.append(f"Can an independent source confirm the same subject and key details about {topic}?")
    if re.search(r"\b(latest|current|today|now|recent|202[4-9])\b", q, re.I) or is_current_population_question(q):
        plan.append(f"What are the publication or update dates, and which source is most current about {topic}?")
    if re.search(r"\b(how many|population|number|rate|percent|percentage)\b", q, re.I):
        plan.append(f"What figures are reported for {topic}, and what dates or methods explain differences?")
    if re.search(r"\b(benefit|risk|effect|impact|compare|versus| vs )\b", q, re.I):
        plan.append(f"What documented benefits, risks, or differences are reported for {topic}?")
    plan.append(f"Which parts of the answer about {topic} remain unsupported or unclear?")
    return list(dict.fromkeys(plan))

def focus_query(question: str) -> str:
    """Remove conversational framing while retaining the important topic terms."""
    focused = SOURCE_REQUEST.sub(" ", question)
    focused = GENERIC_QUESTION.sub(" ", focused)
    focused = re.sub(r"\bpeople\s+(?:live|reside)\s+in\b", "population in", focused, flags=re.I)
    focused = re.sub(r"\b(?:the|a|an|by|from|for|in|on|of|to|with|company)\b", " ", focused, flags=re.I)
    focused = re.sub(r"[^\w\s'-]", " ", focused)
    return re.sub(r"\s+", " ", focused).strip()

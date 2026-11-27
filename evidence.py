"""Passage extraction, relevance, numeric checks, and evidence grouping."""
from __future__ import annotations
from difflib import SequenceMatcher
from functools import lru_cache
import math
import re

import requests

from settings import OLLAMA_EMBED_MODEL, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL, STOP
from sources import domain, publisher_key

def source_class(url: str, query: str) -> tuple[str, int]:
    host = domain(url)
    if host.endswith(("unfpa.org", "worldbank.org", "population.un.org", "un.org")):
        return "International organization", 4
    if host.endswith(".gov") or ".gov." in host:
        return "Government domain", 3
    if host.endswith(".edu") or host.endswith(".ac.in") or ".edu." in host:
        return "Academic domain", 3
    host_terms = tokens(" ".join(host.split(".")[:-1]))
    if host_terms & tokens(query):
        return "Topic-matched domain (ownership unverified)", 2
    return "Other publisher", 1

def tokens(text: str) -> set[str]:
    words = set()
    aliases = {
        "advantage": "benefit", "advantages": "benefit", "benefits": "benefit",
        "saving": "benefit", "savings": "benefit",
        "safe": "safe", "safer": "safe", "safest": "safe", "safety": "safe",
        "vehicle": "car", "vehicles": "car", "automobile": "car", "automobiles": "car",
        "car": "car", "cars": "car",
        "homeowner": "home", "homeowners": "home", "household": "home",
        "residential": "home", "houses": "home",
        "photovoltaic": "panel", "photovoltaics": "panel", "module": "panel", "modules": "panel",
        "people": "population", "person": "population", "persons": "population",
        "population": "population", "populations": "population", "resident": "population",
        "residents": "population", "inhabitant": "population", "inhabitants": "population",
    }
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", text):
        word = re.sub(r"'s$", "", raw.lower())
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        word = aliases.get(word, word)
        if word not in STOP:
            words.add(word)
    return words

def relevance_score(text: str, query: str) -> float:
    words = tokens(query)
    return len(words & tokens(text)) / max(1, len(words))

def claim_passage_overlap(claim: str, passage: str) -> float:
    """Lexical sanity check that a cited excerpt actually discusses the claim."""
    claim_words = tokens(claim)
    if not claim_words:
        return 0.0
    return len(claim_words & tokens(passage)) / len(claim_words)

def _is_author_bio(sentence: str) -> bool:
    return bool(re.match(r"^(?:he|she|they)\s+(?:focus(?:es|ed)|previously|earned|wrote|reported|led)\b", sentence.strip(), re.I))

def asks_for_quantity(question: str) -> bool:
    return bool(re.search(
        r"\b(?:how many|how much|population(?: of| size)?|number of people|percentage|percent|rate|how long|how far)\b",
        question, re.I,
    ))

def quantity_values(text: str) -> list[float]:
    """Extract comparable counts while ignoring standalone years such as 1901."""
    text = re.sub(r"\b(?:18|19|20)\d{2}\b", " ", text)
    pattern = re.compile(
        r"(?<![\w])(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*)(?P<unit>billion|bn|b|million|mn|m|thousand|crore|lakh|people|persons|inhabitants|years?|months?|weeks?|days?|hours?|minutes?|seconds?|%|percent)?\b",
        re.I,
    )
    scales = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6,
              "thousand": 1e3, "crore": 1e7, "lakh": 1e5}
    values = []
    for match in pattern.finditer(text):
        raw = match.group("number").replace(",", "")
        unit = (match.group("unit") or "").lower()
        try:
            number = float(raw)
        except ValueError:
            continue
        if unit in {"%", "percent"}:
            continue
        if unit in scales:
            number *= scales[unit]
        elif number < 10000 and unit not in {"year", "years", "month", "months", "week", "weeks", "day", "days", "hour", "hours", "minute", "minutes", "second", "seconds"}:
            continue
        values.append(number)
    return values

def has_recent_reference(source: dict, current_year: int) -> bool:
    passage = source.get("passage", "")
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", passage)]
    if any(current_year - 3 <= year <= current_year for year in years):
        return True
    if any(year < current_year - 3 for year in years):
        return False
    published = re.search(r"\b(?:19|20)\d{2}\b", source.get("date", ""))
    return bool(published and current_year - 2 <= int(published.group()) <= current_year)

def country_from_question(question: str) -> str | None:
    """Extract an explicitly named country from common population-question phrasing."""
    match = re.search(r"\b(?:population\s+of|people\s+(?:live|reside)\s+in|population\s+in|in)\s+([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3})\b", question, re.I)
    if not match:
        return None
    candidate = match.group(1).strip(" ?.!,")
    # Avoid treating a trailing request phrase as part of a country name.
    candidate = re.split(r"\b(?:show|give|provide|include|today|currently|right now)\b", candidate, flags=re.I)[0].strip()
    return candidate or None


@lru_cache(maxsize=256)
def _ollama_embedding(text: str) -> tuple[float, ...] | None:
    """Return a local Ollama embedding; unavailable embedding models fail closed to lexical matching."""
    try:
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text[:4000]},
            timeout=min(OLLAMA_TIMEOUT_SECONDS, 20),
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings", [])
        if not vectors or not vectors[0]:
            return None
        return tuple(float(value) for value in vectors[0])
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def semantic_similarity(text: str, query: str) -> float | None:
    """Cosine similarity from the configured local Ollama embedding model, if available."""
    left, right = _ollama_embedding(text), _ollama_embedding(query)
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else None


def extract_population_estimates(groups: list[dict], question: str = "") -> list[dict]:
    """Extract absolute estimates when the country or population label is in a page title/table header."""
    number = r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    unit = r"(?P<unit>billion|bn|b|million|mn|m|thousand|crore|lakh|people|persons|inhabitants)"
    country = country_from_question(question)
    reverse_unit = re.compile(r"\bpopulation\b.{0,55}?\b" + unit + r"\b[\s):]*(?:most recent value)?\s*" + number, re.I)
    number_then_unit = re.compile(r"\bpopulation\b.{0,100}?" + number + r"\s*" + unit + r"\b", re.I)
    raw_count = re.compile(r"\bpopulation\b.{0,55}?\b" + number + r"\b", re.I)
    scales = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6,
              "thousand": 1e3, "crore": 1e7, "lakh": 1e5}
    estimates = []
    seen = set()
    for group in groups:
        excerpt = group.get("excerpt", "")
        sources = group.get("sources", [])
        source_context = " ".join(source.get("title", "") for source in sources)
        context = f"{source_context} {excerpt}"
        if not re.search(r"\bpopulation\b", context, re.I):
            continue
        if country and not re.search(r"\b" + re.escape(country) + r"\b", context, re.I):
            continue
        no_years = re.sub(r"\b(?:18|19|20)\d{2}\b", " ", context)
        match = reverse_unit.search(no_years) or number_then_unit.search(no_years)
        display = ""
        if match:
            unit_name = match.group("unit").lower()
            raw_number = match.group("number")
            value = float(raw_number.replace(",", "")) * scales.get(unit_name, 1)
            display = f"{raw_number} {unit_name}"
        else:
            match = raw_count.search(no_years)
            if not match:
                continue
            raw_number = match.group("number")
            value = float(raw_number.replace(",", ""))
            if value < 1e8:  # reject ages, percentages, and unrelated small values
                continue
            display = f"{raw_number} people"
        ref_years = re.findall(r"\b(?:19|20)\d{2}\b", excerpt) or re.findall(r"\b(?:19|20)\d{2}\b", source_context)
        ref_year = max(map(int, ref_years)) if ref_years else None
        for source in sources:
            key = (round(value), source.get("url"))
            if key in seen:
                continue
            seen.add(key)
            estimates.append({"value": value, "display": display,
                              "reference_year": ref_year, "source_id": source.get("source_id"),
                              "title": source.get("title", "Source"), "url": source.get("url", ""),
                              "date": source.get("date", "")})
    return estimates

def extract_population_leads(sources: list[dict], estimates: list[dict], question: str = "") -> list[dict]:
    """Keep numbers seen only in titles/previews visible, but separate from read-page evidence."""
    verified_urls = {item["url"] for item in estimates}
    pattern = re.compile(r"(?<![\w.])(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>billion|bn|b|million|mn|m)\b", re.I)
    leads, seen = [], set()
    for source in sources:
        title = source.get("title", "")
        preview = source.get("preview", "")
        if source.get("url") in verified_urls or source.get("evidence_status") == "relevant passage found":
            continue
        context = title + " " + preview
        country = country_from_question(question)
        if not re.search(r"\bpopulation\b", context, re.I):
            continue
        if country and not re.search(r"\b" + re.escape(country) + r"\b", context, re.I):
            continue
        match = pattern.search(title + " " + preview)
        if not match:
            continue
        key = (source.get("url"), match.group("number"), match.group("unit").lower())
        if key in seen:
            continue
        seen.add(key)
        unit = match.group("unit").lower()
        unit = {"b": "billion", "bn": "billion", "m": "million", "mn": "million"}.get(unit, unit)
        leads.append({"display": f"{match.group('number')} {unit}",
                      "title": title, "url": source.get("url", ""),
                      "reason": source.get("evidence_status") or source.get("error") or "No matching passage was extracted"})
    return leads

def evidence_from(source: dict, query: str) -> str | None:
    """Select a relevant passage using lexical overlap plus optional local embeddings."""
    qwords = tokens(query)
    if not qwords:
        return None
    page_context = tokens(source.get("title", ""))
    title_overlap = qwords & page_context
    sentences = re.split(r"(?<=[.!?])\s+|\s+[•|]\s+|\n+", source.get("text", ""))
    ranked = []
    for sentence in sentences:
        sentence = sentence.strip()
        if _is_author_bio(sentence):
            continue
        sentence_words = tokens(sentence)
        sentence_overlap = qwords & sentence_words
        coverage = len(title_overlap | sentence_overlap) / len(qwords)
        minimum_sentence_terms = 1 if len(title_overlap) >= 2 else min(2, len(qwords))
        lexical_match = len(sentence_overlap) >= minimum_sentence_terms and coverage >= 0.4
        semantic_score = semantic_similarity(sentence, query) if 55 <= len(sentence) <= 520 and not lexical_match else None
        semantic_match = semantic_score is not None and semantic_score >= 0.48
        # Embedding similarity can rescue paraphrases; the title and minimum length
        # constraints reduce accidental matches. Lexical matching remains fallback.
        if 55 <= len(sentence) <= 520 and (lexical_match or semantic_match):
            combined = 0.65 * coverage + 0.35 * (semantic_score or 0.0)
            ranked.append((combined, len(sentence_overlap) / max(1, len(sentence_words) ** 0.45), sentence))
    ranked.sort(reverse=True)
    return ranked[0][2] if ranked else None

def compare_evidence(sources: list[dict], query: str) -> tuple[list[dict], str]:
    """Group only near-verbatim passages; never infer agreement from domain count alone."""
    results = []
    for source in sources:
        if source.get("status") != "read":
            continue
        semantic_excerpt = source.get("semantic_evidence", "")
        normalized_text = re.sub(r"\s+", " ", source.get("text", "")).strip()
        normalized_semantic = re.sub(r"\s+", " ", semantic_excerpt).strip()
        excerpt = (semantic_excerpt if normalized_semantic and normalized_semantic in normalized_text
                   and not _is_author_bio(semantic_excerpt)
                   else evidence_from(source, source.get("evidence_query") or query))
        source["relevance"] = round(relevance_score(source.get("title", "") + " " + source.get("text", ""), query), 2)
        if not excerpt:
            source["evidence_status"] = "excluded: no passage matched enough topic terms"
            continue
        source["evidence"] = excerpt
        source["evidence_status"] = "relevant passage found"
        results.append({"text": excerpt, "sources": [source], "terms": tokens(excerpt)})
    parents = list(range(len(results)))
    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for i, current in enumerate(results):
        for j in range(i + 1, len(results)):
            other = results[j]
            union_terms = current["terms"] | other["terms"]
            similarity = len(current["terms"] & other["terms"]) / max(1, len(union_terms))
            left_words = re.findall(r"[a-z0-9]+", current["text"].lower())
            right_words = re.findall(r"[a-z0-9]+", other["text"].lower())
            sequence_similarity = SequenceMatcher(None, left_words, right_words, autojunk=False).ratio()
            if similarity >= 0.72 and (sequence_similarity >= 0.68 or similarity >= 0.9):
                left, right = root(i), root(j)
                if left != right:
                    parents[right] = left
    components: dict[int, list[dict]] = {}
    for i, row in enumerate(results):
        components.setdefault(root(i), []).append(row)
    groups = []
    for component in components.values():
        component_sources = [source for row in component for source in row["sources"]]
        if not component_sources:
            continue
        uniq = {publisher_key(source["url"]) for source in component_sources}
        representative = max(component, key=lambda row: len(row["text"]))
        groups.append({"excerpt": representative["text"], "sources": component_sources,
                       "similar_wording_domains": len(uniq)})
    if len(groups) == 0:
        assessment = "No opened page contained a passage that matched enough of the question's key terms. The search results do not support an answer."
    elif re.search(r"\b(how many|population|number|rate|percent|percentage)\b", query, re.I):
        observations = []
        for group in groups:
            for source in group["sources"]:
                passage = source.get("evidence", "")
                requested_country = country_from_question(query)
                source_context = passage + " " + source.get("title", "")
                if requested_country and not re.search(r"\b" + re.escape(requested_country) + r"\b", source_context, re.I):
                    continue
                year_match = re.search(r"\b(?:19|20)\d{2}\b", passage)
                year = int(year_match.group()) if year_match else None
                # Only treat population-labelled values as comparable population observations.
                population_match = re.search(r"\bpopulation\b", passage, re.I)
                if not population_match:
                    continue
                nearby = passage[population_match.start():population_match.start() + 150]
                number_pattern = re.compile(r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>billion|bn|million|mn|thousand|crore|lakh|people|persons|inhabitants)?", re.I)
                for match in number_pattern.finditer(nearby):
                    raw = match.group("number")
                    value = float(raw.replace(",", ""))
                    if 1900 <= value <= 2100 or (value < 10000 and not match.group("unit")):
                        continue
                    unit = (match.group("unit") or "people").lower()
                    scale = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "thousand": 1e3, "crore": 1e7, "lakh": 1e5}.get(unit, 1)
                    definition = "urban" if re.search(r"\burban\b", passage, re.I) else "rural" if re.search(r"\brural\b", passage, re.I) else "working-age" if re.search(r"working[- ]age", passage, re.I) else "total-or-unspecified"
                    observations.append({"value": value * scale, "year": year, "unit": unit,
                                         "definition": definition, "source": source})
                    break
        comparable = [item for item in observations if item["year"] is not None]
        conflicts = []
        for index, left in enumerate(comparable):
            for right in comparable[index + 1:]:
                same_scope = left["definition"] == right["definition"]
                if (left["year"] == right["year"] and same_scope
                        and abs(left["value"] - right["value"]) / max(left["value"], right["value"], 1) > 0.01):
                    conflicts.append((left, right))
        if conflicts:
            examples = sorted({f"{int(item['value']):,} ({item['year']})" for pair in conflicts for item in pair})
            assessment = "Potential same-year population discrepancy: " + "; ".join(examples[:6]) + ". Compare geographic coverage, population definition, and estimation method before treating these as conflicting figures."
        elif len(observations) > 1:
            years = sorted({item["year"] for item in observations if item["year"] is not None})
            if len(years) > 1:
                assessment = "Population figures refer to different reference years (" + ", ".join(map(str, years)) + "); they are not directly comparable as a contradiction. Check units, geographic coverage, and definitions."
            else:
                assessment = "Relevant population figures were found, but available context is insufficient to establish a like-for-like numerical comparison."
        else:
            assessment = "Relevant passages were found. No comparable same-year population difference was detected; this does not establish agreement."
    else:
        publisher_families = max(g["similar_wording_domains"] for g in groups)
        if publisher_families >= 2:
            assessment = "Similar wording appears across multiple publisher families. This is a lead for cross-checking, not proof of independent confirmation."
        else:
            assessment = "Relevant passages were found, but this prototype does not determine whether separate claims agree or conflict. Review each linked page and its context."
    return groups, assessment

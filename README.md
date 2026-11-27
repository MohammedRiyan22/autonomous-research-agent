# Fieldnotes: Autonomous Research Agent

A small local web app that demonstrates the assessment workflow. It plans a search, uses DuckDuckGo through the `ddgs` Python package, opens candidate pages, extracts page text, shows passages with clickable source links, and saves completed runs to SQLite.

## Run it

Requires Python 3.10+, [Ollama](https://ollama.com/download), the selected local model, and internet access for the public search service and source pages. Each person running the app needs Ollama and the model installed on their own computer; model weights are not included in this repository. `requirements.txt` installs only the Python packages.

```powershell
cd path\to\Fieldnotes_Autonomous_Research_Agent
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma4:12b
.venv\Scripts\python.exe app.py
```

The `ollama pull` command downloads the default model. You only need to run it once per computer. Start the Ollama app/service before starting Fieldnotes. If Ollama is not running, open the Ollama app or run `ollama serve` in a separate terminal. Open <http://127.0.0.1:8000>. Set `PORT` to use a different port. Set `OLLAMA_MODEL` or `OLLAMA_URL` to override the defaults. The first model response can be slow; the default Ollama timeout is 120 seconds. Set `OLLAMA_TIMEOUT_SECONDS` to change it. Page reads get up to 24 seconds by default; set `PAGE_TIMEOUT_SECONDS` to adjust that limit. Runs are stored in `research_runs.sqlite3` in the current directory. To run the offline checks: `.venv\Scripts\python.exe -m unittest -v`.

## Project structure

- `app.py`: small launcher; run this file to start the local server.
- `settings.py`: environment-based settings, limits, and shared patterns.
- `planning.py`: question cleanup, research planning, and freshness checks.
- `sources.py`: web search, page reading, dates, and source selection.
- `evidence.py`: passage extraction, numeric checks, and evidence grouping.
- `model_review.py`: local Ollama analysis and claim validation.
- `research_pipeline.py`: research workflow, persistence, and background jobs.
- `web_interface.py`: HTML interface and HTTP API routes.

## What happens when a question is submitted

1. A rule-based planner breaks the topic into visible research subquestions and adds date or number checks when relevant.
2. The app searches the natural question, a focused topic query, an official-source query, and an independent-source query. After reading initial pages, it targets up to four evidence groups with claim-focused searches and may try a shorter fallback query. It deduplicates URLs, ranks government/academic and topic-matched domains first, and prefers distinct domains when choosing pages. Domain diversity is a selection hint, not proof of independent ownership or reporting.
3. Beautiful Soup prefers article/main text and looks for dates in search results, page metadata, JSON-LD structured data, and time tags. Search previews only help filter candidates; report evidence comes from pages the app successfully opens.
4. A key-term filter first selects likely passages. For readable pages it misses, the local model can select relevant sentences by meaning; the app accepts only candidate IDs that map to exact text from the opened page, never model-written passage text. Near-identical passages are grouped and flagged as possible copied wording, never as proof of independence. Different numbers are compared across all linked passages.
5. After the first pages yield relevant passages, the app performs targeted searches for up to four findings and opens up to four additional pages from other publishers. Ollama reviews the combined passages using a JSON schema and a concise output limit. It breaks the answer into claims, cites passages that support and contradict each claim, and flags gaps. The app rejects malformed output and unknown source IDs, requires both sides to be cited for a conflict, and downgrades “supported” claims that lack multi-domain, wording-diverse corroboration. Distinct domains and wording are only independence hints, not proof.
6. The browser polls the run status and shows each workflow step as it happens. Search/page/model errors and the 300-second overall run limit are surfaced. The report is saved in SQLite.

See [EXPLAINER.md](EXPLAINER.md) for a presentation-ready walkthrough, and [architecture.md](architecture.md) for the diagram.

## Recent improvements

- Population estimate extraction now accepts the country named in the research question rather than requiring India. Current-population searches use the requested country in their query wording.
- Passage selection can use Ollama embeddings (`OLLAMA_EMBED_MODEL`, default `nomic-embed-text`) to recover relevant paraphrases missed by keyword overlap. If the embedding endpoint/model is unavailable, the original lexical matching remains as a fallback.
- Population comparisons normalize common count units, distinguish reference years, and avoid automatically flagging urban/rural or working-age/total-definition differences as like-for-like conflicts. Users should still inspect geography and source methodology.

## Assessment checklist

- Question box and report view: `app.py`
- Plan, multi-query search/open, model-backed synthesis, evidence checks, clickable claim-level links, uncertainty and failure messages: visible in the report view
- Research run storage: SQLite (`research_runs.sqlite3`)
- Source-link and failed-search checks: `test_app.py`
- Example reports: `examples/`
- Architecture diagram: `architecture.md`
- Demo narration outline: `DEMO.md` (record a screen-capture video from it before final submission)

## Limits and source choice

The `ddgs` package uses public web-search backends (including DuckDuckGo) without a paid API key. A run makes up to four initial searches and considers up to four evidence groups for claim-focused searches; each finding can trigger a compact fallback query when needed. Each query requests up to five results. The app opens up to five initial pages plus four cross-check pages. Search timeouts are 12 seconds, page reads 24 seconds, Ollama model calls up to 120 seconds, and a run is capped at 300 seconds. The public search service and individual websites may also impose changing rate, access, or availability limits. The app surfaces failures rather than guaranteeing results; follow each provider's and site's terms.

The selected AI model is your local `gemma4:12b`, served by Ollama's [chat API](https://docs.ollama.com/api/chat). For optional embedding-based passage matching, install the `nomic-embed-text` embedding model with `ollama pull nomic-embed-text` (or set `OLLAMA_EMBED_MODEL` to another Ollama embedding model). If that embedding model is unavailable, Fieldnotes automatically falls back to its lexical passage matcher; the rest of the research workflow continues. Model inference stays on your machine; the question is still sent to public search backends, and source pages are fetched from their websites. If lexical passage matching misses a readable page, Ollama can choose exact candidate sentences by meaning. The app validates the selected IDs against the opened page, so the model cannot supply or paraphrase evidence. If Ollama or the model is unavailable, times out, or returns output that fails schema validation, Fieldnotes reports the failure and supplies a source-linked extractive fallback from read passages. It labels this fallback as unverified and does not present it as an AI-checked answer. Model judgments can still be wrong. Keyword matching is supplemented by optional local embedding similarity when the embedding model is available. Population number comparisons normalize common count units and avoid treating different reference years or population definitions as direct contradictions. Geographic coverage and measurement methodology still require source-level review; the system does not prove that two estimates are fully comparable. Similar wording can indicate copying, but the app cannot prove provenance or factual truth. Inspect the cited pages before relying on a claim.

Respect each website's access rules. Do not raise concurrency or scrape limits without checking provider and site policies.

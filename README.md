# Fieldnotes: Autonomous Research Agent

A small local web app that demonstrates the assessment workflow. It plans a search, uses DuckDuckGo through the `ddgs` Python package, opens candidate pages, extracts page text, shows passages with clickable source links, and saves completed runs to SQLite.

## Run it

Requires Python 3.10+, [Ollama](https://ollama.com/download), the selected local model, and internet access for the public search service and source pages.

```powershell
cd C:\Users\sim\Documents\Codex\2026-11-24\h\outputs\autonomous-research-agent
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

The app defaults to your installed `gemma4:12b` model. Start the Ollama app/service before starting Fieldnotes. If Ollama is not running, open the Ollama app or run `ollama serve` in a separate terminal. Open <http://127.0.0.1:8000>. Set `PORT` to use a different port. Set `OLLAMA_MODEL` or `OLLAMA_URL` to override the defaults. The first model response can be slow; the default Ollama timeout is 120 seconds. Set `OLLAMA_TIMEOUT_SECONDS` to change it. Runs are stored in `research_runs.sqlite3` in the current directory. To run the offline checks: `.venv\Scripts\python.exe -m unittest -v`.

## What happens when a question is submitted

1. A rule-based planner breaks the topic into visible research subquestions and adds date or number checks when relevant.
2. The app searches the natural question, a focused topic query, an official-source query, and an independent-source query. It deduplicates URLs, ranks government/academic and topic-matched domains first, and opens up to five pages. Domain labels are hints, not authority guarantees.
3. Beautiful Soup prefers article/main text and extracts dates from common metadata fields. Search previews only help filter candidates; report evidence comes from pages the app successfully opens.
4. A key-term filter rejects results and passages that do not match enough topic words. Near-identical passages are grouped and flagged as possible copied wording, never as proof of independence. Different numbers are compared across all linked passages.
5. Ollama runs the selected local Gemma 4 12B model over the retrieved passages. It writes a short answer, separates important claims, assigns source IDs, and flags possible agreement, conflict, and gaps. The app rejects claim citations that reference missing source IDs, then makes accepted source links clickable beside each claim.
6. The browser polls the run status and shows each workflow step as it happens. Search/page/model errors and the 240-second overall run limit are surfaced. The report is saved in SQLite.

See [EXPLAINER.md](EXPLAINER.md) for a presentation-ready walkthrough, and [architecture.md](architecture.md) for the diagram.

## Assessment checklist

- Question box and report view: `app.py`
- Plan, multi-query search/open, model-backed synthesis, evidence checks, clickable claim-level links, uncertainty and failure messages: visible in the report view
- Research run storage: SQLite (`research_runs.sqlite3`)
- Source-link and failed-search checks: `test_app.py`
- Example reports: `examples/`
- Architecture diagram: `architecture.md`
- Demo narration outline: `DEMO.md` (record a screen-capture video from it before final submission)

## Limits and source choice

The `ddgs` package uses public web-search backends (including DuckDuckGo) without a paid API key. It makes up to four queries, requests up to five results per query, opens at most five pages, and uses 12-second search/page timeouts. Public service limits and access rules can change; follow each provider and site's terms. The app reports timeouts and blocked pages. The whole research run has a 240-second limit, including local model analysis.

The selected AI model is your local `gemma4:12b`, served by Ollama's [chat API](https://docs.ollama.com/api/chat). Model inference stays on your machine; the question is still sent to public search backends, and source pages are fetched from their websites. If Ollama or the model is unavailable, Fieldnotes reports the error and shows source excerpts without claiming to have produced an AI-checked answer. Model judgments can still be wrong. Keyword relevance and number comparison can miss paraphrases, units, years, and non-numeric contradictions. Similar wording can indicate copying, but the app cannot prove provenance or factual truth. Inspect the cited pages before relying on a claim.

Respect each website's access rules. Do not raise concurrency or scrape limits without checking provider and site policies.

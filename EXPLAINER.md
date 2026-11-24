# How to explain the project

## 30-second version

“Fieldnotes is a local research assistant. It turns a question into visible subquestions, searches the web, opens pages, and sends relevant passages to the local Gemma 4 12B model using Ollama. The model writes a short answer and lists important claims with source IDs. The app checks that those IDs exist, links each claim to the original page, and reports missing or conflicting evidence. It shows its steps and saves the run in SQLite. It is useful for research, but a person still needs to check the sources and the model's interpretation.”

## Walkthrough, in order

1. **Input:** The user enters a question in the web page.
2. **Plan:** `plan_question` creates smaller research questions and adds freshness checks for current topics or date/method checks for number questions.
3. **Search:** `search_web` runs the natural question, focused topic terms, an official-source query, and an independent-source query. It uses the `ddgs` package and does not treat result previews as evidence.
4. **Choose and read pages:** Candidates that miss too many key terms are discarded. Government and education domains and topic-matched domains are prioritized as selection hints. `read_page` opens up to five HTML pages, prefers article/main text, and extracts common date metadata. Blocked pages are reported.
5. **Check evidence:** `evidence_from` only selects passages with sufficient key-term overlap. `compare_evidence` groups near-identical wording and checks whether relevant passages contain different numbers. Similar wording is a copying warning, not proof of independence.
6. **Synthesize:** `analyze_with_model` sends the question and selected excerpts to the local Gemma 4 12B model. The model must return a short answer and important claims with source IDs. The app rejects claims with unknown source IDs and renders clickable source links beside accepted claims. Source passages are treated as untrusted data, not instructions.
7. **Progress and save:** The browser polls the background job and shows query, page-read, checking, and report steps. The report includes source dates, failed searches/pages/model calls, uncertainty and gaps. `save_run` stores the full result in SQLite.

## Technology choices

- **Python:** readable and has mature search and web-reading libraries.
- **Built-in HTTP server:** keeps the demo small and avoids framework setup.
- **Public search via `ddgs`:** no paid key needed. The prototype runs up to four queries and follows the service's current access rules.
- **Requests + Beautiful Soup:** fetches pages and extracts main/article text.
- **Ollama + Gemma 4 12B:** the user's local model for report synthesis and claim comparison. Model inference stays on the user's computer; search questions and page fetches still use the internet.
- **SQLite:** local file database for research history.
- **Plain HTML/CSS/JavaScript:** small, responsive browser interface with polling progress.

## Likely questions

**Does it prove a claim is true?** No. The model compares only the retrieved passages, and its judgment can be wrong. Inspect source quality, dates, units, and context.

**How are links checked?** Model claims must cite source IDs present in the current run. The app drops claims with unknown IDs and renders links beside each accepted claim. A valid ID only proves that a source was returned; it does not prove that source supports the claim.

**How does it know sources are independent?** It does not prove independence. It groups near-identical wording as a warning and instructs the model not to treat the same wording group as independent confirmation. Copied articles and shared underlying sources may still be missed.

**What happens when a source or model is unavailable?** The report shows the failure. If the model is unavailable, it displays relevant source passages without presenting them as a synthesized, checked answer.

**What are the known limits?** The search provider can fail or rate-limit queries. Some sites block automated access. Keyword matching can miss paraphrases. The numeric check does not normalize all units or years. The model can miss conflicts or make unsupported interpretations, and on slower computers its review can time out before a cross-check is available. A 240-second limit stops long jobs. This is a local assessment demo, not a production multi-user service.

## Files to point to in a presentation

- `app.py`: planning, search, page extraction, relevance checks, model review, routes, and storage.
- `test_app.py`: checks for source links, source-ID validation, failed search, irrelevant NVIDIA results, and numeric differences.
- `architecture.md`: system diagram.
- `examples/`: three example reports.
- `DEMO.md`: recording outline for the required short video.

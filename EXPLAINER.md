# How to explain the project

## 30-second version

“Fieldnotes is a local research assistant. It turns a question into visible subquestions, searches the web, opens pages, and sends relevant passages to the local Gemma 4 12B model using Ollama. The model writes a short answer and lists important claims with source IDs. The app checks that those IDs exist, links each claim to the original page, and reports missing or conflicting evidence. It shows its steps and saves the run in SQLite. It is useful for research, but a person still needs to check the sources and the model's interpretation.”

## Walkthrough, in order

1. **Input:** The user enters a question in the web page.
2. **Plan:** `plan_question` creates smaller research questions and adds freshness checks for current topics or date/method checks for number questions.
3. **Search:** `search_web` runs the natural question, focused topic terms, an official-source query, and an independent-source query. It uses the `ddgs` package and does not treat result previews as evidence.
4. **Choose and read pages:** Candidates that miss too many key terms are discarded. Government and education domains and topic-matched domains are prioritized as selection hints. The selector prefers distinct domains before taking a second page from one site. `read_page` opens up to five initial HTML pages, prefers article/main text, and extracts common date metadata. Blocked pages are reported.
5. **Target findings:** After relevant passages are extracted, the app runs up to two extra searches using the passages as specific findings to verify, then opens up to two additional pages from other domains. This seeks claim-level corroboration without treating a search snippet as evidence.
6. **Check evidence:** `evidence_from` only selects passages with sufficient key-term overlap. `compare_evidence` groups near-identical wording and checks whether relevant passages contain different numbers. Similar wording is a copying warning, not proof of independence.
7. **Cross-check and synthesize:** `analyze_with_model` sends the question and all selected excerpts to the local Gemma 4 12B model with a JSON response schema and concise output limit. It must return atomic claims with separate source IDs that support and contradict each claim. The app rejects malformed output, unknown IDs, and conflicts without citations on both sides. It downgrades support that comes from one source, one domain, or a same-wording cluster. Multiple domains and wording groups are only a proxy for independent corroboration, not proof. Source passages are treated as untrusted data, not instructions.
8. **Progress and save:** The browser polls the background job and shows broad and claim-focused searches, page reads, checking, and report steps. The report includes source dates, failed searches/pages/model calls, uncertainty and gaps. `save_run` stores the full result in SQLite.

## Technology choices

- **Python:** readable and has mature search and web-reading libraries.
- **Built-in HTTP server:** keeps the demo small and avoids framework setup.
- **Public search via `ddgs`:** no paid key needed. The prototype runs up to four broad queries plus up to two claim-focused searches and follows the service's current access rules.
- **Requests + Beautiful Soup:** fetches pages and extracts main/article text.
- **Ollama + Gemma 4 12B:** the user's local model for report synthesis and claim comparison. Model inference stays on the user's computer; search questions and page fetches still use the internet.
- **SQLite:** local file database for research history.
- **Plain HTML/CSS/JavaScript:** small, responsive browser interface with polling progress.

## Likely questions

**Does it prove a claim is true?** No. The model compares only the retrieved passages, and its judgment can be wrong. Inspect source quality, dates, units, and context.

**How are links checked?** Model claims must cite source IDs present in the current run. The app drops claims with unknown IDs and renders links beside each accepted claim. A valid ID only proves that a source was returned; it does not prove that source supports the claim.

**How does it know sources are independent?** It does not prove independence. A claim needs supporting pages from multiple domains and wording groups to retain a “supported” label. This is a screening proxy only; different domains can share ownership or sources, and copied articles with altered wording may be missed.

**What happens when a source or model is unavailable?** The report shows the failure. If model output is unavailable or invalid but passages were read, the app presents a short source-linked extractive fallback, clearly labeled as not AI-checked. If there are no relevant passages, it gives an explicit non-answer.

**What are the known limits?** The search provider can fail or rate-limit queries. Some sites block automated access. Keyword matching can miss paraphrases. Claim-focused searches target at most two findings and open at most two additional pages, so they cannot exhaustively check every claim. The numeric check does not normalize all units or years. The model can miss conflicts or make unsupported interpretations, and on slower computers its review can time out before a cross-check is available. A 300-second limit stops long jobs. This is a local assessment demo, not a production multi-user service.

## Files to point to in a presentation

- `app.py`: planning, search, page extraction, relevance checks, model review, routes, and storage.
- `test_app.py`: checks for source links, source-ID validation, failed search, irrelevant NVIDIA results, and numeric differences.
- `architecture.md`: system diagram.
- `examples/`: three example reports.
- `DEMO.md`: recording outline for the required short video.

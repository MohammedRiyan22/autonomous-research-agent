"""HTML interface and HTTP routes."""
from __future__ import annotations
import json
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from research_pipeline import JOB_LOCK, JOBS, run_job

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fieldnotes — Research Agent</title><style>
:root{--ink:#172b2a;--muted:#667875;--paper:#f4f6f1;--card:#fff;--green:#174f46;--line:#dce5df;--orange:#cf6a38}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Inter,Segoe UI,Arial,sans-serif}.wrap{max-width:1060px;margin:auto;padding:34px 22px 70px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:52px}.brand{font-weight:800;letter-spacing:.03em}.tag{color:var(--muted);font-size:13px}.hero{max-width:760px;margin-bottom:38px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:var(--orange);font-size:12px;font-weight:800}.hero h1{font-size:clamp(36px,7vw,66px);line-height:1.02;letter-spacing:-.045em;margin:12px 0 15px}.hero p{color:var(--muted);font-size:18px}.panel,.report{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;margin:18px 0;box-shadow:0 8px 30px #24453b0b}label{font-weight:700;display:block;margin-bottom:9px}textarea{width:100%;border:1px solid #becdc5;border-radius:12px;padding:14px;font:inherit;resize:vertical;min-height:110px;color:var(--ink)}button{margin-top:13px;border:0;border-radius:10px;padding:12px 20px;background:var(--green);color:white;font-weight:700;font-size:15px;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.status{margin:18px 0;color:var(--muted);font-size:14px}.steps{padding-left:20px}.steps li{padding:4px 0}.pill{display:inline-block;padding:4px 9px;border-radius:999px;background:#e7f1eb;color:var(--green);font-size:12px;font-weight:700}h2{font-size:23px;letter-spacing:-.02em;margin:0 0 13px}h3{margin-bottom:5px;font-size:17px}.claim{border-left:3px solid #80a991;padding:5px 0 5px 15px;margin:17px 0}.claim p{margin:0 0 8px}.sources{padding-left:20px}.sources li{margin:8px 0}.source-meta{font-size:13px;color:var(--muted)}a{color:#176b59;text-decoration-thickness:1px;text-underline-offset:3px}.notice{background:#fff5e9;border:1px solid #f3d9bc;padding:13px;border-radius:10px;margin:12px 0}.small{font-size:14px;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.spinner{display:inline-block;width:12px;height:12px;border:2px solid #b8c7bf;border-top-color:var(--green);border-radius:50%;animation:spin .8s linear infinite;margin-right:7px}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:650px){.grid{grid-template-columns:1fr}.top{margin-bottom:34px}.panel,.report{padding:18px}}
</style></head><body><main class="wrap"><header class="top"><div class="brand">◉ FIELDNOTES</div><div class="tag">A transparent research prototype</div></header><section class="hero"><div class="eyebrow">Ask · investigate · verify</div><h1>Research with its work shown.</h1><p>Enter a question. Fieldnotes searches the open web, reads pages, and links evidence beside the passages it found.</p></section><section class="panel"><form id="form"><label for="question">What do you want to research?</label><textarea id="question" maxlength="500" placeholder="Example: What are the benefits of solar panels for homes? Show the sources."></textarea><button id="submit">Start research</button></form><div id="status" class="status" aria-live="polite"></div></section><section id="result"></section><footer class="small">Prototype note: automated sentence matching is a starting point, not a substitute for expert review.</footer></main><script>
const form=document.querySelector('#form'),btn=document.querySelector('#submit'),statusBox=document.querySelector('#status'),out=document.querySelector('#result');
function esc(s=''){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function link(s){return `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>`}
form.addEventListener('submit',async e=>{e.preventDefault();out.innerHTML='';btn.disabled=true;statusBox.innerHTML='<span class="spinner"></span>Starting research…';try{const r=await fetch('/api/research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.querySelector('#question').value})});const start=await r.json();if(!r.ok)throw Error(start.error||'Research failed');let done=false;while(!done){await new Promise(resolve=>setTimeout(resolve,650));const check=await fetch('/api/jobs/'+encodeURIComponent(start.job_id));const job=await check.json();if(!check.ok)throw Error(job.error||'Research run not found');const steps=job.steps||[];statusBox.innerHTML='<span class="spinner"></span>'+esc(steps[steps.length-1]||'Working');if(job.status==='complete'){statusBox.textContent='Research complete';out.innerHTML=render(job.report);done=true}else if(job.status==='error'){throw Error(job.error||'Research failed')}}}catch(err){statusBox.textContent='Could not complete research: '+err.message}finally{btn.disabled=false}});
function render(d){
 const plan=(d.plan||[]).map(x=>`<li>${esc(x)}</li>`).join('');
 const sourceForId=id=>(d.sources||[]).find(s=>s.source_id===id);
 const analysis=d.analysis||{};
 const fallbackClaims=(analysis.fallback_claims||[]).map(c=>{const support=(c.source_ids||[]).map(id=>sourceForId(id)).filter(Boolean).map(link).join(' · ');return `<div class="claim"><p>${esc(c.claim)}</p><p><span class="pill">${esc(c.assessment)}</span> ${esc(c.note||'')}</p>${support?`<div><b>Source:</b> ${support}</div>`:''}</div>`}).join('');
 const claims=(analysis.claims||[]).map(c=>{const support=c.source_ids.map(id=>sourceForId(id)).filter(Boolean).map(link).join(' · ');const challenge=(c.contradicting_source_ids||[]).map(id=>sourceForId(id)).filter(Boolean).map(link).join(' · ');return `<div class="claim"><p>${esc(c.claim)}</p><p><span class="pill">${esc(c.assessment)}</span> ${esc(c.note||'')}</p>${support?`<div><b>Supporting sources:</b> ${support}</div>`:''}${challenge?`<div><b>Contradicting sources:</b> ${challenge}</div>`:''}</div>`}).join('')||(analysis.status!=='ready'?fallbackClaims:'');
 const conflicts=(analysis.conflicts||[]).map(c=>`<li>${esc(c)}</li>`).join('');
 const modelNotice=analysis.status==='ready'?`<p class="small">Reviewed with local model <b>${esc(analysis.model)}</b>.</p>`:analysis.status==='insufficient'?`<div class="notice"><b>Evidence insufficient.</b> ${esc(analysis.error||'The retrieved pages do not support a reliable answer.')}</div>`:analysis.status==='skipped'?`<div class="notice"><b>AI review skipped.</b> ${esc(analysis.error||'No relevant passages were available for model analysis.')}</div>`:`<div class="notice"><b>${analysis.status==='invalid'?'AI synthesis unavailable.':'AI review unavailable.'}</b> ${esc(analysis.error||'No model-generated review was returned.')}</div>`;
 const displayedSummary=analysis.status==='ready'?analysis.summary:(analysis.fallback_summary||analysis.summary||d.answer);
 const summaryIds=analysis.status==='ready'?(analysis.summary_source_ids||[]):(analysis.fallback_summary_source_ids||[]);
 const summaryLinks=summaryIds.map(id=>sourceForId(id)).filter(Boolean).map(link).join(' · ');
 const estimates=(d.population_estimates||[]).map(item=>`<li><b>${esc(item.display)}</b>${item.reference_year?` · reference year ${esc(item.reference_year)}`:''} · published ${esc(item.date||'date unavailable')} · ${link(item)}</li>`).join('');
 const leads=(d.population_leads||[]).map(item=>`<li><b>${esc(item.display)}</b> · ${link(item)} <span class="source-meta">Not counted as evidence: ${esc(item.reason)}</span></li>`).join('');
 const estimateReview=estimates?`<h3>Population figures by source</h3><p class="small">These are page-level reported estimates. Different reference periods or methods can explain differences; this list alone does not establish which estimate is most accurate.</p><ol class="sources">${estimates}</ol>${leads?`<div class="notice"><b>Unverified search-result leads</b><p class="small">Numbers found only in titles or previews are shown for follow-up. They are not included in the evidence comparison.</p><ol class="sources">${leads}</ol></div>`:''}`:'';
 const checkLabel=(d.population_estimates||[]).length?'Numeric source comparison':analysis.status==='ready'?'Cross-check':analysis.status==='insufficient'?'Cross-check result':analysis.status==='skipped'?'Cross-check skipped (no relevant passages)':analysis.status==='unavailable'?'AI cross-check unavailable':'Text-only check';
 const reportTitle=analysis.status==='ready'||fallbackClaims?'Short answer':'Evidence review';
 const fallbackLabel=analysis.status!=='ready'&&fallbackClaims?`<p class="small"><b>Extractive fallback:</b> Statements below are copied from retrieved passages and linked to their pages; they have not been model-checked.</p>`:'';
 const summary=`<section class="report"><div class="eyebrow">Evidence-linked report</div><h2>${reportTitle}</h2>${modelNotice}${fallbackLabel}<p>${esc(displayedSummary)}</p>${summaryLinks?`<div>${summaryLinks}</div>`:''}${claims?`<h3>Important claims and sources</h3>${claims}`:''}${conflicts?`<div class="notice"><b>Possible conflicts to review</b><ul>${conflicts}</ul></div>`:''}${analysis.evidence_gaps&&analysis.status==='ready'?`<p class="small"><b>Evidence gaps:</b> ${esc(analysis.evidence_gaps)}</p>`:''}${estimateReview}<div class="notice"><b>${checkLabel}:</b> ${esc(d.comparison)}</div><p class="small"><b>Uncertainty:</b> ${esc(d.uncertainty)}</p></section>`;
 const readPages=(d.sources||[]).filter(s=>s.status==='read').length;
 const failedPages=(d.sources||[]).filter(s=>s.status==='error').length;
 const noEvidenceDetail=readPages?`${readPages} page${readPages===1?' was':'s were'} read, but none contained a passage relevant enough to verify. ${failedPages?`${failedPages} other page${failedPages===1?' could not':'s could not'} be read. `:''}Search-result titles and previews are leads only; they are not used as evidence.`:`No source page was successfully read, so there are no page passages to check. Search-result titles and previews are leads only; they are not used as evidence.`;
 const evidence=(d.groups||[]).map(g=>`<div class="claim"><p>${esc(g.excerpt)}</p><div>${g.sources.map(link).join(' · ')} ${g.similar_wording_domains>1?`<span class="pill">Similar wording across ${g.similar_wording_domains} publisher families</span>`:''}</div></div>`).join('')||`<div class="notice"><b>No verified passages found.</b> ${esc(noEvidenceDetail)}</div>`;
 const sources=(d.sources||[]).map(s=>{const state=s.evidence_status||(s.status==='excluded'?s.error:s.status==='read'?'page read':'could not read: '+s.error);const cross=s.cross_check_finding?` · claim-focused check for finding ${esc(s.cross_check_finding)}`:'';return `<li>${link(s)} <span class="source-meta">Published: ${esc(s.date||'date unavailable')} · ${esc(s.source_class||'Publisher type unknown')}${cross} · ${esc(state||'status unavailable')}</span></li>`}).join('');
 const progress=(d.progress||[]).map(x=>`<li>${esc(x)}</li>`).join('');
 const evidenceTitle=analysis.related_passages_suppressed?'Source passages':analysis.status==='insufficient'?'Related search results (not evidence)':'Source passages';
 const evidenceContext=analysis.related_passages_suppressed?'<p class="small">No opened passage directly answers the question. Related search results are listed under Sources and progress only, marked as not used as evidence.</p>':analysis.status==='insufficient'?'<p class="small">These pages matched some search terms, but they do not verify the specific claim in your question. They are shown for transparency and should not be treated as evidence for or against it.</p>':'';
 return `<section class="report"><div class="eyebrow">Research plan</div><h2>${esc(d.question)}</h2><ol class="steps">${plan}</ol><p class="small"><b>Searches used:</b> ${(d.queries||[]).map(esc).join(' · ')}</p></section>${d.search_error?`<div class="notice"><b>Search issue:</b> ${esc(d.search_error)}</div>`:''}${d.semantic_issue?`<div class="notice"><b>Semantic passage selection issue:</b> ${esc(d.semantic_issue)}</div>`:''}${summary}<section class="report"><h2>${evidenceTitle}</h2>${evidenceContext}${evidence}</section><section class="report"><h2>Sources and progress</h2><h3>Steps completed</h3><ol class="steps">${progress}</ol><h3>Sources</h3><ol class="sources">${sources||'<li>No candidate sources returned.</li>'}</ol></section>`;
}
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, status: int, body: bytes, content_type="text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            return self.send(204, b"")
        if path == "/":
            return self.send(200, PAGE.encode())
        if path == "/health":
            return self.send(200, b'{"ok":true}', "application/json")
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOB_LOCK:
                job = JOBS.get(job_id)
                snapshot = dict(job) if job else None
            if not snapshot:
                return self.send(404, b'{"error":"Research run not found"}', "application/json")
            return self.send(200, json.dumps(snapshot, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        return self.send(404, b"Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/research":
            return self.send(404, b'{"error":"Not found"}', "application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 10000:
                raise ValueError("Request is too large.")
            payload = json.loads(self.rfile.read(size) or b"{}")
            question = str(payload.get("question", ""))
            if len(question.strip()) < 5:
                raise ValueError("Enter a research question of at least five characters.")
            job_id = str(time.time_ns())
            with JOB_LOCK:
                JOBS[job_id] = {"status": "running", "steps": ["Research request accepted"], "report": None, "error": ""}
            threading.Thread(target=run_job, args=(job_id, question), daemon=True).start()
            return self.send(202, json.dumps({"job_id": job_id}).encode(), "application/json; charset=utf-8")
        except ValueError as exc:
            return self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")
        except Exception as exc:
            return self.send(500, json.dumps({"error": f"Unexpected server error: {type(exc).__name__}: {str(exc)[:200]}"}).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

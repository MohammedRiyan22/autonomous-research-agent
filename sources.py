"""Web search, page retrieval, and source metadata."""
from __future__ import annotations
import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

from settings import MAX_INITIAL_PAGES, MAX_RESULTS, NOISE_CLASS, PAGE_TIMEOUT, TIMEOUT, USER_AGENT

def search_web(query: str, timeout: float = TIMEOUT) -> list[dict]:
    """Search public web results. A failed query is surfaced to the caller."""
    # Lazy import keeps the deterministic offline tests usable without network packages.
    from ddgs import DDGS
    found = []
    with DDGS(timeout=timeout) as ddgs:
        for item in ddgs.text(query, max_results=MAX_RESULTS):
            url = item.get("href") or item.get("url")
            if not url or not url.startswith(("http://", "https://")):
                continue
            found.append({"title": item.get("title") or url, "url": url,
                          "preview": item.get("body") or "", "date": item.get("date") or ""})
    return found

def page_date(soup: BeautifulSoup, fallback: str = "") -> str:
    """Find a publication/update date from search metadata, page metadata, JSON-LD, or time tags."""
    value = fallback or ""
    if not value:
        date_keys = {
            "article:published_time", "article:modified_time", "og:published_time", "og:updated_time",
            "date", "datepublished", "datecreated", "datemodified", "pubdate", "publish-date",
            "publication_date", "parsely-pub-date", "dc.date", "dc.date.issued",
        }
        for tag in soup.find_all("meta"):
            key = str(tag.get("property") or tag.get("name") or tag.get("itemprop") or "").strip().lower()
            content = str(tag.get("content") or "").strip()
            if key in date_keys and content:
                value = content
                break
    if not value:
        def find_structured_date(item):
            if isinstance(item, dict):
                for key in ("datePublished", "dateCreated", "uploadDate", "dateModified"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                for child in item.values():
                    found = find_structured_date(child)
                    if found:
                        return found
            elif isinstance(item, list):
                for child in item:
                    found = find_structured_date(child)
                    if found:
                        return found
            return ""

        for tag in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
            try:
                value = find_structured_date(json.loads(tag.string or tag.get_text()))
            except (json.JSONDecodeError, TypeError):
                value = ""
            if value:
                break
    if not value:
        tag = soup.find("time", attrs={"datetime": True})
        if tag:
            value = str(tag.get("datetime") or "").strip()
        if not value:
            tag = soup.find("time")
            if tag:
                value = tag.get_text(" ", strip=True)
    value = re.sub(r"\s+", " ", str(value)).strip()
    iso_date = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    if iso_date:
        return iso_date.group(1)
    return value[:80]

def read_page(source: dict, timeout: float = PAGE_TIMEOUT) -> dict:
    """Fetch and extract readable page text; search previews are never evidence."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    response = requests.get(source["url"], headers=headers, timeout=timeout)
    response.raise_for_status()
    ctype = response.headers.get("content-type", "").lower()
    if "html" not in ctype and "xhtml" not in ctype:
        raise ValueError("The page did not return HTML text")
    # Let Beautiful Soup inspect the raw bytes and HTML charset declarations.
    # requests.Response.text can mojibake UTF-8 pages whose server omits a charset.
    soup = BeautifulSoup(response.content, "html.parser")
    date = page_date(soup, source.get("date", ""))
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "button", "form", "input", "select", "iframe"]):
        node.decompose()
    for node in soup.find_all(class_=NOISE_CLASS):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else source["title"]
    # Prefer article/main content to avoid breadcrumbs, menus, and footers becoming evidence.
    content = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    blocks = content.find_all(["p", "li", "tr", "dt", "dd"])
    paragraphs = [re.sub(r"\s+", " ", block.get_text(" ", strip=True)) for block in blocks]
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) >= 30]
    text = "\n".join(paragraphs) if paragraphs else re.sub(r"\s+", " ", content.get_text(" ", strip=True))
    text = text[:30000]
    if len(text) < 80:
        raise ValueError("The page contained too little readable text")
    source.update({"title": title[:250], "date": date, "text": text, "status": "read", "error": ""})
    return source

def domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")

def publisher_key(url: str) -> str:
    """Approximate the publisher's registrable domain for duplicate-source checks.

    This intentionally errs toward treating related subdomains as one publisher.
    It is not a substitute for checking ownership or editorial relationships.
    """
    host = domain(url)
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return host
    multi_label_suffixes = {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au",
        "gov.au", "edu.au", "co.in", "com.in", "net.in", "org.in", "ac.in",
        "gov.in", "co.nz", "com.br", "com.mx", "co.jp", "co.za",
    }
    suffix = ".".join(labels[-2:])
    if suffix in multi_label_suffixes and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix

def select_diverse_sources(sources: list[dict], limit: int = MAX_INITIAL_PAGES) -> list[dict]:
    """Prefer one page per approximate publisher before opening a second page."""
    selected, remaining, seen_domains = [], [], set()
    for source in sources:
        host = publisher_key(source.get("url", ""))
        if host and host not in seen_domains and len(selected) < limit:
            selected.append(source)
            seen_domains.add(host)
        else:
            remaining.append(source)
    if len(selected) < limit:
        selected.extend(remaining[:limit - len(selected)])
    return selected

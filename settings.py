"""Runtime configuration and shared patterns for Fieldnotes."""
from __future__ import annotations
import os
import re

HOST = os.environ.get("RESEARCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
DB_PATH = os.environ.get("RESEARCH_DB", "research_runs.sqlite3")
TIMEOUT = 12
PAGE_TIMEOUT = int(os.environ.get("PAGE_TIMEOUT_SECONDS", "24"))
MAX_RESULTS = 5
MAX_SEARCH_QUERIES = 4
MAX_INITIAL_PAGES = 5
MAX_CROSSCHECK_SEARCHES = 4
MAX_CROSSCHECK_PAGES = 4
MAX_RUN_SECONDS = 300
USER_AGENT = "AutonomousResearchAssessment/1.0 (educational research prototype)"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "summary_source_ids": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "assessment": {"type": "string", "enum": ["supported", "conflicting", "uncertain"]},
                "source_ids": {"type": "array", "items": {"type": "string"}},
                "contradicting_source_ids": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            },
            "required": ["claim", "assessment", "source_ids", "contradicting_source_ids", "note"],
            "additionalProperties": False,
        }},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "evidence_gaps": {"type": "string"},
    },
    "required": ["summary", "summary_source_ids", "claims", "conflicts", "evidence_gaps"],
    "additionalProperties": False,
}
STOP = set("a an and are as at be by can could did do for from has have how i in is it of on or please show tell that the this to was what when where which who why will with company launched launch about me tell latest newest current currently today now recent".split())
GENERIC_QUESTION = re.compile(r"\b(?:tell me about|please|show me|find out|what is|what are|who is|who are|which is|which was|how many|how much|how does|how do|why does|why do|when did|where is|where are|can you|could you|i want to know|explain)\b", re.I)
SOURCE_REQUEST = re.compile(r"\b(?:show|include|list|provide|give|with)\s+(?:me\s+)?(?:the\s+)?(?:publication\s+dates?|sources?|citations?|references?|dates?|links?)\b.*$", re.I)
NOISE_CLASS = re.compile(r"breadcrumb|table.of.contents|\btoc\b|share|social|related|sidebar|cookie|toolbar|\bmenu\b|\bnav\b|call.to.action", re.I)

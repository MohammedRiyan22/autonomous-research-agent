"""Fieldnotes entry point. Run with `python app.py`."""
from __future__ import annotations
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import os
import re
import requests
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlparse

# Re-export the core functions to keep the simple app.py import surface available.
from settings import *  # noqa: F401,F403
from sources import *  # noqa: F401,F403
from evidence import *  # noqa: F401,F403
from planning import *  # noqa: F401,F403
from model_review import analyze_with_model
from research_pipeline import JOB_LOCK, JOBS, research, run_job, save_run
from web_interface import Handler, PAGE

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Fieldnotes is running at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()

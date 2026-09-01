"""Polls this repo's "Multi-Agent Trading Cycle" runs (the only workflow that runs the
Sentiment -> Strategy chain — exit-management-v2.yml doesn't), downloads each new completed run's
logs, and appends one row per symbol to a local CSV: the sentiment reading and the strategy
chosen for every market the pipeline looked at that cycle.

Needs GITHUB_TOKEN in .env (see scripts/watch_github_runs.py's docstring for how to get one).

Run: python scripts/track_strategy_choices.py
Log: strategy_choices.csv (appended to, never overwritten) in the repo root.
"""
import ast
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO = "khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON"
WORKFLOW_FILE = "multi-agent-trading-v2.yml"
TOKEN = os.environ.get("GITHUB_TOKEN")
POLL_SECONDS = 60
ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / ".strategy_log_state.json"
CSV_FILE = ROOT / "strategy_choices.csv"

CSV_FIELDS = [
    "cycle_time_local", "run_number", "run_url", "symbol",
    "final_signal", "overridden_by_sentiment", "sentiment", "sentiment_reasoning",
    "strategy", "strategy_reasoning",
]

# GitHub Actions log lines are prefixed with an RFC3339 timestamp, e.g.
# "2026-09-01T18:07:14.1234567Z the actual line". Strip it to get back the real print() output.
_TIMESTAMP_PREFIX = re.compile(r"^\S+Z\s?")
_SYMBOL_HEADER = re.compile(r"^--- (\S+) ---$")


def _api_get(url):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_new_runs(since_run_id):
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs?per_page=20&status=completed"
    data = json.loads(_api_get(url))
    return [r for r in data["workflow_runs"] if r["id"] > since_run_id]


def find_pipeline_log_text(run_id):
    """Downloads a run's logs and returns the text of the "Run the multi-agent trading pipeline"
    step, or None if the run has no such step (e.g. it failed before reaching it)."""
    zip_bytes = _api_get(f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/logs")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if "run the multi-agent trading pipeline" in name.lower():
                return zf.read(name).decode("utf-8", errors="replace")
    return None


def _strip_timestamp(line):
    return _TIMESTAMP_PREFIX.sub("", line, count=1).rstrip("\n")


def _parse_dict_line(line):
    """The agents print plain Python dicts (not JSON) — ast.literal_eval is the safe equivalent
    of eval() for that, with no arbitrary code execution risk."""
    try:
        return ast.literal_eval(line.strip())
    except (ValueError, SyntaxError):
        return None


def parse_symbol_rows(log_text, run):
    """Walks the log line by line, tracking which symbol's "--- SYMBOL ---" block we're in, and
    pulls the Sentiment Agent / Strategy Agent dicts that immediately follow their section
    headers within that block."""
    lines = [_strip_timestamp(l) for l in log_text.splitlines()]
    rows = []
    symbol = None
    sentiment = None
    strategy = None
    triggered_local = datetime.fromisoformat(run["run_started_at"]).astimezone().isoformat(timespec="seconds")

    def flush():
        if symbol and sentiment and strategy:
            rows.append({
                "cycle_time_local": triggered_local,
                "run_number": run["run_number"],
                "run_url": run["html_url"],
                "symbol": symbol,
                "final_signal": sentiment.get("signal"),
                "overridden_by_sentiment": sentiment.get("overridden"),
                "sentiment": sentiment.get("sentiment"),
                "sentiment_reasoning": sentiment.get("reasoning"),
                "strategy": strategy.get("strategy"),
                "strategy_reasoning": strategy.get("reasoning"),
            })

    i = 0
    while i < len(lines):
        line = lines[i]
        header_match = _SYMBOL_HEADER.match(line)
        if header_match:
            flush()
            symbol, sentiment, strategy = header_match.group(1), None, None
        elif line == "=== Sentiment Agent ===" and i + 1 < len(lines):
            sentiment = _parse_dict_line(lines[i + 1])
        elif line == "=== Strategy Agent ===" and i + 1 < len(lines):
            strategy = _parse_dict_line(lines[i + 1])
        i += 1
    flush()
    return rows


def load_last_run_id():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())["last_run_id"]
    return 0


def save_last_run_id(run_id):
    STATE_FILE.write_text(json.dumps({"last_run_id": run_id}))


def append_rows(rows):
    is_new_file = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    if not TOKEN:
        print(
            "GITHUB_TOKEN not set in .env — add a token with 'Actions: Read-only' access "
            "to poll this private repo's workflow runs. See github.com/settings/tokens."
        )
        sys.exit(1)

    last_run_id = load_last_run_id()
    print(f"Watching {REPO}'s {WORKFLOW_FILE} runs, polling every {POLL_SECONDS}s. Appending to {CSV_FILE}")

    while True:
        try:
            new_runs = sorted(fetch_new_runs(last_run_id), key=lambda r: r["id"])
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"Poll failed ({exc}), retrying next cycle.")
            time.sleep(POLL_SECONDS)
            continue

        for run in new_runs:
            log_text = find_pipeline_log_text(run["id"])
            if log_text is not None:
                rows = parse_symbol_rows(log_text, run)
                if rows:
                    append_rows(rows)
                    print(f"Run #{run['run_number']}: logged {len(rows)} symbol(s) — {[r['symbol'] for r in rows]}")
                else:
                    print(f"Run #{run['run_number']}: no candidates crossed the momentum threshold — nothing to log.")
            last_run_id = run["id"]
            save_last_run_id(last_run_id)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

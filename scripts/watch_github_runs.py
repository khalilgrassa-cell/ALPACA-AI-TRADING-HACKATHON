"""Polls this repo's GitHub Actions runs and appends new ones to a local text file, so you can
see when the scheduled cron pipeline (multi-agent-trading.yml / exit-management.yml) fired
without opening the GitHub UI.

The repo is private, so this needs a token: add GITHUB_TOKEN=... to .env, using a token with
'Actions: Read-only' access (github.com/settings/tokens -> fine-grained token scoped to this
repo, or a classic token with the 'repo' scope).

Run: python scripts/watch_github_runs.py
Log: github_runs.log (appended to, never overwritten) in the repo root.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO = "khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON"
TOKEN = os.environ.get("GITHUB_TOKEN")
POLL_SECONDS = 60
ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / ".github_watch_state.json"
LOG_FILE = ROOT / "github_runs.log"


def fetch_runs():
    url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page=20"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)["workflow_runs"]


def load_seen():
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()


def save_seen(seen):
    STATE_FILE.write_text(json.dumps(sorted(seen)))


def format_entry(run):
    when = datetime.now().astimezone().isoformat(timespec="seconds")
    triggered = datetime.fromisoformat(run["run_started_at"]).astimezone().isoformat(timespec="seconds")
    if run["status"] == "completed":
        state = f"finished — {run['conclusion']}"
    else:
        state = f"started — {run['status']}"
    return (
        f"[{when}] {run['name']} run #{run['run_number']} {state} "
        f"(triggered {triggered}) {run['html_url']}\n"
    )


def main():
    if not TOKEN:
        print(
            "GITHUB_TOKEN not set in .env — add a token with 'Actions: Read-only' access "
            "to poll this private repo's workflow runs. See github.com/settings/tokens."
        )
        sys.exit(1)

    seen = load_seen()
    print(f"Watching {REPO} for Actions runs, polling every {POLL_SECONDS}s. Appending to {LOG_FILE}")

    while True:
        try:
            runs = fetch_runs()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"Poll failed ({exc}), retrying next cycle.")
            time.sleep(POLL_SECONDS)
            continue

        new_entries = [run for run in runs if f"{run['id']}:{run['status']}" not in seen]

        if new_entries:
            with open(LOG_FILE, "a") as f:
                for run in sorted(new_entries, key=lambda r: r["run_started_at"]):
                    f.write(format_entry(run))
                    seen.add(f"{run['id']}:{run['status']}")
            save_seen(seen)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

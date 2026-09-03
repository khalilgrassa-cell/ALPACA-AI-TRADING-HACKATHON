"""Runs exit management alone — no universe scan, no new candidates, no per-candidate Sentiment/
Strategy/Risk/Order chain. Protecting an already-open position from an adverse intraday move is
more time-sensitive than finding new entries, so this is meant to be scheduled more often (every
10 minutes) than the full orchestrator.py cycle (every 30 minutes, which also runs its own exit
check at the end) — see .github/workflows/exit-management-v2.yml."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))

# 2026-09-03: this was meant to isolate this cycle's Groq usage from the every-30-minutes full
# trading cycle's, since sharing GROQ_API_KEY meant both drew against the same per-model rate-limit
# bucket — observed live, running concurrently was enough to keep openai/gpt-oss-120b permanently
# exhausted, forcing every turn of every agent to pay out its full retry-then-fallback delay.
# Since confirmed NOT to actually help: Groq enforces these limits at the *organization* level, not
# per API key, so a second key draws on the identical shared budget regardless. Left in place since
# it's harmless (falls back to the shared key if unset) — the real mitigations are this cycle's
# 10-minute interval (widened from 5) and a lower TOP_N_HOTTEST on the trading cycle's own side.
if os.environ.get("GROQ_API_KEY_EXITS"):
    os.environ["GROQ_API_KEY"] = os.environ["GROQ_API_KEY_EXITS"]

from momentum_strategy import EXECUTE_EXITS
from mcp_client import connect

from market_conditions_agent import check_market_conditions
from trading_agent import manage_exits


async def main():
    async with connect() as session:
        try:
            conditions_result = await check_market_conditions(session)
            print(conditions_result)

            if not conditions_result.get("market_open", False):
                print("Market is closed — skipping exit check.")
            else:
                exits_result = await manage_exits(session, execute_exits=EXECUTE_EXITS)
                print(exits_result)
        except Exception as exc:
            print(f"ERROR: exit-management cycle aborted — {exc}")


if __name__ == "__main__":
    asyncio.run(main())
    # See orchestrator.py's identical guard: a timed-out Groq call leaves its worker thread
    # running in the background, which would otherwise block process exit until it finishes.
    # Flush first — os._exit() skips the stdio flush that would otherwise happen on normal
    # interpreter shutdown, and stdout is fully block-buffered whenever it isn't a terminal
    # (CI logs, Docker, a backgrounded shell), so every print() above would otherwise vanish.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

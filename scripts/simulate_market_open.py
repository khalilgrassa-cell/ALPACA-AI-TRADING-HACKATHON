"""Runs the real orchestrator.py pipeline locally, exactly as it would run live, except the
market-open gate is forced to True — for testing the pipeline outside real market hours without
touching GitHub Actions or waiting for the next session.

This is not a separate, simplified test harness: it imports and runs the actual production
orchestrator.main(), unmodified, so every stage (universe scan, Sentiment -> Strategy -> Risk ->
Trading, exit management) behaves identically to a real scheduled cycle — including submitting
real paper orders on Alpaca if the Risk Agent decides to trade. There is no dry-run flag here;
whatever the real pipeline would do, this does too.

Run: python scripts/simulate_market_open.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent / "multi_agent"))

import orchestrator

SIMULATED_CONDITIONS = {
    "market_open": True,
    "reasoning": "SIMULATED for after-hours local testing — real market status ignored.",
}


async def main():
    orchestrator.check_market_conditions = AsyncMock(return_value=SIMULATED_CONDITIONS)
    print("=== SIMULATION: market-open check forced to True, everything else is the real pipeline ===\n")
    await orchestrator.main()


if __name__ == "__main__":
    asyncio.run(main())

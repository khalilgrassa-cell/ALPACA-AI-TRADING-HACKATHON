"""Universe scanner: a deterministic (no-LLM) batch screen across the whole trading universe.

Fetches bars for every symbol in one MCP call, then runs the pure momentum screen locally.
This intentionally isn't an LLM agent — "does this cross a numeric threshold" is exact math,
not judgment, and running it through a model per symbol would multiply cost/latency by the size
of the universe for no benefit. The LLM-driven agents (news, risk, trading) only run on the
short candidate list this produces."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from mcp_client import mcp_data, mcp_error
from momentum_strategy import MOMENTUM_WINDOW, UNIVERSE, screen_universe

BARS_LOOKBACK_DAYS = 30
# Comfortably above len(universe) * ~21 trading days, so Alpaca never silently truncates the
# batch response — the default get_stock_bars limit (1000) is nowhere near enough for ~170 symbols.
BARS_LIMIT = 10000


async def scan_universe(mcp_session, symbols=None, days=BARS_LOOKBACK_DAYS):
    symbols = symbols if symbols is not None else UNIVERSE
    result = await mcp_session.call_tool(
        "get_stock_bars",
        {"symbols": ",".join(symbols), "timeframe": "1Day", "days": days, "limit": BARS_LIMIT},
    )
    data = mcp_data(result)
    if data is None:
        raise RuntimeError(f"get_stock_bars failed — {mcp_error(result)}")

    closes_by_symbol = {
        symbol: [bar["c"] for bar in bars]
        for symbol, bars in data["bars"].items()
        if len(bars) >= MOMENTUM_WINDOW + 1
    }
    return screen_universe(closes_by_symbol)


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from mcp_client import connect

    async with connect() as session:
        candidates = await scan_universe(session)
        print(f"{len(candidates)} candidate(s) out of {len(UNIVERSE)} symbols screened:")
        for c in candidates:
            print(f"  {c['symbol']}: {c['signal']} ({c['momentum_pct']:+.2f}%)")


if __name__ == "__main__":
    asyncio.run(_main())

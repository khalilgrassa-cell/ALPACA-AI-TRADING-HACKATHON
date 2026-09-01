"""Market-conditions check: deterministic (no LLM) — is_open is a plain boolean field on
get_clock's response, not a judgment call, so running it through a model only adds cost, latency,
and a single point of failure for zero benefit. Same reasoning as universe_scanner.py."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
from mcp_client import mcp_data, mcp_error


async def check_market_conditions(mcp_session):
    result = await mcp_session.call_tool("get_clock", {})
    clock = mcp_data(result)
    if clock is None:
        raise RuntimeError(f"get_clock failed — {mcp_error(result)}")
    market_open = clock["is_open"]
    reasoning = (
        f"get_clock reports is_open={market_open} "
        f"(next_open={clock['next_open']}, next_close={clock['next_close']})"
    )
    return {"market_open": market_open, "reasoning": reasoning}


async def _main():
    from mcp_client import connect

    async with connect() as session:
        print(await check_market_conditions(session))


if __name__ == "__main__":
    asyncio.run(_main())

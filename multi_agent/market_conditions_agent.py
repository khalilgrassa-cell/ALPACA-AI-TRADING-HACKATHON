"""Market-conditions agent: checks whether the market is open before the pipeline trades."""
import asyncio
import sys
from pathlib import Path

from llm_tools import parse_json_response, run_agent

SYSTEM_PROMPT = """You are the market-conditions agent in a multi-agent options trading system.

Your job: call get_clock to check whether the market is currently open. Call get_calendar too if \
you need to check upcoming holidays or early closes.

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"market_open": boolean, "reasoning": string}
"""


async def check_market_conditions(mcp_session):
    user_prompt = "Check whether the US equities market is open right now."
    text, _ = await run_agent(
        SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names={"get_clock", "get_calendar"}, local_tools=[],
    )
    return parse_json_response(text, required_keys={"market_open", "reasoning"})


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from mcp_client import connect

    async with connect() as session:
        print(await check_market_conditions(session))


if __name__ == "__main__":
    asyncio.run(_main())

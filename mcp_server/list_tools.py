"""Dumps the input schema for every MCP tool the trading agent relies on."""
import asyncio
import json

from mcp_client import connect

WANTED_TOOLS = [
    "get_account_info",
    "get_stock_bars",
    "get_option_chain",
    "place_option_order",
    "get_all_positions",
    "close_position",
]


async def main():
    async with connect() as session:
        tools_result = await session.list_tools()
        tool_map = {tool.name: tool for tool in tools_result.tools}
        for name in WANTED_TOOLS:
            dumped = tool_map[name].model_dump()
            schema = dumped.get("inputSchema") or dumped.get("input_schema")
            print(f"--- {name} ---")
            print(json.dumps(schema, indent=2))
            print()


if __name__ == "__main__":
    asyncio.run(main())

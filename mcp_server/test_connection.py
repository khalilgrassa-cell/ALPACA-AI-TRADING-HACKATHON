"""Connectivity smoke test: launches the Alpaca MCP server, lists its tools, and confirms the account is reachable."""
import asyncio

from mcp_client import connect, mcp_data, mcp_error


async def main():
    async with connect() as session:
        tools_result = await session.list_tools()
        print(f"Connected. {len(tools_result.tools)} tools available.")
        for tool in tools_result.tools:
            print(f"  {tool.name}")

        result = await session.call_tool("get_account_info", {})
        account = mcp_data(result)
        if account is None:
            print(f"\nERROR: get_account_info failed — {mcp_error(result)}")
            return
        print()
        print(f"Account {account['account_number']} | status {account['status']} | equity ${account['equity']}")


if __name__ == "__main__":
    asyncio.run(main())

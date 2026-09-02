"""Trading agent: submits risk-gated orders (per candidate) and manages exits on open positions
(once per cycle, across the whole portfolio — a held position doesn't stop needing an exit check
just because its symbol didn't signal again today)."""
import asyncio
import sys
from pathlib import Path

from llm_tools import parse_json_response, run_agent
from local_tools import check_exit_rule

ORDER_SYSTEM_PROMPT = """You are the order-submission agent in a multi-agent options trading system.

You receive a specific option contract, quantity, and position_intent that risk management has \
already approved for one leg of a strategy (a plain single-leg strategy has just one leg; a \
risk-reversal combo submits its long and short legs as two separate calls to you). Your job: call \
place_option_order to submit that exact contract at that exact quantity and position_intent \
(market order, day time-in-force).

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"order_submitted": boolean, "order_result": object | null, "reasoning": string}
"""

EXIT_SYSTEM_PROMPT = """You are the position-management agent in a multi-agent options trading system.

Your job: call get_all_positions to see every currently open position across the whole portfolio, \
then for each one, call check_exit_rule to see if it should be closed (TAKE_PROFIT, STOP_LOSS, or \
TIME_EXIT) — do not judge P&L or days-to-expiration yourself. If execute_exits is true and \
check_exit_rule returned a non-null reason, call close_position for that symbol. If execute_exits \
is false, do not call close_position — just report what you would have closed.

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"open_positions": integer, "exits": [{"symbol": string, "reason": string | null, "closed": boolean}], "reasoning": string}
"""


async def submit_order(mcp_session, chosen_contract, qty, side="long"):
    intent = "buy_to_open" if side == "long" else "sell_to_open"
    user_prompt = (
        f"Submit {qty} contract(s) of {chosen_contract['symbol']} at market, day time-in-force, "
        f"position_intent={intent}."
    )
    text, _ = await run_agent(
        ORDER_SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names={"place_option_order"}, local_tools=[],
        required_keys={"order_submitted", "order_result", "reasoning"},
    )
    return parse_json_response(text, required_keys={"order_submitted", "order_result", "reasoning"})


async def manage_exits(mcp_session, execute_exits=False):
    user_prompt = f"execute_exits={execute_exits} (only call close_position if this is true)."
    text, _ = await run_agent(
        EXIT_SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names={"get_all_positions", "close_position"}, local_tools=[check_exit_rule],
        required_keys={"open_positions", "exits", "reasoning"},
    )
    return parse_json_response(text, required_keys={"open_positions", "exits", "reasoning"})


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from mcp_client import connect

    async with connect() as session:
        print(await manage_exits(session, execute_exits=False))


if __name__ == "__main__":
    asyncio.run(_main())

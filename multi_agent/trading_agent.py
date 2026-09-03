"""Trading agent: submits risk-gated orders (per candidate) and manages exits on open positions
(once per cycle, across the whole portfolio -- a held position doesn't stop needing an exit check
just because its symbol didn't signal again today).

Deterministic, not an LLM call. 2026-09-03: both submit_order and manage_exits used to be
Groq-hosted agents that just glued together a fixed sequence of already-deterministic tool
calls -- place_option_order with an exact contract/qty/side computed upstream, or
get_all_positions -> check_exit_rule -> (maybe) close_position per position -- with zero judgment
anywhere in either sequence (the same reasoning risk_agent.py's rewrite already applied: see that
file's docstring). Rewritten as plain code: removes more calls from Groq's shared,
organization-wide rate-limit budget, and removes any chance of an LLM inferring the wrong
buy/sell side on a real order submission (the original prompt never even told the model which
"side" value paired with which position_intent -- it had to infer that correctly on every call)."""
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from mcp_client import mcp_data, mcp_error
from momentum_strategy import MIN_HOLD_MINUTES_BEFORE_STOP_LOSS

from local_tools import check_exit_rule


def _unwrap_list(data):
    """The Alpaca MCP server nests list-shaped results under a "result" key for some tools
    (get_all_positions, get_orders) but not others (get_account_info) -- unwrap defensively."""
    return data.get("result", data) if isinstance(data, dict) else data


async def submit_order(mcp_session, chosen_contract, qty, side="long"):
    intent = "buy_to_open" if side == "long" else "sell_to_open"
    order_side = "buy" if side == "long" else "sell"
    result = await mcp_session.call_tool("place_option_order", {
        "symbol": chosen_contract["symbol"],
        "qty": str(qty),
        "side": order_side,
        "position_intent": intent,
        "type": "market",
        "time_in_force": "day",
    })
    order_result = mcp_data(result)
    if order_result is None:
        return {
            "order_submitted": False, "order_result": None,
            "reasoning": f"place_option_order failed — {mcp_error(result)}",
        }
    return {
        "order_submitted": True, "order_result": order_result,
        "reasoning": f"Submitted {qty} contract(s) of {chosen_contract['symbol']} ({intent}).",
    }


async def _minutes_since_opened(mcp_session, symbol):
    """Looks up the most recent opening fill (buy_to_open or sell_to_open) for this exact option
    contract, to gate STOP_LOSS on a minimum holding period (see MIN_HOLD_MINUTES_BEFORE_STOP_LOSS).
    Returns None if it can't be determined -- treated as "held long enough" so a lookup failure
    never blocks a real stop-loss."""
    result = await mcp_session.call_tool(
        "get_orders", {"status": "closed", "symbols": symbol, "direction": "desc", "limit": 10},
    )
    orders = mcp_data(result)
    if orders is None:
        return None
    opening_fill = next(
        (
            order for order in _unwrap_list(orders)
            if order.get("position_intent") in ("buy_to_open", "sell_to_open") and order.get("filled_at")
        ),
        None,
    )
    if opening_fill is None:
        return None
    opened_at = datetime.fromisoformat(opening_fill["filled_at"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - opened_at).total_seconds() / 60


async def manage_exits(mcp_session, execute_exits=False):
    positions_result = await mcp_session.call_tool("get_all_positions", {})
    positions_data = mcp_data(positions_result)
    if positions_data is None:
        return {
            "open_positions": 0, "exits": [],
            "reasoning": f"get_all_positions failed — {mcp_error(positions_result)}",
        }
    positions = _unwrap_list(positions_data)

    exits = []
    for position in positions:
        reason = check_exit_rule.func(position)["reason"]

        if reason == "STOP_LOSS":
            minutes_held = await _minutes_since_opened(mcp_session, position["symbol"])
            if minutes_held is not None and minutes_held < MIN_HOLD_MINUTES_BEFORE_STOP_LOSS:
                # Too soon after opening for a leveraged option's own noisy price to reflect a
                # confirmed move rather than bid/ask spread or IV noise — hold instead.
                reason = None

        closed = False
        if reason is not None and execute_exits:
            close_result = await mcp_session.call_tool("close_position", {"symbol_or_asset_id": position["symbol"]})
            closed = mcp_data(close_result) is not None

        exits.append({"symbol": position["symbol"], "reason": reason, "closed": closed})

    flagged = sum(1 for exit_ in exits if exit_["reason"] is not None)
    return {
        "open_positions": len(positions),
        "exits": exits,
        "reasoning": f"Checked {len(positions)} open position(s); {flagged} flagged for exit.",
    }


async def _main():
    from mcp_client import connect

    async with connect() as session:
        print(await manage_exits(session, execute_exits=False))


if __name__ == "__main__":
    asyncio.run(_main())

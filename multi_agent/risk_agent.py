"""Risk agent: selects the option contract(s) a chosen strategy needs and computes a risk-gated
position size — for both plain single-leg strategies and 2-leg risk-reversal combos."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from momentum_strategy import STRATEGY_LEGS

from llm_tools import parse_json_response, run_agent
from local_tools import calculate_combo_position_size, calculate_position_size, select_option_contract

SYSTEM_PROMPT = """You are the risk-management agent in a multi-agent options trading system.

You receive a strategy (LONG_CALL, LONG_PUT, RISK_REVERSAL_BULLISH, RISK_REVERSAL_BEARISH, or \
NO_TRADE) chosen by the strategy agent, plus the leg(s) it needs — each leg is a side \
(long/short) and a directional signal (BUY_CALL or BUY_PUT) telling you which option type and \
strike offset to pick for that leg. Your job:
1. If the strategy is NO_TRADE, skip straight to the final answer with legs: [], qty: 0, should_trade: false.
2. Otherwise, call get_account_info for current equity, then get_option_chain for the underlying's \
option contracts in the given expiration window. This account has no OPRA market-data agreement, \
so always pass feed="indicative" — the default feed will fail with a 403. Always also pass \
strike_price_gte and strike_price_lte narrowed to within 10% of the current price, and limit=50 — \
fetching the full chain wastes tokens on strikes nowhere near what you'll actually pick. Do not \
filter by option type — the same fetched chain covers both legs of a combo.
3. For each leg, call select_option_contract with the fetched contracts, that leg's own signal, \
and the current price, to pick its strike.
4. If there is one leg, call calculate_position_size with the account equity and that leg's ask \
price to get the exact risk-gated quantity. If there are two legs (a combo), call \
calculate_combo_position_size with the long leg's ask, the short leg's bid, and the strategy, \
instead — do not compute either by hand. Use the same quantity for both legs of a combo.
5. If any leg could not be chosen (select_option_contract returned chosen: null), treat the whole \
trade as should_trade: false.

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"strategy": string, "legs": [{"symbol": string, "ask": number, "bid": number, "side": "long" | "short"}], "qty": integer, "should_trade": boolean, "reasoning": string}
legs must be an empty list when should_trade is false.
"""


async def assess_risk(mcp_session, symbol, strategy, current_price, min_dte, max_dte):
    legs = STRATEGY_LEGS.get(strategy, ())
    legs_desc = "; ".join(f"{side} leg: {signal}" for side, signal in legs) or "none"
    user_prompt = (
        f"Strategy from the strategy agent: {strategy} on {symbol} at current price ${current_price:.2f}. "
        f"Legs needed: {legs_desc}. "
        f"Use an expiration window of {min_dte}-{max_dte} days to expiration when fetching the option chain."
    )
    # Only offer the sizing tool the leg count actually needs — every local/MCP tool schema is
    # resent every turn (see llm_tools.py), so a combo call that never needs
    # calculate_position_size (and vice versa) would otherwise pay for both schemas on every
    # single-leg call too. Keeping this lean matters more now than it used to: a combo's extra
    # leg (two contract selections, a second quote in context) already pushes a request closer to
    # the account's fixed per-minute token ceiling — see llm_tools._is_oversized_request_error.
    sizing_tool = calculate_combo_position_size if len(legs) == 2 else calculate_position_size
    text, _ = await run_agent(
        SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names={"get_account_info", "get_option_chain"},
        local_tools=[select_option_contract, sizing_tool],
    )
    return parse_json_response(text, required_keys={"strategy", "legs", "qty", "should_trade", "reasoning"})


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
    from momentum_strategy import MAX_DTE, MIN_DTE, SYMBOL
    from mcp_client import connect

    async with connect() as session:
        print(await assess_risk(session, SYMBOL, "LONG_CALL", 716.43, MIN_DTE, MAX_DTE))


if __name__ == "__main__":
    asyncio.run(_main())

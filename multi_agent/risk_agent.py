"""Risk agent: selects the option contract(s) a chosen strategy needs and computes a risk-gated
position size -- for both plain single-leg strategies and 2-leg risk-reversal combos.

Deterministic, not an LLM call. 2026-09-03: this used to be a Groq-hosted agent that called
get_account_info -> get_option_chain -> select_option_contract (per leg) ->
calculate_position_size/calculate_combo_position_size in a fixed order every time -- pure
plumbing over exact math, with no judgment call anywhere in the sequence (the same reasoning
universe_scanner.py's screen already applies to signal generation). Rewritten as plain code after
live evidence that Groq enforces its per-model rate limits at the *organization* level, not per
API key -- every LLM call anywhere in the pipeline draws on the same shared, exhaustible budget,
so removing an entire agent's calls per candidate (with zero loss of quality, since there was
nothing here an LLM was actually deciding) is a direct, safe cut to that pressure."""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from mcp_client import mcp_data, mcp_error
from momentum_strategy import CHAIN_STRIKE_RANGE_PCT, GET_OPTION_CHAIN_LIMIT, STRATEGY_LEGS

from local_tools import calculate_combo_position_size, calculate_position_size, select_option_contract


def _no_trade(strategy, reasoning):
    return {"strategy": strategy, "legs": [], "qty": 0, "should_trade": False, "reasoning": reasoning}


async def assess_risk(mcp_session, symbol, strategy, current_price, min_dte, max_dte):
    legs = STRATEGY_LEGS.get(strategy, ())
    if not legs:
        return _no_trade(strategy, "Strategy is NO_TRADE, no trade to execute.")

    account_result = await mcp_session.call_tool("get_account_info", {})
    account = mcp_data(account_result)
    if account is None:
        return _no_trade(strategy, f"get_account_info failed — {mcp_error(account_result)}")
    equity = float(account["equity"])

    today = date.today()
    chain_result = await mcp_session.call_tool("get_option_chain", {
        "underlying_symbol": symbol,
        # This account has no OPRA market-data agreement — the default feed fails with a 403.
        "feed": "indicative",
        "strike_price_gte": current_price * (1 - CHAIN_STRIKE_RANGE_PCT),
        "strike_price_lte": current_price * (1 + CHAIN_STRIKE_RANGE_PCT),
        "expiration_date_gte": (today + timedelta(days=min_dte)).isoformat(),
        "expiration_date_lte": (today + timedelta(days=max_dte)).isoformat(),
        "limit": GET_OPTION_CHAIN_LIMIT,
    })
    chain_data = mcp_data(chain_result)
    if chain_data is None:
        return _no_trade(strategy, f"get_option_chain failed — {mcp_error(chain_result)}")

    contracts = [
        {
            "symbol": contract_symbol,
            "ask": snapshot.get("latestQuote", {}).get("ap", 0),
            "bid": snapshot.get("latestQuote", {}).get("bp", 0),
        }
        for contract_symbol, snapshot in chain_data.get("snapshots", {}).items()
    ]

    chosen_legs = []
    for side, signal in legs:
        chosen = select_option_contract.func(contracts, signal, current_price)["chosen"]
        if chosen is None:
            return _no_trade(
                strategy, f"No qualifying {signal} contract found for the {side} leg within the DTE/strike window.",
            )
        chosen_legs.append({"symbol": chosen["symbol"], "ask": chosen["ask"], "bid": chosen["bid"], "side": side})

    if len(chosen_legs) == 1:
        sizing = calculate_position_size.func(equity, chosen_legs[0]["ask"], legs[0][1])
    else:
        long_leg = next(leg for leg in chosen_legs if leg["side"] == "long")
        short_leg = next(leg for leg in chosen_legs if leg["side"] == "short")
        sizing = calculate_combo_position_size.func(equity, long_leg["ask"], short_leg["bid"], strategy)

    should_trade = sizing["should_trade"]
    picked = [leg["symbol"] for leg in chosen_legs]
    reasoning = (
        f"Selected {picked}, sized to {sizing['qty']} contract(s) "
        f"(risk-gated to ${sizing['risk_dollars']:.2f} of equity, contract cost ${sizing['contract_cost']:.2f})."
        if should_trade else
        f"Selected {picked}, but the risk-gated quantity is 0 (contract cost ${sizing['contract_cost']:.2f} "
        f"exceeds the risk budget) — trade not placed."
    )
    return {
        "strategy": strategy,
        "legs": chosen_legs if should_trade else [],
        "qty": sizing["qty"],
        "should_trade": should_trade,
        "reasoning": reasoning,
    }


async def _main():
    from mcp_client import connect
    from momentum_strategy import MAX_DTE, MIN_DTE, SYMBOL

    async with connect() as session:
        print(await assess_risk(session, SYMBOL, "LONG_CALL", 716.43, MIN_DTE, MAX_DTE))


if __name__ == "__main__":
    asyncio.run(_main())

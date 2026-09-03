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
nothing here an LLM was actually deciding) is a direct, safe cut to that pressure.

Sequence today: get_account_info -> get_option_chain once per option type actually needed (see
_fetch_otm_chain -- split per type and biased toward each type's own OTM side, since a single
shared-range fetch was observed live getting exhausted by one side before ever reaching genuinely
OTM strikes on the other) -> get_option_snapshot for delta (see _attach_deltas, best-effort: a
failed call falls back to OTM_PCT-based selection rather than blocking the trade) ->
select_option_contract per leg -> calculate_position_size/calculate_combo_position_size."""
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


async def _fetch_otm_chain(mcp_session, symbol, option_type, current_price, min_dte, max_dte):
    """Fetches one option type's chain, biased toward its own out-of-the-money side (calls: at or
    above current price; puts: at or below). 2026-09-03: a single shared-range fetch across both
    types was observed live getting exhausted by one side before ever reaching genuinely OTM
    strikes -- GET_OPTION_CHAIN_LIMIT applies to the total response, not per type, so a dense
    same-day strike ladder (e.g. QQQ's $1 increments) can fill the whole limit with in-the-money
    calls and return zero puts at all. Splitting per type and biasing each fetch's range toward its
    own OTM side fixes both problems at once. Returns (contracts, error) -- error is None on
    success, contracts is [] (not None) when the call succeeded but nothing matched."""
    today = date.today()
    if option_type == "call":
        strike_gte, strike_lte = current_price, current_price * (1 + CHAIN_STRIKE_RANGE_PCT)
    else:
        strike_gte, strike_lte = current_price * (1 - CHAIN_STRIKE_RANGE_PCT), current_price
    result = await mcp_session.call_tool("get_option_chain", {
        "underlying_symbol": symbol,
        # This account has no OPRA market-data agreement — the default feed fails with a 403.
        "feed": "indicative",
        "type": option_type,
        "strike_price_gte": strike_gte,
        "strike_price_lte": strike_lte,
        "expiration_date_gte": (today + timedelta(days=min_dte)).isoformat(),
        "expiration_date_lte": (today + timedelta(days=max_dte)).isoformat(),
        "limit": GET_OPTION_CHAIN_LIMIT,
    })
    data = mcp_data(result)
    if data is None:
        return None, mcp_error(result)
    contracts = [
        {
            "symbol": contract_symbol,
            "ask": snapshot.get("latestQuote", {}).get("ap", 0),
            "bid": snapshot.get("latestQuote", {}).get("bp", 0),
        }
        for contract_symbol, snapshot in data.get("snapshots", {}).items()
    ]
    return contracts, None


async def _attach_deltas(mcp_session, contracts):
    """Best-effort: fetches each contract's delta via get_option_snapshot and adds it in place as
    contracts[i]["delta"]. A failed or partial response is never fatal -- select_contract() falls
    back to OTM_PCT-based selection for any contract still missing delta afterward, so a snapshot
    outage degrades contract quality slightly rather than blocking the trade."""
    symbols = [c["symbol"] for c in contracts]
    snapshot_result = await mcp_session.call_tool("get_option_snapshot", {
        "symbols": ",".join(symbols),
        "feed": "indicative",
        "limit": len(symbols),
    })
    snapshots = mcp_data(snapshot_result)
    if snapshots is None:
        return
    for contract in contracts:
        snapshot = snapshots.get("snapshots", {}).get(contract["symbol"]) or {}
        contract["delta"] = (snapshot.get("greeks") or {}).get("delta")


async def assess_risk(mcp_session, symbol, strategy, current_price, min_dte, max_dte):
    legs = STRATEGY_LEGS.get(strategy, ())
    if not legs:
        return _no_trade(strategy, "Strategy is NO_TRADE, no trade to execute.")

    account_result = await mcp_session.call_tool("get_account_info", {})
    account = mcp_data(account_result)
    if account is None:
        return _no_trade(strategy, f"get_account_info failed — {mcp_error(account_result)}")
    equity = float(account["equity"])

    needed_types = sorted({"call" if signal == "BUY_CALL" else "put" for _, signal in legs})
    contracts = []
    for option_type in needed_types:
        fetched, error = await _fetch_otm_chain(mcp_session, symbol, option_type, current_price, min_dte, max_dte)
        if fetched is None:
            return _no_trade(strategy, f"get_option_chain ({option_type}) failed — {error}")
        contracts.extend(fetched)

    if contracts:
        await _attach_deltas(mcp_session, contracts)

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

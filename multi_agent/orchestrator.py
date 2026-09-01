"""Runs the Market-Conditions -> Universe Scan (top hottest/trending symbols) -> [Sentiment ->
Strategy -> Risk -> Trader per candidate] -> Exit Management pipeline for one decision cycle.

No reporting/logging stage — every decision is printed to stdout as it happens (see the flush
note at the bottom of this file for why that reaches CI/Docker logs correctly), and actual fills
are tracked in Alpaca's own UI rather than a separate local log."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))

from momentum_strategy import EXECUTE_EXITS, MAX_CYCLE_RISK_PCT, MAX_DTE, MIN_DTE, UNIVERSE, estimate_trade_cost
from mcp_client import connect, mcp_data, mcp_error

from market_conditions_agent import check_market_conditions
from risk_agent import assess_risk
from sentiment_agent import check_sentiment
from strategy_agent import choose_strategy
from trading_agent import manage_exits, submit_order
from universe_scanner import scan_universe


async def get_equity(session):
    result = await session.call_tool("get_account_info", {})
    account = mcp_data(result)
    if account is None:
        raise RuntimeError(f"get_account_info failed — {mcp_error(result)}")
    return float(account["equity"])


async def submit_legs(session, legs, qty):
    """Submits each leg's order in turn. A combo's short leg is only submitted once its long leg
    has actually gone through — this isn't an atomic multi-leg order (place_option_order is
    single-leg), just a sequential safeguard against opening a naked short with no matching long
    leg if the first order fails."""
    orders = []
    for leg in legs:
        order_result = await submit_order(session, leg, qty, side=leg["side"])
        orders.append({"leg": leg, "order": order_result})
        if leg["side"] == "long" and not order_result["order_submitted"]:
            break
    return orders


async def process_candidate(session, candidate, risk_budget, committed):
    symbol, current_price = candidate["symbol"], candidate["current_price"]
    print(f"\n--- {symbol} ---")

    print("=== Sentiment Agent ===")
    sentiment_result = await check_sentiment(session, symbol, candidate["signal"])
    print(sentiment_result)

    print("=== Strategy Agent ===")
    strategy_result = await choose_strategy(
        session, symbol, sentiment_result["signal"], sentiment_result["sentiment"], candidate["momentum_pct"],
    )
    print(strategy_result)

    print("=== Risk Agent ===")
    risk_result = await assess_risk(session, symbol, strategy_result["strategy"], current_price, MIN_DTE, MAX_DTE)
    print(risk_result)

    orders = None
    if risk_result["should_trade"]:
        trade_cost = estimate_trade_cost(risk_result["legs"], risk_result["qty"])
        if committed + trade_cost > risk_budget:
            print(f"Cycle risk cap reached (${committed:.2f} committed of ${risk_budget:.2f} budget) — skipping {symbol}.")
            risk_result["should_trade"] = False
            risk_result["reasoning"] += " [cycle risk cap reached — not executed]"
        else:
            committed += trade_cost
            print("=== Trading Agent (order) ===")
            orders = await submit_legs(session, risk_result["legs"], risk_result["qty"])
            print(orders)

    return {
        "symbol": symbol, "sentiment": sentiment_result, "strategy": strategy_result,
        "risk": risk_result, "orders": orders,
    }, committed


async def main():
    async with connect() as session:
        try:
            print("=== Market Conditions Check ===")
            conditions_result = await check_market_conditions(session)
            print(conditions_result)

            if not conditions_result.get("market_open", False):
                print("\nMarket is closed — skipping the rest of the cycle.")
                return

            print(f"\n=== Universe Scan ({len(UNIVERSE)} symbols) ===")
            candidates = await scan_universe(session)
            print(f"{len(candidates)} hottest/trending candidate(s): {[c['symbol'] for c in candidates]}")

            trades = []
            if candidates:
                equity = await get_equity(session)
                risk_budget = equity * MAX_CYCLE_RISK_PCT
                committed = 0.0
                for candidate in candidates:
                    try:
                        trade, committed = await process_candidate(session, candidate, risk_budget, committed)
                    except Exception as exc:
                        print(f"ERROR: {candidate['symbol']} aborted — {exc}")
                        trade = {"symbol": candidate["symbol"], "error": str(exc)}
                    trades.append(trade)
            else:
                print("No symbols crossed the momentum threshold this cycle.")

            print("\n=== Trading Agent (exits) ===")
            exits_result = await manage_exits(session, execute_exits=EXECUTE_EXITS)
            print(exits_result)

            submitted = [
                t for t in trades
                if t.get("orders") and all(o["order"]["order_submitted"] for o in t["orders"])
            ]
            failed = [t for t in trades if "error" in t]
            print("\n=== Summary ===")
            print(f"Candidates evaluated: {len(candidates)} | Orders submitted: {len(submitted)} "
                  f"| Symbols traded: {[t['symbol'] for t in submitted]} | Failed: {[t['symbol'] for t in failed]} "
                  f"| Open positions: {exits_result['open_positions']}")
        except Exception as exc:
            print(f"\nERROR: pipeline aborted — {exc}")


if __name__ == "__main__":
    asyncio.run(main())
    # A timed-out Groq call (see llm_tools.CALL_TIMEOUT_SECONDS_PER_MODEL) leaves its worker thread running
    # in the background even after we've given up on it — Python's default executor otherwise
    # blocks process exit until every such thread finishes, which could stall a scheduled run
    # well past when its actual work is done. We're finished and don't need graceful cleanup.
    # os._exit() skips the normal interpreter shutdown that would flush stdio buffers, and
    # Python fully block-buffers stdout whenever it isn't a terminal (every real deployment:
    # CI logs, Docker, a backgrounded shell) — without an explicit flush here, every print()
    # above is silently lost instead of reaching the log. Observed live: a full cycle's output
    # vanished except for the MCP server's own banner.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

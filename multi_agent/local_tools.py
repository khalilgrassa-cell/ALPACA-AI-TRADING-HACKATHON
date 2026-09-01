"""Deterministic calculator tools the agents call for exact math instead of doing arithmetic themselves.

Thin wrappers over ../strategy/momentum_strategy.py — the strategy logic itself lives there."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from momentum_strategy import (
    MAX_CONTRACTS, RISK_PCT,
    calculate_combo_position_size as _calculate_combo_position_size,
    calculate_position_size as _calculate_position_size,
    exit_reason, parse_contract, select_contract,
)

from llm_tools import LocalTool


def _select_option_contract(contracts, signal, current_price):
    parsed = [
        c for c in (
            parse_contract(contract["symbol"], {"latestQuote": {"ap": contract.get("ask", 0), "bp": contract.get("bid", 0)}})
            for contract in contracts
        ) if c is not None
    ]
    chosen = select_contract(parsed, signal, current_price)
    if chosen is None:
        return {"chosen": None}
    return {"chosen": {**chosen, "expiration": chosen["expiration"].isoformat()}}


select_option_contract = LocalTool(
    name="select_option_contract",
    description=(
        "Given raw option contract snapshots (symbol, ask, bid), the trading signal, and the "
        "underlying's current price, picks the contract closest to the target OTM strike within "
        "the configured DTE window. Returns {\"chosen\": null} if none qualify."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "contracts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "ask": {"type": "number"},
                        "bid": {"type": "number"},
                    },
                    "required": ["symbol", "ask", "bid"],
                },
            },
            "signal": {"type": "string", "enum": ["BUY_CALL", "BUY_PUT", "NO_TRADE"]},
            "current_price": {"type": "number"},
        },
        "required": ["contracts", "signal", "current_price"],
    },
    func=_select_option_contract,
)


calculate_position_size = LocalTool(
    name="calculate_position_size",
    description=(
        "Computes the risk-gated position size: caps contract quantity at risk_pct of equity and "
        "max_contracts, and reports whether the trade should be placed at all. Always use this "
        "instead of computing position size yourself."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "equity": {"type": "number"},
            "contract_ask": {"type": "number"},
            "signal": {"type": "string", "enum": ["BUY_CALL", "BUY_PUT", "NO_TRADE"]},
            "risk_pct": {"type": "number", "description": f"Fraction of equity to risk (default {RISK_PCT})."},
            "max_contracts": {"type": "integer", "description": f"Hard cap on contract count (default {MAX_CONTRACTS})."},
        },
        "required": ["equity", "contract_ask", "signal"],
    },
    func=_calculate_position_size,
)


calculate_combo_position_size = LocalTool(
    name="calculate_combo_position_size",
    description=(
        "Computes the risk-gated position size for a 2-leg (long+short) combo strategy, sizing "
        "off the net debit per contract (long leg's ask minus short leg's bid). Always use this "
        "instead of calculate_position_size for a RISK_REVERSAL_* strategy."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "equity": {"type": "number"},
            "long_ask": {"type": "number", "description": "The long leg's ask price."},
            "short_bid": {"type": "number", "description": "The short leg's bid price."},
            "strategy": {"type": "string", "enum": ["RISK_REVERSAL_BULLISH", "RISK_REVERSAL_BEARISH", "NO_TRADE"]},
            "risk_pct": {"type": "number", "description": f"Fraction of equity to risk (default {RISK_PCT})."},
            "max_contracts": {"type": "integer", "description": f"Hard cap on contract count (default {MAX_CONTRACTS})."},
        },
        "required": ["equity", "long_ask", "short_bid", "strategy"],
    },
    func=_calculate_combo_position_size,
)


def _check_exit_rule(position):
    return {"reason": exit_reason(position)}


check_exit_rule = LocalTool(
    name="check_exit_rule",
    description=(
        "Checks an open position's unrealized P&L and days-to-expiration against the "
        "TAKE_PROFIT/STOP_LOSS/TIME_EXIT thresholds. Returns {\"reason\": null} to hold. Always use "
        "this instead of judging P&L or expiration yourself."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "position": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "unrealized_plpc": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
        "required": ["position"],
    },
    func=_check_exit_rule,
)

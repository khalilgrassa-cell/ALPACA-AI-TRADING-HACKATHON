"""Unit tests for the deterministic tools the agents call — no network, API keys, or credentials required."""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_tools import calculate_combo_position_size, calculate_position_size, check_exit_rule, select_option_contract


def test_select_option_contract_picks_closest_call():
    exp = (date.today() + timedelta(days=10)).strftime("%y%m%d")
    contracts = [
        {"symbol": f"QQQ{exp}C00700000", "ask": 5.0, "bid": 4.9},
        {"symbol": f"QQQ{exp}C00730000", "ask": 1.0, "bid": 0.9},
    ]
    result = select_option_contract.func(contracts, "BUY_CALL", current_price=716.43)
    assert result["chosen"]["strike"] == 730.0


def test_select_option_contract_no_trade():
    result = select_option_contract.func([], "NO_TRADE", current_price=716.43)
    assert result["chosen"] is None


def test_select_option_contract_prefers_delta_when_present():
    exp = (date.today() + timedelta(days=10)).strftime("%y%m%d")
    contracts = [
        {"symbol": f"QQQ{exp}C00730000", "ask": 5.0, "bid": 4.9, "delta": 0.55},  # closest strike, wrong delta
        {"symbol": f"QQQ{exp}C00750000", "ask": 2.0, "bid": 1.9, "delta": 0.30},  # farther strike, target delta
    ]
    result = select_option_contract.func(contracts, "BUY_CALL", current_price=716.43)
    assert result["chosen"]["strike"] == 750.0


def test_calculate_position_size_gated():
    result = calculate_position_size.func(equity=100000, contract_ask=0.91, signal="BUY_CALL")
    assert result["should_trade"] is True
    assert result["qty"] > 0


def test_calculate_position_size_no_trade():
    result = calculate_position_size.func(equity=100000, contract_ask=0.91, signal="NO_TRADE")
    assert result["should_trade"] is False


def test_calculate_combo_position_size_gated():
    result = calculate_combo_position_size.func(
        equity=100000, long_ask=2.0, short_bid=1.0, strategy="RISK_REVERSAL_BULLISH",
    )
    assert result["should_trade"] is True
    assert result["qty"] > 0


def test_calculate_combo_position_size_no_trade():
    result = calculate_combo_position_size.func(
        equity=100000, long_ask=2.0, short_bid=1.0, strategy="NO_TRADE",
    )
    assert result["should_trade"] is False


def test_check_exit_rule_take_profit():
    result = check_exit_rule.func({"symbol": "QQQ260904C00731000", "unrealized_plpc": "0.55"})
    assert result["reason"] == "TAKE_PROFIT"


def test_check_exit_rule_hold():
    exp = (date.today() + timedelta(days=10)).strftime("%y%m%d")
    result = check_exit_rule.func({"symbol": f"QQQ{exp}C00731000", "unrealized_plpc": "0.03"})
    assert result["reason"] is None

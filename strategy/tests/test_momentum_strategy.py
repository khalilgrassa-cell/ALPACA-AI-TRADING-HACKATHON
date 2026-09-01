"""Unit tests for the momentum strategy's pure decision logic — no network or credentials required."""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from momentum_strategy import (
    NASDAQ_100, SP_100, STRATEGY_LEGS, STRATEGY_SIGNAL, TOP_N_HOTTEST, UNIVERSE,
    calculate_combo_position_size, calculate_momentum, calculate_position_size, estimate_trade_cost,
    exit_reason, parse_contract, screen_universe, select_contract,
)


def test_calculate_momentum_buy_call():
    closes = [100, 100, 100, 100, 100, 102]
    result = calculate_momentum(closes, window=5)
    assert result["signal"] == "BUY_CALL"
    assert round(result["momentum_pct"], 2) == 2.0


def test_calculate_momentum_buy_put():
    closes = [100, 100, 100, 100, 100, 98]
    result = calculate_momentum(closes, window=5)
    assert result["signal"] == "BUY_PUT"


def test_calculate_momentum_no_trade():
    closes = [100, 100, 100, 100, 100, 100.5]
    result = calculate_momentum(closes, window=5)
    assert result["signal"] == "NO_TRADE"


def test_parse_contract_call():
    parsed = parse_contract("QQQ260904C00731000", {"latestQuote": {"ap": 0.91, "bp": 0.90}})
    assert parsed["underlying"] == "QQQ"
    assert parsed["type"] == "C"
    assert parsed["strike"] == 731.0
    assert parsed["ask"] == 0.91


def test_parse_contract_works_for_any_underlying_in_the_universe():
    parsed = parse_contract("AAPL260904C00230000", {"latestQuote": {"ap": 1.0, "bp": 0.9}})
    assert parsed["underlying"] == "AAPL"
    assert parsed["strike"] == 230.0


def test_parse_contract_rejects_malformed_symbols():
    assert parse_contract("NOTANOPTIONSYMBOL", {}) is None
    assert parse_contract("AAPL26090XC00230000", {}) is None


def test_select_contract_picks_closest_otm_call():
    exp = (date.today() + timedelta(days=10)).strftime("%y%m%d")
    contracts = [
        parse_contract(f"QQQ{exp}C00700000", {"latestQuote": {"ap": 5.0, "bp": 4.9}}),
        parse_contract(f"QQQ{exp}C00730000", {"latestQuote": {"ap": 1.0, "bp": 0.9}}),
    ]
    chosen = select_contract(contracts, "BUY_CALL", current_price=716.43)
    assert chosen["strike"] == 730.0


def test_select_contract_excludes_out_of_window_expirations():
    exp = (date.today() + timedelta(days=40)).strftime("%y%m%d")
    contracts = [parse_contract(f"QQQ{exp}C00730000", {"latestQuote": {"ap": 1.0, "bp": 0.9}})]
    assert select_contract(contracts, "BUY_CALL", current_price=716.43) is None


def test_select_contract_no_trade_returns_none():
    assert select_contract([], "NO_TRADE", current_price=716.43) is None


def test_calculate_position_size_gated():
    result = calculate_position_size(equity=100000, contract_ask=0.91, signal="BUY_CALL")
    assert result["should_trade"] is True
    assert result["qty"] > 0


def test_calculate_position_size_no_contract():
    result = calculate_position_size(equity=100000, contract_ask=0, signal="BUY_CALL")
    assert result["should_trade"] is False
    assert result["qty"] == 0


def test_calculate_position_size_no_trade_signal():
    result = calculate_position_size(equity=100000, contract_ask=0.91, signal="NO_TRADE")
    assert result["should_trade"] is False


def test_exit_reason_take_profit():
    position = {"symbol": "QQQ260904C00731000", "unrealized_plpc": "0.55"}
    assert exit_reason(position) == "TAKE_PROFIT"


def test_exit_reason_stop_loss():
    position = {"symbol": "QQQ260904C00731000", "unrealized_plpc": "-0.60"}
    assert exit_reason(position) == "STOP_LOSS"


def test_exit_reason_time_exit():
    exp = (date.today() + timedelta(days=1)).strftime("%y%m%d")
    position = {"symbol": f"QQQ{exp}C00731000", "unrealized_plpc": "0.03"}
    assert exit_reason(position) == "TIME_EXIT"


def test_exit_reason_hold():
    exp = (date.today() + timedelta(days=10)).strftime("%y%m%d")
    position = {"symbol": f"QQQ{exp}C00731000", "unrealized_plpc": "0.03"}
    assert exit_reason(position) is None


def test_universe_is_deduplicated_union_of_both_indices():
    assert set(UNIVERSE) == set(NASDAQ_100) | set(SP_100)
    assert len(UNIVERSE) == len(set(UNIVERSE))


def test_universe_has_no_empty_or_duplicate_entries_within_each_index():
    assert len(NASDAQ_100) == len(set(NASDAQ_100))
    assert len(SP_100) == len(set(SP_100))
    assert all(NASDAQ_100) and all(SP_100)


def test_screen_universe_returns_only_signaling_symbols():
    closes_by_symbol = {
        "AAPL": [100, 100, 100, 100, 100, 102],  # BUY_CALL, +2%
        "MSFT": [100, 100, 100, 100, 100, 100.2],  # NO_TRADE
        "JPM": [100, 100, 100, 100, 100, 98],  # BUY_PUT, -2%
    }
    candidates = screen_universe(closes_by_symbol)
    symbols = {c["symbol"] for c in candidates}
    assert symbols == {"AAPL", "JPM"}
    assert all(c["signal"] != "NO_TRADE" for c in candidates)


def test_screen_universe_ranks_by_momentum_strength():
    closes_by_symbol = {
        "AAPL": [100, 100, 100, 100, 100, 101.5],  # +1.5%
        "JPM": [100, 100, 100, 100, 100, 105],  # +5%
    }
    candidates = screen_universe(closes_by_symbol)
    assert [c["symbol"] for c in candidates] == ["JPM", "AAPL"]


def test_screen_universe_caps_at_top_n_hottest():
    symbol_range = range(2, 70)
    closes_by_symbol = {f"SYM{i}": [100, 100, 100, 100, 100, 100 + i] for i in symbol_range}
    candidates = screen_universe(closes_by_symbol)
    assert len(candidates) == TOP_N_HOTTEST
    # The kept symbols really are the strongest-momentum ones, not an arbitrary slice.
    strongest = sorted(symbol_range, reverse=True)[:TOP_N_HOTTEST]
    assert {c["symbol"] for c in candidates} == {f"SYM{i}" for i in strongest}


def test_screen_universe_skips_symbols_with_insufficient_history():
    closes_by_symbol = {"AAPL": [100, 102]}
    assert screen_universe(closes_by_symbol) == []


def test_strategy_legs_single_leg_strategies():
    assert STRATEGY_LEGS["LONG_CALL"] == (("long", "BUY_CALL"),)
    assert STRATEGY_LEGS["LONG_PUT"] == (("long", "BUY_PUT"),)
    assert STRATEGY_LEGS["NO_TRADE"] == ()


def test_strategy_legs_risk_reversals_pair_opposite_option_types():
    assert STRATEGY_LEGS["RISK_REVERSAL_BULLISH"] == (("long", "BUY_CALL"), ("short", "BUY_PUT"))
    assert STRATEGY_LEGS["RISK_REVERSAL_BEARISH"] == (("long", "BUY_PUT"), ("short", "BUY_CALL"))


def test_strategy_signal_matches_each_strategys_direction():
    assert STRATEGY_SIGNAL["RISK_REVERSAL_BULLISH"] == "BUY_CALL"
    assert STRATEGY_SIGNAL["RISK_REVERSAL_BEARISH"] == "BUY_PUT"
    assert STRATEGY_SIGNAL["NO_TRADE"] == "NO_TRADE"


def test_calculate_combo_position_size_sizes_off_net_debit():
    result = calculate_combo_position_size(equity=100000, long_ask=2.0, short_bid=1.0, strategy="RISK_REVERSAL_BULLISH")
    # net debit = $1.00/contract = $100/contract; 1% of $100k = $1000 risk budget -> 10 contracts, capped at MAX_CONTRACTS.
    assert result["should_trade"] is True
    assert result["qty"] == 5


def test_calculate_combo_position_size_net_credit_falls_back_to_max_contracts():
    result = calculate_combo_position_size(equity=100000, long_ask=1.0, short_bid=1.5, strategy="RISK_REVERSAL_BEARISH")
    assert result["should_trade"] is True
    assert result["qty"] == 5


def test_calculate_combo_position_size_no_trade_strategy():
    result = calculate_combo_position_size(equity=100000, long_ask=2.0, short_bid=1.0, strategy="NO_TRADE")
    assert result["should_trade"] is False


def test_estimate_trade_cost_single_long_leg():
    legs = [{"symbol": "QQQ260904C00731000", "ask": 1.0, "bid": 0.9, "side": "long"}]
    assert estimate_trade_cost(legs, qty=3) == 300.0


def test_estimate_trade_cost_combo_nets_short_leg_credit():
    legs = [
        {"symbol": "QQQ260904C00731000", "ask": 2.0, "bid": 1.9, "side": "long"},
        {"symbol": "QQQ260904P00700000", "ask": 1.1, "bid": 1.0, "side": "short"},
    ]
    # long costs 2.0*100=200/contract, short leg credits its bid 1.0*100=100/contract -> net 100/contract.
    assert estimate_trade_cost(legs, qty=2) == 200.0

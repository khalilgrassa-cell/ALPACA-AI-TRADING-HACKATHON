"""Unit tests for the risk/trading agents — both deterministic (no LLM), so these mock the MCP
session directly. Neither needs a Groq API key, MCP server, or network access."""
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import risk_agent
import trading_agent

EXP = (date.today() + timedelta(days=10)).strftime("%y%m%d")  # inside risk_agent's 5-21 DTE window


def filled_at(minutes_ago):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


def mcp_result(data):
    return SimpleNamespace(is_error=False, structured_content={"data": data}, content=[])


def mcp_error_result(error_text):
    return SimpleNamespace(is_error=True, structured_content={}, content=[SimpleNamespace(text=error_text)])


def chain_snapshot(ask, bid):
    return {"latestQuote": {"ap": ask, "bp": bid}}


def fake_session(*call_tool_results):
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=call_tool_results)
    return session


def test_assess_risk_selects_contract_and_sizes_for_a_single_leg_strategy():
    session = fake_session(
        mcp_result({"equity": "100000"}),
        mcp_result({"snapshots": {
            f"QQQ{EXP}C00700000": chain_snapshot(ask=5.0, bid=4.9),
            f"QQQ{EXP}C00731000": chain_snapshot(ask=0.91, bid=0.90),
        }}),
        mcp_result({"snapshots": {}}),  # get_option_snapshot: no delta data -> falls back to OTM_PCT
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="LONG_CALL", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is True
    assert result["qty"] > 0
    assert result["legs"] == [{"symbol": f"QQQ{EXP}C00731000", "ask": 0.91, "bid": 0.90, "side": "long"}]

    account_call, chain_call, snapshot_call = session.call_tool.call_args_list
    assert account_call.args[0] == "get_account_info"
    assert chain_call.args[0] == "get_option_chain"
    assert chain_call.args[1]["underlying_symbol"] == "QQQ"
    assert chain_call.args[1]["feed"] == "indicative"
    # Biased toward the OTM side (at/above current price for a call) -- not a shared range with puts.
    assert chain_call.args[1]["type"] == "call"
    assert chain_call.args[1]["strike_price_gte"] == 716.43
    assert snapshot_call.args[0] == "get_option_snapshot"


def test_assess_risk_prefers_target_delta_over_strike_distance_when_available():
    session = fake_session(
        mcp_result({"equity": "100000"}),
        mcp_result({"snapshots": {
            f"QQQ{EXP}C00731000": chain_snapshot(ask=5.0, bid=4.9),  # closest strike, wrong delta
            f"QQQ{EXP}C00750000": chain_snapshot(ask=2.0, bid=1.9),  # farther strike, target delta
        }}),
        mcp_result({"snapshots": {
            f"QQQ{EXP}C00731000": {"greeks": {"delta": 0.55}},
            f"QQQ{EXP}C00750000": {"greeks": {"delta": 0.30}},
        }}),
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="LONG_CALL", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["legs"] == [{"symbol": f"QQQ{EXP}C00750000", "ask": 2.0, "bid": 1.9, "side": "long"}]


def test_assess_risk_handles_a_combo_strategy_with_two_legs():
    session = fake_session(
        mcp_result({"equity": "100000"}),
        mcp_result({"snapshots": {f"QQQ{EXP}C00731000": chain_snapshot(ask=2.0, bid=1.9)}}),  # call chain
        mcp_result({"snapshots": {f"QQQ{EXP}P00702000": chain_snapshot(ask=1.1, bid=1.0)}}),  # put chain
        mcp_result({"snapshots": {}}),  # get_option_snapshot: no delta data -> falls back to OTM_PCT
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="RISK_REVERSAL_BULLISH", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is True
    sides = {leg["side"] for leg in result["legs"]}
    assert sides == {"long", "short"}

    call_chain_call, put_chain_call = session.call_tool.call_args_list[1:3]
    assert call_chain_call.args[1]["type"] == "call"
    assert call_chain_call.args[1]["strike_price_gte"] == 716.43
    assert put_chain_call.args[1]["type"] == "put"
    assert put_chain_call.args[1]["strike_price_lte"] == 716.43


def test_assess_risk_propagates_no_trade_strategy_without_calling_mcp():
    session = fake_session()

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="NO_TRADE", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is False
    assert result["legs"] == []
    session.call_tool.assert_not_called()


def test_assess_risk_reports_should_trade_false_when_no_contract_qualifies():
    session = fake_session(
        mcp_result({"equity": "100000"}),
        mcp_result({"snapshots": {}}),
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="LONG_CALL", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is False
    assert result["legs"] == []


def test_assess_risk_reports_should_trade_false_on_get_account_info_failure():
    session = fake_session(mcp_error_result("rate limited"))

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="LONG_CALL", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is False
    assert "rate limited" in result["reasoning"]


def test_submit_order_uses_expected_tools_and_contract():
    session = fake_session(mcp_result({"id": "abc", "status": "pending_new"}))
    chosen_contract = {"symbol": "QQQ260904C00731000", "ask": 0.91}

    result = asyncio.run(trading_agent.submit_order(session, chosen_contract=chosen_contract, qty=3))

    assert result["order_submitted"] is True
    call = session.call_tool.call_args
    assert call.args[0] == "place_option_order"
    assert call.args[1] == {
        "symbol": "QQQ260904C00731000", "qty": "3", "side": "buy",
        "position_intent": "buy_to_open", "type": "market", "time_in_force": "day",
    }


def test_submit_order_short_side_uses_sell_to_open():
    session = fake_session(mcp_result({"id": "abc", "status": "pending_new"}))
    chosen_contract = {"symbol": "QQQ260904P00700000", "ask": 1.1, "bid": 1.0}

    asyncio.run(trading_agent.submit_order(session, chosen_contract=chosen_contract, qty=3, side="short"))

    call = session.call_tool.call_args
    assert call.args[1]["side"] == "sell"
    assert call.args[1]["position_intent"] == "sell_to_open"


def test_submit_order_reports_failure_without_raising():
    session = fake_session(mcp_error_result("insufficient buying power"))

    result = asyncio.run(trading_agent.submit_order(
        session, chosen_contract={"symbol": "QQQ260904C00731000", "ask": 0.91}, qty=3,
    ))

    assert result["order_submitted"] is False
    assert "insufficient buying power" in result["reasoning"]


def test_manage_exits_reports_take_profit_without_closing_when_execute_exits_is_false():
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "0.25"}
    session = fake_session(mcp_result({"result": [position]}))

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=False))

    assert result["open_positions"] == 1
    assert result["exits"] == [{"symbol": position["symbol"], "reason": "TAKE_PROFIT", "closed": False}]
    session.call_tool.assert_called_once_with("get_all_positions", {})  # no close_position call


def test_manage_exits_closes_a_take_profit_position_when_execute_exits_is_true():
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "0.25"}
    session = fake_session(mcp_result({"result": [position]}), mcp_result({"status": "closed"}))

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=True))

    assert result["exits"] == [{"symbol": position["symbol"], "reason": "TAKE_PROFIT", "closed": True}]
    close_call = session.call_tool.call_args_list[1]
    assert close_call.args == ("close_position", {"symbol_or_asset_id": position["symbol"]})


def test_manage_exits_holds_a_stop_loss_position_opened_too_recently():
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "-0.25"}
    orders = [{"position_intent": "buy_to_open", "filled_at": filled_at(minutes_ago=5)}]
    session = fake_session(mcp_result({"result": [position]}), mcp_result({"result": orders}))

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=True))

    # Too soon after opening (< MIN_HOLD_MINUTES_BEFORE_STOP_LOSS) — held instead of closed.
    assert result["exits"] == [{"symbol": position["symbol"], "reason": None, "closed": False}]
    assert session.call_tool.call_count == 2  # get_all_positions, get_orders — no close_position


def test_manage_exits_closes_a_stop_loss_position_held_long_enough():
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "-0.25"}
    orders = [{"position_intent": "buy_to_open", "filled_at": filled_at(minutes_ago=60)}]
    session = fake_session(
        mcp_result({"result": [position]}), mcp_result({"result": orders}), mcp_result({"status": "closed"}),
    )

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=True))

    assert result["exits"] == [{"symbol": position["symbol"], "reason": "STOP_LOSS", "closed": True}]


def test_manage_exits_closes_a_stop_loss_position_when_opening_fill_cant_be_found():
    # Fails open: if the opening fill can't be determined, don't let a lookup gap block a real
    # stop-loss indefinitely.
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "-0.25"}
    session = fake_session(
        mcp_result({"result": [position]}), mcp_result({"result": []}), mcp_result({"status": "closed"}),
    )

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=True))

    assert result["exits"] == [{"symbol": position["symbol"], "reason": "STOP_LOSS", "closed": True}]


def test_manage_exits_does_not_look_up_order_history_for_a_holding_position():
    position = {"symbol": f"AAPL{EXP}C00230000", "unrealized_plpc": "0.05"}
    session = fake_session(mcp_result({"result": [position]}))

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=True))

    assert result["exits"] == [{"symbol": position["symbol"], "reason": None, "closed": False}]
    session.call_tool.assert_called_once_with("get_all_positions", {})  # no get_orders, no close_position


def test_manage_exits_reports_failure_without_raising():
    session = fake_session(mcp_error_result("rate limited"))

    result = asyncio.run(trading_agent.manage_exits(session, execute_exits=False))

    assert result == {"open_positions": 0, "exits": [], "reasoning": "get_all_positions failed — rate limited"}

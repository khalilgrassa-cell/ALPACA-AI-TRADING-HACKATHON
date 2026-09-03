"""Unit tests for the risk/trading agents. risk_agent is deterministic (no LLM) — mocks the MCP
session directly. trading_agent is still LLM-driven — mocks llm_tools.run_agent instead. Neither
needs a Groq API key, MCP server, or network access."""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import risk_agent
import trading_agent

EXP = (date.today() + timedelta(days=10)).strftime("%y%m%d")  # inside risk_agent's 5-21 DTE window


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
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="LONG_CALL", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is True
    assert result["qty"] > 0
    assert result["legs"] == [{"symbol": f"QQQ{EXP}C00731000", "ask": 0.91, "bid": 0.90, "side": "long"}]

    account_call, chain_call = session.call_tool.call_args_list
    assert account_call.args[0] == "get_account_info"
    assert chain_call.args[0] == "get_option_chain"
    assert chain_call.args[1]["underlying_symbol"] == "QQQ"
    assert chain_call.args[1]["feed"] == "indicative"


def test_assess_risk_handles_a_combo_strategy_with_two_legs():
    session = fake_session(
        mcp_result({"equity": "100000"}),
        mcp_result({"snapshots": {
            f"QQQ{EXP}C00731000": chain_snapshot(ask=2.0, bid=1.9),
            f"QQQ{EXP}P00702000": chain_snapshot(ask=1.1, bid=1.0),
        }}),
    )

    result = asyncio.run(risk_agent.assess_risk(
        session, symbol="QQQ", strategy="RISK_REVERSAL_BULLISH", current_price=716.43, min_dte=5, max_dte=21,
    ))

    assert result["should_trade"] is True
    sides = {leg["side"] for leg in result["legs"]}
    assert sides == {"long", "short"}


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
    canned = '{"order_submitted": true, "order_result": {"id": "abc"}, "reasoning": "done"}'
    chosen_contract = {"symbol": "QQQ260904C00731000", "ask": 0.91}
    with patch.object(trading_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        result = asyncio.run(trading_agent.submit_order(
            mcp_session="fake-session", chosen_contract=chosen_contract, qty=3,
        ))

    assert result["order_submitted"] is True
    args, kwargs = mock_run.call_args
    assert "QQQ260904C00731000" in args[1] and "3" in args[1] and "buy_to_open" in args[1]
    assert kwargs["mcp_tool_names"] == {"place_option_order"}
    assert kwargs["local_tools"] == []


def test_submit_order_short_side_uses_sell_to_open():
    canned = '{"order_submitted": true, "order_result": {"id": "abc"}, "reasoning": "done"}'
    chosen_contract = {"symbol": "QQQ260904P00700000", "ask": 1.1, "bid": 1.0}
    with patch.object(trading_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        asyncio.run(trading_agent.submit_order(
            mcp_session="fake-session", chosen_contract=chosen_contract, qty=3, side="short",
        ))

    args, _ = mock_run.call_args
    assert "sell_to_open" in args[1]


def test_manage_exits_uses_expected_tools_regardless_of_todays_candidates():
    canned = '{"open_positions": 1, "exits": [{"symbol": "AAPL260904C00230000", "reason": "TAKE_PROFIT", "closed": false}], "reasoning": "one exit found"}'
    with patch.object(trading_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        result = asyncio.run(trading_agent.manage_exits(mcp_session="fake-session", execute_exits=False))

    assert result["open_positions"] == 1
    args, kwargs = mock_run.call_args
    assert "execute_exits=False" in args[1]
    assert kwargs["mcp_tool_names"] == {"get_all_positions", "close_position"}
    assert [tool.name for tool in kwargs["local_tools"]] == ["check_exit_rule"]


def test_manage_exits_passes_execute_exits_flag_through():
    canned = '{"open_positions": 0, "exits": [], "reasoning": "nothing open"}'
    with patch.object(trading_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        asyncio.run(trading_agent.manage_exits(mcp_session="fake-session", execute_exits=True))

    args, _ = mock_run.call_args
    assert "execute_exits=True" in args[1]

"""Unit tests for the risk/trading agent wrappers — mocks llm_tools.run_agent, so no Groq API key, MCP server, or network access is needed."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import risk_agent
import trading_agent


def test_assess_risk_parses_decision_and_uses_expected_tools():
    canned = (
        '{"strategy": "LONG_CALL", "legs": [{"symbol": "QQQ260904C00731000", "ask": 0.91, "bid": 0.9, "side": "long"}], '
        '"qty": 3, "should_trade": true, "reasoning": "ok"}'
    )
    with patch.object(risk_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        result = asyncio.run(risk_agent.assess_risk(
            mcp_session="fake-session", symbol="QQQ", strategy="LONG_CALL",
            current_price=716.43, min_dte=5, max_dte=21,
        ))

    assert result["should_trade"] is True
    assert result["qty"] == 3
    args, kwargs = mock_run.call_args
    assert "LONG_CALL" in args[1] and "5-21" in args[1] and "long leg: BUY_CALL" in args[1]
    assert kwargs["mcp_tool_names"] == {"get_account_info", "get_option_chain"}
    assert {tool.name for tool in kwargs["local_tools"]} == {"select_option_contract", "calculate_position_size"}


def test_assess_risk_describes_both_legs_for_a_combo_strategy_and_uses_combo_sizing_tool():
    canned = '{"strategy": "RISK_REVERSAL_BULLISH", "legs": [], "qty": 0, "should_trade": false, "reasoning": "n/a"}'
    with patch.object(risk_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        asyncio.run(risk_agent.assess_risk(
            mcp_session="fake-session", symbol="QQQ", strategy="RISK_REVERSAL_BULLISH",
            current_price=716.43, min_dte=5, max_dte=21,
        ))

    args, kwargs = mock_run.call_args
    assert "long leg: BUY_CALL" in args[1] and "short leg: BUY_PUT" in args[1]
    # Only the combo sizing tool is offered — not calculate_position_size too — so a 2-leg call's
    # tool schemas don't carry a tool it will never use.
    assert {tool.name for tool in kwargs["local_tools"]} == {"select_option_contract", "calculate_combo_position_size"}


def test_assess_risk_propagates_no_trade_strategy():
    canned = '{"strategy": "NO_TRADE", "legs": [], "qty": 0, "should_trade": false, "reasoning": "flat"}'
    with patch.object(risk_agent, "run_agent", new=AsyncMock(return_value=(canned, []))):
        result = asyncio.run(risk_agent.assess_risk(
            mcp_session="fake-session", symbol="QQQ", strategy="NO_TRADE",
            current_price=716.43, min_dte=5, max_dte=21,
        ))

    assert result["should_trade"] is False
    assert result["legs"] == []


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

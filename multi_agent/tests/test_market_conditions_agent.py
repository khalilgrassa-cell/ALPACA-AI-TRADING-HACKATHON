"""Unit tests for the market-conditions agent wrapper — mocks llm_tools.run_agent."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import market_conditions_agent


def test_check_market_conditions_open():
    canned = '{"market_open": true, "reasoning": "clock says open"}'
    with patch.object(market_conditions_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        result = asyncio.run(market_conditions_agent.check_market_conditions(mcp_session="fake-session"))

    assert result["market_open"] is True
    args, kwargs = mock_run.call_args
    assert args[2] == "fake-session"
    assert kwargs["mcp_tool_names"] == {"get_clock", "get_calendar"}
    assert kwargs["local_tools"] == []


def test_check_market_conditions_closed():
    canned = '{"market_open": false, "reasoning": "outside trading hours"}'
    with patch.object(market_conditions_agent, "run_agent", new=AsyncMock(return_value=(canned, []))):
        result = asyncio.run(market_conditions_agent.check_market_conditions(mcp_session="fake-session"))

    assert result["market_open"] is False

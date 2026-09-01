"""Unit tests for the deterministic market-conditions check — mocks the MCP session, so no LLM,
network, or credentials are needed."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import market_conditions_agent


def mcp_result(data=None, is_error=False, error_text="failed"):
    if is_error:
        return SimpleNamespace(is_error=True, structured_content={}, content=[SimpleNamespace(text=error_text)])
    return SimpleNamespace(is_error=False, structured_content={"data": data}, content=[])


def test_check_market_conditions_open():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(data={
        "is_open": True, "next_open": "2026-09-02T09:30:00-04:00", "next_close": "2026-09-01T16:00:00-04:00",
    }))

    result = asyncio.run(market_conditions_agent.check_market_conditions(session))

    assert result["market_open"] is True
    session.call_tool.assert_awaited_once_with("get_clock", {})


def test_check_market_conditions_closed():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(data={
        "is_open": False, "next_open": "2026-09-02T09:30:00-04:00", "next_close": "2026-09-01T16:00:00-04:00",
    }))

    result = asyncio.run(market_conditions_agent.check_market_conditions(session))

    assert result["market_open"] is False


def test_check_market_conditions_raises_on_mcp_error():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(is_error=True, error_text="clock unavailable"))

    try:
        asyncio.run(market_conditions_agent.check_market_conditions(session))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "clock unavailable" in str(exc)

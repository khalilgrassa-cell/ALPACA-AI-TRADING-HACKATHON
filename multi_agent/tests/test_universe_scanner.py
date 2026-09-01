"""Unit tests for the deterministic universe scanner — mocks the MCP session, so no network or
credentials are needed. No LLM is involved in this module at all, so there's nothing to mock there."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from universe_scanner import scan_universe


def mcp_result(data=None, is_error=False, error_text="failed"):
    if is_error:
        return SimpleNamespace(is_error=True, structured_content={}, content=[SimpleNamespace(text=error_text)])
    return SimpleNamespace(is_error=False, structured_content={"data": data}, content=[])


def bars(*closes):
    return [{"c": c} for c in closes]


def test_scan_universe_screens_batched_bars():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(data={
        "bars": {
            "AAPL": bars(100, 100, 100, 100, 100, 102),  # BUY_CALL
            "MSFT": bars(100, 100, 100, 100, 100, 100.2),  # NO_TRADE
            "JPM": bars(100, 100, 100, 100, 100, 98),  # BUY_PUT
        }
    }))

    candidates = asyncio.run(scan_universe(session, symbols=["AAPL", "MSFT", "JPM"]))

    call_args = session.call_tool.call_args
    assert call_args.args[0] == "get_stock_bars"
    assert call_args.args[1]["symbols"] == "AAPL,MSFT,JPM"

    symbols = {c["symbol"] for c in candidates}
    assert symbols == {"AAPL", "JPM"}


def test_scan_universe_skips_symbols_with_insufficient_bars():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(data={
        "bars": {
            "AAPL": bars(100, 100, 100, 100, 100, 102),
            "EA": bars(100, 99),  # delisted/halted mid-window — too little history
        }
    }))

    candidates = asyncio.run(scan_universe(session, symbols=["AAPL", "EA"]))

    assert {c["symbol"] for c in candidates} == {"AAPL"}


def test_scan_universe_raises_clearly_on_failure():
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mcp_result(is_error=True, error_text="rate limited"))

    try:
        asyncio.run(scan_universe(session, symbols=["AAPL"]))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "rate limited" in str(exc)

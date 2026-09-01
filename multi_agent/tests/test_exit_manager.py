"""Unit tests for the standalone exit-management entry point — mocks the market-conditions/exit
agents and the MCP connection, so no API key, MCP server, or network access is needed."""
import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import exit_manager


def patched_connect(mock_connect):
    mock_connect.return_value.__aenter__ = AsyncMock(return_value="fake-session")
    mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)


def patch_all(**overrides):
    defaults = {
        "check_market_conditions": AsyncMock(return_value={"market_open": True, "reasoning": "open"}),
        "manage_exits": AsyncMock(return_value={"open_positions": 0, "exits": [], "reasoning": "nothing to do"}),
    }
    defaults.update(overrides)
    return defaults


def run_with(agents):
    with ExitStack() as stack:
        for name, fn in agents.items():
            stack.enter_context(patch.object(exit_manager, name, new=fn))
        mock_connect = stack.enter_context(patch.object(exit_manager, "connect"))
        patched_connect(mock_connect)
        asyncio.run(exit_manager.main())
    return agents


def test_market_closed_skips_exit_check(capsys):
    agents = patch_all(check_market_conditions=AsyncMock(return_value={"market_open": False, "reasoning": "weekend"}))
    run_with(agents)

    agents["manage_exits"].assert_not_called()

    out = capsys.readouterr().out
    assert "Market is closed" in out


def test_market_open_runs_exit_check():
    agents = patch_all()
    run_with(agents)

    agents["manage_exits"].assert_called_once_with("fake-session", execute_exits=exit_manager.EXECUTE_EXITS)


def test_agent_failure_is_caught_and_reported_to_stdout(capsys):
    agents = patch_all(manage_exits=AsyncMock(side_effect=RuntimeError("boom")))
    run_with(agents)

    out = capsys.readouterr().out
    assert "ERROR: exit-management cycle aborted" in out
    assert "boom" in out

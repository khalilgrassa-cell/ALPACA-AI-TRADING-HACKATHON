"""Unit tests for the strategy agent — mocks llm_tools.run_agent, with emphasis on the code-level
safety clamp: the chosen strategy can never point a different direction than the signal it was
given, and can never manufacture a trade out of NO_TRADE, even if the model's own JSON output
tries to."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import strategy_agent
from strategy_agent import _clamp_strategy


def run_with_canned_response(canned, signal="BUY_CALL"):
    with patch.object(strategy_agent, "run_agent", new=AsyncMock(return_value=(canned, []))):
        return asyncio.run(strategy_agent.choose_strategy(
            mcp_session="fake-session", symbol="QQQ", signal=signal, sentiment="positive", momentum_pct=2.5,
        ))


def test_clamp_allows_matching_direction():
    assert _clamp_strategy("BUY_CALL", "LONG_CALL") == "LONG_CALL"
    assert _clamp_strategy("BUY_CALL", "RISK_REVERSAL_BULLISH") == "RISK_REVERSAL_BULLISH"
    assert _clamp_strategy("BUY_PUT", "LONG_PUT") == "LONG_PUT"
    assert _clamp_strategy("BUY_PUT", "RISK_REVERSAL_BEARISH") == "RISK_REVERSAL_BEARISH"


def test_clamp_blocks_opposite_direction():
    assert _clamp_strategy("BUY_CALL", "LONG_PUT") == "NO_TRADE"
    assert _clamp_strategy("BUY_CALL", "RISK_REVERSAL_BEARISH") == "NO_TRADE"
    assert _clamp_strategy("BUY_PUT", "LONG_CALL") == "NO_TRADE"


def test_clamp_blocks_upgrade_from_no_trade():
    assert _clamp_strategy("NO_TRADE", "LONG_CALL") == "NO_TRADE"
    assert _clamp_strategy("NO_TRADE", "RISK_REVERSAL_BULLISH") == "NO_TRADE"


def test_clamp_rejects_unrecognized_strategy_name():
    assert _clamp_strategy("BUY_CALL", "SOMETHING_MADE_UP") == "NO_TRADE"
    assert _clamp_strategy("BUY_CALL", None) == "NO_TRADE"


def test_choose_strategy_allows_conviction_combo():
    canned = '{"strategy": "RISK_REVERSAL_BULLISH", "reasoning": "momentum and sentiment agree"}'
    result = run_with_canned_response(canned, signal="BUY_CALL")
    assert result["strategy"] == "RISK_REVERSAL_BULLISH"


def test_choose_strategy_clamps_model_trying_to_flip_direction():
    canned = '{"strategy": "RISK_REVERSAL_BEARISH", "reasoning": "trying to flip"}'
    result = run_with_canned_response(canned, signal="BUY_CALL")
    assert result["strategy"] == "NO_TRADE"


def test_choose_strategy_uses_no_mcp_tools():
    canned = '{"strategy": "LONG_CALL", "reasoning": "plain bullish"}'
    with patch.object(strategy_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        asyncio.run(strategy_agent.choose_strategy(
            mcp_session="fake-session", symbol="QQQ", signal="BUY_CALL", sentiment="neutral", momentum_pct=1.5,
        ))

    args, kwargs = mock_run.call_args
    assert kwargs["mcp_tool_names"] == set()
    assert kwargs["local_tools"] == []
    assert "QQQ" in args[1] and "BUY_CALL" in args[1] and "neutral" in args[1]

"""Unit tests for the news agent — mocks llm_tools.run_agent, with emphasis on the code-level
safety clamp: the news agent may only veto a signal toward NO_TRADE, never invent or flip one,
even if the model's own JSON output tries to."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import sentiment_agent
from sentiment_agent import _clamp_signal


def run_with_canned_response(canned, signal="BUY_CALL"):
    with patch.object(sentiment_agent, "run_agent", new=AsyncMock(return_value=(canned, []))):
        return asyncio.run(sentiment_agent.check_sentiment(mcp_session="fake-session", symbol="QQQ", signal=signal))


def test_clamp_keeps_unchanged_signal():
    assert _clamp_signal("BUY_CALL", "BUY_CALL") == "BUY_CALL"


def test_clamp_allows_veto_to_no_trade():
    assert _clamp_signal("BUY_CALL", "NO_TRADE") == "NO_TRADE"


def test_clamp_blocks_flip_between_directions():
    assert _clamp_signal("BUY_CALL", "BUY_PUT") == "BUY_CALL"
    assert _clamp_signal("BUY_PUT", "BUY_CALL") == "BUY_PUT"


def test_clamp_blocks_upgrade_from_no_trade():
    assert _clamp_signal("NO_TRADE", "BUY_CALL") == "NO_TRADE"
    assert _clamp_signal("NO_TRADE", "BUY_PUT") == "NO_TRADE"


def test_check_sentiment_neutral_keeps_signal():
    canned = '{"sentiment": "neutral", "signal": "BUY_CALL", "reasoning": "nothing notable"}'
    result = run_with_canned_response(canned, signal="BUY_CALL")
    assert result["signal"] == "BUY_CALL"
    assert result["overridden"] is False


def test_check_sentiment_negative_vetoes_call():
    canned = '{"sentiment": "negative", "signal": "NO_TRADE", "reasoning": "bad earnings"}'
    result = run_with_canned_response(canned, signal="BUY_CALL")
    assert result["signal"] == "NO_TRADE"
    assert result["overridden"] is True


def test_check_sentiment_cannot_flip_direction_even_if_model_tries():
    canned = '{"sentiment": "negative", "signal": "BUY_PUT", "reasoning": "trying to flip"}'
    result = run_with_canned_response(canned, signal="BUY_CALL")
    assert result["signal"] == "BUY_CALL"
    assert result["overridden"] is False


def test_check_sentiment_uses_get_news_tool_only():
    canned = '{"sentiment": "neutral", "signal": "NO_TRADE", "reasoning": "n/a"}'
    with patch.object(sentiment_agent, "run_agent", new=AsyncMock(return_value=(canned, []))) as mock_run:
        asyncio.run(sentiment_agent.check_sentiment(mcp_session="fake-session", symbol="QQQ", signal="NO_TRADE"))

    args, kwargs = mock_run.call_args
    assert kwargs["mcp_tool_names"] == {"get_news"}

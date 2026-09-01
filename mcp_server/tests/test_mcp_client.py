"""Unit tests for mcp_client.py's env-var resolution — no MCP server or network access needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_client import _resolve_env


def test_resolve_env_strips_whitespace_from_referenced_env_vars(monkeypatch):
    # Reproduces a real live failure: GROQ_API_KEY had a trailing newline from copy-pasting into
    # a GitHub Actions secret, which made httpx reject every Groq request outright. The Alpaca
    # keys are pasted the same way, so the MCP server subprocess needs the same guard.
    monkeypatch.setenv("ALPACA_API_KEY", "  test_key\n")
    resolved = _resolve_env({"ALPACA_API_KEY": "${ALPACA_API_KEY}"})
    assert resolved == {"ALPACA_API_KEY": "test_key"}


def test_resolve_env_passes_through_literal_values_unchanged():
    resolved = _resolve_env({"ALPACA_PAPER_TRADE": "true"})
    assert resolved == {"ALPACA_PAPER_TRADE": "true"}

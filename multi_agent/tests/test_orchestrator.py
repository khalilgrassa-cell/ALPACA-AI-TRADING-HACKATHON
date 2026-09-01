"""Unit tests for the orchestrator's full pipeline wiring — mocks every agent, the universe
scanner, and the MCP connection, so no API key, MCP server, or network access is needed."""
import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator


def patched_connect(mock_connect):
    mock_connect.return_value.__aenter__ = AsyncMock(return_value="fake-session")
    mock_connect.return_value.__aexit__ = AsyncMock(return_value=False)


def patch_all(**overrides):
    defaults = {
        "check_market_conditions": AsyncMock(return_value={"market_open": True, "reasoning": "open"}),
        "scan_universe": AsyncMock(return_value=[]),
        "check_sentiment": AsyncMock(return_value={"sentiment": "neutral", "signal": "BUY_CALL", "overridden": False, "reasoning": "n/a"}),
        "choose_strategy": AsyncMock(return_value={"strategy": "LONG_CALL", "reasoning": "n/a"}),
        "assess_risk": AsyncMock(return_value={"strategy": "NO_TRADE", "legs": [], "qty": 0, "should_trade": False, "reasoning": "n/a"}),
        "submit_order": AsyncMock(return_value={"order_submitted": True, "order_result": {}, "reasoning": "done"}),
        "manage_exits": AsyncMock(return_value={"open_positions": 0, "exits": [], "reasoning": "nothing to do"}),
        "get_equity": AsyncMock(return_value=100000.0),
    }
    defaults.update(overrides)
    return defaults


def run_with(agents):
    with ExitStack() as stack:
        for name, fn in agents.items():
            stack.enter_context(patch.object(orchestrator, name, new=fn))
        mock_connect = stack.enter_context(patch.object(orchestrator, "connect"))
        patched_connect(mock_connect)
        asyncio.run(orchestrator.main())
    return agents


def test_market_closed_skips_scan_and_exits(capsys):
    agents = patch_all(check_market_conditions=AsyncMock(return_value={"market_open": False, "reasoning": "weekend"}))
    run_with(agents)

    agents["scan_universe"].assert_not_called()
    agents["manage_exits"].assert_not_called()

    out = capsys.readouterr().out
    assert "Market is closed" in out


def test_no_candidates_still_runs_exit_management(capsys):
    agents = patch_all(scan_universe=AsyncMock(return_value=[]))
    run_with(agents)

    agents["check_sentiment"].assert_not_called()
    agents["choose_strategy"].assert_not_called()
    agents["assess_risk"].assert_not_called()
    agents["submit_order"].assert_not_called()
    agents["manage_exits"].assert_called_once()
    agents["get_equity"].assert_not_called()

    out = capsys.readouterr().out
    assert "No symbols crossed the momentum threshold" in out


def test_candidate_chain_passes_sentiment_and_strategy_through_to_risk_and_submits_order(capsys):
    candidates = [{"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 2.0}]
    long_leg = {"symbol": "AAPL260904C00230000", "ask": 1.0, "bid": 0.9, "side": "long"}
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        check_sentiment=AsyncMock(return_value={"sentiment": "positive", "signal": "BUY_CALL", "overridden": False, "reasoning": "n/a"}),
        choose_strategy=AsyncMock(return_value={"strategy": "LONG_CALL", "reasoning": "bullish"}),
        assess_risk=AsyncMock(return_value={
            "strategy": "LONG_CALL", "legs": [long_leg], "qty": 4, "should_trade": True, "reasoning": "ok",
        }),
    )
    run_with(agents)

    sentiment_args = agents["check_sentiment"].call_args.args
    assert sentiment_args == ("fake-session", "AAPL", "BUY_CALL")

    strategy_args = agents["choose_strategy"].call_args.args
    assert strategy_args == ("fake-session", "AAPL", "BUY_CALL", "positive", 2.0)

    risk_args = agents["assess_risk"].call_args.args
    assert risk_args[2] == "LONG_CALL"  # the strategy agent's choice
    assert risk_args[3] == 230.0

    agents["submit_order"].assert_called_once_with("fake-session", long_leg, 4, side="long")
    agents["manage_exits"].assert_called_once()

    out = capsys.readouterr().out
    assert "AAPL" in out


def test_combo_strategy_submits_both_legs_long_then_short(capsys):
    candidates = [{"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 3.0}]
    long_leg = {"symbol": "AAPL260904C00230000", "ask": 2.0, "bid": 1.9, "side": "long"}
    short_leg = {"symbol": "AAPL260904P00210000", "ask": 1.1, "bid": 1.0, "side": "short"}
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        choose_strategy=AsyncMock(return_value={"strategy": "RISK_REVERSAL_BULLISH", "reasoning": "high conviction"}),
        assess_risk=AsyncMock(return_value={
            "strategy": "RISK_REVERSAL_BULLISH", "legs": [long_leg, short_leg], "qty": 2, "should_trade": True, "reasoning": "ok",
        }),
    )
    run_with(agents)

    assert agents["submit_order"].call_args_list == [
        (("fake-session", long_leg, 2), {"side": "long"}),
        (("fake-session", short_leg, 2), {"side": "short"}),
    ]


def test_combo_strategy_skips_short_leg_when_long_leg_order_fails(capsys):
    candidates = [{"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 3.0}]
    long_leg = {"symbol": "AAPL260904C00230000", "ask": 2.0, "bid": 1.9, "side": "long"}
    short_leg = {"symbol": "AAPL260904P00210000", "ask": 1.1, "bid": 1.0, "side": "short"}
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        choose_strategy=AsyncMock(return_value={"strategy": "RISK_REVERSAL_BULLISH", "reasoning": "high conviction"}),
        assess_risk=AsyncMock(return_value={
            "strategy": "RISK_REVERSAL_BULLISH", "legs": [long_leg, short_leg], "qty": 2, "should_trade": True, "reasoning": "ok",
        }),
        submit_order=AsyncMock(return_value={"order_submitted": False, "order_result": None, "reasoning": "rejected"}),
    )
    run_with(agents)

    agents["submit_order"].assert_called_once_with("fake-session", long_leg, 2, side="long")


def test_no_trade_strategy_reaches_risk_agent_but_skips_order():
    candidates = [{"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 2.0}]
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        check_sentiment=AsyncMock(return_value={"sentiment": "negative", "signal": "NO_TRADE", "overridden": True, "reasoning": "bad news"}),
        choose_strategy=AsyncMock(return_value={"strategy": "NO_TRADE", "reasoning": "no signal"}),
    )
    run_with(agents)

    assert agents["assess_risk"].call_args.args[2] == "NO_TRADE"
    agents["submit_order"].assert_not_called()


def test_one_candidates_failure_does_not_abort_the_others(capsys):
    candidates = [
        {"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 5.0},
        {"symbol": "MSFT", "signal": "BUY_CALL", "current_price": 500.0, "momentum_pct": 4.0},
    ]
    long_leg = {"symbol": "MSFT260904C00500000", "ask": 1.0, "bid": 0.9, "side": "long"}
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        check_sentiment=AsyncMock(side_effect=[
            RuntimeError("token limit exceeded"),
            {"sentiment": "neutral", "signal": "BUY_CALL", "overridden": False, "reasoning": "n/a"},
        ]),
        assess_risk=AsyncMock(return_value={
            "strategy": "LONG_CALL", "legs": [long_leg], "qty": 2, "should_trade": True, "reasoning": "ok",
        }),
    )
    run_with(agents)

    # AAPL fails at the sentiment step; MSFT should still be fully evaluated afterward.
    assert agents["assess_risk"].call_count == 1
    assert agents["assess_risk"].call_args.args[1] == "MSFT"
    agents["submit_order"].assert_called_once()
    agents["manage_exits"].assert_called_once()

    out = capsys.readouterr().out
    assert "ERROR: AAPL aborted" in out
    assert "Failed: ['AAPL']" in out


def test_cycle_risk_cap_blocks_later_candidates_but_not_earlier_ones(capsys):
    candidates = [
        {"symbol": "AAPL", "signal": "BUY_CALL", "current_price": 230.0, "momentum_pct": 5.0},
        {"symbol": "MSFT", "signal": "BUY_CALL", "current_price": 500.0, "momentum_pct": 4.0},
    ]
    aapl_leg = {"symbol": "AAPL260904C00230000", "ask": 8.0, "bid": 7.9, "side": "long"}
    msft_leg = {"symbol": "MSFT260904C00500000", "ask": 10.0, "bid": 9.9, "side": "long"}
    risk_results = [
        {"strategy": "LONG_CALL", "legs": [aapl_leg], "qty": 5, "should_trade": True, "reasoning": "ok"},
        {"strategy": "LONG_CALL", "legs": [msft_leg], "qty": 5, "should_trade": True, "reasoning": "ok"},
    ]
    # Budget is 5% of $100k equity = $5,000. AAPL costs 5*8*100=$4,000 (fits, committed -> $4,000).
    # MSFT would cost 5*10*100=$5,000 more, which pushes cumulative committed to $9,000 > the
    # $5,000 budget, so MSFT should be blocked by the cap even though AAPL went through.
    agents = patch_all(
        scan_universe=AsyncMock(return_value=candidates),
        assess_risk=AsyncMock(side_effect=risk_results),
        get_equity=AsyncMock(return_value=100000.0),
    )
    run_with(agents)

    assert agents["submit_order"].call_count == 1
    agents["submit_order"].assert_called_once_with("fake-session", aapl_leg, 5, side="long")

    out = capsys.readouterr().out
    assert "Cycle risk cap reached" in out
    assert "skipping MSFT" in out


def test_agent_failure_is_caught_and_reported_to_stdout(capsys):
    agents = patch_all(scan_universe=AsyncMock(side_effect=RuntimeError("boom")))
    run_with(agents)

    agents["manage_exits"].assert_not_called()

    out = capsys.readouterr().out
    assert "ERROR: pipeline aborted" in out
    assert "boom" in out

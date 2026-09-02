"""Strategy agent: picks which options strategy (if any) to run on a candidate, given its
momentum-and-sentiment-vetted signal. Needs no MCP tools — it's a pure judgment call over the
signal/sentiment/momentum context it's handed, not a data-fetching step."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from momentum_strategy import STRATEGY_SIGNAL

from llm_tools import parse_json_response, run_agent

SYSTEM_PROMPT = """You are the strategy agent in a multi-agent options trading system.

You receive a symbol's momentum-and-sentiment-vetted signal (BUY_CALL, BUY_PUT, or NO_TRADE), its \
momentum strength, and its news sentiment. Your job: choose which options strategy to run, from \
this menu:

- LONG_CALL: a plain long call. The default bullish choice.
- LONG_PUT: a plain long put. The default bearish choice.
- RISK_REVERSAL_BULLISH: buy a call and sell a put (a synthetic-long combo that reduces net cost). \
Only choose this for a BUY_CALL signal when sentiment is *also* positive — i.e. momentum and news \
agree, giving higher conviction for the more aggressive, undefined-risk combo.
- RISK_REVERSAL_BEARISH: buy a put and sell a call. Only choose this for a BUY_PUT signal when \
sentiment is *also* negative, for the same reason.
- NO_TRADE: if the signal is NO_TRADE, or you don't have enough conviction to trade at all.

Rules:
- Never choose a strategy whose direction disagrees with the given signal (e.g. never choose \
LONG_PUT or RISK_REVERSAL_BEARISH for a BUY_CALL signal).
- Never choose a strategy at all if the signal is NO_TRADE.
- When signal and sentiment disagree (e.g. BUY_CALL signal but negative sentiment), prefer the \
plain single-leg strategy (LONG_CALL/LONG_PUT) over the risk-reversal combo, or NO_TRADE if you \
have real doubts — don't reach for the higher-risk combo without conviction from both signals.

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"strategy": "LONG_CALL" | "LONG_PUT" | "RISK_REVERSAL_BULLISH" | "RISK_REVERSAL_BEARISH" | "NO_TRADE", "reasoning": string}
"""


def _clamp_strategy(signal, proposed_strategy):
    """Enforces in code, not just in the prompt, that the chosen strategy can never point a
    different direction than the vetted signal, nor manufacture a trade out of NO_TRADE."""
    if signal == "NO_TRADE":
        return "NO_TRADE"
    if proposed_strategy not in STRATEGY_SIGNAL:
        return "NO_TRADE"
    if STRATEGY_SIGNAL[proposed_strategy] != signal:
        return "NO_TRADE"
    return proposed_strategy


async def choose_strategy(mcp_session, symbol, signal, sentiment, momentum_pct):
    user_prompt = (
        f"Symbol: {symbol}. Signal: {signal}. Momentum: {momentum_pct:+.2f}%. News sentiment: {sentiment}. "
        "Choose the strategy to run."
    )
    text, _ = await run_agent(
        SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names=set(), local_tools=[], required_keys={"strategy", "reasoning"},
    )
    result = parse_json_response(text, required_keys={"strategy", "reasoning"})
    result["strategy"] = _clamp_strategy(signal, result.get("strategy"))
    return result


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from mcp_client import connect

    async with connect() as session:
        print(await choose_strategy(session, "QQQ", "BUY_CALL", "positive", 2.5))


if __name__ == "__main__":
    asyncio.run(_main())

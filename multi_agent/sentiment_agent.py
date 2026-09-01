"""Sentiment agent: model-based research over recent news for the symbol. Can veto (never invent or
flip) the scanner's signal on contradicting news; its sentiment reading also feeds the Strategy
Agent's choice of which strategy (if any) to run on this symbol."""
import asyncio
import sys
from pathlib import Path

from llm_tools import parse_json_response, run_agent

SYSTEM_PROMPT = """You are the news/sentiment agent in a multi-agent options trading system.

You receive a trading signal (BUY_CALL, BUY_PUT, or NO_TRADE) from the scanning agent, based purely \
on price momentum. Your job: call get_news for the underlying symbol to check for recent, market-moving \
news (earnings, guidance, litigation, regulatory action, etc.), and decide whether it supports, is \
neutral to, or contradicts the momentum signal.

- If the news clearly contradicts the signal (e.g. sharply negative news alongside a BUY_CALL signal, \
or sharply positive news alongside a BUY_PUT signal), veto it by setting signal to NO_TRADE.
- If the news is neutral, unremarkable, or unavailable, keep the original signal unchanged.
- You can only veto a trade down to NO_TRADE — never turn a NO_TRADE into a trade, and never flip \
BUY_CALL to BUY_PUT or vice versa.

Respond with your final answer as a single JSON object, and nothing else, with this shape:
{"sentiment": "positive" | "negative" | "neutral" | "unknown", "signal": "BUY_CALL" | "BUY_PUT" | "NO_TRADE", "reasoning": string}
"""


def _clamp_signal(original_signal, proposed_signal):
    """Enforces in code, not just in the prompt, that news can only veto toward NO_TRADE."""
    if original_signal == "NO_TRADE":
        return "NO_TRADE"
    if proposed_signal == "NO_TRADE":
        return "NO_TRADE"
    return original_signal


async def check_sentiment(mcp_session, symbol, signal):
    user_prompt = f"Scanning agent's signal for {symbol}: {signal}. Check recent news and confirm or veto it."
    text, _ = await run_agent(
        SYSTEM_PROMPT, user_prompt, mcp_session,
        mcp_tool_names={"get_news"}, local_tools=[],
    )
    result = parse_json_response(text, required_keys={"sentiment", "signal", "reasoning"})
    result["signal"] = _clamp_signal(signal, result.get("signal"))
    result["overridden"] = result["signal"] != signal
    return result


async def _main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from momentum_strategy import SYMBOL
    from mcp_client import connect

    async with connect() as session:
        print(await check_sentiment(session, SYMBOL, "BUY_CALL"))


if __name__ == "__main__":
    asyncio.run(_main())

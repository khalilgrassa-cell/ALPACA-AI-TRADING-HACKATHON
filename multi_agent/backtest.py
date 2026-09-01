"""Backtests the momentum signal's directional accuracy and charts price/momentum/signals."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strategy"))
from mcp_client import connect, mcp_data, mcp_error
from momentum_strategy import MOMENTUM_THRESHOLD, MOMENTUM_WINDOW, SYMBOL, calculate_momentum

BACKTEST_DAYS = 180
HOLD_DAYS = 5
CHART_PATH = "backtest_chart.png"


def bt_signal(closes, i):
    return calculate_momentum(closes[: i + 1])["signal"]


def bt_forward_return(closes, i):
    return (closes[i + HOLD_DAYS] - closes[i]) / closes[i] * 100


async def fetch_bars():
    async with connect() as session:
        result = await session.call_tool(
            "get_stock_bars", {"symbols": SYMBOL, "timeframe": "1Day", "days": BACKTEST_DAYS},
        )
        data = mcp_data(result)
        if data is None:
            print(f"ERROR: get_stock_bars failed — {mcp_error(result)}")
            return None
        return data["bars"][SYMBOL]


def run_backtest(bars):
    closes = [b["c"] for b in bars]
    dates = [datetime.fromisoformat(b["t"].replace("Z", "+00:00")) for b in bars]

    records = [(i, bt_signal(closes, i), bt_forward_return(closes, i)) for i in range(MOMENTUM_WINDOW, len(closes) - HOLD_DAYS)]
    calls = [r for r in records if r[1] == "BUY_CALL"]
    puts = [r for r in records if r[1] == "BUY_PUT"]
    flat = [r for r in records if r[1] == "NO_TRADE"]

    call_win_rate = (sum(1 for r in calls if r[2] > 0) / len(calls) * 100) if calls else 0
    put_win_rate = (sum(1 for r in puts if r[2] < 0) / len(puts) * 100) if puts else 0
    call_avg_return = (sum(r[2] for r in calls) / len(calls)) if calls else 0
    put_avg_return = (sum(-r[2] for r in puts) / len(puts)) if puts else 0

    print(f"Backtest data: {len(closes)} daily closes over ~{BACKTEST_DAYS} calendar days")
    print(f"Trading days evaluated: {len(records)} | BUY_CALL: {len(calls)} | BUY_PUT: {len(puts)} | NO_TRADE: {len(flat)}")
    print(f"BUY_CALL directional win rate ({HOLD_DAYS}d fwd return > 0): {call_win_rate:.1f}% | avg fwd return: {call_avg_return:+.2f}%")
    print(f"BUY_PUT directional win rate ({HOLD_DAYS}d fwd return < 0): {put_win_rate:.1f}% | avg captured move: {put_avg_return:+.2f}%")
    print("Note: validates directional accuracy of the momentum signal only — real options P&L also depends on theta decay, bid/ask spread, and IV changes, which are not modeled here.")

    plot_backtest(dates, closes)


def plot_backtest(dates, closes):
    momentum = [
        (closes[i] - closes[i - MOMENTUM_WINDOW]) / closes[i - MOMENTUM_WINDOW] * 100 if i >= MOMENTUM_WINDOW else None
        for i in range(len(closes))
    ]
    signals = [bt_signal(closes, i) if i >= MOMENTUM_WINDOW else "NO_TRADE" for i in range(len(closes))]
    call_idx = [i for i, s in enumerate(signals) if s == "BUY_CALL"]
    put_idx = [i for i, s in enumerate(signals) if s == "BUY_PUT"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(dates, closes, color="#4C72B0", linewidth=1.2, label=f"{SYMBOL} close")
    ax1.scatter([dates[i] for i in call_idx], [closes[i] for i in call_idx], color="green", marker="^", s=35, label="BUY_CALL signal", zorder=3)
    ax1.scatter([dates[i] for i in put_idx], [closes[i] for i in put_idx], color="red", marker="v", s=35, label="BUY_PUT signal", zorder=3)
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{SYMBOL} price with momentum signals")
    ax1.legend(loc="upper left")

    ax2.plot(dates, momentum, color="#DD8452", linewidth=1)
    ax2.axhline(MOMENTUM_THRESHOLD, color="green", linestyle="--", linewidth=0.8)
    ax2.axhline(-MOMENTUM_THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(f"{MOMENTUM_WINDOW}-day momentum (%)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"Saved chart to {CHART_PATH}")


async def main():
    bars = await fetch_bars()
    if bars is not None:
        run_backtest(bars)


if __name__ == "__main__":
    asyncio.run(main())

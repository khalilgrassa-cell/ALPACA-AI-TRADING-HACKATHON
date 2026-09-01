"""The momentum-based long-options strategy: signal generation, contract selection, position
sizing, and exit rules.

Single source of truth for the multi-agent pipeline (../multi_agent/) — change the strategy
here and the agents' local tools pick it up. Contains no I/O and no MCP/LLM-provider
dependencies, so it's trivially unit-testable and swappable independent of how it's executed.
"""
import re
from datetime import date, datetime

SYMBOL = "QQQ"  # used only by backtest.py's single-symbol historical validation

# Nasdaq-100 constituents (as tracked by QQQ) — verified against the June 2026 quarterly
# reconstitution (added ALAB, CRWV, NBIS, RKLB, TER; removed CHTR, CTSH, INSM, VRSK, ZS).
NASDAQ_100 = [
    "ADBE", "ADP", "AMD", "ABNB", "ALNY", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI", "AAPL",
    "AMAT", "APP", "ARM", "ASML", "TEAM", "ADSK", "AXON", "BKR", "BKNG", "AVGO", "CDNS", "CTAS",
    "CSCO", "CCEP", "CMCSA", "CEG", "CPRT", "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG",
    "DASH", "EA", "EXC", "FAST", "FER", "FTNT", "GEHC", "GILD", "IDXX", "INTC", "INTU", "ISRG",
    "KDP", "KLAC", "KHC", "LRCX", "LIN", "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT",
    "MSTR", "MDLZ", "MPWR", "MNST", "NFLX", "NVDA", "NXPI", "ODFL", "ORLY", "PCAR", "PLTR", "PANW",
    "PAYX", "PYPL", "PDD", "PEP", "QCOM", "REGN", "ROP", "ROST", "STX", "SHOP", "SBUX", "SNPS",
    "TTWO", "TSLA", "TXN", "TRI", "TMUS", "VRTX", "WMT", "WBD", "WDC", "WDAY", "XEL",
    "ALAB", "CRWV", "NBIS", "RKLB", "TER",
]

# S&P 100 constituents — the broad blue-chip index Nasdaq-100 explicitly excludes financials
# from, so this is where genuinely new sector exposure (banks, energy, industrials, healthcare,
# staples) comes from. Verified against live Alpaca asset data for tricky tickers (BRK.B format,
# BNY, GEV).
SP_100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA",
    "BAC", "BKNG", "BLK", "BMY", "BNY", "BRK.B", "C", "CAT", "CL", "CMCSA", "COF", "COP", "COST",
    "CRM", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GEV", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO",
    "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "META", "MMM", "MO", "MRK",
    "MS", "MSFT", "MU", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR",
    "PM", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TMO", "TMUS", "TSLA", "TXN", "UBER",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

# The live multi-agent pipeline's trading universe: everything above, deduplicated.
UNIVERSE = sorted(set(NASDAQ_100) | set(SP_100))

MOMENTUM_WINDOW = 5
MOMENTUM_THRESHOLD = 1.0

MIN_DTE = 5
MAX_DTE = 21
OTM_PCT = 0.02

RISK_PCT = 0.01
MAX_CONTRACTS = 5

TAKE_PROFIT_PCT = 0.50
STOP_LOSS_PCT = -0.20
EXIT_DTE_BUFFER = 2

MAX_CYCLE_RISK_PCT = 0.05  # hard cap on aggregate new-position risk across all candidates in one cycle
TOP_N_HOTTEST = 5  # universe scan keeps only the top-N symbols by momentum strength ("hottest"/trending)

# The options-strategy menu the Strategy Agent chooses from, and the leg(s) each one needs.
# Each leg is (side, directional_signal): "long" -> buy_to_open, "short" -> sell_to_open, and the
# signal tells select_option_contract which option type/strike offset to pick for that leg.
# RISK_REVERSAL_* combos (buy one type, sell the other) are only ever chosen for a symbol whose
# momentum signal is already directional — see strategy_agent._clamp_strategy, which enforces in
# code (not just the prompt) that a strategy can never point a different direction than the signal
# it was given, nor manufacture a trade out of NO_TRADE.
STRATEGY_LEGS = {
    "LONG_CALL": (("long", "BUY_CALL"),),
    "LONG_PUT": (("long", "BUY_PUT"),),
    "RISK_REVERSAL_BULLISH": (("long", "BUY_CALL"), ("short", "BUY_PUT")),
    "RISK_REVERSAL_BEARISH": (("long", "BUY_PUT"), ("short", "BUY_CALL")),
    "NO_TRADE": (),
}
# The base directional signal each strategy expresses — used to keep the Strategy Agent's choice
# consistent with the momentum/sentiment-derived signal it was handed.
STRATEGY_SIGNAL = {
    "LONG_CALL": "BUY_CALL",
    "LONG_PUT": "BUY_PUT",
    "RISK_REVERSAL_BULLISH": "BUY_CALL",
    "RISK_REVERSAL_BEARISH": "BUY_PUT",
    "NO_TRADE": "NO_TRADE",
}

# Safety gate shared by orchestrator.py (full cycle) and exit_manager.py (exits-only, run on a
# much tighter schedule) — single source of truth so it can't be flipped in one entry point and
# forgotten in the other. False: the exit agent reports what it would close but never actually
# calls close_position.
EXECUTE_EXITS = True

# Generic OCC option symbol: {root}{YYMMDD}{C|P}{strike*1000, 8 digits} — root is 1-6 letters,
# not hardcoded to one underlying, since the universe now spans ~150 symbols.
CONTRACT_PATTERN = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def calculate_momentum(closes, window=MOMENTUM_WINDOW):
    """Signal generation: N-day momentum vs a +/- threshold, from daily closes (oldest first)."""
    current_price = closes[-1]
    past_price = closes[-1 - window]
    momentum_pct = (current_price - past_price) / past_price * 100
    signal = (
        "BUY_CALL" if momentum_pct > MOMENTUM_THRESHOLD
        else "BUY_PUT" if momentum_pct < -MOMENTUM_THRESHOLD
        else "NO_TRADE"
    )
    return {"current_price": current_price, "momentum_pct": momentum_pct, "signal": signal}


def parse_contract(symbol, data):
    """Parses an OCC option symbol (e.g. QQQ260904C00731000) plus its quote snapshot."""
    m = CONTRACT_PATTERN.match(symbol)
    if not m:
        return None
    return {
        "symbol": symbol,
        "underlying": m.group(1),
        "expiration": datetime.strptime(m.group(2), "%y%m%d").date(),
        "type": m.group(3),
        "strike": int(m.group(4)) / 1000,
        "ask": data.get("latestQuote", {}).get("ap", 0),
        "bid": data.get("latestQuote", {}).get("bp", 0),
    }


def select_contract(contracts, signal, current_price):
    """Contract selection: closest strike to the target OTM offset, within the DTE window."""
    opt_type = {"BUY_CALL": "C", "BUY_PUT": "P"}.get(signal)
    if opt_type is None:
        return None
    target = current_price * (1 + OTM_PCT) if opt_type == "C" else current_price * (1 - OTM_PCT)
    today = date.today()
    candidates = [
        c for c in contracts
        if c["type"] == opt_type and MIN_DTE <= (c["expiration"] - today).days <= MAX_DTE and c["ask"] > 0
    ]
    return min(candidates, key=lambda c: abs(c["strike"] - target)) if candidates else None


def calculate_position_size(equity, contract_ask, signal, risk_pct=RISK_PCT, max_contracts=MAX_CONTRACTS):
    """Risk management: caps contract quantity at risk_pct of equity and max_contracts."""
    contract_cost = contract_ask * 100
    risk_dollars = equity * risk_pct
    qty = min(max_contracts, int(risk_dollars // contract_cost)) if contract_cost > 0 else 0
    should_trade = signal in ("BUY_CALL", "BUY_PUT") and qty > 0
    return {"contract_cost": contract_cost, "risk_dollars": risk_dollars, "qty": qty, "should_trade": should_trade}


def calculate_combo_position_size(equity, long_ask, short_bid, strategy, risk_pct=RISK_PCT, max_contracts=MAX_CONTRACTS):
    """Risk management for a 2-leg (long+short) combo: sizes off the net debit per contract
    (long leg's ask minus short leg's bid), floored at a cent so a net-credit combo still sizes
    off the max_contracts hard cap instead of being treated as a free, unlimited-size trade."""
    net_debit = max(long_ask - short_bid, 0.01)
    contract_cost = net_debit * 100
    risk_dollars = equity * risk_pct
    qty = min(max_contracts, int(risk_dollars // contract_cost))
    should_trade = strategy in ("RISK_REVERSAL_BULLISH", "RISK_REVERSAL_BEARISH") and qty > 0
    return {"contract_cost": contract_cost, "risk_dollars": risk_dollars, "qty": qty, "should_trade": should_trade}


def estimate_trade_cost(legs, qty):
    """Net capital committed for a set of option legs at the given quantity: ask paid for long
    legs minus bid received for short legs. Used to enforce the aggregate per-cycle risk cap
    across candidates regardless of how many legs each one's strategy needs."""
    return sum(
        qty * 100 * (leg["ask"] if leg["side"] == "long" else -leg.get("bid", leg["ask"]))
        for leg in legs
    )


def position_expiration(symbol):
    m = CONTRACT_PATTERN.match(symbol)
    return datetime.strptime(m.group(2), "%y%m%d").date() if m else None


def exit_reason(position):
    """Exit rules: take-profit / stop-loss on unrealized P&L, or time-based near expiration."""
    plpc = float(position.get("unrealized_plpc", 0))
    expiration = position_expiration(position["symbol"])
    dte = (expiration - date.today()).days if expiration else None
    if plpc >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"
    if plpc <= STOP_LOSS_PCT:
        return "STOP_LOSS"
    if dte is not None and dte <= EXIT_DTE_BUFFER:
        return "TIME_EXIT"
    return None


def screen_universe(closes_by_symbol):
    """Runs calculate_momentum() across a batch of {symbol: closes} and returns only the symbols
    that actually crossed the threshold, ranked by signal strength and capped at the top
    TOP_N_HOTTEST ("hottest"/trending) — bounding worst-case LLM calls downstream on a
    broad-signal day while still giving the sentiment/strategy/risk chain a wide candidate pool."""
    candidates = []
    for symbol, closes in closes_by_symbol.items():
        if len(closes) < MOMENTUM_WINDOW + 1:
            continue
        momentum = calculate_momentum(closes)
        if momentum["signal"] == "NO_TRADE":
            continue
        candidates.append({"symbol": symbol, **momentum})
    candidates.sort(key=lambda c: abs(c["momentum_pct"]), reverse=True)
    return candidates[:TOP_N_HOTTEST]

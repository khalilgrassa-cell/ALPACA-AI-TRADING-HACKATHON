# Strategy

The trading strategy itself, decoupled from how it's executed. `../multi_agent/`
wraps these functions as tools the agents call — change the strategy here
and the pipeline picks it up without touching any agent code.

`momentum_strategy.py` has no I/O and no MCP/LLM-provider dependency: it's pure
functions over plain data (lists of closes, dicts of quotes/positions), which
is what makes it trivially unit-testable and swappable independent of the
execution layer.

## Current strategy: momentum-and-sentiment-gated options, from a small strategy menu

5-day price momentum across the whole trading universe (`UNIVERSE`,
`NASDAQ_100 ∪ SP_100`). Each cycle, `screen_universe` keeps only the top
`TOP_N_HOTTEST` (3) symbols by momentum strength that also cross
`MOMENTUM_THRESHOLD` (`BUY_CALL` above +1%, `BUY_PUT` below -1%, otherwise
`NO_TRADE`). Above the strategy layer, `multi_agent/`'s sentiment and
strategy agents then decide, per symbol, whether to trade a plain long
option or a higher-conviction 2-leg risk-reversal combo — see
`STRATEGY_LEGS`/`STRATEGY_SIGNAL` below and
[`../multi_agent/README.md`](../multi_agent/README.md#strategies) for the
full menu and the code-level guardrails on choosing between them.

| Function / constant | Role |
|---|---|
| `calculate_momentum(closes, window)` | Signal generation: N-day momentum vs. a threshold → `BUY_CALL` / `BUY_PUT` / `NO_TRADE` |
| `screen_universe(closes_by_symbol)` | Runs `calculate_momentum` across the universe, keeps the top `TOP_N_HOTTEST` by strength |
| `STRATEGY_LEGS` / `STRATEGY_SIGNAL` | The strategy menu: which leg(s) (side + option-type signal) each strategy needs, and its base direction |
| `parse_contract(symbol, data)` | Parses an OCC option symbol + quote snapshot into a plain dict |
| `select_contract(contracts, signal, current_price)` | Contract selection: closest strike to the target OTM offset, within the DTE window, among contracts liquid enough to exit later (`MAX_SPREAD_PCT`) — used per leg |
| `calculate_position_size(equity, contract_ask, signal)` | Risk management for a single-leg strategy: caps quantity at `RISK_PCT` of equity and `MAX_CONTRACTS` |
| `calculate_combo_position_size(equity, long_ask, short_bid, strategy)` | Risk management for a 2-leg combo: sizes off the net debit per contract |
| `estimate_trade_cost(legs, qty)` | Net capital committed for a set of legs — long asks paid minus short bids received |
| `exit_reason(position)` | Exit rules: `TAKE_PROFIT` / `STOP_LOSS` on unrealized P&L, or `TIME_EXIT` near expiration — applies per leg/symbol, combo or not |

All the tunable constants (`MOMENTUM_WINDOW`, `MOMENTUM_THRESHOLD`,
`TOP_N_HOTTEST`, `MIN_DTE`/`MAX_DTE`, `OTM_PCT`, `MAX_SPREAD_PCT`,
`RISK_PCT`/`MAX_CONTRACTS`, `TAKE_PROFIT_PCT`/`STOP_LOSS_PCT`/`EXIT_DTE_BUFFER`,
`MIN_HOLD_MINUTES_BEFORE_STOP_LOSS`, `MAX_CYCLE_RISK_PCT`) live at the top of
`momentum_strategy.py`.

## Research-informed changes

Several changes since the strategy first shipped were driven by live trading
evidence plus a review of practitioner writeups and academic momentum
literature on why naive momentum-into-options strategies commonly lose money
— not just parameter tuning for its own sake:

- **`MOMENTUM_WINDOW` — not yet applied, worth revisiting.** A backtest
  across 8 large-cap symbols (~2.2yr daily data, de-meaned correlation
  between past momentum and forward return, controlling for shared
  bull-market drift) found the current 5-day window sits in a short-horizon
  *reversal* regime (wrong sign for a momentum-following signal), while
  40-60 days showed the best continuation correlation in the same sweep —
  consistent with academic momentum literature placing continuation at
  multi-month horizons: [Alpha Architect](https://alphaarchitect.com/short-term-momentum-and-long-term-reversals-can-coexist/),
  [Quantpedia](https://quantpedia.com/strategies/short-term-reversal-in-stocks),
  [Review of Financial Studies](https://academic.oup.com/rfs/article-abstract/38/12/3673/8240327).
  A comparable open-source Alpaca momentum bot, [davidalv2/algo-trading-bot](https://github.com/davidalv2/algo-trading-bot),
  independently uses a 20-day window rather than a multi-day one. Tried
  live once and reverted (unrelated infrastructure issues that day made it
  hard to isolate its effect); the finding stands and is worth revisiting
  on its own.
- **`MAX_SPREAD_PCT` (new — contract-selection liquidity filter).** Live
  evidence: an illiquid contract's own bid/ask spread can exceed a trade's
  entire edge before any real price move happens (an ALNY risk-reversal
  combo lost $945 within ~2 minutes of opening). Liquidity/spread filtering
  on contract selection is standard practice: [Quantamental Trader](https://quantamentaltrader.substack.com/p/how-to-choose-options-based-on-highest).
- **`MIN_HOLD_MINUTES_BEFORE_STOP_LOSS` (new — exit-rule change).** A
  leveraged option's own price can swing 20%+ intraday on bid/ask/IV noise
  from a 1% underlying move alone, and checking a stop against the option's
  own noisy price gets whipsawed by exactly that noise rather than a real
  move: [SteadyOptions](https://steadyoptions.com/articles/why-you-should-never-use-a-stop-loss-in-options-trading-r736/),
  [OneOption](https://oneoption.com/questions/how-can-i-exit-this-option-trade-without-getting-whipsawed/).
  Gating `STOP_LOSS` specifically (never `TAKE_PROFIT` or `TIME_EXIT`) on a
  minimum holding period is a lower-risk mitigation than removing the stop
  entirely, given position sizing (`RISK_PCT`) already bounds the worst case.
- **Not yet done, worth revisiting:** delta/volatility-adjusted strike
  selection instead of a flat `OTM_PCT` — a fixed 2% OTM is too far out for
  low-volatility names and too close for high-volatility ones across a
  ~170-symbol universe with very different vol profiles
  ([TradePro Academy](https://tradeproacademy.com/momentum-swing-trading-options-big-returns-on-investments/)
  suggests ~0.30 delta for momentum/swing setups). Not implemented yet
  because it needs a volatility estimate at selection time that the risk
  agent doesn't currently fetch — a real change, not a quick constant tweak.

## What's deliberately *not* here

Execution-layer concerns stay in `../multi_agent/`: `EXECUTE_EXITS` (the
live-trading safety gate), MCP tool calls, and the LLM prompts and
tool-calling loop. This module only answers "given this data, what should
the strategy do."

## Tests

```bash
pytest strategy/tests
```

Covers every function above with hand-computed cases — momentum thresholds
in both directions, contract selection (including DTE-window exclusion and
the liquidity/spread filter), position sizing (including the
zero-contract-cost edge case), and all three exit reasons plus the hold
case. No network or credentials required. `MIN_HOLD_MINUTES_BEFORE_STOP_LOSS`
is tested in `../multi_agent/tests/test_agents.py` instead, since it needs
an order-history lookup (I/O) that this module deliberately doesn't do.

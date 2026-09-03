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
| `select_contract(contracts, signal, current_price)` | Contract selection: closest strike to the target OTM offset, within the DTE window — used per leg |
| `calculate_position_size(equity, contract_ask, signal)` | Risk management for a single-leg strategy: caps quantity at `RISK_PCT` of equity and `MAX_CONTRACTS` |
| `calculate_combo_position_size(equity, long_ask, short_bid, strategy)` | Risk management for a 2-leg combo: sizes off the net debit per contract |
| `estimate_trade_cost(legs, qty)` | Net capital committed for a set of legs — long asks paid minus short bids received |
| `exit_reason(position)` | Exit rules: `TAKE_PROFIT` / `STOP_LOSS` on unrealized P&L, or `TIME_EXIT` near expiration — applies per leg/symbol, combo or not |

All the tunable constants (`MOMENTUM_WINDOW`, `MOMENTUM_THRESHOLD`,
`TOP_N_HOTTEST`, `MIN_DTE`/`MAX_DTE`, `OTM_PCT`, `RISK_PCT`/`MAX_CONTRACTS`,
`TAKE_PROFIT_PCT`/`STOP_LOSS_PCT`/`EXIT_DTE_BUFFER`,
`MAX_CYCLE_RISK_PCT`) live at the top of `momentum_strategy.py`.

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
in both directions, contract selection (including DTE-window exclusion),
position sizing (including the zero-contract-cost edge case), and all three
exit reasons plus the hold case. No network or credentials required.

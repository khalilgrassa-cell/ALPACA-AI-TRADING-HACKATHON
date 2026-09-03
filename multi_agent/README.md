# Multi-Agent Trading Pipeline

The trading strategy, executed as two specialized LLM agents (plus four
deterministic steps). Each LLM stage is a real model call (via the Groq API,
with automatic fallback across several Groq-hosted models — see "Model &
cost" below) with its own system prompt and its own narrow slice of tools,
orchestrated into a chain by `orchestrator.py`.

```
Market Conditions   Universe Scan     Sentiment      Strategy         Risk            Trading
(deterministic)  →  (deterministic) → Agent      →   Agent    →    (deterministic) → (deterministic)
(is it open?)      (top 3 hottest/    (news         (which        (contract         (execution +
                    trending symbols)  sentiment)     strategy?)    + sizing)         exit checks)
```

If the market-conditions check reports the market closed, the pipeline skips
straight past everything else for that cycle — no scanning, no
sentiment/strategy/risk, no orders. There is no reporting/logging stage —
every decision is printed to stdout as it happens, and actual fills are
tracked in Alpaca's own UI rather than a separate local log.

Per candidate, the chain runs Sentiment → Strategy → Risk → Trading in turn:
the Universe Scan ranks the whole trading universe by momentum strength and
keeps only the top `TOP_N_HOTTEST` (3) symbols; each of those gets a
sentiment read, then a strategy choice (a plain long option, or a
higher-conviction 2-leg combo — see "Strategies" below), then contract
selection/sizing, then execution.

## Why give each agent a calculator tool?

LLMs are unreliable at exact arithmetic across many data points. Rather than
asking the model to compute momentum, strike selection, position sizing, or
P&L thresholds in its head, each agent gets a **local tool** (`local_tools.py`)
that does that one calculation deterministically — thin wrappers over
`../strategy/momentum_strategy.py` (`calculate_momentum`, `select_contract`,
`calculate_position_size`, `exit_reason`). For the LLM agents (Sentiment,
Strategy), the model's job is to decide *when* to call which tool and *how*
to interpret the result, not to do the math itself — `risk_agent.py` and
`trading_agent.py` call the exact same local tools directly in code instead,
since there's no judgment involved in either's sequence (see the pipeline
stages table above). `backtest.py`, further down, reuses the same strategy
functions to validate the signal against history rather than trade it live.

## The pipeline stages

| Stage | File | MCP tools | Local tools | Produces |
|---|---|---|---|---|
| Market Conditions | `market_conditions_agent.py` (no LLM — pure lookup) | `get_clock` | — | `{market_open, reasoning}` |
| Universe Scan | `universe_scanner.py` (no LLM — pure math) | `get_stock_bars` | — | list of `{symbol, signal, current_price, momentum_pct}`, top `TOP_N_HOTTEST` by strength |
| Sentiment | `sentiment_agent.py` | `get_news` | — | `{sentiment, signal, overridden, reasoning}` |
| Strategy | `strategy_agent.py` | — | — | `{strategy, reasoning}` |
| Risk | `risk_agent.py` (no LLM — deterministic) | `get_account_info`, `get_option_chain` (once per option type needed, each biased toward its own OTM side), `get_option_snapshot` (for delta) | `select_option_contract`, `calculate_position_size`, `calculate_combo_position_size` | `{strategy, legs, qty, should_trade, reasoning}` |
| Trader | `trading_agent.py` (no LLM — deterministic) | `place_option_order`, `get_all_positions`, `get_orders`, `close_position` | `check_exit_rule` | `{order_submitted, order_result, reasoning}` per leg; `{open_positions, exits, reasoning}` for exit management |

Each LLM stage's final answer is a JSON object (enforced by its system
prompt and parsed with `llm_tools.parse_json_response`), which becomes the
next stage's input. Market Conditions, the Universe Scan, Risk, and Trading
are the four non-LLM stages: "is the market open" is a single boolean field
on `get_clock`'s response, "does this cross a numeric momentum threshold,
and is it in the top `TOP_N_HOTTEST` by strength" is exact math, and
contract selection/position sizing/order submission/exit checks (2026-09-03:
Risk and Trading were both previously LLM agents) are each the same fixed
sequence of already-deterministic tool calls every time — none of these are
a judgment call, so all four run as plain code instead of a model call.
Beyond the obvious cost/latency savings, this also removes a single point
of failure: an LLM call is the one part of this pipeline that can stall on
a degraded
connection for minutes (see `llm_tools.CALL_TIMEOUT_SECONDS_PER_MODEL`),
and market-open is checked at the very start of every cycle — running it as
a network+parsing call was adding that risk before anything else even ran,
for a question with no ambiguity in the answer.

**The sentiment agent can only veto, never invent or flip, a trade.** Its
prompt says so, but that's not trusted alone —
`sentiment_agent._clamp_signal()` enforces it in code: if the model's JSON
tries to turn a `BUY_CALL` into a `BUY_PUT`, or a `NO_TRADE` into a trade,
the clamp overrides it back. The only transition the sentiment agent can
actually cause is `BUY_CALL`/`BUY_PUT` → `NO_TRADE`. See
`tests/test_sentiment_agent.py` for the cases this covers, including "the
model tries to flip it anyway."

## Strategies

The strategy agent picks from a small menu, defined in
`../strategy/momentum_strategy.py`'s `STRATEGY_LEGS`:

| Strategy | Legs | Chosen when |
|---|---|---|
| `LONG_CALL` | buy 1 call | default bullish choice |
| `LONG_PUT` | buy 1 put | default bearish choice |
| `RISK_REVERSAL_BULLISH` | buy 1 call + sell 1 put | `BUY_CALL` signal *and* positive sentiment agree (higher conviction) |
| `RISK_REVERSAL_BEARISH` | buy 1 put + sell 1 call | `BUY_PUT` signal *and* negative sentiment agree |
| `NO_TRADE` | none | signal is `NO_TRADE`, or the model has real doubts |

**The strategy agent can never point a different direction than the signal
it was given, or manufacture a trade out of `NO_TRADE`.** Same code-level
clamp pattern as the sentiment agent —
`strategy_agent._clamp_strategy()` checks the proposed strategy's direction
(`momentum_strategy.STRATEGY_SIGNAL`) against the signal it was handed, and
forces `NO_TRADE` on any mismatch. See `tests/test_strategy_agent.py`.

The risk-reversal combos are **not submitted as a single atomic multi-leg
order** — `place_option_order` is single-leg, so `orchestrator.submit_legs`
submits the long leg first and only submits the short leg if the long leg's
order actually went through. This bounds (but doesn't eliminate) the risk of
ending up with a naked short and no matching long leg from a partial
failure. Selling options short (the short leg of either combo) requires a
higher Alpaca options trading level than buying long options alone — confirm
your paper account is approved for it before enabling these strategies live.

**There is no reporting/logging stage.** Every decision is printed to stdout
as it happens (see `os._exit`'s explicit `sys.stdout.flush()` in
`orchestrator.py`/`exit_manager.py` — needed because stdout is fully
block-buffered whenever it isn't a terminal, e.g. CI logs, and `os._exit()`
skips the flush that would otherwise happen on normal shutdown), and actual
fills are tracked directly in Alpaca's own UI rather than a separate local
log or JSONL file.

## Files

- `llm_tools.py` — the shared agent runner: converts MCP tool schemas to
  Groq's OpenAI-compatible function-calling format, runs the manual tool-use
  loop (call the model → execute any requested tool calls, against either
  the MCP session or a local tool → feed results back as `role: "tool"`
  messages → repeat until the model answers with plain text), parses the
  final JSON, and falls back across `MODELS` (several Groq-hosted models)
  when the current one is rate-limited — see "Model & cost" below.
- `local_tools.py` — the deterministic calculator tools (wrapping
  `../strategy/momentum_strategy.py`).
- `market_conditions_agent.py`, `universe_scanner.py`, `sentiment_agent.py`,
  `strategy_agent.py`, `risk_agent.py`, `trading_agent.py` — one pipeline
  stage each; the LLM-backed ones also have a `python <file>.py` smoke-test
  entry point.
- `orchestrator.py` — runs every stage in sequence for one decision cycle,
  with the market-closed early-exit and per-candidate error handling
  described above.
- `backtest.py` — a ~180-day backtest of the momentum signal's raw
  directional accuracy against forward returns (not full options P&L — theta
  decay, spreads, and IV changes aren't modeled), plus a price/momentum
  chart saved to `backtest_chart.png`. Pure validation — it never trades.
- `tests/test_local_tools.py` — unit tests for the calculator tools.
- `tests/test_llm_tools.py` — unit tests for the tool-calling loop itself
  (`run_agent`), against a mocked Groq client and a mocked MCP session:
  verifies it calls MCP tools and local tools correctly, feeds results back,
  stops once the model returns no further tool calls, surfaces local-tool
  exceptions and malformed tool-call arguments as error content instead of
  crashing, raises after `max_turns`, and falls back to the next model in
  `MODELS` once the current one exhausts its rate-limit retries or hits a
  too-large-for-this-minute request (Groq returns that as a 413, not a 429 —
  see `_is_oversized_request_error`).
- `tests/test_agents.py` — unit tests for `assess_risk`, `submit_order`, and
  `manage_exits` — all deterministic, all against a mocked MCP session
  (including the `MIN_HOLD_MINUTES_BEFORE_STOP_LOSS` gate's order-history
  lookup).
- `tests/test_market_conditions_agent.py` — unit tests for the open/closed
  cases.
- `tests/test_universe_scanner.py` — unit tests for the deterministic scan.
- `tests/test_sentiment_agent.py` — unit tests for the sentiment wrapper,
  with particular emphasis on `_clamp_signal`'s veto-only invariant.
- `tests/test_strategy_agent.py` — unit tests for the strategy wrapper, with
  particular emphasis on `_clamp_strategy`'s direction-matching invariant.
- `tests/test_orchestrator.py` — unit tests for the full pipeline wiring,
  against mocked agents: verifies each stage's output feeds the next stage's
  arguments correctly, the market-closed short-circuit skips scanning/
  sentiment/strategy/risk/trading, a sentiment veto reaches the risk agent as
  `NO_TRADE`, a combo strategy submits its long leg before its short leg
  (and skips the short leg if the long leg's order fails), and a failure at
  any stage is caught and printed rather than crashing.

All tests above need no API keys, MCP server, or network access.

## Setup

In addition to the root project setup (`../README.md`), you need a Groq API
key:

```bash
cp ../.env.example ../.env   # if you haven't already; fill in GROQ_API_KEY too
```

## Run

From the project root, with the venv active:

```bash
pytest multi_agent/tests                      # offline unit tests for every stage
python multi_agent/market_conditions_agent.py # smoke-test just the market-conditions check
python multi_agent/universe_scanner.py        # smoke-test just the scan
python multi_agent/sentiment_agent.py         # smoke-test just the sentiment check
python multi_agent/strategy_agent.py          # smoke-test just the strategy choice
python multi_agent/risk_agent.py              # smoke-test just contract selection + sizing
python multi_agent/orchestrator.py            # full pipeline
python multi_agent/backtest.py                # backtest + saves multi_agent/backtest_chart.png
```

Sample backtest result:

```
Trading days evaluated: 114 | BUY_CALL: 57 | BUY_PUT: 39 | NO_TRADE: 18
BUY_CALL directional win rate (5d fwd return > 0): 59.6% | avg fwd return: +0.86%
BUY_PUT directional win rate (5d fwd return < 0): 43.6% | avg captured move: -0.86%
```

## Model & cost

Each agent call tries `llm_tools.MODELS` in order — several tool-calling-
capable Groq models (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`qwen/qwen3.6-27b`, `qwen/qwen3.8-27b` as of this writing — Groq's available
model set isn't static; two earlier entries in this list were silently
deprecated and had to be swapped out) under `GROQ_API_KEY`. Groq's on-demand
tier caps both tokens-per-minute (TPM) and tokens-per-day (TPD) per model —
observed live, TPD is real and can bind even when TPM headroom looks fine,
on a heavy-usage day. Once a model's own retries (`MAX_RATE_LIMIT_RETRIES`)
are exhausted, or a request is already too large for a per-minute budget by
itself (Groq returns that as a 413, not a 429; see
`_is_oversized_request_error`), `_create_completion_with_fallback` moves to
the next model in the list — and once every model on the primary account
is exhausted, to the same model list again on a second Groq *account* if
`GROQ_API_KEY_EXITS` is set (see `llm_tools.get_secondary_client` and the
"Scheduled runs" section below for why this needs a different account, not
just a second key on the same one). The watchdog around each turn
(`CALL_TIMEOUT_SECONDS_PER_MODEL` in `run_agent`) scales with the number of
models times the number of accounts being tried, since a single call can
now walk the whole fallback chain on every account in turn.

`TOP_N_HOTTEST` (currently 3 — lowered from 5 on 2026-09-03 after live evidence
that Groq's per-model rate limits are enforced org-wide, not per API key, so a
single cycle's own candidate loop alone could exhaust every model in the
fallback chain) caps how many candidates the Universe Scan
hands to the Sentiment → Strategy → Risk → Trader chain each cycle — raising
it multiplies both Groq token usage and Alpaca order activity roughly
linearly, since each candidate runs its own multi-turn agent chain
sequentially against one MCP session (not concurrently).

## Safety

Enforced by the two remaining LLM agents' own tool choices and code-level
clamps, plus the deterministic stages' own logic, rather than a single `if`
statement:

- The sentiment agent can only veto a signal toward `NO_TRADE`
  (`sentiment_agent._clamp_signal`); the strategy agent can only choose a
  strategy matching the signal's direction, or `NO_TRADE`
  (`strategy_agent._clamp_strategy`).
- The (deterministic) risk agent only sets `should_trade: true` when every
  leg's contract was actually found and the risk-gated quantity is > 0 —
  guaranteed by plain code now, not an LLM's tool-calling discipline.
- The (deterministic) trading agent only calls `place_option_order` when
  `should_trade` is true, and only calls `close_position` when
  `EXECUTE_EXITS` (in `strategy/momentum_strategy.py`, currently `True`) is
  true — and even then, never for a `STOP_LOSS` reason on a position opened
  less than `MIN_HOLD_MINUTES_BEFORE_STOP_LOSS` ago.
- A combo strategy's short leg is only submitted once its long leg's order
  has actually gone through (`orchestrator.submit_legs`).
- Paper trading only — same `mcp_config.json` as the rest of the project.

## Deployment

`orchestrator.py` runs one decision cycle and exits — the deployment story is
"run this container on a schedule," not a long-lived service. All files live
at the **project root** (not inside `multi_agent/`) since the Docker build
needs `mcp_server/`, `strategy/`, and `multi_agent/` in one build context;
the Dockerfile itself lives at `multi_agent/Dockerfile` for discoverability.

### Docker

```bash
# from the project root
docker build -f multi_agent/Dockerfile -t alpaca-multi-agent .
docker run --rm --env-file .env alpaca-multi-agent
```

Or with Compose (also from the project root):

```bash
docker compose run --rm multi-agent
```

The image installs `alpaca-mcp-server` as a `uv tool` at build time (same as
local setup) and never bakes in credentials — `.env` is read at container
*run* time via `--env-file`/`env_file`, and `.dockerignore` excludes it from
the build context entirely.

### Scheduled runs (GitHub Actions + an external scheduler)

`.github/workflows/multi-agent-trading-v2.yml` runs the full pipeline;
`.github/workflows/exit-management-v2.yml` runs the lighter exit-only check
(`exit_manager.py`), tighter than the full cycle since protecting an open
position is more time-sensitive than finding new entries — the full cycle
also runs its own exit check at the end of each run, so this is a tighter
safety net layered on top, not the only place exits happen. Both use the
same `pip install` + `uv tool install` steps as local setup — no Docker
needed for this path, no server to host. Add `ALPACA_API_KEY`,
`ALPACA_SECRET_KEY`, and `GROQ_API_KEY` as repository secrets.

**`multi-agent-trading-v2.yml` also reads an optional `GROQ_API_KEY_EXITS`
secret — a genuinely separate Groq account, used as pipeline-wide overflow
capacity.** 2026-09-03: this secret was originally added to give
exit-management its own key, isolated from the full trading cycle's — but
sharing GROQ_API_KEY between them was never really the mechanism worth
isolating; a single Trading Cycle run's own candidate loop was independently
shown to exhaust every model in `MODELS` on its own, including hitting a
hard per-model *tokens-per-day* cap (not just the usual per-minute one),
which no key-per-workflow split addresses. A key generated *inside the same
Groq account* shares that account's identical TPM/TPD budget regardless of
which key authenticates the request — confirmed live, that specific
approach adds nothing. A genuinely *different* Groq account (a separate
signup, which is what this secret now holds) has its own separate budget,
though: `llm_tools.get_secondary_client()` reads it and
`_create_completion_with_fallback` reaches it only once every model on the
primary account is exhausted for a turn — real overflow capacity for
Sentiment/Strategy, tried from whichever entry point calls `run_agent`.
`exit_manager.py` doesn't read this secret at all anymore: its whole call
chain (Trading Agent, Market Conditions) became fully deterministic (see
the pipeline stages table above), so it has no LLM calls left to fall back
on in the first place.

**Both workflows are triggered by `workflow_dispatch` only — GitHub's own
`schedule` trigger is deliberately not used.** A syntax error briefly
pushed to an earlier version of one of these files got GitHub's cron
*scheduler* stuck: manual `workflow_dispatch` runs worked fine again once
the file was fixed, but the automatic `schedule` trigger never resumed on
its own — not after a disable/re-enable via the API, and not after
renaming to a fresh file path (`-v2`) to force a brand-new workflow
registration with no inherited state. Both attempts still produced zero
`schedule`-triggered runs. Rather than keep trusting GitHub's internal
scheduler, an **external scheduler** (cron-job.org, free tier) calls each
workflow's `workflow_dispatch` REST endpoint directly on a timer:

```
POST https://api.github.com/repos/khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON/actions/workflows/exit-management-v2.yml/dispatches
POST https://api.github.com/repos/khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON/actions/workflows/multi-agent-trading-v2.yml/dispatches
```

Headers: `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`, `Content-Type: application/json`.
Body: `{"ref": "main"}`.

Set up as two cron-job.org jobs: the exit-management URL every 10 minutes
(widened from 5 on 2026-09-03 to reduce contention while Exit Management was
still LLM-backed -- moot today since it has no LLM calls left at all, but
the interval itself is unchanged since then), the multi-agent-trading URL
every 30 minutes, both restricted to
13:30-20:00 UTC on weekdays. Use a **fine-grained personal access token
scoped to only this repo with just "Actions: Read and write" permission**
for the `Authorization` header — not the broader classic token used
elsewhere in this project — since this one now lives on a third-party
service.

Running the full cycle every 30 minutes instead of once a day multiplies
Groq token usage and Alpaca order activity accordingly — each run puts up to
`TOP_N_HOTTEST` (3) candidates through the Sentiment → Strategy → Risk →
Trader chain, so this is meaningfully more API-hungry than the original
once-daily cadence even at this lower candidate cap. Watch for the
per-minute token-budget behavior described in "Model & cost" above under
this heavier schedule.

For a different scheduler (cron on a VM, a Kubernetes CronJob, etc.), the
unit of work is just `docker run --rm --env-file .env alpaca-multi-agent`
(or the bare `python multi_agent/orchestrator.py` invocation) on whatever
cadence you want — nothing in the code assumes a particular scheduler.

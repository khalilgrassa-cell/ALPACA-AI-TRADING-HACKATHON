# Alpaca AI Trading Agent — Submission Write-Up

**lablab.ai × Alpaca AI Trading Agents Hackathon**

## Title

**Alpaca AI Trading Agent** — a multi-agent momentum-options pipeline on Alpaca's MCP server

## Description

An autonomous options-trading pipeline that scans a 170-symbol universe (the
union of the Nasdaq-100 and S&P 100 constituents) every 30 minutes during
market hours, ranks candidates by short-term price momentum, and — for the
top 3 "hottest" names each cycle — runs a chain of specialized agents that
check news sentiment, pick an options strategy, select a specific contract
by delta, size the position against account risk, and submit the order. A
second, faster job (every 10 minutes) manages open-position exits
(take-profit, stop-loss, time-based) independently of the entry cycle. Every
integration with Alpaca — market data, option chains, orders, positions,
account state — goes through Alpaca's official MCP server, the same
interface an LLM tool-use loop would call, rather than the REST API or SDK
directly.

## AI logic

Two of the six pipeline stages are genuinely LLM-driven (Groq-hosted models,
with automatic fallback across several models and, once the primary
account's budget is exhausted for a turn, a second Groq account as
overflow):

- **Sentiment Agent** — reads recent news for a candidate and can only push
  the momentum-derived signal toward `NO_TRADE`; it can never invent a trade
  or flip a signal's direction. This is enforced in code (`_clamp_signal`),
  not just prompted.
- **Strategy Agent** — given the signal, sentiment, and momentum strength,
  picks from a menu: a plain long call/put, or a higher-conviction 2-leg
  risk-reversal combo (buy one option type, sell the other) when momentum
  and sentiment agree in direction. Like the Sentiment Agent, it can never
  point a different direction than the signal it was given, nor manufacture
  a trade out of `NO_TRADE` (`_clamp_strategy`).

The remaining four stages — is the market open, the momentum scan, contract
selection/position sizing, and order submission/exit management — are
deterministic Python calling MCP tools and pure math directly. None of them
involve judgment: momentum ranking is exact arithmetic across a batch of
symbols, contract selection is "pick the strike whose delta is closest to a
0.30 target," and position sizing is a fixed formula against account
equity. Routing these through an LLM would only add cost, latency, and a
new failure mode for zero benefit — so they don't.

## Risk gates

- **Per-trade sizing**: 1% of account equity per position, capped at 5
  contracts.
- **Per-cycle aggregate cap**: total new-position risk across every
  candidate in one cycle is capped at 5% of equity, tracked as a running
  accumulator across the candidate loop — so a cycle with multiple
  simultaneous signals can't stack risk unboundedly.
- **Liquidity filter**: a contract is rejected if it has no real bid, or if
  `(ask − bid) / mid` exceeds 25% — a wide spread alone can exceed a
  trade's entire edge before any real price move happens (observed live).
- **Delta-based strike selection**: contracts are picked by proximity to a
  0.30 delta rather than a flat percentage-OTM offset, so strike distance
  adapts to each underlying's own volatility instead of being systematically
  too far out for calm names and too close for volatile ones.
- **Stop-loss noise gate**: a position must survive a 30-minute minimum hold
  before a stop-loss can close it (take-profit and time-based exits are not
  gated) — a raw option-price stop otherwise whipsaws on bid/ask and IV
  noise rather than a real move in the underlying (observed live: a
  risk-reversal combo closed at a loss ~2 minutes after opening on spread
  noise alone).
- **Sequential combo safety**: a 2-leg combo's short leg is only submitted
  once the long leg's order has actually gone through — not a true atomic
  multi-leg order (Alpaca's option order endpoint is single-leg), but a
  safeguard against ending up with only the higher-risk leg.
- **Paper trading only** — no real funds at risk, enforced at the account
  level.

## Infrastructure

- **Deployment**: GitHub Actions (no server to host), two workflows
  (`multi-agent-trading-v2.yml` every 30 min, `exit-management-v2.yml` every
  10 min during market hours), both with a 10-minute job timeout as a
  backstop against a genuine hang. An external scheduler (cron-job.org)
  triggers both via `workflow_dispatch`, since GitHub's own `schedule`
  trigger got stuck during development and never recovered.
- **Secrets**: Alpaca and Groq credentials live in GitHub Actions secrets,
  never in the repo (`.env` is gitignored and was verified clean before
  every push).
- **Testing**: 131 unit tests, fully mocked — the strategy math, the
  LLM tool-calling loop (including model-fallback and rate-limit recovery),
  every agent wrapper (including the code-level direction/veto clamps), and
  the full orchestrator/exit-manager wiring — run with no live credentials
  needed.
- **Paper account**: `PA39Q320P4PP`, a dedicated account created for this
  submission (not reused from earlier prototyping), starting at $100,000.

## Links

- **Repo**: https://github.com/khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON
- **Live runs (demo)**: https://github.com/khalilgrassa-cell/ALPACA-AI-TRADING-HACKATHON/actions
- **Paper account ID**: PA39Q320P4PP

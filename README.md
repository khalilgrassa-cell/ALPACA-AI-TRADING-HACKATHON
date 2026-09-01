# Alpaca AI Trading Agent

lablab.ai × Alpaca AI Trading Agents Hackathon submission.

A multi-agent options trading pipeline that talks to Alpaca **only** through
Alpaca's official MCP server — the same interface an LLM agent's tool-use
loop would call — rather than the REST API, SDK, or CLI directly. Two
deterministic (no-LLM) stages — is the market open, and a momentum scan of
the whole trading universe — feed four specialized LLM agents (sentiment,
strategy selection, risk, execution), running on Groq (with automatic
fallback across several Groq-hosted models), each reasoning over a narrow
slice of tools and handing their decision to the next stage in the chain.
Market-open and momentum-threshold checks are exact math, not judgment
calls, so they run as plain code instead of a model call — one fewer
network round-trip and single point of failure per cycle. The universe scan
ranks the whole trading universe by momentum strength and keeps the top 5
"hottest"/trending symbols each cycle; the strategy agent then picks from a
small menu — a plain long call/put, or a higher-conviction 2-leg
risk-reversal combo — per candidate. There's no reporting/logging stage —
every decision prints to stdout, and fills are tracked directly in Alpaca's
own UI.

## Project layout

- **`strategy/`** — the trading strategy itself (signal, contract selection,
  risk sizing, exit rules, strategy menu), as pure functions with no I/O.
  The single source of truth the agents wrap as tools. See
  [`strategy/README.md`](strategy/README.md).
- **`mcp_server/`** — the MCP server: how it's installed, configured, and
  connected to; connectivity test and tool/schema inspection scripts. See
  [`mcp_server/README.md`](mcp_server/README.md).
- **`multi_agent/`** — the pipeline itself, its backtest, its unit tests, and
  its Docker/Compose/CI deployment files. See
  [`multi_agent/README.md`](multi_agent/README.md).

```
.
├── strategy/
│   ├── README.md
│   ├── momentum_strategy.py   # signal, contract selection, sizing, exit rules — pure functions
│   └── tests/
│       └── test_momentum_strategy.py
├── mcp_server/
│   ├── README.md
│   ├── mcp_config.json     # declares the alpaca MCP server (command/args/env)
│   ├── mcp_client.py        # connects using that config; mcp_data()/mcp_error() helpers
│   ├── test_connection.py   # connectivity smoke test
│   └── list_tools.py        # dumps tool schemas
├── multi_agent/
│   ├── README.md
│   ├── Dockerfile                    # build from repo root: -f multi_agent/Dockerfile
│   ├── llm_tools.py                   # MCP↔Groq tool bridge + manual tool-use loop
│   ├── local_tools.py                 # thin wrappers over strategy/momentum_strategy.py
│   ├── market_conditions_agent.py     # is the market open? (deterministic, no LLM)
│   ├── universe_scanner.py            # market data → top-5 hottest momentum candidates (no LLM)
│   ├── sentiment_agent.py             # news sentiment check, veto-only
│   ├── strategy_agent.py              # picks a strategy from the menu (long option or risk-reversal combo)
│   ├── risk_agent.py                  # contract selection (per leg) + position sizing
│   ├── trading_agent.py               # order submission (per leg) + exit management
│   ├── orchestrator.py                # runs every stage in sequence
│   ├── backtest.py                    # directional backtest + chart
│   └── tests/
│       ├── test_local_tools.py
│       ├── test_llm_tools.py
│       ├── test_agents.py
│       ├── test_market_conditions_agent.py
│       ├── test_universe_scanner.py
│       ├── test_sentiment_agent.py
│       ├── test_strategy_agent.py
│       ├── test_exit_manager.py
│       └── test_orchestrator.py
├── .github/workflows/
│   ├── multi-agent-trading.yml   # full pipeline, every 15 min during market hours
│   └── exit-management.yml       # exit-only checks, every 5 min during market hours
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .dockerignore
└── .gitignore
```

## Setup

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install the MCP server (persistent uv tool, not a pip package)
uv tool install alpaca-mcp-server

# 2. Create the venv and install the project's Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your credentials
cp .env.example .env   # fill in ALPACA_API_KEY / ALPACA_SECRET_KEY / GROQ_API_KEY
```

## Run

```bash
python mcp_server/test_connection.py   # verify the server starts and the account is reachable
python mcp_server/list_tools.py        # inspect the schema of every tool the agents call
pytest                                   # offline unit tests, no credentials needed

python multi_agent/orchestrator.py      # full pipeline: one decision cycle
python multi_agent/backtest.py          # directional backtest + saves multi_agent/backtest_chart.png
```

## Deploy

```bash
docker build -f multi_agent/Dockerfile -t alpaca-multi-agent .
docker run --rm --env-file .env alpaca-multi-agent
# or: docker compose run --rm multi-agent
```

`.github/workflows/multi-agent-trading.yml` and `exit-management.yml` run
the pipeline on a cron schedule via GitHub Actions (repo secrets, no server
to host) — see
[`multi_agent/README.md`](multi_agent/README.md#scheduled-runs-github-actions)
for the schedule and the reasoning behind it.

## Safety

- Order submission is gated behind `should_trade`; position closes behind
  `EXECUTE_EXITS` (`False` by default, `multi_agent/orchestrator.py`).
- The sentiment agent can only veto a signal toward `NO_TRADE` — never invent
  or flip one — enforced in code, not just prompted
  (`sentiment_agent._clamp_signal`). The strategy agent can only choose a
  strategy matching that signal's direction, or `NO_TRADE`
  (`strategy_agent._clamp_strategy`).
- A risk-reversal combo's short leg is only submitted once its long leg's
  order has actually gone through (`orchestrator.submit_legs`) — see
  `multi_agent/README.md` for why this is a sequential safeguard, not a true
  atomic multi-leg order.
- Alpaca **paper trading only** (`ALPACA_PAPER_TRADE=true` in
  `mcp_server/mcp_config.json`) — no real funds are ever at risk.

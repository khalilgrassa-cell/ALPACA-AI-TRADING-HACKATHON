# Alpaca MCP Server

This folder configures and connects to Alpaca's official MCP server
([`alpaca-mcp-server`](https://pypi.org/project/alpaca-mcp-server/)), the
Model Context Protocol interface the trading agents (in `../multi_agent/`)
drive instead of calling Alpaca's REST API or CLI directly.

## Install the server

The server is a standalone executable, installed once as a persistent
[`uv`](https://docs.astral.sh/uv/) tool (not a Python package inside the
project's venv):

```bash
uv tool install alpaca-mcp-server
```

This puts `alpaca-mcp-server` on your `PATH` (`~/.local/bin`). Verify it:

```bash
alpaca-mcp-server --version
```

## Configuration

`mcp_config.json` declares the server the way an MCP host (Claude Desktop,
etc.) would:

```json
{
  "mcpServers": {
    "alpaca": {
      "command": "alpaca-mcp-server",
      "args": ["serve"],
      "env": {
        "ALPACA_API_KEY": "${ALPACA_API_KEY}",
        "ALPACA_SECRET_KEY": "${ALPACA_SECRET_KEY}",
        "ALPACA_PAPER_TRADE": "true"
      }
    }
  }
}
```

`mcp_client.py` reads this file, substitutes the `${...}` placeholders from
the environment (loaded from the project's root `.env` via `python-dotenv`),
launches the server over stdio, and hands back a ready `ClientSession`:

```python
from mcp_client import connect, mcp_data, mcp_error

async with connect() as session:
    result = await session.call_tool("get_account_info", {})
    account = mcp_data(result)  # None on failure — see note below
```

`ALPACA_PAPER_TRADE=true` is hardcoded in the config — this project only
ever talks to Alpaca's paper trading environment.

### A quirk worth knowing: two different error shapes

Most tool failures set `result.is_error = True`, which `mcp_data()` treats as
a failure and returns `None` for. But `alpaca-mcp-server` reports some
upstream API errors (market-data failures, in particular) as a **normal**
result whose `data` is `{"error": {...}}` instead. `mcp_data()` checks for
both shapes; `mcp_error(result)` gives a best-effort human-readable reason
for either one.

## Verify it works

```bash
python test_connection.py   # starts the server, lists every tool, calls get_account_info
python list_tools.py        # dumps the full input schema for each tool the agent uses
```

## Tools

The server exposes 72 tools in total, covering the full Alpaca Trading and
Market Data APIs. `list_tools.py` prints the input schema for the six the
agent actually calls:

| Tool | Used for |
|---|---|
| `get_account_info` | equity, buying power, account status |
| `get_stock_bars` | historical daily bars → momentum signal |
| `get_option_chain` | contract snapshots (quotes, greeks) for the option chain |
| `place_option_order` | submitting the risk-gated order |
| `get_all_positions` | open positions, for the exit-rule check |
| `close_position` | closing a position on a TAKE_PROFIT/STOP_LOSS/TIME_EXIT signal |

Other categories available but unused by this agent: stock/crypto quotes and
trades, order management (cancel/replace/get by id), watchlists, corporate
actions, portfolio history, asset/calendar/clock lookups, locates, and a
built-in Alpaca docs/API-spec search.

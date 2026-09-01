"""Loads the Alpaca MCP server config and hands out a connected session."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "mcp_config.json"


def _resolve_env(env_template):
    resolved = {}
    for key, value in env_template.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            resolved[key] = os.environ[value[2:-1]]
        else:
            resolved[key] = value
    return resolved


def load_server_params(server_name="alpaca"):
    config = json.loads(CONFIG_PATH.read_text())
    server = config["mcpServers"][server_name]
    return StdioServerParameters(
        command=server["command"],
        args=server["args"],
        env=_resolve_env(server.get("env", {})),
    )


@asynccontextmanager
async def connect(server_name="alpaca"):
    server_params = load_server_params(server_name)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def mcp_data(result):
    """Unwraps a successful tool result. Returns None on failure — including the
    alpaca-mcp-server's habit of reporting some upstream API errors (e.g. market
    data failures) as a normal result whose data is {"error": {...}} rather than
    setting is_error=True."""
    if result.is_error:
        return None
    data = result.structured_content.get("data")
    if isinstance(data, dict) and "error" in data:
        return None
    return data


def mcp_error(result):
    """Best-effort human-readable reason a failed mcp_data() call failed."""
    if result.is_error:
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return " ".join(texts) if texts else str(result.content)
    data = result.structured_content.get("data")
    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        return error.get("message", str(error)) if isinstance(error, dict) else str(error)
    return "unknown error"

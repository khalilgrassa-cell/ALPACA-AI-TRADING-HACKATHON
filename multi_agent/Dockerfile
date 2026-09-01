# Build from the project root: docker build -f multi_agent/Dockerfile -t alpaca-multi-agent .
FROM python:3.12-slim

WORKDIR /app

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# alpaca-mcp-server is installed as a persistent uv tool, not a pip dependency —
# same as the local setup documented in mcp_server/README.md.
RUN pip install --no-cache-dir uv && uv tool install alpaca-mcp-server

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server/ ./mcp_server/
COPY strategy/ ./strategy/
COPY multi_agent/ ./multi_agent/

# Credentials are supplied at `docker run` time (--env-file/-e), never baked into the image.
CMD ["python", "multi_agent/orchestrator.py"]

"""Shared plumbing for running a Groq-hosted model against a mix of MCP tools and local (in-process) tools."""
import asyncio
import json
import os
import time

from groq import APIConnectionError, APIStatusError, BadRequestError, Groq, RateLimitError

MODEL = "openai/gpt-oss-120b"

# Every Groq account's on-demand tier caps tokens-per-minute *per model*, not in aggregate — so
# when one model's budget for this minute is exhausted, a different Groq-hosted model still has
# its own untouched budget. MODELS lists tool-calling-capable Groq models in fallback order;
# _create_completion_with_fallback walks it, moving to the next one once a model's own retries
# (see MAX_RATE_LIMIT_RETRIES below) are exhausted. All are pulled from the same GROQ_API_KEY —
# no extra credentials needed.
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
]
# 2026-09-02: llama-3.3-70b-versatile and moonshotai/kimi-k2-instruct were removed from Groq's
# lineup for this account (both now 404 model_not_found) — replaced with qwen/qwen3.6-27b and
# qwen/qwen3.8-27b, verified live to support tool-calling. Re-check client.models.list() if
# fallback models start failing again; Groq's available model set isn't static.

# This Groq account's on-demand tier has an 8000 tokens-per-minute cap — true for every
# tool-calling-capable model we tested, not just this one. A single tool result (e.g. get_news
# returning full article text, or get_option_chain returning dozens of contracts) can blow past
# that alone, on top of the tool schemas resent every turn. Keep everything lean: short tool
# results, trimmed schema descriptions (see mcp_tool_to_tool_schema), and a modest max_tokens on
# the completion itself (short JSON answers don't need much).
MAX_TOOL_RESULT_CHARS = 1500
# 2026-09-02: observed live — every Risk Agent call was hitting a genuine 413 "Request too large"
# on every model (all four share the same 8000 TPM cap), 197-877 tokens over the limit. Not
# rate-limit contention — a real, deterministic size violation: get_option_chain alone can return
# ~60,000 raw characters before truncation, and the Risk Agent's tool schemas (get_option_chain +
# select_option_contract + calculate_position_size/calculate_combo_position_size) add ~1,300-1,400
# tokens on top, resent every turn. Lowered from 2500 to give real headroom instead of hovering
# right at the boundary.
MAX_RATE_LIMIT_RETRIES = 3

# 2026-09-02: observed live — a model near the front of MODELS (hit far more often than the
# others, since every call tries models in the same fixed order) occasionally returns a 429 whose
# retry-after header reflects its own per-minute reset window (tens of seconds), not how long a
# healthy fallback model would take to just try next. Retrying that one model 3 times against its
# own retry-after could alone burn most of CALL_TIMEOUT_SECONDS_PER_MODEL's total budget before
# _create_completion_with_fallback ever reaches a model with headroom — this cap keeps a single
# model's own retries cheap so the fallback chain actually gets used instead of stalling on the
# first model whose limited budget is exhausted.
MAX_RETRY_DELAY_SECONDS = 10.0

# Backstop above the client's own REQUEST_TIMEOUT_SECONDS. A degraded connection can trickle
# occasional bytes and keep resetting httpx's per-read timeout without ever tripping it — observed
# live: a call hung 20+ minutes past the client's nominal 60s timeout with no exception raised.
# Our own code needs a deadline that isn't at the mercy of the HTTP client's internals.
#
# This is a *per-model* budget. run_agent scales the actual wait_for timeout by len(models),
# since _create_completion_with_fallback runs the whole multi-model retry chain — every model's
# own MAX_RATE_LIMIT_RETRIES attempts plus their backoff sleeps — inside one such call. Observed
# live: with a single flat 90s covering all of MODELS, a call that needed to fall back through
# 2-3 rate-limited models before finding room got aborted by this watchdog before it ever reached
# a working one, turning a case the fallback should have handled into a hard failure instead.
CALL_TIMEOUT_SECONDS_PER_MODEL = 90.0

FINAL_ANSWER_TOOL_NAME = "submit_final_answer"
FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": FINAL_ANSWER_TOOL_NAME,
        "description": (
            "Submit your final answer for this task. Call this exactly once, after you are done "
            "reasoning and calling any other tools, with your answer as the arguments — do not "
            "invent any other tool name to deliver your final answer."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
    },
}

_client = None

# The Groq client is synchronous — its network call would otherwise block run_agent's whole
# event loop for however long the connection takes to respond, with no way for an outer
# asyncio.wait_for/timeout to interrupt it (observed live: a stalled request hung well past
# 120s with no error). A hard client-side timeout bounds that; _create_completion_with_retry
# is also run in a thread (see run_agent) so a stall there can't freeze anything else.
REQUEST_TIMEOUT_SECONDS = 60.0


def get_client():
    global _client
    if _client is None:
        # .strip() guards against a stray trailing newline/whitespace from copy-pasting the key
        # into a GitHub Actions secret — that alone makes httpx reject every request outright
        # with "Illegal header value", instantly and with no retry helping (observed live).
        _client = Groq(api_key=os.environ["GROQ_API_KEY"].strip(), timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


MAX_SCHEMA_DESCRIPTION_CHARS = 150


def _trim_schema_descriptions(node):
    """Alpaca's MCP tool schemas carry documentation-length "description" fields (multi-line
    prose per parameter). Those get resent on every single turn of a conversation, and are a
    major contributor to the 8000 TPM account cap this project runs under — trim them to a
    terse hint; the model needs enough to use the tool, not the full docs."""
    if isinstance(node, dict):
        trimmed = {}
        for key, value in node.items():
            if key == "description" and isinstance(value, str) and len(value) > MAX_SCHEMA_DESCRIPTION_CHARS:
                trimmed[key] = value[:MAX_SCHEMA_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "..."
            else:
                trimmed[key] = _trim_schema_descriptions(value)
        return trimmed
    if isinstance(node, list):
        return [_trim_schema_descriptions(item) for item in node]
    return node


def mcp_tool_to_tool_schema(tool):
    dumped = tool.model_dump()
    schema = dumped.get("inputSchema") or dumped.get("input_schema")
    description = tool.description or ""
    if len(description) > MAX_SCHEMA_DESCRIPTION_CHARS:
        description = description[:MAX_SCHEMA_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "..."
    return {
        "type": "function",
        "function": {"name": tool.name, "description": description, "parameters": _trim_schema_descriptions(schema)},
    }


def _truncate(text, limit=MAX_TOOL_RESULT_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} more characters]"


def mcp_result_to_content(result):
    """Best-effort JSON/text rendering of an MCP CallToolResult for a tool result message.

    Passes error payloads through as-is rather than filtering them: the model reads the raw
    tool output and can react to an embedded {"error": ...} shape in its own reasoning.
    Truncated to MAX_TOOL_RESULT_CHARS — some tools (get_news in particular) can return enough
    raw text alone to blow a model's per-minute token budget in one turn."""
    if result.is_error:
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return _truncate(" ".join(texts) if texts else str(result.content))
    data = result.structured_content.get("data") if result.structured_content else None
    return _truncate(json.dumps(data))


class LocalTool:
    """A Python-side tool the model can call, executed in-process alongside MCP tools."""

    def __init__(self, name, description, input_schema, func):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    def to_tool_schema(self):
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.input_schema},
        }


def _create_completion_with_retry(client, **kwargs):
    """Retries on RateLimitError with backoff — a compounding multi-candidate cycle can exceed
    the per-minute token budget even after individual results are truncated. A 413 "request too
    large" for a single oversized message won't be helped by waiting, but a 429-style "you've used
    this minute's budget" will — retrying either way is harmless since the request itself is small.

    Also retries on APIConnectionError (DNS blip, TCP reset, TLS failure) — a transient network
    hiccup shouldn't be indistinguishable from a hard failure. A malformed request that always
    raises this (e.g. an invalid header value) will just exhaust the same retries and fall
    through to the next model in _create_completion_with_fallback, same as a 429 would."""
    last_exc = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            last_exc = exc
            retry_after = getattr(exc, "response", None) and exc.response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2 ** attempt * 5
            time.sleep(min(delay, MAX_RETRY_DELAY_SECONDS))
        except APIConnectionError as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt * 5, MAX_RETRY_DELAY_SECONDS))
    raise last_exc


def _is_oversized_request_error(exc):
    """True for a single request that already exceeds the per-minute token budget by itself.
    Observed live: Groq returns this as a plain 413 APIStatusError (not a 429 RateLimitError),
    but its body still carries code: "rate_limit_exceeded" — it's the same token-budget problem,
    just not the SDK-recognized exception type. Waiting won't help on the same model (the request
    is the same size every retry), so this should skip straight to the next model instead of
    burning _create_completion_with_retry's backoff on a request that will fail identically."""
    if isinstance(exc, RateLimitError):
        return False  # already its own exception type, handled separately
    body = getattr(exc, "body", None)
    return isinstance(exc, APIStatusError) and isinstance(body, dict) and body.get("error", {}).get("code") == "rate_limit_exceeded"


def _create_completion_with_fallback(client, models, **kwargs):
    """Tries each model in order, moving on once a model's own MAX_RATE_LIMIT_RETRIES are
    exhausted (a 429) or a single request is already too large for that model's per-minute budget
    (a 413 — see _is_oversized_request_error) — see MODELS above for why a same-account, same-key
    model swap helps. Raises the last model's error only if every model in the list still fails."""
    last_exc = None
    for model in models:
        try:
            return _create_completion_with_retry(client, model=model, **kwargs), model
        except RateLimitError as exc:
            last_exc = exc
            print(f"Model {model} is rate-limited even after retries — falling back to the next model.")
        except APIConnectionError as exc:
            last_exc = exc
            print(f"Model {model}'s request kept failing to connect even after retries — falling back to the next model.")
        except BadRequestError as exc:
            # Must come before the APIStatusError clause below — BadRequestError is a subclass of
            # it, so listing them in this order is what lets this one run first. Tags which model
            # actually produced the malformed generation, so a caller that can't recover a usable
            # answer from it (see _recover_final_answer_from_tool_use_failure) can retry with that
            # one model excluded instead of aborting outright — reasoning models occasionally emit
            # prose instead of a parseable tool call/JSON (observed live: Groq's own
            # "output_parse_failed"), and a different model in the list often just works.
            exc.failed_model = model
            raise
        except APIStatusError as exc:
            if not _is_oversized_request_error(exc):
                raise
            last_exc = exc
            print(f"Model {model}'s request exceeded its per-minute token budget — falling back to the next model.")
    raise last_exc


def _recover_final_answer_from_tool_use_failure(exc):
    """openai/gpt-oss-120b occasionally invents an undeclared tool name (observed: "JSON",
    "commentary") to deliver its final answer despite the explicit submit_final_answer tool.
    Groq rejects the call with a 400, but its response body still contains the model's intended
    arguments in error.failed_generation — recover the answer from there instead of aborting the
    whole agent run. Returns the answer as a JSON string, or None if this isn't that failure mode."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", {})
    if error.get("code") != "tool_use_failed":
        return None
    try:
        attempted = json.loads(error["failed_generation"])
        return json.dumps(attempted["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


async def run_agent(system_prompt, user_prompt, mcp_session, mcp_tool_names, local_tools=None, max_turns=8, models=None, required_keys=None):
    """Runs one Groq-hosted model to completion (a manual tool-use loop) against a subset of the
    MCP server's tools plus any local_tools, returning (final_text, full_message_history)."""
    local_tools = local_tools or []
    local_tool_map = {tool.name: tool for tool in local_tools}
    models = models or MODELS

    all_mcp_tools = (await mcp_session.list_tools()).tools
    tool_schemas = [mcp_tool_to_tool_schema(t) for t in all_mcp_tools if t.name in mcp_tool_names]
    tool_schemas += [tool.to_tool_schema() for tool in local_tools]
    tool_schemas.append(FINAL_ANSWER_TOOL)

    client = get_client()
    system_prompt = (
        system_prompt
        + f"\n\nWhen you have your final answer, call the {FINAL_ANSWER_TOOL_NAME} tool with it as "
        "the arguments, instead of writing it as plain text."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # See CALL_TIMEOUT_SECONDS_PER_MODEL: one call below can walk the whole of `models`, so the
    # watchdog has to cover all of them, not just one.
    call_timeout = CALL_TIMEOUT_SECONDS_PER_MODEL * len(models)

    for _ in range(max_turns):
        remaining_models = models
        while True:
            try:
                response, used_model = await asyncio.wait_for(
                    asyncio.to_thread(
                        _create_completion_with_fallback,
                        client, remaining_models, messages=messages, max_tokens=1024,
                        tools=tool_schemas or None, tool_choice="auto" if tool_schemas else None,
                    ),
                    timeout=call_timeout,
                )
                message = response.choices[0].message
                if not message.tool_calls and not (message.content and message.content.strip()):
                    # Observed live: a model can return successfully (no exception raised) with
                    # neither a tool call nor any text — an unusable empty turn, not a real final
                    # answer. Treat it the same as a failed model rather than letting an empty
                    # string get accepted as the "final answer" and crash parse_json_response with
                    # an opaque "Expecting value: line 1 column 1" three call frames downstream.
                    remaining_models = [m for m in remaining_models if m != used_model]
                    if not remaining_models:
                        raise RuntimeError(
                            "Every model in the fallback chain returned an empty response "
                            "(no tool call, no content) for this turn."
                        )
                    continue
                if required_keys:
                    # Observed live: a model can call submit_final_answer with technically valid
                    # but semantically empty JSON (e.g. "{}") — passes as a response, but is
                    # missing every field the caller actually needs. Validate here, before
                    # accepting this turn, so an unusable final answer gets the same
                    # drop-this-model-and-retry treatment as an empty response, rather than
                    # surfacing as a confusing "missing expected key(s)" error two frames upstream
                    # with no chance for the fallback chain to recover.
                    final_call = next(
                        (tc for tc in (message.tool_calls or []) if tc.function.name == FINAL_ANSWER_TOOL_NAME), None,
                    )
                    candidate_answer = (
                        final_call.function.arguments if final_call is not None
                        else message.content if not message.tool_calls else None
                    )
                    if candidate_answer is not None:
                        try:
                            parse_json_response(candidate_answer, required_keys)
                        except ValueError:
                            remaining_models = [m for m in remaining_models if m != used_model]
                            if not remaining_models:
                                raise RuntimeError(
                                    f"Every model in the fallback chain returned a final answer "
                                    f"missing required key(s) {required_keys}."
                                )
                            continue
                break
            except asyncio.TimeoutError:
                raise RuntimeError(f"Groq completion call did not return within {call_timeout}s")
            except BadRequestError as exc:
                recovered = _recover_final_answer_from_tool_use_failure(exc)
                if recovered is not None:
                    return recovered, messages
                # Unrecoverable malformed generation from one model — drop just that model and
                # retry with whatever's left, rather than aborting the whole cycle over what's
                # often a one-model quirk (see _create_completion_with_fallback's failed_model tag).
                failed_model = getattr(exc, "failed_model", None)
                remaining_models = [m for m in remaining_models if m != failed_model]
                if failed_model is None or not remaining_models:
                    raise
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ] if message.tool_calls else None,
        })

        if not message.tool_calls:
            return message.content or "", messages

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            if name == FINAL_ANSWER_TOOL_NAME:
                return tool_call.function.arguments, messages
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: invalid arguments JSON — {exc}"})
                continue
            if name in local_tool_map:
                try:
                    content = _truncate(json.dumps(local_tool_map[name].func(**args)))
                except Exception as exc:
                    content = f"Error: {exc}"
            else:
                result = await mcp_session.call_tool(name, args)
                content = mcp_result_to_content(result)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")


def parse_json_response(text, required_keys=None):
    """Extracts a JSON object from a model's final text answer, tolerating ```json fences.

    If required_keys is given, validates the parsed object has all of them — the model
    occasionally returns some other JSON-shaped text as its "final answer" (observed: leftover
    tool-call arguments) instead of the shape its system prompt asked for. Failing loudly here
    with a clear message beats a confusing KeyError three call frames away downstream."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    if required_keys:
        missing = required_keys - parsed.keys()
        if missing:
            raise ValueError(f"Model's final answer is missing expected key(s) {missing}: {parsed}")
    return parsed

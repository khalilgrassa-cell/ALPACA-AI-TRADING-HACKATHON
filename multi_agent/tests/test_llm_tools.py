"""Unit tests for the Groq<->MCP tool-calling loop — mocks both the Groq client and the MCP session, so no API key, MCP server, or network access is needed."""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from groq import BadRequestError

import llm_tools
from llm_tools import LocalTool, mcp_result_to_content, mcp_tool_to_tool_schema, parse_json_response, run_agent
from groq import APIStatusError, RateLimitError


def bad_request_error(body):
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/"))
    return BadRequestError("bad request", response=response, body=body)


def rate_limit_error():
    response = httpx.Response(429, headers={"retry-after": "0"}, request=httpx.Request("POST", "https://api.groq.com/"))
    return RateLimitError("rate limited", response=response, body={"error": {"message": "rate limited"}})


def oversized_request_error():
    # Reproduces a real live failure: Groq returns a single too-large-for-this-minute request as
    # a plain 413 (not 429), so the SDK raises the generic APIStatusError rather than
    # RateLimitError — even though the body's own code still says "rate_limit_exceeded".
    response = httpx.Response(413, request=httpx.Request("POST", "https://api.groq.com/"))
    body = {"error": {"message": "Request too large for model `x` ... tokens per minute (TPM)", "code": "rate_limit_exceeded"}}
    return APIStatusError("request too large", response=response, body=body)


def tool_call(name, arguments, id_="tc1"):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def chat_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def mcp_result(data=None, is_error=False, error_text="failed"):
    if is_error:
        return SimpleNamespace(is_error=True, structured_content={}, content=[SimpleNamespace(text=error_text)])
    return SimpleNamespace(is_error=False, structured_content={"data": data}, content=[])


def test_mcp_tool_to_tool_schema_converts_schema():
    tool = SimpleNamespace(
        name="get_account_info", description="desc",
        model_dump=lambda: {"inputSchema": {"type": "object", "properties": {}}},
    )
    assert mcp_tool_to_tool_schema(tool) == {
        "type": "function",
        "function": {"name": "get_account_info", "description": "desc", "parameters": {"type": "object", "properties": {}}},
    }


def test_mcp_result_to_content_success_is_json():
    assert mcp_result_to_content(mcp_result(data={"equity": "100000"})) == '{"equity": "100000"}'


def test_mcp_result_to_content_truncates_oversized_payloads():
    huge_data = {"articles": ["x" * 1000 for _ in range(20)]}  # ~20,000 chars of JSON
    content = mcp_result_to_content(mcp_result(data=huge_data))
    assert len(content) <= llm_tools.MAX_TOOL_RESULT_CHARS + 60  # + room for the "[truncated ...]" suffix
    assert "truncated" in content


def test_mcp_result_to_content_error_is_text():
    assert mcp_result_to_content(mcp_result(is_error=True, error_text="boom")) == "boom"


def test_parse_json_response_raw():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_fenced():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_response_accepts_object_with_all_required_keys():
    assert parse_json_response('{"a": 1, "b": 2}', required_keys={"a", "b"}) == {"a": 1, "b": 2}


def test_parse_json_response_raises_on_missing_required_keys():
    # Mirrors a real failure observed live: the Risk Agent returned leftover tool-call
    # arguments (a get_option_chain query) instead of its {chosen_contract, qty, ...} shape.
    malformed = json.dumps({"underlying_symbol": "CRWD", "strike_price_gte": 495})
    try:
        parse_json_response(malformed, required_keys={"chosen_contract", "qty", "should_trade", "reasoning"})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "chosen_contract" in str(exc)


def test_run_agent_calls_mcp_tool_then_local_tool_then_returns_final_text():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[
        SimpleNamespace(name="get_stock_bars", description="", model_dump=lambda: {"inputSchema": {"type": "object", "properties": {}}}),
    ]))
    mcp_session.call_tool = AsyncMock(return_value=mcp_result(data={"bars": {"QQQ": [{"c": 100}]}}))

    local_calls = []

    def fake_calc(closes):
        local_calls.append(closes)
        return {"signal": "NO_TRADE"}

    local_tool = LocalTool("calculate_momentum", "desc", {"type": "object", "properties": {}}, fake_calc)

    responses = [
        chat_response(tool_calls=[tool_call("get_stock_bars", {}, "tc1")]),
        chat_response(tool_calls=[tool_call("calculate_momentum", {"closes": [100]}, "tc2")]),
        chat_response(content='{"signal": "NO_TRADE"}'),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=responses)

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user prompt", mcp_session,
            mcp_tool_names={"get_stock_bars"}, local_tools=[local_tool],
        ))

    assert text == '{"signal": "NO_TRADE"}'
    assert fake_client.chat.completions.create.call_count == 3
    mcp_session.call_tool.assert_called_once_with("get_stock_bars", {})
    assert local_calls == [[100]]


def test_run_agent_reports_local_tool_error_without_crashing():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    def failing_tool(**kwargs):
        raise ValueError("bad input")

    local_tool = LocalTool("bad_tool", "desc", {"type": "object", "properties": {}}, failing_tool)

    responses = [
        chat_response(tool_calls=[tool_call("bad_tool", {}, "tc1")]),
        chat_response(content="recovered"),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=responses)

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[local_tool],
        ))

    assert text == "recovered"
    tool_result_message = messages[-2]
    assert tool_result_message["role"] == "tool"
    assert "bad input" in tool_result_message["content"]


def test_run_agent_reports_invalid_tool_arguments_without_crashing():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    local_tool = LocalTool("some_tool", "desc", {"type": "object", "properties": {}}, lambda **kw: {"ok": True})

    bad_call = SimpleNamespace(id="tc1", function=SimpleNamespace(name="some_tool", arguments="{not valid json"))
    responses = [
        chat_response(tool_calls=[bad_call]),
        chat_response(content="handled"),
    ]
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=responses)

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[local_tool],
        ))

    assert text == "handled"
    assert "invalid arguments JSON" in messages[-2]["content"]


def test_is_oversized_request_error_true_for_413_with_rate_limit_code():
    assert llm_tools._is_oversized_request_error(oversized_request_error()) is True


def test_is_oversized_request_error_false_for_other_status_errors():
    assert llm_tools._is_oversized_request_error(bad_request_error({"error": {"code": "invalid_request_error"}})) is False


def test_is_oversized_request_error_false_for_rate_limit_error_itself():
    # RateLimitError is handled by its own except clause in _create_completion_with_fallback —
    # _is_oversized_request_error only needs to recognize the 413-flavored lookalike.
    assert llm_tools._is_oversized_request_error(rate_limit_error()) is False


def test_run_agent_falls_back_to_next_model_when_a_request_is_too_large_for_it():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=[
        oversized_request_error(),  # model A: single request already too large, no retry — straight to B
        chat_response(content='{"ok": true}'),  # model B succeeds
    ])

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[],
            models=["model-a", "model-b"],
        ))

    assert text == '{"ok": true}'
    calls = fake_client.chat.completions.create.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["model-a", "model-b"]


def test_run_agent_reraises_a_non_rate_limit_api_status_error():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    response = httpx.Response(422, request=httpx.Request("POST", "https://api.groq.com/"))
    unrelated_error = APIStatusError("nope", response=response, body={"error": {"code": "invalid_request_error"}})
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=unrelated_error)

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        try:
            asyncio.run(run_agent("system", "user", mcp_session, mcp_tool_names=set(), local_tools=[]))
            assert False, "expected APIStatusError to propagate"
        except APIStatusError:
            pass


def test_run_agent_falls_back_to_next_model_when_first_is_rate_limited():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=[
        *([rate_limit_error()] * llm_tools.MAX_RATE_LIMIT_RETRIES),  # model A exhausts its retries
        chat_response(content='{"ok": true}'),  # model B succeeds
    ])

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[],
            models=["model-a", "model-b"],
        ))

    assert text == '{"ok": true}'
    calls = fake_client.chat.completions.create.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["model-a"] * llm_tools.MAX_RATE_LIMIT_RETRIES + ["model-b"]


def test_run_agent_raises_after_max_turns():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    mcp_session.call_tool = AsyncMock(return_value=mcp_result(data={"ok": True}))

    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(
        return_value=chat_response(tool_calls=[tool_call("some_tool", {}, "tc1")])
    )

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        try:
            asyncio.run(run_agent(
                "system", "user", mcp_session, mcp_tool_names={"some_tool"}, local_tools=[], max_turns=2,
            ))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    assert fake_client.chat.completions.create.call_count == 2


def test_run_agent_recovers_final_answer_from_hallucinated_tool_use_failure():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    error_body = {
        "error": {
            "message": "Tool call validation failed: attempted to call tool 'commentary' which was not in request.tools",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": json.dumps({
                "name": "commentary",
                "arguments": {"market_open": True, "reasoning": "clock says open"},
            }),
        }
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=bad_request_error(error_body))

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[],
        ))

    assert json.loads(text) == {"market_open": True, "reasoning": "clock says open"}


def test_run_agent_scales_call_timeout_by_number_of_models():
    # Regression test for a real live failure: _create_completion_with_fallback can walk the
    # whole of `models` (each with its own MAX_RATE_LIMIT_RETRIES backoff) inside one
    # asyncio.wait_for call. A flat per-model timeout there let the outer watchdog abort a call
    # before the fallback ever reached a model that had room — the timeout must scale with
    # len(models) instead.
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(return_value=chat_response(content='{"ok": true}'))

    captured_timeouts = []
    real_wait_for = asyncio.wait_for

    async def spying_wait_for(awaitable, timeout):
        captured_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    with patch.object(llm_tools, "get_client", return_value=fake_client), \
         patch.object(llm_tools, "CALL_TIMEOUT_SECONDS_PER_MODEL", 10), \
         patch.object(llm_tools.asyncio, "wait_for", new=spying_wait_for):
        asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[],
            models=["model-a", "model-b", "model-c"],
        ))

    assert captured_timeouts == [30]  # 10 (per-model) * 3 models


def test_run_agent_raises_clear_error_when_completion_call_times_out():
    # Reproduces a real live failure: a Groq call can stall well past its own client-side
    # timeout (a degraded connection can keep resetting httpx's per-read timeout without ever
    # tripping it). CALL_TIMEOUT_SECONDS_PER_MODEL is our own backstop, independent of the HTTP
    # client's internals — patched tiny here so the test doesn't actually wait.
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    def hanging_create(**kwargs):
        time.sleep(0.3)
        return chat_response(content="too late")

    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=hanging_create)

    with patch.object(llm_tools, "get_client", return_value=fake_client), \
         patch.object(llm_tools, "CALL_TIMEOUT_SECONDS_PER_MODEL", 0.05):
        try:
            asyncio.run(run_agent("system", "user", mcp_session, mcp_tool_names=set(), local_tools=[]))
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "did not return within" in str(exc)


def test_run_agent_falls_back_to_next_model_on_unrecoverable_malformed_generation():
    # Reproduces a real live failure: openai/gpt-oss-120b (a reasoning model) occasionally emits
    # plain prose instead of a parseable tool call, which Groq rejects as a 400 with code
    # "output_parse_failed" — unlike "tool_use_failed", there's no usable answer to recover from
    # failed_generation here, so the only useful move is trying a different model.
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    error_body = {
        "error": {
            "message": "Parsing failed. The model generated output that could not be parsed.",
            "type": "invalid_request_error",
            "code": "output_parse_failed",
            "failed_generation": "We have get_clock response: is_open true. So market is open now.",
        }
    }
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=[
        bad_request_error(error_body),  # model-a produces unparseable prose
        chat_response(content='{"ok": true}'),  # model-b succeeds
    ])

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        text, messages = asyncio.run(run_agent(
            "system", "user", mcp_session, mcp_tool_names=set(), local_tools=[],
            models=["model-a", "model-b"],
        ))

    assert text == '{"ok": true}'
    calls = fake_client.chat.completions.create.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["model-a", "model-b"]


def test_run_agent_reraises_bad_request_errors_that_arent_tool_use_hallucinations():
    mcp_session = MagicMock()
    mcp_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(
        side_effect=bad_request_error({"error": {"message": "invalid model", "code": "model_not_found"}})
    )

    with patch.object(llm_tools, "get_client", return_value=fake_client):
        try:
            asyncio.run(run_agent("system", "user", mcp_session, mcp_tool_names=set(), local_tools=[]))
            assert False, "expected BadRequestError to propagate"
        except BadRequestError:
            pass


def test_get_client_strips_whitespace_from_api_key(monkeypatch):
    # Reproduces a real live failure: a trailing newline copy-pasted into the GROQ_API_KEY
    # GitHub Actions secret made httpx reject every request outright as an illegal header value,
    # before any network call — instantly, and with no retry or fallback able to help.
    monkeypatch.setattr(llm_tools, "_client", None)
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_test_key\n")

    with patch.object(llm_tools, "Groq") as fake_groq_cls:
        llm_tools.get_client()

    assert fake_groq_cls.call_args.kwargs["api_key"] == "gsk_test_key"

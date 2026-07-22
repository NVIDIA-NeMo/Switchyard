# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`AdvisorToolCallBackend` on the OpenAI-Chat wire.

Mirror of ``test_advisor_tool_call_backend.py`` (the Anthropic suite) with
OpenAI-format executors: fixtures are plain ``chat.completion`` bodies and
``chat.completion.chunk`` dicts, matching what ``OpenAiNativeBackend``'s SSE
parser yields. Covers the OpenAI-specific shapes — function-tool injection,
system-message steering, ``tool_calls`` interception, ``role:"tool"``
feedback with 1:1 ids, delta reassembly + final-usage chunk, sibling drop
with vendor-field whitelisting — plus the format-dispatch seams (mixed tiers,
rejected formats).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from switchyard.lib.backends.advisor_tool_call_backend import (
    AdvisorToolCallBackend,
    _prepend_system_message,
)
from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse, ChatResponseType

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _completion_resp(
    message: dict, *, finish_reason: str = "stop", usage: dict | None = None,
) -> ChatResponse:
    """An OpenAI chat.completion ChatResponse with the given assistant message."""
    body = {
        "id": "chatcmpl-x", "object": "chat.completion", "model": "exec-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }
    return ChatResponse.openai_completion(body)


def _text_turn(text: str = "done") -> ChatResponse:
    return _completion_resp({"role": "assistant", "content": text})


def _advisor_call(call_id: str = "adv1", *, arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": "advisor", "arguments": arguments}}


def _advisor_turn(
    *, extra_calls: list[dict] | None = None, finish_reason: str = "tool_calls",
    arguments: str = "{}",
) -> ChatResponse:
    """A completion turn calling the advisor (optionally with sibling calls)."""
    message = {
        "role": "assistant",
        "content": "let me ask",
        "reasoning_content": "hidden chain of thought",
        "tool_calls": [*(extra_calls or []), _advisor_call(arguments=arguments)],
    }
    return _completion_resp(message, finish_reason=finish_reason)


async def _agen(events):
    for event in events:
        yield event


def _stream_resp(events: list[dict]) -> ChatResponse:
    return ChatResponse.openai_stream(ResponseStream(_agen(events)))


def _chunk(delta: dict, *, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-x", "object": "chat.completion.chunk", "model": "exec-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _usage_chunk(*, prompt: int = 12, completion: int = 7, cached: int = 5) -> dict:
    """The final usage-only chunk stream_options.include_usage produces."""
    return {
        "id": "chatcmpl-x", "object": "chat.completion.chunk", "model": "exec-model",
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


def _advisor_stream_events() -> list[dict]:
    """Chunks for a streamed advisor call: id/name first, arguments in fragments."""
    return [
        _chunk({"role": "assistant", "content": "consul"}),
        _chunk({"content": "ting"}),
        _chunk({"tool_calls": [{"index": 0, "id": "adv-s", "type": "function",
                                "function": {"name": "advisor", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
        _chunk({}, finish_reason="tool_calls"),
        _usage_chunk(),
    ]


def _terminal_stream_events() -> list[dict]:
    return [
        _chunk({"role": "assistant", "content": "final"}),
        _chunk({"content": " answer"}),
        _chunk({}, finish_reason="stop"),
        _usage_chunk(prompt=20, completion=5, cached=0),
    ]


def _exec_backend(*responses) -> MagicMock:
    b = MagicMock()
    b.call = AsyncMock(side_effect=list(responses))
    b.startup = AsyncMock()
    b.shutdown = AsyncMock()
    return b


def _advisor(*advice: str) -> MagicMock:
    c = MagicMock()
    c.advise = AsyncMock(side_effect=[(a, {"prompt_tokens": 100, "completion_tokens": 9})
                                      for a in advice])
    return c


def _failing_advisor(exc: Exception) -> MagicMock:
    c = MagicMock()
    c.advise = AsyncMock(side_effect=exc)
    return c


def _config(**overrides) -> AdvisorConfig:
    base: dict = {
        "executor": {"model": "exec-model", "base_url": "http://exec.test", "api_key": "k",
                     "format": "openai"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "openai"},
        "executor_steering": "STEER",
        "advisor_length_line": "(LENGTH)",
    }
    base.update(overrides)
    return AdvisorConfig(**base)


def _backend(config, executor_backend, advisor_caller) -> AdvisorToolCallBackend:
    return AdvisorToolCallBackend(
        config, executor_backend=executor_backend, advisor_caller=advisor_caller,
    )


def _request(**overrides) -> ChatRequest:
    body: dict = {
        "model": "incoming",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "build X"},
        ],
        "tools": [{"type": "function",
                   "function": {"name": "bash", "description": "Run a shell command",
                                "parameters": {"type": "object"}}}],
    }
    body.update(overrides)
    return ChatRequest.openai_chat(body)  # type: ignore[arg-type]


def _sent_body(exec_b: MagicMock, call_index: int) -> dict:
    return exec_b.call.await_args_list[call_index].args[1].to_body()


def _content_text(body: dict) -> str:
    return str(body["choices"][0]["message"].get("content") or "")


# ---------------------------------------------------------------------------
# Request shaping: tool injection + steering
# ---------------------------------------------------------------------------


async def test_advisor_tool_appended_in_function_shape() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(), exec_b, _advisor()).call(ProxyContext(), _request())
    tools = _sent_body(exec_b, 0)["tools"]
    assert [t["function"]["name"] for t in tools] == ["bash", "advisor"]
    assert tools[-1]["type"] == "function"
    assert tools[-1]["function"]["parameters"]["properties"] == {}  # parameterless


async def test_steering_prepends_system_message_and_appends_length_line() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(), exec_b, _advisor()).call(ProxyContext(), _request())
    messages = _sent_body(exec_b, 0)["messages"]
    assert messages[0] == {"role": "system", "content": "STEER\n\nsys"}
    assert messages[1]["content"] == "build X\n\n(LENGTH)"


async def test_steering_inserts_system_message_when_absent() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(), exec_b, _advisor()).call(
        ProxyContext(),
        _request(messages=[{"role": "user", "content": "build X"}]),
    )
    messages = _sent_body(exec_b, 0)["messages"]
    assert messages[0] == {"role": "system", "content": "STEER"}
    assert messages[1]["content"] == "build X\n\n(LENGTH)"


async def test_steering_disabled_leaves_request_untouched() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(inject_steering=False), exec_b, _advisor()).call(
        ProxyContext(), _request(),
    )
    body = _sent_body(exec_b, 0)
    assert body["messages"][0]["content"] == "sys"
    assert body["messages"][1]["content"] == "build X"
    assert [t["function"]["name"] for t in body["tools"]] == ["bash", "advisor"]


def test_prepend_system_message_shapes() -> None:
    assert _prepend_system_message([], "S") == [{"role": "system", "content": "S"}]
    assert _prepend_system_message(
        [{"role": "developer", "content": "base"}], "S",
    ) == [{"role": "developer", "content": "S\n\nbase"}]
    assert _prepend_system_message(
        [{"role": "system", "content": [{"type": "text", "text": "base"}]}], "S",
    ) == [{"role": "system", "content": [{"type": "text", "text": "S"},
                                         {"type": "text", "text": "base"}]}]


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


async def test_advisor_free_turn_returns_without_consult() -> None:
    exec_b = _exec_backend(_text_turn("done, all good"))
    adv = _advisor()
    ctx = ProxyContext()
    resp = await _backend(_config(), exec_b, adv).call(ctx, _request())
    assert _content_text(resp.to_body()) == "done, all good"
    assert exec_b.call.await_count == 1
    assert adv.advise.await_count == 0
    assert ctx.selected_model == "exec-model"


async def test_client_tool_call_turn_returns_without_consult() -> None:
    """A turn calling only the client's own tools is terminal — hand it back."""
    exec_b = _exec_backend(_completion_resp(
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "b1", "type": "function",
                         "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
        finish_reason="tool_calls",
    ))
    adv = _advisor()
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert resp.to_body()["choices"][0]["finish_reason"] == "tool_calls"
    assert adv.advise.await_count == 0


async def test_advisor_call_is_intercepted_and_fed_back() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn("informed answer"))
    adv = _advisor("try the channel-based pattern")
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 2
    messages = _sent_body(exec_b, 1)["messages"]
    assistant_turn = messages[-2]
    assert assistant_turn["role"] == "assistant"
    assert [tc["function"]["name"] for tc in assistant_turn["tool_calls"]] == ["advisor"]
    result_turn = messages[-1]
    assert result_turn == {
        "role": "tool", "tool_call_id": "adv1",
        "content": "try the channel-based pattern",
    }
    assert _content_text(resp.to_body()) == "informed answer"


async def test_advisor_detected_even_with_stop_finish_reason() -> None:
    """Some OSS servers mislabel tool-call turns; detection is by presence."""
    exec_b = _exec_backend(_advisor_turn(finish_reason="stop"), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert adv.advise.await_count == 1


async def test_parallel_advisor_calls_one_consult_fanned_results() -> None:
    """N advisor calls in one turn → one consult, one tool message per id."""
    message = {
        "role": "assistant", "content": None,
        "tool_calls": [_advisor_call("adv1"), _advisor_call("adv2")],
    }
    exec_b = _exec_backend(
        _completion_resp(message, finish_reason="tool_calls"), _text_turn(),
    )
    adv = _advisor("shared advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    messages = _sent_body(exec_b, 1)["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["adv1", "adv2"]
    assert all(m["content"] == "shared advice" for m in tool_msgs)


async def test_nonempty_arguments_round_trip_verbatim() -> None:
    """Arguments are never parsed (the tool is parameterless) but round-trip."""
    exec_b = _exec_backend(
        _advisor_turn(arguments='{"question": "which approach?"}'), _text_turn(),
    )
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assistant_turn = _sent_body(exec_b, 1)["messages"][-2]
    assert assistant_turn["tool_calls"][0]["function"]["arguments"] == (
        '{"question": "which approach?"}'
    )


async def test_transcript_carries_tools_conversation_and_current_turn() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    transcript = adv.advise.await_args.kwargs["transcript"]
    assert "bash" in transcript                      # function-tool summary
    assert "Run a shell command" in transcript
    assert "build X" in transcript                   # conversation
    assert "it is consulting you now" in transcript  # current-turn section
    assert "let me ask" in transcript                # current-turn text
    assert "hidden chain of thought" not in transcript  # reasoning_content excluded
    assert adv.advise.await_args.kwargs["system"] == _config().advisor_system_prompt


async def test_max_uses_returns_error_result_without_consult() -> None:
    exec_b = _exec_backend(_advisor_turn(), _advisor_turn(), _text_turn())
    adv = _advisor("first advice")
    resp = await _backend(_config(max_uses=1), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 3
    second_result = _sent_body(exec_b, 2)["messages"][-1]
    assert second_result["content"] == "[advisor unavailable: max_uses exceeded]"
    assert _content_text(resp.to_body()) == "done"


async def test_fail_open_marks_advisor_unavailable_and_continues() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn("proceeded unadvised"))
    adv = _failing_advisor(RuntimeError("advisor down"))
    resp = await _backend(_config(fail_open=True), exec_b, adv).call(
        ProxyContext(), _request(),
    )
    result = _sent_body(exec_b, 1)["messages"][-1]
    assert result["content"] == "[advisor unavailable: RuntimeError]"
    assert _content_text(resp.to_body()) == "proceeded unadvised"


async def test_fail_closed_raises() -> None:
    exec_b = _exec_backend(_advisor_turn())
    adv = _failing_advisor(RuntimeError("advisor down"))
    with pytest.raises(RuntimeError, match="advisor down"):
        await _backend(_config(fail_open=False), exec_b, adv).call(
            ProxyContext(), _request(),
        )


async def test_sibling_tool_calls_dropped_and_vendor_fields_whitelisted() -> None:
    """Mixed advisor + client calls: only the advisor call is kept, and vendor
    fields like reasoning_content are dropped from the rebuilt turn."""
    sibling = {"id": "b9", "type": "function",
               "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}
    exec_b = _exec_backend(_advisor_turn(extra_calls=[sibling]), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assistant_turn = _sent_body(exec_b, 1)["messages"][-2]
    assert [tc["id"] for tc in assistant_turn["tool_calls"]] == ["adv1"]
    assert assistant_turn["content"] == "let me ask"       # text survives
    assert "reasoning_content" not in assistant_turn        # vendor field dropped
    # Every kept tool_call id has exactly one tool message (OpenAI 1:1 rule).
    tool_msgs = [m for m in _sent_body(exec_b, 1)["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["adv1"]


async def test_hard_cap_returns_last_round() -> None:
    exec_b = _exec_backend(*[_advisor_turn() for _ in range(8)])
    adv = _advisor(*["advice"] * 8)
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert exec_b.call.await_count == 8
    assert adv.advise.await_count == 2  # default max_uses=2; the rest get error results
    assert resp.to_body()["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_advisor_call_reassembled_then_terminal_replayed() -> None:
    terminal_events = _terminal_stream_events()
    exec_b = _exec_backend(
        _stream_resp(_advisor_stream_events()), _stream_resp(terminal_events),
    )
    adv = _advisor("advice")
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    # Reassembled from deltas: the fed-back tool message targets the streamed
    # id, the fragmented arguments merged, and the streamed text reached the
    # transcript.
    messages = _sent_body(exec_b, 1)["messages"]
    assert messages[-1] == {"role": "tool", "tool_call_id": "adv-s", "content": "advice"}
    assert messages[-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "consulting" in adv.advise.await_args.kwargs["transcript"]
    # The terminal turn's buffered events are replayed verbatim.
    assert resp.response_type == ChatResponseType.OPENAI_STREAM
    replayed = [event async for event in resp.stream]
    assert replayed == terminal_events


async def test_streaming_missing_tool_call_id_is_synthesized() -> None:
    events = [
        _chunk({"role": "assistant",
                "tool_calls": [{"index": 0,
                                "function": {"name": "advisor", "arguments": ""}}]}),
        _chunk({}, finish_reason="tool_calls"),
    ]
    exec_b = _exec_backend(_stream_resp(events), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    result = _sent_body(exec_b, 1)["messages"][-1]
    assert result["tool_call_id"] == "call_switchyard_0"


# ---------------------------------------------------------------------------
# Stats accounting
# ---------------------------------------------------------------------------


async def test_planner_bucket_prices_internal_turns_and_consults() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn())
    adv = _advisor("advice")
    stats = MagicMock()
    stats.record_planner_usage = AsyncMock()
    stats.record_success = AsyncMock()
    backend = AdvisorToolCallBackend(
        _config(), stats_accumulator=stats, executor_backend=exec_b, advisor_caller=adv,
    )
    await backend.call(ProxyContext(), _request())

    assert stats.record_planner_usage.await_count == 2
    by_model = {c.kwargs["model"]: c.kwargs for c in stats.record_planner_usage.await_args_list}
    # The intercepted executor turn is priced under the executor model, with
    # OpenAI cached tokens read from prompt_tokens_details.
    assert by_model["exec-model"]["prompt_tokens"] == 10
    assert by_model["exec-model"]["cached_tokens"] == 4
    assert by_model["adv-model"]["prompt_tokens"] == 100
    assert by_model["adv-model"]["completion_tokens"] == 9
    stats.record_success.assert_awaited_once()


async def test_missing_usage_records_zeros() -> None:
    exec_b = _exec_backend(
        _completion_resp(
            {"role": "assistant", "content": None, "tool_calls": [_advisor_call()]},
            finish_reason="tool_calls", usage={},
        ),
        _text_turn(),
    )
    adv = _advisor("advice")
    stats = MagicMock()
    stats.record_planner_usage = AsyncMock()
    stats.record_success = AsyncMock()
    backend = AdvisorToolCallBackend(
        _config(), stats_accumulator=stats, executor_backend=exec_b, advisor_caller=adv,
    )
    await backend.call(ProxyContext(), _request())
    internal = stats.record_planner_usage.await_args_list[0].kwargs
    assert internal["prompt_tokens"] == 0
    assert internal["cached_tokens"] == 0


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def test_openai_executor_advertises_openai_chat() -> None:
    backend = _backend(_config(), _exec_backend(), _advisor())
    assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]


def test_mixed_tiers_follow_executor_wire() -> None:
    anthropic_advisor = {"model": "claude-opus-4-8", "base_url": "http://adv.test",
                         "api_key": "k", "format": "anthropic"}
    backend = _backend(
        _config(advisor=anthropic_advisor), _exec_backend(), _advisor(),
    )
    assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]

    openai_advisor = {"model": "deepseek/deepseek-r2", "base_url": "http://adv.test",
                      "api_key": "k", "format": "openai"}
    anthropic_exec = {"model": "claude-opus-4-7", "base_url": "http://exec.test",
                      "api_key": "k", "format": "anthropic"}
    backend = _backend(
        _config(executor=anthropic_exec, advisor=openai_advisor),
        _exec_backend(), _advisor(),
    )
    assert backend.supported_request_types == [ChatRequestType.ANTHROPIC]


def test_auto_format_with_injected_backend_raises() -> None:
    auto_exec = {"model": "m", "base_url": "http://exec.test", "api_key": "k",
                 "format": "auto"}
    with pytest.raises(ValueError, match="pin format"):
        _backend(_config(executor=auto_exec), _exec_backend(), _advisor())


# ---------------------------------------------------------------------------
# seed_plan_advice
# ---------------------------------------------------------------------------


async def test_seed_consults_before_first_executor_turn_and_injects() -> None:
    """Seed advice is fetched before the executor runs and lands in the first
    user message, ahead of the steering length line."""
    exec_b = _exec_backend(_text_turn())
    adv = _advisor("1. read the docs first")
    config = _config(seed_plan_advice=True)
    await _backend(config, exec_b, adv).call(ProxyContext(), _request())
    assert adv.advise.await_count == 1
    assert adv.advise.await_args.kwargs["system"] == config.advisor_system_prompt
    user = _sent_body(exec_b, 0)["messages"][1]
    assert user["role"] == "user"
    assert config.seed_advice_prefix.strip() in user["content"]
    assert "1. read the docs first" in user["content"]
    assert "(LENGTH)" in user["content"]  # steering still applied after the seed


async def test_seed_cached_for_later_turns_of_the_session() -> None:
    """The second request of a session re-injects the cached advice without a
    fresh consult; the advisor tool loop still works on top."""
    exec_b = _exec_backend(_text_turn(), _text_turn())
    adv = _advisor("1. plan")
    backend = _backend(_config(seed_plan_advice=True), exec_b, adv)
    await backend.call(ProxyContext(), _request())
    await backend.call(ProxyContext(), _request(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "build X"},
        {"role": "assistant", "content": "working on it"},
        {"role": "user", "content": "tool output: ok"},
    ]))
    assert adv.advise.await_count == 1
    assert "1. plan" in str(_sent_body(exec_b, 1)["messages"][1]["content"])


async def test_seed_fail_open_proceeds_unseeded() -> None:
    exec_b = _exec_backend(_text_turn())
    adv = _failing_advisor(RuntimeError("advisor down"))
    config = _config(seed_plan_advice=True)
    resp = await _backend(config, exec_b, adv).call(ProxyContext(), _request())
    assert _content_text(resp.to_body()) == "done"
    user = _sent_body(exec_b, 0)["messages"][1]
    assert "advisor reviewed" not in str(user["content"])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tool-call :class:`AdvisorToolCallBackend` (native Anthropic).

All loop tests inject a fake executor backend (returns ``ChatResponse``) and a
fake advisor caller — no network. They cover the loop contract: the advisor
tool and steering are injected; an advisor-free turn returns verbatim; an
advisor ``tool_use`` is intercepted, consulted, and fed back as a
``tool_result``; ``max_uses`` returns an error result without a consult;
fail-open marks the advisor unavailable; sibling client tool calls are dropped
from the appended turn; the hard cap bounds a runaway loop. Both streaming
(block reassembly + verbatim replay) and completion paths are exercised, plus
the transcript / injection helpers.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from switchyard.lib.backends.advisor_loop_backend import _anthropic_text
from switchyard.lib.backends.advisor_tool_call_backend import (
    AdvisorToolCallBackend,
    _prepend_system,
    _with_length_line,
)
from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest, ChatResponse, ChatResponseType

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _completion_resp(content: list[dict], *, stop_reason: str = "end_turn") -> ChatResponse:
    """An Anthropic completion ChatResponse with the given content blocks."""
    body = {
        "id": "msg-x", "type": "message", "role": "assistant", "model": "exec-model",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 4},
    }
    return ChatResponse.anthropic_completion(body)


def _text_turn(text: str = "done") -> ChatResponse:
    return _completion_resp([{"type": "text", "text": text}])


def _advisor_turn(*, extra_blocks: list[dict] | None = None) -> ChatResponse:
    """A completion turn calling the advisor (optionally with sibling blocks)."""
    content = [
        {"type": "text", "text": "let me ask"},
        *(extra_blocks or []),
        {"type": "tool_use", "id": "adv1", "name": "advisor", "input": {}},
    ]
    return _completion_resp(content, stop_reason="tool_use")


async def _agen(events):
    for event in events:
        yield event


def _stream_resp(events: list[dict]) -> ChatResponse:
    return ChatResponse.anthropic_stream(AnthropicResponseStream(_agen(events)))


def _advisor_stream_events() -> list[dict]:
    """SSE events for a streamed turn that calls the advisor (tool input via deltas)."""
    return [
        {"type": "message_start",
         "message": {"usage": {"input_tokens": 12, "cache_read_input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "consulting"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "adv-s", "name": "advisor", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": ""}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]


def _terminal_stream_events() -> list[dict]:
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 20}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "final answer"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]


def _exec_backend(*responses) -> MagicMock:
    b = MagicMock()
    b.call = AsyncMock(side_effect=list(responses))
    b.startup = AsyncMock()
    b.shutdown = AsyncMock()
    return b


def _advisor(*advice: str) -> MagicMock:
    c = MagicMock()
    c.advise = AsyncMock(side_effect=[(a, {"input_tokens": 100, "output_tokens": 9})
                                      for a in advice])
    return c


def _failing_advisor(exc: Exception) -> MagicMock:
    c = MagicMock()
    c.advise = AsyncMock(side_effect=exc)
    return c


def _config(**overrides) -> AdvisorConfig:
    base: dict = {
        "executor": {"model": "exec-model", "base_url": "http://exec.test", "api_key": "k",
                     "format": "anthropic"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
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
        "model": "incoming", "system": "sys",
        "messages": [{"role": "user", "content": "build X"}],
        "tools": [{"name": "bash", "description": "Run a shell command",
                   "input_schema": {"type": "object"}}],
    }
    body.update(overrides)
    return ChatRequest.anthropic(body)  # type: ignore[arg-type]


def _sent_body(exec_b: MagicMock, call_index: int) -> dict:
    return exec_b.call.await_args_list[call_index].args[1].to_body()


# ---------------------------------------------------------------------------
# Request shaping: tool injection + steering
# ---------------------------------------------------------------------------


async def test_advisor_tool_appended_to_client_tools() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(), exec_b, _advisor()).call(ProxyContext(), _request())
    tools = _sent_body(exec_b, 0)["tools"]
    assert [t["name"] for t in tools] == ["bash", "advisor"]
    assert tools[-1]["input_schema"]["properties"] == {}  # parameterless


async def test_steering_prepends_system_and_appends_length_line() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(), exec_b, _advisor()).call(ProxyContext(), _request())
    body = _sent_body(exec_b, 0)
    assert body["system"] == "STEER\n\nsys"
    assert body["messages"][0]["content"] == "build X\n\n(LENGTH)"


async def test_steering_disabled_leaves_request_untouched() -> None:
    exec_b = _exec_backend(_text_turn())
    await _backend(_config(inject_steering=False), exec_b, _advisor()).call(
        ProxyContext(), _request(),
    )
    body = _sent_body(exec_b, 0)
    assert body["system"] == "sys"
    assert body["messages"][0]["content"] == "build X"
    assert [t["name"] for t in body["tools"]] == ["bash", "advisor"]  # tool still offered


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


async def test_advisor_free_turn_returns_without_consult() -> None:
    exec_b = _exec_backend(_text_turn("done, all good"))
    adv = _advisor()
    ctx = ProxyContext()
    resp = await _backend(_config(), exec_b, adv).call(ctx, _request())
    assert _anthropic_text(resp.to_body()) == "done, all good"
    assert exec_b.call.await_count == 1
    assert adv.advise.await_count == 0
    assert ctx.selected_model == "exec-model"


async def test_client_tool_call_turn_returns_without_consult() -> None:
    """A turn calling only the client's own tools is terminal — hand it back."""
    exec_b = _exec_backend(_completion_resp(
        [{"type": "tool_use", "id": "b1", "name": "bash", "input": {"command": "ls"}}],
        stop_reason="tool_use",
    ))
    adv = _advisor()
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert resp.to_body()["stop_reason"] == "tool_use"
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
    assert any(b.get("type") == "tool_use" and b.get("name") == "advisor"
               for b in assistant_turn["content"])
    result_turn = messages[-1]
    assert result_turn["role"] == "user"
    assert result_turn["content"] == [{
        "type": "tool_result", "tool_use_id": "adv1",
        "content": "try the channel-based pattern",
    }]
    assert _anthropic_text(resp.to_body()) == "informed answer"


async def test_transcript_carries_tools_conversation_and_current_turn() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    transcript = adv.advise.await_args.kwargs["transcript"]
    assert "bash" in transcript                      # tool summary
    assert "build X" in transcript                   # conversation
    assert "it is consulting you now" in transcript  # current-turn section
    assert "let me ask" in transcript                # current-turn text
    assert adv.advise.await_args.kwargs["system"] == _config().advisor_system_prompt


async def test_max_uses_returns_error_result_without_consult() -> None:
    exec_b = _exec_backend(_advisor_turn(), _advisor_turn(), _text_turn())
    adv = _advisor("first advice")  # only one consult available on purpose
    resp = await _backend(_config(max_uses=1), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 3
    second_result = _sent_body(exec_b, 2)["messages"][-1]["content"][0]
    assert second_result["content"] == "[advisor unavailable: max_uses exceeded]"
    assert _anthropic_text(resp.to_body()) == "done"


async def test_fail_open_marks_advisor_unavailable_and_continues() -> None:
    exec_b = _exec_backend(_advisor_turn(), _text_turn("proceeded unadvised"))
    adv = _failing_advisor(RuntimeError("advisor down"))
    resp = await _backend(_config(fail_open=True), exec_b, adv).call(
        ProxyContext(), _request(),
    )
    result = _sent_body(exec_b, 1)["messages"][-1]["content"][0]
    assert result["content"] == "[advisor unavailable: RuntimeError]"
    assert _anthropic_text(resp.to_body()) == "proceeded unadvised"


async def test_fail_closed_raises() -> None:
    exec_b = _exec_backend(_advisor_turn())
    adv = _failing_advisor(RuntimeError("advisor down"))
    with pytest.raises(RuntimeError, match="advisor down"):
        await _backend(_config(fail_open=False), exec_b, adv).call(
            ProxyContext(), _request(),
        )


async def test_sibling_tool_calls_dropped_from_appended_turn() -> None:
    """Mixed advisor + client calls: the appended turn keeps only the advisor call."""
    sibling = {"type": "tool_use", "id": "b9", "name": "bash", "input": {"command": "ls"}}
    exec_b = _exec_backend(_advisor_turn(extra_blocks=[sibling]), _text_turn())
    adv = _advisor("advice")
    await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assistant_turn = _sent_body(exec_b, 1)["messages"][-2]
    tool_uses = [b for b in assistant_turn["content"] if b.get("type") == "tool_use"]
    assert [b["id"] for b in tool_uses] == ["adv1"]
    # Non-tool blocks (the text) survive.
    assert any(b.get("type") == "text" for b in assistant_turn["content"])


async def test_hard_cap_returns_last_round() -> None:
    exec_b = _exec_backend(*[_advisor_turn() for _ in range(8)])
    adv = _advisor(*["advice"] * 8)
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert exec_b.call.await_count == 8
    assert adv.advise.await_count == 2  # default max_uses=2; the rest get error results
    assert resp.to_body()["stop_reason"] == "tool_use"


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
    # The advisor call was reassembled from deltas: the fed-back tool_result
    # targets the streamed tool_use id, and the current-turn text made it into
    # the transcript.
    result_turn = _sent_body(exec_b, 1)["messages"][-1]
    assert result_turn["content"][0]["tool_use_id"] == "adv-s"
    assert "consulting" in adv.advise.await_args.kwargs["transcript"]
    # The terminal turn's buffered events are replayed verbatim.
    assert resp.response_type == ChatResponseType.ANTHROPIC_STREAM
    replayed = [event async for event in resp.stream]
    assert replayed == terminal_events


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
    # The intercepted executor turn (client never sees it) is priced under the
    # executor model, cache reads included.
    assert by_model["exec-model"]["prompt_tokens"] == 10
    assert by_model["exec-model"]["cached_tokens"] == 4
    # The consult is priced under the advisor model.
    assert by_model["adv-model"]["prompt_tokens"] == 100
    assert by_model["adv-model"]["completion_tokens"] == 9
    stats.record_success.assert_awaited_once()
    assert stats.record_success.await_args.args[0] == "exec-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_serialize_transcript_drops_oldest_when_over_cap() -> None:
    backend = _backend(_config(transcript_max_chars=300), _exec_backend(), _advisor())
    messages = [
        {"role": "user", "content": f"turn-{i}: " + "x" * 80} for i in range(10)
    ]
    transcript = backend._serialize_transcript(messages, "current", [])
    assert "turn-9" in transcript          # newest kept
    assert "turn-0" not in transcript      # oldest dropped
    assert "earlier messages omitted" in transcript
    assert transcript.endswith("current")


def test_serialize_transcript_no_truncation_under_cap() -> None:
    backend = _backend(_config(), _exec_backend(), _advisor())
    messages = [{"role": "user", "content": "hello"}]
    transcript = backend._serialize_transcript(messages, "", [])
    assert "hello" in transcript
    assert "omitted" not in transcript
    assert transcript.endswith("(no text)")


def test_prepend_system_shapes() -> None:
    assert _prepend_system(None, "S") == "S"
    assert _prepend_system("base", "S") == "S\n\nbase"
    assert _prepend_system([{"type": "text", "text": "base"}], "S") == [
        {"type": "text", "text": "S"}, {"type": "text", "text": "base"},
    ]


def test_with_length_line_targets_first_user_message() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
    ]
    out = _with_length_line(messages, "(L)")
    assert out[0]["content"] == "first\n\n(L)"
    assert out[2]["content"][-1]["type"] == "tool_result"  # later user turns untouched


def test_with_length_line_block_content() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "first"}]}]
    out = _with_length_line(messages, "(L)")
    assert out[0]["content"][-1] == {"type": "text", "text": "(L)"}

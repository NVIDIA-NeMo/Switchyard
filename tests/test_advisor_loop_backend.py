# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the review-gate :class:`AdvisorLoopBackend` (native Anthropic).

All loop tests inject a fake executor backend (returns ``ChatResponse``) and a
fake advisor caller — no network. They cover the gate contract: tool-use turns
pass through unreviewed; the first no-tool-use turn is reviewed once; APPROVE
returns it; REDO re-invokes the executor to continue; the review is
once-per-session and the session is pure passthrough afterward; fail-open
approves. Separate tests cover the Anthropic reviewer caller (respx) and the
pure helpers. Both streaming and completion executor responses are exercised.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from switchyard.lib.backends.advisor_loop_backend import (
    AdvisorLoopBackend,
    _anthropic_text,
    _AnthropicAdvisorCaller,
    _build_advisor_caller,
    _messages_url,
    _OpenAiAdvisorCaller,
    _parse_verdict,
    _session_key,
    _usage_tokens,
)
from switchyard.lib.backends.llm_target import coerce_llm_target
from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
from switchyard.lib.chat_response.openai_chat import ResponseStream as OpenAIResponseStream
from switchyard.lib.profiles.advisor_config import AdvisorConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _completion_resp(*, text=None, tool_use=False, model="exec-model") -> ChatResponse:
    """An Anthropic completion ChatResponse (text and/or a tool_use block)."""
    content: list[dict] = []
    if tool_use:
        content.append({"type": "tool_use", "id": "t1", "name": "bash", "input": {}})
    if text is not None:
        content.append({"type": "text", "text": text})
    body = {
        "id": "msg-x", "type": "message", "role": "assistant", "model": model,
        "content": content,
        "stop_reason": "tool_use" if tool_use else "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    return ChatResponse.anthropic_completion(body)


async def _agen(events):
    for event in events:
        yield event


def _stream_resp(*, text=None, tool_use=False) -> ChatResponse:
    """An Anthropic streaming ChatResponse (SSE event dicts)."""
    events: list[dict] = [{"type": "message_start", "message": {"usage": {"input_tokens": 10}}}]
    if tool_use:
        events.append({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}})
        events.append({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                       "usage": {"output_tokens": 1}})
    else:
        events.append({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}})
        if text:
            events.append({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": text}})
        events.append({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": 3}})
    events.append({"type": "message_stop"})
    return ChatResponse.anthropic_stream(AnthropicResponseStream(_agen(events)))


def _exec_backend(*responses) -> MagicMock:
    b = MagicMock()
    b.call = AsyncMock(side_effect=list(responses))
    b.startup = AsyncMock()
    b.shutdown = AsyncMock()
    return b


def _reviewer(*verdicts: str) -> MagicMock:
    """Fake advisor reviewer: ``advise`` yields successive ``(verdict_text, usage)``."""
    c = MagicMock()
    c.advise = AsyncMock(side_effect=[(v, None) for v in verdicts])
    return c


def _failing_reviewer(exc: Exception) -> MagicMock:
    c = MagicMock()
    c.advise = AsyncMock(side_effect=exc)
    return c


def _config(**overrides) -> AdvisorConfig:
    base: dict = {
        "executor": {"model": "exec-model", "base_url": "http://exec.test", "api_key": "k",
                     "format": "anthropic"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


def _backend(config, executor_backend, advisor_caller) -> AdvisorLoopBackend:
    return AdvisorLoopBackend(
        config, executor_backend=executor_backend, advisor_caller=advisor_caller,
    )


def _request(**overrides) -> ChatRequest:
    body: dict = {"model": "incoming", "system": "sys",
                  "messages": [{"role": "user", "content": "build X"}]}
    body.update(overrides)
    return ChatRequest.anthropic(body)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------


async def test_tool_use_turn_passes_through_unreviewed() -> None:
    """A turn with a tool_use block means the executor is working — no review."""
    exec_b = _exec_backend(_completion_resp(text="reading", tool_use=True))
    adv = _reviewer()
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert resp.to_body()["stop_reason"] == "tool_use"
    assert exec_b.call.await_count == 1
    assert adv.advise.await_count == 0  # advisor never consulted


async def test_terminal_turn_approved_returns_as_is() -> None:
    """First no-tool-use turn is reviewed; APPROVE returns it unchanged."""
    exec_b = _exec_backend(_completion_resp(text="done, all good"))
    adv = _reviewer("APPROVE")
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())
    assert _anthropic_text(resp.to_body()) == "done, all good"
    assert exec_b.call.await_count == 1
    assert adv.advise.await_count == 1


async def test_review_records_advisor_cost_into_stats() -> None:
    """The advisor review's tokens are recorded into the accumulator so the run's
    own cost output (routing_stats) includes the advisor, not just the executor."""
    exec_b = _exec_backend(_completion_resp(text="done"))
    adv = MagicMock()
    adv.advise = AsyncMock(return_value=("APPROVE", {"input_tokens": 500, "output_tokens": 8}))
    stats = MagicMock()
    stats.record_planner_usage = AsyncMock()
    stats.record_success = AsyncMock()
    backend = AdvisorLoopBackend(
        _config(), stats_accumulator=stats, executor_backend=exec_b, advisor_caller=adv,
    )
    await backend.call(ProxyContext(), _request())
    stats.record_planner_usage.assert_awaited_once()
    kw = stats.record_planner_usage.await_args.kwargs
    assert kw["model"] == "adv-model"
    assert kw["prompt_tokens"] == 500
    assert kw["completion_tokens"] == 8


async def test_redo_reinvokes_executor_with_feedback() -> None:
    """REDO feeds the plan back and re-invokes the executor to keep working."""
    exec_b = _exec_backend(
        _completion_resp(text="I think I'm done"),            # terminal → review
        _completion_resp(text="continuing", tool_use=True),   # redo continuation (passthrough)
    )
    adv = _reviewer("REDO: you forgot the empty-input case; add a guard and test it")
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request())

    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 2  # original + redo re-invocation
    redo_request = exec_b.call.await_args_list[1].args[1]
    redo_msgs = redo_request.to_body()["messages"]
    assert any(m.get("role") == "assistant" and m.get("content") == "I think I'm done" for m in redo_msgs)
    assert any(m.get("role") == "user" and "empty-input case" in (m.get("content") or "") for m in redo_msgs)
    # the returned response is the continuation (has the real tool call)
    assert resp.to_body()["stop_reason"] == "tool_use"


async def test_review_is_once_per_session() -> None:
    """Two terminal turns in the same session → advisor consulted only once."""
    exec_b = _exec_backend(_completion_resp(text="done1"), _completion_resp(text="done2"))
    adv = _reviewer("APPROVE")  # only one verdict provided on purpose
    backend = _backend(_config(), exec_b, adv)
    await backend.call(ProxyContext(), _request())      # review #1
    await backend.call(ProxyContext(), _request())      # same prefix → no review
    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 2


async def test_reviewed_session_passes_through_verbatim() -> None:
    """After the gate fires, later turns are pure passthrough (no advisor, returned as-is)."""
    exec_b = _exec_backend(
        _completion_resp(text="plan"),                     # review fires here
        _completion_resp(text="more", tool_use=True),      # passthrough verbatim
    )
    adv = _reviewer("APPROVE")
    backend = _backend(_config(), exec_b, adv)
    await backend.call(ProxyContext(), _request())
    resp = await backend.call(ProxyContext(), _request())
    assert adv.advise.await_count == 1
    assert resp.to_body()["stop_reason"] == "tool_use"


async def test_fail_open_approves_on_review_error() -> None:
    exec_b = _exec_backend(_completion_resp(text="done"))
    adv = _failing_reviewer(RuntimeError("advisor down"))
    resp = await _backend(_config(fail_open=True), exec_b, adv).call(ProxyContext(), _request())
    assert _anthropic_text(resp.to_body()) == "done"
    assert exec_b.call.await_count == 1  # no redo


async def test_fail_closed_propagates_review_error() -> None:
    exec_b = _exec_backend(_completion_resp(text="done"))
    adv = _failing_reviewer(RuntimeError("advisor down"))
    with pytest.raises(RuntimeError, match="advisor down"):
        await _backend(_config(fail_open=False), exec_b, adv).call(ProxyContext(), _request())


async def test_streaming_terminal_approved_replays() -> None:
    """Streaming no-tool-use turn → review → APPROVE → replay (one generation)."""
    exec_b = _exec_backend(_stream_resp(text="done"))
    adv = _reviewer("APPROVE")
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request(stream=True))
    assert exec_b.call.await_count == 1
    assert adv.advise.await_count == 1
    assert response_type_matches(resp, ChatResponseType.ANTHROPIC_STREAM)
    events = [e async for e in resp.stream]
    assert any(isinstance(e, dict) and e.get("type") == "content_block_delta" for e in events)


async def test_streaming_tool_use_passes_through() -> None:
    """Streaming turn with a tool_use block → pass through, no review."""
    exec_b = _exec_backend(_stream_resp(tool_use=True))
    adv = _reviewer()
    resp = await _backend(_config(), exec_b, adv).call(ProxyContext(), _request(stream=True))
    assert adv.advise.await_count == 0
    assert response_type_matches(resp, ChatResponseType.ANTHROPIC_STREAM)


# ---------------------------------------------------------------------------
# OpenAI-wire gate (executor format: openai)
# ---------------------------------------------------------------------------


def _openai_config(**overrides) -> AdvisorConfig:
    base: dict = {
        "strategy": "review_gate",
        "executor": {"model": "qwen/qwen3-max", "base_url": "http://exec.test",
                     "api_key": "k", "format": "openai"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


def _openai_completion_resp(*, text=None, tool_calls=False) -> ChatResponse:
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [{"id": "b1", "type": "function",
                                  "function": {"name": "bash", "arguments": "{}"}}]
    body = {
        "id": "chatcmpl-x", "object": "chat.completion", "model": "exec-model",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    return ChatResponse.openai_completion(body)


def _openai_request(**overrides) -> ChatRequest:
    body: dict = {"model": "incoming",
                  "messages": [{"role": "system", "content": "sys"},
                               {"role": "user", "content": "build X"}]}
    body.update(overrides)
    return ChatRequest.openai_chat(body)  # type: ignore[arg-type]


def test_openai_gate_advertises_openai_wire() -> None:
    backend = _backend(_openai_config(), _exec_backend(), _reviewer())
    assert backend.supported_request_types == [ChatRequestType.OPENAI_CHAT]


async def test_openai_tool_call_turn_passes_through_unreviewed() -> None:
    exec_b = _exec_backend(_openai_completion_resp(tool_calls=True))
    adv = _reviewer()
    resp = await _backend(_openai_config(), exec_b, adv).call(ProxyContext(), _openai_request())
    assert resp.to_body()["choices"][0]["finish_reason"] == "tool_calls"
    assert adv.advise.await_count == 0


async def test_openai_terminal_turn_approved_returns_as_is() -> None:
    exec_b = _exec_backend(_openai_completion_resp(text="done, all good"))
    adv = _reviewer("APPROVE")
    resp = await _backend(_openai_config(), exec_b, adv).call(ProxyContext(), _openai_request())
    assert resp.to_body()["choices"][0]["message"]["content"] == "done, all good"
    assert adv.advise.await_count == 1


async def test_openai_redo_uses_configured_prefix_and_wire() -> None:
    exec_b = _exec_backend(
        _openai_completion_resp(text="I think I'm done"),
        _openai_completion_resp(text="continuing", tool_calls=True),
    )
    adv = _reviewer("REDO: verify the output file exists")
    config = _openai_config(redo_feedback_prefix="REVIEWER SAYS: ")
    resp = await _backend(config, exec_b, adv).call(ProxyContext(), _openai_request())

    assert exec_b.call.await_count == 2
    redo_request = exec_b.call.await_args_list[1].args[1]
    assert redo_request.request_type == ChatRequestType.OPENAI_CHAT
    redo_msgs = redo_request.to_body()["messages"]
    assert redo_msgs[-1]["role"] == "user"
    assert redo_msgs[-1]["content"].startswith("REVIEWER SAYS: verify the output file")
    assert redo_msgs[-2] == {"role": "assistant", "content": "I think I'm done"}
    assert resp.to_body()["choices"][0]["finish_reason"] == "tool_calls"


async def test_openai_streaming_terminal_approved_replays() -> None:
    events = [
        {"id": "c", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"},
                      "finish_reason": None}]},
        {"id": "c", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    exec_b = _exec_backend(
        ChatResponse.openai_stream(OpenAIResponseStream(_agen(list(events)))),
    )
    adv = _reviewer("APPROVE")
    resp = await _backend(_openai_config(), exec_b, adv).call(
        ProxyContext(), _openai_request(stream=True),
    )
    assert adv.advise.await_count == 1
    assert response_type_matches(resp, ChatResponseType.OPENAI_STREAM)
    replayed = [e async for e in resp.stream]
    assert replayed == events


# ---------------------------------------------------------------------------
# Pattern trigger (text-protocol harnesses)
# ---------------------------------------------------------------------------


def _pattern_config(**overrides) -> AdvisorConfig:
    base: dict = {
        "strategy": "review_gate",
        "gate_trigger": "pattern",
        "gate_trigger_pattern": r'task_complete["\s>:]*true',
        "executor": {"model": "nvidia/nemotron-3-ultra", "base_url": "http://exec.test",
                     "api_key": "k", "format": "openai"},
        "advisor": {"model": "adv-model", "base_url": "http://adv.test", "api_key": "k",
                    "format": "anthropic"},
    }
    base.update(overrides)
    return AdvisorConfig(**base)


async def test_pattern_trigger_ignores_non_matching_turns() -> None:
    """Ordinary command turns (no marker) pass through unreviewed, even with
    no tool calls — the whole point for text-protocol harnesses."""
    exec_b = _exec_backend(_openai_completion_resp(text='{"commands": ["ls -la"]}'))
    adv = _reviewer()
    resp = await _backend(_pattern_config(), exec_b, adv).call(ProxyContext(), _openai_request())
    assert adv.advise.await_count == 0
    assert resp.to_body()["choices"][0]["message"]["content"] == '{"commands": ["ls -la"]}'


async def test_pattern_trigger_gates_done_marker_and_redos() -> None:
    exec_b = _exec_backend(
        _openai_completion_resp(text='All checks pass. "task_complete": true'),
        _openai_completion_resp(text='{"commands": ["pytest tests/"]}'),
    )
    adv = _reviewer("REDO: the output file was never written; create it and re-verify")
    config = _pattern_config(redo_feedback_prefix="REVIEWER: not done. ")
    resp = await _backend(config, exec_b, adv).call(ProxyContext(), _openai_request())

    assert adv.advise.await_count == 1
    assert exec_b.call.await_count == 2
    redo_msgs = exec_b.call.await_args_list[1].args[1].to_body()["messages"]
    assert redo_msgs[-1]["content"].startswith("REVIEWER: not done. the output file")
    # The client receives the continuation turn, not the premature done-claim.
    assert "pytest" in resp.to_body()["choices"][0]["message"]["content"]


async def test_pattern_trigger_reviews_once_per_session() -> None:
    exec_b = _exec_backend(
        _openai_completion_resp(text='"task_complete": true'),
        _openai_completion_resp(text='<task_complete>true</task_complete>'),
    )
    adv = _reviewer("APPROVE")
    backend = _backend(_pattern_config(), exec_b, adv)
    await backend.call(ProxyContext(), _openai_request())   # gate fires
    await backend.call(ProxyContext(), _openai_request())   # passthrough
    assert adv.advise.await_count == 1


def test_pattern_trigger_requires_pattern() -> None:
    import pydantic
    with pytest.raises(pydantic.ValidationError, match="gate_trigger_pattern"):
        _pattern_config(gate_trigger_pattern="")


def test_invalid_pattern_rejected() -> None:
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        _pattern_config(gate_trigger_pattern="[unclosed")


# ---------------------------------------------------------------------------
# Pure helpers + advisor caller
# ---------------------------------------------------------------------------


def test_parse_verdict() -> None:
    assert _parse_verdict("APPROVE") == ("APPROVE", "")
    assert _parse_verdict("approve, looks complete")[0] == "APPROVE"
    v, plan = _parse_verdict("REDO: add a guard for empty input and re-run tests")
    assert v == "REDO" and "add a guard" in plan
    assert _parse_verdict("hmm not sure")[0] == "APPROVE"  # unclear → approve


def test_session_key_stable_across_turns() -> None:
    msgs = [{"role": "user", "content": "the task"}]
    later = msgs + [{"role": "assistant", "content": "..."}, {"role": "user", "content": "tool result"}]
    assert _session_key("sys", msgs) == _session_key("sys", later)
    assert _session_key("sys", msgs) != _session_key("sys", [{"role": "user", "content": "DIFFERENT"}])
    assert _session_key("sys", msgs) != _session_key("OTHER sys", msgs)  # system is part of the key


def test_build_advisor_caller_is_anthropic() -> None:
    assert isinstance(_build_advisor_caller(_config()), _AnthropicAdvisorCaller)


def test_build_advisor_caller_dispatches_openai() -> None:
    config = _config(advisor={"model": "deepseek/deepseek-r2", "base_url": "http://adv.test",
                              "api_key": "k", "format": "openai"})
    assert isinstance(_build_advisor_caller(config), _OpenAiAdvisorCaller)


@respx.mock
async def test_openai_advisor_hits_chat_completions_endpoint() -> None:
    route = respx.post("https://adv.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-x", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": " use a heap "}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 6},
        })
    )
    target = coerce_llm_target({
        "model": "deepseek/deepseek-r2", "base_url": "https://adv.example/v1",
        "api_key": "secret-key", "format": "openai",
        "extra_headers": {"X-Gateway": "test"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }, default_id="advisor")
    caller = _OpenAiAdvisorCaller(target=target, max_tokens=256, temperature=None)

    text, usage = await caller.advise(system="advise this", transcript="conversation")

    assert text == "use a heap"
    prompt_tokens, completion_tokens = _usage_tokens(usage)
    assert (prompt_tokens, completion_tokens) == (42, 6)
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.headers["x-gateway"] == "test"
    body = json.loads(request.content)
    assert body["messages"][0] == {"role": "system", "content": "advise this"}
    assert body["messages"][1] == {"role": "user", "content": "conversation"}
    assert body["max_tokens"] == 256
    assert "temperature" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@respx.mock
async def test_anthropic_reviewer_hits_messages_endpoint() -> None:
    route = respx.post("https://inference-api.nvidia.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "text", "text": "APPROVE"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        })
    )
    caller = _AnthropicAdvisorCaller(
        api_key="secret-key", base_url="https://inference-api.nvidia.com/v1",
        model="aws/anthropic/bedrock-claude-opus-4-8", max_tokens=256,
        temperature=None, timeout=5.0,
    )
    text, usage = await caller.advise(system="review this", transcript="conversation")
    assert text == "APPROVE"
    assert usage["input_tokens"] == 10
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer secret-key"
    body = json.loads(request.content)
    assert body["system"] == "review this"
    assert "temperature" not in body


def test_messages_url_resolution() -> None:
    base = "https://inference-api.nvidia.com"
    assert _messages_url(f"{base}/v1") == f"{base}/v1/messages"
    assert _messages_url(f"{base}/v1/messages") == f"{base}/v1/messages"
    assert _messages_url(base) == f"{base}/v1/messages"


def test_anthropic_text_joins_text_blocks() -> None:
    data = {"content": [{"type": "text", "text": "RE"}, {"type": "thinking", "text": "x"},
                        {"type": "text", "text": "DO: fix it"}]}
    assert _anthropic_text(data) == "REDO: fix it"

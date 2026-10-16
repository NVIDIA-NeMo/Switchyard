# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured routes use Python transport and native execution policy."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

import pytest

from switchyard.libsy import LibsyError
from switchyard.runner import Runner, UpstreamError, WireFormat
from switchyard_rust.server import ServerConfigError


def deployment(format: str = "openai_chat", *, classify: bool = True) -> str:
    algorithm = (
        "type = 'llm_classifier'\nclassifier_target = 'judge'\nweak_target = 'fast'\n"
        "strong_target = 'capable'\nbase_threshold = 0.5"
        if classify
        else "type = 'passthrough'\ntarget = 'fast'"
    )
    return f"""
schema_version = 1
[llm_clients]
primary = {{ format = '{format}' }}
other = {{ format = '{format}' }}
[targets]
judge = {{ id = 'judge', llm_client = 'primary' }}
fast = {{ id = 'fast', llm_client = 'primary', system_prompt = 'fast prompt' }}
[targets.capable]
id = 'capable'
llm_client = 'other'
system_prompt = 'capable prompt'
extra_body = {{ seed = 7 }}
omit_body_fields = ['temperature']
[routes.coding]
id = 'coding'
{algorithm}
"""


def answer(format: str, model: str, text: str = "host answer") -> dict[str, Any]:
    body: dict[str, Any] = {"id": f"resp_{model}", "model": model}
    if format == "openai_chat":
        body.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    elif format == "openai_responses":
        body.update(
            object="response",
            status="completed",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    else:
        body.update(
            type="message",
            role="assistant",
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )
    return body


def judge_answer(format: str, body: dict[str, Any]) -> dict[str, Any]:
    verdict = {
        "crux": "task",
        "primary_rule": "SUP-1",
        "capability_boundary": "supported",
        "p_solve": 0.1 if "hard follow-up" in json.dumps(body) else 0.9,
    }
    return answer(format, "judge", json.dumps(verdict))


def chat(text: str = "hello") -> dict[str, Any]:
    return {"model": "coding", "messages": [{"role": "user", "content": text}]}


def chat_frame(delta: dict[str, Any], *, stop: bool = False) -> bytes:
    body = {
        "id": "chatcmpl-live",
        "object": "chat.completion.chunk",
        "model": "provider-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if stop else None}],
    }
    if stop:
        body["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


async def test_python_clients_keep_session_and_candidate_policy() -> None:
    """A retained choice still applies each fallback's own prompt and body settings."""
    calls = []

    async def primary(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        if body["model"] == "judge":
            return judge_answer("openai_chat", body)
        raise UpstreamError(503, {"error": {"message": "unavailable"}})

    async def capable(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return answer("openai_chat", body["model"])

    source = deployment() + "\nclassify_trigger = 'new_session'"
    source = source.replace(
        "extra_body = { seed = 7 }", "reasoning_effort = 'low'\nextra_body = { seed = 7 }"
    )
    runner = Runner.from_toml(source, clients={"primary": primary, "other": capable})
    for text in ["hello", "hard follow-up"]:
        result = await runner.execute(
            {**chat(text), "temperature": 0.8}, headers={"x-switchyard-session-id": "task"}
        )
        assert result["model"] == "capable"
    assert [body["model"] for body in calls] == ["judge", "fast", "capable", "fast", "capable"]
    assert "fast prompt" not in json.dumps(calls[0])
    assert calls[1]["messages"][0]["content"] == "fast prompt"
    assert calls[2]["messages"][0]["content"] == "capable prompt"
    assert len(calls[2]["messages"]) == 2
    assert calls[2]["seed"] == 7 and calls[2]["reasoning_effort"] == "low"
    assert "temperature" not in calls[2]


@pytest.mark.parametrize("format", ["openai_chat", "openai_responses"])
async def test_python_clients_keep_responses_continuation_state(format: str) -> None:
    """Continuation IDs pin native providers or restore history for a Chat target."""
    calls = []

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return judge_answer(format, body) if body["model"] == "judge" else answer(format, "fast")

    async def unexpected(body: dict[str, Any]) -> dict[str, Any]:
        pytest.fail(f"continuation switched providers: {body['model']}")

    runner = Runner.from_toml(
        deployment(format), clients={"primary": provider, "other": unexpected}
    )
    seed = await runner.execute({"model": "coding", "input": "hello"}, format="openai_responses")
    result = await runner.execute(
        {"model": "coding", "previous_response_id": seed["id"], "input": "hard follow-up"},
        format="openai_responses",
    )
    assert result["model"] == "fast"
    assert [body["model"] for body in calls] == ["judge", "fast", "fast"]
    if format == "openai_chat":
        assert [message["content"] for message in calls[2]["messages"][1:]] == [
            "hello", "host answer", "hard follow-up",
        ]
    else:
        assert calls[2]["previous_response_id"] == seed["id"]


async def test_python_context_and_cancellation_are_owned_by_each_request() -> None:
    """Concurrent callbacks retain task context; cancellation waits for their cleanup."""
    identity: ContextVar[str] = ContextVar("identity")
    seen = []

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0)
        seen.append(identity.get())
        return answer("openai_chat", body["model"])

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )

    async def run(name: str) -> None:
        token = identity.set(name)
        try:
            await runner.execute(chat(name), headers={"x-switchyard-session-id": name})
        finally:
            identity.reset(token)

    await asyncio.gather(run("a"), run("b"))
    assert sorted(seen) == ["a", "b"]
    started, cleaning, finish, cleaned = (asyncio.Event() for _ in range(4))

    async def pending(body: dict[str, Any]) -> dict[str, Any]:
        try:
            started.set()
            await asyncio.Future()
            raise AssertionError("pending provider returned")
        finally:
            cleaning.set()
            await finish.wait()
            cleaned.set()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": pending, "other": pending}
    )
    task = asyncio.create_task(runner.execute(chat()))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 2)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert cleaned.is_set()


@pytest.mark.parametrize("format", list(WireFormat))
@pytest.mark.parametrize("upstream", list(WireFormat))
async def test_wire_formats_and_unsupported_configuration(
    format: WireFormat, upstream: WireFormat,
) -> None:
    """Caller and target formats are independent; callbacks see prepared provider JSON."""
    calls = []

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return answer(upstream, body["model"])

    source = deployment(upstream.value, classify=False)
    clients = {"primary": provider, "other": provider}
    runner = Runner.from_toml(source, clients=clients)
    body = {"model": "coding", "input": "hello"} if format == "openai_responses" else chat()
    if format == "anthropic_messages":
        body["max_tokens"] = 16
    result = await runner.execute(body, format=format)
    assert result["model"] == "fast" and calls[0]["model"] == "fast"
    assert ("input" if upstream == "openai_responses" else "messages") in calls[0]
    assert "host answer" in json.dumps(result)
    with pytest.raises(ServerConfigError, match="max_retries"):
        Runner.from_toml(
            source.replace("primary = {", "primary = { max_retries = 2,"), clients=clients
        )
    with pytest.raises(ServerConfigError, match="other"):
        Runner.from_toml(source, clients={"primary": provider})
    assert len(calls) == 1


async def test_buffered_provider_failure_uses_fallback_but_sdk_error_is_terminal() -> None:
    """Failed Responses bodies permit fallback; unrelated host exceptions do not."""
    calls = []
    is_sdk_error = False

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        model = body["model"]
        calls.append(model)
        if model == "judge":
            return judge_answer("openai_responses", body)
        if is_sdk_error:
            raise RuntimeError("host SDK failed")
        response = answer("openai_responses", model)
        if model == "fast":
            response.update(status="failed", error={"code": "server_error", "message": "unavailable"})
        return response

    runner = Runner.from_toml(
        deployment("openai_responses"), clients={"primary": provider, "other": provider}
    )
    assert (await runner.execute(chat()))["model"] == "capable"
    assert calls == ["judge", "fast", "capable"]
    calls.clear()
    is_sdk_error = True
    with pytest.raises(LibsyError, match="host SDK failed"):
        await runner.execute(chat())
    assert calls == ["judge", "fast"]


@pytest.mark.parametrize("format", list(WireFormat))
async def test_fragmented_sse_translates_to_each_caller_format(format: WireFormat) -> None:
    """Raw bytes may split UTF-8; caller events retain text, usage and a real terminal."""
    cleaned = asyncio.Event()
    payload = (
        chat_frame({"role": "assistant", "content": "café"})
        + chat_frame({}, stop=True)
        + b"data: [DONE]\n\n"
    )
    split = payload.index("é".encode()) + 1

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield payload[:split]
            yield payload[split : split + 1]
            yield payload[split + 1 :]
        finally:
            cleaned.set()

    async def provider(body: dict[str, Any]) -> AsyncIterator[bytes]:
        assert body["stream"] is True and body["model"] == "fast"
        return chunks()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )
    body = {"model": "coding", "input": "hello"} if format == "openai_responses" else chat()
    if format == "anthropic_messages":
        body["max_tokens"] = 16
    stream = await runner.execute(
        {**body, "stream": True}, format=format
    )
    assert not cleaned.is_set()
    async with stream:
        events = [event async for event in stream]
    assert cleaned.is_set()
    if format == "openai_chat":
        deltas = (
            event["choices"][0]["delta"].get("content", "") for event in events if event["choices"]
        )
        assert any(
            event["choices"] and event["choices"][0]["finish_reason"] == "stop" for event in events
        )
        usage = next(event["usage"] for event in events if event.get("usage"))
        assert all(event["model"] == "fast" for event in events)
    elif format == "openai_responses":
        deltas = (
            event["delta"] for event in events if event["type"] == "response.output_text.delta"
        )
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["model"] == "fast"
        usage = events[-1]["response"]["usage"]
    else:
        deltas = (
            event["delta"].get("text", "")
            for event in events
            if event["type"] == "content_block_delta"
        )
        assert events[-1]["type"] == "message_stop"
        assert events[0]["message"]["model"] == "fast"
        usage = next(event["usage"] for event in events if event["type"] == "message_delta")
    assert "".join(deltas) == "café"
    counts = (
        ("prompt_tokens", "completion_tokens")
        if format == "openai_chat"
        else ("input_tokens", "output_tokens")
    )
    assert [usage[key] for key in counts] == [2, 3]


async def test_unterminated_provider_stream_is_not_a_successful_response() -> None:
    """A valid first frame does not make truncated upstream output complete."""

    async def chunks() -> AsyncIterator[bytes]:
        yield chat_frame({"role": "assistant", "content": "partial"})

    async def provider(body: dict[str, Any]) -> AsyncIterator[bytes]:
        return chunks()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )
    stream = await runner.execute({**chat(), "stream": True})
    async with stream:
        with pytest.raises(LibsyError):
            _ = [event async for event in stream]


@pytest.mark.parametrize("early_close", [False, True])
async def test_provider_iterator_keeps_one_task_and_context_until_close(early_close: bool) -> None:
    """Provider context managers must enter, resume and exit in their owning task."""
    identity: ContextVar[str] = ContextVar("provider_identity", default="caller")
    cleaned = asyncio.Event()

    async def chunks() -> AsyncIterator[bytes]:
        owner = asyncio.current_task()
        token = identity.set("provider")
        try:
            yield chat_frame({"role": "assistant", "content": "hello"})
            assert asyncio.current_task() is owner and identity.get() == "provider"
            yield chat_frame({}, stop=True) + b"data: [DONE]\n\n"
        finally:
            assert asyncio.current_task() is owner and identity.get() == "provider"
            identity.reset(token)
            cleaned.set()

    async def provider(body: dict[str, Any]) -> AsyncIterator[bytes]:
        return chunks()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )
    stream = await runner.execute({**chat(), "stream": True})
    async with stream:
        if not early_close:
            assert [event async for event in stream]
    assert cleaned.is_set() and identity.get() == "caller"


async def test_provider_timeout_remains_active_between_stream_chunks() -> None:
    """A provider's timeout must still cancel its task after yielding the first chunk."""
    timeout = getattr(asyncio, "timeout", None)
    if timeout is None:
        pytest.skip("asyncio.timeout requires Python 3.11")
    expired = asyncio.Event()

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async with timeout(None) as deadline:
                yield chat_frame({"role": "assistant", "content": "hello"})
                deadline.reschedule(asyncio.get_running_loop().time())
                await asyncio.sleep(0.01)
                yield chat_frame({}, stop=True) + b"data: [DONE]\n\n"
        except TimeoutError:
            expired.set()
            raise

    async def provider(body: dict[str, Any]) -> AsyncIterator[bytes]:
        return chunks()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )
    stream = await runner.execute({**chat(), "stream": True})
    async with stream:
        with pytest.raises(LibsyError, match="TimeoutError"):
            _ = [event async for event in stream]
    assert expired.is_set()


@pytest.mark.parametrize("close_mode", ["idle", "pending", "cancel"])
async def test_stream_close_waits_for_provider_cleanup_in_original_context(
    close_mode: str,
) -> None:
    """Streams own their provider until EOF, close or cancellation, even after execute returns."""
    identity: ContextVar[str] = ContextVar("stream_identity")
    started, cleaning, finish, cleaned = (asyncio.Event() for _ in range(4))
    seen = []

    async def chunks() -> AsyncIterator[bytes]:
        try:
            seen.append(identity.get())
            yield chat_frame({"role": "assistant"})
            started.set()
            await asyncio.Future()
        finally:
            cleaning.set()
            await finish.wait()
            seen.append(identity.get())
            cleaned.set()

    async def provider(body: dict[str, Any]) -> AsyncIterator[bytes]:
        return chunks()

    runner = Runner.from_toml(
        deployment(classify=False), clients={"primary": provider, "other": provider}
    )
    token = identity.set("request")
    try:
        stream = await runner.execute({**chat(), "stream": True})
    finally:
        identity.reset(token)
    assert not cleaned.is_set()
    read = None
    if close_mode != "idle":
        await anext(stream)
        read = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), 2)
    if close_mode == "cancel":
        assert read is not None
        read.cancel()
        closing = read
    else:
        closing = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(cleaning.wait(), 2)
    assert not closing.done()
    finish.set()
    if close_mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 2)
    else:
        await asyncio.wait_for(closing, 2)
        if read is not None:
            await asyncio.gather(read, return_exceptions=True)
    await stream.aclose()
    assert cleaned.is_set() and seen == ["request", "request"]


async def test_lazy_stream_http_failure_uses_fallback() -> None:
    """An HTTP failure before the first event keeps native fallback behavior."""
    calls = []

    async def chunks(model: str) -> AsyncIterator[bytes]:
        if model == "fast":
            raise UpstreamError(503, {"error": {"message": "unavailable"}})
        yield chat_frame({"role": "assistant", "content": "fallback answer"})
        yield chat_frame({}, stop=True) + b"data: [DONE]\n\n"

    async def provider(body: dict[str, Any]) -> Any:
        model = body["model"]
        calls.append(model)
        return judge_answer("openai_chat", body) if model == "judge" else chunks(model)

    runner = Runner.from_toml(deployment(), clients={"primary": provider, "other": provider})
    stream = await runner.execute({**chat(), "stream": True})
    async with stream:
        events = [event async for event in stream]
    assert events and all(event["model"] == "capable" for event in events)
    assert calls == ["judge", "fast", "capable"]


async def test_inband_error_keeps_provider_fields_without_success_or_fallback() -> None:
    """After text reaches the caller, a provider error is terminal, not another answer attempt."""
    calls = []
    error = {
        "error": {
            "message": "provider detail",
            "type": "server_error",
            "request_id": "upstream-123",
        }
    }

    async def chunks() -> AsyncIterator[bytes]:
        yield chat_frame({"role": "assistant", "content": "partial"})
        yield f"data: {json.dumps(error)}\n\n".encode()

    async def provider(body: dict[str, Any]) -> Any:
        calls.append(body["model"])
        if body["model"] == "judge":
            return judge_answer("openai_chat", body)
        return chunks()

    runner = Runner.from_toml(deployment(), clients={"primary": provider, "other": provider})
    stream = await runner.execute({**chat(), "stream": True})
    async with stream:
        events = [event async for event in stream]
    assert events[-1]["error"] == error["error"]
    assert not any(
        event.get("choices") and event["choices"][0]["finish_reason"] for event in events
    )
    assert calls == ["judge", "fast"]


async def test_escalation_releases_stream_transport_before_judge_call() -> None:
    """A decoded terminal frame must release a one-connection pool before the judge runs."""
    pool = asyncio.Semaphore(1)
    released = asyncio.Event()
    calls = []

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield chat_frame({"role": "assistant", "content": "efficient answer"})
            yield chat_frame({}, stop=True)
            yield b"data: [DONE]\n\n"
        finally:
            pool.release()
            released.set()

    async def provider(body: dict[str, Any]) -> Any:
        await pool.acquire()
        calls.append(body["model"])
        if body["model"] == "judge":
            try:
                assert released.is_set()
                return answer(
                    "openai_chat",
                    "judge",
                    '{"escalate":false,"category":"none","new_evidence":false,"reason":"progress"}',
                )
            finally:
                pool.release()
        return chunks()

    source = deployment() + "\nescalation = { confirmations = 1 }"
    runner = Runner.from_toml(source, clients={"primary": provider, "other": provider})
    stream = await asyncio.wait_for(runner.execute({**chat(), "stream": True}), 2)
    async with stream:
        events = [event async for event in stream]
    assert events and calls == ["fast", "judge"]
    assert released.is_set() and not pool.locked()


async def test_advisor_review_budget_keeps_host_session_headers() -> None:
    """Raw benchmark session headers scope reviews even without a normalized session ID."""
    calls = []

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body["model"])
        return answer("openai_chat", body["model"], "APPROVE")

    source = deployment().split("[routes.coding]")[0] + """
[routes.coding]
id = 'coding'
type = 'advisor'
executor_target = 'fast'
advisor_target = 'judge'
max_reviews = 1
"""
    runner = Runner.from_toml(source, clients={"primary": provider, "other": provider})
    for session in ["a", "a", "b"]:
        await runner.execute(chat(), headers={"proxy_x_session_id": session})
    assert calls == ["fast", "judge", "fast", "fast", "judge"]


async def test_ingress_strips_reserved_preservation_and_keeps_user_metadata() -> None:
    """Caller metadata cannot replace the request with a preserved foreign-format body."""
    calls = []

    async def provider(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return answer("openai_responses", body["model"])

    source = deployment("openai_responses", classify=False).replace(
        ", system_prompt = 'fast prompt'", ""
    )
    runner = Runner.from_toml(source, clients={"primary": provider, "other": provider})
    await runner.execute({
        **chat(),
        "metadata": {
            "customer": "kept",
            "_switchyard_translation": {
                "requests": {"openai_responses": {"model": "forged", "input": "injected"}}
            },
        },
    })
    assert "hello" in json.dumps(calls[0]["input"])
    assert "injected" not in json.dumps(calls[0])
    assert calls[0]["metadata"] == {"customer": "kept"}

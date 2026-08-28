# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `craft_taskgen.llm_judge`.

Mocks `litellm.acompletion` to exercise:
- Happy path: valid JSON matching schema on first call.
- Parse-error retry: malformed JSON on first call, valid on retry with
  system-message feedback.
- Schema-error retry: parses but fails schema on first call, valid on retry.
- Transient-error retry: `litellm.Timeout` on first call, succeeds on second.
- Max transient retries exhausted: always raises Timeout.
- Max parse retries exhausted (one retry budget): both responses malformed
  → RuntimeError.
- Model normalization: bare model string gets `openai/` prefix.
- Gateway guardrail: missing creds raises RuntimeError.

Uses `asyncio.run()` to avoid a pytest-asyncio dependency (matches the pattern
used elsewhere in this repo's tests).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from craft_taskgen import llm_judge

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "bad"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def _fake_response(content: str, *, input_tokens: int = 100, output_tokens: int = 20) -> SimpleNamespace:
    """Build an object shaped like litellm's completion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
    return SimpleNamespace(choices=[choice], usage=usage)


class _CallRecorder:
    """Scripts a sequence of acompletion responses or exceptions."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Happy path + simple variants
# ---------------------------------------------------------------------------


def test_happy_path_returns_parsed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder([_fake_response('{"verdict":"ok","reason":"clean"}')])
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
        )
    )

    assert out.result == {"verdict": "ok", "reason": "clean"}
    assert out.usage == {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 0}
    assert out.model == "openai/aws/anthropic/bedrock-claude-opus-4-6"
    assert out.latency_s >= 0
    assert len(recorder.calls) == 1


def test_strips_markdown_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = '```json\n{"verdict":"ok","reason":"fenced"}\n```'
    recorder = _CallRecorder([_fake_response(fenced)])
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
        )
    )
    assert out.result == {"verdict": "ok", "reason": "fenced"}


def test_openai_prefix_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder([_fake_response('{"verdict":"ok","reason":"x"}')])
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="openai/us/azure/openai/gpt-5.4",
        )
    )
    assert out.model == "openai/us/azure/openai/gpt-5.4"


# ---------------------------------------------------------------------------
# Parse / schema retry paths
# ---------------------------------------------------------------------------


def test_parse_error_triggers_one_retry_with_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder(
        [
            _fake_response("not json at all"),
            _fake_response('{"verdict":"ok","reason":"recovered"}'),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
        )
    )

    assert out.result["reason"] == "recovered"
    # Retry injected the error feedback into the system message.
    second_messages = recorder.calls[1]["messages"]
    system_messages = [m for m in second_messages if m["role"] == "system"]
    assert any("previous response" in m["content"].lower() for m in system_messages)


def test_parse_retry_preserves_system_prompt_in_single_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Parse-retry must not emit two back-to-back role="system" messages
    # (gateway behavior for that is inconsistent). The build step is the
    # only caller that passes system_prompt=, so its recovery path would
    # be the one broken if this regressed.
    recorder = _CallRecorder(
        [
            _fake_response("not json at all"),
            _fake_response('{"verdict":"ok","reason":"recovered"}'),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
            system_prompt="You are a careful judge. Follow the rubric.",
        )
    )

    second_messages = recorder.calls[1]["messages"]
    system_messages = [m for m in second_messages if m["role"] == "system"]
    assert len(system_messages) == 1
    merged = system_messages[0]["content"]
    assert "careful judge" in merged
    assert "previous response" in merged.lower()


def test_schema_error_triggers_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder(
        [
            _fake_response('{"verdict":"ok"}'),
            _fake_response('{"verdict":"bad","reason":"violates rubric"}'),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
        )
    )
    assert out.result == {"verdict": "bad", "reason": "violates rubric"}


def test_parse_error_after_retry_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder(
        [
            _fake_response("garbage one"),
            _fake_response("garbage two"),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    with pytest.raises(RuntimeError, match="failed parse/schema validation"):
        _run(
            llm_judge.judge(
                prompt="decide",
                schema=_SCHEMA,
                model="aws/anthropic/bedrock-claude-opus-4-6",
            )
        )


# ---------------------------------------------------------------------------
# Transient-error retry path
# ---------------------------------------------------------------------------


def test_transient_error_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_judge.asyncio, "sleep", _no_sleep)
    recorder = _CallRecorder(
        [
            litellm.Timeout(message="boom", model="x", llm_provider="y"),
            _fake_response('{"verdict":"ok","reason":"after timeout"}'),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    out = _run(
        llm_judge.judge(
            prompt="decide",
            schema=_SCHEMA,
            model="aws/anthropic/bedrock-claude-opus-4-6",
        )
    )
    assert out.result["reason"] == "after timeout"
    assert len(recorder.calls) == 2


def test_transient_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_judge.asyncio, "sleep", _no_sleep)
    recorder = _CallRecorder([litellm.Timeout(message="persistent", model="x", llm_provider="y")] * 10)
    monkeypatch.setattr(litellm, "acompletion", recorder)

    with pytest.raises(litellm.Timeout):
        _run(
            llm_judge.judge(
                prompt="decide",
                schema=_SCHEMA,
                model="aws/anthropic/bedrock-claude-opus-4-6",
            )
        )
    assert len(recorder.calls) == llm_judge._MAX_TRANSIENT_RETRIES + 1


def test_non_transient_error_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder(
        [
            litellm.AuthenticationError(
                message="bad key",
                model="x",
                llm_provider="y",
            )
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", recorder)

    with pytest.raises(litellm.AuthenticationError):
        _run(
            llm_judge.judge(
                prompt="decide",
                schema=_SCHEMA,
                model="aws/anthropic/bedrock-claude-opus-4-6",
            )
        )
    assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# Gateway guardrail
# ---------------------------------------------------------------------------


def test_missing_creds_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _run(
            llm_judge.judge(
                prompt="decide",
                schema=_SCHEMA,
                model="aws/anthropic/bedrock-claude-opus-4-6",
            )
        )

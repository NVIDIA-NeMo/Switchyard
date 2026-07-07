# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from switchyard.lib.backends import (
    backend_format_resolver as resolver_mod,
)
from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)


class _RecordingProbe:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> bool:
        self.calls.append(dict(kwargs))
        return self.result


def _no_probe(name: str):
    def fail(**_: object) -> bool:
        pytest.fail(f"explicit backend formats must not probe ({name})")
    return fail


# ---------------------------------------------------------------------------
# Explicit format — no probing at all
# ---------------------------------------------------------------------------


def test_explicit_format_does_not_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _no_probe("chat-completions"))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(model="m", format=BackendFormat.ANTHROPIC),
    )

    assert resolution.format is BackendFormat.ANTHROPIC
    assert resolution.reason == "backend format is explicitly configured"


# ---------------------------------------------------------------------------
# Model-prefix fast-path — skips all probes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-haiku",
    "claude-3-opus-20240229",
    "claude-sonnet-4-6",
])
def test_auto_model_prefix_anthropic_skips_probes(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """anthropic/ and claude prefixes → ANTHROPIC without probing."""
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _no_probe("chat-completions"))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model=model,
            format=BackendFormat.AUTO,
            base_url="https://provider.test/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.ANTHROPIC
    assert "prefix" in resolution.reason


@pytest.mark.parametrize("model", [
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "models/gemini-2.5-pro",
])
def test_auto_model_prefix_gemini_skips_probes(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Bare gemini and models/gemini IDs → GEMINI without probing."""
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _no_probe("chat-completions"))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model=model,
            format=BackendFormat.AUTO,
            base_url="https://generativelanguage.googleapis.com",
            api_key="AIza-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.GEMINI
    assert "prefix" in resolution.reason


@pytest.mark.parametrize("model", [
    "google/gemini-2.5-flash",
    "openrouter/google/gemini-2.5-pro",
])
def test_auto_gateway_gemini_model_probes_chat_completions_first(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Gateway-namespaced Gemini models are NOT fast-pathed; probing runs."""
    chat_probe = _RecordingProbe(result=True)
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync", chat_probe)
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model=model,
            format=BackendFormat.AUTO,
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.OPENAI
    assert len(chat_probe.calls) == 1


@pytest.mark.parametrize("model", [
    "openrouter/anthropic/claude-3-5-sonnet",
    "aws/anthropic/bedrock-claude-opus-4-7",
])
def test_auto_gateway_anthropic_model_probes_chat_completions_first(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Gateway-namespaced models are NOT fast-pathed; Chat Completions probe runs first."""
    chat_probe = _RecordingProbe(result=True)
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync", chat_probe)
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model=model,
            format=BackendFormat.AUTO,
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.OPENAI
    assert len(chat_probe.calls) == 1


# ---------------------------------------------------------------------------
# AUTO probe order: Chat Completions → Anthropic → Responses
# ---------------------------------------------------------------------------


def test_auto_chat_completions_wins_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat Completions probe succeeds → OPENAI; remaining probes are never called."""
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _RecordingProbe(result=True))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _no_probe("anthropic"))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="nvidia/nvidia/nemotron-nano-9b-v2",
            format=BackendFormat.AUTO,
            base_url="https://inference-api.nvidia.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.OPENAI


def test_auto_resolves_to_anthropic_when_chat_completions_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat Completions probe fails, Anthropic probe succeeds → ANTHROPIC."""
    probe = _RecordingProbe(result=True)
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _RecordingProbe(result=False))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync", probe)
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        _no_probe("responses"))

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="some-model",
            format=BackendFormat.AUTO,
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.ANTHROPIC
    assert probe.calls == [{
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "sk-test",
        "model": "some-model",
        "timeout_s": 3.0,
    }]


def test_auto_resolves_to_responses_when_only_responses_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat Completions and Anthropic probes fail, Responses probe succeeds → RESPONSES."""
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _RecordingProbe(result=False))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _RecordingProbe(result=False))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        lambda **_: True)

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="some-model",
            format=BackendFormat.AUTO,
            base_url="https://provider.test/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.RESPONSES


def test_auto_falls_back_to_openai_when_all_probes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three probes fail → OPENAI (Chat Completions assumed universal)."""
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync",
                        _RecordingProbe(result=False))
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync",
                        _RecordingProbe(result=False))
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync",
                        lambda **_: False)

    resolution = resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="some-model",
            format=BackendFormat.AUTO,
            base_url="https://provider.test/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert resolution.format is BackendFormat.OPENAI


# ---------------------------------------------------------------------------
# Missing inputs
# ---------------------------------------------------------------------------


def test_auto_format_requires_base_url() -> None:
    with pytest.raises(ValueError, match="requires base_url"):
        resolver_mod.BackendFormatResolver.resolve(
            LlmTarget(
                model="m",
                format=BackendFormat.AUTO,
                api_key="sk-test",  # pragma: allowlist secret
            ),
        )


def test_auto_format_requires_api_key() -> None:
    with pytest.raises(ValueError, match="requires api_key"):
        resolver_mod.BackendFormatResolver.resolve(
            LlmTarget(
                model="m",
                format=BackendFormat.AUTO,
                base_url="https://provider.test/v1",
            ),
        )


# ---------------------------------------------------------------------------
# Model + timeout forwarding
# ---------------------------------------------------------------------------


def test_auto_forwards_endpoint_timeout_to_all_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier.endpoint.timeout_secs is forwarded as timeout_s to all three probes."""
    chat_probe = _RecordingProbe(result=False)
    anthropic_probe = _RecordingProbe(result=False)
    responses_probe = _RecordingProbe(result=False)
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync", chat_probe)
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync", anthropic_probe)
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync", responses_probe)

    resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="nvidia/nvidia/nemotron-nano",
            format=BackendFormat.AUTO,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="sk-test",  # pragma: allowlist secret
            timeout_secs=30.0,
        ),
    )

    assert chat_probe.calls[0]["timeout_s"] == 30.0
    assert anthropic_probe.calls[0]["timeout_s"] == 30.0
    assert responses_probe.calls[0]["timeout_s"] == 30.0


def test_auto_passes_model_to_all_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model is forwarded to all three probes so mixed providers are correctly scoped."""
    chat_probe = _RecordingProbe(result=False)
    anthropic_probe = _RecordingProbe(result=False)
    responses_probe = _RecordingProbe(result=False)
    monkeypatch.setattr(resolver_mod, "probe_openai_chat_completions_support_sync", chat_probe)
    monkeypatch.setattr(resolver_mod, "probe_anthropic_messages_support_sync", anthropic_probe)
    monkeypatch.setattr(resolver_mod, "probe_openai_responses_support_sync", responses_probe)

    resolver_mod.BackendFormatResolver.resolve(
        LlmTarget(
            model="openrouter/non-claude-model",
            format=BackendFormat.AUTO,
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",  # pragma: allowlist secret
        ),
    )

    assert chat_probe.calls[0]["model"] == "openrouter/non-claude-model"
    assert anthropic_probe.calls[0]["model"] == "openrouter/non-claude-model"
    assert responses_probe.calls[0]["model"] == "openrouter/non-claude-model"


# ---------------------------------------------------------------------------
# _interpret_status — model-error body detection
# ---------------------------------------------------------------------------


def test_interpret_status_400_with_model_not_found_body_returns_false() -> None:
    """400 whose body names the model as not found is a probe failure."""
    import json as _json

    body = _json.dumps({
        "error": {"type": "invalid_request_error", "message": "model: gpt-4o not found"},
    }).encode()
    assert resolver_mod._interpret_status(400, body) is False


def test_interpret_status_400_with_not_found_error_type_returns_false() -> None:
    """400/404-style 'not_found_error' type is treated as a probe failure."""
    import json as _json

    body = _json.dumps({"error": {"type": "not_found_error", "message": "model not found"}}).encode()
    assert resolver_mod._interpret_status(400, body) is False


def test_interpret_status_400_with_field_error_returns_true() -> None:
    """400 about missing fields (not the model) means route exists — True."""
    import json as _json

    body = _json.dumps({
        "error": {"type": "invalid_request_error", "message": "messages: field required"},
    }).encode()
    assert resolver_mod._interpret_status(400, body) is True


def test_interpret_status_400_without_body_returns_true() -> None:
    """400 with no body (legacy call-site) preserves old behaviour — True."""
    assert resolver_mod._interpret_status(400) is True

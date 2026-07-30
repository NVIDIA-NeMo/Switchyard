# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profile-backed deterministic routing construction."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.llm_classifier import (
    LLMClassifierRequestProcessor,
    SignalTierSelectorRequestProcessor,
)
from switchyard.lib.processors.stats_request_processor import StatsRequestProcessor
from switchyard.lib.processors.stats_response_processor_accumulator import (
    StatsResponseProcessor,
)
from switchyard.lib.profiles import (
    DeterministicRoutingConfig,
    DeterministicRoutingPresets,
    DeterministicRoutingProfileConfig,
    ProfileSwitchyard,
)
from switchyard.lib.profiles.tier_target_builders import (
    apply_deepseek_overrides,
    apply_default_tier_timeout,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.route_table_builders import deterministic_routing_virtual_model_id
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardContextWindowExceededError,
)
from switchyard_rust.profiles import ProfileInput
from switchyard_rust.translation import TranslationEngine


def _config(
    *,
    profile_name: str = "coding_agent",
    enable_stats: bool = True,
    classifier_min_confidence: float = 0.0,
    classifier_model: str = "nvidia/deepseek-ai/deepseek-v4-flash",
    classifier_system_prompt: str | None = None,
    classifier_max_request_chars: int = 16_000,
    session_affinity: bool = False,
    affinity_max_sessions: int = 10_000,
    affinity_warmup_turns: int = 0,
) -> DeterministicRoutingConfig:
    return DeterministicRoutingConfig(
        strong=LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        weak=LlmTarget(
            id="weak",
            model="nvidia/deepseek-ai/evals-deepseek-v4-pro",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        classifier=LlmTarget(
            id="classifier",
            model=classifier_model,
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://example.invalid/v1",
        ),
        profile_name=profile_name,  # type: ignore[arg-type]
        classifier_min_confidence=classifier_min_confidence,
        classifier_system_prompt=classifier_system_prompt,
        classifier_max_request_chars=classifier_max_request_chars,
        enable_stats=enable_stats,
        fallback_target_on_evict="strong",
        session_affinity=session_affinity,
        affinity_max_sessions=affinity_max_sessions,
        affinity_warmup_turns=affinity_warmup_turns,
    )


def _deterministic_routing_switchyard(
    config: DeterministicRoutingConfig,
    *,
    stats_accumulator: StatsAccumulator | None = None,
    pre_routing_request_processors: list[Any] | None = None,
    extra_request_processors: list[Any] | None = None,
    extra_response_processors: list[Any] | None = None,
) -> ProfileSwitchyard:
    """Build the profile-backed runtime used by these tests."""
    return ProfileSwitchyard(
        DeterministicRoutingProfileConfig.from_config(config)
        .build()
        .with_runtime_components(
            stats_accumulator=stats_accumulator,
            enable_stats=config.enable_stats,
            pre_request_processors=pre_routing_request_processors or (),
            post_request_processors=extra_request_processors or (),
            response_processors=extra_response_processors or (),
        )
    )


class _NoopRequestProcessor:
    async def process(self, _ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        return request


class _NoopResponseProcessor:
    async def process(self, _ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        return response


class TestProfileStructure:
    def test_returns_profile_backed_switchyard_adapter(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        assert isinstance(switchyard, ProfileSwitchyard)

    def test_backend_is_deterministic_routing(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        backends = [
            c for c in switchyard.iter_components()
            if isinstance(c, DeterministicRoutingLLMBackend)
        ]
        assert len(backends) == 1

    def test_classifier_processor_present(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        classifiers = [
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        ]
        assert len(classifiers) == 1

    def test_classifier_processor_uses_prompt_and_context_overrides(self) -> None:
        switchyard = _deterministic_routing_switchyard(
            _config(
                classifier_system_prompt="custom classifier prompt",
                classifier_max_request_chars=1024,
            ),
        )
        classifier = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        assert classifier._config.system_prompt == "custom classifier prompt"
        assert classifier._config.max_request_chars == 1024

    def test_tier_selector_processor_present(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        selectors = [
            c for c in switchyard.iter_components()
            if isinstance(c, SignalTierSelectorRequestProcessor)
        ]
        assert len(selectors) == 1

    def test_affinity_warmup_reaches_shared_processors(self) -> None:
        switchyard = _deterministic_routing_switchyard(
            _config(session_affinity=True, affinity_warmup_turns=2),
        )
        classifier = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        selector = next(
            c for c in switchyard.iter_components()
            if isinstance(c, SignalTierSelectorRequestProcessor)
        )
        assert classifier._affinity is selector._affinity
        assert classifier._affinity is not None
        assert classifier._affinity.warmup_turns == 2

    def test_translator_present(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        translators = [
            c for c in switchyard.iter_components()
            if isinstance(c, TranslationEngine)
        ]
        assert len(translators) == 1

    def test_classifier_runs_before_tier_selector(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        components = list(switchyard.iter_components())
        classifier_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        selector_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, SignalTierSelectorRequestProcessor)
        )
        assert classifier_idx < selector_idx

    def test_stats_processors_wired_when_enabled(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        assert any(
            isinstance(c, StatsRequestProcessor)
            for c in switchyard.iter_components()
        )
        assert any(
            isinstance(c, StatsResponseProcessor)
            for c in switchyard.iter_components()
        )

    def test_stats_processors_absent_when_disabled(self) -> None:
        switchyard = _deterministic_routing_switchyard(
            _config(enable_stats=False),
        )
        assert not any(
            isinstance(c, StatsRequestProcessor)
            for c in switchyard.iter_components()
        )
        assert not any(
            isinstance(c, StatsResponseProcessor)
            for c in switchyard.iter_components()
        )

    def test_extra_processors_are_wired(self) -> None:
        request_processor = _NoopRequestProcessor()
        response_processor = _NoopResponseProcessor()
        switchyard = _deterministic_routing_switchyard(
            _config(),
            extra_request_processors=[request_processor],
            extra_response_processors=[response_processor],
        )
        components = list(switchyard.iter_components())
        assert request_processor in components
        assert response_processor in components

    def test_pre_routing_runs_before_classifier(self) -> None:
        pre = _NoopRequestProcessor()
        switchyard = _deterministic_routing_switchyard(
            _config(),
            pre_routing_request_processors=[pre],
        )
        components = list(switchyard.iter_components())
        stats_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, StatsRequestProcessor)
        )
        pre_idx = components.index(pre)
        classifier_idx = next(
            idx for idx, c in enumerate(components)
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        assert stats_idx < pre_idx < classifier_idx

    async def test_shared_stats_accumulator(self) -> None:
        """Recording on the shared accumulator must surface in the response processor."""
        stats = StatsAccumulator()
        switchyard = _deterministic_routing_switchyard(
            _config(),
            stats_accumulator=stats,
        )
        response_processor = next(
            c for c in switchyard.iter_components()
            if isinstance(c, StatsResponseProcessor)
        )
        await stats.record_success("aws/anthropic/bedrock-claude-opus-4-7")
        assert response_processor.accumulator.snapshot_sync()["total_requests"] == 1


class TestStderrSuppression:
    """The launcher path shares stderr with the spawned agent's TUI, so the
    classifier processor's ``classifier_signals=...`` dump must be off when the
    profile builds the chain (benchmark callers still get it via the
    ``LLMClassifierConfig.dump_signals_to_stderr=True`` default)."""

    def test_profile_disables_classifier_stderr_dump(self) -> None:
        switchyard = _deterministic_routing_switchyard(_config())
        classifier = next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )
        assert classifier._config.dump_signals_to_stderr is False


class TestClassifierReasoningHint:
    def _classifier(self, config: DeterministicRoutingConfig) -> LLMClassifierRequestProcessor:
        switchyard = _deterministic_routing_switchyard(config)
        return next(
            c for c in switchyard.iter_components()
            if isinstance(c, LLMClassifierRequestProcessor)
        )

    def test_bedrock_claude_classifier_disables_reasoning(self) -> None:
        classifier = self._classifier(
            _config(classifier_model="aws/anthropic/bedrock-claude-sonnet-4-6"),
        )
        assert classifier._config.disable_reasoning is False

    def test_deepseek_classifier_keeps_reasoning_disabled(self) -> None:
        classifier = self._classifier(
            _config(classifier_model="nvidia/deepseek-ai/deepseek-v4-flash"),
        )
        assert classifier._config.disable_reasoning is True


class TestProfileSelection:
    @pytest.mark.parametrize("profile", ["general", "coding_agent", "openclaw"])
    def test_known_profiles_build(self, profile: str) -> None:
        switchyard = _deterministic_routing_switchyard(
            _config(profile_name=profile),
        )
        assert isinstance(switchyard, ProfileSwitchyard)

    def test_unknown_profile_rejected_at_config_construction(self) -> None:
        with pytest.raises(ValueError):
            DeterministicRoutingConfig(
                strong={"model": "s"},
                weak={"model": "w"},
                classifier={"model": "c"},
                profile_name="invented_profile",  # type: ignore[arg-type]
                fallback_target_on_evict="strong",
            )


class TestDeepSeekOverrides:
    """The profile layers DeepSeek-specific extras onto tier targets."""

    def test_deepseek_v4_pro_gets_thinking_off(self) -> None:
        target = LlmTarget(
            id="weak",
            model="nvidia/deepseek-ai/evals-deepseek-v4-pro",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
        )
        out = apply_deepseek_overrides(target)
        assert out.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_deepseek_gets_batch_priority_header(self) -> None:
        target = LlmTarget(
            id="weak",
            model="nvidia/deepseek-ai/deepseek-v4-flash",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
        )
        out = apply_deepseek_overrides(target)
        assert out.extra_headers == {"X-Inference-Priority": "batch"}

    def test_non_deepseek_passes_through_unchanged(self) -> None:
        target = LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
        )
        out = apply_deepseek_overrides(target)
        assert out is target  # no rebuild needed

    def test_caller_supplied_extras_win(self) -> None:
        target = LlmTarget(
            id="weak",
            model="nvidia/deepseek-ai/evals-deepseek-v4-pro",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            extra_headers={"X-Inference-Priority": "interactive"},
        )
        out = apply_deepseek_overrides(target)
        assert out.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
        assert out.extra_headers == {"X-Inference-Priority": "interactive"}

    def test_caller_supplied_empty_body_wins(self) -> None:
        target = LlmTarget(
            id="weak",
            model="nvidia/deepseek-ai/evals-deepseek-v4-pro",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
            extra_body={},
        )
        out = apply_deepseek_overrides(target)
        assert out.extra_body == {}


class TestTierTimeoutDefaults:
    """Deterministic tiers get a bounded timeout unless callers set one."""

    def test_default_timeout_applies_when_target_has_no_timeout(self) -> None:
        target = LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
        )

        out = apply_default_tier_timeout(target, 123.0)

        assert out.endpoint.timeout_secs == 123.0

    def test_existing_timeout_wins(self) -> None:
        target = LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
            timeout_secs=45.0,
        )

        out = apply_default_tier_timeout(target, 123.0)

        assert out is target
        assert out.endpoint.timeout_secs == 45.0

    def test_none_disables_default_timeout(self) -> None:
        target = LlmTarget(
            id="strong",
            model="aws/anthropic/bedrock-claude-opus-4-7",
            format=BackendFormat.OPENAI,
            api_key="k",
            base_url="https://e/v1",
        )

        out = apply_default_tier_timeout(target, None)

        assert out is target
        assert out.endpoint.timeout_secs is None


class TestPresetIntegration:
    """Profile round-trip with the shipping preset."""

    def test_preset_builds_profile(self) -> None:
        config = DeterministicRoutingPresets.coding_agent_default(api_key="nvapi-test")
        switchyard = _deterministic_routing_switchyard(config)
        backends = [
            c for c in switchyard.iter_components()
            if isinstance(c, DeterministicRoutingLLMBackend)
        ]
        assert len(backends) == 1

    def test_preset_metadata_round_trips(self) -> None:
        config = DeterministicRoutingPresets.coding_agent_default(api_key="nvapi-test")
        assert config.preset == "coding_agent_default"
        assert config.profile_name == "coding_agent"
        assert config.strong.model == "anthropic/claude-opus-4.7"
        assert config.weak.model == "moonshotai/kimi-k2.6"
        assert config.classifier.model == "google/gemini-3.5-flash"

    def test_preset_owns_per_tier_formats(self) -> None:
        # Strong is a Claude model: AUTO probes so cache_control survives.
        config = DeterministicRoutingPresets.coding_agent_default(api_key="nvapi-test")
        assert config.strong.format is BackendFormat.AUTO
        assert config.weak.format is BackendFormat.OPENAI
        assert config.classifier.format is BackendFormat.OPENAI


def test_session_affinity_requires_nonzero_capacity() -> None:
    """Enabling affinity with a zero-capacity store is rejected as a footgun."""
    with pytest.raises(ValidationError):
        _config(session_affinity=True, affinity_max_sessions=0)
    # Enabled with capacity, or disabled with zero, are both fine.
    _config(session_affinity=True, affinity_max_sessions=1)
    _config(session_affinity=False, affinity_max_sessions=0)


def test_affinity_warmup_turns_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        _config(session_affinity=True, affinity_warmup_turns=-1)


def test_blank_classifier_prompt_is_treated_as_unset() -> None:
    config = _config(classifier_system_prompt="   ")
    assert config.classifier_system_prompt is None


def test_virtual_model_id_changes_with_prompt_and_context() -> None:
    base = _config()
    base_id = deterministic_routing_virtual_model_id(base)
    prompt_id = deterministic_routing_virtual_model_id(
        _config(classifier_system_prompt="custom classifier prompt"),
    )
    max_chars_id = deterministic_routing_virtual_model_id(
        _config(classifier_max_request_chars=1024),
    )

    assert prompt_id != base_id
    assert max_chars_id != base_id


def _custom_id_config() -> DeterministicRoutingConfig:
    return DeterministicRoutingConfig(
        strong=LlmTarget(
            id="frontier",
            model="strong-model",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://strong.invalid/v1",
        ),
        weak=LlmTarget(
            id="cheap",
            model="weak-model",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://weak.invalid/v1",
        ),
        classifier=LlmTarget(
            id="classifier",
            model="classifier-model",
            format=BackendFormat.OPENAI,
            api_key="sk-test",
            base_url="https://classifier.invalid/v1",
        ),
        profile_name="coding_agent",
        fallback_target_on_evict="frontier",
    )


def test_duplicate_tier_ids_rejected() -> None:
    config = _custom_id_config()
    with pytest.raises(ValidationError) as exc:
        DeterministicRoutingConfig.model_validate({
            **config.model_dump(),
            "strong": LlmTarget(id="tier", model="strong-model"),
            "weak": LlmTarget(id="tier", model="weak-model"),
            "fallback_target_on_evict": "tier",
        })
    assert "must differ" in str(exc.value)


def test_custom_tier_ids_key_backend_and_selector() -> None:
    """Backend tiers and the selector's labels follow the configured ids."""
    profile = DeterministicRoutingProfileConfig.from_config(_custom_id_config()).build()

    backend = profile._backend
    assert isinstance(backend, DeterministicRoutingLLMBackend)
    assert set(backend._backends) == {"frontier", "cheap"}
    assert backend._default_tier == "frontier"


class _OverflowableTier(LLMBackend):
    """Stub tier backend: overflows when told to, else returns a completion."""

    def __init__(self, *, overflow: bool, target_id: str) -> None:
        self.calls = 0
        self._overflow = overflow
        self._target_id = target_id

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self._overflow:
            error = SwitchyardContextWindowExceededError(f"{self._target_id} overflowed")
            error.target_id = self._target_id  # type: ignore[attr-defined]
            raise error
        return ChatResponse.openai_completion({
            "id": "deterministic-overflow-test",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        })


class _StampTier:
    """Stand-in for the classifier+selector pair: stamp a fixed tier label."""

    def __init__(self, tier: str) -> None:
        self._tier = tier

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = self._tier
        return request


async def test_overflow_reroutes_to_custom_strong_id_through_full_profile() -> None:
    """Weak overflow retries onto the configured strong id, not an unknown label.

    Regression test for tiers keyed as literal strong/weak while the chain's
    evict-and-retry rewrote ``selected_target`` to the *configured* fallback
    id: with ids ``cheap``/``frontier``, the rewritten pick was unrecognised,
    the retry hit weak again, and the pool exhausted.
    """
    profile = DeterministicRoutingProfileConfig.from_config(_custom_id_config()).build()
    backend = profile._backend
    assert isinstance(backend, DeterministicRoutingLLMBackend)
    cheap = _OverflowableTier(overflow=True, target_id="cheap")
    frontier = _OverflowableTier(overflow=False, target_id="frontier")
    backend._backends = {"cheap": cheap, "frontier": frontier}
    profile._request_processors = (_StampTier("cheap"),)

    request = ChatRequest.openai_chat({
        "model": "client/model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    response = await profile.run(ProfileInput(request))

    assert cheap.calls == 1
    assert frontier.calls == 1
    assert response.body["choices"][0]["message"]["content"] == "ok"

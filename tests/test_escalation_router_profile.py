# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the escalation-router config model and profile construction."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.processors.escalation_judge_request_processor import (
    ESCALATION_JUDGE_SYSTEM_PROMPT,
    EscalationJudgeRequestProcessor,
)
from switchyard.lib.processors.reasoning_effort_normalizer import ReasoningEffortNormalizer
from switchyard.lib.profiles import EscalationRouterConfig, EscalationRouterProfileConfig
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardContextWindowExceededError,
)
from switchyard_rust.profiles import ProfileInput


def _target(tier: str) -> LlmTarget:
    return LlmTarget(
        id=tier,
        model=f"{tier}-model",
        base_url=f"https://{tier}.invalid/v1",
        api_key=f"sk-{tier}",
    )


def _config(**overrides: Any) -> EscalationRouterConfig:
    data: dict[str, Any] = {
        "strong": _target("strong"),
        "weak": _target("weak"),
        "judge": _target("judge"),
        "fallback_target_on_evict": "strong",
    }
    data.update(overrides)
    return EscalationRouterConfig.model_validate(data)


def test_fallback_must_match_a_tier_id() -> None:
    with pytest.raises(ValidationError) as exc:
        _config(fallback_target_on_evict="judge")
    assert "fallback_target_on_evict" in str(exc.value)


def test_judge_target_is_required() -> None:
    with pytest.raises(ValidationError):
        EscalationRouterConfig.model_validate({
            "strong": _target("strong"),
            "weak": _target("weak"),
            "fallback_target_on_evict": "strong",
        })


def test_blank_judge_prompt_is_unset() -> None:
    assert _config(judge_system_prompt="   ").judge_system_prompt is None


def test_build_composes_judge_chain() -> None:
    profile = EscalationRouterProfileConfig.from_config(_config()).build()

    processors = profile._request_processors
    assert [type(p) for p in processors] == [
        ReasoningEffortNormalizer,
        EscalationJudgeRequestProcessor,
    ]
    backend = profile._backend
    assert isinstance(backend, DeterministicRoutingLLMBackend)
    assert set(backend._backends) == {"strong", "weak"}
    assert backend._default_tier == "weak"
    assert profile._fallback_target_on_evict == "strong"


def test_build_pins_deepseek_judge_to_batch_gateway() -> None:
    """A DeepSeek judge inherits the benchmark-gateway header like DeepSeek tiers."""
    profile = EscalationRouterProfileConfig.from_config(
        _config(
            judge=LlmTarget(
                id="judge",
                model="nvidia/deepseek-ai/deepseek-v4-flash",
                base_url="https://judge.invalid/v1",
                api_key="sk-judge",
            ),
        ),
    ).build()

    judge = profile._request_processors[1]
    assert isinstance(judge, EscalationJudgeRequestProcessor)
    assert judge._config.extra_headers == {"X-Inference-Priority": "batch"}


def test_build_non_deepseek_judge_has_no_default_headers() -> None:
    profile = EscalationRouterProfileConfig.from_config(_config()).build()

    judge = profile._request_processors[1]
    assert isinstance(judge, EscalationJudgeRequestProcessor)
    assert judge._config.extra_headers is None


def test_build_threads_judge_settings() -> None:
    profile = EscalationRouterProfileConfig.from_config(
        _config(
            judge_min_turn=5,
            judge_recent_turn_window=20,
            judge_window_message_chars=500,
            judge_disable_reasoning=False,
            judge_timeout_s=12.0,
            session_key_depth=2,
        ),
    ).build()

    judge = profile._request_processors[1]
    assert isinstance(judge, EscalationJudgeRequestProcessor)
    assert judge._config.min_judge_turn == 5
    assert judge._config.recent_turn_window == 20
    assert judge._config.window_message_chars == 500
    assert judge._config.disable_reasoning is False
    assert judge._config.timeout_s == 12.0
    assert judge._config.system_prompt == ESCALATION_JUDGE_SYSTEM_PROMPT
    assert judge._session_key_depth == 2
    assert judge._affinity.enabled


def _custom_id_config() -> EscalationRouterConfig:
    return _config(
        strong=LlmTarget(
            id="frontier",
            model="strong-model",
            base_url="https://strong.invalid/v1",
            api_key="sk-strong",
        ),
        weak=LlmTarget(
            id="cheap",
            model="weak-model",
            base_url="https://weak.invalid/v1",
            api_key="sk-weak",
        ),
        fallback_target_on_evict="frontier",
    )


def test_duplicate_tier_ids_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _config(
            strong=LlmTarget(id="tier", model="strong-model"),
            weak=LlmTarget(id="tier", model="weak-model"),
            fallback_target_on_evict="tier",
        )
    assert "must differ" in str(exc.value)


def test_custom_tier_ids_key_backend_and_judge() -> None:
    """Backend tiers and judge labels follow the configured target ids."""
    profile = EscalationRouterProfileConfig.from_config(_custom_id_config()).build()

    backend = profile._backend
    assert isinstance(backend, DeterministicRoutingLLMBackend)
    assert set(backend._backends) == {"frontier", "cheap"}
    assert backend._default_tier == "cheap"
    judge = profile._request_processors[1]
    assert isinstance(judge, EscalationJudgeRequestProcessor)
    assert judge._strong_tier == "frontier"
    assert judge._weak_tier == "cheap"


class _TierFake(LLMBackend):
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
            "id": "escalation-overflow-test",
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


async def test_overflow_reroutes_to_custom_strong_id_through_full_profile() -> None:
    """Weak overflow retries onto the configured strong id, not an unknown label.

    Regression test for tiers keyed as literal strong/weak while the chain's
    evict-and-retry rewrote ``selected_target`` to the *configured* fallback
    id: with ids ``cheap``/``frontier``, the rewritten pick was unrecognised,
    the retry hit weak again, and the pool exhausted.
    """
    profile = EscalationRouterProfileConfig.from_config(_custom_id_config()).build()
    backend = profile._backend
    assert isinstance(backend, DeterministicRoutingLLMBackend)
    cheap = _TierFake(overflow=True, target_id="cheap")
    frontier = _TierFake(overflow=False, target_id="frontier")
    backend._backends = {"cheap": cheap, "frontier": frontier}

    request = ChatRequest.openai_chat({
        "model": "client/model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    response = await profile.run(ProfileInput(request))

    assert cheap.calls == 1
    assert frontier.calls == 1
    assert response.body["choices"][0]["message"]["content"] == "ok"

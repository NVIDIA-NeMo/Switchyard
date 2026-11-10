# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from typing import Any

import pytest
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RoutingContext
from switchyard_litellm import ClassifierRoutingPlugin, SwitchyardRoutingPlugin
from switchyard_litellm.plugins.judge import serving_judge_call

from switchyard.libsy import LlmClassifierConfig, LlmResponse, TaskClassifierConfig, algorithms

CAPABLE = "provider/capable"
EFFICIENT = "provider/efficient"
JUDGE = "provider/judge"


def routing_context(
    messages: list[dict[str, Any]],
    candidates: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RoutingContext:
    return RoutingContext(
        raw_messages=messages,
        structured_messages=messages,
        candidate_models=candidates or [CAPABLE, EFFICIENT],
        metadata=metadata or {},
        signals={},
    )


class FakeJudge:
    """Return a fixed verdict, recording what Switchyard asked for."""

    def __init__(self, p_solve: float, boundary: str = "supported", rule: str = "SUP-1") -> None:
        self.verdict = {
            "crux": "x",
            "primary_rule": rule,
            "capability_boundary": boundary,
            "p_solve": p_solve,
        }
        self.calls: list[tuple[Mapping[str, object], str]] = []

    async def __call__(self, request: Mapping[str, object], model: str) -> LlmResponse.Agg:
        self.calls.append((request, model))
        return LlmResponse.Agg(
            {
                "model": model,
                "outputs": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(self.verdict)}],
                        "stop_reason": "end_turn",
                    }
                ],
            }
        )


class FailingJudge:
    async def __call__(self, request: Mapping[str, object], model: str) -> LlmResponse.Agg:
        raise RuntimeError("judge deployment unavailable")


def plugin(judge: Any, **kwargs: Any) -> ClassifierRoutingPlugin:
    return ClassifierRoutingPlugin(
        judge=judge, judge_model=JUDGE, base_threshold=0.5, threshold_step=0.1, **kwargs
    )


async def test_supported_verdict_above_threshold_selects_efficient() -> None:
    judge = FakeJudge(0.9)
    context = routing_context([{"role": "user", "content": "Rewrite this politely."}])

    result = await plugin(judge).run(context)

    assert result.candidate_models == [EFFICIENT]
    assert result.signals["switchyard"]["selected_model_id"] == EFFICIENT
    assert result.signals["switchyard"]["fallback_models"] == [CAPABLE]
    assert result.signals["switchyard"]["evidence"] == {
        "source": "llm-classifier",
        "score": 0.9,
        "threshold": 0.5,
    }
    assert result.signals["switchyard"]["algorithm"] == "llm_task_classifier"
    assert len(judge.calls) == 1
    assert judge.calls[0][1] == JUDGE


async def test_unsupported_verdict_selects_capable_with_raised_threshold() -> None:
    judge = FakeJudge(0.3, boundary="unsupported", rule="LIM-2")
    context = routing_context([{"role": "user", "content": "Prove the series diverges."}])

    result = await plugin(judge).run(context)

    assert result.candidate_models == [CAPABLE]
    assert result.signals["switchyard"]["evidence"] == {
        "source": "llm-classifier",
        "score": 0.3,
        "threshold": 0.7,
    }


async def test_judge_sees_the_custom_prompt_and_the_user_task() -> None:
    judge = FakeJudge(0.9)
    context = routing_context([{"role": "user", "content": "Rewrite this politely."}])

    await plugin(judge, prompt="Judge for a small text model.", max_output_tokens=256).run(context)

    request, _ = judge.calls[0]
    texts = [
        block["text"]
        for message in [*request.get("instructions", []), *request.get("messages", [])]
        for block in message["content"]
    ]
    assert any("Judge for a small text model." in text for text in texts)
    assert any("Rewrite this politely." in text for text in texts)
    assert request["output"]["max_output_tokens"] == 256
    assert request["output"]["response_format"]["type"] == "json_schema"


async def test_failing_judge_falls_open_to_capable_without_raising() -> None:
    context = routing_context([{"role": "user", "content": "Anything."}])

    result = await plugin(FailingJudge()).run(context)

    assert result.candidate_models == [CAPABLE]
    assert result.signals["switchyard"]["evidence"]["source"] == "fail_open"


async def test_judge_requests_pass_through_untouched() -> None:
    judge = FakeJudge(0.9)
    context = routing_context(
        [{"role": "user", "content": "the judge prompt"}],
        candidates=[JUDGE],
        metadata={"tenant": "t"},
    )

    with serving_judge_call():
        result = await plugin(judge).run(context)

    assert result is context
    assert result.candidate_models == [JUDGE]
    assert result.signals == {}
    assert judge.calls == []


async def test_client_supplied_judge_metadata_does_not_skip_routing() -> None:
    judge = FakeJudge(0.9)
    context = routing_context(
        [{"role": "user", "content": "Rewrite this politely."}],
        metadata={"switchyard_judge": True},
    )

    result = await plugin(judge).run(context)

    assert result.candidate_models == [EFFICIENT]
    assert len(judge.calls) == 1


async def test_core_plugin_also_passes_judge_requests_through() -> None:
    algorithm = algorithms.llm_classifier(
        LlmClassifierConfig.capability(config=TaskClassifierConfig(0.5, max_output_tokens=64))
    )
    core = SwitchyardRoutingPlugin(algorithm, judge=FakeJudge(0.9))
    context = routing_context([{"role": "user", "content": "x"}], candidates=[JUDGE])

    with serving_judge_call():
        assert await core.run(context) is context
    assert context.candidate_models == [JUDGE]


async def test_core_plugin_without_a_judge_still_rejects_intermediate_calls() -> None:
    algorithm = algorithms.llm_classifier(
        LlmClassifierConfig.capability(config=TaskClassifierConfig(0.5, max_output_tokens=64))
    )
    core = SwitchyardRoutingPlugin(
        algorithm,
        models={
            "any": [CAPABLE, EFFICIENT],
            "capable": [CAPABLE],
            "efficient": [EFFICIENT],
            "judge": [JUDGE],
        },
    )

    with pytest.raises(ValueError, match="configure a judge client"):
        await core.run(routing_context([{"role": "user", "content": "x"}]))


@pytest.mark.parametrize("candidates", [[CAPABLE], [CAPABLE, EFFICIENT, "provider/third"]])
async def test_requires_exactly_two_unique_candidates(candidates: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly two unique"):
        await plugin(FakeJudge(0.9)).run(
            routing_context([{"role": "user", "content": "x"}], candidates=candidates)
        )


async def test_judge_model_must_not_be_a_candidate() -> None:
    with pytest.raises(ValueError, match="must not be a routing candidate"):
        await plugin(FakeJudge(0.9)).run(
            routing_context([{"role": "user", "content": "x"}], candidates=[CAPABLE, JUDGE])
        )


def test_plugin_is_a_litellm_callback_for_the_deployment_hook() -> None:
    assert isinstance(plugin(FakeJudge(0.9)), CustomLogger)


def test_rejects_unknown_response_format_type() -> None:
    with pytest.raises(ValueError, match="response_format_type"):
        plugin(FakeJudge(0.9), response_format_type="yaml")

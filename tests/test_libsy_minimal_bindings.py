# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dictionary-based libsy Python API."""

from typing import Any

import pytest

from switchyard.libsy import LibsyError, LlmTarget, algorithms


def request_body() -> dict[str, Any]:
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
    }


def stage_request(*, failed: bool) -> dict[str, Any]:
    result = "fatal runtime error: out of memory" if failed else "ok"
    return {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "fix the build"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "Bash",
                        "arguments": {"command": "cargo test"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "content": [{"type": "text", "text": result}],
                        "is_error": failed,
                    }
                ],
            },
        ],
    }


class EchoClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "model": self.model,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.model}],
                    "stop_reason": "end_turn",
                }
            ],
        }


class JudgeClient:
    def __init__(
        self,
        *,
        p_solve: float = 0.9,
        confidence: float = 0.9,
        capability_boundary: str = "supported",
        fail: bool = False,
        malformed: bool = False,
    ) -> None:
        self.p_solve = p_solve
        self.confidence = confidence
        self.capability_boundary = capability_boundary
        self.fail = fail
        self.malformed = malformed
        self.calls: list[dict[str, Any]] = []

    async def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("judge failed")
        verdict = "not json"
        if not self.malformed:
            verdict = (
                '{"recommended_route":"efficient",'
                f'"p_solve":{self.p_solve},'
                f'"confidence":{self.confidence},'
                '"abstain":false,'
                f'"capability_boundary":"{self.capability_boundary}",'
                '"primary_rule":"SUP-1",'
                '"crux":"bounded task"}'
            )
        return {
            "model": "judge",
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": verdict}],
                    "stop_reason": "end_turn",
                }
            ],
        }


async def test_random_runs_with_a_python_client() -> None:
    client = EchoClient("fast")
    algorithm = algorithms.random([LlmTarget("fast", client)])

    decisions, response = await algorithm.run(request_body())

    assert decisions == [
        {
            "selected_model": "fast",
            "reasoning": "random routing selected target 'fast'",
            "routing_tier": None,
        }
    ]
    assert client.calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert response["model"] == "fast"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "fast"}]


async def test_random_weights_and_seed_are_reproducible() -> None:
    def algorithm():
        return algorithms.random(
            [
                LlmTarget("fast", EchoClient("fast")),
                LlmTarget("capable", EchoClient("capable")),
            ],
            weights=[1, 3],
            seed=42,
        )

    first_router = algorithm()
    second_router = algorithm()
    first = [(await first_router.run(request_body()))[1]["model"] for _ in range(100)]
    second = [(await second_router.run(request_body()))[1]["model"] for _ in range(100)]

    assert first == second
    assert 65 <= second.count("capable") <= 85


def test_random_rejects_invalid_weights() -> None:
    targets = [
        LlmTarget("fast", EchoClient("fast")),
        LlmTarget("capable", EchoClient("capable")),
    ]

    with pytest.raises(ValueError, match="expected 2 weights, got 1"):
        algorithms.random(targets, weights=[1])


async def test_noop_needs_no_client() -> None:
    decisions, response = await algorithms.noop().run(request_body())

    assert decisions[0]["selected_model"] == "auto"
    assert response["outputs"][0]["content"] == [{"type": "text", "text": "OK"}]


async def test_random_decides_with_a_clientless_target() -> None:
    algorithm = algorithms.random([LlmTarget("fast")], seed=42)

    decision = await algorithm.decide(request_body())

    assert decision == {
        "selected_model": "fast",
        "reasoning": "random routing selected target 'fast'",
        "routing_tier": None,
    }


async def test_random_decision_weights_and_seed_share_the_run_sequence() -> None:
    targets = [LlmTarget("fast"), LlmTarget("capable")]
    first = algorithms.random(targets, weights=[1, 3], seed=42)
    second = algorithms.random(targets, weights=[1, 3], seed=42)

    first_decisions = [
        (await first.decide(request_body()))["selected_model"] for _ in range(100)
    ]
    second_decisions = [
        (await second.decide(request_body()))["selected_model"] for _ in range(100)
    ]

    assert first_decisions == second_decisions
    assert 65 <= second_decisions.count("capable") <= 85

    def runnable():
        return algorithms.random(
            [
                LlmTarget("fast", EchoClient("fast")),
                LlmTarget("capable", EchoClient("capable")),
            ],
            weights=[1, 3],
            seed=42,
        )

    mixed = runnable()
    runs_only = runnable()
    mixed_sequence = [
        (await mixed.decide(request_body()))["selected_model"],
        (await mixed.run(request_body()))[1]["model"],
        (await mixed.decide(request_body()))["selected_model"],
        (await mixed.run(request_body()))[1]["model"],
    ]
    run_sequence = [
        (await runs_only.run(request_body()))[1]["model"] for _ in range(4)
    ]

    assert mixed_sequence == run_sequence


async def test_noop_decides_without_a_client() -> None:
    decision = await algorithms.noop().decide(request_body())

    assert decision == {
        "selected_model": "auto",
        "reasoning": None,
        "routing_tier": None,
    }


async def test_passthrough_decides_with_a_clientless_target() -> None:
    algorithm = algorithms.passthrough(LlmTarget("fast"))

    decision = await algorithm.decide(request_body())

    assert decision == {
        "selected_model": "fast",
        "reasoning": None,
        "routing_tier": None,
    }


async def test_run_rejects_a_clientless_selected_target() -> None:
    algorithm = algorithms.random([LlmTarget("fast")])

    with pytest.raises(LibsyError, match="has no client"):
        await algorithm.run(request_body())


async def test_run_trace_includes_routing_tier() -> None:
    algorithm = algorithms.random([LlmTarget("fast", EchoClient("fast"))])

    decisions, _ = await algorithm.run(request_body())

    assert decisions == [
        {
            "selected_model": "fast",
            "reasoning": "random routing selected target 'fast'",
            "routing_tier": None,
        }
    ]


async def test_classifier_decides_and_calls_only_the_judge() -> None:
    judge = JudgeClient()
    fast = EchoClient("fast")
    strong = EchoClient("strong")
    algorithm = algorithms.llm_classifier(
        judge=LlmTarget("judge", judge),
        efficient=LlmTarget("fast", fast),
        capable=LlmTarget("strong", strong),
        base_threshold=0.5,
    )

    decision = await algorithm.decide(request_body())

    assert decision == {
        "selected_model": "fast",
        "reasoning": "fall-through selected fast (confidence 1.000)",
        "routing_tier": "weak",
    }
    assert len(judge.calls) == 1
    assert fast.calls == []
    assert strong.calls == []


async def test_classifier_decides_capable_and_fails_open() -> None:
    capable = algorithms.llm_classifier(
        judge=LlmTarget("judge", JudgeClient(p_solve=0.1)),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
    )
    failed_judge = algorithms.llm_classifier(
        judge=LlmTarget("judge", JudgeClient(fail=True)),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
    )
    malformed_judge = algorithms.llm_classifier(
        judge=LlmTarget("judge", JudgeClient(malformed=True)),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
    )

    capable_decision = await capable.decide(request_body())
    failed_decision = await failed_judge.decide(request_body())
    malformed_decision = await malformed_judge.decide(request_body())

    assert capable_decision["selected_model"] == "strong"
    assert capable_decision["routing_tier"] == "strong"
    assert failed_decision["selected_model"] == "strong"
    assert failed_decision["routing_tier"] == "strong"
    assert malformed_decision["selected_model"] == "strong"
    assert malformed_decision["routing_tier"] == "strong"


async def test_classifier_decision_honors_confidence_and_elevated_floor() -> None:
    low_confidence = algorithms.llm_classifier(
        judge=LlmTarget("judge", JudgeClient(confidence=0.8)),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
        min_confidence=0.9,
    )
    elevated_boundary = algorithms.llm_classifier(
        judge=LlmTarget(
            "judge",
            JudgeClient(p_solve=0.7, capability_boundary="uncertain"),
        ),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
        capability_elevated_floor=0.8,
    )

    low_confidence_decision = await low_confidence.decide(request_body())
    elevated_boundary_decision = await elevated_boundary.decide(request_body())

    assert low_confidence_decision["selected_model"] == "strong"
    assert elevated_boundary_decision["selected_model"] == "strong"


async def test_classifier_accepts_a_recent_turn_window() -> None:
    judge = JudgeClient()
    algorithm = algorithms.llm_classifier(
        judge=LlmTarget("judge", judge),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
        recent_turn_window=3,
    )

    decision = await algorithm.decide(request_body())

    assert decision["selected_model"] == "fast"
    assert len(judge.calls) == 1


async def test_stage_router_decides_from_tool_signals_without_clients() -> None:
    algorithm = algorithms.stage_router(
        capable=LlmTarget("strong"),
        efficient=LlmTarget("fast"),
        picker="efficient_first",
        confidence_threshold=0.5,
    )

    capable = await algorithm.decide(stage_request(failed=True))
    efficient = await algorithm.decide(stage_request(failed=False))

    assert capable["selected_model"] == "strong"
    assert capable["routing_tier"] == "strong"
    assert efficient["selected_model"] == "fast"
    assert efficient["routing_tier"] == "weak"


async def test_stage_router_decision_calls_only_its_optional_judge() -> None:
    judge = JudgeClient(p_solve=0.1)
    algorithm = algorithms.stage_router(
        capable=LlmTarget("strong"),
        efficient=LlmTarget("fast"),
        picker="efficient_first",
        confidence_threshold=0.5,
        judge=LlmTarget("judge", judge),
        classifier_base_threshold=0.5,
    )

    decision = await algorithm.decide(stage_request(failed=False))

    assert decision["selected_model"] == "strong"
    assert decision["routing_tier"] == "strong"
    assert len(judge.calls) == 1


async def test_stage_router_accepts_the_full_configuration() -> None:
    algorithm = algorithms.stage_router(
        capable=LlmTarget("strong"),
        efficient=LlmTarget("fast"),
        picker="capable_first",
        confidence_threshold=0.5,
        recent_turn_window=2,
        handoff_escalation_note="Continue the failed diagnosis.",
        handoff_deescalation_note="The build is healthy again.",
        handoff_only_on_wrong_signal_escalation=False,
        capable_system_prompt="Solve difficult failures.",
        efficient_system_prompt="Handle routine work.",
        judge=LlmTarget("judge", JudgeClient()),
        classifier_base_threshold=0.5,
        classifier_min_confidence=0.2,
        classifier_capability_elevated_floor=0.8,
        classifier_recent_turn_window=4,
    )

    decision = await algorithm.decide(stage_request(failed=True))

    assert decision["selected_model"] == "strong"
    assert decision["routing_tier"] == "strong"


def test_stage_router_rejects_invalid_configuration() -> None:
    target = LlmTarget("target")

    with pytest.raises(ValueError, match="picker"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="unknown",
            confidence_threshold=0.5,
        )

    with pytest.raises(ValueError, match="confidence_threshold"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="efficient_first",
            confidence_threshold=1.5,
        )

    with pytest.raises(ValueError, match="judge and classifier_base_threshold"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="efficient_first",
            confidence_threshold=0.5,
            judge=target,
        )

    with pytest.raises(ValueError, match="judge and classifier_base_threshold"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="efficient_first",
            confidence_threshold=0.5,
            classifier_base_threshold=0.5,
        )

    with pytest.raises(ValueError, match="requires handoff_escalation_note"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="efficient_first",
            confidence_threshold=0.5,
            handoff_deescalation_note="healthy",
        )

    with pytest.raises(ValueError, match="require judge"):
        algorithms.stage_router(
            capable=target,
            efficient=target,
            picker="efficient_first",
            confidence_threshold=0.5,
            classifier_min_confidence=0.2,
        )


async def test_classifier_decision_preserves_session_affinity() -> None:
    judge = JudgeClient()
    algorithm = algorithms.llm_classifier(
        judge=LlmTarget("judge", judge),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
        session_affinity=True,
    )
    headers = {"x-switchyard-session-id": "session-1"}

    first = await algorithm.decide(request_body(), headers=headers)
    second = await algorithm.decide(request_body(), headers=headers)

    assert first["selected_model"] == "fast"
    assert second["selected_model"] == "fast"
    assert len(judge.calls) == 1


async def test_classifier_requires_a_judge_client_when_deciding() -> None:
    algorithm = algorithms.llm_classifier(
        judge=LlmTarget("judge"),
        efficient=LlmTarget("fast"),
        capable=LlmTarget("strong"),
        base_threshold=0.5,
    )

    with pytest.raises(LibsyError, match="judge.*has no client"):
        await algorithm.decide(request_body())


def test_classifier_rejects_invalid_configuration() -> None:
    target = LlmTarget("target")

    with pytest.raises(ValueError, match="base_threshold"):
        algorithms.llm_classifier(
            judge=target,
            efficient=target,
            capable=target,
            base_threshold=1.5,
        )

    with pytest.raises(ValueError, match="message_hash_fallback requires session_affinity"):
        algorithms.llm_classifier(
            judge=target,
            efficient=target,
            capable=target,
            base_threshold=0.5,
            message_hash_fallback=True,
        )


def test_algorithm_exposes_only_managed_execution() -> None:
    algorithm = algorithms.noop()

    assert callable(algorithm.run)
    assert not hasattr(algorithm, "run_stream")


def test_target_requires_a_callable_client() -> None:
    with pytest.raises(TypeError, match="client must define async call"):
        LlmTarget("fast", object())

    with pytest.raises(TypeError, match="client.call must be callable"):
        LlmTarget("fast", type("Client", (), {"call": None})())


def test_random_requires_a_target() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        algorithms.random([])


async def test_invalid_request_is_rejected_at_the_boundary() -> None:
    algorithm = algorithms.random([LlmTarget("fast", EchoClient("fast"))])

    with pytest.raises(ValueError, match="unknown variant"):
        await algorithm.run(
            {
                "model": "auto",
                "messages": [{"role": "invalid", "content": []}],
            }
        )


async def test_client_failure_becomes_libsy_error() -> None:
    class FailingClient:
        async def call(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("client failed")

    algorithm = algorithms.random([LlmTarget("broken", FailingClient())])

    with pytest.raises(LibsyError, match="client failed"):
        await algorithm.run(request_body())

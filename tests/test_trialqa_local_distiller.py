# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-network tests for native, container-free TrialQA distillation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import benchmark.trialqa_local_distiller as distiller_module
import benchmark.trialqa_tooluniverse_mcp as adapter_module
from benchmark.trialqa_local_distiller import (
    DISTILLER_MODEL,
    DISTILLER_ROUTE,
    EXECUTOR_MODEL,
    EXECUTOR_ROUTE,
    NAMESPACE,
    DistillationPlan,
    DonorEvidence,
    ModelCall,
    ModelCallResult,
    TrialQADistillationError,
    _assert_no_sensitive,
    _sanitize,
    build_distillation_plan,
    execute_distillation,
    main,
)
from switchyard.lib.skill_distillation_native import (
    NativeTrialQAEvidenceError,
    import_native_trialqa_evidence,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore

REFERENCE_REPO = (
    Path(__file__).resolve().parents[1] / ".experiments" / "skill-distillation-demo" / "reference"
)
ROUTING_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "routing-profiles"
    / "skill-distillation-nemotron-ultra.yaml"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("public_name", sorted(distiller_module.PUBLIC_TRIALQA_TOOLS))
def test_namespace_tool_names_are_decoded_for_public_skill_rules(
    public_name: str,
) -> None:
    encoded = f"__sy1n17_mcp__tooluniverse{public_name}"
    events = [
        {
            "kind": "tool_call",
            "payload": {"function": {"name": encoded, "arguments": "{}"}},
        }
    ]

    assert distiller_module._public_tool_name(public_name, structural=True) == public_name
    assert distiller_module._public_tool_name(encoded) == public_name
    assert distiller_module._observed_tools(events) == frozenset({public_name})
    assert distiller_module._public_tool_view(events) == [
        {
            "kind": "tool_call",
            "payload": {"function": {"name": public_name, "arguments": "{}"}},
        }
    ]


@pytest.mark.parametrize(
    "name",
    [
        "__sy1n_mcp__tooluniversetrialqa_get_study",
        "__sy1n999_mcp__tooluniversetrialqa_get_study",
    ],
)
def test_namespace_tool_decoder_rejects_malformed_names(name: str) -> None:
    with pytest.raises(TrialQADistillationError, match="namespace|length"):
        distiller_module._public_tool_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "__sy1n3_webtrialqa_get_study",
        "__sy1n17_mcp__tooluniverseother_tool",
        "__sy1n17_mcp__tooluniverseget_protocol",
        "__sy1n17_mcp__tooluniversesearch_trials",
        "__sy1n17_mcp__trialqa_search_trials",
        "mcp__trialqa__search_trials",
        "trialqa__search_trials",
        "trialqa_get_protocol",
        "trialqa_search_trials",
        "exec_command",
        "get_interventions",
        "get_trial",
        "list_tools",
        "search_trials",
        "tools",
    ],
)
def test_unsupported_tool_names_map_to_non_callable_sentinel(name: str) -> None:
    assert (
        distiller_module._public_tool_name(name, structural=True)
        == distiller_module.UNSUPPORTED_TOOL_SENTINEL
    )


def test_public_tool_view_normalizes_embedded_names_and_rejects_residual_marker() -> None:
    encoded = "__sy1n17_mcp__tooluniversetrialqa_get_study"
    unsupported = "__sy1n17_mcp__tooluniverseget_protocol"

    assert distiller_module._public_tool_view(
        f"Use {encoded}; never retry {unsupported} or mcp__trialqa__search_trials."
    ) == ("Use trialqa_get_study; never retry unsupported_tool_call or unsupported_tool_call.")
    with pytest.raises(TrialQADistillationError, match="residual internal tool"):
        distiller_module._public_tool_view("broken __sy1n_bad marker")


def test_observed_tools_exclude_unsupported_arbitrary_and_ambiguous_calls() -> None:
    encoded = "__sy1n17_mcp__tooluniversetrialqa_search"
    events = [
        {
            "kind": "tool_call",
            "payload": {"function": {"name": encoded, "arguments": "{}"}},
        },
        {
            "kind": "tool_call",
            "payload": {"function": {"name": "search_trials", "arguments": "{}"}},
        },
        {
            "kind": "tool_call",
            "payload": {"function": {"name": "bash", "arguments": "{}"}},
        },
        {
            "kind": "tool_call",
            "payload": {
                "name": "trialqa_get_study",
                "function": {"name": encoded, "arguments": "{}"},
            },
        },
        {"kind": "tool_call", "payload": {"function": {"arguments": "{}"}}},
    ]

    assert distiller_module._observed_tools(events) == frozenset({"trialqa_search"})
    assert distiller_module._public_tool_name("bash", structural=True) == (
        distiller_module.UNSUPPORTED_TOOL_SENTINEL
    )


def test_analyst_canonicalization_defaults_category_and_decodes_tool_name() -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "1" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="a" * 64,
    )
    output = {
        "source_task_name": "model-invented-source-name",
        "memory_items": [
            {
                "rule_type": "tool_rule",
                "tool_name": "__sy1n17_mcp__tooluniversetrialqa_get_study",
            },
            {"rule_type": "gotcha", "category": "exactness"},
        ],
    }

    canonical = distiller_module._canonicalize_model_output(call, output)

    assert canonical["source_task_name"] == call.key
    assert canonical["skill_patch"] == {
        "target": "tooluniverse-trialqa/SKILL.md",
        "sections": [],
    }
    assert canonical["memory_items"] == [
        {
            "rule_type": "tool_rule",
            "tool_name": "trialqa_get_study",
            "category": "evidence_retrieval",
        },
        {"rule_type": "gotcha", "category": "exactness"},
    ]


@pytest.mark.parametrize(
    ("rule_type", "expected_category"),
    sorted(distiller_module.DEFAULT_CATEGORY_BY_RULE_TYPE.items()),
)
def test_analyst_canonicalization_defaults_only_missing_categories(
    rule_type: str, expected_category: str
) -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "2" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="b" * 64,
    )
    canonical = distiller_module._canonicalize_model_output(
        call,
        {
            "memory_items": [
                {"rule_type": rule_type},
                {"rule_type": rule_type, "category": "verification"},
                {"rule_type": rule_type, "category": None},
                {"rule_type": "unknown"},
            ]
        },
    )

    assert canonical["memory_items"] == [
        {"rule_type": rule_type, "category": expected_category},
        {"rule_type": rule_type, "category": "verification"},
        {"rule_type": rule_type, "category": None},
        {"rule_type": "unknown"},
    ]


def test_analyst_canonicalization_repairs_category_used_as_unambiguous_rule_type() -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "3" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="c" * 64,
    )
    canonical = distiller_module._canonicalize_model_output(
        call,
        {
            "memory_items": [
                {
                    "rule_type": "failure_avoidance",
                    "category": "failure_avoidance",
                    "trigger": "trigger",
                    "symptom": "symptom",
                    "prevention": "prevention",
                },
                {
                    "rule_type": "other",
                    "category": "other",
                    "tool_name": "trialqa_search",
                    "rule": "ambiguous rule",
                    "when": "searching",
                    "rationale": "also a workflow shape",
                },
                {"rule_type": "unknown", "category": "other", "fact": "fact"},
            ]
        },
    )

    items = canonical["memory_items"]
    assert isinstance(items, list)
    assert items[0]["rule_type"] == "failure_mode"
    assert items[1]["rule_type"] == "other"
    assert items[2]["rule_type"] == "unknown"


def test_analyst_canonicalization_defaults_only_missing_section_action() -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "4" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="d" * 64,
    )
    canonical = distiller_module._canonicalize_model_output(
        call,
        {
            "skill_patch": {
                "sections": [
                    {"heading": "A", "content": "missing action"},
                    {"heading": "B", "content": "explicit action", "action": "replace"},
                ],
            }
        },
    )

    assert canonical["schema_version"] == "trace2skill_patch.v2"
    assert canonical["skill_patch"]["target"] == "tooluniverse-trialqa/SKILL.md"
    assert canonical["skill_patch"]["sections"] == [
        {"heading": "A", "content": "missing action", "action": "append"},
        {"heading": "B", "content": "explicit action", "action": "replace"},
    ]


@pytest.mark.parametrize(
    "tool_name",
    [
        "__sy1n17_mcp__tooluniversetrialqa_search",
        "mcp__trialqa__search_trials",
        "unsupported_tool_call",
        "trialqa_get_protocol",
        "search_trials",
        "bash",
    ],
)
def test_rendered_skill_rejects_internal_or_unsupported_tool_headings(
    tool_name: str,
) -> None:
    catalog = {
        "tool_rules": [
            {
                "tool_name": tool_name,
                "rules": [
                    {
                        "when": "retrieving a clinical-trial record",
                        "rule": "Inspect the structured response before answering the question.",
                    }
                ],
            },
        ],
        "workflow_rules": [],
        "failure_modes": [],
        "gotchas": [],
    }

    with pytest.raises(TrialQADistillationError, match="internal|unsupported"):
        distiller_module.render_skill_markdown(catalog)


def _task(*, repeat_index: int = 1, partition: str = "train") -> dict[str, object]:
    group = "trialqa-0001-a1b2c3d4e5f6"
    return {
        "id": f"{group}-r{repeat_index:03d}",
        "question_id": "trialqa-row-0001",
        "question_group_key": group,
        "question": "Which reusable field identifies the matching clinical trial record?",
        "condition": "donor",
        "partition": partition,
        "repeat_index": repeat_index,
        "n_repeats": 5,
    }


def _run(task: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": f"donor-{task['id']}",
        "phase": "donor",
        "model": EXECUTOR_MODEL,
        "route": EXECUTOR_ROUTE,
        "harness": "codex",
        "skill_loaded": False,
        "candidate_id": None,
        "candidate_manifest_sha256": None,
        "candidate_skill_sha256": None,
    }


def _outcome(
    *,
    score: float = 1.0,
    answer: str = "The record identifier is in identificationModule.nctId.",
) -> dict[str, object]:
    return {
        "score": score,
        "verifier": "trialqa-semantic-judge-v1",
        "label": "correct" if score else "incorrect",
        "judge_rationale": "The submitted answer matches the donor ideal.",
        "ideal_answer": "Use the protocolSection.identificationModule.nctId field.",
        "submitted_answer": answer,
    }


def _turns(session_id: str) -> list[dict[str, object]]:
    question = {
        "role": "user",
        "content": "Which reusable field identifies the matching clinical trial record?",
    }
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "trialqa_search",
                    "arguments": '{"query":"matching trial"}',
                },
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "trialqa_get_study",
                    "arguments": '{"record":"first result"}',
                },
            },
        ],
    }
    common = {
        "schema_version": 1,
        "session_id": session_id,
        "active_skill_version": None,
        "active_skill_candidate_id": None,
        "active_skill_manifest_sha256": None,
        "served_model": EXECUTOR_MODEL,
    }
    return [
        {
            **common,
            "turn_index": 0,
            "recorded_at": "2026-07-06T10:00:00Z",
            "request": {"model": EXECUTOR_ROUTE, "messages": [question]},
            "response": {"choices": [{"message": call, "finish_reason": "tool_calls"}]},
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
        {
            **common,
            "turn_index": 1,
            "recorded_at": "2026-07-06T10:00:01Z",
            "request": {
                "model": EXECUTOR_ROUTE,
                "messages": [
                    question,
                    call,
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "trialqa_search",
                        "content": '{"protocolSection":{"identificationModule":{"nctId":"NCT99999999"}}}',
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-2",
                        "name": "trialqa_get_study",
                        "content": '{"status":"verified"}',
                    },
                ],
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "answer": "The record identifier is in identificationModule.nctId."
                                },
                                separators=(",", ":"),
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        },
    ]


def _import_evidence(
    project: Path,
    *,
    repeat_index: int = 1,
    partition: str = "train",
    context_overrides: Mapping[str, object] | None = None,
    active_loaded: bool = False,
    stats_errors: int = 0,
    score: float = 1.0,
    outcome_answer: str = "The record identifier is in identificationModule.nctId.",
) -> str:
    task = _task(repeat_index=repeat_index, partition=partition)
    run = _run(task)
    session_id = f"codex-{repeat_index}-{partition}"
    store = SkillDistillationStore(NAMESPACE, project)
    session = store.sessions_path / session_id
    session.mkdir()
    turns = _turns(session_id)
    turns_bytes = "".join(
        json.dumps(turn, ensure_ascii=False, sort_keys=True) + "\n" for turn in turns
    ).encode("utf-8")
    (session / "turns.jsonl").write_bytes(turns_bytes)
    _write_json(
        session / "stats.json",
        {
            "total_requests": 2,
            "total_errors": stats_errors,
            "models": {EXECUTOR_MODEL: {"calls": 2}},
            "classifier": {"total_requests": 0, "total_errors": 0},
            "planner": {"total_requests": 0, "total_errors": 0},
            "openai_transport": {
                "physical_attempts": 2,
                "null_eof_retries": 0,
                "retry_usage_charges": 0,
                "unpriced_null_eof_retries": 0,
                "retry_token_sensitivity": {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "cache_creation": 0,
                    "reasoning": 0,
                    "total": 0,
                },
            },
        },
    )
    context = {
        "task_id": task["id"],
        "row_id": task["question_id"],
        "question_group_key": task["question_group_key"],
        "partition": task["partition"],
        "condition": task["condition"],
        "repeat_index": task["repeat_index"],
        "n_repeats": task["n_repeats"],
        "manifest_id": run["run_id"],
        "phase": run["phase"],
        "route": run["route"],
        "executor_model": run["model"],
        "skill_loaded": False,
        "candidate_id": None,
        "candidate_manifest_sha256": None,
        "candidate_skill_sha256": None,
    }
    context.update(context_overrides or {})
    _write_json(
        session / "session.json",
        {
            "schema_version": 1,
            "session_id": session_id,
            "namespace": NAMESPACE,
            "launch_target": "codex",
            "display_model": EXECUTOR_ROUTE,
            "strategy_summary": f"profile: {EXECUTOR_ROUTE}",
            "started_at": "2026-07-06T10:00:00Z",
            "ended_at": "2026-07-06T10:00:02Z",
            "status": "completed",
            "exit_code": 0,
            "turn_count": len(turns),
            "trajectory_sha256": f"sha256:{hashlib.sha256(turns_bytes).hexdigest()}",
            "run_context": context,
            "active_skill": {
                "loaded": active_loaded,
                "candidate_id": "unexpected" if active_loaded else None,
                "manifest_sha256": "sha256:" + "a" * 64 if active_loaded else None,
                "skill_sha256": "sha256:" + "b" * 64 if active_loaded else None,
                "path": "/tmp/unexpected" if active_loaded else None,
            },
        },
    )
    outcome = _outcome(score=score, answer=outcome_answer)
    outcome.update(
        {
            "partition": task["partition"],
            "row_id": task["question_id"],
            "question": task["question"],
            "question_group_key": task["question_group_key"],
            "repeat_index": task["repeat_index"],
            "n_repeats": task["n_repeats"],
            "task_name": task["id"],
            "condition": task["condition"],
        }
    )
    result = import_native_trialqa_evidence(
        session,
        namespace=NAMESPACE,
        task=task,
        outcome=outcome,
        run=run,
        project_dir=project,
    )
    return result.evidence_id


def _plan(project: Path, evidence_id: str) -> DistillationPlan:
    return build_distillation_plan(
        project_dir=project,
        namespace=NAMESPACE,
        evidence_ids=[evidence_id],
        work_dir=project / "distillation-runs",
        reference_repo=REFERENCE_REPO,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        expected_question_count=24,
        expected_repeats=5,
        mode="pilot",
    )


class _FakeCaller:
    def __init__(self) -> None:
        self.calls: list[ModelCall] = []

    def __call__(self, call: ModelCall) -> ModelCallResult:
        self.calls.append(call)
        user_content = str(call.payload["messages"][1]["content"])
        is_error = '"role": "error"' in user_content
        if call.stage == "analyst":
            output: dict[str, Any] = {
                "schema_version": "trace2skill_patch.v2",
                "role": "error" if is_error else "success",
                "source_task_name": call.key,
                "memory_items": [
                    {
                        "rule_type": "tool_rule",
                        "category": "evidence_retrieval",
                        "tool_name": "trialqa_search",
                        "rule": "Read the nested identificationModule field returned by the search result before selecting a record.",
                        "when": "matching a search result to the requested clinical trial",
                        "confidence": 0.9,
                    }
                ],
                "skill_patch": {
                    "target": "tooluniverse-trialqa/SKILL.md",
                    "sections": [],
                },
            }
            if is_error:
                output["diagnosis"] = {
                    "failure_surface": "The final response selected the wrong evidence field.",
                    "expected_vs_actual": "The expected structured field differed from the submitted field.",
                    "root_cause": "The nested result structure was not inspected before answering.",
                    "corrected_strategy": "Inspect the nested identification module and verify the field meaning before answering.",
                    "causal_trace_steps": [
                        "step 2: the nested search result was not checked against the requested field"
                    ],
                }
        elif call.stage == "question_merge":
            output = {
                "schema_version": "trace2skill_question_merge.v2",
                "question_group_key": call.key,
                "summary": "One repeat supports a nested-field retrieval rule.",
                "source_patch_count": 1,
                "repeat_count": 1,
                "role_counts": {"error": 1} if is_error else {"success": 1},
                "judge_result_counts": {"incorrect": 1} if is_error else {"correct": 1},
                "tool_rules": [
                    {
                        "tool_name": "trialqa_search",
                        "rule": "Read the nested identificationModule field returned by the search result before selecting a record.",
                        "when": "matching a search result to the requested clinical trial",
                        "confidence": 0.9,
                        "source_patch_count": 1,
                    }
                ],
                "workflow_rules": [],
                "failure_modes": [],
                "gotchas": [],
            }
        else:
            output = {
                "schema_version": "trace2skill_merge.v2",
                "skill_name": "tooluniverse-trialqa",
                "summary": "Use structured search evidence before answering.",
                "tool_rules": [
                    {
                        "tool_name": "trialqa_search",
                        "rules": [
                            {
                                "rule": "Read the nested identificationModule field returned by the search result before selecting a record.",
                                "when": "matching a search result to the requested clinical trial",
                                "confidence": 0.9,
                                "source_patch_count": 1,
                            }
                        ],
                    }
                ],
                "workflow_rules": [],
                "failure_modes": [],
                "gotchas": [],
            }
        return ModelCallResult(
            content=json.dumps(output),
            route_model=DISTILLER_ROUTE,
            upstream_model=DISTILLER_MODEL,
            request_id=f"request-{call.stage}-{call.key}",
            usage={"total_tokens": 100},
        )


class _Stats:
    def __init__(self, caller: _FakeCaller) -> None:
        self.caller = caller

    def __call__(self) -> Mapping[str, object]:
        count = len(self.caller.calls)
        return {
            "total_requests": count,
            "total_errors": 0,
            "models": ({DISTILLER_MODEL: {"calls": count}} if count else {}),
        }


class _LeakyAnalystCaller(_FakeCaller):
    def __call__(self, call: ModelCall) -> ModelCallResult:
        result = super().__call__(call)
        if call.stage != "analyst":
            return result
        output = json.loads(result.content)
        output["memory_items"][0]["rule"] += (
            " Never hard-code NCT99999999 or source row trialqa-row-0001. "
            "The record identifier is in identificationModule.nctId. "
            "The submitted answer matches the donor ideal. "
            f"Source evidence was {call.key}."
        )
        return ModelCallResult(
            content=json.dumps(output),
            route_model=result.route_model,
            upstream_model=result.upstream_model,
            request_id=result.request_id,
            usage=result.usage,
        )


class _WrongStructuralGroupCaller(_FakeCaller):
    def __call__(self, call: ModelCall) -> ModelCallResult:
        result = super().__call__(call)
        if call.stage != "question_merge":
            return result
        output = json.loads(result.content)
        output["question_group_key"] = "model-invented-group-key"
        return replace(result, content=json.dumps(output))


def test_plan_binds_imported_metadata_to_finalized_session(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)

    plan = _plan(tmp_path, evidence_id)

    assert plan.mode == "pilot"
    assert plan.manifest["matrix"]["performance_eligible"] is False
    assert plan.manifest["source_evidence"][0]["evidence_id"] == evidence_id
    assert plan.run_id.startswith("trialqa-distill-")
    assert "trialqa-row-0001" in plan.evidence[0].sensitive_literals


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"partition": "test"}, "train-partition"),
        ({"context_overrides": {"route": "other-route"}}, "run_context.route"),
        ({"active_loaded": True}, "skill_loaded|no skill was loaded"),
        ({"stats_errors": 1}, "recovered errors|executor errors"),
        ({"outcome_answer": "a different answer"}, "submitted_answer|trajectory final output"),
    ],
)
def test_plan_rejects_untrusted_or_non_donor_source(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TrialQADistillationError, NativeTrialQAEvidenceError), match=message):
        evidence_id = _import_evidence(tmp_path, **kwargs)
        _plan(tmp_path, evidence_id)


def test_full_mode_requires_exact_question_repeat_matrix(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)

    with pytest.raises(TrialQADistillationError, match="120 evidence"):
        build_distillation_plan(
            project_dir=tmp_path,
            namespace=NAMESPACE,
            evidence_ids=[evidence_id],
            work_dir=tmp_path / "runs",
            reference_repo=REFERENCE_REPO,
            routing_profile=ROUTING_PROFILE,
            proxy_url="http://127.0.0.1:18181/v1",
            expected_question_count=24,
            expected_repeats=5,
            mode="full",
        )


def _synthetic_full_donors(tmp_path: Path) -> tuple[DonorEvidence, ...]:
    donors: list[DonorEvidence] = []
    index = 0
    for question_index in range(24):
        group = f"trialqa-group-{question_index:02d}"
        for repeat_index in range(1, 6):
            evidence_id = f"native-{index:032x}"
            donors.append(
                DonorEvidence(
                    evidence_id=evidence_id,
                    path=tmp_path,
                    document={
                        "task": {
                            "question": f"Synthetic question {question_index}",
                            "n_repeats": 5,
                        }
                    },
                    manifest_sha256="0" * 64,
                    donor_run_id="common-donor-manifest",
                    question_group_key=group,
                    repeat_index=repeat_index,
                    role="success",
                    judge_result="correct",
                    observed_tools=frozenset({"trialqa_search"}),
                    sensitive_literals=(evidence_id,),
                )
            )
            index += 1
    return tuple(donors)


def test_full_mode_accepts_exact_24_by_5_and_rejects_malformed_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    donors = _synthetic_full_donors(tmp_path)
    by_id = {item.evidence_id: item for item in donors}
    monkeypatch.setattr(
        distiller_module,
        "_load_donor_evidence",
        lambda _store, evidence_id: by_id[evidence_id],
    )

    plan = build_distillation_plan(
        project_dir=tmp_path,
        namespace=NAMESPACE,
        evidence_ids=list(by_id),
        work_dir=tmp_path / "full-runs",
        reference_repo=REFERENCE_REPO,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        mode="full",
    )

    assert len(plan.evidence) == 120
    assert plan.manifest["matrix"]["performance_eligible"] is True
    assert len({item.question_group_key for item in plan.evidence}) == 24

    mixed_runs = list(donors)
    mixed_runs[-1] = replace(mixed_runs[-1], donor_run_id="other-donor-manifest")
    mixed_by_id = {item.evidence_id: item for item in mixed_runs}
    monkeypatch.setattr(
        distiller_module,
        "_load_donor_evidence",
        lambda _store, evidence_id: mixed_by_id[evidence_id],
    )
    with pytest.raises(TrialQADistillationError, match="one experiment manifest"):
        build_distillation_plan(
            project_dir=tmp_path,
            namespace=NAMESPACE,
            evidence_ids=list(mixed_by_id),
            work_dir=tmp_path / "mixed-runs",
            reference_repo=REFERENCE_REPO,
            routing_profile=ROUTING_PROFILE,
            proxy_url="http://127.0.0.1:18181/v1",
            mode="full",
        )

    malformed = list(donors)
    malformed[-1] = replace(
        malformed[-1],
        question_group_key=malformed[0].question_group_key,
        repeat_index=malformed[0].repeat_index,
    )
    malformed_by_id = {item.evidence_id: item for item in malformed}
    monkeypatch.setattr(
        distiller_module,
        "_load_donor_evidence",
        lambda _store, evidence_id: malformed_by_id[evidence_id],
    )
    with pytest.raises(TrialQADistillationError, match="duplicate donor pair"):
        build_distillation_plan(
            project_dir=tmp_path,
            namespace=NAMESPACE,
            evidence_ids=list(malformed_by_id),
            work_dir=tmp_path / "malformed-runs",
            reference_repo=REFERENCE_REPO,
            routing_profile=ROUTING_PROFILE,
            proxy_url="http://127.0.0.1:18181/v1",
            mode="full",
        )


def test_execute_validates_saves_and_activates_candidate_without_network(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    caller = _FakeCaller()

    result = execute_distillation(
        plan,
        caller=caller,
        stats_reader=_Stats(caller),
        activate=True,
    )

    assert [call.stage for call in caller.calls] == [
        "analyst",
        "question_merge",
        "final_merge",
    ]
    assert all(call.payload["model"] == DISTILLER_ROUTE for call in caller.calls)
    assert result.activated is True
    assert result.model_call_count == 3
    assert result.skill_path.is_file()
    skill = result.skill_path.read_text(encoding="utf-8")
    assert skill.startswith("---\nname: tooluniverse-trialqa\n")
    assert "NCT99999999" not in skill
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    assert (store.active_path / "tooluniverse-trialqa" / "SKILL.md").read_text() == skill
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["distillation_mode"] == "pilot"
    assert report["performance_eligible"] is False
    assert report["source_evidence_ids"] == [evidence_id]
    assert report["routing"]["attested_call_count"] == 3
    assert report["artifacts"]["raw_response_count"] == 3


def test_question_merge_canonicalizes_only_the_structural_group_key(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    caller = _WrongStructuralGroupCaller()

    execute_distillation(plan, caller=caller, stats_reader=_Stats(caller))

    question_call = next(call for call in caller.calls if call.stage == "question_merge")
    prompt = str(question_call.payload["messages"][1]["content"])
    assert "top-level structural question_group_key exactly" in prompt
    assert "group-ID prohibition applies only to free text" in prompt
    stage_path = plan.run_path / "questions" / f"{plan.evidence[0].question_group_key}.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    raw = json.loads(distiller_module._raw_response_path(stage_path).read_text(encoding="utf-8"))
    assert stage["output"]["question_group_key"] == plan.evidence[0].question_group_key
    assert "model-invented-group-key" not in json.dumps(stage["output"])
    assert "model-invented-group-key" in raw["result"]["content"]


def test_merge_prompts_spell_out_exact_leaf_schemas_and_wrapper_is_normalized() -> None:
    group = "trialqa-0001-a1b2c3d4e5f6"
    _system, question = distiller_module._question_prompt(
        group,
        [{"role": "success", "memory_items": [], "skill_patch": {"sections": []}}],
        [{"repeat_index": 1, "role": "success", "judge_result": "correct"}],
    )
    assert '"schema_version": "trace2skill_question_merge.v2"' in question
    assert '"source_patch_count": 1' in question
    assert '"rationale": "at least 15 characters"' in question
    assert '"fact": "at least 20 characters"' in question
    assert "Do not emit a\n  `trace2skill_question_merge.v2` wrapper key" in question

    question_call = ModelCall(
        stage="question_merge",
        key=group,
        payload={},
        input_sha256="d" * 64,
    )
    canonical_question = distiller_module._canonicalize_model_output(
        question_call,
        {
            "trace2skill_question_merge.v2": {
                "summary": "Reusable merge summary.",
            }
        },
    )
    assert canonical_question == {
        "schema_version": "trace2skill_question_merge.v2",
        "question_group_key": group,
        "summary": "Reusable merge summary.",
    }

    _system, final = distiller_module._final_prompt(
        [{"source_patch_count": 1, "summary": "Reusable merge summary."}]
    )
    assert '"schema_version": "trace2skill_merge.v2"' in final
    assert '"rules": [{"rule": "at least 20 characters"' in final
    assert '"impact": "at least 15 characters"' in final
    final_call = replace(question_call, stage="final_merge")
    canonical_final = distiller_module._canonicalize_model_output(
        final_call,
        {"trace2skill_merge.v2": {}},
    )
    assert canonical_final == {
        "schema_version": "trace2skill_merge.v2",
        "skill_name": NAMESPACE,
    }


def _compact_catalog() -> dict[str, object]:
    return {
        "schema_version": "trace2skill_merge.v2",
        "skill_name": NAMESPACE,
        "summary": "Resolve one trial, retrieve one targeted evidence slice, and answer.",
        "tool_rules": [
            {
                "tool_name": "trialqa_load_active_skill",
                "rules": [
                    {
                        "when": "starting a TrialQA task",
                        "rule": (
                            "Call the active-skill loader exactly once, then continue without "
                            "ever retrying it."
                        ),
                        "confidence": 0.95,
                        "source_patch_count": 5,
                    }
                ],
            },
            {
                "tool_name": "trialqa_search",
                "rules": [
                    {
                        "when": "resolving the trial identifier",
                        "rule": (
                            "Use at most 3 semantically distinct searches. Stop searching after "
                            "an exact title match, never invent an acronym expansion, and never "
                            "repeat the same search or arguments."
                        ),
                        "confidence": 0.95,
                        "source_patch_count": 5,
                    }
                ],
            },
            {
                "tool_name": "trialqa_get_study",
                "rules": [
                    {
                        "when": "retrieving the resolved trial record",
                        "rule": (
                            "Use the resolved NCT identifier to retrieve the one structured "
                            "study record needed by the question."
                        ),
                        "confidence": 0.9,
                        "source_patch_count": 5,
                    }
                ],
            },
        ],
        "workflow_rules": [
            {
                "rule": (
                    "After resolving the NCT identifier, call one question-specific getter; "
                    "use a second getter only when the required field is absent, then answer."
                ),
                "rationale": "A bounded retrieval path avoids redundant fixed-content calls.",
                "confidence": 0.95,
                "source_patch_count": 8,
            }
        ],
        "failure_modes": [],
        "gotchas": [],
    }


def _train_cached_catalog() -> dict[str, object]:
    catalog = _compact_catalog()
    catalog["workflow_rules"] = [
        {
            "rule": (
                "Use trialqa_search to resolve the NCT id, then use the field-specific getter "
                "matching intent: get_eligibility for criteria or thresholds, "
                "get_outcome_measures for outcome counts or timeFrames, get_descriptions for "
                "narrative or per-arm detail, and get_study for enrollment."
            ),
            "rationale": "Search metadata rarely contains the field needed for the answer.",
            "confidence": 0.9,
            "source_patch_count": 40,
        },
        {
            "rule": (
                "Once a tool response already contains every field needed to answer, extract "
                "and finalize immediately without repeating fixed-content calls."
            ),
            "rationale": "Repeated calls add cost without adding evidence.",
            "confidence": 0.88,
            "source_patch_count": 20,
        },
    ]
    catalog["gotchas"] = [
        {
            "fact": (
                "An empty result from one wrapper means its slice is silent, not that the "
                "datum is absent across disjoint slices other trialqa_* tools expose."
            ),
            "impact": (
                "Treating one silent endpoint as terminal misses fields held by another getter."
            ),
            "confidence": 0.8,
            "source_patch_count": 5,
        }
    ]
    return catalog


def _compact_source_plan(
    tmp_path: Path, *, tool_contract: distiller_module.ToolContract = "direct"
) -> distiller_module.CompactDistillationPlan:
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    questions = source / "questions"
    questions.mkdir(parents=True)
    run_id = "trialqa-distill-" + "c" * 32
    source_evidence = []
    source_ids = []
    question_entries = []
    for group_index in range(24):
        group = f"trialqa-{group_index:04d}-{'a' * 12}"
        for repeat in range(1, 6):
            evidence_id = f"native-{group_index * 5 + repeat:032x}"
            source_ids.append(evidence_id)
            source_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "manifest_sha256": "d" * 64,
                    "donor_run_id": "trialqa-donor-" + "e" * 20,
                    "question_group_key": group,
                    "repeat_index": repeat,
                    "role": "success",
                }
            )
        aggregate = {
            "schema_version": distiller_module.QUESTION_MERGE_SCHEMA,
            "question_group_key": group,
            "summary": "Five repeats support one bounded structured-search rule.",
            "source_patch_count": 5,
            "repeat_count": 5,
            "role_counts": {"success": 5},
            "judge_result_counts": {"correct": 5},
            "tool_rules": [
                {
                    "tool_name": "trialqa_load_active_skill",
                    "rule": (
                        "Call the active-skill loader exactly once, then continue without "
                        "ever retrying it."
                    ),
                    "when": "starting a TrialQA task",
                    "confidence": 0.9,
                    "source_patch_count": 5,
                },
                {
                    "tool_name": "trialqa_search",
                    "rule": "Inspect the structured title field before selecting the matching trial record.",
                    "when": "resolving the requested clinical trial",
                    "confidence": 0.9,
                    "source_patch_count": 5,
                },
                {
                    "tool_name": "trialqa_get_study",
                    "rule": (
                        "Retrieve the resolved study record and read only the field needed "
                        "for the current question."
                    ),
                    "when": "reading the resolved clinical trial record",
                    "confidence": 0.9,
                    "source_patch_count": 5,
                },
            ],
            "workflow_rules": [],
            "failure_modes": [],
            "gotchas": [],
        }
        path = questions / f"{group}.json"
        distiller_module._write_stage_artifact(
            path,
            {
                "schema_version": distiller_module.SCHEMA_VERSION,
                "stage": "question_merge",
                "key": group,
                "input_sha256": "f" * 64,
                "output": aggregate,
                "attestation": {
                    "stage": "question_merge",
                    "key": group,
                    "route_model": DISTILLER_ROUTE,
                    "upstream_model": DISTILLER_MODEL,
                    "request_id": f"question-request-{group_index}",
                    "usage": {"total_tokens": 100},
                },
            },
        )
        question_entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": f"sha256:{distiller_module._file_sha256(path)}",
                "size_bytes": path.stat().st_size,
            }
        )
    source_manifest = {
        "schema_version": distiller_module.SCHEMA_VERSION,
        "run_id": run_id,
        "namespace": NAMESPACE,
        "matrix": {
            "mode": "full",
            "performance_eligible": True,
            "expected_question_count": 24,
            "expected_repeats": 5,
        },
        "source_evidence": source_evidence,
    }
    _write_json(source / "run_manifest.json", source_manifest)
    final_path = source / "final_catalog.json"
    distiller_module._write_stage_artifact(
        final_path,
        {
            "schema_version": distiller_module.SCHEMA_VERSION,
            "stage": "final_merge",
            "key": run_id,
            "input_sha256": "e" * 64,
            "output": _train_cached_catalog(),
            "attestation": {
                "stage": "final_merge",
                "key": run_id,
                "route_model": DISTILLER_ROUTE,
                "upstream_model": DISTILLER_MODEL,
                "request_id": "source-final-request",
                "usage": {"total_tokens": 100},
            },
        },
    )
    analyst_entries = [
        {
            "path": f"analyst/{evidence_id}.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1,
        }
        for evidence_id in source_ids
    ]
    completion = {
        "schema_version": distiller_module.SCHEMA_VERSION,
        "run_id": run_id,
        "stage_artifacts": [
            *analyst_entries,
            *question_entries,
            {
                "path": "final_catalog.json",
                "sha256": f"sha256:{distiller_module._file_sha256(final_path)}",
                "size_bytes": final_path.stat().st_size,
            },
        ],
    }
    completion_path = source / "completion_manifest.json"
    _write_json(completion_path, completion)
    checks = {
        "all_evidence_native_and_content_validated": True,
        "all_evidence_train_donor": True,
        "all_executor_sessions_unskilled": True,
        "one_analyst_patch_per_evidence": True,
        "repeats_grouped_by_question": True,
        "distiller_route_only": True,
        "distiller_model_only": True,
        "routing_stats_accounted": True,
    }
    validation = {
        "schema_version": distiller_module.SCHEMA_VERSION,
        "run_id": run_id,
        "status": "passed",
        "performance_eligible": True,
        "source_evidence_ids": source_ids,
        "checks": checks,
        "artifacts": {
            "completion_manifest_sha256": (
                f"sha256:{distiller_module._file_sha256(completion_path)}"
            )
        },
    }
    _write_json(source / "candidate_validation.json", validation)
    return distiller_module.build_compact_distillation_plan(
        project_dir=project,
        namespace=NAMESPACE,
        work_dir=tmp_path / "runs",
        source_run=source,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        tool_contract=tool_contract,
    )


def test_compact_final_prompt_declares_hard_efficiency_contract() -> None:
    _system, prompt = distiller_module._compact_final_prompt(
        [{"source_patch_count": 5, "summary": "Reusable merge summary."}]
    )

    assert "at most 10 total rule leaves" in prompt
    assert "at most 3 semantically distinct searches" in prompt
    assert "never invent an acronym expansion" in prompt
    assert "one question-specific getter" in prompt
    assert "4,096 UTF-8 bytes" in prompt


def test_compact_skill_validation_accepts_bounded_control_flow() -> None:
    catalog = _compact_catalog()
    skill = distiller_module.render_skill_markdown(catalog)

    metrics = distiller_module.validate_compact_skill(catalog, skill)

    assert metrics["size_bytes"] <= 4096
    assert metrics["word_count"] <= 650
    assert metrics["rule_count"] == 4


def test_compact_transport_maps_paid_aliases_to_exact_meta_tool_calls() -> None:
    catalog = distiller_module.compact_final_catalog(_compact_catalog(), tool_contract="compact")
    skill = distiller_module.render_skill_markdown(catalog, tool_contract="compact")

    metrics = distiller_module.validate_compact_skill(catalog, skill, tool_contract="compact")

    assert catalog["tool_contract"] == "compact"
    transport = catalog["transport"]
    assert isinstance(transport, dict)
    assert transport["public_tools"] == [
        "trialqa_load_active_skill",
        "execute_tool",
        "grep_tools",
        "get_tool_info",
    ]
    mappings = transport["source_alias_mapping"]
    assert isinstance(mappings, list)
    assert len(mappings) == 9
    assert all(mapping["arguments_parameter"] == "arguments_json" for mapping in mappings)
    for source_alias, tool_name, arguments in distiller_module.COMPACT_TRIALQA_TOOL_MAP:
        assert source_alias in skill
        assert tool_name in skill
        for argument in arguments:
            assert argument in skill
    assert "call `execute_tool` directly" in skill
    assert "`arguments_json` string encoding one JSON object" in skill
    assert "Discovery is fallback-only" in skill
    assert "host-enforced" not in skill
    assert metrics["size_bytes"] <= 4096
    assert metrics["word_count"] <= 650


def test_compact_transport_matches_schema_safe_adapter_contract() -> None:
    mapped_targets = tuple(
        target for _alias, target, _arguments in distiller_module.COMPACT_TRIALQA_TOOL_MAP
    )
    execute_schema = adapter_module.TOOL_SPEC_BY_NAME["execute_tool"].input_schema
    info_schema = adapter_module.TOOL_SPEC_BY_NAME["get_tool_info"].input_schema

    assert mapped_targets == adapter_module.ALLOWED_EXECUTION_TOOL_NAMES
    assert set(execute_schema["properties"]) == {"tool_name", "arguments_json"}
    assert execute_schema["required"] == ["tool_name", "arguments_json"]
    assert info_schema["properties"]["tool_names"]["type"] == "array"


def test_compact_tool_contract_is_bound_into_plan_identity(tmp_path: Path) -> None:
    direct = _compact_source_plan(tmp_path)
    compact = distiller_module.build_compact_distillation_plan(
        project_dir=direct.project_dir,
        namespace=NAMESPACE,
        work_dir=tmp_path / "compact-runs",
        source_run=direct.source_run,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        tool_contract="compact",
    )

    assert direct.tool_contract == "direct"
    assert compact.tool_contract == "compact"
    assert direct.run_id != compact.run_id
    assert direct.manifest["tool_contract"] == "direct"
    assert compact.manifest["tool_contract"] == "compact"
    assert (
        compact.manifest["compact_policy"]["transport_adaptation"]
        == "source-alias-to-tooluniverse-compact-meta-tools-v1"
    )


def test_compact_skill_validation_rejects_placeholders_and_missing_budget() -> None:
    catalog = _compact_catalog()
    skill = distiller_module.render_skill_markdown(catalog)

    with pytest.raises(TrialQADistillationError, match="placeholder"):
        distiller_module.validate_compact_skill(
            catalog,
            skill.replace("at most 3", "<REDACTED_TASK_LITERAL>"),
        )
    with pytest.raises(TrialQADistillationError, match="search budget"):
        distiller_module.validate_compact_skill(
            catalog,
            skill.replace("at most 3 semantically distinct searches", "several searches"),
        )
    with pytest.raises(TrialQADistillationError, match="placeholder"):
        distiller_module.validate_compact_skill(
            catalog,
            skill.replace("exact title match", "REDACTED_TASK_LITERAL"),
        )


def test_compact_skill_validation_rejects_contradictory_unbounded_rules() -> None:
    catalog = _compact_catalog()
    cast_rules = catalog["workflow_rules"]
    assert isinstance(cast_rules, list)
    cast_rules.append(
        {
            "rule": (
                "Continue searching indefinitely and call every getter even after an exact "
                "title match."
            ),
            "rationale": "This intentionally contradicts the bounded workflow contract.",
            "confidence": 0.1,
            "source_patch_count": 1,
        }
    )
    skill = distiller_module.render_skill_markdown(catalog)

    with pytest.raises(TrialQADistillationError, match="contradictory"):
        distiller_module.validate_compact_skill(catalog, skill)


def test_compact_execution_makes_one_call_and_saves_bounded_candidate(
    tmp_path: Path,
) -> None:
    plan = _compact_source_plan(tmp_path)

    class CompactCaller(_FakeCaller):
        def __call__(self, call: ModelCall) -> ModelCallResult:
            self.calls.append(call)
            return ModelCallResult(
                content=json.dumps(_compact_catalog()),
                route_model=DISTILLER_ROUTE,
                upstream_model=DISTILLER_MODEL,
                request_id="compact-request-1",
                usage={"total_tokens": 100},
            )

    caller = CompactCaller()
    result = distiller_module.execute_compact_distillation(
        plan,
        caller=caller,
        stats_reader=_Stats(caller),
    )

    assert len(caller.calls) == 1
    assert caller.calls[0].stage == "final_merge"
    assert result.model_call_count == 1
    skill = result.skill_path.read_text(encoding="utf-8")
    assert len(skill.encode("utf-8")) <= 4096
    assert len(skill.split()) <= 650
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["distillation_mode"] == "compact-final-only"
    assert report["artifacts"]["rule_count"] == 3


def test_compact_transport_reuses_paid_raw_catalog_with_zero_model_calls(
    tmp_path: Path,
) -> None:
    direct_plan = _compact_source_plan(tmp_path)

    class PaidCaller(_FakeCaller):
        def __call__(self, call: ModelCall) -> ModelCallResult:
            self.calls.append(call)
            return ModelCallResult(
                content=json.dumps(_compact_catalog()),
                route_model=DISTILLER_ROUTE,
                upstream_model=DISTILLER_MODEL,
                request_id="paid-compact-request",
                usage={"total_tokens": 100},
            )

    paid_caller = PaidCaller()
    paid_result = distiller_module.execute_compact_distillation(
        direct_plan,
        caller=paid_caller,
        stats_reader=_Stats(paid_caller),
    )
    compact_plan = distiller_module.build_compact_distillation_plan(
        project_dir=direct_plan.project_dir,
        namespace=NAMESPACE,
        work_dir=tmp_path / "compact-transport-runs",
        source_run=direct_plan.source_run,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        paid_raw_run=direct_plan.run_path,
        tool_contract="compact",
    )

    def unexpected_call(_call: ModelCall) -> ModelCallResult:
        pytest.fail("transport adaptation must not make a model call")

    result = distiller_module.execute_compact_distillation(
        compact_plan,
        caller=unexpected_call,
        stats_reader=lambda: {
            "total_requests": 0,
            "total_errors": 0,
            "models": {},
        },
    )

    assert result.model_call_count == 0
    assert result.candidate_id != paid_result.candidate_id
    skill = result.skill_path.read_text(encoding="utf-8")
    assert "## Compact ToolUniverse contract" in skill
    assert "ClinicalTrials_search_studies" in skill
    assert "execute_tool" in skill
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["tool_contract"] == "compact"
    assert report["new_model_call_count"] == 0
    assert report["checks"]["tool_contract_bound"] is True
    assert report["checks"]["compact_transport_map_exact"] is True
    assert report["checks"]["paid_raw_response_reused_without_recall"] is True
    completion = json.loads(
        (compact_plan.run_path / "completion_manifest.json").read_text(encoding="utf-8")
    )
    assert completion["tool_contract"] == "compact"
    assert completion["new_model_call_count"] == 0
    assert completion["paid_raw_recovery"] == compact_plan.paid_raw_binding
    recovered = json.loads(
        (compact_plan.run_path / "final_catalog.raw-response.json").read_text(encoding="utf-8")
    )
    paid = json.loads(
        (direct_plan.run_path / "final_catalog.raw-response.json").read_text(encoding="utf-8")
    )
    assert recovered["result"] == paid["result"]
    assert recovered["recovered_from"] == compact_plan.paid_raw_binding


def test_source_final_catalog_transport_is_train_only_generic_and_zero_call(
    tmp_path: Path,
) -> None:
    source_plan = _compact_source_plan(tmp_path)
    plan = distiller_module.build_compact_distillation_plan(
        project_dir=source_plan.project_dir,
        namespace=NAMESPACE,
        work_dir=tmp_path / "source-final-transport-runs",
        source_run=source_plan.source_run,
        routing_profile=ROUTING_PROFILE,
        proxy_url="http://127.0.0.1:18181/v1",
        tool_contract="compact",
        transport_source_final_catalog=True,
    )

    def unexpected_call(_call: ModelCall) -> ModelCallResult:
        pytest.fail("cached final catalog transport must not make a model call")

    result = distiller_module.execute_compact_distillation(
        plan,
        caller=unexpected_call,
        stats_reader=lambda: {"total_requests": 0, "total_errors": 0, "models": {}},
    )

    assert plan.run_id.startswith("trialqa-transport-")
    assert plan.run_id != source_plan.run_id
    assert plan.manifest["pipeline"] == "cached-final-catalog/deterministic-compact-transport"
    assert plan.manifest["model_call_budget"] == 0
    assert plan.manifest["compact_policy"]["deterministic_pruning"] == (
        distiller_module.CACHED_CATALOG_TRANSPORT_MODE
    )
    assert plan.source_final_binding is not None
    assert result.model_call_count == 0

    skill = result.skill_path.read_text(encoding="utf-8")
    assert "field-specific getter" in skill
    assert "non-exhaustive" in skill
    assert "If the selected slice lacks the requested field" in skill
    assert "another relevant getter" in skill
    assert "Answer only when retrieved evidence directly supports every requested field" in skill
    assert "one question-specific getter" not in skill
    assert "Never exceed 5 operational" not in skill
    assert "PF-06463922" not in skill
    assert "NCT01970865" not in skill
    assert "10 mg" not in skill
    assert "dose" not in skill.lower()

    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["distillation_mode"] == distiller_module.CACHED_CATALOG_TRANSPORT_MODE
    assert report["new_model_call_count"] == 0
    assert report["source_evidence_ids"] == list(plan.source_evidence_ids)
    assert len(report["source_evidence_ids"]) == 120
    assert report["checks"]["train_only_source_provenance"] is True
    assert report["checks"]["source_final_catalog_hash_bound"] is True
    assert report["checks"]["zero_new_model_calls"] is True
    assert report["checks"]["generic_field_specific_getter_routing"] is True
    assert report["checks"]["evidence_sufficiency_fallback"] is True
    assert report["checks"]["non_exhaustive_routes_preserved"] is True
    assert report["routing"]["attested_call_count"] == 0
    assert report["routing"]["source_final_attestation"]["request_id"] == ("source-final-request")

    transported = json.loads((plan.run_path / "final_catalog.json").read_text(encoding="utf-8"))
    assert transported["stage"] == "catalog_transport"
    assert transported["provenance"]["new_model_call_count"] == 0
    assert transported["provenance"]["source_final_catalog"] == plan.source_final_binding


def test_source_final_catalog_transport_cli_never_constructs_model_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_plan = _compact_source_plan(tmp_path)
    monkeypatch.setattr(
        distiller_module,
        "LocalSwitchyardCaller",
        lambda _proxy: pytest.fail("source-final transport must stay local"),
    )

    return_code = main(
        [
            "compact-execute",
            "--project-dir",
            str(source_plan.project_dir),
            "--work-dir",
            str(tmp_path / "source-final-cli-runs"),
            "--source-run",
            str(source_plan.source_run),
            "--routing-profile",
            str(ROUTING_PROFILE),
            "--proxy-url",
            "http://127.0.0.1:18181/v1",
            "--tool-contract",
            "compact",
            "--transport-source-final-catalog",
        ]
    )

    assert return_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_call_count"] == 0
    assert output["run_id"].startswith("trialqa-transport-")


def test_development_layer_rule_is_conditional_generic_and_compact() -> None:
    parent = distiller_module.compact_cached_final_catalog(
        _train_cached_catalog(), tool_contract="compact"
    )

    catalog = distiller_module.layer_exposed_development_catalog(parent)
    skill = distiller_module.render_skill_markdown(catalog, tool_contract="compact")
    metrics = distiller_module.validate_compact_skill(catalog, skill, tool_contract="compact")

    assert metrics["size_bytes"] <= distiller_module.COMPACT_SKILL_MAX_BYTES
    assert "intervention starting-dose, regimen, or ordered arm/group questions" in skill
    assert "if the selected slice lacks direct support" in skill
    assert "trialqa_extract_adverse_events as a fallback evidence slice" in skill
    assert "Do not infer starting, lowest, or highest values from outcome timeFrames" in skill
    assert "trialqa_get_outcome_measures for outcome counts or timeFrames" in skill
    assert "all dose questions" not in skill.lower()
    assert "PF-06463922" not in skill
    assert "NCT01970865" not in skill
    assert "10 mg" not in skill
    assert "25 mg" not in skill


def test_mechanism_repair_is_generic_compact_and_preserves_q7_rule() -> None:
    train_parent = distiller_module.compact_cached_final_catalog(
        _train_cached_catalog(), tool_contract="compact"
    )
    parent = distiller_module.layer_exposed_development_catalog(train_parent)
    parent_q7 = next(
        rule
        for rule in parent["workflow_rules"]
        if "trialqa_extract_adverse_events" in rule["rule"]
    )

    repaired = distiller_module.layer_exposed_mechanism_repair_catalog(parent)
    skill = distiller_module.render_skill_markdown(repaired, tool_contract="compact")
    metrics = distiller_module.validate_compact_skill(repaired, skill, tool_contract="compact")
    repaired_q7 = next(
        rule
        for rule in repaired["workflow_rules"]
        if "trialqa_extract_adverse_events" in rule["rule"]
    )
    repaired_search = next(
        group for group in repaired["tool_rules"] if group["tool_name"] == "trialqa_search"
    )["rules"][0]
    repaired_field_route = next(
        rule
        for rule in repaired["workflow_rules"]
        if "trialqa_get_eligibility first" in rule["rule"]
    )

    assert metrics["size_bytes"] <= distiller_module.COMPACT_SKILL_MAX_BYTES
    assert metrics["word_count"] <= distiller_module.COMPACT_SKILL_MAX_WORDS
    assert metrics["rule_count"] == 4
    assert repaired_q7 == parent_q7
    assert repaired_search["source_patch_count"] == 1
    assert repaired_field_route["source_patch_count"] == 1
    assert "never search after resolution" in skill
    assert "never repeat a query or arguments" in skill
    assert "trialqa_get_eligibility first" in skill
    assert "never trialqa_get_study" in skill
    assert "trialqa_get_outcome_measures for outcome counts or timeFrames" in skill
    assert "trialqa_get_descriptions for narrative or per-arm detail" in skill
    assert "trialqa_get_study for enrollment or structured summary fields" in skill
    assert all(
        literal not in skill
        for literal in (
            "MK-2118-001",
            "NCT03249792",
            "12 weeks",
            "4 weeks",
            "NCT01693562",
            "Dose 6",
            "NCT01970865",
            "10 mg",
            "25 mg",
        )
    )
    assert re.search(r"\bNCT\d{8}\b", skill) is None


def test_search_discipline_repair_changes_only_the_search_rule() -> None:
    train_parent = distiller_module.compact_cached_final_catalog(
        _train_cached_catalog(), tool_contract="compact"
    )
    development = distiller_module.layer_exposed_development_catalog(train_parent)
    parent = distiller_module.layer_exposed_mechanism_repair_catalog(development)

    repaired = distiller_module.layer_exposed_search_discipline_repair_catalog(parent)
    skill = distiller_module.render_skill_markdown(repaired, tool_contract="compact")
    metrics = distiller_module.validate_compact_skill(repaired, skill, tool_contract="compact")

    assert repaired["workflow_rules"] == parent["workflow_rules"]
    assert repaired["failure_modes"] == parent["failure_modes"]
    assert repaired["gotchas"] == parent["gotchas"]
    assert repaired["tool_rules"][0]["tool_name"] == parent["tool_rules"][0]["tool_name"]
    assert repaired["tool_rules"][0]["rules"] != parent["tool_rules"][0]["rules"]
    assert metrics["size_bytes"] <= distiller_module.COMPACT_SKILL_MAX_BYTES
    assert metrics["rule_count"] == 4
    assert "Search once" in skill
    assert "stop searching even if the topic is absent" in skill
    assert "call the field getter" in skill
    assert "Never repeat a query or arguments" in skill
    assert re.search(r"\bNCT\d{8}\b", skill) is None


def test_identifier_terminal_repair_changes_only_the_search_rule() -> None:
    train_parent = distiller_module.compact_cached_final_catalog(
        _train_cached_catalog(), tool_contract="compact"
    )
    development = distiller_module.layer_exposed_development_catalog(train_parent)
    mechanism = distiller_module.layer_exposed_mechanism_repair_catalog(development)
    parent = distiller_module.layer_exposed_search_discipline_repair_catalog(mechanism)

    repaired = distiller_module.layer_exposed_identifier_terminal_repair_catalog(parent)
    skill = distiller_module.render_skill_markdown(repaired, tool_contract="compact")
    metrics = distiller_module.validate_compact_skill(repaired, skill, tool_contract="compact")

    assert repaired["workflow_rules"] == parent["workflow_rules"]
    assert repaired["failure_modes"] == parent["failure_modes"]
    assert repaired["gotchas"] == parent["gotchas"]
    assert repaired["tool_rules"][0]["tool_name"] == parent["tool_rules"][0]["tool_name"]
    assert repaired["tool_rules"][0]["rules"] != parent["tool_rules"][0]["rules"]
    assert metrics["size_bytes"] <= distiller_module.COMPACT_SKILL_MAX_BYTES
    assert metrics["word_count"] <= distiller_module.COMPACT_SKILL_MAX_WORDS
    assert metrics["rule_count"] == 4
    assert "One NCT result with id in its title is an exact title match" in skill
    assert "stop searching; call field getter now" in skill
    assert "No answer-term searches" in skill
    assert "Never repeat a query or arguments" in skill
    assert re.search(r"\bNCT\d{8}\b", skill) is None


def _development_evidence(
    tmp_path: Path,
    *,
    evidence_id: str,
    role: str,
    repeat_index: int,
    direct_support_observed: bool,
) -> distiller_module.DevelopmentEvidence:
    task_id = f"trialqa-0007-aabbccddeeff-r{repeat_index:03d}-treatment"
    return distiller_module.DevelopmentEvidence(
        evidence_id=evidence_id,
        path=tmp_path / evidence_id,
        document={
            "task": {
                "id": task_id,
                "question_group_key": "trialqa-0007-aabbccddeeff",
                "repeat_index": repeat_index,
                "question": "Which starting regimen is directly supported?",
            },
            "outcome": {"submitted_answer": "A directly supported regimen."},
            "events": [],
        },
        manifest_sha256="a" * 64,
        question_group_key="trialqa-0007-aabbccddeeff",
        repeat_index=repeat_index,
        role=cast(Any, role),
        direct_support_observed=direct_support_observed,
    )


def test_development_verdict_rejects_started_primary_capture(tmp_path: Path) -> None:
    parent_binding = {
        "candidate_id": "trialqa-parent",
        "manifest_sha256": "sha256:" + "1" * 64,
        "skill_sha256": "sha256:" + "2" * 64,
    }
    support = _development_evidence(
        tmp_path,
        evidence_id="native-" + "1" * 32,
        role="support",
        repeat_index=1,
        direct_support_observed=True,
    )
    failure = _development_evidence(
        tmp_path,
        evidence_id="native-" + "2" * 32,
        role="failure",
        repeat_index=2,
        direct_support_observed=False,
    )
    groups = tuple(
        "trialqa-0007-aabbccddeeff" if index == 7 else f"trialqa-{index:04d}-aabbccddeeff"
        for index in range(96)
    )
    descriptive = {"manifest_id": "descriptive"}
    primary = {"manifest_id": "primary"}
    verdict = {
        "schema_version": "switchyard.trialqa_exposed_regression_verdict.v1",
        "decision": "kill",
        "performance_eligible": False,
        "source_attestation_current": True,
        "candidate": parent_binding,
        "manifest": {
            "manifest_id": "descriptive",
            "manifest_sha256": "sha256:" + "3" * 64,
        },
        "primary_evaluation": {
            "manifest_id": "primary",
            "manifest_sha256": "sha256:" + "4" * 64,
            "capture_started": False,
        },
        "scope": {
            "question_start": 7,
            "question_limit": 1,
            "repeat_limit": 2,
            "heldout_classification": "exposed-heldout-quarantine",
            "scope_attestation_sha256": "sha256:" + "5" * 64,
        },
        "policy": {
            "name": "exposed-mechanism-regression-v1",
            "required_treatment_correct_repeats": 2,
            "required_treatment_direct_support_repeats": 2,
        },
        "results": [
            {
                "task_id": support.document["task"]["id"],
                "score": 1.0,
                "evidence_id": support.evidence_id,
                "direct_support_operation": "extract_clinical_trial_adverse_events",
                "direct_support_observed": True,
                "generation_sha256": "sha256:" + "6" * 64,
                "codex_events_sha256": "sha256:" + "7" * 64,
                "result_sha256": "sha256:" + "8" * 64,
            },
            {
                "task_id": failure.document["task"]["id"],
                "score": 0.0,
                "evidence_id": failure.evidence_id,
                "direct_support_operation": "extract_clinical_trial_adverse_events",
                "direct_support_observed": False,
                "generation_sha256": "sha256:" + "9" * 64,
                "codex_events_sha256": "sha256:" + "a" * 64,
                "result_sha256": "sha256:" + "b" * 64,
            },
        ],
        "summary": {
            "treatment_correct_repeats": 1,
            "treatment_direct_support_repeats": 1,
            "required_repeats": 2,
        },
    }
    distiller_module._validate_regression_verdict(
        verdict,
        parent_binding=parent_binding,
        failure=failure,
        support=support,
        descriptive_manifest=descriptive,
        descriptive_sha256="3" * 64,
        primary_manifest=primary,
        primary_sha256="4" * 64,
        descriptive_groups=groups,
        primary_groups=groups[8:],
    )
    verdict["primary_evaluation"]["capture_started"] = True

    with pytest.raises(TrialQADistillationError, match="untouched primary"):
        distiller_module._validate_regression_verdict(
            verdict,
            parent_binding=parent_binding,
            failure=failure,
            support=support,
            descriptive_manifest=descriptive,
            descriptive_sha256="3" * 64,
            primary_manifest=primary,
            primary_sha256="4" * 64,
            descriptive_groups=groups,
            primary_groups=groups[8:],
        )


def test_execute_development_layer_is_zero_call_and_does_not_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    parent = distiller_module.compact_cached_final_catalog(
        _train_cached_catalog(), tool_contract="compact"
    )
    support = _development_evidence(
        tmp_path,
        evidence_id="native-" + "e" * 32,
        role="support",
        repeat_index=1,
        direct_support_observed=True,
    )
    failure = _development_evidence(
        tmp_path,
        evidence_id="native-" + "f" * 32,
        role="failure",
        repeat_index=2,
        direct_support_observed=False,
    )
    train_ids = tuple(f"native-{index:032x}" for index in range(120))
    strata = {
        "train_base": {"source_evidence_ids": list(train_ids), "evidence_count": 120},
        "exposed_development": {
            "performance_eligible": False,
            "failure": {"evidence_id": failure.evidence_id},
            "support": {"evidence_id": support.evidence_id},
        },
        "evaluation_scope": {
            "excluded_question_start": 0,
            "excluded_question_count": 8,
            "primary_question_start": 8,
            "primary_question_count": 88,
            "capture_started": False,
        },
    }
    run_id = "trialqa-development-" + "c" * 32
    plan = distiller_module.DevelopmentLayerPlan(
        run_id=run_id,
        run_path=tmp_path / "runs" / run_id,
        namespace=NAMESPACE,
        project_dir=project,
        parent_candidate_id="trialqa-parent",
        parent_candidate_path=tmp_path / "parent",
        parent_manifest_sha256="1" * 64,
        parent_skill_sha256="2" * 64,
        parent_catalog=parent,
        parent_catalog_binding={"sha256": "sha256:" + "3" * 64},
        train_evidence_ids=train_ids,
        failure_evidence=failure,
        support_evidence=support,
        verdict_path=tmp_path / "verdict.json",
        verdict={},
        verdict_binding={"sha256": "sha256:" + "4" * 64},
        descriptive_manifest_path=tmp_path / "descriptive.json",
        descriptive_manifest_binding={"sha256": "sha256:" + "5" * 64},
        primary_manifest_path=tmp_path / "primary.json",
        primary_manifest_binding={"sha256": "sha256:" + "6" * 64},
        manifest={
            "run_id": run_id,
            "schema_version": distiller_module.SCHEMA_VERSION,
            "mode": distiller_module.DEVELOPMENT_LAYER_MODE,
            "model_call_budget": 0,
            "provenance_strata": strata,
        },
    )
    monkeypatch.setattr(distiller_module, "build_development_layer_plan", lambda **_kw: plan)
    monkeypatch.setattr(distiller_module, "_copy_development_evidence", lambda *_a: None)
    monkeypatch.setattr(
        distiller_module,
        "LocalSwitchyardCaller",
        lambda _proxy: pytest.fail("development layer must never construct a model client"),
    )

    result = distiller_module.execute_development_layer(plan)

    assert result.model_call_count == 0
    assert result.activated is False
    assert not SkillDistillationStore(NAMESPACE, project).active_path.joinpath("SKILL.md").exists()
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["new_model_call_count"] == 0
    assert report["full_96_performance_eligible"] is False
    assert report["checks"]["conditional_adverse_event_fallback"] is True
    assert report["provenance_strata"] == strata


def test_paid_raw_compact_cli_is_local_and_needs_no_model_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    direct_plan = _compact_source_plan(tmp_path)

    class PaidCaller(_FakeCaller):
        def __call__(self, call: ModelCall) -> ModelCallResult:
            self.calls.append(call)
            return ModelCallResult(
                content=json.dumps(_compact_catalog()),
                route_model=DISTILLER_ROUTE,
                upstream_model=DISTILLER_MODEL,
                request_id="paid-cli-request",
            )

    paid_caller = PaidCaller()
    distiller_module.execute_compact_distillation(
        direct_plan,
        caller=paid_caller,
        stats_reader=_Stats(paid_caller),
    )
    monkeypatch.setattr(
        distiller_module,
        "LocalSwitchyardCaller",
        lambda _proxy: pytest.fail("paid raw CLI must not construct a model client"),
    )

    return_code = main(
        [
            "compact-execute",
            "--project-dir",
            str(direct_plan.project_dir),
            "--work-dir",
            str(tmp_path / "cli-compact-runs"),
            "--source-run",
            str(direct_plan.source_run),
            "--paid-raw-run",
            str(direct_plan.run_path),
            "--routing-profile",
            str(ROUTING_PROFILE),
            "--proxy-url",
            "http://127.0.0.1:18181/v1",
            "--tool-contract",
            "compact",
        ]
    )

    assert return_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_call_count"] == 0


def test_compact_execution_rejects_manually_weakened_source_bindings(
    tmp_path: Path,
) -> None:
    plan = _compact_source_plan(tmp_path)
    weakened = replace(plan, source_bindings=())

    with pytest.raises(TrialQADistillationError, match="re-attested source"):
        distiller_module.execute_compact_distillation(
            weakened,
            caller=lambda _call: pytest.fail("invalid plan made a paid call"),
            stats_reader=lambda: {"total_requests": 0, "total_errors": 0, "models": {}},
        )


def test_final_merge_repairs_only_observed_array_closure_syntax() -> None:
    malformed = (
        '{"workflow_rules":[{"rule":"a"}]},{"rule":"b"}]}],'
        '"failure_modes":[{"trigger":"a"}]},{"trigger":"b"}]}],'
        '"gotchas":[{"fact":"a"}]},{"fact":"b"}]}]}'
    )

    assert distiller_module._extract_json(malformed, "final_merge/run-id") == {
        "workflow_rules": [{"rule": "a"}, {"rule": "b"}],
        "failure_modes": [{"trigger": "a"}, {"trigger": "b"}],
        "gotchas": [{"fact": "a"}, {"fact": "b"}],
    }
    with pytest.raises(TrialQADistillationError, match="invalid JSON"):
        distiller_module._extract_json(malformed, "analyst/evidence-id")
    with pytest.raises(TrialQADistillationError, match="invalid JSON"):
        distiller_module._extract_json('{"workflow_rules":[}', "final_merge/run-id")


def test_final_merge_repair_does_not_replace_tokens_inside_strings() -> None:
    text = '"]},{\\"rule\\"" ]},{"rule"'

    repaired, count = distiller_module._replace_outside_json_strings(text, ']},{"rule"', ',{"rule"')

    assert repaired == '"]},{\\"rule\\"" ,{"rule"'
    assert count == 1


def test_error_donor_produces_diagnosed_error_patch(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path, score=0.0)
    plan = _plan(tmp_path, evidence_id)
    assert plan.evidence[0].role == "error"
    assert plan.evidence[0].judge_result == "incorrect"
    caller = _FakeCaller()

    result = execute_distillation(plan, caller=caller, stats_reader=_Stats(caller))

    artifact = json.loads(
        (plan.run_path / "analyst" / f"{evidence_id}.json").read_text(encoding="utf-8")
    )
    assert artifact["output"]["role"] == "error"
    assert artifact["output"]["diagnosis"]["causal_trace_steps"]
    assert result.candidate_path.is_dir()


def test_analyst_task_identifiers_are_sanitized_before_merge(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    caller = _LeakyAnalystCaller()

    result = execute_distillation(plan, caller=caller, stats_reader=_Stats(caller))

    analyst_artifact = (plan.run_path / "analyst" / f"{evidence_id}.json").read_text()
    assert "NCT99999999" not in analyst_artifact
    assert "trialqa-row-0001" not in analyst_artifact
    assert "The record identifier is in identificationModule.nctId." not in analyst_artifact
    output = json.loads(analyst_artifact)["output"]
    memory_text = json.dumps(output["memory_items"])
    assert "The submitted answer matches the donor ideal." not in memory_text
    assert evidence_id not in memory_text
    assert "<REDACTED_TRIAL_ID>" in analyst_artifact
    assert "NCT99999999" not in result.skill_path.read_text()


def test_short_gold_answer_is_redacted_without_corrupting_schema_version() -> None:
    value = {
        "schema_version": "trace2skill_patch.v2",
        "source_patch_count": 1,
        "confidence": 0.1,
        "rule": "The task-specific answer was 1, which must not become a rule.",
    }

    sanitized = _sanitize(value, ["1"])

    assert sanitized["schema_version"] == "trace2skill_patch.v2"
    assert sanitized["source_patch_count"] == 1
    assert sanitized["confidence"] == 0.1
    assert sanitized["rule"] == (
        "The task-specific answer was <REDACTED_TASK_LITERAL>, which must not become a rule."
    )
    _assert_no_sensitive(sanitized, ["1"], "short-answer regression")

    version_like = _sanitize(
        {
            "schema_version": "trace2skill_patch.v2",
            "rule": "The source answer v2 must not appear in a reusable rule.",
        },
        ["v2"],
    )
    assert version_like["schema_version"] == "trace2skill_patch.v2"
    assert "v2" not in version_like["rule"]


def test_resume_reuses_all_stage_artifacts_and_candidate(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    first = _FakeCaller()
    original = execute_distillation(plan, caller=first, stats_reader=_Stats(first))
    resumed = _FakeCaller()

    second = execute_distillation(
        plan,
        caller=resumed,
        stats_reader=_Stats(resumed),
        resume=True,
    )

    assert resumed.calls == []
    assert second.model_call_count == 0
    assert second.candidate_id == original.candidate_id
    assert second.candidate_path == original.candidate_path


def test_invalid_model_output_is_persisted_and_resume_does_not_recall(
    tmp_path: Path,
) -> None:
    call = ModelCall(
        stage="question_merge",
        key="trusted-group",
        payload={"model": DISTILLER_ROUTE},
        input_sha256="a" * 64,
    )
    artifact_path = tmp_path / "question.json"
    calls = 0

    def invalid_caller(_call: ModelCall) -> ModelCallResult:
        nonlocal calls
        calls += 1
        return ModelCallResult(
            content="this is not JSON",
            route_model=DISTILLER_ROUTE,
            upstream_model=DISTILLER_MODEL,
            request_id="request-invalid-output",
            usage={"total_tokens": 17},
        )

    with pytest.raises(TrialQADistillationError, match="returned no JSON object"):
        distiller_module._call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=invalid_caller,
            resume=False,
            validator=lambda value: value,
        )

    raw_path = distiller_module._raw_response_path(artifact_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["call"] == {
        "stage": "question_merge",
        "key": "trusted-group",
        "input_sha256": "a" * 64,
    }
    assert raw["result"] == {
        "content": "this is not JSON",
        "route_model": DISTILLER_ROUTE,
        "upstream_model": DISTILLER_MODEL,
        "request_id": "request-invalid-output",
        "usage": {"total_tokens": 17},
    }
    assert distiller_module._integrity_path(raw_path).is_file()

    with pytest.raises(TrialQADistillationError, match="returned no JSON object"):
        distiller_module._call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=lambda _call: pytest.fail("resume repeated a paid model call"),
            resume=True,
            validator=lambda value: value,
        )
    assert calls == 1


def test_resume_validates_raw_result_and_writes_stage_without_recall(
    tmp_path: Path,
) -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "1" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="b" * 64,
    )
    artifact_path = tmp_path / "analyst.json"
    result = ModelCallResult(
        content='{"value":"locally valid"}',
        route_model=DISTILLER_ROUTE,
        upstream_model=DISTILLER_MODEL,
        request_id="request-valid-raw",
        usage={"total_tokens": 11},
    )

    with pytest.raises(RuntimeError, match="synthetic validation interruption"):
        distiller_module._call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=lambda _call: result,
            resume=False,
            validator=lambda _value: (_ for _ in ()).throw(
                RuntimeError("synthetic validation interruption")
            ),
        )

    output, attestation, called = distiller_module._call_or_resume(
        call=call,
        artifact_path=artifact_path,
        caller=lambda _call: pytest.fail("resume repeated a paid model call"),
        resume=True,
        validator=lambda value: value,
    )

    assert output == {
        "schema_version": "trace2skill_patch.v2",
        "value": "locally valid",
        "source_task_name": call.key,
        "skill_patch": {"target": "tooluniverse-trialqa/SKILL.md", "sections": []},
    }
    assert attestation["request_id"] == "request-valid-raw"
    assert called is False
    stage = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert stage["raw_response"]["artifact"] == "analyst.raw-response.json"
    raw = json.loads(distiller_module._raw_response_path(artifact_path).read_text(encoding="utf-8"))
    assert raw["result"]["content"] == '{"value":"locally valid"}'


def test_resume_rejects_tampered_raw_result_without_recall(tmp_path: Path) -> None:
    call = ModelCall(
        stage="analyst",
        key="native-" + "2" * 32,
        payload={"model": DISTILLER_ROUTE},
        input_sha256="c" * 64,
    )
    artifact_path = tmp_path / "analyst.json"
    result = ModelCallResult(
        content='{"value":"original"}',
        route_model=DISTILLER_ROUTE,
        upstream_model=DISTILLER_MODEL,
        request_id="request-tamper",
    )
    with pytest.raises(RuntimeError, match="stop after persistence"):
        distiller_module._call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=lambda _call: result,
            resume=False,
            validator=lambda _value: (_ for _ in ()).throw(RuntimeError("stop after persistence")),
        )
    raw_path = distiller_module._raw_response_path(artifact_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["result"]["content"] = '{"value":"tampered"}'
    _write_json(raw_path, raw)

    with pytest.raises(TrialQADistillationError, match="integrity mismatch"):
        distiller_module._call_or_resume(
            call=call,
            artifact_path=artifact_path,
            caller=lambda _call: pytest.fail("tampered resume repeated a paid call"),
            resume=True,
            validator=lambda value: value,
        )


def test_resume_rejects_tampered_final_artifact_even_with_rewritten_sidecar(
    tmp_path: Path,
) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    first = _FakeCaller()
    execute_distillation(plan, caller=first, stats_reader=_Stats(first))
    artifact_path = plan.run_path / "final_catalog.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["output"]["summary"] = "Tampered but structurally valid summary text."
    _write_json(artifact_path, artifact)
    payload = artifact_path.read_bytes()
    _write_json(
        plan.run_path / "final_catalog.integrity.json",
        {
            "schema_version": "switchyard.trialqa_native_distillation.v1",
            "artifact": "final_catalog.json",
            "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
            "size_bytes": len(payload),
        },
    )
    resumed = _FakeCaller()

    with pytest.raises(TrialQADistillationError, match="immutable distillation artifact conflict"):
        execute_distillation(
            plan,
            caller=resumed,
            stats_reader=_Stats(resumed),
            resume=True,
        )


def test_resume_quarantines_incomplete_stage_and_reuses_raw_with_current_stats(
    tmp_path: Path,
) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)

    class FailAfterAnalyst(_FakeCaller):
        def __call__(self, call: ModelCall) -> ModelCallResult:
            if self.calls:
                raise RuntimeError("synthetic interruption")
            return super().__call__(call)

    interrupted = FailAfterAnalyst()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute_distillation(
            plan,
            caller=interrupted,
            stats_reader=_Stats(interrupted),
        )
    (plan.run_path / "analyst" / f"{evidence_id}.integrity.json").unlink()
    resumed = _FakeCaller()

    result = execute_distillation(
        plan,
        caller=resumed,
        stats_reader=_Stats(resumed),
        resume=True,
    )

    assert [call.stage for call in resumed.calls] == ["question_merge", "final_merge"]
    assert result.model_call_count == 2
    report = json.loads(result.validation_report_path.read_text())
    assert report["artifacts"]["recovery_artifact_count"] == 1
    assert report["artifacts"]["raw_response_count"] == 3
    assert list((plan.run_path / "analyst").glob("*.orphan-*"))


def test_wrong_distiller_model_fails_before_candidate_save(tmp_path: Path) -> None:
    evidence_id = _import_evidence(tmp_path)
    plan = _plan(tmp_path, evidence_id)
    caller = _FakeCaller()

    def wrong_model(call: ModelCall) -> ModelCallResult:
        result = caller(call)
        return ModelCallResult(
            content=result.content,
            route_model=result.route_model,
            upstream_model="some-other-model",
            request_id=result.request_id,
        )

    with pytest.raises(TrialQADistillationError, match="route/model attestation"):
        execute_distillation(plan, caller=wrong_model, stats_reader=_Stats(caller))
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    assert list(store.candidates_path.iterdir()) == []


def test_plan_cli_never_constructs_or_calls_model_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence_id = _import_evidence(tmp_path)

    result = main(
        [
            "plan",
            "--project-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "runs"),
            "--reference-repo",
            str(REFERENCE_REPO),
            "--routing-profile",
            str(ROUTING_PROFILE),
            "--proxy-url",
            "http://127.0.0.1:18181/v1",
            "--evidence-id",
            evidence_id,
            "--pilot",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["matrix"]["mode"] == "pilot"
    assert not (tmp_path / "runs").exists()

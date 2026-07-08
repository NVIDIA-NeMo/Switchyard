# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for native-session TrialQA evidence normalization."""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from switchyard.lib.skill_distillation_native import (
    NativeTrialQAEvidenceConflictError,
    NativeTrialQAEvidenceError,
    import_native_trialqa_evidence,
    validate_native_trialqa_evidence_directory,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore

NAMESPACE = "tooluniverse-trialqa"
SESSION_ID = "codex-session-1"
EXECUTOR_MODEL = "nvidia/nvidia/nemotron-3-ultra"
EXECUTOR_ROUTE = "sd-executor"
CANDIDATE_ID = "candidate-evaluation-1"
CANDIDATE_MANIFEST = "sha256:" + "a" * 64
CANDIDATE_SKILL = "sha256:" + "b" * 64


def _openai_transport(
    total_requests: int,
    *,
    retries: int = 0,
    charges: int = 0,
    unpriced: int = 0,
    prompt: int = 0,
    completion: int = 0,
) -> dict[str, object]:
    return {
        "physical_attempts": total_requests + retries,
        "null_eof_retries": retries,
        "retry_usage_charges": charges,
        "unpriced_null_eof_retries": unpriced,
        "retry_token_sensitivity": {
            "prompt": prompt,
            "completion": completion,
            "cached": 0,
            "cache_creation": 0,
            "reasoning": 0,
            "total": prompt + completion,
        },
    }


def _tool_call() -> dict:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query":"BRCA1"}'},
    }


def _turns() -> list[dict]:
    question = {"role": "user", "content": "Find the BRCA1 chromosome."}
    assistant_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call()],
    }
    return [
        {
            "schema_version": 1,
            "session_id": SESSION_ID,
            "turn_index": 0,
            "recorded_at": "2026-07-06T10:00:00Z",
            "served_model": EXECUTOR_MODEL,
            "active_skill_version": CANDIDATE_ID,
            "active_skill_candidate_id": CANDIDATE_ID,
            "active_skill_manifest_sha256": CANDIDATE_MANIFEST,
            "request": {"model": EXECUTOR_ROUTE, "messages": [question]},
            "response": {
                "choices": [
                    {
                        "message": assistant_call,
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
        {
            "schema_version": 1,
            "session_id": SESSION_ID,
            "turn_index": 1,
            "recorded_at": "2026-07-06T10:00:01Z",
            "served_model": EXECUTOR_MODEL,
            "active_skill_version": CANDIDATE_ID,
            "active_skill_candidate_id": CANDIDATE_ID,
            "active_skill_manifest_sha256": CANDIDATE_MANIFEST,
            "request": {
                "model": EXECUTOR_ROUTE,
                "messages": [
                    question,
                    assistant_call,
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "search",
                        "content": '{"chromosome":"17"}',
                    },
                ],
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "BRCA1 is on chromosome 17.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            "routing": {"strategy": "passthrough"},
        },
    ]


def _create_session(
    project_dir: Path,
    *,
    status: str = "completed",
    turns: list[dict] | None = None,
) -> tuple[SkillDistillationStore, Path]:
    store = SkillDistillationStore(NAMESPACE, project_dir)
    session_dir = store.sessions_path / SESSION_ID
    session_dir.mkdir()
    turn_rows = _turns() if turns is None else turns
    turns_content = "".join(
        json.dumps(turn, ensure_ascii=False, sort_keys=True) + "\n" for turn in turn_rows
    ).encode("utf-8")
    (session_dir / "turns.jsonl").write_bytes(turns_content)
    (session_dir / "stats.json").write_text(
        json.dumps(
            {
                "total_requests": len(turn_rows),
                "total_errors": 0,
                "models": {
                    EXECUTOR_MODEL: {
                        "calls": len(turn_rows),
                        "errors": 0,
                    }
                },
                "classifier": {"total_requests": 0, "total_errors": 0},
                "planner": {"total_requests": 0, "total_errors": 0},
                "openai_transport": _openai_transport(len(turn_rows)),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    session = {
        "schema_version": 1,
        "session_id": SESSION_ID,
        "namespace": NAMESPACE,
        "launch_target": "codex",
        "display_model": EXECUTOR_ROUTE,
        "strategy_summary": f"passthrough: {EXECUTOR_MODEL}",
        "started_at": "2026-07-06T10:00:00Z",
        "ended_at": "2026-07-06T10:00:02Z",
        "status": status,
        "exit_code": 0 if status == "completed" else 1,
        "turn_count": len(turn_rows),
        "trajectory_sha256": f"sha256:{hashlib.sha256(turns_content).hexdigest()}",
        "run_context": {
            "task_id": "trialqa-17",
            "row_id": "17",
            "question_group_key": "row-0017",
            "partition": "test",
            "condition": "skilled",
            "repeat_index": 1,
            "n_repeats": 3,
            "manifest_id": "demo-run-1",
            "phase": "evaluation",
            "executor_model": EXECUTOR_MODEL,
            "route": EXECUTOR_ROUTE,
            "skill_loaded": True,
            "candidate_id": CANDIDATE_ID,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST,
            "candidate_skill_sha256": CANDIDATE_SKILL,
        },
        "active_skill": {
            "loaded": True,
            "candidate_id": CANDIDATE_ID,
            "manifest_sha256": CANDIDATE_MANIFEST,
            "skill_sha256": CANDIDATE_SKILL,
        },
    }
    (session_dir / "session.json").write_text(
        json.dumps(session, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return store, session_dir


def _task() -> dict:
    return {
        "id": "trialqa-17",
        "question_id": "17",
        "row_id": "17",
        "question": "Find the BRCA1 chromosome.",
        "question_group_key": "row-0017",
        "task_name": "trialqa-17",
        "condition": "skilled",
        "partition": "test",
        "repeat_index": 1,
        "n_repeats": 3,
    }


def _outcome() -> dict:
    return {
        "score": 1,
        "raw_score": 100,
        "source_scale": "0_to_100",
        "label": "passed",
        "verifier": "trialqa-exact-match-v1",
        "metrics": {"exact_match": 1},
        "submitted_answer": "BRCA1 is on chromosome 17.",
        "row_id": "17",
        "question": "Find the BRCA1 chromosome.",
        "question_group_key": "row-0017",
        "partition": "test",
        "condition": "skilled",
        "repeat_index": 1,
        "n_repeats": 3,
        "task_name": "trialqa-17",
    }


def _run() -> dict:
    return {
        "run_id": "demo-run-1",
        "phase": "evaluation",
        "model": EXECUTOR_MODEL,
        "executor_model": EXECUTOR_MODEL,
        "route": EXECUTOR_ROUTE,
        "skill_loaded": True,
        "candidate_id": CANDIDATE_ID,
        "candidate_manifest_sha256": CANDIDATE_MANIFEST,
        "candidate_skill_sha256": CANDIDATE_SKILL,
    }


def _import(session_dir: Path, project_dir: Path, **overrides):
    return import_native_trialqa_evidence(
        session_dir,
        namespace=NAMESPACE,
        task=overrides.get("task", _task()),
        outcome=overrides.get("outcome", _outcome()),
        run=overrides.get("run", _run()),
        project_dir=project_dir,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_import_normalizes_cumulative_turns_and_activates_candidate(tmp_path: Path) -> None:
    store, session_dir = _create_session(tmp_path)

    result = _import(session_dir, tmp_path)

    assert result.imported is True
    assert result.evidence_id.startswith("native-")
    evidence = _read_json(result.evidence_path / "evidence.json")
    assert evidence["task"]["type"] == "trialqa"
    assert evidence["task"]["id"] == "trialqa-17"
    assert evidence["execution"]["session_id"] == SESSION_ID
    assert evidence["execution"]["served_models"] == [EXECUTOR_MODEL]
    assert evidence["outcome"]["score"] == 1.0
    assert [event["sequence"] for event in evidence["events"]] == [0, 1, 2, 3]
    assert [event["kind"] for event in evidence["events"]] == [
        "message",
        "tool_call",
        "tool_result",
        "final_output",
    ]
    assert evidence["events"][1]["payload"]["id"] == "call-1"
    assert evidence["events"][2]["payload"]["tool_call_id"] == "call-1"
    assert evidence["events"][3]["payload"]["content"] == "BRCA1 is on chromosome 17."
    validate_native_trialqa_evidence_directory(
        result.evidence_path,
        expected_evidence_id=result.evidence_id,
    )

    store.save_candidate(
        candidate_id="candidate-1",
        skills={"SKILL.md": "# TrialQA\n"},
        generator=EXECUTOR_MODEL,
        evidence_ids=[result.evidence_id],
        validation={"status": "passed"},
        created_at="2026-07-06T10:01:00Z",
    )
    activation = store.activate("candidate-1")
    assert activation.active_candidate_id == "candidate-1"


def test_import_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)

    first = _import(session_dir, tmp_path)
    second = _import(session_dir, tmp_path)
    changed = _import(
        session_dir,
        tmp_path,
        outcome={**_outcome(), "score": 0.5, "raw_score": 50},
    )

    assert second.evidence_id == first.evidence_id
    assert second.evidence_path == first.evidence_path
    assert second.imported is False
    assert changed.evidence_id != first.evidence_id
    assert changed.imported is True


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("task", "id"),
        ("task", "question"),
        ("task", "condition"),
        ("outcome", "score"),
        ("outcome", "verifier"),
        ("run", "run_id"),
        ("run", "phase"),
    ],
)
def test_import_requires_explicit_trialqa_metadata(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    _store, session_dir = _create_session(tmp_path)
    values = {"task": _task(), "outcome": _outcome(), "run": _run()}
    values[section].pop(field)

    with pytest.raises(NativeTrialQAEvidenceError, match=field):
        _import(session_dir, tmp_path, **values)


@pytest.mark.parametrize(
    ("status", "turns"),
    [
        ("failed", None),
        ("completed", []),
    ],
)
def test_import_rejects_failed_or_empty_session(
    tmp_path: Path,
    status: str,
    turns: list[dict] | None,
) -> None:
    _store, session_dir = _create_session(tmp_path, status=status, turns=turns)

    with pytest.raises(ValueError, match="completed, non-empty"):
        _import(session_dir, tmp_path)


def test_import_binds_task_and_run_to_captured_context(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    session_path = session_dir / "session.json"
    session = _read_json(session_path)
    session["run_context"]["condition"] = "baseline"
    session_path.write_text(json.dumps(session, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="run_context.condition"):
        _import(session_dir, tmp_path)


def test_import_rejects_non_ultra_stats_attribution(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats["models"] = {"other/model": {"calls": 2, "errors": 0}}
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="pinned Ultra model"):
        _import(session_dir, tmp_path)


def test_import_retains_recovered_ultra_error_as_telemetry(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats.update(
        {
            "total_requests": 3,
            "total_errors": 1,
            "models": {EXECUTOR_MODEL: {"calls": 2, "errors": 1}},
            "openai_transport": _openai_transport(3),
        }
    )
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    result = _import(session_dir, tmp_path)

    retained = _read_json(result.evidence_path / "raw" / "stats.json")
    assert retained["total_requests"] == 3
    assert retained["total_errors"] == 1
    assert retained["models"][EXECUTOR_MODEL] == {"calls": 2, "errors": 1}
    validate_native_trialqa_evidence_directory(
        result.evidence_path,
        expected_evidence_id=result.evidence_id,
    )


def test_import_accepts_priced_transport_retry_and_retains_sensitivity(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats["openai_transport"] = _openai_transport(
        2,
        retries=1,
        charges=1,
        prompt=20,
        completion=8,
    )
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    result = _import(session_dir, tmp_path)

    retained = _read_json(result.evidence_path / "raw" / "stats.json")
    assert retained["openai_transport"]["physical_attempts"] == 3
    assert retained["openai_transport"]["retry_token_sensitivity"]["total"] == 28


@pytest.mark.parametrize(
    "transport",
    [
        _openai_transport(2, retries=1, unpriced=1),
        {**_openai_transport(2), "physical_attempts": 3},
    ],
)
def test_import_rejects_unpriced_or_inconsistent_transport(
    tmp_path: Path, transport: dict[str, object]
) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats["openai_transport"] = transport
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="transport accounting"):
        _import(session_dir, tmp_path)


@pytest.mark.parametrize(
    "models",
    [
        {EXECUTOR_MODEL: {"calls": 2, "errors": 0}},
        {
            EXECUTOR_MODEL: {"calls": 2, "errors": 0},
            "other/model": {"calls": 0, "errors": 1},
        },
        {EXECUTOR_MODEL: {"calls": 1, "errors": 2}},
    ],
)
def test_import_rejects_inconsistent_recovered_error_attribution(
    tmp_path: Path,
    models: dict[str, dict[str, int]],
) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats.update({"total_requests": 3, "total_errors": 1, "models": models})
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="attempts|recovered errors"):
        _import(session_dir, tmp_path)


def test_import_rejects_missing_successful_turn_for_reconciled_stats(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path, turns=_turns()[:1])
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats.update(
        {
            "total_requests": 3,
            "total_errors": 1,
            "models": {EXECUTOR_MODEL: {"calls": 2, "errors": 1}},
        }
    )
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="captured turns"):
        _import(session_dir, tmp_path)


@pytest.mark.parametrize("subsystem_name", ["classifier", "planner"])
def test_import_rejects_non_executor_subsystem_activity(
    tmp_path: Path,
    subsystem_name: str,
) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    stats[subsystem_name] = {"total_requests": 1, "total_errors": 0}
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match=subsystem_name):
        _import(session_dir, tmp_path)


@pytest.mark.parametrize("field", ["total_errors", "model_errors"])
def test_import_rejects_boolean_error_counters(tmp_path: Path, field: str) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats = _read_json(stats_path)
    if field == "total_errors":
        stats["total_errors"] = False
    else:
        stats["models"][EXECUTOR_MODEL]["errors"] = False
    stats_path.write_text(json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceError, match="stats"):
        _import(session_dir, tmp_path)


def test_import_rejects_zero_based_trialqa_repeat(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)

    with pytest.raises(NativeTrialQAEvidenceError, match="repeat_index"):
        _import(session_dir, tmp_path, task={**_task(), "repeat_index": 0})


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_import_rejects_linked_source_artifacts(tmp_path: Path, link_kind: str) -> None:
    _store, session_dir = _create_session(tmp_path)
    stats_path = session_dir / "stats.json"
    stats_path.unlink()
    external = tmp_path / "external-stats.json"
    external.write_text("{}\n", encoding="utf-8")
    if link_kind == "symlink":
        stats_path.symlink_to(external)
    else:
        os.link(external, stats_path)

    with pytest.raises(NativeTrialQAEvidenceConflictError, match="regular file"):
        _import(session_dir, tmp_path)


def test_validation_and_reimport_reject_tampered_evidence(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    result = _import(session_dir, tmp_path)
    trajectory = result.evidence_path / "raw" / "turns.jsonl"
    trajectory.chmod(0o600)
    trajectory.write_text("{}\n", encoding="utf-8")

    with pytest.raises(NativeTrialQAEvidenceConflictError, match="Integrity check failed"):
        validate_native_trialqa_evidence_directory(
            result.evidence_path,
            expected_evidence_id=result.evidence_id,
        )
    with pytest.raises(NativeTrialQAEvidenceConflictError, match="Integrity check failed"):
        _import(session_dir, tmp_path)


def test_validation_rejects_symlinked_and_hardlinked_evidence_files(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    symlinked = _import(session_dir, tmp_path)
    stats_path = symlinked.evidence_path / "raw" / "stats.json"
    stats_path.unlink()
    external = tmp_path / "external.json"
    external.write_text("{}\n", encoding="utf-8")
    stats_path.symlink_to(external)

    with pytest.raises(NativeTrialQAEvidenceConflictError, match="regular file"):
        validate_native_trialqa_evidence_directory(
            symlinked.evidence_path,
            expected_evidence_id=symlinked.evidence_id,
        )

    second_project = tmp_path / "second"
    second_project.mkdir()
    _store, second_session = _create_session(second_project)
    hardlinked = _import(second_session, second_project)
    stats_path = hardlinked.evidence_path / "raw" / "stats.json"
    stats_copy = second_project / "stats-copy.json"
    os.link(stats_path, stats_copy)

    with pytest.raises(NativeTrialQAEvidenceConflictError, match="single-link regular file"):
        validate_native_trialqa_evidence_directory(
            hardlinked.evidence_path,
            expected_evidence_id=hardlinked.evidence_id,
        )


def test_import_rejects_session_from_another_store(tmp_path: Path) -> None:
    source_project = tmp_path / "source"
    destination_project = tmp_path / "destination"
    source_project.mkdir()
    destination_project.mkdir()
    _store, session_dir = _create_session(source_project)

    with pytest.raises(ValueError, match="direct child"):
        _import(session_dir, destination_project)


def test_metadata_inputs_are_copied_before_normalization(tmp_path: Path) -> None:
    _store, session_dir = _create_session(tmp_path)
    task = _task()
    outcome = _outcome()
    run = _run()
    originals = deepcopy((task, outcome, run))

    _import(session_dir, tmp_path, task=task, outcome=outcome, run=run)

    assert (task, outcome, run) == originals

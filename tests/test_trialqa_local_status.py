# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_status as status


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    row_ids = [f"row-{index}" for index in range(8)]
    groups = [
        f"trialqa-{index:04d}-{hashlib.sha256(row_id.encode()).hexdigest()[:12]}"
        for index, row_id in enumerate(row_ids)
    ]
    digest = demo._sha256_bytes(demo._canonical_json(groups))
    tasks = [
        {
            "task_id": f"{group}-r{repeat:03d}-{condition}",
            "pair_id": f"{group}-r{repeat:03d}",
            "row_id": row_ids[index],
            "dataset_row_index": index,
            "question_group_key": group,
            "partition": "test",
            "phase": "evaluation",
            "condition": condition,
            "arm": condition,
            "repeat_index": repeat,
            "n_repeats": 5,
        }
        for index, group in enumerate(groups)
        for repeat in range(1, 6)
        for condition in ("baseline", "treatment")
    ]
    manifest = {
        "schema_version": "switchyard.trialqa_experiment_manifest.v1",
        "kind": "full",
        "dataset": {
            "official_labbench2": False,
            "test_count": 8,
            "heldout_ordering": {
                "question_count": 8,
                "question_group_keys": groups,
                "question_group_keys_sha256": digest,
            },
        },
        "protocol": {
            "performance_eligible": True,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": 8,
                "repeat_count": 5,
                "task_count": 80,
                "question_group_keys_sha256": digest,
            },
            "heldout_quarantine": {
                "question_start": 0,
                "question_count": 0,
                "disposition": "none-new-prospective-population",
                "question_group_keys_sha256": demo._sha256_bytes(demo._canonical_json([])),
            },
            "max_generation_concurrency": 4,
        },
        "tasks": tasks,
    }
    manifest = {
        "manifest_id": f"trialqa-full-{hashlib.sha256(demo._canonical_json(manifest)).hexdigest()[:20]}",
        **manifest,
    }
    return _write_json(path, manifest)


def _read_manifest_id(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["manifest_id"])


def _readiness(path: Path, *, manifest_id: str) -> Path:
    states = {
        f"trialqa-000{index}-prospective-r001-{condition}": "not_started"
        for index in range(4)
        for condition in ("baseline", "treatment")
    }
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_canary_readiness.v1",
            "status": "ready_for_generation",
            "manifest": {"manifest_id": manifest_id},
            "first_generation_canary": {
                "task_count": 8,
                "pair_count": 4,
                "selected_task_states": states,
                "scope_attestation": {
                    "selected_question_count": 4,
                    "selected_task_count": 8,
                },
            },
        },
    )


def _reference(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_targets.v1",
            "population": {"trials": 480, "heldout_questions": 96, "repeats_per_question": 5},
            "super": {"r1": {"accuracy": 0.738, "token_reduction": 0.3, "operational_call_reduction": 0.45}},
        },
    )


def _gate(
    path: Path,
    *,
    gate: str,
    decision: str,
    manifest_id: str,
    question_limit: int = 4,
    selected_repeat_indices: list[int] | None = None,
    confirmatory_scope_complete: bool = False,
    performance_eligible: bool = False,
) -> Path:
    repeat_indices = selected_repeat_indices or [1]
    task_count = question_limit * len(repeat_indices) * 2
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_gate_report.v3",
            "manifest_id": manifest_id,
            "gate": gate,
            "decision": decision,
            "performance_eligible": performance_eligible,
            "scope": {
                "task_count": task_count,
                "pair_count": task_count // 2,
                "confirmatory_scope_complete": confirmatory_scope_complete,
                "selection_attestation": {
                    "question_start": 0,
                    "question_limit": question_limit,
                    "selected_repeat_indices": repeat_indices,
                    "selected_question_count": question_limit,
                    "selected_task_count": task_count,
                },
            },
        },
    )


def test_status_points_to_guarded_generation_when_only_readiness_exists(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
    )

    assert report["schema_version"] == status.SCHEMA_VERSION
    assert report["reference_targets"]["trials"] == 480
    assert report["readiness"]["status"] == "ready_for_generation"
    assert report["next_action"] == {
        "action": "run_guarded_generation_canary",
        "reason": "no operational gate report exists yet",
        "requires_yes_spend": True,
    }


def test_status_allows_cumulative_generation_expansion_readiness(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    readiness = _readiness(tmp_path / "readiness.json", manifest_id=manifest_id)
    readiness_payload = json.loads(readiness.read_text(encoding="utf-8"))
    readiness_payload["status"] = "ready_for_generation_expansion"
    readiness_payload["first_generation_canary"]["selected_task_states"][
        "trialqa-0000-prospective-r001-completed"
    ] = "completed"
    _write_json(readiness, readiness_payload)

    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=readiness,
        reference_targets_path=_reference(tmp_path / "reference.json"),
    )

    assert report["readiness"]["status"] == "ready_for_generation_expansion"
    assert report["next_action"]["action"] == "run_guarded_generation_canary"


def test_status_kills_after_operational_gate_failure(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="kill",
            manifest_id=manifest_id,
        ),
    )

    assert report["operational_gate"]["decision"] == "kill"
    assert report["next_action"]["action"] == "kill_candidate"


def test_status_points_to_guarded_score_after_operational_promotion(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
        ),
    )

    assert report["operational_gate"]["decision"] == "promote_to_score"
    assert report["next_action"] == {
        "action": "run_guarded_score_canary",
        "reason": "operational gate promoted to score and no promotion gate exists yet",
        "requires_yes_spend": True,
    }


def test_status_expands_from_four_question_promotion_to_full_question_scope(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
        ),
    )

    assert report["next_action"] == {
        "action": "expand_generation_scope",
        "reason": "promoted partial question canary",
        "question_start": 0,
        "question_limit": 8,
        "repeat_limit": 1,
        "requires_yes_spend": True,
    }


def test_status_expands_from_repeat_one_to_repeat_three(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
        ),
    )

    assert report["next_action"] == {
        "action": "expand_generation_scope",
        "reason": "promoted repeat-1 scope",
        "question_start": 0,
        "question_limit": 8,
        "repeat_limit": 3,
        "requires_yes_spend": True,
    }


def test_status_expands_from_repeat_three_to_repeat_five(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3],
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3],
        ),
    )

    assert report["next_action"] == {
        "action": "expand_generation_scope",
        "reason": "promoted repeat-3 scope",
        "question_start": 0,
        "question_limit": 8,
        "repeat_limit": 5,
        "requires_yes_spend": True,
    }


def test_status_rejects_repeat_five_completion_without_confirmatory_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
        ),
    )

    assert report["next_action"] == {
        "action": "fix_readiness_before_spend",
        "reason": "promotion gate has not marked the confirmatory scope complete",
    }


def test_status_rejects_repeat_five_completion_without_performance_eligible_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
            confirmatory_scope_complete=True,
        ),
    )

    assert report["next_action"] == {
        "action": "fix_readiness_before_spend",
        "reason": "promotion gate is not performance eligible",
    }


def test_status_marks_prospective_directional_scope_complete_at_repeat_five(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _read_manifest_id(manifest)
    report = status.build_status_report(
        manifest_path=manifest,
        readiness_path=_readiness(tmp_path / "readiness.json", manifest_id=manifest_id),
        reference_targets_path=_reference(tmp_path / "reference.json"),
        operational_gate_path=_gate(
            tmp_path / "operational.json",
            gate="operational",
            decision="promote_to_score",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
        ),
        promotion_gate_path=_gate(
            tmp_path / "promotion.json",
            gate="promotion",
            decision="promote_to_next_cohort",
            manifest_id=manifest_id,
            question_limit=8,
            selected_repeat_indices=[1, 2, 3, 4, 5],
            confirmatory_scope_complete=True,
            performance_eligible=True,
        ),
    )

    assert report["next_action"] == {
        "action": "prospective_directional_scope_complete",
        "reason": "promotion gate completed the declared primary scope",
    }

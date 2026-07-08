# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_protocol_audit as audit


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
            "config": "clinicaltrials-gov",
            "official_labbench2": False,
            "id": "trialqa-compatible-prospective",
            "heldout_ordering": {
                "question_count": 8,
                "question_group_keys": groups,
                "question_group_keys_sha256": digest,
            },
            "revision": "clinicaltrials-gov-prospective-v1",
            "row_count": 8,
            "split": "prospective",
        },
        "protocol": {
            "batch_driver": "benchmark/trialqa_local_batch.py",
            "gold_in_manifest": False,
            "performance_eligible": True,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": 8,
                "repeat_count": 5,
                "task_count": 80,
                "question_group_keys_sha256": digest,
            },
            "prospective_population_kind": "trialqa-compatible-clinicaltrials-gov",
        },
        "routing": {
            "executor_route": demo.EXECUTOR_ROUTE,
            "executor_model": demo.EXECUTOR_MODEL,
            "profile_sha256": "sha256:" + "3" * 64,
        },
        "candidate": {
            "candidate_id": "candidate-synthetic",
            "manifest_sha256": "sha256:" + "4" * 64,
            "skill_sha256": "sha256:" + "5" * 64,
        },
        "tasks": tasks,
    }
    manifest = {
        "manifest_id": f"trialqa-full-{hashlib.sha256(demo._canonical_json(manifest)).hexdigest()[:20]}",
        **manifest,
    }
    return _write_json(path, manifest)


def _manifest_id(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["manifest_id"])


def _comparison_invariant() -> dict[str, object]:
    return {
        "status": "proved",
        "design": "concurrent-paired-same-executor-skill-only",
        "control_design": "concurrent-paired",
        "conditions": ["baseline", "treatment"],
        "shared_executor": {
            "route": demo.EXECUTOR_ROUTE,
            "model": demo.EXECUTOR_MODEL,
            "routing_profile_sha256": "sha256:" + "3" * 64,
        },
        "baseline_arm": {
            "condition": "baseline",
            "skill_loaded": False,
            "candidate_id": None,
            "candidate_manifest_sha256": None,
            "candidate_skill_sha256": None,
        },
        "treatment_arm": {
            "condition": "treatment",
            "skill_loaded": True,
            "candidate_id": "candidate-synthetic",
            "candidate_manifest_sha256": "sha256:" + "4" * 64,
            "candidate_skill_sha256": "sha256:" + "5" * 64,
        },
        "runtime_enforcement": [
            "manifest requires exact baseline/treatment pairs",
            "generation launch context binds executor route/model per task",
            "session proof requires Ultra-only served models",
            "session proof binds active-skill evidence per turn",
            "baseline workspace must expose no project skill",
            "treatment workspace must expose only the candidate skill",
        ],
    }


def _status(path: Path, *, manifest_id: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_status.v1",
            "manifest": {
                "manifest_id": manifest_id,
                "official_labbench2": False,
                "primary_evaluation_scope": {
                    "question_start": 0,
                    "question_count": 8,
                    "repeat_count": 5,
                    "task_count": 80,
                },
                "task_count": 80,
            },
            "reference_targets": {
                "trials": 480,
                "heldout_questions": 96,
                "repeats_per_question": 5,
            },
            "readiness": {
                "status": "ready_for_generation",
                "task_count": 8,
                "pair_count": 4,
                "selected_task_state_values": ["not_started"],
                "comparison_invariant": _comparison_invariant(),
            },
            "operational_gate": None,
            "promotion_gate": None,
            "next_action": {
                "action": "run_guarded_generation_canary",
                "reason": "no operational gate report exists yet",
                "requires_yes_spend": True,
            },
            "completion_state": "incomplete",
        },
    )


def _status_after_operational_promotion(path: Path, *, manifest_id: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_protocol_status.v1",
            "manifest": {
                "manifest_id": manifest_id,
                "official_labbench2": False,
                "primary_evaluation_scope": {
                    "question_start": 0,
                    "question_count": 8,
                    "repeat_count": 5,
                    "task_count": 80,
                },
                "task_count": 80,
            },
            "reference_targets": {
                "trials": 480,
                "heldout_questions": 96,
                "repeats_per_question": 5,
            },
            "readiness": {
                "status": "ready_for_generation",
                "task_count": 8,
                "pair_count": 4,
                "selected_task_state_values": ["not_started"],
                "comparison_invariant": _comparison_invariant(),
            },
            "operational_gate": {
                "decision": "promote_to_score",
                "performance_eligible": False,
                "task_count": 8,
                "pair_count": 4,
                "confirmatory_scope_complete": False,
                "selection_attestation": {
                    "question_start": 0,
                    "question_limit": 4,
                    "selected_repeat_indices": [1],
                    "selected_question_count": 4,
                    "selected_task_count": 8,
                },
            },
            "promotion_gate": None,
            "next_action": {
                "action": "run_guarded_score_canary",
                "reason": "operational gate promoted to score and no promotion gate exists yet",
                "requires_yes_spend": True,
            },
            "completion_state": "incomplete",
        },
    )


def _generation_summary(path: Path, *, manifest: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_canary_driver.v1",
            "status": "awaiting_spend_authorization",
            "spend_authorized": False,
            "readiness_status": "ready_for_generation",
            "readiness_output": "readiness.json",
            "generation_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_batch",
                "--stage",
                "generation",
            ],
            "operational_gate_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_gate",
            ],
            "authorized_rerun_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_canary",
                "--manifest",
                str(manifest),
                "--summary-output",
                str(path),
                "--yes-spend",
            ],
        },
    )


def _score_summary(path: Path, *, manifest: Path, operational_gate: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_canary_score_driver.v1",
            "status": "awaiting_spend_authorization",
            "spend_authorized": False,
            "operational_decision": "promote_to_score",
            "operational_gate": str(operational_gate),
            "score_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_batch",
                "--stage",
                "score",
            ],
            "promotion_gate_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_gate",
                "--gate",
                "promotion",
            ],
            "authorized_rerun_command": [
                "python",
                "-m",
                "benchmark.trialqa_local_canary_score",
                "--manifest",
                str(manifest),
                "--operational-gate",
                str(operational_gate),
                "--summary-output",
                str(path),
                "--yes-spend",
            ],
        },
    )


def _requirements_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    requirements = report["requirements"]
    assert isinstance(requirements, list)
    return {str(item["id"]): item for item in requirements if isinstance(item, dict)}


def test_protocol_audit_marks_ready_for_generation_spend_with_dry_run(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status(tmp_path / "status.json", manifest_id=manifest_id),
        generation_canary_summary_path=_generation_summary(
            tmp_path / "generation-summary.json",
            manifest=manifest,
        ),
    )

    requirements = _requirements_by_id(report)
    command = [
        "python",
        "-m",
        "benchmark.trialqa_local_canary",
        "--manifest",
        str(manifest),
        "--summary-output",
        str(tmp_path / "generation-summary.json"),
        "--yes-spend",
    ]
    assert report["schema_version"] == audit.SCHEMA_VERSION
    assert report["completion_state"] == "awaiting_generation_canary_spend_authorization"
    assert report["spend_boundary"] == {
        "requires_yes_spend": True,
        "authorized_by_audit": False,
        "reason": "this audit is read-only and never authorizes model spend",
    }
    assert report["next_command"] == {
        "kind": "guarded_generation_canary",
        "command": command,
        "shell_command": " ".join(
            [
                "python",
                "-m",
                "benchmark.trialqa_local_canary",
                "--manifest",
                str(manifest),
                "--summary-output",
                str(tmp_path / "generation-summary.json"),
                "--yes-spend",
            ]
        ),
        "source": str(tmp_path / "generation-summary.json"),
        "requires_yes_spend": True,
        "authorized_by_audit": False,
    }
    assert requirements["reference_targets_bound"]["status"] == "proved"
    assert requirements["prospective_manifest_bound"]["status"] == "proved"
    assert requirements["paired_skill_off_on_scope"]["status"] == "proved"
    assert requirements["generation_scope_readiness_clean"]["status"] == "proved"
    assert requirements["skill_distillation_ab_invariant_bound"]["status"] == "proved"
    assert requirements["local_switchyard_trialqa_transfer_runtime_bound"]["status"] == "proved"
    assert requirements["guarded_generation_dry_run_persisted"]["status"] == "proved"
    assert requirements["operational_generation_gate_completed"]["status"] == "missing"
    assert requirements["quality_parity_evidence"]["status"] == "missing"
    assert requirements["efficiency_benefit_evidence"]["status"] == "missing"


def test_protocol_audit_requires_persisted_dry_run_before_spend(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status(tmp_path / "status.json", manifest_id=manifest_id),
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "missing_no_spend_evidence"
    assert report["next_command"] is None
    assert requirements["guarded_generation_dry_run_persisted"]["status"] == "missing"


def test_protocol_audit_rejects_status_for_different_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")

    with pytest.raises(audit.TrialQAProtocolAuditError, match="different manifest"):
        audit.build_protocol_audit(
            manifest_path=manifest,
            status_path=_status(tmp_path / "status.json", manifest_id="other"),
            generation_canary_summary_path=_generation_summary(
                tmp_path / "summary.json",
                manifest=manifest,
            ),
        )


def test_protocol_audit_requires_ab_invariant_before_spend(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    status_path = _status(tmp_path / "status.json", manifest_id=manifest_id)
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    del status_payload["readiness"]["comparison_invariant"]
    _write_json(status_path, status_payload)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=status_path,
        generation_canary_summary_path=_generation_summary(
            tmp_path / "summary.json",
            manifest=manifest,
        ),
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "not_ready_for_spend"
    assert report["next_command"] is None
    assert requirements["skill_distillation_ab_invariant_bound"]["status"] == "failed"


def test_protocol_audit_rejects_official_hf_population_before_spend(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"].update(
        {
            "config": "trialqa",
            "id": "EdisonScientific/labbench2",
            "official_labbench2": True,
            "revision": "main",
            "split": "train",
        }
    )
    _write_json(manifest, payload)
    manifest_id = _manifest_id(manifest)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status(tmp_path / "status.json", manifest_id=manifest_id),
        generation_canary_summary_path=_generation_summary(
            tmp_path / "summary.json",
            manifest=manifest,
        ),
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "not_ready_for_spend"
    assert report["next_command"] is None
    assert (
        requirements["local_switchyard_trialqa_transfer_runtime_bound"]["status"]
        == "failed"
    )
    assert "official_labbench2_false" in requirements[
        "local_switchyard_trialqa_transfer_runtime_bound"
    ]["evidence"]


def test_protocol_audit_rejects_ab_invariant_for_different_candidate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    status_path = _status(tmp_path / "status.json", manifest_id=manifest_id)
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["readiness"]["comparison_invariant"]["treatment_arm"][
        "candidate_skill_sha256"
    ] = "sha256:" + "9" * 64
    _write_json(status_path, status_payload)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=status_path,
        generation_canary_summary_path=_generation_summary(
            tmp_path / "summary.json",
            manifest=manifest,
        ),
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "not_ready_for_spend"
    assert report["next_command"] is None
    assert requirements["skill_distillation_ab_invariant_bound"]["status"] == "failed"


def test_protocol_audit_rejects_dry_run_command_for_different_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    summary = _generation_summary(tmp_path / "summary.json", manifest=tmp_path / "other.json")

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status(tmp_path / "status.json", manifest_id=manifest_id),
        generation_canary_summary_path=summary,
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "not_ready_for_spend"
    assert report["next_command"] is None
    assert requirements["guarded_generation_dry_run_persisted"]["status"] == "failed"
    assert "authorized_rerun_manifest" in requirements["guarded_generation_dry_run_persisted"][
        "evidence"
    ]


def test_protocol_audit_marks_ready_for_score_spend_after_operational_promotion(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    operational_gate = tmp_path / "operational.json"
    score_summary = _score_summary(
        tmp_path / "score-summary.json",
        manifest=manifest,
        operational_gate=operational_gate,
    )

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status_after_operational_promotion(
            tmp_path / "status.json",
            manifest_id=manifest_id,
        ),
        score_canary_summary_path=score_summary,
        operational_gate_path=operational_gate,
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "awaiting_score_canary_spend_authorization"
    assert report["next_command"]["kind"] == "guarded_score_canary"
    assert report["next_command"]["command"][-1] == "--yes-spend"
    assert report["next_command"]["source"] == str(score_summary)
    assert report["dry_run_summary"] == {
        "path": str(score_summary),
        "status": "awaiting_spend_authorization",
    }
    assert "guarded_generation_dry_run_persisted" not in requirements
    assert requirements["guarded_score_dry_run_persisted"]["status"] == "proved"
    assert requirements["operational_generation_gate_completed"]["status"] == "proved"
    assert requirements["quality_parity_evidence"]["status"] == "missing"
    assert requirements["efficiency_benefit_evidence"]["status"] == "missing"


def test_protocol_audit_rejects_score_dry_run_for_different_operational_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    score_summary = _score_summary(
        tmp_path / "score-summary.json",
        manifest=manifest,
        operational_gate=tmp_path / "other-operational.json",
    )

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=_status_after_operational_promotion(
            tmp_path / "status.json",
            manifest_id=manifest_id,
        ),
        score_canary_summary_path=score_summary,
        operational_gate_path=tmp_path / "operational.json",
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "not_ready_for_spend"
    assert report["next_command"] is None
    assert requirements["guarded_score_dry_run_persisted"]["status"] == "failed"
    assert "authorized_rerun_operational_gate" in requirements[
        "guarded_score_dry_run_persisted"
    ]["evidence"]


def test_protocol_audit_accepts_cumulative_generation_expansion_readiness(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    status_path = _status(tmp_path / "status.json", manifest_id=manifest_id)
    status_report = json.loads(status_path.read_text(encoding="utf-8"))
    status_report["readiness"] = {
        "status": "ready_for_generation_expansion",
        "task_count": 16,
        "pair_count": 8,
        "selected_task_state_values": ["completed", "not_started"],
        "comparison_invariant": _comparison_invariant(),
    }
    _write_json(status_path, status_report)
    summary_path = _generation_summary(tmp_path / "summary.json", manifest=manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["readiness_status"] = "ready_for_generation_expansion"
    _write_json(summary_path, summary)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=status_path,
        generation_canary_summary_path=summary_path,
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "awaiting_generation_canary_spend_authorization"
    assert requirements["generation_scope_readiness_clean"]["status"] == "proved"


def test_protocol_audit_marks_expansion_ready_for_generation_spend(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    status_path = _status(tmp_path / "status.json", manifest_id=manifest_id)
    status_report = json.loads(status_path.read_text(encoding="utf-8"))
    status_report["readiness"] = {
        "status": "ready_for_generation_expansion",
        "task_count": 16,
        "pair_count": 8,
        "selected_task_state_values": ["completed", "not_started"],
        "comparison_invariant": _comparison_invariant(),
    }
    status_report["operational_gate"] = {
        "decision": "promote_to_score",
        "performance_eligible": False,
    }
    status_report["promotion_gate"] = {
        "decision": "promote_to_next_cohort",
        "performance_eligible": False,
        "confirmatory_scope_complete": False,
    }
    status_report["next_action"] = {
        "action": "expand_generation_scope",
        "reason": "promoted partial question canary",
        "question_start": 0,
        "question_limit": 8,
        "repeat_limit": 1,
        "requires_yes_spend": True,
    }
    _write_json(status_path, status_report)
    summary_path = _generation_summary(tmp_path / "summary.json", manifest=manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["readiness_status"] = "ready_for_generation_expansion"
    _write_json(summary_path, summary)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=status_path,
        generation_canary_summary_path=summary_path,
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "awaiting_generation_canary_spend_authorization"
    assert report["next_action"]["action"] == "expand_generation_scope"
    assert report["next_command"]["kind"] == "guarded_generation_canary"
    assert report["next_command"]["command"][-1] == "--yes-spend"
    assert requirements["guarded_generation_dry_run_persisted"]["status"] == "proved"
    assert requirements["quality_parity_evidence"]["status"] == "missing"
    assert requirements["efficiency_benefit_evidence"]["status"] == "missing"


def test_protocol_audit_marks_final_scope_promotion_as_live_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    manifest_id = _manifest_id(manifest)
    status_path = _status(tmp_path / "status.json", manifest_id=manifest_id)
    status_report = json.loads(status_path.read_text(encoding="utf-8"))
    status_report["readiness"] = {
        "status": "ready_for_generation_expansion",
        "task_count": 80,
        "pair_count": 40,
        "selected_task_state_values": ["completed"],
        "comparison_invariant": _comparison_invariant(),
    }
    status_report["operational_gate"] = {
        "decision": "promote_to_score",
        "performance_eligible": False,
        "confirmatory_scope_complete": True,
    }
    status_report["promotion_gate"] = {
        "decision": "promote_to_next_cohort",
        "performance_eligible": True,
        "confirmatory_scope_complete": True,
    }
    status_report["next_action"] = {
        "action": "prospective_directional_scope_complete",
        "reason": "promotion gate completed the declared primary scope",
    }
    _write_json(status_path, status_report)

    report = audit.build_protocol_audit(
        manifest_path=manifest,
        status_path=status_path,
    )

    requirements = _requirements_by_id(report)
    assert report["completion_state"] == "prospective_directional_scope_complete"
    assert report["next_command"] is None
    assert requirements["quality_parity_evidence"]["status"] == "proved"
    assert requirements["efficiency_benefit_evidence"]["status"] == "proved"

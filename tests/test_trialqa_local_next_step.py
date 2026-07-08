# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import benchmark.trialqa_local_next_step as next_step


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_experiment_manifest.v1",
            "manifest_id": "trialqa-full-test",
        },
    )


def _base_status(action: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "switchyard.trialqa_protocol_status.v1",
        "manifest": {"manifest_id": "trialqa-full-test"},
        "readiness": {
            "status": "ready_for_generation",
            "scope_attestation": {
                "question_start": 0,
                "question_limit": 4,
                "selected_repeat_indices": [1],
            },
        },
        "operational_gate": None,
        "promotion_gate": None,
        "next_action": action,
    }


def _score_status() -> dict[str, object]:
    status = _base_status(
        {
            "action": "run_guarded_score_canary",
            "reason": "operational gate promoted to score and no promotion gate exists yet",
            "requires_yes_spend": True,
        }
    )
    status["operational_gate"] = {
        "decision": "promote_to_score",
        "selection_attestation": {
            "question_start": 0,
            "question_limit": 4,
            "selected_repeat_indices": [1],
        },
    }
    return status


def _config(
    tmp_path: Path,
    *,
    status: Path,
    operational_gate: Path | None = None,
    promotion_gate: Path | None = None,
) -> next_step.NextStepConfig:
    return next_step.NextStepConfig(
        status=status,
        manifest=_manifest(tmp_path / "manifest.json"),
        dataset=tmp_path / "dataset.parquet",
        experiment_root=tmp_path / "experiments",
        doctor=tmp_path / "doctor.json",
        population_report=tmp_path / "population.json",
        candidate=tmp_path / "candidate",
        switchyard=tmp_path / "bin" / "switchyard",
        codex=tmp_path / "bin" / "codex",
        tooluniverse=tmp_path / "tooluniverse" / "bin" / "tooluniverse-smcp-stdio",
        profile=tmp_path / "profile.yaml",
        reference_targets=tmp_path / "reference.json",
        runbook=tmp_path / "runbook.md",
        artifact_dir=tmp_path / "artifacts",
        artifact_stem="ctgov-prospective-v1-compact-v5",
        workers=4,
        max_generation_attempts=1,
        operational_gate=operational_gate,
        promotion_gate=promotion_gate,
        skills_distillation_repo=tmp_path / "skills-distillation",
    )


def test_next_step_emits_generation_preflight_from_readiness_scope(tmp_path: Path) -> None:
    status = _write_json(
        tmp_path / "status.json",
        _base_status(
            {
                "action": "run_guarded_generation_canary",
                "reason": "no operational gate report exists yet",
                "requires_yes_spend": True,
            }
        ),
    )

    report = next_step.build_next_step_plan(_config(tmp_path, status=status), python="python")

    command = report["safe_next_command"]["command"]
    assert report["schema_version"] == next_step.SCHEMA_VERSION
    assert report["terminal"] is False
    assert report["scope"]["suffix"] == "q0-q3-r1"
    assert report["safe_next_command"]["kind"] == "generation_preflight"
    assert "benchmark.trialqa_local_preflight" in command
    assert "--yes-spend" not in command
    assert command[command.index("--generation-summary-output") + 1].endswith(
        "canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json"
    )
    assert command[command.index("--reference-alignment-output") + 1].endswith(
        "reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json"
    )
    assert command[command.index("--skills-distillation-repo") + 1] == str(
        tmp_path / "skills-distillation"
    )
    assert command[command.index("--ladder-rehearsal") + 1] == str(
        tmp_path / "artifacts" / "ladder-rehearsal-ctgov-prospective-v1-compact-v5.json"
    )


def test_next_step_emits_score_preflight_after_operational_promotion(tmp_path: Path) -> None:
    status = _write_json(tmp_path / "status.json", _score_status())

    report = next_step.build_next_step_plan(_config(tmp_path, status=status), python="python")

    command = report["safe_next_command"]["command"]
    assert report["safe_next_command"]["kind"] == "score_preflight"
    assert report["scope"]["suffix"] == "q0-q3-r1"
    assert "benchmark.trialqa_local_score_preflight" in command
    assert "--yes-spend" not in command
    assert command[command.index("--score-summary-output") + 1].endswith(
        "canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json"
    )
    assert command[command.index("--reference-alignment-output") + 1].endswith(
        "reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json"
    )
    assert command[command.index("--operational-gate") + 1].endswith(
        "gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json"
    )
    assert command[command.index("--skills-distillation-repo") + 1] == str(
        tmp_path / "skills-distillation"
    )
    assert command[command.index("--ladder-rehearsal") + 1] == str(
        tmp_path / "artifacts" / "ladder-rehearsal-ctgov-prospective-v1-compact-v5.json"
    )


def test_next_step_emits_expansion_preflight_from_next_action(tmp_path: Path) -> None:
    status = _write_json(
        tmp_path / "status.json",
        _base_status(
            {
                "action": "expand_generation_scope",
                "reason": "promoted repeat-1 scope",
                "question_start": 0,
                "question_limit": 8,
                "repeat_limit": 3,
            }
        ),
    )

    report = next_step.build_next_step_plan(_config(tmp_path, status=status), python="python")

    command = report["safe_next_command"]["command"]
    assert report["safe_next_command"]["kind"] == "generation_expansion_preflight"
    assert report["scope"] == {
        "question_start": 0,
        "question_limit": 8,
        "repeat_limit": 3,
        "suffix": "q0-q7-r3",
    }
    assert command[command.index("--question-limit") + 1] == "8"
    assert command[command.index("--repeat-limit") + 1] == "3"
    assert command[command.index("--skills-distillation-repo") + 1] == str(
        tmp_path / "skills-distillation"
    )
    assert command[command.index("--ladder-rehearsal") + 1] == str(
        tmp_path / "artifacts" / "ladder-rehearsal-ctgov-prospective-v1-compact-v5.json"
    )
    assert command[command.index("--output") + 1].endswith(
        "no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q7-r3.json"
    )


def test_next_step_expansion_preflight_preserves_gate_context(tmp_path: Path) -> None:
    status = _write_json(
        tmp_path / "status.json",
        _base_status(
            {
                "action": "expand_generation_scope",
                "reason": "promoted partial question canary",
                "question_start": 0,
                "question_limit": 8,
                "repeat_limit": 1,
                "requires_yes_spend": True,
            }
        ),
    )
    operational_gate = tmp_path / "operational.json"
    promotion_gate = tmp_path / "promotion.json"

    report = next_step.build_next_step_plan(
        _config(
            tmp_path,
            status=status,
            operational_gate=operational_gate,
            promotion_gate=promotion_gate,
        ),
        python="python",
    )

    command = report["safe_next_command"]["command"]
    assert report["safe_next_command"]["kind"] == "generation_expansion_preflight"
    assert command[command.index("--operational-gate") + 1] == str(operational_gate)
    assert command[command.index("--promotion-gate") + 1] == str(promotion_gate)


def test_next_step_terminal_kill_has_no_safe_command(tmp_path: Path) -> None:
    status = _write_json(
        tmp_path / "status.json",
        _base_status(
            {
                "action": "kill_candidate",
                "reason": "operational decision is 'kill'",
            }
        ),
    )

    report = next_step.build_next_step_plan(_config(tmp_path, status=status), python="python")

    assert report["terminal"] is True
    assert report["decision"] == "kill_candidate"
    assert report["safe_next_command"] is None

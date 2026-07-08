# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_ladder_rehearsal as rehearsal


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path, *, question_count: int = 8, repeat_count: int = 5) -> Path:
    row_ids = [f"row-{index}" for index in range(question_count)]
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
            "n_repeats": repeat_count,
        }
        for index, group in enumerate(groups)
        for repeat in range(1, repeat_count + 1)
        for condition in ("baseline", "treatment")
    ]
    manifest = {
        "schema_version": "switchyard.trialqa_experiment_manifest.v1",
        "kind": "full",
        "dataset": {
            "official_labbench2": False,
            "test_count": question_count,
            "heldout_ordering": {
                "question_count": question_count,
                "question_group_keys": groups,
                "question_group_keys_sha256": digest,
            },
        },
        "protocol": {
            "performance_eligible": True,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": question_count,
                "repeat_count": repeat_count,
                "task_count": question_count * repeat_count * 2,
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


def _reference(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_targets.v1",
            "population": {
                "trials": 480,
                "heldout_questions": 96,
                "repeats_per_question": 5,
            },
            "super": {
                "r1": {
                    "accuracy": 0.738,
                    "token_reduction": 0.3,
                    "operational_call_reduction": 0.45,
                }
            },
        },
    )


def _config(tmp_path: Path) -> rehearsal.RehearsalConfig:
    return rehearsal.RehearsalConfig(
        manifest=_manifest(tmp_path / "manifest.json"),
        reference_targets=_reference(tmp_path / "reference.json"),
        dataset=tmp_path / "dataset.parquet",
        experiment_root=tmp_path / "experiments",
        doctor=tmp_path / "doctor.json",
        population_report=tmp_path / "population.json",
        candidate=tmp_path / "candidate",
        switchyard=tmp_path / "bin" / "switchyard",
        codex=tmp_path / "bin" / "codex",
        tooluniverse=tmp_path / "tooluniverse" / "bin" / "tooluniverse-smcp-stdio",
        profile=tmp_path / "profile.yaml",
        runbook=tmp_path / "runbook.md",
        artifact_dir=tmp_path / "artifacts",
        artifact_stem="ctgov-prospective-v1-compact-v5",
        rehearsal_dir=tmp_path / "rehearsal",
        workers=4,
        max_generation_attempts=1,
    )


def test_ladder_rehearsal_passes_full_expected_sequence(tmp_path: Path) -> None:
    report = rehearsal.run_ladder_rehearsal(_config(tmp_path), python="python")

    scenarios = {item["scenario_id"]: item for item in report["scenarios"]}
    assert report["schema_version"] == rehearsal.SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["spend_authorized"] is False
    assert report["model_calls"] == 0
    assert report["judge_calls"] == 0
    assert report["ladder_budget"]["first_spend_boundary"] == {
        "stage": "generation",
        "scope": {
            "question_start": 0,
            "question_limit": 4,
            "repeat_limit": 1,
            "suffix": "q0-q3-r1",
        },
        "expected_model_calls": 8,
        "expected_judge_calls": 0,
    }
    assert report["ladder_budget"]["max_model_calls_before_directional_completion"] == 80
    assert report["ladder_budget"]["max_judge_calls_before_directional_completion"] == 80
    assert report["ladder_budget"]["max_total_live_calls_before_directional_completion"] == 160
    boundaries = report["ladder_budget"]["all_promote_boundaries"]
    assert [item["incremental_model_calls"] for item in boundaries] == [8, 0, 8, 0, 32, 0, 32, 0]
    assert [item["incremental_judge_calls"] for item in boundaries] == [0, 8, 0, 8, 0, 32, 0, 32]
    assert report["scenario_count"] == 8
    assert report["failed_scenario_count"] == 0
    assert scenarios["initial-generation"]["observed"] == {
        "action": "run_guarded_generation_canary",
        "terminal": False,
        "safe_kind": "generation_preflight",
        "decision": None,
        "scope": {
            "question_start": 0,
            "question_limit": 4,
            "repeat_limit": 1,
            "suffix": "q0-q3-r1",
        },
    }
    assert scenarios["post-generation-promote"]["observed"]["safe_kind"] == "score_preflight"
    assert scenarios["post-score-q4-promote"]["observed"]["scope"]["suffix"] == "q0-q7-r1"
    assert scenarios["post-score-q8-r1-promote"]["observed"]["scope"]["suffix"] == "q0-q7-r3"
    assert scenarios["post-score-q8-r3-promote"]["observed"]["scope"]["suffix"] == "q0-q7-r5"
    assert scenarios["post-score-q8-r5-promote"]["observed"]["decision"] == (
        "prospective_directional_scope_complete"
    )
    for item in report["scenarios"]:
        next_step = item["artifacts"]["next_step"]
        assert isinstance(next_step, str)
        assert Path(next_step).exists()


def test_ladder_rehearsal_reports_failed_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    real_build = rehearsal.next_step.build_next_step_plan

    def fake_build_next_step_plan(*args: object, **kwargs: object) -> dict[str, object]:
        report = real_build(*args, **kwargs)
        if report["action"] == "run_guarded_score_canary":
            report = {**report, "action": "kill_candidate"}
        return report

    monkeypatch.setattr(
        rehearsal.next_step,
        "build_next_step_plan",
        fake_build_next_step_plan,
    )

    report = rehearsal.run_ladder_rehearsal(config, python="python")

    failed = [item for item in report["scenarios"] if item["status"] == "failed"]
    assert report["status"] == "failed"
    assert report["failed_scenario_count"] == 1
    assert failed[0]["scenario_id"] == "post-generation-promote"
    assert "run_guarded_score_canary" in failed[0]["failures"][0]


def test_ladder_rehearsal_rejects_too_small_primary_scope(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        manifest=_manifest(tmp_path / "small-manifest.json", question_count=3),
    )

    with pytest.raises(
        rehearsal.TrialQALadderRehearsalError,
        match="at least four questions",
    ):
        rehearsal.run_ladder_rehearsal(config, python="python")

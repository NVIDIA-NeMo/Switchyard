# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import benchmark.trialqa_local_batch as batch
import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_readiness as readiness
from benchmark.trialqa_local_dataset import TRIALQA_SCHEMA
from benchmark.trialqa_local_runner import validate_candidate_skill


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _profile(path: Path) -> Path:
    path.write_text(
        "defaults:\n"
        "  api_key: test\n"
        "  base_url: https://invalid.example/v1\n"
        "routes:\n"
        "  sd-executor:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {demo.EXECUTOR_MODEL}\n"
        "      format: openai\n"
        "  sd-distiller:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {demo.JUDGE_MODEL}\n"
        "      format: openai\n"
        "  sd-judge:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {demo.JUDGE_MODEL}\n"
        "      format: openai\n",
        encoding="utf-8",
    )
    return path


def _tooluniverse(root: Path) -> Path:
    binary = _write_executable(root / "bin" / "tooluniverse-smcp-stdio")
    _write_executable(root / "bin" / "python")
    metadata = (
        root / "lib" / "python3.12" / "site-packages" / "tooluniverse-1.1.11.dist-info" / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: tooluniverse\nVersion: 1.1.11\n", encoding="utf-8")
    return binary


def _candidate(root: Path) -> Path:
    skill = (
        b"---\n"
        b"name: tooluniverse-trialqa\n"
        b"description: Answer TrialQA questions with targeted ToolUniverse evidence.\n"
        b"---\n\n"
        b"# TrialQA\n\nSearch trial registries before answering.\n"
    )
    skill_path = root / "tooluniverse-trialqa" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_bytes(skill)
    manifest = {
        "schema_version": 1,
        "candidate_id": "candidate-synthetic",
        "validation": {"status": "passed"},
        "skills": [
            {
                "path": "tooluniverse-trialqa/SKILL.md",
                "sha256": hashlib.sha256(skill).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _namespace_translation_attestation() -> dict[str, object]:
    flattened = "__sy1n17_mcp__tooluniversetrialqa_load_active_skill"
    return {
        "schema_version": "switchyard.trialqa_namespace_translation.v1",
        "source_format": "openai_responses",
        "target_format": "openai_chat",
        "namespace": "mcp__tooluniverse",
        "child_name": "trialqa_load_active_skill",
        "flattened_name": flattened,
        "flattened_name_sha256": demo._sha256_bytes(flattened.encode("ascii")),
        "request_flattened_tool_count": 1,
        "response_namespace": "mcp__tooluniverse",
        "response_child_name": "trialqa_load_active_skill",
        "response_call_id": "trialqa-doctor-call",
        "model_calls": 0,
    }


def _doctor_report(
    path: Path,
    *,
    switchyard: Path,
    codex: Path,
    tooluniverse: Path,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": demo.DOCTOR_SCHEMA_VERSION,
                "status": "passed",
                "model_calls": 0,
                "dataset": {
                    "id": demo.TRIALQA_DATASET_ID,
                    "config": demo.TRIALQA_DATASET_CONFIG,
                    "revision": demo.TRIALQA_DATASET_REVISION,
                    "parquet_sha256": demo.TRIALQA_PARQUET_SHA256,
                    "row_count": 120,
                    "split_counts": {"train": 24, "test": 96},
                },
                "routing": {
                    "first_route": demo.EXECUTOR_ROUTE,
                    "executor_model": demo.EXECUTOR_MODEL,
                    "judge_route": demo.JUDGE_ROUTE,
                    "judge_model": demo.JUDGE_MODEL,
                },
                "implementation": {"source_sha256": demo._execution_source_sha256()},
                "mcp_adapter": demo._adapter_schema_attestation(),
                "codex_safety": demo._codex_safety_attestation(),
                "namespace_translation": _namespace_translation_attestation(),
                "runtime_artifacts": {
                    "switchyard": demo._runtime_binary_attestation(
                        switchyard, "Switchyard binary"
                    ),
                    "codex": demo._runtime_binary_attestation(codex, "Codex binary"),
                    "switchyard_rust_native_extension": demo._native_extension_attestation(),
                    "tooluniverse": {
                        **demo._runtime_binary_attestation(
                            tooluniverse, "ToolUniverse binary"
                        ),
                        "version": demo.TOOLUNIVERSE_VERSION,
                        "python": demo._runtime_binary_attestation(
                            tooluniverse.parent / "python",
                            "ToolUniverse venv Python",
                        ),
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _prospective_dataset(path: Path, *, rows: int = 8) -> object:
    records = []
    for index in range(rows):
        records.append(
            {
                "id": f"prospective-{index:03d}",
                "tag": "trialqa",
                "version": "1.0",
                "question": f"What is the primary outcome measure for trial {index:03d}?",
                "ideal": f"Outcome measure {index:03d}",
                "files": "",
                "sources": [f"https://clinicaltrials.gov/study/NCT9{index:07d}"],
                "key_passage": f"Prospective metadata passage {index:03d}",
                "canary": "",
                "is_opensource": True,
                "ground_truth": True,
                "prompt_suffix": "Answer with the exact field value.",
                "type": "registry",
                "mode": {"file": False, "retrieve": True, "inject": False},
                "validator_params": "{}",
                "answer_regex": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=TRIALQA_SCHEMA), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return demo.load_trialqa_compatible_parquet(
        path,
        expected_sha256=digest,
        expected_row_count=rows,
        revision="clinicaltrials-gov-prospective-v1",
    )


def _population_report(path: Path, dataset: object) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "switchyard.trialqa_prospective_population.v1",
                "status": "passed",
                "population": {
                    "kind": "trialqa-compatible-clinicaltrials-gov-prospective",
                    "official_labbench2": False,
                    "sha256": dataset.parquet_sha256,
                    "row_count": len(dataset.rows),
                },
                "official_trialqa_exclusion": {
                    "selected_ncts_overlap_official_trialqa": [],
                },
                "use_constraints": {
                    "model_calls": 0,
                    "must_not_be_reported_as_official_labbench2_trialqa": True,
                    "performance_eligible_only_if_manifest_is_frozen_before_generation": True,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_readiness_report_rebuilds_prospective_manifest_and_first_scope(
    tmp_path: Path,
) -> None:
    switchyard = _write_executable(tmp_path / "bin" / "switchyard").absolute()
    codex = _write_executable(tmp_path / "bin" / "codex").absolute()
    tooluniverse = _tooluniverse(tmp_path / "tooluniverse-venv").absolute()
    profile = _profile(tmp_path / "profile.yaml").absolute()
    candidate_root = _candidate(tmp_path / "candidate").absolute()
    dataset = _prospective_dataset(tmp_path / "prospective.parquet")
    population = _population_report(tmp_path / "population-report.json", dataset)
    doctor = _doctor_report(
        tmp_path / "doctor.json",
        switchyard=switchyard,
        codex=codex,
        tooluniverse=tooluniverse,
    )
    candidate = validate_candidate_skill(candidate_root, demo.NAMESPACE)
    manifest = demo.build_prospective_experiment_manifest(
        dataset=dataset,
        population_report=population,
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=switchyard,
        codex_bin=codex,
        tooluniverse_bin=tooluniverse,
        doctor_report=doctor,
    )
    manifest_path = tmp_path / "manifest.json"
    demo._write_json_atomic(manifest_path, manifest)

    report = readiness.build_readiness_report(
        manifest_path=manifest_path,
        dataset_path=dataset.path,
        experiment_root=tmp_path / "experiments",
        doctor_report=doctor,
        population_report=population,
        candidate_root=candidate_root,
        switchyard_bin=switchyard,
        codex_bin=codex,
        tooluniverse_bin=tooluniverse,
        routing_profile=profile,
        question_start=0,
        question_limit=4,
        repeat_limit=1,
    )

    assert report["schema_version"] == readiness.SCHEMA_VERSION
    assert report["status"] == "ready_for_generation"
    assert report["manifest"]["official_labbench2"] is False
    assert report["manifest"]["task_count"] == 80
    assert report["comparison_invariant"] == {
        "status": "proved",
        "design": "concurrent-paired-same-executor-skill-only",
        "control_design": "concurrent-paired",
        "conditions": ["baseline", "treatment"],
        "shared_executor": {
            "route": demo.EXECUTOR_ROUTE,
            "model": demo.EXECUTOR_MODEL,
            "routing_profile_sha256": manifest["routing"]["profile_sha256"],
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
            "candidate_id": manifest["candidate"]["candidate_id"],
            "candidate_manifest_sha256": manifest["candidate"]["manifest_sha256"],
            "candidate_skill_sha256": manifest["candidate"]["skill_sha256"],
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
    assert report["first_generation_canary"]["task_count"] == 8
    assert report["first_generation_canary"]["pair_count"] == 4
    assert report["first_generation_canary"]["selected_repeat_indices"] == [1]
    assert report["first_generation_canary"]["ledger_exists"] is False
    assert report["first_generation_canary"]["selected_task_state_counts"] == {
        "not_started": 8
    }
    assert set(report["first_generation_canary"]["selected_task_states"].values()) == {
        "not_started"
    }


def test_readiness_allows_cumulative_generation_expansion_with_completed_prefix(
    tmp_path: Path,
) -> None:
    switchyard = _write_executable(tmp_path / "bin" / "switchyard").absolute()
    codex = _write_executable(tmp_path / "bin" / "codex").absolute()
    tooluniverse = _tooluniverse(tmp_path / "tooluniverse-venv").absolute()
    profile = _profile(tmp_path / "profile.yaml").absolute()
    candidate_root = _candidate(tmp_path / "candidate").absolute()
    dataset = _prospective_dataset(tmp_path / "prospective.parquet")
    population = _population_report(tmp_path / "population-report.json", dataset)
    doctor = _doctor_report(
        tmp_path / "doctor.json",
        switchyard=switchyard,
        codex=codex,
        tooluniverse=tooluniverse,
    )
    candidate = validate_candidate_skill(candidate_root, demo.NAMESPACE)
    manifest = demo.build_prospective_experiment_manifest(
        dataset=dataset,
        population_report=population,
        candidate=candidate,
        routing_profile=profile,
        switchyard_bin=switchyard,
        codex_bin=codex,
        tooluniverse_bin=tooluniverse,
        doctor_report=doctor,
    )
    manifest_path = tmp_path / "manifest.json"
    demo._write_json_atomic(manifest_path, manifest)
    first_scope = batch._build_manifest_task_scope(
        manifest,
        list(manifest["tasks"]),
        limit=None,
        question_start=0,
        question_limit=4,
        repeat_limit=1,
        condition="both",
    )
    ledger = demo.ResumableLedger(
        tmp_path / "experiments" / str(manifest["manifest_id"]) / "ledger.jsonl",
        manifest,
    )
    for task in first_scope.tasks:
        task_id = str(task["task_id"])
        ledger.append(task_id, "generation_started")
        ledger.append(task_id, "generation_completed")
        ledger.append(task_id, "scored")
        ledger.append(task_id, "evidence_imported")
        ledger.append(task_id, "completed")

    report = readiness.build_readiness_report(
        manifest_path=manifest_path,
        dataset_path=dataset.path,
        experiment_root=tmp_path / "experiments",
        doctor_report=doctor,
        population_report=population,
        candidate_root=candidate_root,
        switchyard_bin=switchyard,
        codex_bin=codex,
        tooluniverse_bin=tooluniverse,
        routing_profile=profile,
        question_start=0,
        question_limit=8,
        repeat_limit=1,
    )

    assert report["status"] == "ready_for_generation_expansion"
    assert report["first_generation_canary"]["task_count"] == 16
    assert report["first_generation_canary"]["pair_count"] == 8
    assert report["first_generation_canary"]["selected_task_state_counts"] == {
        "completed": 8,
        "not_started": 8,
    }

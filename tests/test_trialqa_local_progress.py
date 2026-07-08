# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import benchmark.trialqa_local_batch as batch
import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_progress as progress


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    rows = [f"row-{index}" for index in range(4)]
    groups = [
        f"trialqa-{index:04d}-{hashlib.sha256(row.encode()).hexdigest()[:12]}"
        for index, row in enumerate(rows)
    ]
    digest = demo._sha256_bytes(demo._canonical_json(groups))
    tasks = [
        {
            "task_id": f"{group}-r{repeat:03d}-{condition}",
            "pair_id": f"{group}-r{repeat:03d}",
            "row_id": rows[index],
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
    seed = {
        "schema_version": "switchyard.trialqa_experiment_manifest.v1",
        "kind": "full",
        "dataset": {
            "official_labbench2": False,
            "test_count": len(groups),
            "heldout_ordering": {
                "question_count": len(groups),
                "question_group_keys": groups,
                "question_group_keys_sha256": digest,
            },
        },
        "protocol": {
            "conditions": ["baseline", "treatment"],
            "performance_eligible": True,
            "max_generation_concurrency": 4,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": len(groups),
                "repeat_count": 5,
                "task_count": len(tasks),
                "question_group_keys_sha256": digest,
            },
            "heldout_quarantine": {
                "question_start": 0,
                "question_count": 0,
                "disposition": "none-new-prospective-population",
                "question_group_keys_sha256": demo._sha256_bytes(demo._canonical_json([])),
            },
        },
        "tasks": tasks,
    }
    manifest = {
        "manifest_id": f"trialqa-full-{hashlib.sha256(demo._canonical_json(seed)).hexdigest()[:20]}",
        **seed,
    }
    return _write_json(path, manifest)


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_first_scope_tasks(manifest: dict[str, object]) -> list[dict[str, object]]:
    return list(
        batch._build_manifest_task_scope(
            manifest,
            list(manifest["tasks"]),  # type: ignore[arg-type]
            limit=None,
            question_start=0,
            question_limit=2,
            repeat_limit=1,
            condition="both",
        ).tasks
    )


def test_progress_reports_clean_not_started_generation_scope(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    report = progress.build_progress_report(
        manifest_path=manifest_path,
        experiment_root=tmp_path,
        stage="generation",
        question_start=0,
        question_limit=2,
        repeat_limit=1,
    )

    assert report["schema_version"] == progress.SCHEMA_VERSION
    assert report["ledger"]["exists"] is False
    assert report["progress"]["selected_task_count"] == 4
    assert report["progress"]["category_counts"] == {"not_started": 4}
    assert report["recommendation"] == {
        "action": "run_guarded_generation_canary_after_spend_review",
        "requires_spend": True,
        "reason": "no selected generation work has started",
    }


def test_progress_treats_started_generation_without_lock_as_interrupted(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    manifest = _read_manifest(manifest_path)
    manifest_id = str(manifest["manifest_id"])
    ledger = demo.ResumableLedger(tmp_path / manifest_id / "ledger.jsonl", manifest)
    tasks = list(manifest["tasks"])
    assert isinstance(tasks[0], dict)
    assert isinstance(tasks[1], dict)
    ledger.append(str(tasks[0]["task_id"]), "generation_started", {"generation_attempt": 1})
    ledger.append(str(tasks[1]["task_id"]), "generation_started", {"generation_attempt": 1})
    ledger.append(str(tasks[1]["task_id"]), "generation_completed", {"generation_path": "g.json"})

    report = progress.build_progress_report(
        manifest_path=manifest_path,
        experiment_root=tmp_path,
        stage="generation",
        question_start=0,
        question_limit=2,
        repeat_limit=1,
    )

    assert report["ledger"]["exists"] is True
    assert report["ledger"]["record_count"] == 3
    assert report["progress"]["category_counts"] == {
        "generated": 1,
        "in_progress": 1,
        "not_started": 2,
    }
    assert report["batch_lock"]["held"] is False
    assert report["recommendation"] == {
        "action": "recover_interrupted_generation",
        "requires_spend": True,
        "reason": "1 selected task(s) are generation_started, but no batch driver holds the lock",
        "recovery_flag": "--recover-interrupted",
    }


def test_progress_reports_partial_running_generation_when_lock_is_held(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    manifest = _read_manifest(manifest_path)
    manifest_id = str(manifest["manifest_id"])
    ledger = demo.ResumableLedger(tmp_path / manifest_id / "ledger.jsonl", manifest)
    tasks = list(manifest["tasks"])
    assert isinstance(tasks[0], dict)
    ledger.append(str(tasks[0]["task_id"]), "generation_started", {"generation_attempt": 1})
    lock_path = tmp_path / manifest_id / "batch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = progress.build_progress_report(
            manifest_path=manifest_path,
            experiment_root=tmp_path,
            stage="generation",
            question_start=0,
            question_limit=2,
            repeat_limit=1,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert report["batch_lock"]["held"] is True
    assert report["progress"]["category_counts"] == {
        "in_progress": 1,
        "not_started": 3,
    }
    assert report["recommendation"]["action"] == "wait_or_monitor_generation"
    assert report["recommendation"]["requires_spend"] is False


def test_progress_reports_ready_to_score_scope(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    manifest = _read_manifest(manifest_path)
    manifest_id = str(manifest["manifest_id"])
    ledger = demo.ResumableLedger(tmp_path / manifest_id / "ledger.jsonl", manifest)
    tasks = _selected_first_scope_tasks(manifest)
    for task in tasks[:4]:
        ledger.append(str(task["task_id"]), "generation_started", {"generation_attempt": 1})
        ledger.append(str(task["task_id"]), "generation_completed", {"generation_path": "g.json"})

    report = progress.build_progress_report(
        manifest_path=manifest_path,
        experiment_root=tmp_path,
        stage="score",
        question_start=0,
        question_limit=2,
        repeat_limit=1,
    )

    assert report["progress"]["category_counts"] == {"ready_to_score": 4}
    assert report["recommendation"] == {
        "action": "run_guarded_score_canary_after_spend_review",
        "requires_spend": True,
        "reason": "4 selected task(s) have generation evidence and await scoring",
    }


def test_progress_surfaces_generation_failure_for_review(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "manifest.json")
    manifest = _read_manifest(manifest_path)
    manifest_id = str(manifest["manifest_id"])
    ledger = demo.ResumableLedger(tmp_path / manifest_id / "ledger.jsonl", manifest)
    task = list(manifest["tasks"])[0]
    assert isinstance(task, dict)
    task_id = str(task["task_id"])
    ledger.append(task_id, "generation_started", {"generation_attempt": 1})
    ledger.append(
        task_id,
        "failed",
        {
            "stage": "generation",
            "retry_permitted": False,
            "manual_review": True,
        },
    )

    report = progress.build_progress_report(
        manifest_path=manifest_path,
        experiment_root=tmp_path,
        stage="generation",
        question_start=0,
        question_limit=2,
        repeat_limit=1,
    )

    assert report["progress"]["category_counts"] == {
        "generation_failed": 1,
        "not_started": 3,
    }
    assert report["task_states"][task_id]["failure_stage"] == "generation"
    assert report["recommendation"]["action"] == "inspect_or_recover_generation_failure"

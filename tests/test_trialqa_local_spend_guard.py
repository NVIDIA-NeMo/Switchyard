# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_spend_guard as spend_guard


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _command(tmp_path: Path, *, spend_review_name: str = "spend-review.json") -> list[str]:
    return [
        "python",
        "-m",
        "benchmark.trialqa_local_canary",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--experiment-root",
        str(tmp_path / "experiments"),
        "--question-start",
        "0",
        "--question-limit",
        "4",
        "--repeat-limit",
        "1",
        "--spend-review",
        str(tmp_path / spend_review_name),
        "--yes-spend",
    ]


def _current_progress_verification(
    *,
    action: str = "run_guarded_generation_canary_after_spend_review",
    done_task_count: int = 0,
    remaining_task_count: int = 8,
    category_counts: dict[str, int] | None = None,
    ledger_record_count: int = 0,
    batch_lock_state: str = "missing",
) -> dict[str, object]:
    return {
        "status": "matched",
        "stage": "generation",
        "action": action,
        "requires_spend": True,
        "selected_task_count": 8,
        "done_task_count": done_task_count,
        "remaining_task_count": remaining_task_count,
        "category_counts": category_counts if category_counts is not None else {"not_started": 8},
        "ledger_record_count": ledger_record_count,
        "batch_lock_state": batch_lock_state,
    }


def _progress_report(
    *,
    action: str = "run_guarded_generation_canary_after_spend_review",
    done_task_count: int = 0,
    remaining_task_count: int = 8,
    category_counts: dict[str, int] | None = None,
    ledger_record_count: int = 0,
    batch_lock_state: str = "missing",
) -> dict[str, object]:
    return {
        "schema_version": "switchyard.trialqa_progress.v1",
        "manifest_id": "trialqa-full-test",
        "stage": "generation",
        "scope": {
            "question_start": 0,
            "question_limit": 4,
            "selected_task_count": 8,
            "condition": "both",
            "selected_repeat_indices": [1],
        },
        "recommendation": {"action": action, "requires_spend": True},
        "progress": {
            "selected_task_count": 8,
            "done_task_count": done_task_count,
            "remaining_task_count": remaining_task_count,
            "category_counts": category_counts if category_counts is not None else {"not_started": 8},
        },
        "ledger": {"record_count": ledger_record_count},
        "batch_lock": {"state": batch_lock_state},
    }


@pytest.fixture(autouse=True)
def _patch_fresh_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spend_guard.progress,
        "build_progress_report",
        lambda **_kwargs: _progress_report(),
    )


def _spend_review(
    path: Path,
    *,
    command: list[str],
    bundle_sha: str = "sha256:" + "1" * 64,
) -> Path:
    verification = _write_json(
        path.with_name("bundle-verification.json"),
        {
            "schema_version": "switchyard.trialqa_pre_spend_audit_bundle_verification.v1",
            "status": "passed",
            "bundle": {
                "path": str(path.with_name("bundle.json")),
                "sha256": bundle_sha,
            },
        },
    )
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_spend_review_packet.v1",
            "status": "ready_for_user_spend_decision",
            "authorized_by_packet": False,
            "manifest_id": "trialqa-full-test",
            "bundle_verification": {
                "path": str(verification),
                "bundle_sha256": bundle_sha,
            },
            "guarded_spend_scope": {
                "stage": "generation",
                "question_start": 0,
                "question_limit": 4,
                "repeat_limit": 1,
                "task_count": 8,
            },
            "current_progress_verification": _current_progress_verification(),
            "guarded_spend_command": {
                "command": command,
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
            "guarded_recovery_command": {
                "command": [*command[:-1], "--recover-interrupted", "--yes-spend"],
                "requires_yes_spend": True,
                "authorized_by_packet": False,
            },
        },
    )


def _fresh_bundle_report(bundle_sha: str) -> dict[str, object]:
    return {
        "status": "passed",
        "bundle": {"path": "bundle.json", "sha256": bundle_sha},
        "artifact_checks": [{"status": "matched"}],
        "source_file_checks": [{"status": "matched"}],
    }


def test_spend_guard_accepts_exact_reviewed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    review = _spend_review(tmp_path / "spend-review.json", command=command)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "1" * 64),
    )

    report = spend_guard.validate_spend_review_for_command(
        spend_review=review,
        expected_command=command,
        expected_stage="generation",
    )

    assert report["reviewed_command_matched"] is True
    assert report["spend_authorized_by_packet"] is False
    assert report["bundle"]["bundle_sha256"] == "sha256:" + "1" * 64
    assert report["current_progress"]["done_task_count"] == 0


def test_spend_guard_check_validates_reviewed_command_without_authorizing_spend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    review = _spend_review(tmp_path / "spend-review.json", command=command)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "1" * 64),
    )

    report = spend_guard.build_spend_guard_check(spend_review=review)

    assert report["schema_version"] == spend_guard.SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["spend_authorized"] is False
    assert report["next_stage_spend_authorized"] is False
    assert report["reviewed_command"]["command"] == command
    assert report["reviewed_command"]["contains_spend_review_guard"] is True


def test_spend_guard_rejects_mismatched_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed = _command(tmp_path)
    review = _spend_review(tmp_path / "spend-review.json", command=reviewed)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "1" * 64),
    )
    expected = list(reviewed)
    expected[expected.index("--question-limit") + 1] = "8"

    with pytest.raises(spend_guard.TrialQASpendGuardError, match="reviewed command"):
        spend_guard.validate_spend_review_for_command(
            spend_review=review,
            expected_command=expected,
            expected_stage="generation",
        )


def test_spend_guard_rejects_wrong_spend_review_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path, spend_review_name="other-review.json")
    review = _spend_review(tmp_path / "spend-review.json", command=command)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "1" * 64),
    )

    with pytest.raises(spend_guard.TrialQASpendGuardError, match="--spend-review"):
        spend_guard.validate_spend_review_for_command(
            spend_review=review,
            expected_command=command,
            expected_stage="generation",
        )


def test_spend_guard_rejects_stale_bundle_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    review = _spend_review(tmp_path / "spend-review.json", command=command)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "2" * 64),
    )

    with pytest.raises(spend_guard.TrialQASpendGuardError, match="stale"):
        spend_guard.validate_spend_review_for_command(
            spend_review=review,
            expected_command=command,
            expected_stage="generation",
        )


def test_spend_guard_rejects_fresh_progress_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    review = _spend_review(tmp_path / "spend-review.json", command=command)
    monkeypatch.setattr(
        spend_guard.bundle_verify,
        "verify_audit_bundle",
        lambda **_kwargs: _fresh_bundle_report("sha256:" + "1" * 64),
    )
    monkeypatch.setattr(
        spend_guard.progress,
        "build_progress_report",
        lambda **_kwargs: _progress_report(
            done_task_count=1,
            remaining_task_count=7,
            category_counts={"generated": 1, "not_started": 7},
            ledger_record_count=3,
        ),
    )

    with pytest.raises(spend_guard.TrialQASpendGuardError, match="fresh progress"):
        spend_guard.validate_spend_review_for_command(
            spend_review=review,
            expected_command=command,
            expected_stage="generation",
        )

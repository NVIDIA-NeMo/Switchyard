# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Last-mile no-spend guard for TrialQA canary commands.

The pre-spend audit bundle proves a frozen source/artifact snapshot, and the
spend-review packet exposes the exact guarded command an operator may choose to
run. This module is the final check performed inside a guarded canary before it
starts model or judge calls: the command must be the one reviewed, the reviewed
bundle must still match the current filesystem, and the selected ledger/lock
state must still be at the reviewed spend boundary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_audit_bundle_verify as bundle_verify  # noqa: E402
import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_progress as progress  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_spend_guard_check.v1"
JsonObject = dict[str, Any]


class TrialQASpendGuardError(RuntimeError):
    """The reviewed spend packet does not authorize this exact canary command."""


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQASpendGuardError(f"{label} must be an object")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrialQASpendGuardError(f"{label} must be a string list")
    return list(value)


def _guarded_command(
    review: Mapping[str, object],
    *,
    recover_interrupted: bool,
) -> list[str]:
    field = "guarded_recovery_command" if recover_interrupted else "guarded_spend_command"
    command = _require_mapping(review.get(field), field)
    argv = _require_string_list(command.get("command"), f"{field} command")
    if not argv or argv[-1] != "--yes-spend":
        raise TrialQASpendGuardError(f"{field} must end in --yes-spend")
    if command.get("requires_yes_spend") is not True:
        raise TrialQASpendGuardError(f"{field} must be marked requires_yes_spend=true")
    if command.get("authorized_by_packet") is not False:
        raise TrialQASpendGuardError(f"{field} must not be packet-authorized")
    return argv


def _option_value(argv: Sequence[str], option: str) -> str:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise TrialQASpendGuardError(f"reviewed command is missing {option}") from exc
    value_index = index + 1
    if value_index >= len(argv) or argv[value_index].startswith("--"):
        raise TrialQASpendGuardError(f"reviewed command has no value for {option}")
    return argv[value_index]


def _option_int(argv: Sequence[str], option: str) -> int:
    value = _option_value(argv, option)
    try:
        return int(value)
    except ValueError as exc:
        raise TrialQASpendGuardError(
            f"reviewed command has a non-integral value for {option}"
        ) from exc


def _require_matching_path(actual: str, expected: Path, label: str) -> None:
    if Path(actual).resolve() != expected.resolve():
        raise TrialQASpendGuardError(f"{label} does not match the spend review path")


def _require_stage_module(argv: Sequence[str], expected_stage: str) -> None:
    expected_module = {
        "generation": "benchmark.trialqa_local_canary",
        "score": "benchmark.trialqa_local_canary_score",
    }.get(expected_stage)
    if expected_module is None:
        raise TrialQASpendGuardError(f"unsupported spend stage {expected_stage!r}")
    if len(argv) < 3 or argv[1:3] != ["-m", expected_module]:
        raise TrialQASpendGuardError(
            f"reviewed command for {expected_stage!r} must target {expected_module}"
        )


def _verify_bundle_freshness(review: Mapping[str, object]) -> JsonObject:
    bundle_summary = _require_mapping(
        review.get("bundle_verification"),
        "spend-review bundle_verification",
    )
    verification_path = bundle_summary.get("path")
    expected_bundle_sha = bundle_summary.get("bundle_sha256")
    if not isinstance(verification_path, str) or not verification_path:
        raise TrialQASpendGuardError("spend review has no bundle verification path")
    if not isinstance(expected_bundle_sha, str) or not expected_bundle_sha:
        raise TrialQASpendGuardError("spend review has no bundle sha256")

    stored_verification = demo._read_json_object(
        Path(verification_path),
        "stored bundle verification",
    )
    stored_bundle = _require_mapping(stored_verification.get("bundle"), "stored bundle")
    bundle_path = stored_bundle.get("path")
    if not isinstance(bundle_path, str) or not bundle_path:
        raise TrialQASpendGuardError("stored bundle verification has no bundle path")

    fresh = bundle_verify.verify_audit_bundle(bundle_path=Path(bundle_path))
    if fresh.get("status") != "passed":
        raise TrialQASpendGuardError("fresh bundle verification did not pass")
    fresh_bundle = _require_mapping(fresh.get("bundle"), "fresh bundle")
    fresh_sha = fresh_bundle.get("sha256")
    if fresh_sha != expected_bundle_sha:
        raise TrialQASpendGuardError(
            f"spend-review bundle sha is stale: {fresh_sha!r} != {expected_bundle_sha!r}"
        )
    return {
        "bundle_path": bundle_path,
        "bundle_sha256": fresh_sha,
        "artifact_check_count": len(fresh.get("artifact_checks", []))
        if isinstance(fresh.get("artifact_checks"), list)
        else None,
        "source_file_check_count": len(fresh.get("source_file_checks", []))
        if isinstance(fresh.get("source_file_checks"), list)
        else None,
    }


def _current_progress_report(
    reviewed_command: Sequence[str],
    *,
    stage: str,
) -> JsonObject:
    return progress.build_progress_report(
        manifest_path=Path(_option_value(reviewed_command, "--manifest")),
        experiment_root=Path(_option_value(reviewed_command, "--experiment-root")),
        stage=cast(progress.Stage, stage),
        question_start=_option_int(reviewed_command, "--question-start"),
        question_limit=_option_int(reviewed_command, "--question-limit"),
        repeat_limit=_option_int(reviewed_command, "--repeat-limit"),
    )


def _compact_progress_summary(report: Mapping[str, object]) -> JsonObject:
    recommendation = _require_mapping(report.get("recommendation"), "current recommendation")
    progress_summary = _require_mapping(report.get("progress"), "current progress summary")
    ledger = _require_mapping(report.get("ledger"), "current progress ledger")
    batch_lock = _require_mapping(report.get("batch_lock"), "current progress batch_lock")
    return {
        "stage": report.get("stage"),
        "action": recommendation.get("action"),
        "requires_spend": recommendation.get("requires_spend"),
        "selected_task_count": progress_summary.get("selected_task_count"),
        "done_task_count": progress_summary.get("done_task_count"),
        "remaining_task_count": progress_summary.get("remaining_task_count"),
        "category_counts": progress_summary.get("category_counts"),
        "ledger_record_count": ledger.get("record_count"),
        "batch_lock_state": batch_lock.get("state"),
    }


def _verify_current_progress_freshness(
    review: Mapping[str, object],
    reviewed_command: Sequence[str],
    *,
    expected_stage: str,
    recover_interrupted: bool,
) -> JsonObject:
    reviewed_progress = _require_mapping(
        review.get("current_progress_verification"),
        "spend-review current_progress_verification",
    )
    scope = _require_mapping(review.get("guarded_spend_scope"), "guarded_spend_scope")
    fresh = _current_progress_report(reviewed_command, stage=expected_stage)
    if fresh.get("schema_version") != "switchyard.trialqa_progress.v1":
        raise TrialQASpendGuardError("fresh progress report has invalid schema_version")
    if review.get("manifest_id") is not None and fresh.get("manifest_id") != review.get("manifest_id"):
        raise TrialQASpendGuardError("fresh progress report belongs to a different manifest")
    if fresh.get("stage") != expected_stage:
        raise TrialQASpendGuardError("fresh progress report has the wrong stage")

    fresh_scope = _require_mapping(fresh.get("scope"), "fresh progress scope")
    expected_fields = {
        "question_start": scope.get("question_start"),
        "question_limit": scope.get("question_limit"),
        "selected_task_count": scope.get("task_count"),
        "condition": "both",
    }
    for field, expected in expected_fields.items():
        if fresh_scope.get(field) != expected:
            raise TrialQASpendGuardError(
                f"fresh progress scope differs at {field}: "
                f"{fresh_scope.get(field)!r} != {expected!r}"
            )
    repeat_limit = scope.get("repeat_limit")
    if (
        not isinstance(repeat_limit, int)
        or isinstance(repeat_limit, bool)
        or fresh_scope.get("selected_repeat_indices") != list(range(1, repeat_limit + 1))
    ):
        raise TrialQASpendGuardError("fresh progress scope has the wrong repeat indices")

    fresh_summary = _compact_progress_summary(fresh)
    if recover_interrupted:
        acceptable = (
            {"recover_interrupted_generation"}
            if expected_stage == "generation"
            else {"recover_interrupted_score"}
        )
        if fresh_summary["action"] not in acceptable or fresh_summary["requires_spend"] is not True:
            raise TrialQASpendGuardError(
                "fresh progress is not at an interrupted recovery boundary: "
                f"action={fresh_summary['action']!r}, "
                f"requires_spend={fresh_summary['requires_spend']!r}"
            )
    else:
        for field, actual in fresh_summary.items():
            if reviewed_progress.get(field) != actual:
                raise TrialQASpendGuardError(
                    f"fresh progress differs from the reviewed packet at {field}: "
                    f"{actual!r} != {reviewed_progress.get(field)!r}"
                )

    return {
        **fresh_summary,
        "status": "matched",
        "recover_interrupted": recover_interrupted,
    }


def validate_spend_review_for_command(
    *,
    spend_review: Path,
    expected_command: Sequence[str],
    expected_stage: str,
    recover_interrupted: bool = False,
) -> JsonObject:
    """Verify the reviewed packet still authorizes this exact spend boundary.

    The return value is intentionally compact so canary summaries can record the
    last-mile guard evidence without copying the full review packet.
    """

    review = demo._read_json_object(spend_review, "spend review")
    if review.get("schema_version") != "switchyard.trialqa_spend_review_packet.v1":
        raise TrialQASpendGuardError("spend review has invalid schema_version")
    if review.get("status") != "ready_for_user_spend_decision":
        raise TrialQASpendGuardError("spend review is not ready for a spend decision")
    if review.get("authorized_by_packet") is not False:
        raise TrialQASpendGuardError("spend review must not authorize spend")

    scope = _require_mapping(review.get("guarded_spend_scope"), "guarded_spend_scope")
    if scope.get("stage") != expected_stage:
        raise TrialQASpendGuardError(
            f"spend review stage {scope.get('stage')!r} does not match {expected_stage!r}"
        )

    reviewed_command = _guarded_command(review, recover_interrupted=recover_interrupted)
    expected = list(expected_command)
    if reviewed_command != expected:
        raise TrialQASpendGuardError("current canary command does not match the reviewed command")
    if "--spend-review" not in reviewed_command:
        raise TrialQASpendGuardError("reviewed command is missing --spend-review")
    if "--yes-spend" not in reviewed_command:
        raise TrialQASpendGuardError("reviewed command is missing --yes-spend")
    _require_matching_path(
        _option_value(reviewed_command, "--spend-review"),
        spend_review,
        "reviewed command --spend-review",
    )
    _require_stage_module(reviewed_command, expected_stage)

    freshness = _verify_bundle_freshness(review)
    current_progress = _verify_current_progress_freshness(
        review,
        reviewed_command,
        expected_stage=expected_stage,
        recover_interrupted=recover_interrupted,
    )
    return {
        "spend_review": str(spend_review),
        "stage": expected_stage,
        "recover_interrupted": recover_interrupted,
        "reviewed_command_matched": True,
        "bundle": freshness,
        "current_progress": current_progress,
        "spend_authorized_by_packet": False,
    }


def build_spend_guard_check(
    *,
    spend_review: Path,
    recover_interrupted: bool = False,
) -> JsonObject:
    """Run the last-mile guard against the reviewed command without spending."""

    review = demo._read_json_object(spend_review, "spend review")
    scope = _require_mapping(review.get("guarded_spend_scope"), "guarded_spend_scope")
    stage = scope.get("stage")
    if not isinstance(stage, str):
        raise TrialQASpendGuardError("spend review has no guarded stage")
    reviewed_command = _guarded_command(review, recover_interrupted=recover_interrupted)
    guard = validate_spend_review_for_command(
        spend_review=spend_review,
        expected_command=reviewed_command,
        expected_stage=stage,
        recover_interrupted=recover_interrupted,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "spend_authorized": False,
        "next_stage_spend_authorized": False,
        "spend_review": str(spend_review),
        "stage": stage,
        "recover_interrupted": recover_interrupted,
        "reviewed_command": {
            "command": reviewed_command,
            "shell_command": " ".join(reviewed_command),
            "requires_yes_spend": True,
            "contains_yes_spend": True,
            "contains_spend_review_guard": True,
        },
        "guard": guard,
        "review_note": (
            "This command is read-only and does not authorize spend. It proves "
            "the reviewed guarded command, current hash-bound audit bundle, "
            "and selected ledger/lock progress still match before an operator "
            "decides whether to run --yes-spend."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend-review", type=Path, required=True)
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="validate the reviewed recovery command instead of the first-run guarded command",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_spend_guard_check(
        spend_review=args.spend_review,
        recover_interrupted=args.recover_interrupted,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

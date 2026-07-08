# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a compact no-spend review packet before a TrialQA spend decision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_gate as gate  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_spend_review_packet.v1"
JsonObject = dict[str, Any]
EXPECTED_NEXT_STEP_BY_BUNDLE_STATE = {
    "awaiting_generation_canary_spend_authorization": {
        ("run_guarded_generation_canary", "generation_preflight"),
        ("expand_generation_scope", "generation_expansion_preflight"),
    },
    "awaiting_score_canary_spend_authorization": {
        ("run_guarded_score_canary", "score_preflight"),
    },
}


class TrialQASpendReviewError(RuntimeError):
    """The no-spend artifacts are not ready for a user spend decision."""


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQASpendReviewError(f"{label} must be an object")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrialQASpendReviewError(f"{label} must be a string list")
    return list(value)


def _attach_spend_review_guard(
    guarded_command: Sequence[str],
    spend_review_path: Path | None,
) -> list[str]:
    command = list(guarded_command)
    if spend_review_path is None:
        return command
    if not command or command[-1] != "--yes-spend":
        raise TrialQASpendReviewError("guarded spend command must end in --yes-spend")
    if "--spend-review" in command:
        raise TrialQASpendReviewError("guarded spend command already has --spend-review")
    return [*command[:-1], "--spend-review", str(spend_review_path), "--yes-spend"]


def _argv_value(argv: Sequence[str], option: str) -> str:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise TrialQASpendReviewError(f"guarded spend command is missing {option}") from exc
    value_index = index + 1
    if value_index >= len(argv) or argv[value_index].startswith("--"):
        raise TrialQASpendReviewError(f"guarded spend command has no value for {option}")
    return argv[value_index]


def _argv_int(argv: Sequence[str], option: str) -> int:
    value = _argv_value(argv, option)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TrialQASpendReviewError(
            f"guarded spend command has a non-integral value for {option}"
        ) from exc
    return parsed


def _guarded_spend_scope(guarded_command: Sequence[str], command_kind: object) -> JsonObject:
    if command_kind == "guarded_generation_canary":
        stage = "generation"
    elif command_kind == "guarded_score_canary":
        stage = "score"
    else:
        raise TrialQASpendReviewError(f"unsupported guarded spend command kind {command_kind!r}")

    question_start = _argv_int(guarded_command, "--question-start")
    question_limit = _argv_int(guarded_command, "--question-limit")
    repeat_limit = _argv_int(guarded_command, "--repeat-limit")
    workers = _argv_int(guarded_command, "--workers")
    max_generation_attempts = _argv_int(guarded_command, "--max-generation-attempts")
    if question_start < 0:
        raise TrialQASpendReviewError("guarded spend command has a negative question start")
    if question_limit < 1:
        raise TrialQASpendReviewError("guarded spend command has a non-positive question limit")
    if repeat_limit < 1:
        raise TrialQASpendReviewError("guarded spend command has a non-positive repeat limit")
    if workers < 1:
        raise TrialQASpendReviewError("guarded spend command has a non-positive worker count")
    if max_generation_attempts < 1:
        raise TrialQASpendReviewError(
            "guarded spend command has a non-positive max generation attempt count"
        )

    paired_draw_count = question_limit * repeat_limit
    task_count = paired_draw_count * 2
    question_stop_inclusive = question_start + question_limit - 1
    return {
        "stage": stage,
        "question_start": question_start,
        "question_limit": question_limit,
        "question_stop_inclusive": question_stop_inclusive,
        "repeat_limit": repeat_limit,
        "paired_draw_count": paired_draw_count,
        "arm_count": 2,
        "task_count": task_count,
        "expected_model_calls": task_count if stage == "generation" else 0,
        "expected_judge_calls": task_count if stage == "score" else 0,
        "configured_worker_limit": workers,
        "configured_max_generation_attempts": max_generation_attempts,
        "maximum_generation_attempts": (
            task_count * max_generation_attempts if stage == "generation" else 0
        ),
        "scope_label": (
            f"q{question_start}-q{question_stop_inclusive}, "
            f"{repeat_limit} repeat(s), 2 arms, {task_count} task(s)"
        ),
    }


def _progress_monitor_command(
    guarded_command: Sequence[str],
    *,
    stage: object,
) -> JsonObject:
    if stage not in {"generation", "score"}:
        raise TrialQASpendReviewError(f"unsupported monitor stage {stage!r}")
    command = [
        guarded_command[0],
        "-m",
        "benchmark.trialqa_local_progress",
        "--manifest",
        _argv_value(guarded_command, "--manifest"),
        "--experiment-root",
        _argv_value(guarded_command, "--experiment-root"),
        "--stage",
        str(stage),
        "--question-start",
        str(_argv_int(guarded_command, "--question-start")),
        "--question-limit",
        str(_argv_int(guarded_command, "--question-limit")),
        "--repeat-limit",
        str(_argv_int(guarded_command, "--repeat-limit")),
    ]
    return {
        "command": command,
        "shell_command": " ".join(command),
        "contains_yes_spend": False,
        "requires_spend": False,
        "stage": stage,
        "review_note": (
            "Read-only ledger progress monitor; safe to run before, during, "
            "or after the guarded spend command."
        ),
    }


def _guarded_recovery_command(
    guarded_command: Sequence[str],
    *,
    stage: object,
) -> JsonObject:
    if stage not in {"generation", "score"}:
        raise TrialQASpendReviewError(f"unsupported recovery stage {stage!r}")
    if not guarded_command or guarded_command[-1] != "--yes-spend":
        raise TrialQASpendReviewError("guarded recovery command must end in --yes-spend")
    if "--recover-interrupted" in guarded_command:
        command = list(guarded_command)
    else:
        command = [*guarded_command[:-1], "--recover-interrupted", "--yes-spend"]
    if command.count("--recover-interrupted") != 1:
        raise TrialQASpendReviewError("guarded recovery command has duplicate recovery flags")
    if command[-2:] != ["--recover-interrupted", "--yes-spend"]:
        raise TrialQASpendReviewError(
            "guarded recovery command must place --recover-interrupted before --yes-spend"
        )
    return {
        "command": command,
        "shell_command": " ".join(command),
        "requires_yes_spend": True,
        "authorized_by_audit": False,
        "authorized_by_packet": False,
        "recovery_flag": "--recover-interrupted",
        "stage": stage,
        "review_note": (
            "Use only if the read-only progress monitor reports an interrupted "
            f"{stage} canary; still requires explicit spend approval."
        ),
    }


def _decision_policy_summary(*, stage: object) -> JsonObject:
    if stage not in {"generation", "score"}:
        raise TrialQASpendReviewError(f"unsupported decision-policy stage {stage!r}")
    boundary = (
        {
            "stage": "generation",
            "post_spend_gate": "operational",
            "promote_decision": "promote_to_score",
            "kill_decision": "kill",
            "judge_spend_deferred": True,
        }
        if stage == "generation"
        else {
            "stage": "score",
            "post_spend_gate": "promotion",
            "promote_decision": "promote_to_next_cohort",
            "kill_decision": "kill",
            "judge_spend_deferred": False,
        }
    )
    return {
        "name": gate.POLICY_NAME,
        "thresholds": {
            "token_reduction_min": gate.TOKEN_REDUCTION_MIN,
            "operational_call_reduction_min": gate.OPERATIONAL_CALL_REDUCTION_MIN,
            "quality_delta_min": gate.QUALITY_DELTA_MIN,
            "quality_confidence_level": gate.QUALITY_CONFIDENCE_LEVEL,
            "futility_confidence_level": gate.FUTILITY_CONFIDENCE_LEVEL,
        },
        "quality_modes": {
            "interim": gate.INTERIM_QUALITY_MODE,
            "confirmatory": gate.CONFIRMATORY_QUALITY_MODE,
        },
        "population_and_retry_policy": {
            "analysis_population": "intention-to-treat",
            "completed_draw_policy": "terminal-no-retry-or-replacement",
            "empty_answer_policy": "score-zero",
            "performance_retry_policy": "zero-null-eof-retries",
        },
        "current_boundary": boundary,
        "review_note": (
            "Frozen gate policy that will interpret the post-spend checkpoint; "
            "do not revise thresholds after seeing TrialQA outcomes."
        ),
    }


def _expected_repeat_indices(repeat_limit: object) -> list[int]:
    if not isinstance(repeat_limit, int) or isinstance(repeat_limit, bool) or repeat_limit < 1:
        raise TrialQASpendReviewError("guarded spend scope has invalid repeat_limit")
    return list(range(1, repeat_limit + 1))


def _current_progress_verification(
    *,
    progress_path: Path,
    manifest_id: object,
    guarded_spend_scope: Mapping[str, object],
) -> JsonObject:
    progress = demo._read_json_object(progress_path, "current progress report")
    if progress.get("schema_version") != "switchyard.trialqa_progress.v1":
        raise TrialQASpendReviewError("current progress report has invalid schema_version")
    if progress.get("manifest_id") != manifest_id:
        raise TrialQASpendReviewError("current progress report belongs to a different manifest")
    stage = guarded_spend_scope.get("stage")
    if progress.get("stage") != stage:
        raise TrialQASpendReviewError("current progress report has the wrong stage")
    scope = _require_mapping(progress.get("scope"), "current progress scope")
    expected_fields = {
        "question_start": guarded_spend_scope.get("question_start"),
        "question_limit": guarded_spend_scope.get("question_limit"),
        "selected_task_count": guarded_spend_scope.get("task_count"),
        "condition": "both",
    }
    for field, expected in expected_fields.items():
        if scope.get(field) != expected:
            raise TrialQASpendReviewError(
                f"current progress scope differs at {field}: "
                f"{scope.get(field)!r} != {expected!r}"
            )
    expected_repeats = _expected_repeat_indices(guarded_spend_scope.get("repeat_limit"))
    if scope.get("selected_repeat_indices") != expected_repeats:
        raise TrialQASpendReviewError("current progress scope has the wrong repeat indices")

    recommendation = _require_mapping(
        progress.get("recommendation"),
        "current progress recommendation",
    )
    action = recommendation.get("action")
    requires_spend = recommendation.get("requires_spend")
    acceptable_actions = (
        {
            "run_guarded_generation_canary_after_spend_review",
            "resume_guarded_generation_if_still_authorized",
        }
        if stage == "generation"
        else {"run_guarded_score_canary_after_spend_review"}
    )
    if action not in acceptable_actions or requires_spend is not True:
        raise TrialQASpendReviewError(
            "current progress is not at a clean spend boundary: "
            f"action={action!r}, requires_spend={requires_spend!r}"
        )

    progress_summary = _require_mapping(progress.get("progress"), "current progress summary")
    ledger = _require_mapping(progress.get("ledger"), "current progress ledger")
    batch_lock = _require_mapping(progress.get("batch_lock"), "current progress batch_lock")
    return {
        "path": str(progress_path),
        "status": "matched",
        "stage": stage,
        "action": action,
        "requires_spend": True,
        "selected_task_count": progress_summary.get("selected_task_count"),
        "done_task_count": progress_summary.get("done_task_count"),
        "remaining_task_count": progress_summary.get("remaining_task_count"),
        "category_counts": progress_summary.get("category_counts"),
        "ledger_record_count": ledger.get("record_count"),
        "batch_lock_state": batch_lock.get("state"),
        "review_note": (
            "Current read-only progress matched the reviewed scope and is still "
            "at a clean spend boundary."
        ),
    }


def _all_matched(items: object, label: str) -> int:
    if not isinstance(items, list):
        raise TrialQASpendReviewError(f"{label} must be a list")
    for item in items:
        entry = _require_mapping(item, f"{label} entry")
        if entry.get("status") != "matched":
            raise TrialQASpendReviewError(f"{label} contains non-matching entry")
    return len(items)


def _optional_no_spend_command(value: object, label: str) -> JsonObject | None:
    if value is None:
        return None
    command = _require_mapping(value, label)
    argv = _require_string_list(command.get("command"), f"{label} command")
    if "--yes-spend" in argv:
        raise TrialQASpendReviewError(f"{label} must not include --yes-spend")
    if command.get("requires_spend") is not False:
        raise TrialQASpendReviewError(f"{label} must be marked requires_spend=false")
    if command.get("contains_yes_spend") is not False:
        raise TrialQASpendReviewError(f"{label} must be marked contains_yes_spend=false")
    return {
        **dict(command),
        "command": argv,
        "requires_spend": False,
        "contains_yes_spend": False,
    }


def _require_checkpoint_kind(command: JsonObject, *, stage: object) -> None:
    if stage not in {"generation", "score"}:
        raise TrialQASpendReviewError(f"unsupported checkpoint stage {stage!r}")
    expected = {
        "generation": "post_generation_checkpoint",
        "score": "post_score_checkpoint",
    }[stage]
    if command.get("kind") != expected:
        raise TrialQASpendReviewError(
            f"post_spend_checkpoint_command kind must be {expected!r} for {stage} spend"
        )


def _post_spend_acceptance_criteria(
    *,
    guarded_command: Sequence[str],
    stage: object,
    post_spend_checkpoint: Mapping[str, object] | None,
) -> JsonObject:
    if stage == "generation":
        gate_option = "--gate-output"
        required_gate = "operational"
        promote_decision = "promote_to_score"
        next_boundary_if_promoted = "score_spend_review"
        checkpoint_kind = "post_generation_checkpoint"
        spend_type_before_checkpoint = "judge"
    elif stage == "score":
        gate_option = "--promotion-gate-output"
        required_gate = "promotion"
        promote_decision = "promote_to_next_cohort"
        next_boundary_if_promoted = "generation_expansion_spend_review_or_complete"
        checkpoint_kind = "post_score_checkpoint"
        spend_type_before_checkpoint = "model"
    else:
        raise TrialQASpendReviewError(f"unsupported acceptance stage {stage!r}")

    checkpoint_command = (
        post_spend_checkpoint.get("command")
        if isinstance(post_spend_checkpoint, Mapping)
        else None
    )
    return {
        "stage": stage,
        "required_gate": required_gate,
        "required_gate_artifact": _argv_value(guarded_command, gate_option),
        "required_gate_schema_version": "switchyard.trialqa_gate_report.v3",
        "promote_decision": promote_decision,
        "kill_decision": "kill",
        "next_no_spend_checkpoint_kind": checkpoint_kind,
        "checkpoint_command_available": isinstance(checkpoint_command, list),
        "must_run_checkpoint_before_more_spend": True,
        "next_boundary_if_promoted": next_boundary_if_promoted,
        f"{spend_type_before_checkpoint}_spend_before_checkpoint_allowed": False,
        "review_note": (
            f"After the guarded {stage} canary finishes, inspect the "
            f"{required_gate} gate at the listed artifact path. Proceed only "
            f"if its decision is {promote_decision!r}; otherwise stop on "
            "the kill decision. Run the no-spend checkpoint before any "
            "additional spend."
        ),
    }


def build_spend_review_packet(
    *,
    preflight_path: Path,
    bundle_verification_path: Path,
    next_step_path: Path,
    progress_path: Path | None = None,
    spend_review_path: Path | None = None,
) -> JsonObject:
    """Build a compact review packet that never authorizes spend."""

    preflight = demo._read_json_object(preflight_path, "preflight report")
    verification = demo._read_json_object(bundle_verification_path, "bundle verification")
    next_step = demo._read_json_object(next_step_path, "next-step plan")
    if preflight.get("schema_version") not in {
        "switchyard.trialqa_no_spend_preflight.v1",
        "switchyard.trialqa_no_spend_score_preflight.v1",
    }:
        raise TrialQASpendReviewError("preflight report has invalid schema_version")
    if preflight.get("status") != "passed":
        raise TrialQASpendReviewError("preflight report has not passed")
    if preflight.get("spend_authorized") is not False:
        raise TrialQASpendReviewError("preflight report must not authorize spend")
    if verification.get("schema_version") != "switchyard.trialqa_pre_spend_audit_bundle_verification.v1":
        raise TrialQASpendReviewError("bundle verification has invalid schema_version")
    if verification.get("status") != "passed":
        raise TrialQASpendReviewError("bundle verification has not passed")
    if next_step.get("schema_version") != "switchyard.trialqa_next_step_plan.v1":
        raise TrialQASpendReviewError("next-step plan has invalid schema_version")
    if next_step.get("terminal") is not False:
        raise TrialQASpendReviewError("next-step plan is terminal and has no spend boundary")

    next_command = _require_mapping(preflight.get("next_command"), "preflight next_command")
    raw_guarded_command = _require_string_list(next_command.get("command"), "guarded command")
    if not raw_guarded_command or raw_guarded_command[-1] != "--yes-spend":
        raise TrialQASpendReviewError("guarded spend command must end in --yes-spend")
    if next_command.get("requires_yes_spend") is not True:
        raise TrialQASpendReviewError("guarded spend command is not marked spend-gated")
    if next_command.get("authorized_by_audit") is not False:
        raise TrialQASpendReviewError("guarded spend command must not be audit-authorized")
    guarded_command = _attach_spend_review_guard(raw_guarded_command, spend_review_path)
    guarded_spend_scope = _guarded_spend_scope(guarded_command, next_command.get("kind"))
    monitor_command = _progress_monitor_command(
        guarded_command,
        stage=guarded_spend_scope["stage"],
    )
    recovery_command = _guarded_recovery_command(
        guarded_command,
        stage=guarded_spend_scope["stage"],
    )
    decision_policy = _decision_policy_summary(stage=guarded_spend_scope["stage"])

    bundle = _require_mapping(verification.get("bundle"), "verification bundle")
    bundle_state = bundle.get("bundle_state")
    if not isinstance(bundle_state, str):
        raise TrialQASpendReviewError("bundle state must be a string")
    expected_pairs = EXPECTED_NEXT_STEP_BY_BUNDLE_STATE.get(bundle_state)
    if expected_pairs is None:
        raise TrialQASpendReviewError(f"unsupported bundle state {bundle_state!r}")
    safe_command = _require_mapping(next_step.get("safe_next_command"), "safe_next_command")
    next_step_pair = (next_step.get("action"), safe_command.get("kind"))
    if next_step_pair not in expected_pairs:
        raise TrialQASpendReviewError(
            "next-step action/safe command kind does not match bundle state"
        )
    safe_argv = _require_string_list(safe_command.get("command"), "safe command")
    if "--yes-spend" in safe_argv:
        raise TrialQASpendReviewError("safe next command must not include --yes-spend")
    artifact_check_count = _all_matched(verification.get("artifact_checks"), "artifact_checks")
    source_file_check_count = _all_matched(
        verification.get("source_file_checks"),
        "source_file_checks",
    )
    if source_file_check_count == 0:
        raise TrialQASpendReviewError("bundle verification has no source-file checks")
    post_spend_checkpoint = _optional_no_spend_command(
        preflight.get("post_spend_checkpoint_command"),
        "post_spend_checkpoint_command",
    )
    if post_spend_checkpoint is not None:
        _require_checkpoint_kind(
            post_spend_checkpoint,
            stage=guarded_spend_scope["stage"],
        )
    post_spend_acceptance = _post_spend_acceptance_criteria(
        guarded_command=guarded_command,
        stage=guarded_spend_scope["stage"],
        post_spend_checkpoint=post_spend_checkpoint,
    )
    current_progress = (
        _current_progress_verification(
            progress_path=progress_path,
            manifest_id=preflight.get("manifest_id"),
            guarded_spend_scope=guarded_spend_scope,
        )
        if progress_path is not None
        else None
    )

    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_user_spend_decision",
        "authorized_by_packet": False,
        "manifest_id": preflight.get("manifest_id"),
        "bundle_state": bundle_state,
        "preflight": {
            "path": str(preflight_path),
            "status": preflight.get("status"),
            "next_command_kind": next_command.get("kind"),
        },
        "bundle_verification": {
            "path": str(bundle_verification_path),
            "status": verification.get("status"),
            "bundle_sha256": bundle.get("sha256"),
            "artifact_check_count": artifact_check_count,
            "source_file_check_count": source_file_check_count,
        },
        "next_step": {
            "path": str(next_step_path),
            "action": next_step.get("action"),
            "safe_command_kind": safe_command.get("kind"),
        },
        "safe_no_spend_command": {
            "command": safe_argv,
            "shell_command": safe_command.get("shell_command"),
            "contains_yes_spend": False,
        },
        "guarded_spend_command": {
            "command": guarded_command,
            "shell_command": " ".join(guarded_command),
            "requires_yes_spend": True,
            "authorized_by_audit": False,
            "authorized_by_packet": False,
            "requires_spend_review_guard": spend_review_path is not None,
        },
        "guarded_spend_scope": guarded_spend_scope,
        "decision_policy": decision_policy,
        "post_spend_acceptance_criteria": post_spend_acceptance,
        "progress_monitor_command": monitor_command,
        "guarded_recovery_command": recovery_command,
        "review_note": (
            "This packet verifies pre-spend evidence and exposes the guarded command, "
            "but it does not authorize model or judge spend."
        ),
    }
    if post_spend_checkpoint is not None:
        packet["post_spend_checkpoint_command"] = post_spend_checkpoint
    if current_progress is not None:
        packet["current_progress_verification"] = current_progress
    return packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--bundle-verification", type=Path, required=True)
    parser.add_argument("--next-step", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_spend_review_packet(
        preflight_path=args.preflight,
        bundle_verification_path=args.bundle_verification,
        next_step_path=args.next_step,
        progress_path=args.progress,
        spend_review_path=args.output,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

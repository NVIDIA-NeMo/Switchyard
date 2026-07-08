# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact operator summary for the staged local TrialQA validation ladder.

The spend-review and goal-audit artifacts intentionally contain enough detail
to audit the next boundary, but they are large. This command distills those
artifacts into the small set of facts an operator needs before deciding whether
to run the next guarded command. It is read-only and never authorizes spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_decision_summary.v1"
JsonObject = dict[str, Any]


class TrialQADecisionSummaryError(RuntimeError):
    """The staged artifacts cannot be summarized safely."""


@dataclass(frozen=True)
class DecisionSummaryConfig:
    spend_review: Path
    goal_audit: Path
    decision_summary_output: Path | None = None


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQADecisionSummaryError(f"{label} must be an object")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrialQADecisionSummaryError(f"{label} must be a string list")
    return list(value)


def _safe_command(value: object, label: str) -> JsonObject:
    command = _require_mapping(value, label)
    argv = _require_string_list(command.get("command"), f"{label} command")
    if "--yes-spend" in argv:
        raise TrialQADecisionSummaryError(f"{label} must not contain --yes-spend")
    return {
        "command": argv,
        "shell_command": command.get("shell_command"),
        "contains_yes_spend": False,
    }


def _guarded_command(value: object) -> JsonObject:
    guarded = _require_mapping(value, "guarded_spend_command")
    argv = _require_string_list(guarded.get("command"), "guarded spend command")
    if not argv or argv[-1] != "--yes-spend":
        raise TrialQADecisionSummaryError("guarded spend command must end in --yes-spend")
    if "--spend-review" not in argv:
        raise TrialQADecisionSummaryError("guarded spend command must include --spend-review")
    if guarded.get("requires_yes_spend") is not True:
        raise TrialQADecisionSummaryError("guarded spend command is not marked yes-spend gated")
    if guarded.get("authorized_by_packet") is not False:
        raise TrialQADecisionSummaryError("guarded spend command must not be packet-authorized")
    return {
        "command": argv,
        "shell_command": guarded.get("shell_command"),
        "requires_yes_spend": True,
        "authorized_by_packet": False,
    }


def _gate_inspection_output(gate_path: Path) -> Path:
    name = gate_path.name
    if name.startswith("gate-operational-"):
        tail = name.removeprefix("gate-operational-")
    elif name.startswith("gate-promotion-"):
        tail = name.removeprefix("gate-promotion-")
    else:
        tail = name
    return gate_path.with_name(f"gate-inspection-{tail}")


def _spend_guard_check_output(spend_review_path: Path) -> Path:
    name = spend_review_path.name
    tail = name.removeprefix("spend-review-") if name.startswith("spend-review-") else name
    return spend_review_path.with_name(f"spend-guard-check-{tail}")


def _spend_guard_check_command(
    *,
    config: DecisionSummaryConfig,
    guarded_command: Mapping[str, object],
) -> JsonObject:
    guarded_argv = _require_string_list(guarded_command.get("command"), "guarded command")
    command = [
        guarded_argv[0],
        "-m",
        "benchmark.trialqa_local_spend_guard",
        "--spend-review",
        str(config.spend_review),
        "--output",
        str(_spend_guard_check_output(config.spend_review)),
    ]
    return {
        "command": command,
        "shell_command": " ".join(command),
        "contains_yes_spend": False,
        "review_note": (
            "Run immediately before approving spend; this rechecks the reviewed "
            "guarded command, current hash-bound bundle, and selected ledger/lock "
            "progress without making model or judge calls."
        ),
    }


def _gate_inspection_command(
    *,
    config: DecisionSummaryConfig,
    guarded_command: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> JsonObject:
    gate_artifact = acceptance.get("required_gate_artifact")
    if not isinstance(gate_artifact, str) or not gate_artifact:
        raise TrialQADecisionSummaryError("acceptance criteria has no required gate artifact")
    guarded_argv = _require_string_list(guarded_command.get("command"), "guarded command")
    if config.decision_summary_output is None:
        return {
            "command_available": False,
            "contains_yes_spend": False,
            "review_note": (
                "Pass --output when generating the decision summary to make "
                "the post-spend gate-inspection command copy/pasteable."
            ),
        }
    command = [
        guarded_argv[0],
        "-m",
        "benchmark.trialqa_local_gate_inspect",
        "--gate",
        gate_artifact,
        "--decision-summary",
        str(config.decision_summary_output),
        "--output",
        str(_gate_inspection_output(Path(gate_artifact))),
    ]
    return {
        "command": command,
        "shell_command": " ".join(command),
        "command_available": True,
        "contains_yes_spend": False,
        "review_note": (
            "Run after the guarded canary writes the required gate; this "
            "inspects the gate and still does not authorize next-stage spend."
        ),
    }


def _missing_goal_evidence(goal_audit: Mapping[str, object]) -> list[JsonObject]:
    requirements = goal_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQADecisionSummaryError("goal audit requirements must be a list")
    missing: list[JsonObject] = []
    for item in requirements:
        if not isinstance(item, Mapping):
            raise TrialQADecisionSummaryError("goal audit requirement must be an object")
        if item.get("status") == "missing":
            missing.append(
                {
                    "id": item.get("id"),
                    "evidence": item.get("evidence"),
                }
            )
    return missing


def _failed_goal_evidence(goal_audit: Mapping[str, object]) -> list[JsonObject]:
    requirements = goal_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQADecisionSummaryError("goal audit requirements must be a list")
    failed: list[JsonObject] = []
    for item in requirements:
        if not isinstance(item, Mapping):
            raise TrialQADecisionSummaryError("goal audit requirement must be an object")
        if item.get("status") == "failed":
            failed.append(
                {
                    "id": item.get("id"),
                    "evidence": item.get("evidence"),
                }
            )
    return failed


def _goal_requirement_summary(goal_audit: Mapping[str, object]) -> JsonObject:
    summary = goal_audit.get("requirement_summary")
    if not isinstance(summary, Mapping):
        raise TrialQADecisionSummaryError("goal audit requirement_summary must be an object")
    required_missing = summary.get("required_missing_ids")
    required_failed = summary.get("required_failed_ids")
    status_counts = summary.get("status_counts")
    total = summary.get("total")
    if not isinstance(total, int) or total <= 0:
        raise TrialQADecisionSummaryError("goal audit requirement_summary total is invalid")
    if not isinstance(status_counts, Mapping):
        raise TrialQADecisionSummaryError("goal audit requirement_summary status_counts is invalid")
    if not isinstance(required_missing, list) or not all(
        isinstance(item, str) for item in required_missing
    ):
        raise TrialQADecisionSummaryError(
            "goal audit requirement_summary required_missing_ids is invalid"
        )
    if not isinstance(required_failed, list) or not all(
        isinstance(item, str) for item in required_failed
    ):
        raise TrialQADecisionSummaryError(
            "goal audit requirement_summary required_failed_ids is invalid"
        )
    if required_failed:
        raise TrialQADecisionSummaryError("goal audit has failed required requirements")
    return dict(summary)


def _next_required_action(
    goal_audit: Mapping[str, object],
    *,
    stage: object,
) -> JsonObject:
    action = goal_audit.get("next_required_action")
    if not isinstance(action, Mapping):
        raise TrialQADecisionSummaryError("goal audit next_required_action must be an object")
    action_name = action.get("action")
    if stage == "score":
        expected = "request_explicit_score_canary_spend_approval"
    else:
        expected = "request_explicit_generation_canary_spend_approval"
    if action_name != expected:
        raise TrialQADecisionSummaryError("goal audit next_required_action does not match stage")
    if action.get("requires_spend") is not True:
        raise TrialQADecisionSummaryError("goal audit next_required_action must require spend")
    instruction = action.get("instruction")
    if not isinstance(instruction, str) or "--yes-spend" not in instruction:
        raise TrialQADecisionSummaryError(
            "goal audit next_required_action must mention --yes-spend"
        )
    return dict(action)


def _proved_setup_evidence(goal_audit: Mapping[str, object]) -> list[JsonObject]:
    requirements = goal_audit.get("requirements")
    if not isinstance(requirements, list):
        raise TrialQADecisionSummaryError("goal audit requirements must be a list")
    selected_ids = {
        "prospective_manifest_bound",
        "reference_workflow_alignment_bound",
        "local_switchyard_trialqa_transfer_runtime_bound",
        "switchyard_only_skill_distillation_ab_invariant_bound",
        "frozen_promotion_kill_policy_bound",
        "human_spend_review_packet_ready",
    }
    proved: list[JsonObject] = []
    for item in requirements:
        if not isinstance(item, Mapping):
            raise TrialQADecisionSummaryError("goal audit requirement must be an object")
        if item.get("id") in selected_ids and item.get("status") == "proved":
            proved.append(
                {
                    "id": item.get("id"),
                    "evidence": item.get("evidence"),
                }
            )
    return proved


def _stage_status(stage: object) -> str:
    if stage == "generation":
        return "awaiting_explicit_generation_spend_authorization"
    if stage == "score":
        return "awaiting_explicit_score_spend_authorization"
    raise TrialQADecisionSummaryError(f"unsupported spend stage {stage!r}")


def _operator_checklist(
    *,
    stage: object,
    acceptance: Mapping[str, object],
) -> list[JsonObject]:
    gate = acceptance.get("required_gate")
    gate_path = acceptance.get("required_gate_artifact")
    promote_decision = acceptance.get("promote_decision")
    kill_decision = acceptance.get("kill_decision")
    checkpoint_kind = acceptance.get("next_no_spend_checkpoint_kind")
    more_spend = "judge" if stage == "generation" else "model"
    return [
        {
            "id": "review_packet",
            "requires_spend": False,
            "instruction": "Review this summary and the spend-review packet before approving spend.",
        },
        {
            "id": "validate_spend_guard",
            "requires_spend": False,
            "instruction": (
                "Run the pre-spend guard check to prove the reviewed command, "
                "hash-bound bundle, and selected ledger/lock progress are still current."
            ),
        },
        {
            "id": "run_guarded_canary_if_approved",
            "requires_spend": True,
            "instruction": (
                "Run the guarded command only after explicit user approval; "
                "it must include the reviewed --spend-review guard and end with --yes-spend."
            ),
        },
        {
            "id": "monitor_without_spend",
            "requires_spend": False,
            "instruction": "Use the progress monitor command while waiting instead of starting a larger benchmark.",
        },
        {
            "id": "inspect_post_spend_gate",
            "requires_spend": False,
            "instruction": (
                f"After the canary finishes, inspect the {gate!r} gate at {gate_path!r}."
            ),
        },
        {
            "id": "promote_or_kill",
            "requires_spend": False,
            "instruction": (
                f"Continue only on decision {promote_decision!r}; stop on "
                f"decision {kill_decision!r}."
            ),
        },
        {
            "id": "checkpoint_before_more_spend",
            "requires_spend": False,
            "instruction": (
                f"Run the {checkpoint_kind!r} no-spend checkpoint before any "
                f"additional {more_spend} spend."
            ),
        },
    ]


def build_decision_summary(config: DecisionSummaryConfig) -> JsonObject:
    """Return a compact, non-authorizing summary for the next spend boundary."""

    spend_review = demo._read_json_object(config.spend_review, "spend review")
    goal_audit = demo._read_json_object(config.goal_audit, "goal audit")
    if spend_review.get("schema_version") != "switchyard.trialqa_spend_review_packet.v1":
        raise TrialQADecisionSummaryError("spend review has invalid schema_version")
    if goal_audit.get("schema_version") != "switchyard.trialqa_goal_audit.v1":
        raise TrialQADecisionSummaryError("goal audit has invalid schema_version")
    if spend_review.get("status") != "ready_for_user_spend_decision":
        raise TrialQADecisionSummaryError("spend review is not ready for a spend decision")
    if spend_review.get("authorized_by_packet") is not False:
        raise TrialQADecisionSummaryError("spend review must not authorize spend")
    if goal_audit.get("spend_authorized") is not False:
        raise TrialQADecisionSummaryError("goal audit must not authorize spend")

    scope = _require_mapping(spend_review.get("guarded_spend_scope"), "guarded spend scope")
    stage = scope.get("stage")
    expected_goal_status = (
        "ready_for_score_spend_decision"
        if stage == "score"
        else "ready_for_generation_spend_decision"
    )
    if goal_audit.get("status") != expected_goal_status:
        raise TrialQADecisionSummaryError("goal audit status does not match spend stage")
    guarded = _guarded_command(spend_review.get("guarded_spend_command"))
    monitor = _safe_command(spend_review.get("progress_monitor_command"), "progress monitor command")
    safe_preflight = _safe_command(spend_review.get("safe_no_spend_command"), "safe no-spend command")
    checkpoint = _safe_command(
        spend_review.get("post_spend_checkpoint_command"),
        "post-spend checkpoint command",
    )
    acceptance = _require_mapping(
        spend_review.get("post_spend_acceptance_criteria"),
        "post-spend acceptance criteria",
    )
    if acceptance.get("stage") != stage:
        raise TrialQADecisionSummaryError("acceptance criteria stage does not match spend scope")
    if acceptance.get("must_run_checkpoint_before_more_spend") is not True:
        raise TrialQADecisionSummaryError("acceptance criteria must require a checkpoint")
    gate_inspection = _gate_inspection_command(
        config=config,
        guarded_command=guarded,
        acceptance=acceptance,
    )
    spend_guard_check = _spend_guard_check_command(
        config=config,
        guarded_command=guarded,
    )

    progress = _require_mapping(
        spend_review.get("current_progress_verification"),
        "current progress verification",
    )
    if progress.get("status") != "matched":
        raise TrialQADecisionSummaryError("current progress is not matched")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": _stage_status(stage),
        "spend_authorized": False,
        "goal_status": goal_audit.get("status"),
        "goal_complete": goal_audit.get("goal_complete") is True,
        "goal_requirement_summary": _goal_requirement_summary(goal_audit),
        "next_required_action": _next_required_action(goal_audit, stage=stage),
        "manifest_id": spend_review.get("manifest_id"),
        "bundle_sha256": _require_mapping(
            spend_review.get("bundle_verification"),
            "bundle verification",
        ).get("bundle_sha256"),
        "next_boundary": {
            "stage": stage,
            "guarded_command_kind": _require_mapping(
                spend_review.get("preflight"),
                "preflight summary",
            ).get("next_command_kind"),
            "requires_yes_spend": True,
            "authorized_by_packet": False,
            "scope_label": scope.get("scope_label"),
            "task_count": scope.get("task_count"),
            "expected_model_calls": scope.get("expected_model_calls"),
            "expected_judge_calls": scope.get("expected_judge_calls"),
        },
        "commands": {
            "guarded_spend": guarded,
            "progress_monitor": monitor,
            "safe_preflight_refresh": safe_preflight,
            "pre_spend_guard_check": spend_guard_check,
            "post_spend_gate_inspection": gate_inspection,
            "post_spend_checkpoint": checkpoint,
        },
        "post_spend_acceptance_criteria": dict(acceptance),
        "current_progress": {
            "status": progress.get("status"),
            "done_task_count": progress.get("done_task_count"),
            "remaining_task_count": progress.get("remaining_task_count"),
            "ledger_record_count": progress.get("ledger_record_count"),
            "batch_lock_state": progress.get("batch_lock_state"),
        },
        "operator_checklist": _operator_checklist(stage=stage, acceptance=acceptance),
        "blocked_actions": [
            {
                "id": "no_full_benchmark_before_canaries",
                "instruction": "Do not expand to the full primary scope until the current gate promotes.",
            },
            {
                "id": "no_next_stage_spend_before_checkpoint",
                "instruction": "Do not buy the next stage until the post-spend checkpoint is complete.",
            },
        ],
        "proved_setup_evidence": _proved_setup_evidence(goal_audit),
        "missing_goal_evidence": _missing_goal_evidence(goal_audit),
        "failed_goal_evidence": _failed_goal_evidence(goal_audit),
        "goal_completion_note": goal_audit.get("completion_note"),
        "review_note": (
            "This summary is read-only and non-authorizing. It exists to make "
            "the next spend decision and immediate post-spend actions explicit."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend-review", type=Path, required=True)
    parser.add_argument("--goal-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> DecisionSummaryConfig:
    return DecisionSummaryConfig(
        spend_review=args.spend_review,
        goal_audit=args.goal_audit,
        decision_summary_output=args.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_decision_summary(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a post-spend TrialQA gate against the staged decision summary.

This command is the fast no-spend readout to run immediately after a guarded
generation or score canary finishes. It verifies that the gate is the exact
artifact named by the decision summary, checks the manifest and gate kind, and
emits the next safe action without authorizing further model or judge spend.
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
import benchmark.trialqa_local_gate as gate_module  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_gate_inspection.v1"
JsonObject = dict[str, Any]


class TrialQAGateInspectError(RuntimeError):
    """The gate cannot be safely interpreted against the decision summary."""


@dataclass(frozen=True)
class GateInspectConfig:
    gate: Path
    decision_summary: Path


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQAGateInspectError(f"{label} must be an object")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrialQAGateInspectError(f"{label} must be a string list")
    return list(value)


def _expected_summary_status(stage: object) -> str:
    if stage == "generation":
        return "awaiting_explicit_generation_spend_authorization"
    if stage == "score":
        return "awaiting_explicit_score_spend_authorization"
    raise TrialQAGateInspectError(f"unsupported decision summary stage {stage!r}")


def _expected_goal_status(stage: object) -> str:
    if stage == "generation":
        return "ready_for_generation_spend_decision"
    if stage == "score":
        return "ready_for_score_spend_decision"
    raise TrialQAGateInspectError(f"unsupported decision summary stage {stage!r}")


def _expected_action(stage: object) -> str:
    if stage == "generation":
        return "request_explicit_generation_canary_spend_approval"
    if stage == "score":
        return "request_explicit_score_canary_spend_approval"
    raise TrialQAGateInspectError(f"unsupported decision summary stage {stage!r}")


def _expected_command_kind(stage: object) -> str:
    if stage == "generation":
        return "guarded_generation_canary"
    if stage == "score":
        return "guarded_score_canary"
    raise TrialQAGateInspectError(f"unsupported decision summary stage {stage!r}")


def _safe_command(command: Mapping[str, object], label: str) -> list[str]:
    argv = _require_string_list(command.get("command"), f"{label} command")
    if "--yes-spend" in argv:
        raise TrialQAGateInspectError(f"{label} must not contain --yes-spend")
    shell_command = command.get("shell_command")
    if isinstance(shell_command, str) and "--yes-spend" in shell_command:
        raise TrialQAGateInspectError(f"{label} shell_command must not contain --yes-spend")
    return argv


def _same_path(expected: object, observed: Path) -> bool:
    if not isinstance(expected, str) or not expected:
        raise TrialQAGateInspectError("decision summary has no required gate artifact")
    return Path(expected).resolve() == observed.resolve()


def _option_value(argv: Sequence[str], option: str, label: str) -> str:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise TrialQAGateInspectError(f"{label} is missing {option}") from exc
    value_index = index + 1
    if value_index >= len(argv) or argv[value_index].startswith("--"):
        raise TrialQAGateInspectError(f"{label} has no value for {option}")
    return argv[value_index]


def _require_matching_path(actual: str, expected: Path, label: str) -> None:
    if Path(actual).resolve() != expected.resolve():
        raise TrialQAGateInspectError(f"{label} does not match this inspection")


def _gate_inspection_output(gate_path: Path) -> Path:
    name = gate_path.name
    if name.startswith("gate-operational-"):
        tail = name.removeprefix("gate-operational-")
    elif name.startswith("gate-promotion-"):
        tail = name.removeprefix("gate-promotion-")
    else:
        tail = name
    return gate_path.with_name(f"gate-inspection-{tail}")


def _validate_decision_summary_state(
    *,
    summary: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> JsonObject:
    stage = acceptance.get("stage")
    if summary.get("status") != _expected_summary_status(stage):
        raise TrialQAGateInspectError("decision summary status does not match gate stage")
    if summary.get("goal_status") != _expected_goal_status(stage):
        raise TrialQAGateInspectError("decision summary goal_status does not match gate stage")
    if summary.get("goal_complete") is not False:
        raise TrialQAGateInspectError("decision summary goal_complete must be false")

    boundary = _require_mapping(summary.get("next_boundary"), "decision summary next_boundary")
    if boundary.get("stage") != stage:
        raise TrialQAGateInspectError("decision summary next_boundary stage mismatch")
    if boundary.get("authorized_by_packet") is not False:
        raise TrialQAGateInspectError("decision summary next_boundary must not authorize spend")
    if boundary.get("requires_yes_spend") is not True:
        raise TrialQAGateInspectError("decision summary next_boundary must require --yes-spend")
    if boundary.get("guarded_command_kind") != _expected_command_kind(stage):
        raise TrialQAGateInspectError("decision summary next_boundary command kind mismatch")

    requirement_summary = _require_mapping(
        summary.get("goal_requirement_summary"),
        "decision summary goal_requirement_summary",
    )
    failed_ids = requirement_summary.get("required_failed_ids")
    if not isinstance(failed_ids, list) or not all(isinstance(item, str) for item in failed_ids):
        raise TrialQAGateInspectError(
            "decision summary goal_requirement_summary required_failed_ids is invalid"
        )
    if failed_ids:
        raise TrialQAGateInspectError("decision summary has failed required requirements")
    missing_ids = requirement_summary.get("required_missing_ids")
    if not isinstance(missing_ids, list) or not all(
        isinstance(item, str) for item in missing_ids
    ):
        raise TrialQAGateInspectError(
            "decision summary goal_requirement_summary required_missing_ids is invalid"
        )
    if not isinstance(requirement_summary.get("status_counts"), Mapping):
        raise TrialQAGateInspectError(
            "decision summary goal_requirement_summary status_counts is invalid"
        )
    if not isinstance(requirement_summary.get("total"), int):
        raise TrialQAGateInspectError(
            "decision summary goal_requirement_summary total is invalid"
        )

    next_action = _require_mapping(
        summary.get("next_required_action"),
        "decision summary next_required_action",
    )
    if next_action.get("action") != _expected_action(stage):
        raise TrialQAGateInspectError("decision summary next_required_action mismatch")
    if next_action.get("requires_spend") is not True:
        raise TrialQAGateInspectError("decision summary next_required_action must require spend")
    instruction = next_action.get("instruction")
    if not isinstance(instruction, str) or "--yes-spend" not in instruction:
        raise TrialQAGateInspectError(
            "decision summary next_required_action must mention --yes-spend"
        )

    failed_evidence = summary.get("failed_goal_evidence")
    if not isinstance(failed_evidence, list):
        raise TrialQAGateInspectError("decision summary failed_goal_evidence must be a list")
    if failed_evidence:
        raise TrialQAGateInspectError("decision summary failed_goal_evidence must be empty")
    completion_note = summary.get("goal_completion_note")
    if not isinstance(completion_note, str) or not completion_note:
        raise TrialQAGateInspectError("decision summary goal_completion_note must be set")

    return {
        "status": summary.get("status"),
        "goal_status": summary.get("goal_status"),
        "goal_complete": False,
        "goal_requirement_summary": dict(requirement_summary),
        "next_required_action": dict(next_action),
        "next_boundary": dict(boundary),
    }


def _validate_gate_inspection_command(
    *,
    commands: Mapping[str, object],
    gate: Path,
    decision_summary: Path,
) -> JsonObject:
    command = _require_mapping(
        commands.get("post_spend_gate_inspection"),
        "post-spend gate inspection",
    )
    if command.get("command_available") is not True:
        raise TrialQAGateInspectError(
            "post-spend gate inspection command is unavailable; "
            "regenerate the decision summary with --output"
        )
    argv = _safe_command(command, "post-spend gate inspection")
    if len(argv) < 3 or argv[1:3] != ["-m", "benchmark.trialqa_local_gate_inspect"]:
        raise TrialQAGateInspectError("post-spend gate inspection command targets the wrong module")
    _require_matching_path(
        _option_value(argv, "--gate", "post-spend gate inspection"),
        gate,
        "post-spend gate inspection --gate",
    )
    _require_matching_path(
        _option_value(argv, "--decision-summary", "post-spend gate inspection"),
        decision_summary,
        "post-spend gate inspection --decision-summary",
    )
    _require_matching_path(
        _option_value(argv, "--output", "post-spend gate inspection"),
        _gate_inspection_output(gate),
        "post-spend gate inspection --output",
    )
    return {
        "command": argv,
        "shell_command": command.get("shell_command"),
        "contains_yes_spend": False,
    }


def _failed_criteria(gate: Mapping[str, object]) -> list[JsonObject]:
    criteria = gate.get("criteria")
    if not isinstance(criteria, list):
        raise TrialQAGateInspectError("gate criteria must be a list")
    failed: list[JsonObject] = []
    for item in criteria:
        criterion = _require_mapping(item, "gate criterion")
        if criterion.get("passed") is False:
            failed.append(
                {
                    "name": criterion.get("name"),
                    "value": criterion.get("value"),
                    "operator": criterion.get("operator"),
                    "threshold": criterion.get("threshold"),
                }
            )
    return failed


def _decision_status(
    *,
    decision: object,
    promote_decision: object,
    kill_decision: object,
) -> tuple[str, str, bool]:
    if decision == promote_decision:
        return "gate_promoted", "run_post_spend_checkpoint", True
    if decision == kill_decision:
        return "gate_killed", "run_post_spend_checkpoint_to_record_terminal_decision", False
    return "gate_not_decisive", "inspect_gate_before_more_spend", False


def _compact_gate_summary(gate: Mapping[str, object]) -> JsonObject:
    benefit = _require_mapping(gate.get("benefit"), "gate benefit")
    quality = gate.get("quality")
    scope = _require_mapping(gate.get("scope"), "gate scope")
    return {
        "benefit": {
            "token_reduction_fraction": benefit.get("token_reduction_fraction"),
            "operational_call_reduction_fraction": benefit.get(
                "operational_call_reduction_fraction"
            ),
            "mean_score_delta": benefit.get("mean_score_delta"),
            "terminal_rate_delta": benefit.get("terminal_rate_delta"),
        },
        "quality": (
            {
                "available": quality.get("available"),
                "mode": quality.get("mode"),
                "point_delta": quality.get("point_delta"),
                "decision_bound": quality.get("decision_bound"),
                "decision_threshold": quality.get("decision_threshold"),
            }
            if isinstance(quality, Mapping)
            else None
        ),
        "scope": {
            "pair_count": scope.get("pair_count"),
            "task_count": scope.get("task_count"),
            "confirmatory_scope_complete": scope.get("confirmatory_scope_complete"),
            "selection_attestation": scope.get("selection_attestation"),
        },
    }


def inspect_gate(config: GateInspectConfig) -> JsonObject:
    """Inspect a gate and return the next safe no-spend action."""

    gate = demo._read_json_object(config.gate, "gate report")
    summary = demo._read_json_object(config.decision_summary, "decision summary")
    if gate.get("schema_version") != gate_module.SCHEMA_VERSION:
        raise TrialQAGateInspectError("gate report has invalid schema_version")
    if summary.get("schema_version") != "switchyard.trialqa_decision_summary.v1":
        raise TrialQAGateInspectError("decision summary has invalid schema_version")
    if summary.get("spend_authorized") is not False:
        raise TrialQAGateInspectError("decision summary must not authorize spend")

    acceptance = _require_mapping(
        summary.get("post_spend_acceptance_criteria"),
        "post-spend acceptance criteria",
    )
    decision_summary_state = _validate_decision_summary_state(
        summary=summary,
        acceptance=acceptance,
    )
    if not _same_path(acceptance.get("required_gate_artifact"), config.gate):
        raise TrialQAGateInspectError("gate path does not match the decision summary")
    if gate.get("gate") != acceptance.get("required_gate"):
        raise TrialQAGateInspectError("gate kind does not match the decision summary")
    if gate.get("manifest_id") != summary.get("manifest_id"):
        raise TrialQAGateInspectError("gate manifest_id does not match the decision summary")
    commands = _require_mapping(summary.get("commands"), "decision summary commands")
    gate_inspection_command = _validate_gate_inspection_command(
        commands=commands,
        gate=config.gate,
        decision_summary=config.decision_summary,
    )
    checkpoint = _require_mapping(commands.get("post_spend_checkpoint"), "post-spend checkpoint")
    checkpoint_command = _safe_command(checkpoint, "post-spend checkpoint")

    status, next_action, promoted = _decision_status(
        decision=gate.get("decision"),
        promote_decision=acceptance.get("promote_decision"),
        kill_decision=acceptance.get("kill_decision"),
    )
    failed_criteria = _failed_criteria(gate)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "spend_authorized": False,
        "next_stage_spend_authorized": False,
        "manifest_id": gate.get("manifest_id"),
        "gate": gate.get("gate"),
        "gate_path": str(config.gate),
        "decision": gate.get("decision"),
        "expected_promote_decision": acceptance.get("promote_decision"),
        "expected_kill_decision": acceptance.get("kill_decision"),
        "promoted": promoted,
        "next_action": next_action,
        "must_run_checkpoint_before_more_spend": True,
        "validated_gate_inspection_command": gate_inspection_command,
        "decision_summary_state": decision_summary_state,
        "post_spend_checkpoint_command": {
            "command": checkpoint_command,
            "shell_command": checkpoint.get("shell_command"),
            "contains_yes_spend": False,
        },
        "failed_criteria": failed_criteria,
        "gate_summary": _compact_gate_summary(gate),
        "review_note": (
            "This inspection is read-only and does not authorize more spend. "
            "Run the post-spend checkpoint before any next-stage model or judge calls."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--decision-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _config_from_args(args: argparse.Namespace) -> GateInspectConfig:
    return GateInspectConfig(
        gate=args.gate,
        decision_summary=args.decision_summary,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = inspect_gate(_config_from_args(args))
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

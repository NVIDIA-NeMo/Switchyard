# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find and verify the newest local TrialQA operator packet.

The artifact directory intentionally keeps old compact-vN packets as immutable
audit history. This helper prevents an operator from picking an older spend
packet by hand. It is read-only: it may re-run the no-spend spend guard, but it
never runs a guarded canary and never authorizes model or judge calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_spend_guard as spend_guard  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_current_packet.v1"
DECISION_SUMMARY_SCHEMA = "switchyard.trialqa_decision_summary.v1"
Stage = Literal["generation", "score"]
JsonObject = dict[str, Any]
_DECISION_SUMMARY_RE = re.compile(
    r"^decision-summary-(?P<stem>.+)-compact-v(?P<version>[0-9]+)-"
    r"(?P<scope>q[0-9]+-q[0-9]+-r[0-9]+)\.json$"
)


class TrialQACurrentPacketError(RuntimeError):
    """The latest operator packet cannot be identified or verified safely."""


@dataclass(frozen=True)
class PacketCandidate:
    path: Path
    version: int
    stem: str
    scope: str
    payload: Mapping[str, object]


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQACurrentPacketError(f"{label} must be an object")
    return value


def _require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrialQACurrentPacketError(f"{label} must be a string list")
    return list(value)


def _require_object_list(value: object, label: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise TrialQACurrentPacketError(f"{label} must be a list")
    result: list[JsonObject] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TrialQACurrentPacketError(f"{label} items must be objects")
        result.append(dict(item))
    return result


def _require_ids(items: Sequence[Mapping[str, object]], label: str) -> set[str]:
    ids: set[str] = set()
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise TrialQACurrentPacketError(f"{label} items must have string ids")
        ids.add(item_id)
    return ids


def _operator_handoff(
    summary: Mapping[str, object],
    *,
    boundary: Mapping[str, object],
) -> JsonObject:
    blocked_actions = _require_object_list(
        summary.get("blocked_actions"),
        "decision summary blocked_actions",
    )
    operator_checklist = _require_object_list(
        summary.get("operator_checklist"),
        "decision summary operator_checklist",
    )
    acceptance = dict(
        _require_mapping(
            summary.get("post_spend_acceptance_criteria"),
            "decision summary post_spend_acceptance_criteria",
        )
    )

    blocked_ids = _require_ids(blocked_actions, "blocked_actions")
    required_blocked = {
        "no_full_benchmark_before_canaries",
        "no_next_stage_spend_before_checkpoint",
    }
    missing_blocked = required_blocked - blocked_ids
    if missing_blocked:
        raise TrialQACurrentPacketError(
            "decision summary blocked_actions is missing "
            f"{sorted(missing_blocked)!r}"
        )

    checklist_ids = _require_ids(operator_checklist, "operator_checklist")
    required_checklist = {
        "validate_spend_guard",
        "run_guarded_canary_if_approved",
        "monitor_without_spend",
        "inspect_post_spend_gate",
        "promote_or_kill",
        "checkpoint_before_more_spend",
    }
    missing_checklist = required_checklist - checklist_ids
    if missing_checklist:
        raise TrialQACurrentPacketError(
            "decision summary operator_checklist is missing "
            f"{sorted(missing_checklist)!r}"
        )

    if acceptance.get("stage") != boundary.get("stage"):
        raise TrialQACurrentPacketError(
            "decision summary acceptance criteria stage does not match boundary"
        )
    if acceptance.get("kill_decision") != "kill":
        raise TrialQACurrentPacketError("decision summary acceptance criteria must define kill")
    if acceptance.get("must_run_checkpoint_before_more_spend") is not True:
        raise TrialQACurrentPacketError(
            "decision summary acceptance criteria must require a no-spend checkpoint"
        )

    return {
        "blocked_actions": blocked_actions,
        "operator_checklist": operator_checklist,
        "post_spend_acceptance_criteria": acceptance,
    }


def _goal_state_handoff(
    summary: Mapping[str, object],
    *,
    boundary: Mapping[str, object],
) -> JsonObject:
    requirement_summary = dict(
        _require_mapping(
            summary.get("goal_requirement_summary"),
            "decision summary goal_requirement_summary",
        )
    )
    required_failed = requirement_summary.get("required_failed_ids")
    if not isinstance(required_failed, list) or not all(
        isinstance(item, str) for item in required_failed
    ):
        raise TrialQACurrentPacketError(
            "decision summary goal_requirement_summary required_failed_ids is invalid"
        )
    if required_failed:
        raise TrialQACurrentPacketError(
            "decision summary has failed required goal requirements"
        )
    required_missing = requirement_summary.get("required_missing_ids")
    if not isinstance(required_missing, list) or not all(
        isinstance(item, str) for item in required_missing
    ):
        raise TrialQACurrentPacketError(
            "decision summary goal_requirement_summary required_missing_ids is invalid"
        )
    status_counts = requirement_summary.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise TrialQACurrentPacketError(
            "decision summary goal_requirement_summary status_counts is invalid"
        )
    total = requirement_summary.get("total")
    if not isinstance(total, int) or total <= 0:
        raise TrialQACurrentPacketError(
            "decision summary goal_requirement_summary total is invalid"
        )

    next_action = dict(
        _require_mapping(
            summary.get("next_required_action"),
            "decision summary next_required_action",
        )
    )
    if next_action.get("requires_spend") is not True:
        raise TrialQACurrentPacketError(
            "decision summary next_required_action must require spend"
        )
    expected_action = (
        "request_explicit_score_canary_spend_approval"
        if boundary.get("stage") == "score"
        else "request_explicit_generation_canary_spend_approval"
    )
    if next_action.get("action") != expected_action:
        raise TrialQACurrentPacketError(
            "decision summary next_required_action does not match boundary"
        )
    instruction = next_action.get("instruction")
    if not isinstance(instruction, str) or "--yes-spend" not in instruction:
        raise TrialQACurrentPacketError(
            "decision summary next_required_action must mention --yes-spend"
        )

    proved_setup = _require_object_list(
        summary.get("proved_setup_evidence"),
        "decision summary proved_setup_evidence",
    )
    missing_goal = _require_object_list(
        summary.get("missing_goal_evidence"),
        "decision summary missing_goal_evidence",
    )
    failed_goal = _require_object_list(
        summary.get("failed_goal_evidence"),
        "decision summary failed_goal_evidence",
    )
    if failed_goal:
        raise TrialQACurrentPacketError("decision summary failed_goal_evidence must be empty")
    completion_note = summary.get("goal_completion_note")
    if not isinstance(completion_note, str) or not completion_note:
        raise TrialQACurrentPacketError("decision summary goal_completion_note must be set")

    return {
        "goal_requirement_summary": requirement_summary,
        "next_required_action": next_action,
        "proved_setup_evidence": proved_setup,
        "missing_goal_evidence": missing_goal,
        "failed_goal_evidence": failed_goal,
        "goal_completion_note": completion_note,
    }


def _option_value(argv: Sequence[str], option: str) -> str:
    try:
        index = argv.index(option)
    except ValueError as exc:
        raise TrialQACurrentPacketError(f"command is missing {option}") from exc
    value_index = index + 1
    if value_index >= len(argv) or argv[value_index].startswith("--"):
        raise TrialQACurrentPacketError(f"command has no value for {option}")
    return argv[value_index]


def _stage(payload: Mapping[str, object]) -> str | None:
    boundary = payload.get("next_boundary")
    if not isinstance(boundary, Mapping):
        return None
    stage = boundary.get("stage")
    return stage if isinstance(stage, str) else None


def _expected_summary_status(stage: object) -> str:
    if stage == "generation":
        return "awaiting_explicit_generation_spend_authorization"
    if stage == "score":
        return "awaiting_explicit_score_spend_authorization"
    raise TrialQACurrentPacketError(f"unsupported decision summary stage {stage!r}")


def _expected_goal_status(stage: object) -> str:
    if stage == "generation":
        return "ready_for_generation_spend_decision"
    if stage == "score":
        return "ready_for_score_spend_decision"
    raise TrialQACurrentPacketError(f"unsupported decision summary stage {stage!r}")


def _load_candidates(
    artifact_dir: Path,
    *,
    stage: Stage | None,
    stem_contains: str | None,
    scope: str | None,
) -> list[PacketCandidate]:
    candidates: list[PacketCandidate] = []
    for path in artifact_dir.glob("decision-summary-*.json"):
        match = _DECISION_SUMMARY_RE.match(path.name)
        if match is None:
            continue
        candidate_scope = match.group("scope")
        candidate_stem = match.group("stem")
        if scope is not None and candidate_scope != scope:
            continue
        if stem_contains is not None and stem_contains not in candidate_stem:
            continue
        payload = demo._read_json_object(path, "decision summary")
        if payload.get("schema_version") != DECISION_SUMMARY_SCHEMA:
            raise TrialQACurrentPacketError(f"{path} has invalid schema_version")
        candidate_stage = _stage(payload)
        if stage is not None and candidate_stage != stage:
            continue
        candidates.append(
            PacketCandidate(
                path=path,
                version=int(match.group("version")),
                stem=candidate_stem,
                scope=candidate_scope,
                payload=payload,
            )
        )
    return candidates


def _select_latest(candidates: Sequence[PacketCandidate]) -> PacketCandidate:
    if not candidates:
        raise TrialQACurrentPacketError("no matching decision-summary compact-vN packet found")
    return max(candidates, key=lambda item: (item.version, item.path.name))


def _spend_review_from_summary(
    summary: Mapping[str, object],
    *,
    repo_root: Path,
) -> Path:
    commands = _require_mapping(summary.get("commands"), "decision summary commands")
    guard_command = _require_mapping(
        commands.get("pre_spend_guard_check"),
        "pre_spend_guard_check command",
    )
    argv = _require_string_list(guard_command.get("command"), "pre_spend_guard_check argv")
    if "--yes-spend" in argv:
        raise TrialQACurrentPacketError("pre-spend guard command must not contain --yes-spend")
    spend_review = Path(_option_value(argv, "--spend-review"))
    return spend_review if spend_review.is_absolute() else repo_root / spend_review


def _command_summary(summary: Mapping[str, object]) -> JsonObject:
    commands = _require_mapping(summary.get("commands"), "decision summary commands")
    names = (
        "guarded_spend",
        "pre_spend_guard_check",
        "progress_monitor",
        "post_spend_gate_inspection",
        "post_spend_checkpoint",
    )
    return {
        name: _require_mapping(commands.get(name), f"{name} command").get("shell_command")
        for name in names
    }


def _verify_guarded_command_matches_guard(
    *,
    summary_guarded: Mapping[str, object],
    guard_report: Mapping[str, object],
) -> JsonObject:
    reviewed = _require_mapping(
        guard_report.get("reviewed_command"),
        "fresh spend guard reviewed_command",
    )
    reviewed_argv = _require_string_list(
        reviewed.get("command"),
        "fresh spend guard reviewed command argv",
    )
    summary_argv = _require_string_list(
        summary_guarded.get("command"),
        "decision summary guarded spend argv",
    )
    if summary_argv != reviewed_argv:
        raise TrialQACurrentPacketError(
            "decision summary guarded command differs from the fresh spend guard"
        )
    summary_shell = summary_guarded.get("shell_command")
    reviewed_shell = reviewed.get("shell_command")
    if (
        isinstance(summary_shell, str)
        and isinstance(reviewed_shell, str)
        and summary_shell != reviewed_shell
    ):
        raise TrialQACurrentPacketError(
            "decision summary guarded shell command differs from the fresh spend guard"
        )
    return dict(reviewed)


def build_current_packet_report(
    *,
    artifact_dir: Path,
    stage: Stage | None = "generation",
    stem_contains: str | None = None,
    scope: str | None = "q0-q3-r1",
    verify_guard: bool = True,
    repo_root: Path | None = None,
) -> JsonObject:
    """Return a compact handoff for the newest matching operator packet."""

    root = Path.cwd() if repo_root is None else repo_root
    candidates = _load_candidates(
        artifact_dir,
        stage=stage,
        stem_contains=stem_contains,
        scope=scope,
    )
    latest = _select_latest(candidates)
    summary = latest.payload
    boundary = _require_mapping(summary.get("next_boundary"), "decision summary next_boundary")
    boundary_stage = boundary.get("stage")
    if summary.get("status") != _expected_summary_status(boundary_stage):
        raise TrialQACurrentPacketError(
            "decision summary status does not match its spend boundary"
        )
    if summary.get("goal_status") != _expected_goal_status(boundary_stage):
        raise TrialQACurrentPacketError(
            "decision summary goal_status does not match its spend boundary"
        )
    if summary.get("spend_authorized") is not False:
        raise TrialQACurrentPacketError("decision summary must not authorize spend")
    if summary.get("goal_complete") is not False:
        raise TrialQACurrentPacketError(
            "decision summary goal_complete must be false while it exposes a spend packet"
        )
    if boundary.get("authorized_by_packet") is not False:
        raise TrialQACurrentPacketError("decision summary boundary must not authorize spend")
    if boundary.get("requires_yes_spend") is not True:
        raise TrialQACurrentPacketError("decision summary boundary must require --yes-spend")
    commands = _require_mapping(summary.get("commands"), "decision summary commands")
    guarded = _require_mapping(commands.get("guarded_spend"), "guarded spend command")
    guarded_argv = _require_string_list(guarded.get("command"), "guarded spend argv")
    if not guarded_argv or guarded_argv[-1] != "--yes-spend":
        raise TrialQACurrentPacketError("guarded spend command must end in --yes-spend")
    if "--spend-review" not in guarded_argv:
        raise TrialQACurrentPacketError("guarded spend command must include --spend-review")
    handoff = _operator_handoff(summary, boundary=boundary)
    goal_handoff = _goal_state_handoff(summary, boundary=boundary)

    spend_review = _spend_review_from_summary(summary, repo_root=root)
    guard_report: Mapping[str, object] | None = None
    verified_guarded_command: JsonObject | None = None
    if verify_guard:
        guard_report = spend_guard.build_spend_guard_check(spend_review=spend_review)
        if guard_report.get("status") != "passed":
            raise TrialQACurrentPacketError("fresh spend guard did not pass")
        guard = _require_mapping(guard_report.get("guard"), "fresh spend guard")
        bundle = _require_mapping(guard.get("bundle"), "fresh spend guard bundle")
        if bundle.get("bundle_sha256") != summary.get("bundle_sha256"):
            raise TrialQACurrentPacketError("fresh spend guard bundle does not match summary")
        verified_guarded_command = _verify_guarded_command_matches_guard(
            summary_guarded=guarded,
            guard_report=guard_report,
        )

    commands_summary = _command_summary(summary)
    if verified_guarded_command is not None:
        commands_summary["guarded_spend"] = verified_guarded_command.get("shell_command")
    else:
        commands_summary["guarded_spend"] = None
    guard_verified = guard_report is not None

    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "ready_for_user_spend_decision"
            if guard_verified
            else "selected_without_fresh_spend_guard"
        ),
        "spend_authorized": False,
        "fresh_spend_guard_required_before_spend": not guard_verified,
        "artifact_dir": str(artifact_dir),
        "selected": {
            "path": str(latest.path),
            "version": latest.version,
            "stem": latest.stem,
            "scope": latest.scope,
            "stage": boundary.get("stage"),
            "bundle_sha256": summary.get("bundle_sha256"),
        },
        "goal_status": summary.get("goal_status"),
        "goal_complete": False,
        "candidate_count": len(candidates),
        "older_candidate_count": max(0, len(candidates) - 1),
        "next_boundary": dict(boundary),
        **goal_handoff,
        **handoff,
        "commands": commands_summary,
        "spend_review": str(spend_review),
        "fresh_spend_guard": (
            {
                "status": guard_report.get("status"),
                "stage": guard_report.get("stage"),
                "spend_authorized": guard_report.get("spend_authorized"),
                "next_stage_spend_authorized": guard_report.get(
                    "next_stage_spend_authorized"
                ),
                "bundle_sha256": _require_mapping(
                    _require_mapping(guard_report.get("guard"), "fresh guard").get("bundle"),
                    "fresh guard bundle",
                ).get("bundle_sha256"),
                "current_progress": _require_mapping(
                    _require_mapping(guard_report.get("guard"), "fresh guard").get(
                        "current_progress"
                    ),
                    "fresh guard current_progress",
                ),
                "reviewed_command": verified_guarded_command,
            }
            if guard_report is not None
            else None
        ),
        "review_note": (
            (
                "This report is read-only and non-authorizing. Use the selected "
                "decision summary as the handoff, and run the guarded command only "
                "after explicit approval for --yes-spend."
            )
            if guard_verified
            else (
                "This report only selected the newest packet. It did not re-run "
                "the fresh spend guard, so it is not ready for a spend decision; "
                "rerun without --no-verify-guard before considering --yes-spend."
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("generation", "score"), default="generation")
    parser.add_argument("--stem-contains")
    parser.add_argument("--scope", default="q0-q3-r1")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--no-verify-guard",
        action="store_true",
        help=(
            "only select the newest packet; do not re-run the no-spend spend "
            "guard, do not print a guarded spend command, and do not treat the "
            "packet as ready for a spend decision"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_current_packet_report(
        artifact_dir=args.artifact_dir,
        stage=cast(Stage, args.stage),
        stem_contains=args.stem_contains,
        scope=args.scope,
        verify_guard=not args.no_verify_guard,
        repo_root=args.repo_root,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-spend staged-status report for the local TrialQA validation ladder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_readiness as readiness_module  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_protocol_status.v1"
JsonObject = dict[str, Any]


class TrialQAStatusError(RuntimeError):
    """A status input is missing, stale, or inconsistent."""


def _optional_json(path: Path | None, label: str) -> JsonObject | None:
    if path is None:
        return None
    return demo._read_json_object(path, label)


def _require_schema(report: Mapping[str, object], schema: str, label: str) -> None:
    if report.get("schema_version") != schema:
        raise TrialQAStatusError(f"{label} has invalid schema_version")


def _gate_summary(
    gate: Mapping[str, object] | None,
    *,
    expected_manifest_id: str,
    expected_gate: str,
) -> JsonObject | None:
    if gate is None:
        return None
    _require_schema(gate, "switchyard.trialqa_gate_report.v3", f"{expected_gate} gate")
    if gate.get("manifest_id") != expected_manifest_id:
        raise TrialQAStatusError(f"{expected_gate} gate belongs to a different manifest")
    if gate.get("gate") != expected_gate:
        raise TrialQAStatusError(f"expected {expected_gate} gate, got {gate.get('gate')!r}")
    scope = gate.get("scope")
    if not isinstance(scope, Mapping):
        raise TrialQAStatusError(f"{expected_gate} gate has no scope")
    attestation = scope.get("selection_attestation")
    if attestation is not None and not isinstance(attestation, Mapping):
        raise TrialQAStatusError(f"{expected_gate} gate selection attestation is invalid")
    return {
        "decision": gate.get("decision"),
        "performance_eligible": gate.get("performance_eligible"),
        "task_count": scope.get("task_count"),
        "pair_count": scope.get("pair_count"),
        "confirmatory_scope_complete": scope.get("confirmatory_scope_complete"),
        "selection_attestation": dict(attestation) if isinstance(attestation, Mapping) else None,
    }


def _readiness_summary(
    readiness: Mapping[str, object],
    *,
    expected_manifest_id: str,
) -> JsonObject:
    _require_schema(readiness, "switchyard.trialqa_canary_readiness.v1", "readiness report")
    manifest = readiness.get("manifest")
    if not isinstance(manifest, Mapping):
        raise TrialQAStatusError("readiness report has no manifest section")
    if manifest.get("manifest_id") != expected_manifest_id:
        raise TrialQAStatusError("readiness report belongs to a different manifest")
    first = readiness.get("first_generation_canary")
    if not isinstance(first, Mapping):
        raise TrialQAStatusError("readiness report has no first_generation_canary section")
    states = first.get("selected_task_states")
    if not isinstance(states, Mapping):
        raise TrialQAStatusError("readiness report has invalid selected_task_states")
    comparison_invariant = readiness.get("comparison_invariant")
    if comparison_invariant is not None and not isinstance(comparison_invariant, Mapping):
        raise TrialQAStatusError("readiness report has invalid comparison_invariant")
    return {
        "status": readiness.get("status"),
        "task_count": first.get("task_count"),
        "pair_count": first.get("pair_count"),
        "selected_task_state_values": sorted({str(value) for value in states.values()}),
        "scope_attestation": first.get("scope_attestation"),
        "comparison_invariant": (
            dict(comparison_invariant) if isinstance(comparison_invariant, Mapping) else None
        ),
    }


def _next_expansion_from_gate(
    promotion: Mapping[str, object],
    *,
    primary_scope: Mapping[str, object],
) -> JsonObject:
    attestation = promotion.get("selection_attestation")
    if not isinstance(attestation, Mapping):
        raise TrialQAStatusError("promotion gate has no selection attestation")
    question_count = attestation.get("selected_question_count")
    repeat_indices = attestation.get("selected_repeat_indices")
    primary_question_count = primary_scope.get("question_count")
    primary_repeat_count = primary_scope.get("repeat_count")
    if (
        isinstance(question_count, int)
        and isinstance(primary_question_count, int)
        and question_count < primary_question_count
    ):
        return {
            "action": "expand_generation_scope",
            "reason": "promoted partial question canary",
            "question_start": primary_scope.get("question_start"),
            "question_limit": primary_question_count,
            "repeat_limit": max(cast(list[int], repeat_indices)) if isinstance(repeat_indices, list) else 1,
            "requires_yes_spend": True,
        }
    if isinstance(repeat_indices, list) and repeat_indices:
        max_repeat = max(int(value) for value in repeat_indices)
        if max_repeat < 3:
            return {
                "action": "expand_generation_scope",
                "reason": "promoted repeat-1 scope",
                "question_start": primary_scope.get("question_start"),
                "question_limit": primary_question_count,
                "repeat_limit": 3,
                "requires_yes_spend": True,
            }
        if isinstance(primary_repeat_count, int) and max_repeat < primary_repeat_count:
            return {
                "action": "expand_generation_scope",
                "reason": "promoted repeat-3 scope",
                "question_start": primary_scope.get("question_start"),
                "question_limit": primary_question_count,
                "repeat_limit": primary_repeat_count,
                "requires_yes_spend": True,
            }
    if promotion.get("confirmatory_scope_complete") is not True:
        return {
            "action": "fix_readiness_before_spend",
            "reason": "promotion gate has not marked the confirmatory scope complete",
        }
    if promotion.get("performance_eligible") is not True:
        return {
            "action": "fix_readiness_before_spend",
            "reason": "promotion gate is not performance eligible",
        }
    return {
        "action": "prospective_directional_scope_complete",
        "reason": "promotion gate completed the declared primary scope",
    }


def _next_action(
    *,
    readiness: Mapping[str, object],
    operational: Mapping[str, object] | None,
    promotion: Mapping[str, object] | None,
    primary_scope: Mapping[str, object],
) -> JsonObject:
    if readiness.get("status") not in readiness_module.GENERATION_READY_STATUSES:
        return {
            "action": "fix_readiness_before_spend",
            "reason": f"readiness status is {readiness.get('status')!r}",
        }
    if operational is None:
        return {
            "action": "run_guarded_generation_canary",
            "reason": "no operational gate report exists yet",
            "requires_yes_spend": True,
        }
    if operational.get("decision") != "promote_to_score":
        return {
            "action": "kill_candidate",
            "reason": f"operational decision is {operational.get('decision')!r}",
        }
    if promotion is None:
        return {
            "action": "run_guarded_score_canary",
            "reason": "operational gate promoted to score and no promotion gate exists yet",
            "requires_yes_spend": True,
        }
    if promotion.get("decision") != "promote_to_next_cohort":
        return {
            "action": "kill_candidate",
            "reason": f"promotion decision is {promotion.get('decision')!r}",
        }
    return _next_expansion_from_gate(promotion, primary_scope=primary_scope)


def build_status_report(
    *,
    manifest_path: Path,
    readiness_path: Path,
    reference_targets_path: Path,
    operational_gate_path: Path | None = None,
    promotion_gate_path: Path | None = None,
) -> JsonObject:
    """Build an audit-backed status report for the current TrialQA ladder point."""

    manifest = demo._read_json_object(manifest_path, "experiment manifest")
    demo.validate_manifest_pairing(manifest)
    manifest_id = cast(str, manifest["manifest_id"])
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TrialQAStatusError("manifest has no protocol")
    primary_scope = protocol.get("primary_evaluation_scope")
    if not isinstance(primary_scope, Mapping):
        raise TrialQAStatusError("manifest has no primary_evaluation_scope")

    readiness = demo._read_json_object(readiness_path, "readiness report")
    reference = demo._read_json_object(reference_targets_path, "reference targets")
    _require_schema(reference, "switchyard.trialqa_reference_targets.v1", "reference targets")
    readiness_report = _readiness_summary(readiness, expected_manifest_id=manifest_id)
    operational_report = _gate_summary(
        _optional_json(operational_gate_path, "operational gate report"),
        expected_manifest_id=manifest_id,
        expected_gate="operational",
    )
    promotion_report = _gate_summary(
        _optional_json(promotion_gate_path, "promotion gate report"),
        expected_manifest_id=manifest_id,
        expected_gate="promotion",
    )
    next_action = _next_action(
        readiness=readiness_report,
        operational=operational_report,
        promotion=promotion_report,
        primary_scope=primary_scope,
    )
    reference_population = cast(Mapping[str, object], reference["population"])
    reference_super = cast(Mapping[str, object], reference["super"])
    reference_super_r1 = cast(Mapping[str, object], reference_super["r1"])
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": {
            "manifest_id": manifest_id,
            "kind": manifest.get("kind"),
            "task_count": len(cast(list[object], manifest["tasks"])),
            "official_labbench2": cast(Mapping[str, object], manifest["dataset"]).get(
                "official_labbench2"
            ),
            "primary_evaluation_scope": dict(primary_scope),
        },
        "reference_targets": {
            "path": str(reference_targets_path),
            "trials": reference_population.get("trials"),
            "heldout_questions": reference_population.get("heldout_questions"),
            "repeats_per_question": reference_population.get("repeats_per_question"),
            "super_r1_accuracy": reference_super_r1.get("accuracy"),
            "super_r1_token_reduction": reference_super_r1.get("token_reduction"),
            "super_r1_operational_call_reduction": reference_super_r1.get(
                "operational_call_reduction"
            ),
        },
        "readiness": readiness_report,
        "operational_gate": operational_report,
        "promotion_gate": promotion_report,
        "next_action": next_action,
        "completion_state": "incomplete",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--reference-targets", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path)
    parser.add_argument("--promotion-gate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_status_report(
        manifest_path=args.manifest,
        readiness_path=args.readiness,
        reference_targets_path=args.reference_targets,
        operational_gate_path=args.operational_gate,
        promotion_gate_path=args.promotion_gate,
    )
    if args.output is not None:
        demo._write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

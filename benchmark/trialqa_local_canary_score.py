# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded TrialQA canary scoring driver.

This driver is the post-generation companion to ``trialqa_local_canary``. It
requires an operational gate report with ``decision: promote_to_score`` before
it will run score-stage judge calls. The default mode is a zero-spend dry-run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_spend_guard as spend_guard  # noqa: E402

JsonObject = dict[str, Any]
Run = Callable[..., subprocess.CompletedProcess[object]]


class TrialQACanaryScoreError(RuntimeError):
    """The canary score stage cannot safely proceed."""


@dataclass(frozen=True)
class ScoreConfig:
    manifest: Path
    dataset: Path
    experiment_root: Path
    doctor: Path
    population_report: Path
    candidate: Path
    switchyard: Path
    codex: Path
    tooluniverse: Path
    profile: Path
    operational_gate: Path
    question_start: int
    question_limit: int
    repeat_limit: int
    workers: int
    max_generation_attempts: int
    promotion_gate_output: Path


def _path_args(config: ScoreConfig) -> list[str]:
    return [
        "--manifest",
        str(config.manifest),
        "--dataset",
        str(config.dataset),
        "--experiment-root",
        str(config.experiment_root),
        "--doctor",
        str(config.doctor),
        "--population-report",
        str(config.population_report),
        "--candidate",
        str(config.candidate),
        "--switchyard",
        str(config.switchyard),
        "--codex",
        str(config.codex),
        "--tooluniverse",
        str(config.tooluniverse),
        "--profile",
        str(config.profile),
    ]


def score_command(
    config: ScoreConfig,
    *,
    python: Path | str = sys.executable,
    recover_interrupted: bool = False,
) -> list[str]:
    """Return the exact score-stage batch command."""

    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_batch",
        *_path_args(config),
        "--stage",
        "score",
        "--question-start",
        str(config.question_start),
        "--question-limit",
        str(config.question_limit),
        "--repeat-limit",
        str(config.repeat_limit),
        "--workers",
        str(config.workers),
        "--max-generation-attempts",
        str(config.max_generation_attempts),
    ]
    if recover_interrupted:
        command.extend(["--recover-interrupted", "--retry-failed"])
    return command


def promotion_gate_command(
    config: ScoreConfig, *, python: Path | str = sys.executable
) -> list[str]:
    """Return the exact promotion gate command for scored canary results."""

    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    capture = config.experiment_root / str(manifest["manifest_id"])
    return [
        str(python),
        "-m",
        "benchmark.trialqa_local_gate",
        "--manifest",
        str(config.manifest),
        "--capture",
        str(capture),
        "--gate",
        "promotion",
        "--question-start",
        str(config.question_start),
        "--question-limit",
        str(config.question_limit),
        "--repeat-limit",
        str(config.repeat_limit),
        "--output",
        str(config.promotion_gate_output),
    ]


def guarded_score_canary_command(
    config: ScoreConfig,
    *,
    python: Path | str = sys.executable,
    summary_output: Path | None = None,
    spend_review: Path | None = None,
    yes_spend: bool = False,
    recover_interrupted: bool = False,
) -> list[str]:
    """Return the top-level guarded score canary command for this exact config."""

    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_canary_score",
        *_path_args(config),
        "--operational-gate",
        str(config.operational_gate),
        "--question-start",
        str(config.question_start),
        "--question-limit",
        str(config.question_limit),
        "--repeat-limit",
        str(config.repeat_limit),
        "--workers",
        str(config.workers),
        "--max-generation-attempts",
        str(config.max_generation_attempts),
        "--promotion-gate-output",
        str(config.promotion_gate_output),
    ]
    if summary_output is not None:
        command.extend(["--summary-output", str(summary_output)])
    if spend_review is not None:
        command.extend(["--spend-review", str(spend_review)])
    if recover_interrupted:
        command.append("--recover-interrupted")
    if yes_spend:
        command.append("--yes-spend")
    return command


def _validate_operational_gate(
    config: ScoreConfig,
) -> Mapping[str, object]:
    gate = demo._read_json_object(config.operational_gate, "operational gate report")
    manifest = demo._read_json_object(config.manifest, "experiment manifest")
    if gate.get("schema_version") != "switchyard.trialqa_gate_report.v3":
        raise TrialQACanaryScoreError("operational gate report has an invalid schema")
    if gate.get("gate") != "operational":
        raise TrialQACanaryScoreError("gate report is not an operational gate")
    if gate.get("manifest_id") != manifest.get("manifest_id"):
        raise TrialQACanaryScoreError("operational gate report belongs to a different manifest")
    if gate.get("decision") != "promote_to_score":
        raise TrialQACanaryScoreError(
            f"operational gate did not promote to score: {gate.get('decision')!r}"
        )
    scope = gate.get("scope")
    if not isinstance(scope, Mapping):
        raise TrialQACanaryScoreError("operational gate scope is missing")
    attestation = scope.get("selection_attestation")
    if not isinstance(attestation, Mapping):
        raise TrialQACanaryScoreError("operational gate selection attestation is missing")
    expected = {
        "question_start": config.question_start,
        "question_limit": config.question_limit,
        "selected_repeat_indices": [config.repeat_limit]
        if config.repeat_limit == 1
        else list(range(1, config.repeat_limit + 1)),
        "selected_task_count": config.question_limit * config.repeat_limit * 2,
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            raise TrialQACanaryScoreError(
                f"operational gate selection attestation differs at {field}"
            )
    return gate


def run_score_canary(
    config: ScoreConfig,
    *,
    yes_spend: bool,
    recover_interrupted: bool = False,
    run: Run = subprocess.run,
    python: Path | str = sys.executable,
    summary_output: Path | None = None,
    spend_review: Path | None = None,
) -> JsonObject:
    """Validate the operational gate, optionally score, then run promotion gate."""

    gate = _validate_operational_gate(config)
    score = score_command(
        config,
        python=python,
        recover_interrupted=recover_interrupted,
    )
    promotion = promotion_gate_command(config, python=python)
    summary: JsonObject = {
        "schema_version": "switchyard.trialqa_canary_score_driver.v1",
        "operational_gate": str(config.operational_gate),
        "operational_decision": gate["decision"],
        "score_command": score,
        "promotion_gate_command": promotion,
        "authorized_rerun_command": guarded_score_canary_command(
            config,
            python=python,
            summary_output=summary_output,
            spend_review=spend_review,
            yes_spend=True,
            recover_interrupted=recover_interrupted,
        ),
        "spend_authorized": yes_spend,
        "recover_interrupted": recover_interrupted,
    }
    if not yes_spend:
        summary["status"] = "awaiting_spend_authorization"
        summary["next_step"] = "rerun with --yes-spend to execute score and promotion gate"
        if summary_output is not None:
            demo._write_json_atomic(summary_output, summary)
        return summary
    if spend_review is None:
        raise TrialQACanaryScoreError("refusing to spend without --spend-review")
    summary["spend_review_guard"] = spend_guard.validate_spend_review_for_command(
        spend_review=spend_review,
        expected_command=guarded_score_canary_command(
            config,
            python=python,
            summary_output=summary_output,
            spend_review=spend_review,
            yes_spend=True,
            recover_interrupted=recover_interrupted,
        ),
        expected_stage="score",
        recover_interrupted=recover_interrupted,
    )

    score_result = run(score, check=False)
    summary["score_returncode"] = score_result.returncode
    if score_result.returncode != 0:
        summary["status"] = "score_failed"
        if summary_output is not None:
            demo._write_json_atomic(summary_output, summary)
        return summary

    promotion_result = run(promotion, check=False)
    summary["promotion_gate_returncode"] = promotion_result.returncode
    summary["promotion_gate_output"] = str(config.promotion_gate_output)
    summary["status"] = (
        "promotion_gate_completed"
        if promotion_result.returncode == 0
        else "promotion_gate_failed"
    )
    if summary_output is not None:
        demo._write_json_atomic(summary_output, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--population-report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--switchyard", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--tooluniverse", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path, required=True)
    parser.add_argument("--question-start", type=int, required=True)
    parser.add_argument("--question-limit", type=int, required=True)
    parser.add_argument("--repeat-limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--promotion-gate-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--spend-review",
        type=Path,
        help=(
            "required with --yes-spend; verifies the reviewed spend packet and "
            "hash-bound audit bundle still match before judge calls"
        ),
    )
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help=(
            "pass interrupted-score recovery through to the batch runner; "
            "adds --recover-interrupted and --retry-failed to the score command, "
            "and still requires --yes-spend to execute scoring"
        ),
    )
    parser.add_argument("--yes-spend", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> ScoreConfig:
    return ScoreConfig(
        manifest=args.manifest,
        dataset=args.dataset,
        experiment_root=args.experiment_root,
        doctor=args.doctor,
        population_report=args.population_report,
        candidate=args.candidate,
        switchyard=args.switchyard,
        codex=args.codex,
        tooluniverse=args.tooluniverse,
        profile=args.profile,
        operational_gate=args.operational_gate,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        promotion_gate_output=args.promotion_gate_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_score_canary(
        _config_from_args(args),
        yes_spend=args.yes_spend,
        recover_interrupted=args.recover_interrupted,
        summary_output=args.summary_output,
        spend_review=args.spend_review,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())

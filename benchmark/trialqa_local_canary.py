# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded first-checkpoint TrialQA canary driver.

The default mode performs only the zero-spend readiness check and prints the
exact generation/gate commands. Passing ``--yes-spend`` executes generation and
then the operational gate. The driver intentionally stops before scoring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark.trialqa_local_demo as demo  # noqa: E402
import benchmark.trialqa_local_readiness as readiness  # noqa: E402
import benchmark.trialqa_local_spend_guard as spend_guard  # noqa: E402

JsonObject = dict[str, Any]
Run = Callable[..., subprocess.CompletedProcess[object]]


class TrialQACanaryError(RuntimeError):
    """The guarded canary cannot safely proceed."""


@dataclass(frozen=True)
class CanaryConfig:
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
    question_start: int
    question_limit: int
    repeat_limit: int
    workers: int
    max_generation_attempts: int
    readiness_output: Path
    gate_output: Path


def _path_args(config: CanaryConfig) -> list[str]:
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


def generation_command(
    config: CanaryConfig,
    *,
    python: Path | str = sys.executable,
    recover_interrupted: bool = False,
) -> list[str]:
    """Return the exact generation-only batch command."""

    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_batch",
        *_path_args(config),
        "--stage",
        "generation",
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
        command.append("--recover-interrupted")
    return command


def operational_gate_command(
    config: CanaryConfig, *, python: Path | str = sys.executable
) -> list[str]:
    """Return the exact manifest-bound operational gate command."""

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
        "operational",
        "--question-start",
        str(config.question_start),
        "--question-limit",
        str(config.question_limit),
        "--repeat-limit",
        str(config.repeat_limit),
        "--output",
        str(config.gate_output),
    ]


def guarded_canary_command(
    config: CanaryConfig,
    *,
    python: Path | str = sys.executable,
    summary_output: Path | None = None,
    spend_review: Path | None = None,
    yes_spend: bool = False,
    recover_interrupted: bool = False,
) -> list[str]:
    """Return the top-level guarded canary command for this exact config."""

    command = [
        str(python),
        "-m",
        "benchmark.trialqa_local_canary",
        *_path_args(config),
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
        "--readiness-output",
        str(config.readiness_output),
        "--gate-output",
        str(config.gate_output),
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


def build_readiness(config: CanaryConfig) -> JsonObject:
    """Run the zero-spend readiness check and persist its report."""

    report = readiness.build_readiness_report(
        manifest_path=config.manifest,
        dataset_path=config.dataset,
        experiment_root=config.experiment_root,
        doctor_report=config.doctor,
        population_report=config.population_report,
        candidate_root=config.candidate,
        switchyard_bin=config.switchyard,
        codex_bin=config.codex,
        tooluniverse_bin=config.tooluniverse,
        routing_profile=config.profile,
        question_start=config.question_start,
        question_limit=config.question_limit,
        repeat_limit=config.repeat_limit,
    )
    demo._write_json_atomic(config.readiness_output, report)
    return report


def run_canary(
    config: CanaryConfig,
    *,
    yes_spend: bool,
    recover_interrupted: bool = False,
    run: Run = subprocess.run,
    python: Path | str = sys.executable,
    summary_output: Path | None = None,
    spend_review: Path | None = None,
) -> JsonObject:
    """Run readiness, optionally generation, and then the operational gate."""

    readiness_report = build_readiness(config)
    generation = generation_command(
        config,
        python=python,
        recover_interrupted=recover_interrupted,
    )
    summary: JsonObject = {
        "schema_version": "switchyard.trialqa_canary_driver.v1",
        "readiness_status": readiness_report["status"],
        "readiness_output": str(config.readiness_output),
        "generation_command": generation,
        "authorized_rerun_command": guarded_canary_command(
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
    if readiness_report.get("status") not in readiness.GENERATION_READY_STATUSES:
        if yes_spend:
            raise TrialQACanaryError(
                f"refusing to spend because readiness status is {readiness_report.get('status')!r}"
            )
        summary["status"] = "readiness_not_clean"
        summary["next_step"] = "inspect readiness_output before rerunning"
        if summary_output is not None:
            demo._write_json_atomic(summary_output, summary)
        return summary
    gate = operational_gate_command(config, python=python)
    summary["operational_gate_command"] = gate
    if not yes_spend:
        summary["status"] = "awaiting_spend_authorization"
        summary["next_step"] = "rerun with --yes-spend to execute generation and operational gate"
        if summary_output is not None:
            demo._write_json_atomic(summary_output, summary)
        return summary
    if spend_review is None:
        raise TrialQACanaryError("refusing to spend without --spend-review")
    summary["spend_review_guard"] = spend_guard.validate_spend_review_for_command(
        spend_review=spend_review,
        expected_command=guarded_canary_command(
            config,
            python=python,
            summary_output=summary_output,
            spend_review=spend_review,
            yes_spend=True,
            recover_interrupted=recover_interrupted,
        ),
        expected_stage="generation",
        recover_interrupted=recover_interrupted,
    )

    generation_result = run(generation, check=False)
    summary["generation_returncode"] = generation_result.returncode
    if generation_result.returncode != 0:
        summary["status"] = "generation_failed"
        if summary_output is not None:
            demo._write_json_atomic(summary_output, summary)
        return summary

    gate_result = run(gate, check=False)
    summary["operational_gate_returncode"] = gate_result.returncode
    summary["gate_output"] = str(config.gate_output)
    summary["status"] = (
        "operational_gate_completed" if gate_result.returncode == 0 else "operational_gate_failed"
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
    parser.add_argument("--question-start", type=int, required=True)
    parser.add_argument("--question-limit", type=int, required=True)
    parser.add_argument("--repeat-limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=1)
    parser.add_argument("--readiness-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--spend-review",
        type=Path,
        help=(
            "required with --yes-spend; verifies the reviewed spend packet and "
            "hash-bound audit bundle still match before model calls"
        ),
    )
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help=(
            "pass interrupted-generation recovery through to the batch runner; "
            "still requires --yes-spend to execute generation"
        ),
    )
    parser.add_argument("--yes-spend", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> CanaryConfig:
    return CanaryConfig(
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
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        workers=args.workers,
        max_generation_attempts=args.max_generation_attempts,
        readiness_output=args.readiness_output,
        gate_output=args.gate_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_canary(
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

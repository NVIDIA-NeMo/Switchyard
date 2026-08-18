#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare Switchyard routing algorithms with oha and NVIDIA AIPerf."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RequiredBinary:
    """One command the benchmark must find before it starts."""

    label: str
    env_var: str
    value: str
    setup: str


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs shared by every algorithm comparison run."""

    base_url: str
    models: tuple[tuple[str, str], ...]
    concurrency: int
    request_count: int
    backend_label: str
    output_dir: Path
    oha_bin: str
    aiperf_bin: str
    tokenizer: str = "builtin"
    input_sequence_length: int = 32
    output_sequence_length: int = 8


@dataclass(frozen=True)
class BenchmarkResult:
    """Selected oha and AIPerf metrics for one routing algorithm."""

    algorithm: str
    model: str
    oha_requests_per_second: float
    oha_latency_p50_ms: float
    oha_latency_p99_ms: float
    aiperf_requests_per_second: float
    aiperf_request_latency_p50_ms: float
    aiperf_ttft_p50_ms: float
    aiperf_ttft_p99_ms: float
    aiperf_itl_p50_ms: float | None
    aiperf_output_tokens_per_second: float
    aiperf_output_tokens_per_second_per_user_p50: float | None


def positive_int(value: str) -> int:
    """Parse a command-line integer that must be greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def model_spec(value: str) -> tuple[str, str]:
    """Parse LABEL=MODEL and reject labels that cannot be directory names."""
    label, separator, model = value.partition("=")
    if not separator or not label or not model:
        raise argparse.ArgumentTypeError("must use LABEL=MODEL")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None:
        raise argparse.ArgumentTypeError(
            "LABEL may contain only letters, numbers, periods, underscores, and hyphens"
        )
    return label, model


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    command = argparse.ArgumentParser(
        description=(
            "Run sequential oha and AIPerf jobs for Switchyard models, then write one routing "
            "algorithm performance report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command.add_argument("--base-url", required=True, help="URL of a running Switchyard server")
    command.add_argument(
        "--model",
        action="append",
        required=True,
        type=model_spec,
        dest="models",
        metavar="LABEL=MODEL",
        help="algorithm label and exact model id; repeat for every route to compare",
    )
    command.add_argument(
        "--concurrency",
        type=positive_int,
        default=4,
        help="concurrent requests used by both load tools",
    )
    command.add_argument(
        "--request-count",
        type=positive_int,
        default=100,
        help="measured requests sent by each tool for each algorithm",
    )
    command.add_argument(
        "--tokenizer",
        default="builtin",
        help="AIPerf tokenizer; use the real model tokenizer for provider benchmarks",
    )
    command.add_argument(
        "--input-sequence-length",
        type=positive_int,
        default=32,
        help="synthetic AIPerf input token count",
    )
    command.add_argument(
        "--output-sequence-length",
        type=positive_int,
        default=8,
        help="requested AIPerf output token count",
    )
    command.add_argument(
        "--backend-label",
        default="unspecified backend",
        help="backend description recorded in the generated report",
    )
    command.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for logs, raw tool results, and the combined report",
    )
    command.add_argument(
        "--oha-bin",
        default=os.environ.get("OHA_BIN", "oha"),
        help="oha executable path",
    )
    command.add_argument(
        "--aiperf-bin",
        default=os.environ.get("AIPERF_BIN", "aiperf"),
        help="AIPerf executable path",
    )
    return command


def find_binary(value: str) -> str | None:
    """Return an executable path, or None when the command cannot run."""
    if os.sep in value:
        path = Path(value).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    else:
        found = shutil.which(value)
        if found is not None:
            return found
    return None


def resolve_binaries(requirements: Sequence[RequiredBinary]) -> dict[str, str]:
    """Resolve every command and print all missing-command setup instructions."""
    resolved = {}
    missing = []
    for requirement in requirements:
        path = find_binary(requirement.value)
        if path is None:
            missing.append(requirement)
        else:
            resolved[requirement.label] = path

    for requirement in missing:
        print(
            f"warning: {requirement.label} executable not found: {requirement.value}",
            file=sys.stderr,
        )
        print(f"  {requirement.setup}", file=sys.stderr)
        print(
            f"  Or set {requirement.env_var}=/path/to/{Path(requirement.value).name}",
            file=sys.stderr,
        )
    if missing:
        raise RuntimeError(f"install or build the {len(missing)} missing command(s), then rerun")
    return resolved


def run_checked(name: str, command: Sequence[str], log_path: Path) -> None:
    """Run one finite tool and keep its output in a named log file."""
    print(f"Running {name}; log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with status {result.returncode}; see {log_path}")


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object and name the invalid result file in any error."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read JSON result {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return document


def number_at(document: dict[str, object], group: str, field: str, path: Path) -> float:
    """Read one required numeric metric from a nested result object."""
    metrics = document.get(group)
    value = metrics.get(field) if isinstance(metrics, dict) else None
    if not isinstance(value, int | float):
        raise RuntimeError(f"missing numeric {group}.{field} in {path}")
    return float(value)


def optional_number_at(document: dict[str, object], group: str, field: str) -> float | None:
    """Read one metric that AIPerf may omit for a one-token response."""
    metrics = document.get(group)
    value = metrics.get(field) if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, int | float) else None


def milliseconds(value: float, unit: object, metric: str, path: Path) -> float:
    """Convert an AIPerf latency metric to milliseconds."""
    if unit == "ms":
        return value
    if unit in {"s", "seconds"}:
        return value * 1000
    raise RuntimeError(f"unsupported {metric} unit {unit!r} in {path}")


def aiperf_latency(document: dict[str, object], metric: str, field: str, path: Path) -> float:
    """Read one required AIPerf latency statistic in milliseconds."""
    value = number_at(document, metric, field, path)
    group = document.get(metric)
    unit = group.get("unit") if isinstance(group, dict) else None
    return milliseconds(value, unit, metric, path)


def optional_aiperf_latency(
    document: dict[str, object], metric: str, field: str, path: Path
) -> float | None:
    """Read one optional AIPerf latency statistic in milliseconds."""
    value = optional_number_at(document, metric, field)
    if value is None:
        return None
    group = document.get(metric)
    unit = group.get("unit") if isinstance(group, dict) else None
    return milliseconds(value, unit, metric, path)


def parse_result(algorithm: str, model: str, oha_path: Path, aiperf_path: Path) -> BenchmarkResult:
    """Combine the stable JSON summaries emitted by oha and AIPerf."""
    oha = read_json_object(oha_path)
    aiperf = read_json_object(aiperf_path)
    success_rate = number_at(oha, "summary", "successRate", oha_path)
    if success_rate != 1:
        raise RuntimeError(f"oha success rate for {model} was {success_rate:.2%}; see {oha_path}")
    error_count = optional_number_at(aiperf, "error_request_count", "avg")
    if error_count is not None and error_count != 0:
        raise RuntimeError(f"AIPerf recorded {error_count:g} errors for {model}; see {aiperf_path}")

    return BenchmarkResult(
        algorithm=algorithm,
        model=model,
        oha_requests_per_second=number_at(oha, "summary", "requestsPerSec", oha_path),
        oha_latency_p50_ms=number_at(oha, "latencyPercentiles", "p50", oha_path) * 1000,
        oha_latency_p99_ms=number_at(oha, "latencyPercentiles", "p99", oha_path) * 1000,
        aiperf_requests_per_second=number_at(aiperf, "request_throughput", "avg", aiperf_path),
        aiperf_request_latency_p50_ms=aiperf_latency(aiperf, "request_latency", "p50", aiperf_path),
        aiperf_ttft_p50_ms=aiperf_latency(aiperf, "time_to_first_token", "p50", aiperf_path),
        aiperf_ttft_p99_ms=aiperf_latency(aiperf, "time_to_first_token", "p99", aiperf_path),
        aiperf_itl_p50_ms=optional_aiperf_latency(
            aiperf, "inter_token_latency", "p50", aiperf_path
        ),
        aiperf_output_tokens_per_second=number_at(
            aiperf, "output_token_throughput", "avg", aiperf_path
        ),
        aiperf_output_tokens_per_second_per_user_p50=optional_number_at(
            aiperf, "output_token_throughput_per_user", "p50"
        ),
    )


def format_metric(value: float | None) -> str:
    """Render a report value without implying precision the benchmark lacks."""
    return "n/a" if value is None else f"{value:.2f}"


def write_report(config: BenchmarkConfig, results: Sequence[BenchmarkResult]) -> None:
    """Write machine-readable and reviewer-readable comparison reports."""
    rows = [asdict(result) for result in results]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": config.backend_label,
        "base_url": config.base_url,
        "concurrency": config.concurrency,
        "request_count_per_tool_per_algorithm": config.request_count,
        "aiperf_tokenizer": config.tokenizer,
        "aiperf_input_sequence_length": config.input_sequence_length,
        "aiperf_output_sequence_length": config.output_sequence_length,
        "results": rows,
    }
    (config.output_dir / "report.json").write_text(
        f"{json.dumps(payload, indent=2)}\n", encoding="utf-8"
    )

    fieldnames = list(rows[0])
    with (config.output_dir / "report.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Routing algorithm performance",
        "",
        f"- Backend: {config.backend_label}",
        f"- Concurrency: {config.concurrency}",
        f"- Measured requests per tool per algorithm: {config.request_count}",
        f"- AIPerf workload: streaming Chat Completions, ISL {config.input_sequence_length}, "
        f"OSL {config.output_sequence_length}, tokenizer `{config.tokenizer}`",
        "- AIPerf warmup: 1 second per algorithm",
        "- Run order: all oha and AIPerf jobs ran sequentially",
        "",
        "| Algorithm | oha req/s | oha p50 ms | oha p99 ms | AIPerf req/s | "
        "request p50 ms | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | output tok/s | "
        "output tok/s/user p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.algorithm} | {format_metric(result.oha_requests_per_second)} | "
            f"{format_metric(result.oha_latency_p50_ms)} | "
            f"{format_metric(result.oha_latency_p99_ms)} | "
            f"{format_metric(result.aiperf_requests_per_second)} | "
            f"{format_metric(result.aiperf_request_latency_p50_ms)} | "
            f"{format_metric(result.aiperf_ttft_p50_ms)} | "
            f"{format_metric(result.aiperf_ttft_p99_ms)} | "
            f"{format_metric(result.aiperf_itl_p50_ms)} | "
            f"{format_metric(result.aiperf_output_tokens_per_second)} | "
            f"{format_metric(result.aiperf_output_tokens_per_second_per_user_p50)} |"
        )
    lines.extend(
        (
            "",
            "oha uses non-streaming fixed-body requests to measure the HTTP request-rate and "
            "full-response latency ceiling. AIPerf uses streaming requests to measure LLM-aware "
            "latency and token throughput. Compare algorithms within one tool's columns; do not "
            "compare oha values directly with AIPerf values.",
            "",
            "A VidaiMock run isolates Switchyard routing and protocol overhead. It does not "
            "predict production model capacity. Use the same command against routes backed by "
            "the same real model deployment to measure end-to-end token performance.",
            "",
        )
    )
    (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_oha(config: BenchmarkConfig, algorithm: str, model: str, output_dir: Path) -> Path:
    """Run oha's fixed-body HTTP load for one algorithm."""
    request_path = output_dir / f"{algorithm}-request.json"
    result_path = output_dir / f"{algorithm}.json"
    request_path.write_text(
        json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": config.output_sequence_length,
                "stream": False,
            }
        ),
        encoding="utf-8",
    )
    run_checked(
        f"oha {algorithm} load",
        [
            config.oha_bin,
            "-n",
            str(config.request_count),
            "-c",
            str(config.concurrency),
            "--latency-correction",
            "--no-tui",
            "--method",
            "POST",
            "-T",
            "application/json",
            "-D",
            str(request_path),
            "--output-format",
            "json",
            "--output",
            str(result_path),
            f"{config.base_url.rstrip('/')}/v1/chat/completions",
        ],
        output_dir / f"{algorithm}.log",
    )
    return result_path


def run_aiperf(
    config: BenchmarkConfig,
    algorithm: str,
    model: str,
    output_dir: Path,
) -> Path:
    """Run one isolated AIPerf profile for an algorithm."""
    artifact_dir = output_dir / algorithm
    run_checked(
        f"AIPerf {algorithm} streaming profile",
        [
            config.aiperf_bin,
            "profile",
            "--model",
            model,
            "--url",
            config.base_url,
            "--endpoint-type",
            "chat",
            "--streaming",
            "--tokenizer",
            config.tokenizer,
            "--use-legacy-max-tokens",
            "--isl",
            str(config.input_sequence_length),
            "--osl",
            str(config.output_sequence_length),
            "--random-seed",
            "42",
            "--warmup-duration",
            "1",
            "--concurrency",
            str(config.concurrency),
            "--request-count",
            str(config.request_count),
            "--failed-request-threshold",
            "0",
            "--request-timeout-seconds",
            "30",
            "--ui",
            "none",
            "--artifact-dir",
            str(artifact_dir),
        ],
        output_dir / f"{algorithm}.log",
    )
    return artifact_dir / "profile_export_aiperf.json"


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Run both tools for every algorithm and return the report directory."""
    if not config.models:
        raise RuntimeError("at least one algorithm model is required")
    labels = [label for label, _model in config.models]
    if len(labels) != len(set(labels)):
        raise RuntimeError("algorithm labels must be unique")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    oha_dir = config.output_dir / "oha"
    aiperf_dir = config.output_dir / "aiperf"
    oha_dir.mkdir()
    aiperf_dir.mkdir()

    oha_paths = {}
    for algorithm, model in config.models:
        oha_paths[algorithm] = run_oha(config, algorithm, model, oha_dir)
    aiperf_paths = {}
    for algorithm, model in config.models:
        aiperf_paths[algorithm] = run_aiperf(config, algorithm, model, aiperf_dir)
    results = [
        parse_result(algorithm, model, oha_paths[algorithm], aiperf_paths[algorithm])
        for algorithm, model in config.models
    ]

    write_report(config, results)
    return config.output_dir


def default_output_dir() -> Path:
    """Choose a new timestamped directory under the repository."""
    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "routing-benchmark-results" / stamp


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the comparison, and print one actionable failure."""
    args = parser().parse_args(argv)
    try:
        binaries = resolve_binaries(
            (
                RequiredBinary(
                    "oha", "OHA_BIN", args.oha_bin, "Install oha with: cargo install oha"
                ),
                RequiredBinary(
                    "AIPerf",
                    "AIPERF_BIN",
                    args.aiperf_bin,
                    "Install AIPerf with: uv tool install --python 3.12 aiperf",
                ),
            )
        )
        output_dir = run_benchmark(
            BenchmarkConfig(
                base_url=args.base_url,
                models=tuple(args.models),
                concurrency=args.concurrency,
                request_count=args.request_count,
                tokenizer=args.tokenizer,
                input_sequence_length=args.input_sequence_length,
                output_sequence_length=args.output_sequence_length,
                backend_label=args.backend_label,
                output_dir=(args.output_dir or default_output_dir()).resolve(),
                oha_bin=binaries["oha"],
                aiperf_bin=binaries["AIPerf"],
            )
        )
    except (OSError, RuntimeError) as error:
        print(f"routing algorithm benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"Routing algorithm benchmark passed; report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

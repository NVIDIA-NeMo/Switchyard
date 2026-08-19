#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare Switchyard routes with Rust-owned scenarios, oha, and NVIDIA AIPerf."""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .aiperf_runner import aggregate_exports, run_profile, validate_aiperf_version
else:
    from aiperf_runner import aggregate_exports, run_profile, validate_aiperf_version

MAX_MATERIALIZED_REQUESTS = 1_000_000
MAX_MATERIALIZED_BYTES = 256 * 1024 * 1024
AIPERF_LIFECYCLE_ALLOWANCE_SECONDS = 120


@dataclass(frozen=True)
class RequiredBinary:
    """One command the benchmark must find before it starts."""

    label: str
    env_var: str
    value: str
    setup: str


@dataclass(frozen=True)
class DirectBaseline:
    """A backend endpoint that AIPerf calls without going through Switchyard."""

    base_url: str
    model: str
    api_key_env: str | None = None


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
    soak_bin: str
    tokenizer: str = "builtin"
    scenario_set: str = "standard"
    scenarios: tuple[str, ...] = ()
    load_profiles: tuple[str, ...] = ("fixed",)
    profile_runs: int = 3
    prompt_bytes: int = 1024
    max_output_tokens: int = 32
    context_window_tokens: int = 32_768
    request_rate: float | None = None
    scenario_backend_reset_url: str | None = None
    direct_baseline: DirectBaseline | None = None


@dataclass(frozen=True)
class BenchmarkArm:
    """One endpoint and model pair included in the comparison."""

    label: str
    model: str
    base_url: str
    bypasses_switchyard: bool
    api_key_env: str | None = None


@dataclass(frozen=True)
class ScenarioDefinition:
    """One validated entry exported by switchyard-soak."""

    id: str
    group: str
    description: str
    expected: str
    expected_error_rate_min: float
    expected_error_rate_max: float
    input_file: Path
    load_profiles: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RoutingDelta:
    """Routing counters attributable to one sequential AIPerf run."""

    model_calls: str
    model_share: str
    model_errors: str
    classifier_calls: int
    classifier_errors: int
    classifier_latency_avg_ms: float | None
    routing_overhead_avg_ms: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Selected load-tool and routing metrics for one comparison row."""

    algorithm: str
    model: str
    scenario: str
    scenario_description: str
    scenario_group: str
    load_profile: str
    expected_behavior: str
    expected_error_rate_min: float
    expected_error_rate_max: float
    expectation_met: bool
    oha_requests_per_second: float | None
    oha_latency_p50_ms: float | None
    oha_latency_p99_ms: float | None
    aiperf_requests_per_second: float | None
    aiperf_request_latency_p50_ms: float | None
    aiperf_ttft_p50_ms: float | None
    aiperf_ttft_p99_ms: float | None
    aiperf_itl_p50_ms: float | None
    aiperf_output_tokens_per_second: float | None
    aiperf_output_tokens_per_second_per_user_p50: float | None
    aiperf_error_requests: float
    aiperf_error_rate: float
    aiperf_request_throughput_cv: float | None
    aiperf_request_throughput_ci_low: float | None
    aiperf_request_throughput_ci_high: float | None
    selected_model_calls: str
    selected_model_share: str
    selected_model_errors: str
    classifier_calls: int
    classifier_errors: int
    classifier_latency_avg_ms: float | None
    routing_overhead_avg_ms: float | None
    bypasses_switchyard: bool
    aiperf_request_throughput_delta_pct: float | None = None
    aiperf_request_latency_p50_delta_ms: float | None = None
    aiperf_ttft_p50_delta_ms: float | None = None
    aiperf_ttft_p99_delta_ms: float | None = None
    aiperf_output_tokens_per_second_delta_pct: float | None = None


def positive_int(value: str) -> int:
    """Parse a command-line integer that must be greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a finite command-line number that must be greater than zero."""
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def model_spec(value: str) -> tuple[str, str]:
    """Parse LABEL=MODEL without coupling the report label to a filesystem name."""
    label, separator, model = value.partition("=")
    if not separator or not label.strip() or not model.strip():
        raise argparse.ArgumentTypeError("must use LABEL=MODEL")
    return label.strip(), model.strip()


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    command = argparse.ArgumentParser(
        description=(
            "Export Rust-owned request scenarios, run oha for the short baseline and AIPerf "
            "for every selected scenario, then write one routing performance report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command.add_argument("--base-url", required=True, help="URL of a running Switchyard server")
    command.add_argument(
        "--direct-base-url",
        help="backend URL for an AIPerf baseline that bypasses Switchyard",
    )
    command.add_argument(
        "--direct-model",
        help="backend model id for the direct baseline; requires --direct-base-url",
    )
    command.add_argument(
        "--direct-api-key-env",
        help="environment variable containing the direct backend API key",
    )
    command.add_argument(
        "--model",
        action="append",
        required=True,
        type=model_spec,
        dest="models",
        metavar="LABEL=MODEL",
        help="report label and exact model id; repeat for every route to compare",
    )
    command.add_argument("--concurrency", type=positive_int, default=4)
    command.add_argument("--request-count", type=positive_int, default=100)
    command.add_argument("--tokenizer", default="builtin")
    command.add_argument(
        "--scenario-set",
        choices=("core", "agentic", "resilience", "standard", "all"),
        default="standard",
    )
    command.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="exact Rust scenario id; repeat to preserve an explicit order",
    )
    command.add_argument(
        "--load-profile",
        action="append",
        choices=("fixed", "concurrency-knee", "traffic-burst", "all"),
        default=[],
        help="load schedule to run; repeat as needed",
    )
    command.add_argument(
        "--profile-runs",
        type=positive_int,
        default=3,
        help="AIPerf repetitions used for confidence intervals; maximum 10",
    )
    command.add_argument(
        "--request-rate",
        type=positive_float,
        help="base requests/second for traffic-burst; defaults to concurrency",
    )
    command.add_argument("--prompt-bytes", type=positive_int, default=1024)
    command.add_argument("--max-output-tokens", type=positive_int, default=32)
    command.add_argument("--context-window-tokens", type=positive_int, default=32_768)
    command.add_argument("--backend-label", default="unspecified backend")
    command.add_argument(
        "--scenario-backend-reset-url",
        help="optional local scenario-backend endpoint reset before each AIPerf cell",
    )
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--oha-bin", default=os.environ.get("OHA_BIN", "oha"))
    command.add_argument("--aiperf-bin", default=os.environ.get("AIPERF_BIN", "aiperf"))
    command.add_argument(
        "--soak-bin",
        default=os.environ.get("SWITCHYARD_SOAK_BIN", "target/release/switchyard-soak"),
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


def capture_stats(base_url: str, path: Path) -> dict[str, object]:
    """Capture Switchyard's routing counters before or after one isolated run."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/stats", timeout=10) as response:
            document = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read Switchyard stats: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError("Switchyard /v1/stats did not return a JSON object")
    path.write_text(f"{json.dumps(document, indent=2)}\n", encoding="utf-8")
    return document


def reset_scenario_backend(url: str | None) -> None:
    """Reset local failure injection so every algorithm receives the same attempts."""
    if url is None:
        return
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            document = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not reset the scenario backend: {error}") from error
    status = document.get("status") if isinstance(document, dict) else None
    if response.status != 200 or status != "reset":
        raise RuntimeError(f"scenario backend reset returned an invalid response from {url}")


def _nested_number(document: dict[str, object], section: str, field: str) -> float:
    section_value = document.get(section)
    value = section_value.get(field) if isinstance(section_value, dict) else None
    return float(value) if isinstance(value, int | float) else 0.0


def _histogram_delta(
    before: dict[str, object], after: dict[str, object], histogram: str
) -> float | None:
    count = _nested_number(after, histogram, "count") - _nested_number(before, histogram, "count")
    total = _nested_number(after, histogram, "total_ms") - _nested_number(
        before, histogram, "total_ms"
    )
    return total / count if count > 0 else None


def stats_delta(before: dict[str, object], after: dict[str, object]) -> RoutingDelta:
    """Return routing and target-counter changes for one run."""
    before_value = before.get("models")
    after_value = after.get("models")
    before_models: dict[str, object] = before_value if isinstance(before_value, dict) else {}
    after_models: dict[str, object] = after_value if isinstance(after_value, dict) else {}
    classifier_before = before.get("classifier")
    classifier_after = after.get("classifier")
    before_classifier_models = (
        classifier_before.get("models") if isinstance(classifier_before, dict) else None
    )
    after_classifier_models = (
        classifier_after.get("models") if isinstance(classifier_after, dict) else None
    )
    calls = {}
    errors = {}
    for model, current in after_models.items():
        if not isinstance(model, str) or not isinstance(current, dict):
            continue
        previous = before_models.get(model)
        previous_calls = previous.get("calls", 0) if isinstance(previous, dict) else 0
        classifier_current = (
            after_classifier_models.get(model)
            if isinstance(after_classifier_models, dict)
            else None
        )
        classifier_previous = (
            before_classifier_models.get(model)
            if isinstance(before_classifier_models, dict)
            else None
        )
        classifier_call_delta = float(
            classifier_current.get("calls", 0) if isinstance(classifier_current, dict) else 0
        ) - float(
            classifier_previous.get("calls", 0) if isinstance(classifier_previous, dict) else 0
        )
        call_delta = int(
            float(current.get("calls", 0)) - float(previous_calls) - classifier_call_delta
        )
        if call_delta > 0:
            calls[model] = call_delta
        previous_errors = previous.get("errors", 0) if isinstance(previous, dict) else 0
        classifier_error_delta = float(
            classifier_current.get("errors", 0) if isinstance(classifier_current, dict) else 0
        ) - float(
            classifier_previous.get("errors", 0) if isinstance(classifier_previous, dict) else 0
        )
        error_delta = int(
            float(current.get("errors", 0)) - float(previous_errors) - classifier_error_delta
        )
        if error_delta > 0:
            errors[model] = error_delta

    classifier_count = classifier_total = 0.0
    if isinstance(after_classifier_models, dict):
        for model, current in after_classifier_models.items():
            if not isinstance(model, str) or not isinstance(current, dict):
                continue
            previous = (
                before_classifier_models.get(model)
                if isinstance(before_classifier_models, dict)
                else None
            )
            before_record = previous if isinstance(previous, dict) else {}
            classifier_count += _nested_number(current, "model_call_latency", "count")
            classifier_count -= _nested_number(before_record, "model_call_latency", "count")
            classifier_total += _nested_number(current, "model_call_latency", "total_ms")
            classifier_total -= _nested_number(before_record, "model_call_latency", "total_ms")

    total_calls = sum(calls.values())
    shares = {model: round(call_count / total_calls, 4) for model, call_count in calls.items()}
    return RoutingDelta(
        model_calls=json.dumps(calls, sort_keys=True, separators=(",", ":")),
        model_share=json.dumps(shares, sort_keys=True, separators=(",", ":")),
        model_errors=json.dumps(errors, sort_keys=True, separators=(",", ":")),
        classifier_calls=int(
            _nested_number(after, "classifier", "total_requests")
            - _nested_number(before, "classifier", "total_requests")
        ),
        classifier_errors=int(
            _nested_number(after, "classifier", "total_errors")
            - _nested_number(before, "classifier", "total_errors")
        ),
        classifier_latency_avg_ms=(
            classifier_total / classifier_count if classifier_count > 0 else None
        ),
        routing_overhead_avg_ms=_histogram_delta(before, after, "routing_overhead"),
    )


def validate_load_profile(profile: dict[str, object], scenario_id: str) -> None:
    """Reject malformed or unbounded schedules before invoking a load generator."""
    profile_id = profile.get("id")
    kind = profile.get("kind")
    if (profile_id, kind) == ("fixed", "fixed"):
        return
    if (profile_id, kind) == ("concurrency-knee", "concurrency_knee"):
        steps = profile.get("concurrency_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(step, int) and 1 <= step <= 10_000 for step in steps)
        ):
            raise RuntimeError(f"invalid concurrency knee for scenario {scenario_id}")
        return
    if (profile_id, kind) == ("traffic-burst", "traffic_burst"):
        duration = profile.get("duration_seconds")
        points = profile.get("points")
        if not isinstance(duration, int) or not 1 <= duration <= 3_600:
            raise RuntimeError(f"invalid traffic-burst duration for scenario {scenario_id}")
        if not isinstance(points, list) or not 2 <= len(points) <= 64:
            raise RuntimeError(f"invalid traffic-burst points for scenario {scenario_id}")
        previous = -1.0
        for point in points:
            time_s = point.get("time_s") if isinstance(point, dict) else None
            multiplier = point.get("rate_multiplier") if isinstance(point, dict) else None
            if (
                not isinstance(time_s, int | float)
                or not isinstance(multiplier, int | float)
                or not math.isfinite(float(time_s))
                or not math.isfinite(float(multiplier))
                or float(time_s) <= previous
                or float(time_s) > duration
                or float(multiplier) <= 0
            ):
                raise RuntimeError(f"invalid traffic-burst point for scenario {scenario_id}")
            previous = float(time_s)
        return
    raise RuntimeError(f"unknown load profile for scenario {scenario_id}: {profile_id!r}")


def export_scenarios(config: BenchmarkConfig, index: int, model: str) -> list[ScenarioDefinition]:
    """Ask the Rust crate for the only authoritative scenario manifest."""
    output_dir = config.output_dir / "scenario-definitions" / f"algorithm-{index:02d}"
    command = [
        config.soak_bin,
        "--model",
        model,
        "--scenario-set",
        config.scenario_set,
        "--prompt-bytes",
        str(config.prompt_bytes),
        "--max-output-tokens",
        str(config.max_output_tokens),
        "--context-window-tokens",
        str(config.context_window_tokens),
        "--export-scenarios",
        str(output_dir),
    ]
    for scenario in config.scenarios:
        command.extend(("--scenario", scenario))
    run_checked(
        "scenario export",
        command,
        config.output_dir / f"scenario-export-{index:02d}.log",
    )
    document = read_json_object(output_dir / "manifest.json")
    if document.get("schema_version") != 1 or document.get("model") != model:
        raise RuntimeError(f"unsupported scenario manifest in {output_dir}")
    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise RuntimeError(f"scenario manifest has no scenarios: {output_dir}")
    definitions = []
    seen = set()
    root = output_dir.resolve()
    for entry in raw_scenarios:
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid scenario entry in {output_dir}")
        scenario_id = entry.get("id")
        group = entry.get("group")
        description = entry.get("description")
        expected = entry.get("expected")
        expected_error_rate = entry.get("expected_error_rate")
        min_error_rate = (
            expected_error_rate.get("min_rate") if isinstance(expected_error_rate, dict) else None
        )
        max_error_rate = (
            expected_error_rate.get("max_rate") if isinstance(expected_error_rate, dict) else None
        )
        input_name = entry.get("input_file")
        profiles = entry.get("load_profiles")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in seen
            or group not in {"core", "agentic", "resilience"}
            or not isinstance(description, str)
            or not description
            or not isinstance(expected, str)
            or not expected
            or not isinstance(min_error_rate, int | float)
            or not isinstance(max_error_rate, int | float)
            or not 0 <= float(min_error_rate) <= float(max_error_rate) <= 1
            or not isinstance(input_name, str)
            or not isinstance(profiles, list)
        ):
            raise RuntimeError(f"invalid scenario entry in {output_dir}")
        input_file = (output_dir / input_name).resolve()
        if not input_file.is_relative_to(root) or not input_file.is_file():
            raise RuntimeError(f"scenario input escapes or is missing from {output_dir}")
        if not all(
            isinstance(profile, dict) and isinstance(profile.get("id"), str) for profile in profiles
        ):
            raise RuntimeError(f"invalid load profile for scenario {scenario_id}")
        for profile in profiles:
            validate_load_profile(profile, scenario_id)
        seen.add(scenario_id)
        definitions.append(
            ScenarioDefinition(
                id=scenario_id,
                group=group,
                description=description,
                expected=expected,
                expected_error_rate_min=float(min_error_rate),
                expected_error_rate_max=float(max_error_rate),
                input_file=input_file,
                load_profiles=tuple(profiles),
            )
        )
    return definitions


def selected_profiles(
    config: BenchmarkConfig, scenario: ScenarioDefinition
) -> list[dict[str, object]]:
    """Select only schedules supported by this request shape."""
    requested = set(config.load_profiles)
    if "all" in requested:
        return list(scenario.load_profiles)
    return [profile for profile in scenario.load_profiles if profile.get("id") in requested]


def _metric(document: dict[str, object], name: str, statistic: str) -> tuple[float | None, object]:
    group = document.get(name)
    if isinstance(group, dict):
        value = group.get(statistic)
        return (float(value) if isinstance(value, int | float) else None, group.get("unit"))
    metrics = document.get("metrics")
    aggregate = metrics.get(f"{name}_{statistic}") if isinstance(metrics, dict) else None
    if isinstance(aggregate, dict):
        value = aggregate.get("mean")
        return (
            float(value) if isinstance(value, int | float) else None,
            aggregate.get("unit"),
        )
    return None, None


def _latency(document: dict[str, object], metric: str, statistic: str, path: Path) -> float | None:
    value, unit = _metric(document, metric, statistic)
    if value is None:
        return None
    if unit == "ms":
        return value
    if unit in {"s", "seconds"}:
        return value * 1000
    raise RuntimeError(f"unsupported {metric} unit {unit!r} in {path}")


def parse_result(
    arm: BenchmarkArm,
    scenario: ScenarioDefinition,
    load_profile: str,
    oha_path: Path | None,
    aiperf_path: Path,
    before_stats: dict[str, object],
    after_stats: dict[str, object],
) -> BenchmarkResult:
    """Combine one AIPerf run, an optional oha baseline, and routing counter deltas."""
    oha_rps = oha_p50 = oha_p99 = None
    if oha_path is not None:
        oha = read_json_object(oha_path)
        success, _unit = _metric(oha, "summary", "successRate")
        if success != 1:
            raise RuntimeError(f"oha success rate for {arm.model} was {success}; see {oha_path}")
        oha_rps, _unit = _metric(oha, "summary", "requestsPerSec")
        p50, _unit = _metric(oha, "latencyPercentiles", "p50")
        p99, _unit = _metric(oha, "latencyPercentiles", "p99")
        oha_p50 = p50 * 1000 if p50 is not None else None
        oha_p99 = p99 * 1000 if p99 is not None else None
    aiperf = read_json_object(aiperf_path)
    errors, _unit = _metric(aiperf, "error_request_count", "avg")
    error_count = errors or 0.0
    measured_requests, _unit = _metric(aiperf, "request_count", "avg")
    error_rate = (
        error_count / measured_requests
        if measured_requests is not None and measured_requests > 0
        else 1.0
    )
    expectation_met = (
        scenario.expected_error_rate_min <= error_rate <= scenario.expected_error_rate_max
    )
    throughput, _unit = _metric(aiperf, "request_throughput", "avg")
    output_throughput, _unit = _metric(aiperf, "output_token_throughput", "avg")
    user_throughput, _unit = _metric(aiperf, "e2e_output_token_throughput", "p50")
    aggregate = aiperf.get("metrics")
    throughput_stats = (
        aggregate.get("request_throughput_avg") if isinstance(aggregate, dict) else None
    )
    routing = stats_delta(before_stats, after_stats)
    return BenchmarkResult(
        algorithm=arm.label,
        model=arm.model,
        scenario=scenario.id,
        scenario_description=scenario.description,
        scenario_group=scenario.group,
        load_profile=load_profile,
        expected_behavior=scenario.expected,
        expected_error_rate_min=scenario.expected_error_rate_min,
        expected_error_rate_max=scenario.expected_error_rate_max,
        expectation_met=expectation_met,
        oha_requests_per_second=oha_rps,
        oha_latency_p50_ms=oha_p50,
        oha_latency_p99_ms=oha_p99,
        aiperf_requests_per_second=throughput,
        aiperf_request_latency_p50_ms=_latency(aiperf, "request_latency", "p50", aiperf_path),
        aiperf_ttft_p50_ms=_latency(aiperf, "time_to_first_token", "p50", aiperf_path),
        aiperf_ttft_p99_ms=_latency(aiperf, "time_to_first_token", "p99", aiperf_path),
        aiperf_itl_p50_ms=_latency(aiperf, "inter_token_latency", "p50", aiperf_path),
        aiperf_output_tokens_per_second=output_throughput,
        aiperf_output_tokens_per_second_per_user_p50=user_throughput,
        aiperf_error_requests=error_count,
        aiperf_error_rate=error_rate,
        aiperf_request_throughput_cv=(
            float(throughput_stats["cv"])
            if isinstance(throughput_stats, dict)
            and isinstance(throughput_stats.get("cv"), int | float)
            else None
        ),
        aiperf_request_throughput_ci_low=(
            float(throughput_stats["ci_low"])
            if isinstance(throughput_stats, dict)
            and isinstance(throughput_stats.get("ci_low"), int | float)
            else None
        ),
        aiperf_request_throughput_ci_high=(
            float(throughput_stats["ci_high"])
            if isinstance(throughput_stats, dict)
            and isinstance(throughput_stats.get("ci_high"), int | float)
            else None
        ),
        selected_model_calls=routing.model_calls,
        selected_model_share=routing.model_share,
        selected_model_errors=routing.model_errors,
        classifier_calls=routing.classifier_calls,
        classifier_errors=routing.classifier_errors,
        classifier_latency_avg_ms=routing.classifier_latency_avg_ms,
        routing_overhead_avg_ms=routing.routing_overhead_avg_ms,
        bypasses_switchyard=arm.bypasses_switchyard,
    )


def _percent_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (value / baseline - 1) * 100


def _difference(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _direct_baselines(
    results: Sequence[BenchmarkResult],
) -> dict[tuple[str, str], BenchmarkResult]:
    """Index the one direct-backend result allowed for each workload."""
    baselines = {}
    for result in results:
        if not result.bypasses_switchyard:
            continue
        key = (result.scenario, result.load_profile)
        if key in baselines:
            raise RuntimeError(
                f"duplicate direct baseline for {result.scenario} {result.load_profile}"
            )
        baselines[key] = result
    return baselines


def compare_to_direct_backend(results: Sequence[BenchmarkResult]) -> list[BenchmarkResult]:
    """Attach AIPerf deltas from the direct-backend row for each workload."""
    baselines = _direct_baselines(results)
    if not baselines:
        return list(results)

    compared = []
    for result in results:
        if result.bypasses_switchyard:
            compared.append(result)
            continue
        key = (result.scenario, result.load_profile)
        baseline = baselines.get(key)
        if baseline is None:
            if result.scenario_group != "resilience":
                raise RuntimeError(
                    f"missing direct baseline for {result.scenario} {result.load_profile}"
                )
            compared.append(result)
            continue
        if result.aiperf_error_rate != 0 or baseline.aiperf_error_rate != 0:
            compared.append(result)
            continue
        compared.append(
            replace(
                result,
                aiperf_request_throughput_delta_pct=_percent_change(
                    result.aiperf_requests_per_second,
                    baseline.aiperf_requests_per_second,
                ),
                aiperf_request_latency_p50_delta_ms=_difference(
                    result.aiperf_request_latency_p50_ms,
                    baseline.aiperf_request_latency_p50_ms,
                ),
                aiperf_ttft_p50_delta_ms=_difference(
                    result.aiperf_ttft_p50_ms,
                    baseline.aiperf_ttft_p50_ms,
                ),
                aiperf_ttft_p99_delta_ms=_difference(
                    result.aiperf_ttft_p99_ms,
                    baseline.aiperf_ttft_p99_ms,
                ),
                aiperf_output_tokens_per_second_delta_pct=_percent_change(
                    result.aiperf_output_tokens_per_second,
                    baseline.aiperf_output_tokens_per_second,
                ),
            )
        )
    return compared


def format_metric(value: float | None) -> str:
    """Render a report value without implying precision the benchmark lacks."""
    return "n/a" if value is None else f"{value:.2f}"


def format_delta(value: float | None, suffix: str = "") -> str:
    """Render a signed direct-backend comparison."""
    return "n/a" if value is None else f"{value:+.2f}{suffix}"


def write_report(config: BenchmarkConfig, results: Sequence[BenchmarkResult]) -> None:
    """Write machine-readable and reviewer-readable comparison reports."""
    if not results:
        raise RuntimeError("no scenario and load-profile combinations were selected")
    rows = [asdict(result) for result in results]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": config.backend_label,
        "base_url": config.base_url,
        "direct_baseline": (
            asdict(config.direct_baseline) if config.direct_baseline is not None else None
        ),
        "concurrency": config.concurrency,
        "request_count": config.request_count,
        "profile_runs": config.profile_runs,
        "scenario_set": config.scenario_set,
        "selected_scenarios": list(config.scenarios),
        "selected_load_profiles": list(config.load_profiles),
        "aiperf_tokenizer": config.tokenizer,
        "results": rows,
    }
    (config.output_dir / "report.json").write_text(
        f"{json.dumps(payload, indent=2)}\n", encoding="utf-8"
    )
    with (config.output_dir / "report.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Routing algorithm performance", ""]
    run_details = [
        f"- Backend: {config.backend_label}",
        f"- AIPerf repetitions per row: {config.profile_runs}",
        f"- Scenario set: {config.scenario_set}",
        "- Run order: algorithms ran back-to-back within each scenario and load; all jobs "
        "remained sequential",
    ]
    if config.direct_baseline is not None:
        run_details.append(
            f"- Direct baseline: `{config.direct_baseline.model}` at "
            f"`{config.direct_baseline.base_url}`; this arm bypasses Switchyard"
        )
        overhead_results = [
            result
            for result in results
            if not result.bypasses_switchyard and result.scenario_group != "resilience"
        ]
        lines.extend(
            (
                "## Routing overhead versus the direct backend",
                "",
                "The direct arm sends the same Rust-exported requests without Switchyard. These "
                "deltas isolate Switchyard overhead only when both arms use the same backend "
                "deployment, model, and response settings. Positive latency means the routed "
                "request took longer. Negative throughput means the routed path processed less "
                "work. Resilience workloads are shown later but are not compared because their "
                "failure behavior differs.",
                "",
            )
        )
        if not overhead_results:
            lines.append("No successful direct-backend comparisons were available.")
        else:
            lines.extend(
                (
                    "| Workload | Route | Load | Request p50 | TTFT p50 | Request throughput | "
                    "Token throughput |",
                    "|---|---|---|---:|---:|---:|---:|",
                )
            )
            for result in overhead_results:
                lines.append(
                    f"| {result.scenario.replace('-', ' ')} | {result.algorithm} | "
                    f"{result.load_profile} | "
                    f"{format_delta(result.aiperf_request_latency_p50_delta_ms, ' ms')} | "
                    f"{format_delta(result.aiperf_ttft_p50_delta_ms, ' ms')} | "
                    f"{format_delta(result.aiperf_request_throughput_delta_pct, '%')} | "
                    f"{format_delta(result.aiperf_output_tokens_per_second_delta_pct, '%')} |"
                )
        lines.extend(
            (
                "",
                "Use the confidence intervals in the detailed results before treating a small "
                "throughput change as meaningful. A small negative latency or positive "
                "throughput delta is run-to-run noise when those intervals overlap. Request and "
                "token throughput percentage changes match when every arm emits the same fixed "
                "output length; the detailed table shows the absolute token rates.",
                "",
                "## Workloads measured",
                "",
                "| Workload | Request pattern |",
                "|---|---|",
            )
        )
        seen_scenarios = set()
        for result in overhead_results:
            if result.scenario in seen_scenarios:
                continue
            seen_scenarios.add(result.scenario)
            lines.append(
                f"| {result.scenario.replace('-', ' ')} | {result.scenario_description} |"
            )
    lines.extend(("", "## Run details", "", *run_details))
    lines.extend(
        (
            "",
            "## Throughput and latency",
            "",
            "| Algorithm | Scenario | Load | oha req/s | AIPerf req/s | request p50 ms | "
            "TTFT p50 ms | TTFT p99 ms | ITL p50 ms | output tok/s | error rate | gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        )
    )
    for result in results:
        if result.scenario_group == "resilience":
            continue
        lines.append(
            f"| {result.algorithm} | {result.scenario} | {result.load_profile} | "
            f"{format_metric(result.oha_requests_per_second)} | "
            f"{format_metric(result.aiperf_requests_per_second)} | "
            f"{format_metric(result.aiperf_request_latency_p50_ms)} | "
            f"{format_metric(result.aiperf_ttft_p50_ms)} | "
            f"{format_metric(result.aiperf_ttft_p99_ms)} | "
            f"{format_metric(result.aiperf_itl_p50_ms)} | "
            f"{format_metric(result.aiperf_output_tokens_per_second)} | "
            f"{result.aiperf_error_rate:.2%} | "
            f"{'PASS' if result.expectation_met else 'FAIL'} |"
        )
    if config.profile_runs > 1:
        lines.extend(
            (
                "",
                "## Repeatability",
                "",
                "The script calculates variation and confidence intervals across independent "
                "AIPerf runs. "
                "Treat a small throughput delta as noise when these intervals overlap.",
                "",
                "| Algorithm | Scenario | Load | req/s average | req/s CV | req/s 95% CI |",
                "|---|---|---|---:|---:|---:|",
            )
        )
        for result in results:
            if result.scenario_group == "resilience":
                continue
            confidence_interval = (
                "n/a"
                if result.aiperf_request_throughput_ci_low is None
                or result.aiperf_request_throughput_ci_high is None
                else f"{result.aiperf_request_throughput_ci_low:.2f}–"
                f"{result.aiperf_request_throughput_ci_high:.2f}"
            )
            coefficient_of_variation = (
                "n/a"
                if result.aiperf_request_throughput_cv is None
                else f"{result.aiperf_request_throughput_cv:.2%}"
            )
            lines.append(
                f"| {result.algorithm} | {result.scenario} | {result.load_profile} | "
                f"{format_metric(result.aiperf_requests_per_second)} | "
                f"{coefficient_of_variation} | {confidence_interval} |"
            )
    lines.extend(
        (
            "",
            "## Routing behavior",
            "",
            "| Algorithm | Scenario | Load | selected target calls | selected target share | target errors | "
            "classifier calls | classifier errors | classifier avg ms | "
            "Switchyard routing time avg ms |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|",
        )
    )
    for result in results:
        if result.bypasses_switchyard:
            continue
        lines.append(
            f"| {result.algorithm} | {result.scenario} | {result.load_profile} | "
            f"`{result.selected_model_calls}` | `{result.selected_model_share}` | "
            f"`{result.selected_model_errors}` | "
            f"{result.classifier_calls} | {result.classifier_errors} | "
            f"{format_metric(result.classifier_latency_avg_ms)} | "
            f"{format_metric(result.routing_overhead_avg_ms)} |"
        )

    resilience = [result for result in results if result.scenario_group == "resilience"]
    if resilience:
        lines.extend(
            (
                "",
                "## Resilience",
                "",
                "| Algorithm | Scenario | Load | client error rate | expected range | gate | "
                "target errors | Expected behavior |",
                "|---|---|---|---:|---|---|---|---|",
            )
        )
        for result in resilience:
            if result.bypasses_switchyard:
                continue
            lines.append(
                f"| {result.algorithm} | {result.scenario} | {result.load_profile} | "
                f"{result.aiperf_error_rate:.2%} | "
                f"{result.expected_error_rate_min:.2%}–{result.expected_error_rate_max:.2%} | "
                f"{'PASS' if result.expectation_met else 'FAIL'} | "
                f"`{result.selected_model_errors}` | "
                f"{result.expected_behavior} |"
            )
    lines.extend(
        (
            "",
            "oha runs only for the fixed short-interactive baseline. AIPerf replays the same "
            "Rust-exported request payload for streaming, session-aware LLM metrics. Compare "
            "algorithms within the same scenario and load row.",
            "",
            "The local scenario backend isolates Switchyard routing and protocol overhead. It "
            "does not predict model capacity. Repeat the same command against routes backed by "
            "one real deployment to include tokenization, model execution, and provider queuing.",
            "",
        )
    )
    (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_oha(
    config: BenchmarkConfig,
    label: str,
    scenario: ScenarioDefinition,
    output_dir: Path,
) -> Path:
    """Run oha only for the fixed short-interactive baseline."""
    inputs = read_json_object(scenario.input_file)
    data = inputs.get("data")
    try:
        payload = data[0]["payloads"][0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid AIPerf input file {scenario.input_file}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid AIPerf payload in {scenario.input_file}")
    body = dict(payload)
    body["stream"] = False
    request_path = output_dir / f"{label}-request.json"
    result_path = output_dir / f"{label}.json"
    request_path.write_text(json.dumps(body), encoding="utf-8")
    run_checked(
        f"oha {label}",
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
        output_dir / f"{label}.log",
    )
    return result_path


def materialize_aiperf_input(
    scenario: ScenarioDefinition,
    output_path: Path,
    minimum_request_count: int,
    namespace: str = "load",
) -> Path:
    """Repeat Rust-owned sessions with unique ids so AIPerf never wraps the dataset."""
    if minimum_request_count > MAX_MATERIALIZED_REQUESTS:
        raise RuntimeError(
            f"AIPerf input needs {minimum_request_count} requests; maximum is "
            f"{MAX_MATERIALIZED_REQUESTS}"
        )
    document = read_json_object(scenario.input_file)
    data = document.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"AIPerf input has no sessions: {scenario.input_file}")

    validated = []
    requests_per_replica = 0
    bytes_per_replica = 0
    for raw_session in data:
        if not isinstance(raw_session, dict):
            raise RuntimeError(f"invalid AIPerf session in {scenario.input_file}")
        session_id = raw_session.get("session_id")
        payloads = raw_session.get("payloads")
        if not isinstance(session_id, str) or not isinstance(payloads, list) or not payloads:
            raise RuntimeError(f"invalid AIPerf session in {scenario.input_file}")
        validated.append((raw_session, session_id, len(payloads)))
        requests_per_replica += len(payloads)
        bytes_per_replica += len(json.dumps(raw_session).encode()) + 64
    replica_count = math.ceil(minimum_request_count / requests_per_replica)
    estimated_bytes = replica_count * bytes_per_replica
    if estimated_bytes > MAX_MATERIALIZED_BYTES:
        raise RuntimeError(
            f"AIPerf input would use about {estimated_bytes} bytes; maximum is "
            f"{MAX_MATERIALIZED_BYTES}"
        )

    sessions = []
    request_count = 0
    replica = 0
    for _replica_index in range(replica_count):
        for raw_session, session_id, payload_count in validated:
            sessions.append(
                {
                    **raw_session,
                    "session_id": f"{session_id}-{namespace}-{replica:06d}",
                }
            )
            request_count += payload_count
            replica += 1
            if request_count >= minimum_request_count:
                break
        if request_count >= minimum_request_count:
            break
    output_path.write_text(json.dumps({"data": sessions}), encoding="utf-8")
    return output_path


def profile_request_count(config: BenchmarkConfig, profile: dict[str, object]) -> int:
    """Return a conservative request bound for one AIPerf schedule."""
    if profile.get("kind") != "traffic_burst":
        return config.request_count
    duration = profile.get("duration_seconds")
    points = profile.get("points")
    if not isinstance(duration, int) or not isinstance(points, list):
        raise RuntimeError("traffic-burst profile is missing its duration or points")
    parsed_points = []
    for point in points:
        if not isinstance(point, dict):
            raise RuntimeError("traffic-burst profile has an invalid point")
        time_s = point.get("time_s")
        multiplier = point.get("rate_multiplier")
        if not isinstance(time_s, int | float) or not isinstance(multiplier, int | float):
            raise RuntimeError("traffic-burst profile has an invalid point")
        parsed_points.append((float(time_s), float(multiplier)))
    request_multiplier_seconds = sum(
        (right_time - left_time) * (left_rate + right_rate) / 2
        for (left_time, left_rate), (right_time, right_rate) in zip(
            parsed_points, parsed_points[1:], strict=False
        )
    )
    return math.ceil(request_multiplier_seconds * (config.request_rate or config.concurrency))


def aiperf_timeout_seconds(
    config: BenchmarkConfig,
    scenario: ScenarioDefinition,
    profile: dict[str, object],
    concurrency: int | None,
) -> int:
    """Bound one process by its schedule and per-request timeout."""
    request_timeout = 1 if scenario.id == "client-cancellation" else 30
    if profile.get("kind") == "traffic_burst":
        duration = profile.get("duration_seconds")
        if not isinstance(duration, int):
            raise RuntimeError(f"traffic-burst profile has no duration for {scenario.id}")
        workload_seconds = duration + request_timeout
    else:
        effective_concurrency = concurrency or config.concurrency
        workload_seconds = math.ceil(config.request_count / effective_concurrency) * request_timeout
    return AIPERF_LIFECYCLE_ALLOWANCE_SECONDS + workload_seconds


def run_aiperf(
    config: BenchmarkConfig,
    arm: BenchmarkArm,
    scenario: ScenarioDefinition,
    profile: dict[str, object],
    output_dir: Path,
    artifact_label: str,
    concurrency: int | None = None,
) -> Path:
    """Run independent AIPerf repetitions using Rust-exported inputs-json datasets."""
    artifact_root = output_dir / artifact_label
    effective_concurrency = concurrency or config.concurrency
    load_arguments: list[str] = []
    kind = profile.get("kind")
    if kind == "traffic_burst":
        points = profile.get("points")
        if not isinstance(points, list):
            raise RuntimeError(f"traffic-burst profile has no points for {scenario.id}")
        base_rate = config.request_rate or float(config.concurrency)
        rate_points = []
        for point in points:
            if not isinstance(point, dict):
                raise RuntimeError(f"invalid traffic-burst point for {scenario.id}")
            rate_points.append(
                {
                    "time_s": point["time_s"],
                    "qps": base_rate * float(point["rate_multiplier"]),
                }
            )
        series_path = output_dir / f"{artifact_label}-request-rate.json"
        series_path.write_text(
            f"{json.dumps({'points': rate_points}, indent=2)}\n", encoding="utf-8"
        )
        load_arguments.extend(
            ("--request-rate-series", str(series_path), "--arrival-pattern", "constant")
        )
        load_arguments.extend(("--benchmark-duration", str(profile.get("duration_seconds"))))
        load_arguments.extend(("--concurrency", str(config.concurrency)))
    else:
        load_arguments.extend(("--concurrency", str(effective_concurrency)))
        load_arguments.extend(("--request-count", str(config.request_count)))

    exports: list[Path] = []
    for trial in range(1, config.profile_runs + 1):
        run_label = f"{artifact_label}-run-{trial:02d}"
        input_path = materialize_aiperf_input(
            scenario,
            output_dir / f"{run_label}-inputs.json",
            profile_request_count(config, profile),
            namespace=f"{artifact_label}-run-{trial:02d}",
        )
        command = [
            config.aiperf_bin,
            "profile",
            "--model",
            arm.model,
            "--url",
            arm.base_url,
            "--endpoint-type",
            "chat",
            "--streaming",
            "--tokenizer",
            config.tokenizer,
            "--custom-dataset-type",
            "inputs-json",
            "--input-file",
            str(input_path),
            "--session-header",
            "x-switchyard-session-id",
            "--random-seed",
            "42",
            "--num-profile-runs",
            "1",
            "--record-processors",
            "1",
            "--request-timeout-seconds",
            "1" if scenario.id == "client-cancellation" else "30",
            "--ui",
            "none",
            *load_arguments,
        ]
        if arm.api_key_env is not None:
            command.extend(("--api-key", f"${{{arm.api_key_env}}}"))
        trial_root = (
            artifact_root / "profile_runs" / f"run_{trial:04d}"
            if config.profile_runs > 1
            else artifact_root
        )
        exports.append(
            run_profile(
                command,
                output_dir / f"{run_label}.log",
                trial_root,
                aiperf_timeout_seconds(config, scenario, profile, concurrency),
            )
        )
    if len(exports) == 1:
        return exports[0]
    aggregate_path: Path = aggregate_exports(
        exports,
        artifact_root / "aggregate" / "profile_export_aiperf_aggregate.json",
    )
    return aggregate_path


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Run selected scenarios for every algorithm and return the report directory."""
    if not config.models:
        raise RuntimeError("at least one algorithm model is required")
    if config.profile_runs > 10:
        raise RuntimeError("--profile-runs must be between 1 and 10")
    validate_aiperf_version(config.aiperf_bin)
    arms = [BenchmarkArm(label, model, config.base_url, False) for label, model in config.models]
    if config.direct_baseline is not None:
        arms.insert(
            0,
            BenchmarkArm(
                "direct-backend",
                config.direct_baseline.model,
                config.direct_baseline.base_url,
                True,
                config.direct_baseline.api_key_env,
            ),
        )
    labels = [arm.label for arm in arms]
    if len(labels) != len(set(labels)):
        raise RuntimeError("algorithm labels must be unique")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    oha_dir = config.output_dir / "oha"
    aiperf_dir = config.output_dir / "aiperf"
    oha_dir.mkdir()
    aiperf_dir.mkdir()
    results = []
    definition_sets = [export_scenarios(config, index, arm.model) for index, arm in enumerate(arms)]
    expected_ids = [scenario.id for scenario in definition_sets[0]]
    for definitions in definition_sets[1:]:
        if [scenario.id for scenario in definitions] != expected_ids:
            raise RuntimeError("scenario exports differ between algorithms")

    for scenario_index, baseline_scenario in enumerate(definition_sets[0]):
        for profile in selected_profiles(config, baseline_scenario):
            profile_id = str(profile["id"])
            concurrencies: list[int | None] = [None]
            if profile.get("kind") == "concurrency_knee":
                steps = profile.get("concurrency_steps")
                if not isinstance(steps, list) or not all(isinstance(step, int) for step in steps):
                    raise RuntimeError(f"invalid concurrency knee for {baseline_scenario.id}")
                concurrencies = sorted(
                    set(
                        [step for step in steps if step <= config.concurrency]
                        + [config.concurrency]
                    )
                )
            for concurrency in concurrencies:
                load_label = profile_id if concurrency is None else f"{profile_id}@{concurrency}"
                for index, arm in enumerate(arms):
                    scenario = definition_sets[index][scenario_index]
                    if arm.bypasses_switchyard and scenario.group == "resilience":
                        continue
                    artifact_label = (
                        f"algorithm-{index:02d}-{scenario.id}-{load_label.replace('@', '-')}"
                    )
                    oha_path = None
                    if (
                        not arm.bypasses_switchyard
                        and scenario.id == "short-interactive"
                        and profile_id == "fixed"
                    ):
                        oha_path = run_oha(
                            config,
                            artifact_label,
                            scenario,
                            oha_dir,
                        )
                    reset_scenario_backend(config.scenario_backend_reset_url)
                    before = (
                        {}
                        if arm.bypasses_switchyard
                        else capture_stats(
                            config.base_url,
                            aiperf_dir / f"{artifact_label}-stats-before.json",
                        )
                    )
                    aiperf_path = run_aiperf(
                        config,
                        arm,
                        scenario,
                        profile,
                        aiperf_dir,
                        artifact_label,
                        concurrency,
                    )
                    after = (
                        {}
                        if arm.bypasses_switchyard
                        else capture_stats(
                            config.base_url,
                            aiperf_dir / f"{artifact_label}-stats-after.json",
                        )
                    )
                    results.append(
                        parse_result(
                            arm,
                            scenario,
                            load_label,
                            oha_path,
                            aiperf_path,
                            before,
                            after,
                        )
                    )
    results = compare_to_direct_backend(results)
    write_report(config, results)
    failures = [result for result in results if not result.expectation_met]
    if failures:
        raise RuntimeError(
            f"{len(failures)} benchmark row(s) missed their error-rate gate; "
            f"see {config.output_dir / 'report.md'}"
        )
    return config.output_dir


def default_output_dir() -> Path:
    """Choose a new timestamped directory under the repository."""
    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "routing-benchmark-results" / stamp


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the comparison, and print one actionable failure."""
    command = parser()
    args = command.parse_args(argv)
    if (args.direct_base_url is None) != (args.direct_model is None):
        command.error("--direct-base-url and --direct-model must be used together")
    if args.direct_api_key_env is not None and args.direct_base_url is None:
        command.error("--direct-api-key-env requires --direct-base-url and --direct-model")
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
                    "Install AIPerf with: uv tool install --python 3.12 'aiperf==0.11.0'",
                ),
                RequiredBinary(
                    "switchyard-soak",
                    "SWITCHYARD_SOAK_BIN",
                    args.soak_bin,
                    "Build the scenario exporter with: cargo build --release -p switchyard-soak",
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
                scenario_set=args.scenario_set,
                scenarios=tuple(args.scenario),
                load_profiles=tuple(args.load_profile or ["fixed"]),
                profile_runs=args.profile_runs,
                prompt_bytes=args.prompt_bytes,
                max_output_tokens=args.max_output_tokens,
                context_window_tokens=args.context_window_tokens,
                request_rate=args.request_rate,
                backend_label=args.backend_label,
                output_dir=(args.output_dir or default_output_dir()).resolve(),
                oha_bin=binaries["oha"],
                aiperf_bin=binaries["AIPerf"],
                soak_bin=binaries["switchyard-soak"],
                scenario_backend_reset_url=args.scenario_backend_reset_url,
                direct_baseline=(
                    DirectBaseline(
                        args.direct_base_url,
                        args.direct_model,
                        args.direct_api_key_env,
                    )
                    if args.direct_base_url is not None and args.direct_model is not None
                    else None
                ),
            )
        )
    except (OSError, RuntimeError) as error:
        print(f"routing algorithm benchmark failed: {error}", file=sys.stderr)
        return 1
    print(f"Routing algorithm benchmark passed; report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

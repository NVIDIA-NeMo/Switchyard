# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.benchmark_routing_algorithms as benchmark
from scripts.benchmark_routing_algorithms import (
    BenchmarkArm,
    BenchmarkConfig,
    DirectBaseline,
    ScenarioDefinition,
    compare_to_direct_backend,
    materialize_aiperf_input,
    parse_result,
    profile_request_count,
    write_report,
)


def test_report_combines_scenarios_single_and_aggregate_results(tmp_path) -> None:
    oha_path = tmp_path / "oha.json"
    single_path = tmp_path / "profile_export_aiperf.json"
    direct_path = tmp_path / "profile_export_aiperf_direct.json"
    aggregate_path = tmp_path / "profile_export_aiperf_aggregate.json"
    oha_path.write_text(
        json.dumps(
            {
                "summary": {"successRate": 1.0, "requestsPerSec": 120.5},
                "latencyPercentiles": {"p50": 0.012, "p99": 0.045},
            }
        )
    )
    single_path.write_text(
        json.dumps(
            {
                "error_request_count": {"unit": "requests", "avg": 0},
                "request_count": {"unit": "requests", "avg": 4},
                "request_throughput": {"unit": "requests/sec", "avg": 95.5},
                "request_latency": {"unit": "ms", "p50": 40.0},
                "time_to_first_token": {"unit": "ms", "p50": 15.0, "p99": 35.0},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 620.0},
                "e2e_output_token_throughput": {
                    "unit": "tokens/sec/user",
                    "p50": 200.0,
                },
            }
        )
    )
    direct_path.write_text(
        json.dumps(
            {
                "error_request_count": {"unit": "requests", "avg": 0},
                "request_count": {"unit": "requests", "avg": 4},
                "request_throughput": {"unit": "requests/sec", "avg": 100.0},
                "request_latency": {"unit": "ms", "p50": 35.0},
                "time_to_first_token": {"unit": "ms", "p50": 12.0, "p99": 30.0},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 700.0},
            }
        )
    )
    aggregate_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "error_request_count_avg": {"unit": "requests", "mean": 2},
                    "request_count_avg": {"unit": "requests", "mean": 4},
                    "request_throughput_avg": {
                        "unit": "requests/sec",
                        "mean": 80.0,
                        "cv": 0.03,
                        "ci_low": 77.0,
                        "ci_high": 83.0,
                    },
                    "request_latency_p50": {"unit": "seconds", "mean": 0.05},
                    "time_to_first_token_p50": {"unit": "ms", "mean": 20.0},
                    "time_to_first_token_p99": {"unit": "ms", "mean": 45.0},
                    "output_token_throughput_avg": {
                        "unit": "tokens/sec",
                        "mean": 500.0,
                    },
                }
            }
        )
    )
    short = ScenarioDefinition(
        id="short-interactive",
        group="core",
        description="short baseline",
        expected="all requests succeed",
        expected_error_rate_min=0.0,
        expected_error_rate_max=0.0,
        input_file=tmp_path / "short.json",
        load_profiles=(),
    )
    failure = ScenarioDefinition(
        id="failure-pressure",
        group="resilience",
        description="failure injection",
        expected="failures stay bounded",
        expected_error_rate_min=0.01,
        expected_error_rate_max=0.75,
        input_file=tmp_path / "failure.json",
        load_profiles=(),
    )
    before = {
        "models": {
            "mock/weak": {"calls": 10, "errors": 0},
            "mock/classifier": {"calls": 3, "errors": 0},
        },
        "classifier": {
            "total_requests": 3,
            "total_errors": 0,
            "models": {
                "mock/classifier": {
                    "calls": 3,
                    "errors": 0,
                    "model_call_latency": {"count": 3, "total_ms": 6},
                }
            },
        },
        "routing_overhead": {"count": 10, "total_ms": 20},
    }
    after = {
        "models": {
            "mock/weak": {"calls": 20, "errors": 2},
            "mock/strong": {"calls": 2, "errors": 0},
            "mock/classifier": {"calls": 5, "errors": 1},
        },
        "classifier": {
            "total_requests": 5,
            "total_errors": 1,
            "models": {
                "mock/classifier": {
                    "calls": 5,
                    "errors": 1,
                    "model_call_latency": {"count": 5, "total_ms": 14},
                }
            },
        },
        "routing_overhead": {"count": 22, "total_ms": 50},
    }
    results = compare_to_direct_backend(
        (
            parse_result(
                BenchmarkArm(
                    "direct-backend",
                    "mock/weak",
                    "http://127.0.0.1:8100",
                    True,
                ),
                short,
                "fixed",
                None,
                direct_path,
                {},
                {},
            ),
            parse_result(
                BenchmarkArm(
                    "random",
                    "switchyard/random",
                    "http://127.0.0.1:4000",
                    False,
                ),
                short,
                "fixed",
                oha_path,
                single_path,
                before,
                after,
            ),
            parse_result(
                BenchmarkArm(
                    "classifier",
                    "switchyard/classifier",
                    "http://127.0.0.1:4000",
                    False,
                ),
                failure,
                "fixed",
                None,
                aggregate_path,
                before,
                after,
            ),
        )
    )
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:4000",
        models=(("random", "switchyard/random"),),
        concurrency=100,
        request_count=1000,
        backend_label="test backend",
        output_dir=output_dir,
        oha_bin="oha",
        aiperf_bin="aiperf",
        soak_bin="switchyard-soak",
        direct_baseline=DirectBaseline("http://127.0.0.1:8100", "mock/weak"),
    )

    write_report(config, results)

    report = (output_dir / "report.md").read_text()
    assert "## Routing overhead versus the direct backend" in report
    assert (
        report.index("## Routing overhead versus the direct backend")
        < report.index("## Run details")
        < report.index("## Throughput and latency")
    )
    assert (
        "| short interactive | random | fixed | +5.00 ms | +3.00 ms | -4.50% | -11.43% |" in report
    )
    assert "| short interactive | short baseline |" in report
    assert "| random | short-interactive | fixed | 120.50 | 95.50 |" in report
    assert "## Repeatability" in report
    assert "## Resilience" in report
    assert (
        "| classifier | failure-pressure | fixed | 50.00% | 1.00%–75.00% | PASS | `{"
        '"mock/weak":2}` | failures stay bounded |' in report
    )
    rows = list(csv.DictReader((output_dir / "report.csv").open()))
    rows_by_key = {(row["algorithm"], row["scenario"]): row for row in rows}
    random_row = rows_by_key[("random", "short-interactive")]
    assert random_row["selected_model_calls"] == '{"mock/strong":2,"mock/weak":10}'
    assert random_row["selected_model_share"] == '{"mock/strong":0.1667,"mock/weak":0.8333}'
    assert random_row["selected_model_errors"] == '{"mock/weak":2}'
    payload = json.loads((output_dir / "report.json").read_text())
    payload_by_key = {(row["algorithm"], row["scenario"]): row for row in payload["results"]}
    direct_row = payload_by_key[("direct-backend", "short-interactive")]
    random_payload = payload_by_key[("random", "short-interactive")]
    classifier_payload = payload_by_key[("classifier", "failure-pressure")]
    assert direct_row["aiperf_itl_p50_ms"] is None
    assert direct_row["aiperf_request_throughput_delta_pct"] is None
    assert random_payload["aiperf_output_tokens_per_second_per_user_p50"] == 200.0
    assert random_payload["scenario_description"] == "short baseline"
    assert classifier_payload["aiperf_request_latency_p50_ms"] == 50.0
    assert classifier_payload["aiperf_request_throughput_cv"] == 0.03
    assert classifier_payload["classifier_latency_avg_ms"] == 4.0

    with pytest.raises(RuntimeError, match="missing direct baseline for mixed-traffic fixed"):
        compare_to_direct_backend((results[0], replace(results[1], scenario="mixed-traffic")))


def test_materialized_sessions_are_unique_and_cover_request_count(tmp_path) -> None:
    scenario_path = tmp_path / "short.json"
    scenario_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "session_id": "short-0",
                        "payloads": [{"model": "test"}, {"model": "test"}],
                    }
                ]
            }
        )
    )
    scenario = ScenarioDefinition(
        id="short-interactive",
        group="core",
        description="short baseline",
        expected="all requests succeed",
        expected_error_rate_min=0.0,
        expected_error_rate_max=0.0,
        input_file=scenario_path,
        load_profiles=(),
    )

    expanded_input = materialize_aiperf_input(scenario, tmp_path / "expanded.json", 5)
    expanded_sessions = json.loads(expanded_input.read_text())["data"]

    assert len(expanded_sessions) == 3
    assert len({session["session_id"] for session in expanded_sessions}) == 3


def test_traffic_burst_request_count_integrates_rate_series(tmp_path) -> None:
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:4000",
        models=(("random", "switchyard/random"),),
        concurrency=2,
        request_count=100,
        backend_label="test backend",
        output_dir=tmp_path,
        oha_bin="oha",
        aiperf_bin="aiperf",
        soak_bin="switchyard-soak",
    )
    profile = {
        "kind": "traffic_burst",
        "duration_seconds": 30,
        "points": [
            {"time_s": 0, "rate_multiplier": 1},
            {"time_s": 10, "rate_multiplier": 1},
            {"time_s": 11, "rate_multiplier": 10},
            {"time_s": 16, "rate_multiplier": 10},
            {"time_s": 17, "rate_multiplier": 1},
            {"time_s": 30, "rate_multiplier": 1},
        ],
    }

    assert profile_request_count(config, profile) == 168


def test_aiperf_cells_use_disjoint_artifacts(tmp_path, monkeypatch) -> None:
    scenario_path = tmp_path / "short.json"
    scenario_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "session_id": "short-0",
                        "payloads": [{"model": "switchyard/random"}],
                    }
                ]
            }
        )
    )
    scenario = ScenarioDefinition(
        id="short-interactive",
        group="core",
        description="short baseline",
        expected="all requests succeed",
        expected_error_rate_min=0.0,
        expected_error_rate_max=0.0,
        input_file=scenario_path,
        load_profiles=(),
    )
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:4000",
        models=(("random", "switchyard/random"),),
        concurrency=1,
        request_count=1,
        backend_label="test backend",
        output_dir=tmp_path,
        oha_bin="oha",
        aiperf_bin="aiperf",
        soak_bin="switchyard-soak",
        profile_runs=1,
    )
    arm = BenchmarkArm(
        "random",
        "switchyard/random",
        "http://127.0.0.1:4000",
        False,
    )
    observed: list[tuple[Path, Path]] = []

    def fake_run_profile(
        _command, log_path: Path, artifact_dir: Path, _timeout_seconds: int
    ) -> Path:
        observed.append((log_path, artifact_dir))
        artifact_dir.mkdir(parents=True)
        export = artifact_dir / "profile_export_aiperf.json"
        export.write_text("{}")
        return export

    monkeypatch.setattr(benchmark, "run_profile", fake_run_profile)

    first = benchmark.run_aiperf(
        config,
        arm,
        scenario,
        {"kind": "fixed"},
        tmp_path,
        "algorithm-00-short-interactive-fixed",
    )
    second = benchmark.run_aiperf(
        config,
        arm,
        scenario,
        {"kind": "fixed"},
        tmp_path,
        "algorithm-00-long-context-fixed",
    )

    assert first != second
    assert observed[0][0] != observed[1][0]
    assert observed[0][1] != observed[1][1]

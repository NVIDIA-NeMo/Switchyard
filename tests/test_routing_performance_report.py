# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json

from scripts.benchmark_routing_algorithms import (
    BenchmarkConfig,
    ScenarioDefinition,
    parse_result,
    write_report,
)


def test_report_combines_scenarios_single_and_aggregate_results(tmp_path) -> None:
    oha_path = tmp_path / "oha.json"
    single_path = tmp_path / "profile_export_aiperf.json"
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
    results = (
        parse_result(
            "random",
            "switchyard/random",
            short,
            "fixed",
            oha_path,
            single_path,
            before,
            after,
        ),
        parse_result(
            "classifier",
            "switchyard/classifier",
            failure,
            "fixed",
            None,
            aggregate_path,
            before,
            after,
        ),
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
    )

    write_report(config, results)

    report = (output_dir / "report.md").read_text()
    assert "| random | short-interactive | fixed | 120.50 | 95.50 |" in report
    assert "## Resilience" in report
    assert (
        "| classifier | failure-pressure | fixed | 50.00% | 1.00%–75.00% | PASS | `{"
        '"mock/weak":2}` | failures stay bounded |' in report
    )
    rows = list(csv.DictReader((output_dir / "report.csv").open()))
    assert rows[0]["selected_model_calls"] == '{"mock/strong":2,"mock/weak":10}'
    assert rows[0]["selected_model_share"] == '{"mock/strong":0.1667,"mock/weak":0.8333}'
    assert rows[0]["selected_model_errors"] == '{"mock/weak":2}'
    payload = json.loads((output_dir / "report.json").read_text())
    assert payload["results"][0]["aiperf_itl_p50_ms"] is None
    assert payload["results"][0]["aiperf_output_tokens_per_second_per_user_p50"] == 200.0
    assert payload["results"][1]["aiperf_request_latency_p50_ms"] == 50.0
    assert payload["results"][1]["aiperf_request_throughput_cv"] == 0.03
    assert payload["results"][1]["classifier_latency_avg_ms"] == 4.0

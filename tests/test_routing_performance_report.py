# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from scripts.benchmark_routing_algorithms import BenchmarkConfig, parse_result, write_report


def test_report_combines_oha_and_aiperf_metrics(tmp_path) -> None:
    oha_path = tmp_path / "oha.json"
    aiperf_path = tmp_path / "profile_export_aiperf.json"
    oha_path.write_text(
        json.dumps(
            {
                "summary": {"successRate": 1.0, "requestsPerSec": 120.5},
                "latencyPercentiles": {"p50": 0.012, "p99": 0.045},
            }
        )
    )
    aiperf_path.write_text(
        json.dumps(
            {
                "error_request_count": {"unit": "requests", "avg": 0},
                "request_throughput": {"unit": "requests/sec", "avg": 95.5},
                "request_latency": {"unit": "ms", "p50": 40.0},
                "time_to_first_token": {"unit": "ms", "p50": 15.0, "p99": 35.0},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 620.0},
            }
        )
    )
    result = parse_result("random", "switchyard/random", oha_path, aiperf_path)
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    config = BenchmarkConfig(
        base_url="http://127.0.0.1:4000",
        models=(("random", "switchyard/random"),),
        concurrency=100,
        request_count=1000,
        tokenizer="builtin",
        input_sequence_length=32,
        output_sequence_length=8,
        backend_label="test backend",
        output_dir=output_dir,
        oha_bin="oha",
        aiperf_bin="aiperf",
    )

    write_report(config, (result,))

    report = (output_dir / "report.md").read_text()
    assert "| random | 120.50 | 12.00 | 45.00 | 95.50 |" in report
    assert "| 40.00 | 15.00 | 35.00 | n/a | 620.00 | n/a |" in report
    assert (output_dir / "report.csv").is_file()
    assert json.loads((output_dir / "report.json").read_text())["results"] == [
        {
            "algorithm": "random",
            "model": "switchyard/random",
            "oha_requests_per_second": 120.5,
            "oha_latency_p50_ms": 12.0,
            "oha_latency_p99_ms": 45.0,
            "aiperf_requests_per_second": 95.5,
            "aiperf_request_latency_p50_ms": 40.0,
            "aiperf_ttft_p50_ms": 15.0,
            "aiperf_ttft_p99_ms": 35.0,
            "aiperf_itl_p50_ms": None,
            "aiperf_output_tokens_per_second": 620.0,
            "aiperf_output_tokens_per_second_per_user_p50": None,
        }
    ]

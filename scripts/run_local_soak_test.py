#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run every Switchyard route with the local scenario backend and load tools."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from benchmark_routing_algorithms import (
    BenchmarkConfig,
    DirectBaseline,
    RequiredBinary,
    positive_int,
    resolve_binaries,
    run_benchmark,
    run_checked,
)

ROUTES = (
    ("noop", "switchyard/noop"),
    ("random", "switchyard/random"),
    ("passthrough", "switchyard/passthrough"),
    ("plan_execute", "switchyard/plan-execute"),
    ("llm_classifier", "switchyard/classifier"),
    ("stage_router", "switchyard/stage"),
)
MOCK_PORT = 8100


@dataclass
class Child:
    """A child process and the log file that must stay open while it runs."""

    name: str
    process: subprocess.Popen[str]
    log: TextIO
    log_path: Path


def nonnegative_int(value: str) -> int:
    """Parse a command-line integer that may be zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def tcp_port(value: str) -> int:
    """Parse a valid nonzero TCP port."""
    parsed = positive_int(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def parser() -> argparse.ArgumentParser:
    """Build the plain-English command-line interface."""
    command = argparse.ArgumentParser(
        description=(
            "Start the request-aware scenario backend and a local Switchyard server, send one "
            "request through every route, then run oha, AIPerf, and switchyard-soak."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command.add_argument(
        "--duration",
        default="10s",
        help="time to run switchyard-soak; use an s, m, or h suffix",
    )
    command.add_argument(
        "--concurrency",
        type=positive_int,
        default=4,
        help="concurrent requests used by oha, AIPerf, and the soak run",
    )
    command.add_argument(
        "--request-count",
        type=positive_int,
        default=100,
        help="measured requests each load tool sends for each routing algorithm",
    )
    command.add_argument(
        "--mock-latency-ms",
        type=nonnegative_int,
        default=40,
        help="latency before the local scenario backend sends its first token",
    )
    command.add_argument(
        "--mock-token-latency-ms",
        type=nonnegative_int,
        default=1,
        help="latency the local scenario backend adds between streamed output tokens",
    )
    command.add_argument(
        "--server-port",
        type=tcp_port,
        default=4000,
        help="local TCP port used by switchyard-server",
    )
    command.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for generated config, logs, and tool results",
    )
    return command


def start_child(name: str, command: Sequence[str], log_path: Path) -> Child:
    """Start one long-running process and retain its combined log."""
    print(f"Starting {name}; log: {log_path}")
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        log.close()
        raise
    return Child(name=name, process=process, log=log, log_path=log_path)


def stop_child(child: Child) -> None:
    """Stop one child and close its log, escalating to kill after five seconds."""
    if child.process.poll() is None:
        child.process.terminate()
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.process.kill()
            child.process.wait(timeout=5)
    child.log.close()


def wait_for_health(child: Child, url: str, expected_status: str | None = None) -> None:
    """Wait up to 30 seconds for a child process's health endpoint."""
    for _ in range(120):
        if child.process.poll() is not None:
            raise RuntimeError(
                f"{child.name} exited with status {child.process.returncode}; see {child.log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                body = response.read()
                document = json.loads(body) if expected_status is not None else None
                status = document.get("status") if isinstance(document, dict) else None
                if response.status == 200 and (
                    expected_status is None or status == expected_status
                ):
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"{child.name} did not become healthy at {url}; see {child.log_path}")


def check_routes(base_url: str, output_path: Path) -> None:
    """Send one HTTP request through every configured route."""
    records = []
    for _algorithm, route in ROUTES:
        body = json.dumps(
            {
                "model": route,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            records.append(
                {
                    "route": route,
                    "status": response.status,
                    "selected_model": response.headers.get("x-model-router-selected-model"),
                    "response": json.loads(response.read()),
                }
            )
    output_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )


def default_output_dir(repo_root: Path) -> Path:
    """Choose a new timestamped directory under the repository."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "local-soak-test-results" / stamp


def run_local_soak_test(args: argparse.Namespace) -> Path:
    """Run the complete local soak test and return its output directory."""
    repo_root = Path(__file__).resolve().parents[1]
    rust_build = (
        "Build the Rust commands with: cargo build --release -p switchyard-server "
        "-p switchyard-soak --bins --example switchyard-soak-mock"
    )
    binaries = resolve_binaries(
        (
            RequiredBinary(
                "switchyard-server",
                "SWITCHYARD_SERVER_BIN",
                os.environ.get(
                    "SWITCHYARD_SERVER_BIN",
                    str(repo_root / "target/release/switchyard-server"),
                ),
                rust_build,
            ),
            RequiredBinary(
                "switchyard-soak",
                "SWITCHYARD_SOAK_BIN",
                os.environ.get(
                    "SWITCHYARD_SOAK_BIN", str(repo_root / "target/release/switchyard-soak")
                ),
                rust_build,
            ),
            RequiredBinary(
                "scenario backend",
                "SWITCHYARD_SOAK_MOCK_BIN",
                os.environ.get(
                    "SWITCHYARD_SOAK_MOCK_BIN",
                    str(repo_root / "target/release/examples/switchyard-soak-mock"),
                ),
                rust_build,
            ),
            RequiredBinary(
                "oha",
                "OHA_BIN",
                os.environ.get("OHA_BIN", "oha"),
                "Install oha with: cargo install oha",
            ),
            RequiredBinary(
                "AIPerf",
                "AIPERF_BIN",
                os.environ.get("AIPERF_BIN", "aiperf"),
                "Install AIPerf with: uv tool install --python 3.12 'aiperf==0.11.0'",
            ),
        )
    )
    server_bin = binaries["switchyard-server"]
    soak_bin = binaries["switchyard-soak"]
    mock_bin = binaries["scenario backend"]
    oha_bin = binaries["oha"]
    aiperf_bin = binaries["AIPerf"]

    output_dir = (args.output_dir or default_output_dir(repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "routes.toml"
    shutil.copy2(repo_root / "scripts/local_soak_test.toml", config_path)

    run_checked(
        "switchyard-server config validation",
        [server_bin, "--config", str(config_path), "--dry-run"],
        output_dir / "config-validation.log",
    )

    children: list[Child] = []
    try:
        mock = start_child(
            "scenario backend",
            [
                mock_bin,
                "--port",
                str(MOCK_PORT),
                "--latency-ms",
                str(args.mock_latency_ms),
                "--token-latency-ms",
                str(args.mock_token_latency_ms),
            ],
            output_dir / "scenario-backend.log",
        )
        children.append(mock)
        wait_for_health(mock, f"http://127.0.0.1:{MOCK_PORT}/health")

        server = start_child(
            "switchyard-server",
            [server_bin, "--config", str(config_path), "--port", str(args.server_port)],
            output_dir / "switchyard-server.log",
        )
        children.append(server)
        base_url = f"http://127.0.0.1:{args.server_port}"
        wait_for_health(server, f"{base_url}/health", "ok")
        check_routes(base_url, output_dir / "route-checks.jsonl")

        run_benchmark(
            BenchmarkConfig(
                base_url=base_url,
                models=ROUTES,
                concurrency=args.concurrency,
                request_count=args.request_count,
                backend_label=(
                    "request-aware local backend with "
                    f"{args.mock_latency_ms} ms TTFT and "
                    f"{args.mock_token_latency_ms} ms per output token"
                ),
                output_dir=output_dir / "routing-benchmark",
                oha_bin=oha_bin,
                aiperf_bin=aiperf_bin,
                soak_bin=soak_bin,
                profile_runs=1,
                scenario_backend_reset_url=f"http://127.0.0.1:{MOCK_PORT}/reset",
                direct_baseline=DirectBaseline(
                    base_url=f"http://127.0.0.1:{MOCK_PORT}",
                    model="mock/weak",
                ),
            )
        )

        run_checked(
            "switchyard-soak API and streaming requests",
            [
                soak_bin,
                "--base-url",
                base_url,
                "--model",
                "switchyard/passthrough",
                "--duration",
                args.duration,
                "--concurrency",
                str(args.concurrency),
                "--report-interval",
                "2",
                "--invalid-canary-interval",
                "0",
                "--server-pid",
                str(server.process.pid),
                "--results-dir",
                str(output_dir / "soak"),
            ],
            output_dir / "soak.log",
        )
    finally:
        for child in reversed(children):
            stop_child(child)

    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the local soak test, and print one actionable failure."""
    args = parser().parse_args(argv)
    try:
        output_dir = run_local_soak_test(args)
    except (OSError, RuntimeError) as error:
        print(f"local soak test failed: {error}", file=sys.stderr)
        return 1
    print(f"Local soak test passed; results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

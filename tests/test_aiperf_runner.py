# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import time

import pytest

import scripts.aiperf_runner as aiperf_runner
from scripts.aiperf_runner import aggregate_exports, run_profile, validate_aiperf_version


def _write_stubborn_worker(worker) -> None:
    worker.write_text(
        """import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
heartbeat = Path(sys.argv[1])
while True:
    heartbeat.write_text(str(time.monotonic()))
    time.sleep(0.05)
"""
    )


def _assert_heartbeat_stopped(heartbeat) -> None:
    heartbeat_before = heartbeat.read_text()
    time.sleep(0.2)
    assert heartbeat.read_text() == heartbeat_before


def test_run_profile_timeout_kills_the_worker_process_group(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "fake_aiperf.py"
    worker = tmp_path / "worker.py"
    child_pid = tmp_path / "child-pid"
    heartbeat = tmp_path / "heartbeat"
    _write_stubborn_worker(worker)
    monkeypatch.setattr(aiperf_runner, "PROCESS_GROUP_GRACE_SECONDS", 0.2)
    fake.write_text(
        """import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, sys.argv[2], sys.argv[3]])
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    )

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="exceeded its 1-second run limit"):
        run_profile(
            [sys.executable, str(fake), str(child_pid), str(worker), str(heartbeat)],
            tmp_path / "aiperf.log",
            tmp_path / "artifacts",
            timeout_seconds=1,
        )

    assert time.monotonic() - started_at < 10
    assert child_pid.read_text().isdigit()
    _assert_heartbeat_stopped(heartbeat)
    assert "exceeded its 1-second run limit" in (tmp_path / "aiperf.log").read_text()


def test_run_profile_stops_workers_after_the_leader_exits(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "fake_aiperf.py"
    worker = tmp_path / "worker.py"
    child_pid = tmp_path / "child-pid"
    heartbeat = tmp_path / "heartbeat"
    _write_stubborn_worker(worker)
    monkeypatch.setattr(aiperf_runner, "PROCESS_GROUP_GRACE_SECONDS", 0.2)
    fake.write_text(
        """import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, sys.argv[2], sys.argv[3]])
Path(sys.argv[1]).write_text(str(child.pid))
while not Path(sys.argv[3]).exists():
    time.sleep(0.01)
"""
    )

    with pytest.raises(RuntimeError, match="did not write profile_export_aiperf.json"):
        run_profile(
            [sys.executable, str(fake), str(child_pid), str(worker), str(heartbeat)],
            tmp_path / "aiperf.log",
            tmp_path / "artifacts",
            timeout_seconds=10,
        )

    assert child_pid.read_text().isdigit()
    _assert_heartbeat_stopped(heartbeat)


def test_process_group_probe_ignores_an_unowned_reused_group(monkeypatch) -> None:
    def deny_signal(_process_group, _signal) -> None:
        raise PermissionError

    monkeypatch.setattr(aiperf_runner.os, "killpg", deny_signal)

    assert not aiperf_runner._process_group_exists(1234)


def test_validate_aiperf_version_rejects_uncovered_release(tmp_path) -> None:
    fake = tmp_path / "aiperf"
    fake.write_text("#!/bin/sh\nprintf '0.12.0\\n'\n")
    fake.chmod(0o755)

    with pytest.raises(RuntimeError, match="AIPerf 0.12.0 is unsupported"):
        validate_aiperf_version(str(fake))


def test_aggregate_exports_combines_independent_runs(tmp_path) -> None:
    exports = []
    for index, throughput in enumerate((10.0, 14.0), start=1):
        path = tmp_path / f"run-{index}.json"
        document = {
            "aiperf_version": "0.11.0",
            "error_request_count": None,
            "request_count": {"unit": "requests", "avg": 20},
            "request_throughput": {
                "unit": "requests/sec",
                "avg": throughput,
            },
            "time_to_first_token": {"unit": "ms", "p50": 10.0},
        }
        if index == 2:
            del document["error_request_count"]
            document["error_summary"] = []
        path.write_text(json.dumps(document))
        exports.append(path)

    aggregate_path = aggregate_exports(exports, tmp_path / "aggregate" / "report.json")
    metrics = json.loads(aggregate_path.read_text())["metrics"]

    assert metrics["request_throughput_avg"]["mean"] == 12.0
    assert metrics["request_throughput_avg"]["ci_low"] < 0
    assert metrics["error_request_count_avg"]["mean"] == 0.0

    mismatched = json.loads(exports[1].read_text())
    mismatched["time_to_first_token"]["unit"] = "seconds"
    exports[1].write_text(json.dumps(mismatched))
    with pytest.raises(RuntimeError, match="disagree on the unit"):
        aggregate_exports(exports, tmp_path / "invalid" / "report.json")

    mismatched["time_to_first_token"]["unit"] = "ms"
    mismatched["request_count"]["avg"] = True
    exports[1].write_text(json.dumps(mismatched))
    with pytest.raises(RuntimeError, match="has no numeric request_count.avg"):
        aggregate_exports(exports, tmp_path / "malformed" / "report.json")

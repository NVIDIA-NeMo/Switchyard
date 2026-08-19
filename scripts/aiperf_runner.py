#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the supported AIPerf release without allowing its startup race to hang."""

import json
import math
import os
import signal
import statistics
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

SUPPORTED_AIPERF_VERSION = "0.11.0"
PROCESS_GROUP_GRACE_SECONDS = 5
_T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}
_REQUIRED_STATISTICS = (
    ("error_request_count", "avg"),
    ("request_count", "avg"),
    ("request_throughput", "avg"),
)


def validate_aiperf_version(binary: str) -> None:
    """Reject AIPerf versions whose CLI or startup behavior is not covered."""
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not read the AIPerf version from {binary}: {error}") from error
    version = result.stdout.strip()
    if result.returncode != 0 or version != SUPPORTED_AIPERF_VERSION:
        found = version or result.stderr.strip() or "unknown"
        raise RuntimeError(
            f"AIPerf {found} is unsupported; install {SUPPORTED_AIPERF_VERSION} with: "
            f"uv tool install --python 3.12 'aiperf=={SUPPORTED_AIPERF_VERSION}'"
        )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A completed leader's process-group id can be reused by a process we do not own.
        return False
    return True


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop AIPerf and every worker process it started."""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        if process.poll() is None:
            raise RuntimeError(
                f"could not stop running AIPerf process group {process_group}"
            ) from error
        return
    deadline = time.monotonic() + PROCESS_GROUP_GRACE_SECONDS
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            if process.poll() is None:
                raise RuntimeError(
                    f"could not kill running AIPerf process group {process_group}"
                ) from error
    try:
        process.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"could not reap AIPerf process {process.pid}") from error


def run_profile(
    command: Sequence[str],
    log_path: Path,
    artifact_dir: Path,
    timeout_seconds: int,
) -> Path:
    """Run one bounded AIPerf process and return its verified export."""
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(
                [*command, "--artifact-dir", str(artifact_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise RuntimeError(f"could not start AIPerf: {error}") from error
        stopped = False
        try:
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                timeout_message = f"AIPerf exceeded its {timeout_seconds}-second run limit"
                log.write(f"\n{timeout_message}.\n".encode())
                log.flush()
                _stop_process_group(process)
                stopped = True
                raise RuntimeError(f"{timeout_message}; see {log_path}") from error
        finally:
            if not stopped and (process.poll() is None or _process_group_exists(process.pid)):
                _stop_process_group(process)
    if returncode != 0:
        raise RuntimeError(f"AIPerf failed with status {returncode}; see {log_path}")
    export_path = artifact_dir / "profile_export_aiperf.json"
    if not export_path.is_file():
        raise RuntimeError(f"AIPerf did not write {export_path.name}; see {log_path}")
    return export_path


def _summary(values: Sequence[float], unit: str) -> dict[str, float | str | None]:
    """Return the 95% confidence summary used by the benchmark report."""
    count = len(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if count > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(count)
    critical = _T_CRITICAL_95.get(count)
    margin = critical * standard_error if critical is not None else 0.0
    return {
        "mean": mean,
        "std": standard_deviation,
        "min": min(values),
        "max": max(values),
        "cv": standard_deviation / mean if mean else 0.0,
        "se": standard_error,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "t_critical": critical,
        "unit": unit,
    }


def aggregate_exports(exports: Sequence[Path], output_path: Path) -> Path:
    """Combine independent AIPerf exports into the aggregate schema consumed by the report."""
    if not 2 <= len(exports) <= 10:
        raise RuntimeError("AIPerf aggregation requires between 2 and 10 exports")
    collected: dict[tuple[str, str, str], list[float]] = {}
    metric_units: dict[tuple[str, str], str] = {}
    for path in exports:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not read AIPerf export {path}: {error}") from error
        if not isinstance(document, dict):
            raise RuntimeError(f"AIPerf export is not a JSON object: {path}")
        version = document.get("aiperf_version")
        if version != SUPPORTED_AIPERF_VERSION:
            raise RuntimeError(
                f"AIPerf export {path} has version {version!r}; expected {SUPPORTED_AIPERF_VERSION}"
            )
        error_count = document.get("error_request_count")
        clean_error_summary = document.get("error_summary") == []
        if error_count is None and ("error_request_count" in document or clean_error_summary):
            document["error_request_count"] = {"unit": "requests", "avg": 0.0}
        for metric_name, statistic in _REQUIRED_STATISTICS:
            metric = document.get(metric_name)
            value = metric.get(statistic) if isinstance(metric, dict) else None
            unit = metric.get("unit") if isinstance(metric, dict) else None
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not isinstance(unit, str)
            ):
                raise RuntimeError(f"AIPerf export {path} has no numeric {metric_name}.{statistic}")
        for metric_name, metric in document.items():
            if not isinstance(metric, dict) or not isinstance(metric.get("unit"), str):
                continue
            unit = metric["unit"]
            for statistic, value in metric.items():
                if statistic == "unit" or isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    previous_unit = metric_units.setdefault((metric_name, statistic), unit)
                    if unit != previous_unit:
                        raise RuntimeError(
                            "AIPerf exports disagree on the unit for "
                            f"{metric_name}.{statistic}: {previous_unit!r} and {unit!r}"
                        )
                    collected.setdefault((metric_name, statistic, unit), []).append(float(value))
    metrics = {
        f"{metric}_{statistic}": _summary(values, unit)
        for (metric, statistic, unit), values in collected.items()
        if len(values) == len(exports)
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(
        f"{json.dumps({'aiperf_version': SUPPORTED_AIPERF_VERSION, 'metrics': metrics}, indent=2)}\n",
        encoding="utf-8",
    )
    return output_path

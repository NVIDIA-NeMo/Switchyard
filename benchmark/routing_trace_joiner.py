# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join Switchyard request traces to Harbor task results."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield valid JSON object rows, warning about rows that cannot be used."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping invalid JSON at %s:%d", path, line_number)
                continue
            if isinstance(row, dict):
                yield row


def _reward(result: dict[str, Any]) -> Any:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    return rewards.get("reward") if isinstance(rewards, dict) else None


def join_routing_traces(job_dir: Path, trace_path: Path) -> int:
    """Write task-local routing traces and return the joined request count."""
    traces: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(trace_path):
        request_id = row.get("request_id")
        if isinstance(request_id, str):
            traces.setdefault(request_id, []).append(row)

    joined_count = 0
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        request_map = result_path.parent / "artifacts" / "request_map.jsonl"
        joined: list[dict[str, Any]] = []
        for mapping in _read_jsonl(request_map):
            request_id = mapping.get("request_id")
            if not isinstance(request_id, str) or request_id not in traces:
                continue
            joined.append(
                {
                    "task_name": result.get("task_name"),
                    "trial_name": result.get("trial_name"),
                    "request_id": request_id,
                    "reward": _reward(result),
                    "events": traces[request_id],
                }
            )
        output_path = result_path.parent / "artifacts" / "routing_trace.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            for row in joined:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        output_path.chmod(0o600)
        joined_count += len(joined)
    return joined_count


def main() -> int:
    """Run the routing trace joiner CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--routing-trace", type=Path, required=True)
    args = parser.parse_args()
    count = join_routing_traces(args.job_dir, args.routing_trace)
    print(f"joined {count} routed requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

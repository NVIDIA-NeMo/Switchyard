# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract per-call cache usage from captured server logs and reconcile counts."""

import argparse
import json
from pathlib import Path


def summarize(log: str, stats: dict, session_id: str) -> dict:
    """Keep individual calls and reject incomplete or inconsistent capture."""
    session = stats.get("sessions", {}).get(session_id)
    if session is None and stats.get("session_id") == session_id:
        session = stats
    if session is None:
        raise ValueError("session is absent from final routing stats")
    calls = []
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        if "Routing usage recorded" not in line:
            continue
        _, separator, payload = line.partition("record=")
        if not separator:
            raise ValueError("usage event has no record field")
        record, _ = decoder.raw_decode(payload.lstrip())
        if isinstance(record, str):
            record = json.loads(record)
        # RoutingRecord omits this field when its value is zero.
        record.setdefault("cache_creation_1h_tokens", 0)
        if record.get("session_id") == session_id:
            calls.append(record)
    if len(calls) != session.get("total_calls"):
        raise ValueError("captured call count differs from final routing stats")
    for field, total in [
        ("prompt_tokens", "total_prompt_tokens"),
        ("completion_tokens", "total_completion_tokens"),
        ("cached_tokens", "total_cached_tokens"),
        ("cache_creation_tokens", "total_cache_creation_tokens"),
    ]:
        values = [call.get(field) for call in calls]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"missing or invalid {field}")
        if sum(values) != session.get(total):
            raise ValueError(f"captured {field} differs from final routing stats")
    previous_by_agent = {}
    seen_by_agent = {}
    for call in calls:
        agent = call.get("agent_id")
        model = call["model"]
        target = call.get("target") or model
        prompt = call["prompt_tokens"]
        read = call["cached_tokens"]
        write = call["cache_creation_tokens"]
        if read + write > prompt:
            raise ValueError("cache tokens exceed aggregate prompt tokens")
        call["uncached_input_tokens"] = prompt - read - write
        call["cache_read_fraction"] = read / prompt if prompt else None
        call["routing_call"] = call.get("tier") == "classifier"
        if call["routing_call"]:
            call["answer_target_changed"] = None
            call["return_to_previously_served_target"] = None
            continue
        previous = previous_by_agent.get(agent)
        seen = seen_by_agent.setdefault(agent, set())
        call["answer_target_changed"] = previous is not None and previous != target
        call["return_to_previously_served_target"] = previous != target and target in seen
        previous_by_agent[agent] = target
        seen.add(target)
    return {
        "session_id": session_id,
        "status": "captured_usage_reconciles_with_session_stats",
        "calls": calls,
        "call_count": len(calls),
        "answer_target_changes": sum(call["answer_target_changed"] is True for call in calls),
        "cost": session.get("cost"),
        "limits": [
            "Reconciliation covers recorded calls. It does not prove capture of billed failed attempts.",
            "Cache fields retain the server's reported values. Missing provider detail may have become zero upstream.",
            "A target change is not itself a cache miss. Inspect the cache fields on each call.",
            "Provider caches may be shared by other traffic. These observations do not identify an isolated causal switching cost.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("stats", type=Path)
    parser.add_argument("session_id")
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text(), json.loads(args.stats.read_text()), args.session_id), indent=2))


if __name__ == "__main__":
    main()

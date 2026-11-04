# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price unique Switchyard attempts from one workflow's ATOF events using Relay."""

import argparse
import json
import math
from pathlib import Path

from nemo_relay import codecs


def summarize(events, workflow_uuid):
    """Return observed spending; a missing total means coverage or pricing is incomplete.

    Configure Relay's pricing component before calling this function to enable
    estimates. Provider-reported charges work without a pricing catalog.
    This covers instrumented completion attempts, including routing calls.
    """
    scopes = {e["uuid"]: e for e in events if e.get("kind") == "scope"}

    def belongs(event):
        seen = set()
        while event["uuid"] not in seen:
            if event["uuid"] == workflow_uuid:
                return True
            seen.add(event["uuid"])
            event = scopes.get(event.get("parent_uuid"))
            if event is None:
                return False
        return False

    ends = {
        e["uuid"]: e
        for e in events
        if e.get("kind") == "scope" and e.get("scope_category") == "end" and belongs(e)
    }
    calls = [e for e in scopes.values() if e.get("name") == "libsy.client_call" and belongs(e)]
    attempts = [
        e for e in ends.values()
        if e.get("metadata", {}).get("switchyard.accounting.unit") == "completion_attempt"
    ]
    coverage = bool(calls) and workflow_uuid in ends
    for call in calls:
        meta = ends.get(call["uuid"], {}).get("metadata", {})
        children = [e for e in attempts if e.get("parent_uuid") == call["uuid"]]
        expected = meta.get("switchyard.accounting.attempts")
        coverage &= (
            meta.get("switchyard.accounting.unit") == "summary"
            and isinstance(expected, int)
            and expected >= 0
            and expected == meta.get("switchyard.accounting.attempt_records") == len(children)
            and sorted(e["metadata"]["attempt"] for e in children) == list(range(1, expected + 1))
        )
    coverage &= all(e.get("parent_uuid") in {c["uuid"] for c in calls} for e in attempts)
    records = []
    for event in attempts:
        meta = event["metadata"]
        usage = codecs.AnthropicMessagesCodec().decode_response(event["data"]).usage or {}
        cost = usage.get("cost", {})
        amount = cost.get("total")
        priced = isinstance(amount, (float, int)) and math.isfinite(amount) and amount >= 0
        complete = priced and (
            cost.get("source") == "provider_reported"
            or meta.get("switchyard.accounting.usage_status") == "complete"
        )
        records.append({
            "uuid": event["uuid"], "model": event["data"].get("model"),
            "outcome": meta.get("outcome"),
            "usage_status": meta.get("switchyard.accounting.usage_status"),
            "cost": cost if priced else None, "cost_complete": complete,
        })
    subtotal = math.fsum(r["cost"]["total"] for r in records if r["cost"] is not None)
    complete = coverage and all(r["cost_complete"] for r in records)
    return {
        "workflow_uuid": workflow_uuid, "attempts": records,
        "coverage_complete": bool(coverage), "cost_status": "complete" if complete else "incomplete",
        "known_cost_subtotal_usd": subtotal, "total_cost_usd": subtotal if complete else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="ATOF JSONL containing the workflow's scope tree")
    parser.add_argument("--workflow-uuid", required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    print(json.dumps(summarize(events, args.workflow_uuid), indent=2))


if __name__ == "__main__":
    main()

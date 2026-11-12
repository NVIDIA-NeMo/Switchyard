# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price Switchyard completion attempts and decision calls within one ATOF workflow."""

import argparse
import json
import math
from pathlib import Path

from nemo_relay import codecs


def summarize(events, workflow_uuid):
    """Return observed spending; a missing total means coverage or pricing is incomplete.

    Configure Relay's pricing component before calling this function to enable
    estimates. Provider-reported charges work without a pricing catalog.
    Coverage applies only to observed instrumented completion calls and typed
    decision calls. It excludes other workflow work and cannot detect calls
    whose entire telemetry is absent.
    """
    scopes = {e["uuid"]: e for e in events if e.get("kind") == "scope"}

    def belongs(event, roots=(workflow_uuid,)):
        seen = set()
        while event["uuid"] not in seen:
            if event["uuid"] in roots:
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
    decisions = [
        e for e in ends.values()
        if e.get("metadata", {}).get("switchyard.accounting.unit") == "decision_call"
    ]
    known_workflows = {workflow_uuid} | {e["uuid"] for e in scopes.values() if e.get("category") == "agent"}
    unattributed = {
        e["uuid"]: {"uuid": e["uuid"], "context_status": "unattributed"}
        for e in events if e.get("kind") == "scope" and e.get("scope_category") == "end"
        and e.get("metadata", {}).get("switchyard.accounting.unit") == "decision_call"
        and not belongs(e, known_workflows)
    }
    # Existing decision marks establish record coverage, but are never priced.
    decision_marks = {
        e["uuid"]: e for e in events
        if e.get("kind") == "mark" and e.get("name") == "switchyard.routing.decision_call" and belongs(e)
    }
    coverage = bool(calls or decisions) and workflow_uuid in ends
    coverage &= len(decisions) == len(decision_marks)
    coverage &= not unattributed
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
    for event in attempts + decisions:
        meta = event["metadata"]
        usage = codecs.AnthropicMessagesCodec().decode_response(event["data"]).usage or {}
        cost = usage.get("cost", {})
        amount = cost.get("total")
        priced = (
            cost.get("currency", "USD").upper() == "USD"
            and isinstance(amount, (float, int)) and math.isfinite(amount) and amount >= 0
        )
        complete = priced and (
            cost.get("source") == "provider_reported"
            or meta.get("switchyard.accounting.usage_status") == "complete"
        )
        records.append({
            "uuid": event["uuid"], "model": event["data"].get("model"),
            "outcome": meta.get("outcome"),
            "usage_status": meta.get("switchyard.accounting.usage_status"),
            "context_status": "bound",
            "cost": cost if priced else None, "cost_complete": complete,
        })
    subtotal = math.fsum(r["cost"]["total"] for r in records if r["cost"] is not None)
    complete = coverage and all(r["cost_complete"] for r in records)
    return {
        "workflow_uuid": workflow_uuid, "attempts": records[:len(attempts)], "decisions": records[len(attempts):],
        "unattributed_decisions": list(unattributed.values()),
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

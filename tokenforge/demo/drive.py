"""Scenario driver — generates traffic that exercises every design claim.

Five scenarios, each mapping to a specific assertion in the design spec:

  1. Baseline enterprise traffic          -> tenant-attributed rating + margin
  2. Multi-agent run with sub-agents      -> spend tree rolls up to the parent task
  3. Budget walk ok -> throttle -> deny   -> Phase B enforcement, clean 402
  4. Entitlement ceiling                  -> strong request forced to weak
  5. Unattributed + invalid key           -> quarantine and 401
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from _http import get_json, post_json

EDGE = "http://127.0.0.1:9000"
SWITCHYARD = "http://127.0.0.1:8080"

PROMPT = [{"role": "user", "content": "Summarize the Q3 revenue variance."}]


def call(api_key: str, route: str, *, tier: str = "weak",
         session: Optional[Dict[str, str]] = None) -> tuple:
    headers = {"authorization": "Bearer " + api_key,
               "x-tokenforge-requested-tier": tier}
    for name, value in (session or {}).items():
        headers[name] = value
    return post_json(EDGE + "/v1/chat/completions",
                     {"model": route, "messages": PROMPT, "max_tokens": 2048},
                     headers)


def scenario_baseline(log: List[str]) -> None:
    for _ in range(6):
        call("m360_key_acme", "tf-tiered", tier="weak")
    for _ in range(4):
        call("m360_key_acme", "tf-escalating", tier="strong")
    log.append("1. Baseline      : 10 Acme FSI requests across 2 routes")


def scenario_spend_tree(log: List[str]) -> str:
    """A parent agent fans out to three sub-agents on one task."""
    task_id = "task_" + uuid.uuid4().hex[:8]
    session_id = "sess_" + uuid.uuid4().hex[:8]
    parent = "agent_orchestrator"

    call("m360_key_acme", "tf-escalating", tier="strong", session={
        "x-switchyard-session-id": session_id,
        "x-switchyard-task-id": task_id,
        "x-switchyard-agent-id": parent,
        "x-switchyard-is-subagent": "false",
    })
    for index in range(3):
        call("m360_key_acme", "tf-tiered", tier="weak", session={
            "x-switchyard-session-id": session_id,
            "x-switchyard-task-id": task_id,
            "x-switchyard-agent-id": "agent_worker_%d" % index,
            "x-switchyard-parent-agent-id": parent,
            "x-switchyard-is-subagent": "true",
        })
    log.append("2. Spend tree    : 1 orchestrator + 3 sub-agents on task %s" % task_id)
    return task_id


def scenario_budget_walk(log: List[str]) -> None:
    """Northwind has a $0.35 cap. Drive it through every budget state."""
    states: List[str] = []
    denied = 0
    throttled = 0
    for _ in range(40):
        status, response = call("m360_key_northwind", "tf-escalating", tier="strong")
        if status == 402:
            denied += 1
            states.append("deny")
            if denied >= 3:
                break
        else:
            budgets = get_json(EDGE + "/v1/budgets")
            row = next(r for r in budgets["tenants"] if r["tenant_id"] == "acct_9107")
            states.append(row["budget_state"])
    decisions = get_json(EDGE + "/v1/decisions")["decisions"]
    throttled = sum(1 for d in decisions
                    if d.get("tenant_id") == "acct_9107" and d.get("action") == "throttle")
    log.append("3. Budget walk   : Northwind reached %s -- %d throttled, %d denied (402)"
               % (states[-1] if states else "?", throttled, denied))


def scenario_entitlement(log: List[str]) -> None:
    """Sovereign tenant asks for `strong` but is entitled to `weak` only."""
    call("m360_key_sovereign", "tf-nemotron", tier="strong")
    status, _ = call("m360_key_sovereign", "tf-tiered", tier="weak")
    decisions = get_json(EDGE + "/v1/decisions")["decisions"]
    ceiling = sum(1 for d in decisions if d.get("reason") == "entitlement_ceiling")
    not_entitled = sum(1 for d in decisions if d.get("code") == "not_entitled")
    log.append("4. Entitlements  : %d forced to weak by ceiling, %d route denials (403=%s)"
               % (ceiling, not_entitled, status))


def scenario_negative(log: List[str]) -> None:
    """Invalid key at the Edge; unattributed request straight at Switchyard."""
    bad_status, _ = call("m360_key_forged", "tf-tiered")

    # Bypass the Edge entirely -- exactly what an attacker or a misconfigured
    # client would do. Switchyard has no auth, so it SERVES the request; the
    # intake record then arrives with no tenant and Core quarantines it.
    served, _ = post_json(SWITCHYARD + "/v1/chat/completions",
                          {"model": "tf-tiered", "messages": PROMPT},
                          {"x-switchyard-intake-enabled": "true"})
    log.append("5. Negative      : forged key -> %d at Edge; direct-to-Switchyard "
               "bypass -> %d (served, unauthenticated) then quarantined"
               % (bad_status, served))


def run() -> Dict[str, object]:
    log: List[str] = []
    scenario_baseline(log)
    task_id = scenario_spend_tree(log)
    scenario_budget_walk(log)
    scenario_entitlement(log)
    scenario_negative(log)
    return {"log": log, "task_id": task_id}

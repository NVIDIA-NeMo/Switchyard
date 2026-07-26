#!/usr/bin/env python3
"""One-command TokenForge x Switchyard demo.

    python3 tokenforge/demo/run_demo.py

Starts the Switchyard stand-in (:8080), TokenForge Core (:9900) and TokenForge
Edge (:9000), drives five scenarios, prints a report and writes a dashboard to
tokenforge/demo/dashboard.html.

Stdlib only -- no pip install, no venv, no Rust toolchain. Python 3.9+.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashboard
import drive
import switchyard_sim
import tokenforge_core
import tokenforge_edge
from _http import get_json, get_text

CORE_INTAKE = "http://127.0.0.1:9900/v1/intake/switchyard"


def main() -> int:
    print(_banner())

    switchyard_sim.configure(intake_target_url=CORE_INTAKE)
    for service in (switchyard_sim.service, tokenforge_core.service,
                    tokenforge_edge.service):
        service.start()
        print("  started %-18s :%d" % (service.name, service.port))
    time.sleep(0.4)

    print("\n  SWITCHYARD_INTAKE_TARGET_URL = %s\n" % CORE_INTAKE)
    print("  Driving scenarios")
    print("  " + "-" * 68)
    result = drive.run()
    for line in result["log"]:
        print("  " + line)

    # The intake sink is asynchronous; give it a moment to drain before
    # reconciling. In production this is why reconciliation runs on a lag.
    time.sleep(1.2)

    reconcile = get_json("http://127.0.0.1:9900/v1/reconcile")
    margin = get_json("http://127.0.0.1:9900/v1/margin")
    budgets = get_json("http://127.0.0.1:9000/v1/budgets")
    tree = get_json("http://127.0.0.1:9900/v1/spend-tree/%s" % result["task_id"])
    metrics = get_text("http://127.0.0.1:8080/metrics")

    _print_margin(margin)
    _print_budgets(budgets)
    _print_tree(tree)
    _print_reconcile(reconcile)
    _print_metrics(metrics)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(path, "w") as handle:
        handle.write(dashboard.render(margin, budgets, tree, reconcile,
                                      get_json("http://127.0.0.1:9000/v1/decisions")))
    print("\n  Dashboard written to %s" % path)
    print("  Live endpoints (Ctrl-C to stop):")
    for url in ("http://127.0.0.1:9900/v1/margin",
                "http://127.0.0.1:9900/v1/reconcile",
                "http://127.0.0.1:9000/v1/budgets",
                "http://127.0.0.1:8080/metrics"):
        print("    " + url)

    if "--serve" in sys.argv:
        print("\n  Serving. Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def _banner() -> str:
    return """
  TokenForge x NVIDIA Switchyard -- Phase A0/B0 demo
  ====================================================================
  Switchyard stand-in :8080   routing fabric (no auth, no tenancy)
  TokenForge Core     :9900   intake, rating, margin, reconciliation
  TokenForge Edge     :9000   authN/Z, budget, tier shaping, meter
"""


def _print_margin(margin: dict) -> None:
    print("\n  Margin by tenant")
    print("  " + "-" * 68)
    print("  %-12s %8s %10s %10s %10s %7s" % (
        "tenant", "reqs", "revenue", "cost", "margin", "mult"))
    for tenant_id, row in sorted(margin["tenants"].items()):
        print("  %-12s %8d %10.4f %10.4f %10.4f %6.2fx" % (
            tenant_id, row["requests"], row["revenue_usd"], row["cost_usd"],
            row["margin_usd"], row["margin_multiple"]))

    print("\n  Margin by model (the tier-economics argument)")
    print("  " + "-" * 68)
    for model, row in sorted(margin["models"].items()):
        print("  %-42s %6.2fx  %5.1f%%" % (
            model[-42:], row["margin_multiple"], row["margin_pct"]))


def _print_budgets(budgets: dict) -> None:
    print("\n  Budget state (Phase B enforcement)")
    print("  " + "-" * 68)
    print("  %-22s %10s %8s %7s  %s" % ("tenant", "spent", "cap", "used", "state"))
    for row in budgets["tenants"]:
        print("  %-22s %10.4f %8.2f %6.1f%%  %s" % (
            row["name"], row["spent_usd"], row["cap_usd"],
            row["used_pct"], row["budget_state"].upper()))


def _print_tree(tree: dict) -> None:
    print("\n  Spend tree for %s" % tree["task_id"])
    print("  " + "-" * 68)
    for node in tree["nodes"]:
        prefix = "    +-- " if node["is_subagent"] else "  "
        print("%s%-26s %6d tok  rev %.5f  cost %.5f" % (
            prefix, node["agent_id"], node["tokens"],
            node["revenue_usd"], node["cost_usd"]))
    total = sum(n["revenue_usd"] for n in tree["nodes"])
    print("  %-30s rolled-up revenue %.5f" % ("TASK TOTAL", total))


def _print_reconcile(reconcile: dict) -> None:
    print("\n  Metering integrity -- three independent meters")
    print("  " + "-" * 68)
    for name, row in reconcile["meters"].items():
        print("  %-18s %5s reqs  %9s tok   basis=%s" % (
            name, row["requests"], row["tokens"], row["basis"]))
    print("  intake loss vs settled meter : %.2f%%   verdict=%s"
          % (reconcile["intake_loss_pct"], reconcile["verdict"].upper()))
    print("  quarantined                  : %d %s"
          % (reconcile["quarantined"], json.dumps(reconcile["quarantine_reasons"])))


def _print_metrics(metrics: str) -> None:
    print("\n  Switchyard /metrics excerpt (no cost metric, no tenant label)")
    print("  " + "-" * 68)
    for line in metrics.splitlines():
        if line.startswith("switchyard_requests_total{") or line.startswith(
                "switchyard_prompt_tokens_total "):
            print("  " + line)


if __name__ == "__main__":
    sys.exit(main())

"""Renders the demo dashboard to a self-contained HTML file.

Colour roles come from the validated reference palette, used in fixed slot order
and never cycled:
  slot 1 blue   #2a78d6 / #3987e5  -> revenue
  slot 2 orange #eb6834 / #d95926  -> supplier cost
  status good/warning/serious/critical -> budget state, always with a text label
  so state is never carried by colour alone.
"""

from __future__ import annotations

from typing import Any, Dict, List

CSS = """
:root { color-scheme: light; }
.viz {
  --surface-0: #ffffff; --surface-1: #fcfcfb; --surface-2: #f4f4f1;
  --border: #e2e2dd; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #78776f;
  --revenue: #2a78d6; --cost: #eb6834;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz {
    color-scheme: dark;
    --surface-0: #131312; --surface-1: #1a1a19; --surface-2: #232322;
    --border: #33332f; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --text-muted: #96958b;
    --revenue: #3987e5; --cost: #d95926;
  }
}
:root[data-theme="dark"] .viz {
  color-scheme: dark;
  --surface-0: #131312; --surface-1: #1a1a19; --surface-2: #232322;
  --border: #33332f; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --text-muted: #96958b;
  --revenue: #3987e5; --cost: #d95926;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); }
.viz {
  background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
  padding: 32px 28px 56px; max-width: 1080px; margin: 0 auto;
}
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-secondary); font-size: 14px; margin: 0 0 28px; }
h2 { font-size: 15px; margin: 34px 0 4px; letter-spacing: -0.005em; }
.note { color: var(--text-muted); font-size: 13px; margin: 0 0 14px; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.tile .label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase;
  letter-spacing: 0.04em; }
.tile .value { font-size: 27px; font-weight: 600; margin-top: 6px;
  font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.tile .foot { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
.legend { display: flex; gap: 18px; align-items: center; font-size: 13px;
  color: var(--text-secondary); margin: 0 0 12px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px; vertical-align: -1px; }
.rows { display: flex; flex-direction: column; gap: 12px; }
.row { display: grid; grid-template-columns: 190px 1fr 118px; gap: 14px; align-items: center; }
.rowname { font-size: 13px; color: var(--text-secondary); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.track { display: flex; flex-direction: column; gap: 2px; }
.bar { height: 11px; border-radius: 0 4px 4px 0; min-width: 2px; position: relative; }
.bar.rev { background: var(--revenue); }
.bar.cost { background: var(--cost); }
.rowval { font-size: 13px; text-align: right; font-variant-numeric: tabular-nums;
  color: var(--text-primary); }
.gauge { height: 13px; background: var(--surface-2); border-radius: 4px;
  overflow: hidden; border: 1px solid var(--border); }
.gauge > span { display: block; height: 100%; border-radius: 0 3px 3px 0; }
.pill { font-size: 11px; font-weight: 600; letter-spacing: 0.04em; padding: 2px 7px;
  border-radius: 5px; color: #0b0b0b; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--text-secondary); font-weight: 500; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.04em; }
.tree { font: 13px/1.9 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; overflow-x: auto; }
.scroll { overflow-x: auto; }
.tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px;
  background: var(--surface-2); color: var(--text-secondary); margin-left: 6px; }
"""


def _e(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _tile(label: str, value: str, foot: str = "") -> str:
    return ('<div class="tile"><div class="label">%s</div><div class="value">%s</div>'
            '<div class="foot">%s</div></div>' % (_e(label), _e(value), _e(foot)))


def _bars(rows: List[Dict[str, Any]], maximum: float) -> str:
    out = []
    for row in rows:
        rev_w = (row["revenue_usd"] / maximum * 100) if maximum else 0
        cost_w = (row["cost_usd"] / maximum * 100) if maximum else 0
        out.append(
            '<div class="row"><div class="rowname" title="%s">%s</div>'
            '<div class="track">'
            '<div class="bar rev" style="width:%.2f%%" title="revenue $%.4f"></div>'
            '<div class="bar cost" style="width:%.2f%%" title="supplier cost $%.4f"></div>'
            '</div><div class="rowval">%.2fx <span class="tag">%.0f%%</span></div></div>'
            % (_e(row["label"]), _e(row["label"]), rev_w, row["revenue_usd"],
               cost_w, row["cost_usd"], row["margin_multiple"], row["margin_pct"])
        )
    return '<div class="rows">%s</div>' % "".join(out)


_STATE_COLOR = {"ok": "var(--good)", "warn": "var(--warning)",
                "throttle": "var(--serious)", "deny": "var(--critical)"}


def render(margin: dict, budgets: dict, tree: dict, reconcile: dict,
           decisions: dict) -> str:
    tenants = margin.get("tenants", {})
    models = margin.get("models", {})

    revenue = sum(r["revenue_usd"] for r in tenants.values())
    cost = sum(r["cost_usd"] for r in tenants.values())
    requests = sum(r["requests"] for r in tenants.values())
    multiple = (revenue / cost) if cost else 0.0
    escalations = sum(r["escalations"] for r in tenants.values())

    decision_rows = decisions.get("decisions", [])
    denied = sum(1 for d in decision_rows if d.get("action") == "deny")
    throttled = sum(1 for d in decision_rows if d.get("action") == "throttle")

    meters = reconcile.get("meters", {})
    verdict = reconcile.get("verdict", "ok")

    tenant_rows = sorted(
        [dict(r, label=tid) for tid, r in tenants.items()],
        key=lambda r: -r["revenue_usd"])
    model_rows = sorted(
        [dict(r, label=m.split("/")[-1]) for m, r in models.items()],
        key=lambda r: -r["revenue_usd"])
    bar_max = max([r["revenue_usd"] for r in tenant_rows + model_rows] or [1])

    tiles = "".join([
        _tile("Revenue", "$%.4f" % revenue, "%d rated requests" % requests),
        _tile("Supplier cost", "$%.4f" % cost, "from Switchyard cost_usd"),
        _tile("Margin multiple", "%.2fx" % multiple,
              "%.1f%% margin" % ((revenue - cost) / revenue * 100 if revenue else 0)),
        _tile("Escalations", str(escalations), "billable governance events"),
        _tile("Enforcement", "%d / %d" % (denied, throttled), "denied / throttled"),
        _tile("Intake loss", "%.2f%%" % reconcile.get("intake_loss_pct", 0.0),
              "verdict: %s" % verdict.upper()),
    ])

    legend = ('<div class="legend">'
              '<span><span class="swatch" style="background:var(--revenue)"></span>Revenue</span>'
              '<span><span class="swatch" style="background:var(--cost)"></span>'
              'Supplier cost</span></div>')

    budget_rows = []
    for row in budgets.get("tenants", []):
        state = row["budget_state"]
        pct = min(row["used_pct"], 100.0)
        budget_rows.append(
            '<div class="row"><div class="rowname">%s</div>'
            '<div class="gauge"><span style="width:%.1f%%;background:%s"></span></div>'
            '<div class="rowval"><span class="pill" style="background:%s">%s</span></div>'
            '</div>' % (_e(row["name"]), pct, _STATE_COLOR.get(state, "var(--good)"),
                        _STATE_COLOR.get(state, "var(--good)"), _e(state.upper())))

    budget_table = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>$%.4f</td><td>$%.2f</td>"
        "<td>%.1f%%</td><td>%s</td></tr>"
        % (_e(r["name"]), _e(r["route_tag"]), _e(r["max_tier"]), r["spent_usd"],
           r["cap_usd"], r["used_pct"], _e(r["budget_state"].upper()))
        for r in budgets.get("tenants", []))

    tree_lines = []
    for node in tree.get("nodes", []):
        prefix = "    └── " if node["is_subagent"] else "  "
        tree_lines.append("%s%-24s %6d tok   rev $%.5f   cost $%.5f" % (
            prefix, node["agent_id"], node["tokens"],
            node["revenue_usd"], node["cost_usd"]))
    tree_total = sum(n["revenue_usd"] for n in tree.get("nodes", []))
    tree_lines.append("  %-26s rolled up  $%.5f" % ("TASK TOTAL", tree_total))

    meter_table = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (_e(name.replace("_", " ")), _e(row.get("requests")), _e(row.get("tokens")),
           _e(row.get("basis")))
        for name, row in meters.items())

    model_table = "".join(
        "<tr><td>%s</td><td>%d</td><td>%d</td><td>$%.4f</td><td>$%.4f</td><td>%.2fx</td></tr>"
        % (_e(r["label"]), r["requests"], r["tokens"], r["revenue_usd"],
           r["cost_usd"], r["margin_multiple"])
        for r in model_rows)

    return """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TokenForge x Switchyard - demo</title>
<style>%(css)s</style>
<div class="viz">
  <h1>TokenForge &times; NVIDIA Switchyard</h1>
  <p class="sub">Phase A0/B0 demo &mdash; tenant-attributed token metering, margin,
  and budget enforcement over a Switchyard routing fabric.</p>

  <div class="tiles">%(tiles)s</div>

  <h2>Margin by tenant</h2>
  <p class="note">Switchyard reports supplier <em>cost</em>; TokenForge owns the
  customer <em>price</em>. The gap is the product.</p>
  %(legend)s
  %(tenant_bars)s

  <h2>Margin by served model</h2>
  <p class="note">Tier economics: the cheap tier carries the higher multiple, which
  is what makes a budget-driven downgrade profitable rather than merely cheaper.</p>
  %(model_bars)s
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th>Requests</th><th>Tokens</th><th>Revenue</th>
    <th>Cost</th><th>Multiple</th></tr></thead>
    <tbody>%(model_table)s</tbody></table></div>

  <h2>Budget state</h2>
  <p class="note">Enforced at TokenForge Edge, before Switchyard sees the request &mdash;
  so a denial burns no classifier tokens. State is labelled, never colour-only.</p>
  <div class="rows">%(budget_rows)s</div>
  <div class="scroll"><table>
    <thead><tr><th>Tenant</th><th>Route tag</th><th>Max tier</th><th>Spent</th>
    <th>Cap</th><th>Used</th><th>State</th></tr></thead>
    <tbody>%(budget_table)s</tbody></table></div>

  <h2>Spend tree &mdash; %(task_id)s</h2>
  <p class="note">Built from Switchyard's own agent-hierarchy headers. Sub-agent
  spend rolls up to the parent task, and the task to a tenant contract.</p>
  <div class="tree">%(tree)s</div>

  <h2>Metering integrity</h2>
  <p class="note">Three independent meters. The Edge meter is synchronous and
  in-path, so it is the invoice basis; the intake sink is asynchronous and may drop
  on a full queue, so it is enrichment and cross-check only.</p>
  <div class="scroll"><table>
    <thead><tr><th>Meter</th><th>Requests</th><th>Tokens</th><th>Basis</th></tr></thead>
    <tbody>%(meter_table)s</tbody></table></div>
  <p class="note">Intake loss %(loss).2f%% &middot; verdict <strong>%(verdict)s</strong>
  &middot; quarantined %(quarantined)d &middot; block threshold
  %(block).1f%%</p>
</div>
""" % {
        "css": CSS,
        "tiles": tiles,
        "legend": legend,
        "tenant_bars": _bars(tenant_rows, bar_max),
        "model_bars": _bars(model_rows, bar_max),
        "model_table": model_table,
        "budget_rows": "".join(budget_rows),
        "budget_table": budget_table,
        "task_id": _e(tree.get("task_id", "")),
        "tree": _e("\n".join(tree_lines)),
        "meter_table": meter_table,
        "loss": reconcile.get("intake_loss_pct", 0.0),
        "verdict": _e(verdict.upper()),
        "quarantined": reconcile.get("quarantined", 0),
        "block": reconcile.get("drift_block_pct", 2.0),
    }

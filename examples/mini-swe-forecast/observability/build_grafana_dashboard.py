# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generates grafana/dashboards/per-call.json (Grafana reads Langfuse's ClickHouse directly)."""

import json

DS = {"type": "grafana-clickhouse-datasource", "uid": "lfch"}


def M(k):
    return f"metadata_values[indexOf(metadata_names,'attributes.{k}')]"


LATEST = "latest (auto)"
RUNSEL = (
    f"if('$run'='{LATEST}', (SELECT argMax({M('demo.run_id')}, start_time) FROM events_full "
    f"WHERE name='lightgbm.forecast'), '$run')"
)
FC = f"""fc AS (SELECT trace_id, toInt32OrZero({M("demo.call_index")}) AS call FROM events_full
         WHERE name='lightgbm.forecast' AND {M("demo.run_id")} = {RUNSEL})"""


def SN(n):
    return f"if(countIf(s.name='{n}')=0, NULL, maxIf(s.value, s.name='{n}'))"


pid = [100]


def tgt(sql):
    return [
        {
            "refId": "A",
            "datasource": DS,
            "editorType": "sql",
            "format": 1,
            "queryType": "table",
            "rawSql": sql,
        }
    ]


def color(name, c, dash=False, width=2, points=True):
    props = [
        {"id": "color", "value": {"mode": "fixed", "fixedColor": c}},
        {"id": "custom.lineWidth", "value": width},
        {"id": "custom.showPoints", "value": "always" if points else "never"},
    ]
    if dash:
        props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}})
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


def trend(title, desc, sql, x, y, w, h, unit, overrides):
    pid[0] += 1
    return {
        "id": pid[0],
        "type": "trend",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": tgt(sql),
        "options": {
            "xField": "call",
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear",
                    "showPoints": "always",
                    "pointSize": 8,
                    "spanNulls": True,
                },
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "call"},
                    "properties": [
                        {"id": "unit", "value": "none"},
                        {"id": "displayName", "value": "model call #"},
                    ],
                }
            ]
            + overrides,
        },
    }


def simple(
    typ,
    title,
    desc,
    sql,
    x,
    y,
    w,
    h,
    options=None,
    defaults=None,
    overrides=None,
    transformations=None,
):
    pid[0] += 1
    p = {
        "id": pid[0],
        "type": typ,
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": tgt(sql),
        "options": options or {},
        "fieldConfig": {"defaults": defaults or {}, "overrides": overrides or []},
    }
    if transformations:
        p["transformations"] = transformations
    return p


def row(title, y):
    pid[0] += 1
    return {
        "id": pid[0],
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


panels = []
# ---------------- All runs ----------------
PER_RUN = f"""WITH f AS (SELECT trace_id, {M("demo.run_id")} AS run, toInt32OrZero({M("demo.call_index")}) AS call
           FROM events_full WHERE name='lightgbm.forecast'),
c AS (SELECT f.run AS run, f.call AS call, maxIf(s.value, s.name='actual_call_cost') AS actual,
             maxIf(s.value, s.name='pred_next_call_cost_mid') AS pred_mid, maxIf(s.value, s.name='pred_total_task_cost') AS pred_total
      FROM scores s JOIN f ON s.trace_id=f.trace_id GROUP BY run, call)
SELECT run, sum(actual) AS actual_cost, sum(pred_mid) AS predicted_sum_of_next_calls,
       argMin(pred_total, call) AS predicted_at_task_start, argMax(pred_total, call) AS predicted_at_last_call
FROM c WHERE run != '' GROUP BY run"""
panels.append(row("All runs", 0))
panels.append(
    simple(
        "barchart",
        "Overall cost per run: actual vs predicted",
        "actual_cost = real total. predicted_sum_of_next_calls = sum of per-call predictions (= Langfuse predicted:opus-4-8). "
        "predicted_at_task_start = whole-task bill predicted before call 1. predicted_at_last_call = estimate before the final call.",
        PER_RUN + " ORDER BY max(call) DESC",
        0,
        1,
        12,
        9,
        options={
            "xField": "run",
            "orientation": "horizontal",
            "showValue": "always",
            "groupWidth": 0.8,
            "legend": {"showLegend": True, "placement": "bottom", "displayMode": "list"},
            "tooltip": {"mode": "multi"},
        },
        defaults={"unit": "currencyUSD", "decimals": 3},
        overrides=[
            {
                "matcher": {"id": "byName", "options": n},
                "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}],
            }
            for n, c in (
                ("actual_cost", "red"),
                ("predicted_sum_of_next_calls", "blue"),
                ("predicted_at_task_start", "purple"),
                ("predicted_at_last_call", "light-blue"),
            )
        ],
    )
)
panels.append(
    simple(
        "barchart",
        "Cost by model: actual vs predicted (same as Langfuse)",
        "",
        "SELECT if(provided_model_name='predicted:opus-4-8','predicted:opus-4-8', provided_model_name) AS model, sum(total_cost) AS cost "
        "FROM events_full WHERE type='GENERATION' AND provided_model_name != '' GROUP BY model ORDER BY model",
        12,
        1,
        6,
        9,
        options={
            "xField": "model",
            "showValue": "always",
            "colorByField": "model",
            "legend": {"showLegend": False},
        },
        defaults={"unit": "currencyUSD", "decimals": 3, "color": {"mode": "palette-classic"}},
    )
)
panels.append(
    simple(
        "stat",
        "All runs: total actual vs predicted",
        "",
        f"SELECT sum(actual_cost) AS actual, sum(predicted_sum_of_next_calls) AS predicted_per_call_sum, "
        f"sum(predicted_at_task_start) AS predicted_at_start FROM ({PER_RUN})",
        18,
        1,
        6,
        9,
        options={
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "horizontal",
            "textMode": "value_and_name",
            "colorMode": "value",
            "graphMode": "none",
        },
        defaults={"unit": "currencyUSD", "decimals": 3},
        overrides=[
            {
                "matcher": {"id": "byName", "options": n},
                "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}],
            }
            for n, c in (
                ("actual", "red"),
                ("predicted_per_call_sum", "blue"),
                ("predicted_at_start", "purple"),
            )
        ],
    )
)

# ---------------- Selected run ----------------
panels.append(row("Selected run: $run", 10))
INFO = f"""WITH {FC}
SELECT
  anyIf({M("demo.task_id")}, name='lightgbm.forecast') AS task,
  anyIf({M("gen_ai.response.model")}, type='GENERATION' AND provided_model_name!='predicted:opus-4-8') AS model,
  anyIf({M("switchyard.algorithm")}, name='libsy.run') AS router,
  anyIf({M("forecast.checkpoint")}, name='lightgbm.forecast') AS predictor_checkpoint,
  anyIf({M("forecast.variant")}, name='lightgbm.forecast') AS variant,
  substring(anyIf({M("forecast.checkpoint_sha256")}, name='lightgbm.forecast'), 1, 12) AS checkpoint_sha,
  anyIf(metadata_values[indexOf(metadata_names,'resourceAttributes.service.name')], name='lightgbm.forecast') AS agent_service,
  countIf(name='agent.call') AS calls_done,
  sumIf(total_cost, type='GENERATION' AND provided_model_name!='predicted:opus-4-8') AS actual_cost
FROM events_full WHERE trace_id IN (SELECT trace_id FROM fc)"""
panels.append(
    simple(
        "table",
        "Run info: model, router, predictor",
        "From OTel span attributes",
        INFO,
        0,
        11,
        24,
        3,
        options={"showHeader": True, "cellHeight": "sm"},
        overrides=[
            {
                "matcher": {"id": "byName", "options": "actual_cost"},
                "properties": [
                    {"id": "unit", "value": "currencyUSD"},
                    {"id": "decimals", "value": 4},
                ],
            }
        ],
    )
)

NEXT = f"""WITH {FC}
SELECT fc.call AS call, {SN("actual_call_cost")} AS actual, {SN("pred_next_call_cost_mid")} AS predicted_mid,
       {SN("pred_next_call_cost_low")} AS predicted_low, {SN("pred_next_call_cost_high")} AS predicted_high
FROM scores s JOIN fc ON s.trace_id=fc.trace_id GROUP BY call ORDER BY call"""
panels.append(
    trend(
        "Next model call: actual vs predicted cost ($)",
        "Before each call the forecaster predicts a cost band (dashed low/high) and best guess (mid). Red = actual.",
        NEXT,
        0,
        14,
        24,
        9,
        "currencyUSD",
        [
            color("actual", "red", width=3),
            color("predicted_mid", "blue"),
            color("predicted_low", "light-blue", dash=True, points=False),
            color("predicted_high", "light-blue", dash=True, points=False),
        ],
    )
)

TOK = [
    ("Uncached input tokens", "uncached_input_tokens", "actual.uncached_input_tokens"),
    ("Cached input tokens (cache reads)", "cached_input_tokens", "actual.cache_read_input_tokens"),
    ("Cache-write input tokens", "cache_write_input_tokens", "actual.cache_write_input_tokens"),
    ("Output tokens", "output_tokens", "actual.output_tokens"),
]
for i, (title, fk, ak) in enumerate(TOK):
    p = f"forecast.next_call.{fk}."
    sql = f"""WITH
f AS (SELECT trace_id, toInt32OrZero({M("demo.call_index")}) AS call,
             toFloat64OrNull({M(p + "lower_inclusive")}) AS lo,
             if({M(p + "upper_unbounded")}='true', NULL, toFloat64OrNull({M(p + "upper_exclusive")})) AS hi
      FROM events_full WHERE name='lightgbm.forecast' AND {M("demo.run_id")} = {RUNSEL}),
a AS (SELECT trace_id, toFloat64OrNull({M(ak)}) AS actual FROM events_full
      WHERE name='agent.call' AND {M("demo.run_id")} = {RUNSEL})
SELECT f.call AS call, a.actual AS actual, if(f.hi IS NULL, f.lo, (f.lo+f.hi)/2) AS predicted_mid,
       f.lo AS predicted_low, f.hi AS predicted_high
FROM f LEFT JOIN a ON f.trace_id=a.trace_id ORDER BY call"""
    panels.append(
        trend(
            f"Next call tokens: {title}",
            "Predicted band for the next call vs actual tokens. Empty band = not predicted by this checkpoint.",
            sql,
            (i % 2) * 12,
            23 + (i // 2) * 8,
            12,
            8,
            "short",
            [
                color("actual", "red", width=3),
                color("predicted_mid", "blue"),
                color("predicted_low", "light-blue", dash=True, points=False),
                color("predicted_high", "light-blue", dash=True, points=False),
            ],
        )
    )

TASK = f"""WITH {FC}
SELECT call, actual_cost_so_far, predicted_total, predicted_remaining, max(actual_cost_so_far) OVER () AS actual_final_bill
FROM (SELECT fc.call AS call, {SN("actual_cost_so_far")} AS actual_cost_so_far, {SN("pred_total_task_cost")} AS predicted_total,
             {SN("pred_remaining_task_cost")} AS predicted_remaining
      FROM scores s JOIN fc ON s.trace_id=fc.trace_id GROUP BY call) ORDER BY call"""
panels.append(
    trend(
        "Whole task: predicted final bill vs actual ($)",
        "predicted_total = spent so far + predicted remaining, at each call. Dashed red = real final bill (so far, while running).",
        TASK,
        0,
        39,
        12,
        9,
        "currencyUSD",
        [
            color("actual_cost_so_far", "red", width=3),
            color("predicted_total", "blue", width=3),
            color("predicted_remaining", "purple"),
            color("actual_final_bill", "red", dash=True, points=False),
        ],
    )
)
CALLS = f"""WITH {FC}
SELECT call, predicted_remaining_calls, (max(call) OVER ()) - call + 1 AS actual_remaining_calls
FROM (SELECT fc.call AS call, {SN("pred_remaining_calls")} AS predicted_remaining_calls
      FROM scores s JOIN fc ON s.trace_id=fc.trace_id GROUP BY call) ORDER BY call"""
panels.append(
    trend(
        "Remaining model calls: predicted vs actual",
        "Actual remaining is final only once the run finishes.",
        CALLS,
        12,
        39,
        12,
        9,
        "none",
        [
            color("actual_remaining_calls", "red", width=3),
            color("predicted_remaining_calls", "blue", width=3),
        ],
    )
)

PHASE = f"""WITH {FC},
ph AS (SELECT trace_id, {M("ir.observed.phase")} AS phase FROM events_full WHERE name='agent.call')
SELECT ph.phase AS phase, {SN("actual_call_cost")} AS actual, {SN("pred_next_call_cost_mid")} AS predicted
FROM scores s JOIN fc ON s.trace_id=fc.trace_id JOIN ph ON ph.trace_id=fc.trace_id
GROUP BY s.trace_id, phase"""
panels.append(
    simple(
        "barchart",
        "Cost by IR phase: actual vs predicted ($)",
        "Phase = ir.observed.phase of the call (what the call actually did).",
        f"SELECT phase, sum(actual) AS actual, sum(predicted) AS predicted FROM ({PHASE}) WHERE phase != '' GROUP BY phase ORDER BY actual DESC",
        0,
        48,
        8,
        10,
        options={
            "xField": "phase",
            "orientation": "horizontal",
            "showValue": "auto",
            "groupWidth": 0.75,
            "barWidth": 0.9,
            "xTickLabelMaxLength": 24,
            "xTickLabelSpacing": 0,
            "legend": {"showLegend": True, "placement": "bottom", "displayMode": "list"},
        },
        defaults={"unit": "currencyUSD", "decimals": 3},
        overrides=[
            {
                "matcher": {"id": "byName", "options": n},
                "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}],
            }
            for n, c in (("actual", "red"), ("predicted", "blue"))
        ],
    )
)

DETAIL = f"""WITH {FC},
sc AS (SELECT s.trace_id AS trace_id, {SN("actual_call_cost")} AS actual_cost, {SN("pred_next_call_cost_low")} AS pred_low,
              {SN("pred_next_call_cost_high")} AS pred_high, {SN("pred_remaining_calls")} AS pred_remaining_calls,
              {SN("next_call_in_band")} AS in_band
       FROM scores s WHERE s.trace_id IN (SELECT trace_id FROM fc) GROUP BY s.trace_id),
sp AS (SELECT trace_id,
              anyIf({M("ir.source.phase")}, name='lightgbm.forecast') AS phase_before,
              anyIf({M("ir.observed.phase")}, name='agent.call') AS phase_after,
              anyIf({M("ir.observed.categories")}, name='agent.call') AS categories_raw,
              anyIf(toFloat64OrNull({M("actual.uncached_input_tokens")}), name='agent.call') AS uncached_in,
              anyIf(toFloat64OrNull({M("actual.cache_read_input_tokens")}), name='agent.call') AS cache_read,
              anyIf(toFloat64OrNull({M("actual.cache_write_input_tokens")}), name='agent.call') AS cache_write,
              anyIf(toFloat64OrNull({M("actual.output_tokens")}), name='agent.call') AS output
       FROM events_full WHERE trace_id IN (SELECT trace_id FROM fc) GROUP BY trace_id)
SELECT fc.call AS call, sp.phase_before AS phase_before, sp.phase_after AS phase_after,
       sp.uncached_in, sp.cache_read, sp.cache_write, sp.output,
       sc.actual_cost, sc.pred_low, sc.pred_high, sc.in_band, sc.pred_remaining_calls
FROM fc LEFT JOIN sc ON sc.trace_id=fc.trace_id LEFT JOIN sp ON sp.trace_id=fc.trace_id ORDER BY call DESC"""
phase_colors = {
    "Initial": "purple",
    "Exploration": "blue",
    "Experiment/Measurement": "yellow",
    "Artifact Editing": "orange",
    "Verification": "green",
    "Mixed": "light-purple",
    "Unknown": "gray",
}
panels.append(
    simple(
        "table",
        "Per-call details (newest first): IR phase, tokens, cost",
        "phase_before = ir.source.phase at forecast time; phase_after = ir.observed.phase of the call.",
        DETAIL,
        8,
        48,
        16,
        10,
        options={"showHeader": True, "cellHeight": "sm"},
        overrides=[
            {
                "matcher": {"id": "byRegexp", "options": "actual_cost|pred_low|pred_high"},
                "properties": [
                    {"id": "unit", "value": "currencyUSD"},
                    {"id": "decimals", "value": 4},
                ],
            },
            {
                "matcher": {"id": "byRegexp", "options": "phase_.*"},
                "properties": [
                    {"id": "custom.cellOptions", "value": {"type": "color-background"}},
                    {
                        "id": "mappings",
                        "value": [
                            {
                                "type": "value",
                                "options": {
                                    k: {"color": v, "index": i}
                                    for i, (k, v) in enumerate(phase_colors.items())
                                },
                            }
                        ],
                    },
                ],
            },
            {
                "matcher": {"id": "byName", "options": "in_band"},
                "properties": [
                    {
                        "id": "mappings",
                        "value": [
                            {
                                "type": "value",
                                "options": {
                                    "1": {"text": "✓", "color": "green"},
                                    "0": {"text": "✗", "color": "red"},
                                },
                            }
                        ],
                    },
                    {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                ],
            },
            {
                "matcher": {"id": "byName", "options": "pred_remaining_calls"},
                "properties": [{"id": "decimals", "value": 1}],
            },
        ],
    )
)

var = {
    "name": "run",
    "label": "Run",
    "type": "query",
    "datasource": DS,
    "refresh": 2,
    "sort": 0,
    "query": f"SELECT run FROM (SELECT '{LATEST}' AS run, toDateTime64('2100-01-01 00:00:00', 6) AS t UNION ALL "
    f"SELECT {M('demo.run_id')} AS run, max(start_time) AS t FROM events_full WHERE name='lightgbm.forecast' GROUP BY run) "
    f"WHERE run != '' ORDER BY t DESC",
    "current": {"text": LATEST, "value": LATEST},
    "includeAll": False,
    "multi": False,
}
d = {
    "uid": "sy-per-call",
    "title": "Switchyard forecaster: cost & tokens per model call",
    "refresh": "5s",
    "schemaVersion": 39,
    "time": {"from": "now-24h", "to": "now"},
    "templating": {"list": [var]},
    "panels": panels,
    "tags": ["switchyard"],
}
json.dump(d, open("grafana/dashboards/per-call.json", "w"), indent=1)
print("panels:", len(panels))

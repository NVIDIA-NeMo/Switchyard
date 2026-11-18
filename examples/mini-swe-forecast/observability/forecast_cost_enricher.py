# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn Switchyard forecaster token-band spans into predicted cost in Langfuse, in real time.

Polls Langfuse's ClickHouse for new `lightgbm.forecast` and `agent.call` spans and writes back:
  * a GENERATION span "predicted next call" (model `predicted:opus-4-8`, cost = band midpoint)
    in the same trace -> shows next to the real model in "cost by model" charts
  * numeric scores (timestamped as they arrive):
      pred_next_call_cost_low / _mid / _high, pred_remaining_task_cost, pred_total_task_cost,
      pred_remaining_calls, actual_call_cost, actual_cost_so_far, next_call_in_band
Cache writes are not predicted yet (treated as 0): predicted uncached input is priced at the input rate.
"""

import base64
import hashlib
import json
import os
import time
import urllib.request

CH = os.environ.get("CH_URL", "http://127.0.0.1:8123/")
CH_AUTH = (
    "Basic "
    + base64.b64encode(os.environ.get("CLICKHOUSE_AUTH", "clickhouse:clickhouse").encode()).decode()
)
LF = os.environ.get("LF_URL", "http://localhost:3000")
LF_PK = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-switchyard-demo")
LF_SK = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-switchyard-demo")
LF_AUTH = "Basic " + base64.b64encode(f"{LF_PK}:{LF_SK}".encode()).decode()
PRICE = {
    "input": 5.0,
    "cache_read": 0.5,
    "cache_write": 6.25,
    "output": 25.0,
}  # USD / 1M tokens, Opus 4.8
PRED_MODEL = "predicted:opus-4-8"
POLL_S = 3

seen: set = set()
forecasts: dict = {}  # trace_id -> forecast dict
so_far: dict = {}  # run_id -> {call_index: actual_cost}


def http(url, body=None, headers=None):
    req = urllib.request.Request(url, body, headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def ch(sql):
    out = http(CH, sql.encode(), {"Authorization": CH_AUTH}).decode()
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def score(trace_id, session_id, name, value, comment=None):
    # one score per (trace, name); each model call is its own trace, and the trace carries the session
    body = {
        "id": f"{trace_id}-{name}",
        "traceId": trace_id,
        "name": name,
        "value": float(value),
        "dataType": "NUMERIC",
    }
    if comment:
        body["comment"] = comment
    http(
        f"{LF}/api/public/scores",
        json.dumps(body).encode(),
        {"Authorization": LF_AUTH, "Content-Type": "application/json"},
    )


def attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, (int, float)):
        return {"key": k, "value": {"doubleValue": float(v)}}
    return {"key": k, "value": {"stringValue": str(v)}}


def predicted_generation(f):
    t0 = int(f["start_ns"])
    attrs = [
        attr("langfuse.observation.type", "generation"),
        attr("langfuse.observation.model.name", PRED_MODEL),
        attr("langfuse.observation.usage_details", json.dumps(f["usage_mid"])),
        attr(
            "langfuse.observation.cost_details",
            json.dumps({**f["cost_mid_parts"], "total": f["mid"]}),
        ),
        attr("langfuse.observation.metadata.pred_cost_low", f["low"]),
        attr(
            "langfuse.observation.metadata.pred_cost_high",
            f["high"] if f["high"] is not None else "unbounded",
        ),
        attr("langfuse.observation.metadata.call_index", f["call_index"]),
    ]
    if f["session"]:
        attrs.append(attr("langfuse.session.id", f["session"]))
    # deterministic id -> re-processing (e.g. after a restart) updates the same observation instead of duplicating it
    span_id = hashlib.sha256(("pred-" + f["span_id"]).encode()).hexdigest()[:16]
    span = {
        "traceId": f["trace_id"],
        "spanId": span_id,
        "parentSpanId": f["span_id"],
        "name": "predicted next call",
        "kind": 1,
        "startTimeUnixNano": str(t0),
        "endTimeUnixNano": str(t0 + 1_000_000),
        "attributes": attrs,
    }
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": [attr("service.name", "forecast-cost-enricher")]},
                "scopeSpans": [{"scope": {"name": "forecast-cost-enricher"}, "spans": [span]}],
            }
        ]
    }
    http(
        f"{LF}/api/public/otel/v1/traces",
        json.dumps(payload).encode(),
        {"Authorization": LF_AUTH, "Content-Type": "application/json"},
    )


def num(a, k):
    try:
        return float(a.get(k) or 0)
    except ValueError:
        return 0.0


def parse_forecast(r):
    a = {
        k.replace("attributes.", ""): v
        for k, v in zip(r["metadata_names"], r["metadata_values"], strict=True)
    }
    parts_lo, parts_hi, parts_mid, usage_mid, unbounded = {}, {}, {}, {}, False
    for field, price_key, usage_key in (
        ("uncached_input_tokens", "input", "input"),
        ("cached_input_tokens", "cache_read", "input_cached_tokens"),
        ("output_tokens", "output", "output"),
    ):
        p = f"forecast.next_call.{field}."
        lo = num(a, p + "lower_inclusive")
        hi_unb = a.get(p + "upper_unbounded") == "true"
        hi = None if hi_unb else num(a, p + "upper_exclusive")
        unbounded |= hi_unb
        mid = lo if hi is None else (lo + hi) / 2
        price = PRICE[price_key] / 1e6
        parts_lo[usage_key], parts_mid[usage_key] = lo * price, mid * price
        parts_hi[usage_key] = None if hi is None else hi * price
        usage_mid[usage_key] = int(mid)
    # Cache writes (newer forecaster). Uncached input already includes them (same as actual.uncached_input_tokens),
    # so add only the cache-write premium over the input rate to avoid double counting.
    p = "forecast.next_call.cache_write_input_tokens."
    has_cw = (p + "lower_inclusive") in a
    if has_cw:
        lo = num(a, p + "lower_inclusive")
        hi_unb = a.get(p + "upper_unbounded") == "true"
        hi = None if hi_unb else num(a, p + "upper_exclusive")
        unbounded |= hi_unb
        mid = lo if hi is None else (lo + hi) / 2
        prem = (PRICE["cache_write"] - PRICE["input"]) / 1e6
        parts_lo["input_cache_creation"], parts_mid["input_cache_creation"] = lo * prem, mid * prem
        parts_hi["input_cache_creation"] = None if hi is None else hi * prem
        usage_mid["input_cache_creation"] = int(mid)
    low, mid = sum(parts_lo.values()), sum(parts_mid.values())
    high = None if unbounded else sum(parts_hi.values())
    return {
        "trace_id": r["trace_id"],
        "span_id": r["span_id"],
        "start_ns": r["start_ns"],
        "session": r["session_id"] or a.get("langfuse.session.id") or a.get("demo.run_id", ""),
        "run": a.get("demo.run_id") or r["session_id"],
        "call_index": int(num(a, "demo.call_index")),
        "low": low,
        "mid": mid,
        "high": high,
        "cost_mid_parts": parts_mid,
        "usage_mid": usage_mid,
        "remaining_calls": num(a, "forecast.remaining_task.model_calls.count_estimate"),
        "has_cw": has_cw,
    }


def actual_cost(a):
    unc, cr = num(a, "actual.uncached_input_tokens"), num(a, "actual.cache_read_input_tokens")
    cw, out = num(a, "actual.cache_write_input_tokens"), num(a, "actual.output_tokens")
    true_cost = (
        (unc - cw) * PRICE["input"]
        + cw * PRICE["cache_write"]
        + cr * PRICE["cache_read"]
        + out * PRICE["output"]
    ) / 1e6
    no_cw_cost = (
        unc * PRICE["input"] + cr * PRICE["cache_read"] + out * PRICE["output"]
    ) / 1e6  # comparable to prediction
    return true_cost, no_cw_cost


def tick():
    rows = ch("""
        SELECT trace_id, span_id, name, session_id, toUnixTimestamp64Nano(start_time) AS start_ns, metadata_names, metadata_values
        FROM events_full
        WHERE name IN ('lightgbm.forecast', 'agent.call') AND start_time > now() - INTERVAL 1 DAY
        ORDER BY start_time, name DESC
        FORMAT JSONEachRow""")
    for r in rows:
        key = (r["name"], r["span_id"])
        if key in seen:
            continue
        if r["name"] == "lightgbm.forecast":
            f = parse_forecast(r)
            forecasts[f["trace_id"]] = f
            done = so_far.get(f["run"], {})
            spent = sum(v for ci, v in done.items() if ci < f["call_index"])
            remaining = f["remaining_calls"] * f["mid"]
            predicted_generation(f)
            t = f["trace_id"]
            score(t, f["session"], "pred_next_call_cost_low", f["low"])
            score(t, f["session"], "pred_next_call_cost_mid", f["mid"])
            if f["high"] is not None:
                score(t, f["session"], "pred_next_call_cost_high", f["high"])
            score(t, f["session"], "pred_remaining_calls", f["remaining_calls"])
            score(t, f["session"], "pred_remaining_task_cost", remaining)
            score(t, f["session"], "pred_total_task_cost", spent + remaining)
            print(
                f"forecast run={f['run'][-15:]} call={f['call_index']} next=${f['low']:.4f}-{f['high'] if f['high'] is None else round(f['high'], 4)} remaining=${remaining:.4f}",
                flush=True,
            )
        else:
            a = {
                k.replace("attributes.", ""): v
                for k, v in zip(r["metadata_names"], r["metadata_values"], strict=True)
            }
            if "actual.output_tokens" not in a:
                continue  # span without actuals yet
            run = a.get("demo.run_id") or r["session_id"]
            ci = int(num(a, "demo.call_index"))
            true_cost, no_cw = actual_cost(a)
            so_far.setdefault(run, {})[ci] = true_cost
            t = r["trace_id"]
            score(t, run, "actual_call_cost", true_cost)
            score(t, run, "actual_cost_so_far", sum(so_far[run].values()))
            f = forecasts.get(t)
            if f:
                cmp = true_cost if f["has_cw"] else no_cw  # compare like with like
                hit = f["low"] <= cmp and (f["high"] is None or cmp < f["high"])
                score(t, run, "next_call_in_band", 1 if hit else 0)
            print(f"actual   run={run[-15:]} call={ci} cost=${true_cost:.4f}", flush=True)
        seen.add(key)


if __name__ == "__main__":
    print("forecast-cost-enricher polling", CH, "->", LF, flush=True)
    while True:
        try:
            tick()
        except Exception as e:  # keep polling through transient errors
            print("error:", e, flush=True)
        time.sleep(POLL_S)

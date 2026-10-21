// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { Metrics, estimateCost } from "../dist/index.js";

test("Rust costs price all token classes, prefer provider charges, and retain unknown totals", () => {
  const metrics = new Metrics();
  const context = { session_id: "cost-test", agent_id: "root", agent_turn_idx: 0, user_turn_idx: 0 };
  const rates = { input: 2, output: 10, cache_read: 0.2, cache_write: 3, cache_write_1h: 4 };
  const usage = { input_tokens: 100, output_tokens: 20, cached_input_tokens: 300,
    cache_creation_input_tokens: 50, cache_creation_1h_input_tokens: 10, reasoning_tokens: 10 };
  const answer = { id: "answer", context, model: "provider/model", call_kind: "answer", is_success: true,
    duration_ms: 100, usage, rates };
  const projected = estimateCost({ ...usage, provider_cost: 10 }, rates);
  assert.equal(projected.source, "estimated");
  assert.ok(Math.abs(projected.usd - 0.00062) < 1e-12);
  assert.equal(metrics.costs(context).total_usd, 0);
  metrics.record(answer);
  metrics.record({ ...answer, id: "judge", call_kind: "decision", usage: { ...usage, provider_cost: 0.1 } });
  assert.equal(new Metrics().record(answer), false);
  const priced = metrics.costs(context);
  assert.ok(Math.abs(priced.total_usd - 0.10062) < 1e-12);
  assert.equal(priced.unknown_calls, 0);
  assert.equal(priced.estimated_calls, 1);
  assert.equal(priced.provider_reported_calls, 1);
  const costSeries = metrics.stats(context).series.filter((series) => series.name === "switchyard_llm_call_cost_usd_total");
  assert.ok(Math.abs(costSeries.reduce((sum, series) => sum + series.value.value, 0) - priced.known_usd) < 1e-12);

  const next = { ...context, agent_turn_idx: 1 };
  metrics.record({ ...answer, id: "unpriced", context: next, rates: undefined });
  metrics.record({ ...answer, id: "failed-route", context: next, call_kind: "request", rates: undefined, usage: undefined });
  const subtotal = metrics.costs({ session_id: context.session_id, agent_id: "root", user_turn_idx: 0 });
  assert.equal(subtotal.total_usd, null);
  assert.equal(subtotal.unknown_calls, 1);
  assert.ok(Math.abs(subtotal.known_usd - 0.10062) < 1e-12);
  assert.equal(metrics.costs(next).unknown_calls, 1);
  assert.ok(!metrics.stats(next).series.some((series) => series.name === "switchyard_llm_call_cost_usd_total"));
  metrics.record({ ...answer, id: "free", context: { ...context, agent_id: "child" },
    rates: { input: 0, output: 0, cache_read: 0, cache_write: 0, cache_write_1h: 0 } });
  assert.equal(metrics.costs({ ...context, agent_id: "child" }).total_usd, 0);
  assert.deepEqual(metrics.costs({ session_id: "absent" }), {
    known_usd: 0, unknown_calls: 0, estimated_calls: 0, provider_reported_calls: 0, total_usd: 0,
  });
  assert.throws(() => metrics.record({ ...answer, id: "invalid", rates: { ...rates, input: -1 } }), { code: "ERR_INVALID_REQUEST" });
});

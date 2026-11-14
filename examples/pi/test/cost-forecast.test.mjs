// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { observedSpend, costFrame, costView, formatCostError } from "../cost-forecast.ts";

const frame = (turn, spent, remaining, previous = [], taskStart = 0) => costFrame({
  taskStart, agentTurn: turn, spent: { completion_usd: spent, prediction_api_usd: 0.01 },
  sessionSpent: spent, predictionApiCall: 0.001,
  remaining: { lightgbm: remaining, greg: remaining === null ? null : remaining * 2 }, nextCall: 0.1, previous,
});

test("projected totals freeze C0 and final errors use actual minus the original forecast", () => {
  const first = frame(0, 0, 1);
  const second = frame(1, 0.2, 0.7, [first]);
  const third = frame(2, 0.45, 0.4, [first, second]);
  assert.ok(Math.abs(second.predictors.lightgbm.total_usd - 0.9) < 1e-12);
  assert.equal(third.predictors.lightgbm.initial_total_usd, 1);
  assert.equal(third.predictors.greg.initial_total_usd, 2);
  assert.ok(Math.abs(second.predictors.lightgbm.revision_usd + 0.1) < 1e-12);
  const final = costView(third, "lightgbm", 0.95, true);
  assert.equal(final.remaining_usd, 0);
  assert.equal(final.total_usd, 0.95);
  assert.ok(Math.abs(final.actual_minus_initial_usd + 0.05) < 1e-12);
  const terminalRevision = 0.95 - third.predictors.lightgbm.total_usd;
  const revisions = second.predictors.lightgbm.revision_usd + third.predictors.lightgbm.revision_usd + terminalRevision;
  assert.ok(Math.abs(revisions - final.actual_minus_initial_usd) < 1e-12);
  assert.equal(formatCostError(final), "C0 err −$0.0500");
  assert.equal(third.predictors.lightgbm.remaining_usd, 0.4);
});

test("cold-start and mid-task upgrades do not invent C0; first-priced estimates stay frozen", () => {
  const first = frame(0, 0, null);
  const second = frame(1, 0.2, 0.7, [first]);
  const third = frame(2, 0.45, 0.1, [first, second]);
  assert.equal(third.predictors.lightgbm.initial_total_usd, null);
  assert.equal(third.predictors.lightgbm.first_priced_agent_turn_idx, 1);
  assert.equal(third.predictors.lightgbm.first_priced_total_usd, second.predictors.lightgbm.total_usd);
  const final = costView(third, "lightgbm", 1, true);
  assert.equal(final.actual_minus_initial_usd, null);
  assert.equal(formatCostError(final), "first err +$0.1000");
  assert.equal(frame(7, 0.2, 1, [], 5).predictors.lightgbm.initial_total_usd, null);
});

test("spent dollars consume one frozen budget without summing remaining forecasts", () => {
  const estimate = frame(0, 0, 1);
  assert.equal(costView(estimate, "lightgbm", 0.25, false).remaining_usd, 0.75);
  assert.equal(costView(estimate, "lightgbm", 0.25, false).total_usd, 1);
  assert.equal(costView(estimate, "lightgbm", 1.2, false).total_usd, 1.2);
  assert.equal(costView(estimate, "lightgbm", null, false).remaining_usd, null);
  assert.equal(costView(estimate, "lightgbm", 0.25, false).actual_minus_initial_usd, null);
});

test("actual acting cost uses saved per-call prices, excludes predictor fees, and deduplicates branches", () => {
  const call = (id, turn, kind, usage, rates) => ({ type: "custom", customType: "switchyard.metrics.v1", data: {
    id, context: { session_id: "s", agent_turn_idx: turn }, call_kind: kind, usage, rates,
  } });
  const rates = { input: 2, output: 10, cache_read: 0.1, cache_write: 2.5, cache_write_1h: 4 };
  const answer = call("a", 0, "answer", { input_tokens: 100, output_tokens: 20,
    cached_input_tokens: 1000, cache_creation_input_tokens: 100, reasoning_tokens: 5 }, rates);
  const decision = call("d", 0, "decision", { input_tokens: 100, output_tokens: 10 }, rates);
  const retry = call("b", 1, "answer", { input_tokens: 100, output_tokens: 10 }, rates);
  const foreign = structuredClone(answer); foreign.data.context.session_id = "parent";
  const entries = [answer, answer, decision, retry, foreign];
  assert.ok(Math.abs(observedSpend(entries, "s").completion_usd - 0.00105) < 1e-12);
  assert.ok(Math.abs(observedSpend(entries, "s").prediction_api_usd - 0.0003) < 1e-12);
  assert.ok(Math.abs(observedSpend(entries, "s", 1).completion_usd - 0.0003) < 1e-12);
  assert.deepEqual(observedSpend(entries, "new"), { completion_usd: 0, prediction_api_usd: 0 });
  entries.push(call("unknown", 2, "answer", undefined, rates));
  assert.equal(observedSpend(entries, "s").completion_usd, null);
  assert.equal(observedSpend(entries, "s", 3).completion_usd, 0);
});

test("reloading cost frames preserves baseline errors and a new task starts a new C0", () => {
  const first = frame(3, 0, 0.5, [], 3);
  const restored = JSON.parse(JSON.stringify(first));
  assert.equal(costView(restored, "lightgbm", 0.75, true).actual_minus_initial_usd, 0.25);
  const next = frame(9, 0, 2, [], 9);
  assert.equal(next.predictors.lightgbm.initial_total_usd, 2);
});

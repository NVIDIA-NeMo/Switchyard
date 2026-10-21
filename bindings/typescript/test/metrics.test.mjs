// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { CompositeRouter, Metrics } from "../dist/index.js";

const context = { session_id: "metrics-test", agent_id: "root", agent_turn_idx: 2, user_turn_idx: 1 };
const usage = { input_tokens: 10, output_tokens: 4, cached_input_tokens: 3,
  cache_creation_input_tokens: 2, total_tokens: 19, reasoning_tokens: 1 };
const counter = (stats, name) => stats.series.filter((series) => series.name === name)
  .reduce((sum, series) => sum + series.value.value, 0);

// The collector is process-wide, as are the meter provider and replay IDs.
test("hosted metrics retain context, usage, timings and replay identity", () => {
  const metrics = new Metrics();
  const observation = { id: "answer-1", context, model: "provider/strong", call_kind: "answer",
    is_success: true, duration_ms: 120, ttft_ms: 30, usage };
  assert.equal(metrics.record(observation), true);
  assert.equal(metrics.record(observation), false);
  assert.equal(new Metrics().record(observation), false);
  metrics.record({ ...observation, id: "decision-1", call_kind: "decision", duration_ms: 20, ttft_ms: undefined });
  metrics.record({ ...observation, id: "answer-2", context: { ...context, agent_turn_idx: 3 }, duration_ms: 200, ttft_ms: undefined });
  const stats = metrics.stats(context);
  assert.equal(counter(stats, "switchyard_llm_call_prompt_tokens_total"), 30);
  assert.equal(counter(stats, "switchyard_llm_call_completion_tokens_total"), 8);
  assert.equal(counter(stats, "switchyard_llm_call_reasoning_tokens_total"), 2);
  const duration = stats.series.find((series) => series.name === "switchyard_inference_duration_ms").value;
  assert.equal(duration.type, "histogram");
  assert.equal(duration.count, 1);
  assert.equal(duration.sum, 120);
  assert.equal(stats.series.find((series) => series.name === "switchyard_inference_ttft_ms").value.sum, 30);
  assert.equal(counter(metrics.stats({ session_id: context.session_id }), "switchyard_inference_requests_total"), 2);
  assert.match(metrics.prometheus(context), /agent_turn_idx="2"/);
  assert.deepEqual(metrics.stats({ session_id: "absent" }), { series: [] });
  for (const change of [
    { duration_ms: -1 }, { ttft_ms: 121 }, { context: { ...context, agent_turn_idx: -1 } },
    { context: { ...context, session_id: "" } }, { usage: { input_tokens: -1 } },
  ]) {
    assert.throws(() => metrics.record({ ...observation, id: "invalid", ...change }), { code: "ERR_INVALID_REQUEST" });
  }
});

test("routing metrics use host indices without changing branch state", async () => {
  const metrics = new Metrics();
  const router = CompositeRouter.create({
    targets: { capable: "provider/strong", efficient: "provider/weak", classifier: "provider/judge" },
    classifier: { classify_trigger: "user_turn", fail_open: false,
      decision: { cutoff: 0.4, candidates: { a: "capable", b: "efficient" }, evidence: {} } },
    stage: { efficient_first_threshold: 0.5, capable_first_threshold: 0.2 },
  });
  const metricContext = { ...context, session_id: "routing-metrics-test", agent_turn_idx: 7 };
  const messages = [{ role: "user", content: [{ type: "text", text: "Fix a bug" }] }];
  const decide = async () => ({ answers: { route: { value: { type: "choice", data: {
    selected: "advantage", probabilities: { advantage: 0.8, no_advantage: 0.2 },
  } } } }, usage });
  const without = await router.route(messages, { decide });
  const withMetrics = await router.route(messages, { decide, metricContext });
  assert.deepEqual(withMetrics, without);
  const stats = metrics.stats(metricContext);
  assert.equal(counter(stats, "switchyard_runs_total"), 1);
  assert.equal(counter(stats, "switchyard_decision_calls_total"), 1);
  assert.ok(stats.series.some((series) => series.name === "switchyard_run_duration_ms"));
});

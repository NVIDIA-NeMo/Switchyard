// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { CompletionTiming, completionDuration } from "../completion-time.ts";

const call = (changes = {}) => ({
  id: "answer", context: { session_id: "session", agent_id: "root", agent_turn_idx: 0, user_turn_idx: 0 },
  model: "provider/answer", call_kind: "answer", is_success: true, duration_ms: 6000, ttft_ms: 2000,
  usage: { input_tokens: 700, cache_creation_input_tokens: 100, cached_input_tokens: 10000,
    output_tokens: 200, reasoning_tokens: 50 }, ...changes,
});

test("splits the combined budget before applying input latency and output throughput", () => {
  const timing = new CompletionTiming();
  timing.record(call());
  // 1,500 remaining => 1,200 input at 400 tok/s + 300 output at 50 tok/s.
  assert.equal(timing.estimateMs(1500), 9000);
  timing.record(call({ id: "second", ttft_ms: 500, duration_ms: 16500,
    usage: { input_tokens: 200, output_tokens: 800 } }));
  // Matched session totals: 1,000 input / 2.5s TTFT and 1,000 output / 20s decoding.
  assert.equal(timing.estimateMs(1000), 11250);
});

test("requires usable paired measurements and excludes classifier and failed calls", () => {
  const timing = new CompletionTiming();
  assert.equal(timing.estimateMs(1500), undefined);
  assert.equal(timing.estimateMs(0), 0);
  for (const change of [
    { call_kind: "decision" }, { call_kind: "request" }, { is_success: false },
    { usage: undefined }, { ttft_ms: undefined }, { duration_ms: 2000 },
    { usage: { input_tokens: 100 } }, { usage: { output_tokens: 100 } },
    { usage: { input_tokens: 100, output_tokens: 0 } },
  ]) timing.record(call(change));
  assert.equal(timing.estimateMs(1500), undefined);
  timing.record(call());
  assert.equal(timing.estimateMs(1500), 9000);
});

test("fully cached calls still include TTFT and zero TTFT is usable", () => {
  const cached = new CompletionTiming();
  cached.record(call({ duration_ms: 4000, usage: { input_tokens: 0, cached_input_tokens: 10000, output_tokens: 100 } }));
  assert.equal(cached.estimateMs(1000), 40000);
  const instant = new CompletionTiming();
  instant.record(call({ ttft_ms: 0, duration_ms: 2000, usage: { input_tokens: 900, output_tokens: 100 } }));
  assert.equal(instant.estimateMs(1000), 2000);
});

test("formats unavailable, seconds, minutes and hours", () => {
  for (const [value, expected] of [[undefined, "—"], [0, "~0s"], [9000, "~9s"],
    [60000, "~1m"], [60100, "~1m1s"], [3600000, "~1h"], [3660000, "~1h1m"]]) {
    assert.equal(completionDuration(value), expected);
  }
});

test("adds tool wall time per agent response, including responses without tools in the average", () => {
  const timing = new CompletionTiming();
  const first = call();
  timing.record(first);
  timing.recordTools(first.context, 8000);
  // 1.5 future responses: 9s model time + 1.5 * 8s tools.
  assert.equal(timing.estimateMs(1500), 21000);
  timing.record(call({ context: { ...first.context, agent_turn_idx: 1 } }));
  // The second response has no tools: average tools per response is now 4s.
  assert.equal(timing.estimateMs(1500), 15000);
  const failed = call({ context: { ...first.context, agent_turn_idx: 2 }, is_success: false });
  timing.record(failed);
  timing.recordTools(failed.context, 1000000);
  timing.recordTools({ ...first.context, agent_id: "other" }, 1000000);
  timing.recordTools({ ...first.context, agent_turn_idx: 99 }, 1000000);
  assert.equal(timing.estimateMs(1500), 15000);
});

test("a large uncached-input response does not collapse ETA by diluting per-request delay", () => {
  const timing = new CompletionTiming();
  const first = call({ duration_ms: 2582, ttft_ms: 2253,
    usage: { input_tokens: 149, output_tokens: 39 } });
  timing.record(first);
  timing.recordTools(first.context, 23);
  assert.equal(completionDuration(timing.estimateMs(22265)), "~5m9s");
  timing.record(call({ context: { ...first.context, agent_turn_idx: 1 }, duration_ms: 2355, ttft_ms: 2291,
    usage: { input_tokens: 13513, output_tokens: 112 } }));
  // The remaining budget also fell, but the same session still has seconds of TTFT per call.
  const estimate = timing.estimateMs(12915);
  const share = ((39 / 188) + (112 / 13625)) / 2;
  const output = 12915 * share;
  const calls = output / (151 / 2);
  const expected = calls * ((4544 + 23) / 2) + output / (151 / 393);
  assert.ok(Math.abs(estimate - expected) < 1e-8);
  assert.ok(estimate > 40000 && estimate < 50000);
});

test("projected monetary usage splits the uncached budget and estimates cache reads separately", () => {
  const timing = new CompletionTiming();
  assert.equal(timing.remainingUsage(1000), undefined);
  assert.equal(timing.remainingUsage(0).output_tokens, 0);
  timing.record(call({ usage: { input_tokens: 700, cache_creation_input_tokens: 100,
    cache_creation_1h_input_tokens: 20, cached_input_tokens: 10000, output_tokens: 200 } }));
  assert.deepEqual(timing.remainingUsage(1500), {
    input_tokens: 1050, output_tokens: 300, cached_input_tokens: 15000,
    cache_creation_input_tokens: 150, cache_creation_1h_input_tokens: 30,
  });
  assert.equal(timing.meanPromptTokens, 10800);
});

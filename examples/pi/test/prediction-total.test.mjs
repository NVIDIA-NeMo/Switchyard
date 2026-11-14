// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { totalTokenPredictionRequest } from "../prediction-total.ts";

test("Jev's full-token prompt includes cache usage, model and selected reasoning without rewriting task text", () => {
  const messages = [
    { role: "user", content: "Explain uncached input plus output" },
    { role: "assistant", content: [{ type: "text", text: "Inspecting" }],
      usage: { input: 800, cacheRead: 5000, cacheWrite: 200, output: 100, reasoning: 40 } },
  ];
  for (const level of ["medium", "high", "max"]) {
    const request = totalTokenPredictionRequest("provider/judge", messages, "provider/worker", level);
    assert.equal(request.context.token_usage.tokens_used_so_far, 6100);
    assert.equal(request.context.token_usage.per_step[0].input_tokens, 6000);
    assert.equal(request.context.token_usage.per_step[0].output_tokens, 100);
    assert.equal(request.context.token_usage.per_step[0].tokens_used, 6100);
    assert.equal(request.context.completion_model, "provider/worker");
    assert.equal(request.context.thinking_level, level);
    assert.match(request.context.instructions, /including cache reads and writes/);
    assert.match(request.context.instructions, /latest user request/);
    assert.equal(request.context.task_history[0].message, messages[0].content);
    assert.doesNotMatch(request.context.instructions, /uncached input plus output/);
    assert.doesNotMatch(JSON.stringify(request.questions), /uncached/);
    assert.equal(request.questions.minimum_likely_budget.kind.data.options.length, 81);
  }
});

test("missing or invalid cache usage cannot silently change Jev's target", () => {
  for (const usage of [undefined, { input: 100, output: 10, cacheWrite: 0 },
    { input: 100, output: 10, cacheRead: -1, cacheWrite: 0 }]) {
    assert.throws(() => totalTokenPredictionRequest("provider/judge", [{ role: "assistant", content: [], usage }],
      "provider/worker", "max"), /usage is missing/);
  }
});

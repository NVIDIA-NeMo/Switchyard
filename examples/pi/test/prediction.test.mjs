// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { predictionRequest } from "../prediction.ts";

test("prediction history retains task and tool evidence with uncached-input-plus-output accounting", () => {
  const messages = [
    { role: "user", content: "Fix the parser" },
    { role: "assistant", content: [{ type: "thinking", thinking: "check parser" },
      { type: "toolCall", name: "bash", arguments: { command: "pytest" } }],
      usage: { input: 800, output: 100, cacheRead: 5000, cacheWrite: 200, reasoning: 40 } },
    { role: "toolResult", toolName: "bash", isError: true, content: [{ type: "text", text: "test failed" }] },
  ];
  const request = predictionRequest("provider/judge", messages);
  assert.equal(request.context.token_usage.tokens_used_so_far, 1100);
  assert.equal(request.context.token_usage.reasoning_output_tokens_so_far, 40);
  assert.match(request.context.task_history[1].message, /pytest/);
  assert.match(request.context.task_history[1].message, /check parser/);
  assert.equal(request.context.task_history[2].is_error, true);
  assert.equal(request.context.task_history[0].message, "Fix the parser");
  assert.equal(request.questions.minimum_likely_budget.kind.data.options.length, 81);
});

test("history is bounded while preserving initial task, recent content and full branch usage", () => {
  const usageMessages = [{ role: "assistant", usage: { input: 80, output: 20, cacheRead: 999, cacheWrite: 0 } }];
  const messages = [{ role: "user", content: "Initial task" }, ...Array.from({ length: 100 }, (_, i) => ({
    role: "assistant", content: [{ type: "text", text: `${i}: ${"x".repeat(10000)}` }, { type: "image", data: "large-image" }],
  }))];
  const { context } = predictionRequest("provider/judge", messages, usageMessages);
  assert.equal(context.task_history[0].message, "Initial task");
  assert.match(context.task_history.at(-1).message, /^99:/);
  assert.ok(JSON.stringify(context.task_history).length < 61000);
  assert.ok(context.task_history.length < messages.length);
  assert.ok(!JSON.stringify(context).includes("large-image"));
  assert.equal(context.token_usage.tokens_used_so_far, 100);
});

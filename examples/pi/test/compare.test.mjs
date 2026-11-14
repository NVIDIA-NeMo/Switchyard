// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import extension from "../switchyard-compare.ts";

test("both forecasts share a cutoff and completion; rendering survives classifier failure and resume", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "pi-compare-"));
  const previous = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = directory;
  t.after(() => {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previous;
    rmSync(directory, { recursive: true, force: true });
  });
  writeFileSync(join(directory, "switchyard-forecast.json"), JSON.stringify({
    decision_model: "openrouter/~typesafe/jev-latest", completion_model: "openrouter/openai/gpt-6.1-sol",
    worker_command: [process.execPath, fileURLToPath(new URL("./fixtures/forecast-worker.mjs", import.meta.url))],
  }));
  const handlers = new Map(), entries = [], statuses = [], widgets = [], contexts = [];
  // Earlier uncached forecasts and scores must not appear in the full-token comparison.
  entries.push({ type: "custom", customType: "switchyard.compare.forecast.v1",
    data: { session_id: "comparison", greg: { expected_upper_tokens: 999999 } } });
  entries.push({ type: "custom", customType: "switchyard.compare.remaining-errors.v1",
    data: { session_id: "comparison", samples: [{ predictor: "greg", sample_id: "old", session_id: "comparison",
      predicted_tokens: 999999, actual_tokens: 1, absolute_error: 999998 }] } });
  let definition, fail = false;
  const model = { provider: "openrouter", id: "openai/gpt-6.1-sol",
    cost: { input: 1, output: 4, cacheRead: 0.1, cacheWrite: 1 } };
  const ctx = {
    hasUI: true, cwd: directory,
    ui: { setStatus: (_key, value) => statuses.push(value), setWidget: (_key, lines) => widgets.push(lines) },
    sessionManager: { getSessionId: () => "comparison", getEntries: () => entries, getBranch: () => entries },
    modelRegistry: {
      find: () => model, findOfType: () => ({ provider: "openrouter", id: "~typesafe/jev-latest" }),
      classify: async (_judge, context) => {
        contexts.push(context);
        if (fail) throw new Error("offline");
        return { provider: "openrouter", model: "~typesafe/jev-latest", stopReason: "stop", answers: {
          minimum_likely_budget: { type: "choice", choice: "b010", probabilities:
            Object.fromEntries(Object.keys(context.questions.minimum_likely_budget.criteria).map(id => [id, id === "b010" ? 1 : 0])) },
        } };
      },
    },
  };
  const emit = (name, event = {}) => { for (const handler of handlers.get(name) ?? []) handler(event, ctx); };
  extension({ registerVirtualModel: value => { definition = value; },
    on: (name, handler) => handlers.set(name, [...handlers.get(name) ?? [], handler]),
    appendEntry: (customType, data) => entries.push({ type: "custom", customType, data }) });
  t.after(() => emit("session_shutdown"));
  const user = { role: "user", content: "Read the parser and explain it" };
  entries.push({ type: "message", message: user });
  const request = { reason: "user", thinkingLevel: "max", messages: [user] };
  assert.deepEqual(definition.thinkingLevels, ["medium", "high", "max"]);
  emit("session_start");
  assert.match(widgets.at(-1)[1], /^Jev · waiting/);
  const firstRoute = await definition.route(request, ctx);
  assert.equal(firstRoute.model, model);
  assert.equal(firstRoute.thinkingLevel, "max");
  const pair = entries.find(e => e.customType === "switchyard.compare.forecast.v2").data;
  assert.ok(pair.lightgbm.remaining_task.tokens > 0);
  assert.equal(pair.greg.predicted_tokens, 46000);
  assert.equal(pair.token_basis, "full_input_plus_output");
  assert.equal(pair.thinking_level, "max");
  assert.equal(pair.full_used, 0);
  assert.equal(contexts[0].state.task_history[0].message, user.content);
  assert.equal(contexts[0].state.thinking_level, "max");
  assert.equal(widgets.at(-1).length, 2);
  assert.match(widgets.at(-1)[0], /^LightGBM · \$— · ETA — · next .* \| remaining .*\(avg err/);
  assert.match(widgets.at(-1)[1], /^Jev · \$— · ETA — · remaining .*\(avg err/);
  assert.ok(statuses.every(value => value === undefined));
  emit("message_update", { message: { role: "assistant" }, assistantMessageEvent: { type: "text_delta", delta: "a" } });
  await new Promise(resolve => setTimeout(resolve, 2));
  const response = { role: "assistant", provider: model.provider, model: model.id, content: [{ type: "text", text: "Done" }],
    stopReason: "stop", usage: { input: 100, output: 10, cacheRead: 1000, cacheWrite: 0, totalTokens: 1110 } };
  entries.push({ type: "message", message: response });
  emit("message_end", { message: response });
  assert.match(widgets.at(-1)[0], /^LightGBM · ~\$[0-9.]+ · ETA ~[^·]+ · next/);
  assert.match(widgets.at(-1)[1], /^Jev · ~\$[0-9.]+ · ETA ~[^·]+ · remaining/);
  assert.match(widgets.at(-1)[0], /next .*\(avg err [+-][\d,]+\.\d%\) \| remaining/);
  assert.match(widgets.at(-1)[1], /avg err —\)/);
  emit("agent_settled");
  const settlements = () => entries.filter(e => e.customType === "switchyard.compare.remaining-errors.v2");
  assert.equal(settlements().length, 1);
  assert.equal(settlements()[0].data.samples.length, 2);
  assert.equal(settlements()[0].data.samples.find(s => s.predictor === "greg").actual_tokens, 1110);
  assert.equal(settlements()[0].data.samples.find(s => s.predictor === "lightgbm").actual_tokens, 1110);
  for (const line of widgets.at(-1)) {
    assert.match(line, / · \$0\.0000 · ETA 0s · /);
    assert.match(line, /remaining 0 tok \(avg err /);
  }
  assert.match(widgets.at(-1)[0], /next — \(avg err .*\) \| remaining 0 tok/);
  assert.equal(pair.lightgbm.remaining_task.tokens, 6000);
  assert.equal(pair.greg.predicted_tokens, 46000);
  assert.match(widgets.at(-1)[0], /remaining .*avg err \+[\d,]+\.\d%\)/);
  assert.match(widgets.at(-1)[1], /remaining .*avg err \+[\d,]+\.\d%\)/);
  emit("agent_settled");
  assert.equal(settlements().length, 1);
  const before = widgets.at(-1);
  emit("session_start");
  assert.deepEqual(widgets.at(-1), before);
  fail = true;
  const secondRoute = await definition.route({ ...request, thinkingLevel: "medium", reason: "continuation" }, ctx);
  assert.equal(secondRoute.model, model);
  assert.equal(secondRoute.thinkingLevel, "medium");
  assert.equal(contexts.at(-1).state.thinking_level, "medium");
  assert.equal(contexts.at(-1).state.token_usage.tokens_used_so_far, 1110);
  assert.match(widgets.at(-1)[0], /ETA ~.* \| remaining ~/);
  assert.match(widgets.at(-1)[0], /^LightGBM · .* · ETA .* · next .* \| remaining/);
  assert.match(widgets.at(-1)[1], /^Jev · \$— · ETA — · remaining unavailable/);
  const number = contexts.length;
  assert.equal((await definition.route({ ...request, reason: "direct" }, ctx)).model, model);
  assert.equal(contexts.length, number);
  // Cancelling before a new response must not reuse the previous turn's final response.
  emit("agent_settled");
  assert.equal(settlements().length, 2);
  assert.equal(settlements()[1].data.completed, false);
  assert.deepEqual(settlements()[1].data.samples, []);
  assert.doesNotMatch(widgets.at(-1)[0], /ETA 0s|remaining 0 tok/);
  assert.match(widgets.at(-1)[1], /avg err \+[\d,]+\.\d%\)/);
  assert.ok(statuses.every(value => value === undefined));
});

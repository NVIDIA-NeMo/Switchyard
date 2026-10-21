// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { test } from "node:test";
import { CompositeRouter, Metrics } from "@switchyard/libsy";
import extension from "../switchyard-router.ts";

const config = {
  decision_model: "openrouter/inception/mercury-decide:free",
  capable_model: "openai-codex/gpt-6.1-sol",
  efficient_model: "openai-codex/gpt-6-luna",
};
const routingCalls = (registry) => registry.classify.mock.calls.filter((call) => call.arguments[1].questions.route);
const request = (fields = {}) => ({
  reason: "user", messages: [{ role: "user", content: "task" }],
  thinkingLevel: "high", signal: new AbortController().signal, ...fields,
});
const result = (advantage = 0.6) => ({
  provider: "openrouter", model: "inception/mercury-decide:free", stopReason: "stop",
  answers: { route: {
    type: "choice", choice: advantage > 0.5 ? "advantage" : "no_advantage", probabilities: { advantage, no_advantage: 1 - advantage }, confidence: 0.9,
  } },
});

function setup(t, answer = result()) {
  const dir = mkdtempSync(join(tmpdir(), "switchyard-router-"));
  const previous = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  t.after(() => {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previous;
    rmSync(dir, { recursive: true, force: true });
  });
  const path = join(dir, "switchyard-router.json");
  writeFileSync(path, JSON.stringify(config));
  const registry = {
    find: t.mock.fn((provider, id) => ({ provider, id })),
    findOfType: t.mock.fn((type, provider, id) => ({ type, provider, id })),
    classify: t.mock.fn(async () => answer),
  };
  let definition;
  const handlers = new Map();
  const entries = [];
  const sessionId = randomUUID();
  const ctx = {
    modelRegistry: registry, hasUI: true,
    sessionManager: { getSessionId: () => sessionId, getEntries: () => entries, getBranch: () => entries },
    ui: { setStatus: t.mock.fn() },
  };
  extension({
    registerVirtualModel: (value) => { definition = value; },
    on: (name, handler) => { handlers.set(name, handler); },
    appendEntry: (customType, data) => entries.push({ type: "custom", customType, data }),
  });
  return { definition, registry, path, ctx, entries,
    emit: (event) => handlers.get(event.type)?.(event, ctx),
    route: (req) => definition.route(req, ctx) };

}

function tool(id, name, args, text, isError = false) {
  return [
    { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }] },
    { role: "toolResult", toolCallId: id, toolName: name, content: [{ type: "text", text }], isError },
  ];
}

test("Pi classifies the libsy question and retains the capable tier on a read continuation", async (t) => {
  const { route, definition, registry } = setup(t);
  assert.deepEqual(definition.thinkingLevels, ["medium"]);
  const req = request();
  const initial = await route(req);
  assert.deepEqual(initial.model, { provider: "openai-codex", id: "gpt-6.1-sol" });
  assert.equal(initial.thinkingLevel, "medium");
  assert.ok(initial.state);
  assert.equal(registry.findOfType.mock.callCount(), 1);
  assert.deepEqual(registry.findOfType.mock.calls[0].arguments, ["classifier", "openrouter", "inception/mercury-decide:free"]);
  assert.equal(routingCalls(registry).length, 1);
  const [judge, context, options] = registry.classify.mock.calls[0].arguments;
  assert.equal(judge, registry.findOfType.mock.calls[0].result);
  assert.deepEqual(options, { signal: req.signal });
  assert.deepEqual(context.state.comparison, { capable: "a", efficient: "b" });
  assert.equal(context.state.evidence.description, "Candidate a is the larger reasoning-oriented coding and thinking model. It is intended for complex problems. Candidate b is the smaller fast coding model. It is intended for simple questions, simple local reads and edits, routine tests and straightforward fixes. These are qualitative deployment roles, not empirical success rates.");
  assert.deepEqual(context.state.task, [{ role: "user", content: [{ type: "text", text: "task" }] }]);
  assert.equal(context.questions.route.type, "choice");
  assert.equal(typeof context.questions.route.instructions, "string");
  assert.deepEqual(JSON.parse(context.questions.route.instructions), JSON.parse(readFileSync(new URL(
    "../../../crates/libsy/src/prompts/capability-classifier/relative_advantage.json", import.meta.url,
  ), "utf8")));
  assert.deepEqual(Object.keys(context.questions.route.criteria), ["advantage", "no_advantage"]);
  const continued = await route(request({
    reason: "continuation", state: initial.state,
    messages: [...req.messages, ...tool("read", "read", { path: "parser.rs" }, "fn parse() {}")],
  }));
  assert.deepEqual(continued, { model: initial.model, thinkingLevel: "medium", state: undefined });
  assert.equal(registry.findOfType.mock.callCount(), 1);
  assert.equal(routingCalls(registry).length, 1);
});

test("cutoff equality selects efficient, then a tool failure holds capable for two requests", async (t) => {
  const { route, registry } = setup(t, result(0.4));
  const messages = request().messages;
  let selected = await route(request({ messages }));
  assert.equal(selected.model.id, "gpt-6-luna");
  messages.push(...tool("failed", "bash", { command: "pytest" }, "out of memory", true));
  selected = await route(request({ reason: "continuation", messages, state: selected.state }));
  assert.equal(selected.model.id, "gpt-6.1-sol");
  assert.deepEqual(selected.state.capable_hold_turns, { count: 2 });
  for (let i = 0; i < 3; i++) messages.push(...tool(`write-${i}`, "write", { path: `file-${i}` }, "wrote file"));
  let state = selected.state;
  for (const expected of ["gpt-6.1-sol", "gpt-6.1-sol", "gpt-6-luna"]) {
    selected = await route(request({ reason: "continuation", messages, state }));
    state = selected.state ?? state;
    assert.equal(selected.model.id, expected);
  }
  assert.equal(routingCalls(registry).length, 1);
});

test("classifier lookup, answer validation, failures, and cancellation", async (t) => {
  const { route, registry } = setup(t);
  for (const [answer, expected] of [
    [{ stopReason: "error", errorMessage: "classifier unavailable" }, /classifier unavailable/],
    [{ stopReason: "aborted" }, { name: "AbortError", code: "ABORT_ERR" }],
    ...[undefined, { type: "bool", probability: 0.6 }, { type: "choice", choice: "other" }].map((answer) => [
      { ...result(), answers: { route: answer } }, /Expected a declared choice for question: route/,
    ]),
  ]) {
    registry.classify.mock.mockImplementation(async () => answer);
    await assert.rejects(route(request()), expected);
  }
  const calls = routingCalls(registry).length;
  await assert.rejects(route(request({ signal: AbortSignal.abort() })), { name: "AbortError", code: "ABORT_ERR" });
  assert.equal(routingCalls(registry).length, calls);
  assert.equal(registry.findOfType.mock.callCount(), 1);
});

test("a missing classifier fails until it appears, then the lookup is cached", async (t) => {
  const { route, registry } = setup(t);
  registry.findOfType.mock.mockImplementationOnce(() => undefined);
  await assert.rejects(route(request()), { message: `Unknown classifier: ${config.decision_model}` });
  assert.equal(routingCalls(registry).length, 0);
  await route(request());
  await route(request());
  assert.equal(registry.findOfType.mock.callCount(), 2);
  assert.equal(routingCalls(registry).length, 2);
});

test("config loads lazily, names missing fields, and is cached until reload", async (t) => {
  const { route, path } = setup(t);
  rmSync(path);
  const check = (error) => {
    for (const value of [path, ...Object.keys(config)]) assert.ok(error.message.includes(value));
    return true;
  };
  await assert.rejects(route(request()), check);
  for (const field of Object.keys(config)) {
    const incomplete = { ...config };
    delete incomplete[field];
    writeFileSync(path, JSON.stringify(incomplete));
    await assert.rejects(route(request()), check);
  }
  writeFileSync(path, JSON.stringify({ ...config, efficient_model: config.capable_model }));
  await assert.rejects(route(request()), { code: "ERR_CONFIG", message: /^capable and efficient targets must differ/ });
  writeFileSync(path, JSON.stringify(config));
  const initial = await route(request());
  rmSync(path);
  assert.deepEqual(await route(request()), initial);
});

test("message and classifier adapters preserve content and model identities", async (t) => {
  const { route, registry } = setup(t);
  const nativeRoute = t.mock.fn(async (_, { decide }) => {
    const response = await decide({ context: { task: "task" }, questions: { route: {
      instructions: "Choose a model",
      kind: { type: "choice", data: { options: [
        { id: "advantage", description: "Complex work" }, { id: "no_advantage", description: { task: "Simple work" } }, { id: "unspecified" },
      ] } },
    } } });
    assert.deepEqual(response, { model: config.decision_model, answers: { route: {
      value: { type: "choice", data: { selected: "advantage", probabilities: result().answers.route.probabilities } },
      provider_confidence: 0.9,
    } } });
    await assert.rejects(decide({ context: {}, questions: { route: { kind: { type: "score" } } } }), /Expected a choice question/);
    return { model: "openrouter/vendor/model" };
  });
  t.mock.method(CompositeRouter, "create", () => ({ route: nativeRoute }));
  const req = request({ messages: [
    { role: "system", content: "rules", sections: { extra: "more rules" } },
    { role: "user", content: [{ type: "text", text: "task" }, { type: "image", data: "image" }] },
    ...tool("call", "bash", { command: "pytest" }, "out of memory", true),
    { role: "assistant", content: [{ type: "thinking", thinking: "private" }] },
  ] });
  assert.deepEqual((await route(req)).model, { provider: "openrouter", id: "vendor/model" });
  const [messages, options] = nativeRoute.mock.calls[0].arguments;
  assert.deepEqual(messages, [
    { role: "system", content: [{ type: "text", text: "rules" }, { type: "text", text: "more rules" }] },
    { role: "user", content: [{ type: "text", text: "task" }] },
    { role: "assistant", content: [{ type: "tool_call", id: "call", name: "bash", arguments: { command: "pytest" } }] },
    { role: "tool", content: [{ type: "tool_result", tool_call_id: "call", content: [{ type: "text", text: "out of memory" }], is_error: true }] },
    { role: "assistant", content: [] },
  ]);
  assert.equal(options.signal, req.signal);
  assert.equal(options.state, req.state);
  assert.deepEqual(registry.classify.mock.calls[0].arguments[1], {
    state: { task: "task" }, questions: { route: {
      type: "choice", instructions: "Choose a model", criteria: { advantage: "Complex work", no_advantage: '{"task":"Simple work"}', unspecified: "null" },
    } },
  });
});

test("the virtual model reports routing and answer metrics in the same native context", async (t) => {
  let now = 0;
  t.mock.method(performance, "now", () => now);
  const usage = { input: 10, output: 2, cacheRead: 3, cacheWrite: 0, totalTokens: 15,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const { route, ctx, emit, entries } = setup(t, { ...result(), usage });
  const selection = await route(request());
  const identity = entries[0].data.context;
  now = 20;
  emit({ type: "message_update", message: { role: "assistant" },
    assistantMessageEvent: { type: "text_delta", delta: "hello" } });
  now = 60;
  emit({ type: "message_end", message: { role: "assistant", ...selection.model,
    model: selection.model.id, usage, stopReason: "stop" } });
  assert.equal(ctx.ui.setStatus.mock.calls.at(-1).arguments[1],
    "SY remaining unavailable | session 45/20ms/60ms | last 45/20ms/60ms");
  const snapshot = new Metrics().stats(identity);
  const count = (name) => snapshot.series.filter((series) => series.name === name)
    .reduce((sum, series) => sum + series.value.value, 0);
  assert.equal(count("switchyard_llm_call_total_tokens_total"), 45);
  assert.equal(count("switchyard_runs_total"), 1);
  assert.equal(count("switchyard_decision_calls_total"), 1);
  assert.equal(count("switchyard_inference_requests_total"), 1);
});

function forecast(choice = "b002") {
  const probabilities = Object.fromEntries(Array.from({ length: 80 }, (_, i) => [`b${String(i + 1).padStart(3, "0")}`, 0]));
  probabilities.impossible = 0;
  probabilities[choice] = 1;
  return { ...result(), answers: { minimum_likely_budget: { type: "choice", choice, probabilities } } };
}

test("task prediction appears before completion and refreshes for continuations using branch usage", async (t) => {
  const { route, registry, ctx, emit, entries } = setup(t);
  const weighted = forecast();
  weighted.answers.minimum_likely_budget.probabilities.b001 = 0.5;
  weighted.answers.minimum_likely_budget.probabilities.b002 = 0.5;
  registry.classify.mock.mockImplementation(async (_, context) => context.questions.route ? result() : weighted);
  const initial = await route(request());
  assert.equal(registry.classify.mock.callCount(), 2);
  const [judge, prediction, options] = registry.classify.mock.calls[1].arguments;
  assert.equal(judge, registry.findOfType.mock.calls[0].result);
  assert.equal(Object.keys(prediction.questions.minimum_likely_budget.criteria).length, 81);
  assert.equal(prediction.state.task_history[0].message, "task");
  assert.equal(prediction.state.token_usage.tokens_used_so_far, 0);
  assert.ok(options.signal instanceof AbortSignal);
  assert.match(ctx.ui.setStatus.mock.calls.at(-1).arguments[1], /^SY remaining ~3\.0k tok ETA [^|]+ \|/);
  const assistant = { role: "assistant", provider: "openai-codex", model: initial.model.id,
    content: [{ type: "text", text: "checking parser" }], stopReason: "toolUse",
    usage: { input: 800, cacheWrite: 200, cacheRead: 5000, output: 100, reasoning: 40, totalTokens: 6100 } };
  emit({ type: "message_end", message: assistant });
  const completedStatus = ctx.ui.setStatus.mock.calls.at(-1).arguments[1];
  assert.ok(completedStatus.length <= 80);
  assert.match(completedStatus, /^SY remaining ~1\.9k tok ETA [^|]+ \|/);
  assert.ok(!completedStatus.includes("impossible"));
  assert.match(completedStatus, /session 6\.1k\/.+ \| last 6\.1k\//);
  const branch = [
    { type: "message", message: { role: "user", content: "original task" } },
    { type: "message", message: assistant },
    { type: "message", message: { role: "toolResult", toolName: "bash", isError: false, content: [{ type: "text", text: "tests pass" }] } },
  ];
  ctx.sessionManager.getBranch = () => branch;
  registry.classify.mock.mockImplementation(async (_, context) => context.questions.route ? result() : forecast("impossible"));
  await route(request({ reason: "continuation", state: initial.state, messages: [{ role: "user", content: "compacted summary" }] }));
  const continued = registry.classify.mock.calls.at(-1).arguments[1];
  assert.equal(continued.state.task_history[0].message, "original task");
  assert.equal(continued.state.token_usage.tokens_used_so_far, 1100);
  assert.equal(continued.state.token_usage.reasoning_output_tokens_so_far, 40);
  assert.match(ctx.ui.setStatus.mock.calls.at(-1).arguments[1], /^SY remaining unavailable \|/);
  assert.equal(entries.filter((entry) => entry.customType === "switchyard.prediction.v1").length, 2);
  emit({ type: "agent_settled" });
  assert.ok(!ctx.ui.setStatus.mock.calls.at(-1).arguments[1].includes("remaining"));
});

test("prediction failure allows completion routing, direct helpers skip prediction, and cancellation propagates", async (t) => {
  const { route, registry, ctx } = setup(t);
  registry.classify.mock.mockImplementation(async (_, context) => {
    if (context.questions.route) return result();
    throw new Error("prediction unavailable");
  });
  const initial = await route(request());
  assert.equal(initial.model.id, "gpt-6.1-sol");
  assert.match(ctx.ui.setStatus.mock.calls.at(-1).arguments[1], /^SY remaining unavailable/);
  const calls = registry.classify.mock.callCount();
  await route(request({ reason: "direct" }));
  assert.equal(registry.classify.mock.callCount(), calls + 1);

  const controller = new AbortController();
  registry.classify.mock.mockImplementation(async (_, context) => {
    if (context.questions.route) return result();
    controller.abort();
    return { ...forecast(), stopReason: "aborted" };
  });
  await assert.rejects(route(request({ signal: controller.signal })), { name: "AbortError", code: "ABORT_ERR" });
});

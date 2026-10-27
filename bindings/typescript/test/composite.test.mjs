// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { CompositeRouter } from "../dist/index.js";

const options = {
  targets: { capable: "provider/strong", efficient: "provider/weak", classifier: "provider/judge" },
  classifier: {
    classify_trigger: "user_turn",
    fail_open: false,
    decision: {
      cutoff: 0.4,
      candidates: { a: "capable", b: "efficient" },
      evidence: { description: "a handles complex work; b handles routine work" },
    },
  },
  stage: { efficient_first_threshold: 0.5, capable_first_threshold: 0.2 },
};
const messages = [{ role: "user", content: [{ type: "text", text: "Fix the parser" }] }];
const continuation = [
  ...messages,
  { role: "assistant", content: [{ type: "tool_call", id: "read-1", name: "read", arguments: { path: "parser.rs" } }] },
  { role: "tool", content: [{ type: "tool_result", tool_call_id: "read-1", is_error: false, content: [{ type: "text", text: "fn parse() {}" }] }] },
];
const instructions = JSON.parse(readFileSync(new URL(
  "../../../crates/libsy/src/prompts/capability-classifier/relative_advantage.json", import.meta.url,
), "utf8"));
const unexpectedDecision = async () => assert.fail("unexpected decision call");

function response(advantage) {
  return { answers: { route: { value: {
    type: "choice", data: { selected: "advantage", probabilities: { advantage, no_advantage: 1 - advantage } },
  } } } };
}

test("decision request, cutoff, and branch-owned state", async () => {
  const router = CompositeRouter.create(options);
  const states = await Promise.all([0.41, 0.4].map(async (advantage) => {
    let calls = 0;
    const selected = await router.route(messages, { decide: async (request) => {
      calls++;
      assert.equal(request.model, options.targets.classifier);
      assert.deepEqual(request.context, {
        task: messages, candidates: ["a", "b"], comparison: { capable: "a", efficient: "b" },
        evidence: options.classifier.decision.evidence,
      });
      assert.deepEqual(Object.keys(request.questions), ["route"]);
      assert.deepEqual(request.questions.route.instructions, instructions);
      assert.equal(request.questions.route.kind.type, "choice");
      assert.deepEqual(request.questions.route.kind.data.options.map((option) => option.id), ["advantage", "no_advantage"]);
      return response(advantage);
    } });
    assert.equal(calls, 1);
    assert.equal(selected.model, advantage > 0.4 ? options.targets.capable : options.targets.efficient);
    assert.ok(selected.state);
    return selected;
  }));
  for (const { model, state } of states) {
    const saved = structuredClone(state);
    assert.deepEqual(await router.route(continuation, { state, decide: unexpectedDecision }), { model });
    assert.deepEqual(state, saved);
  }
  assert.deepEqual(await router.route(continuation, { decide: unexpectedDecision }), {
    model: options.targets.efficient,
    state: { stage_previous_tier: { string: "weak" } },
  });

  const once = structuredClone(options);
  once.classifier.classify_trigger = "new_session";
  const sessionRouter = CompositeRouter.create(once);
  const { state } = await sessionRouter.route(messages, { decide: async () => response(0.41) });
  assert.deepEqual(await sessionRouter.route(messages, { state, decide: unexpectedDecision }), { model: options.targets.capable });
});

test("serialized state preserves capable holds until they expire", async () => {
  const router = CompositeRouter.create(options);
  let selected = await router.route(continuation, { decide: unexpectedDecision });
  assert.equal(selected.model, options.targets.efficient);

  const failed = structuredClone(continuation);
  failed.at(-1).content[0].is_error = true;
  failed.at(-1).content[0].content[0].text = "Out of memory: kill process 1234";
  const requests = [failed, continuation, continuation, continuation];
  const expected = [options.targets.capable, options.targets.capable, options.targets.capable, options.targets.efficient];
  for (const [i, request] of requests.entries()) {
    assert.ok(selected.state);
    const state = JSON.parse(JSON.stringify(selected.state));
    const saved = structuredClone(state);
    selected = await router.route(request, { state, decide: unexpectedDecision });
    assert.equal(selected.model, expected[i]);
    assert.deepEqual(state, saved);
    // A separate branch without the returned state has no hold.
    assert.equal((await router.route(continuation, { decide: unexpectedDecision })).model, options.targets.efficient);
  }
});

test("capable-first pins efficient by default and accepts an explicit opt-out", async () => {
  for (const turns of [undefined, 0]) {
    const config = structuredClone(options);
    config.stage.efficient_hold_turns = turns;
    const router = CompositeRouter.create(config);
    const initial = await router.route(messages, { decide: async () => response(0.41) });
    assert.equal(initial.model, options.targets.capable);
    const write = structuredClone(continuation);
    write[1].content[0].name = "write";
    let selected = await router.route(write, { state: initial.state, decide: unexpectedDecision });
    assert.equal(selected.model, options.targets.efficient);
    const failed = structuredClone(continuation);
    failed.at(-1).content[0].is_error = true;
    failed.at(-1).content[0].content[0].text = "Out of memory: kill process 1234";
    let state = selected.state;
    for (let i = 0; i < 3; i++) {
      assert.ok(state);
      selected = await router.route(failed, { state: JSON.parse(JSON.stringify(state)), decide: unexpectedDecision });
      assert.equal(selected.model, turns === 0 ? options.targets.capable : options.targets.efficient);
      state = selected.state ?? state;
    }
  }
});

test("validation and decision failure policy", async () => {
  for (const change of [
    (config) => { config.targets.efficient = config.targets.capable; },
    (config) => { config.classifier.decision.cutoff = 2; },
    (config) => { config.classifier.decision.extra = true; },
    (config) => { config.classifier.classify_trigger = "every_request"; },
    (config) => { config.stage.efficient_hold_turns = -2; },
  ]) {
    const config = structuredClone(options);
    change(config);
    assert.throws(() => CompositeRouter.create(config), { code: "ERR_CONFIG" });
  }
  const router = CompositeRouter.create(options);
  await assert.rejects(router.route([{ role: "invalid", content: [] }], { decide: unexpectedDecision }), { code: "ERR_INVALID_REQUEST" });
  await assert.rejects(router.route(messages, { state: { fall_open: "strong" }, decide: unexpectedDecision }), { code: "ERR_INVALID_REQUEST" });
  for (const answer of [{}, undefined, { answers: 1n }]) {
    await assert.rejects(router.route(messages, { decide: async () => answer }), { code: "ERR_INVALID_REQUEST" });
  }

  const cause = new Error("classifier unavailable");
  const decide = async () => { throw cause; };
  await assert.rejects(router.route(messages, { decide }), (error) => {
    assert.equal(error.code, "ERR_ROUTING");
    assert.match(error.message, /classifier unavailable/);
    assert.equal(error.cause, cause);
    return true;
  });
  const config = structuredClone(options);
  config.classifier.fail_open = true;
  const fallback = CompositeRouter.create(config);
  const { state } = await fallback.route(messages, { decide: async () => response(0.41) });
  assert.deepEqual(await fallback.route(messages, { state, decide }), { model: options.targets.capable });
  assert.deepEqual(await fallback.route(messages, { decide }), {
    model: options.targets.efficient,
    state: { stage_previous_tier: { string: "weak" } },
  });
  assert.deepEqual(await router.route(messages, { state, decide: async () => ({ answers: {} }) }), { model: options.targets.capable });
});

test("abort settles promptly, ignores late answers, and removes listeners", { timeout: 5000 }, async () => {
  const router = CompositeRouter.create(options);
  await assert.rejects(router.route(messages, {
    signal: AbortSignal.abort(), decide: unexpectedDecision,
  }), { name: "AbortError", code: "ABORT_ERR" });

  for (const timing of ["native", "resolve", "reject"]) {
    const entered = Promise.withResolvers();
    const pending = Promise.withResolvers();
    const controller = new AbortController();
    const rejected = assert.rejects(router.route(messages, {
      signal: controller.signal,
      decide: () => { entered.resolve(); return pending.promise; },
    }), { name: "AbortError", code: "ABORT_ERR" });
    if (timing !== "native") await entered.promise;
    controller.abort();
    await rejected;
    assert.equal(getEventListeners(controller.signal, "abort").length, 0);
    if (timing === "reject") pending.reject(new Error("late rejection"));
    else pending.resolve(response(0.41));
  }
  const controller = new AbortController();
  await router.route(messages, { signal: controller.signal, decide: async () => response(0.4) });
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

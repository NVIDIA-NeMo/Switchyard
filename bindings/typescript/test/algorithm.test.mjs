// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { AlgorithmFactory, ConfiguredAlgorithm, LibsyError, LlmResponse, Step } from "../dist/index.js";

const deployment = `
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
[targets]
fast = { id = "upstream/fast", llm_client = "answers" }
capable = { id = "upstream/capable", llm_client = "answers" }
[decision_clients.judge]
format = "openai_decisions"
endpoint = "https://example.test/v1/decisions"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
timeout_ms = 2000
[decision_targets.judge]
id = "upstream/judge"
decision_client = "judge"
[routes.parent]
id = "public/main"
type = "passthrough"
target = "fast"
[routes.parent.subagents]
type = "passthrough"
target = "capable"
[routes.classifier]
id = "public/classify"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
[routes.advisor]
id = "public/advisor"
type = "advisor"
executor_target = "fast"
advisor_target = "capable"
gate_min_tool_results = 1
`;
const routes = AlgorithmFactory.fromToml(deployment).build();
const request = { model: "public/main", messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }] };
const aggregate = (text = "hello") => ({
  model: "upstream/fast", outputs: [{ role: "assistant", content: [{ type: "text", text }], stop_reason: "end_turn" }],
});
const answer = { answers: { route: { value: {
  type: "choice", data: { selected: "capable", probabilities: { capable: 0.9, efficient: 0.1 } },
} } } };

async function collect(source) {
  const result = [];
  for await (const item of source) result.push(item);
  return result;
}

async function start(t) {
  const run = routes["public/advisor"].runStream(request);
  t.after(() => run.return());
  const step = (await run.next()).value;
  assert.ok(step instanceof Step.CallModel);
  return { run, call: step.call };
}

test("factory returns logical targets, full outcomes, and header-driven subagent selection", async () => {
  assert.deepEqual(Object.keys(routes), ["public/advisor", "public/classify", "public/main"]);
  assert.ok(routes["public/main"] instanceof ConfiguredAlgorithm);
  for (const [headers, expected] of [[undefined, "fast"], [{ "x-switchyard-is-subagent": "true" }, "capable"]]) {
    const steps = await collect(routes["public/main"].runStream(request, { headers }));
    assert.equal(steps.length, 1);
    assert.ok(steps[0] instanceof Step.Done);
    const outcome = steps[0].outcome;
    assert.deepEqual(outcome.selected_model_ids, [expected]);
    assert.equal(outcome.request.model, expected);
    assert.deepEqual(outcome.request.messages, request.messages);
    assert.equal(outcome.fallback_request, null);
    assert.deepEqual(outcome.candidate_requests, {});
    assert.equal(outcome.response, null);
    assert.equal(typeof outcome.metadata.outcome_id, "string");
  }
  assert.throws(() => AlgorithmFactory.fromToml("invalid"), { code: "ERR_CONFIG" });
  assert.throws(() => routes["public/main"].runStream({ messages: [{ role: "invalid", content: [] }] }), { code: "ERR_INVALID_REQUEST" });
});

test("factory rejects configuration requiring Runner admission", () => {
  assert.throws(() => AlgorithmFactory.fromToml(`${deployment}\n[routes.parent.eligibility]\n`).build(), {
    code: "ERR_CONFIG", message: /requires Runner admission/,
  });
});

test("decision calls accept async replies and failures", async (t) => {
  for (const fail of [false, true]) {
    const run = routes["public/classify"].runStream(request);
    t.after(() => run.return());
    const step = (await run.next()).value;
    assert.ok(step instanceof Step.CallDecision);
    assert.equal(step.call.model, "upstream/judge");
    assert.equal(step.call.request.model, "upstream/judge");
    if (fail) await step.call.fail(new Error("judge unavailable"));
    else await step.call.respond(Promise.resolve(answer));
    await assert.rejects(step.call.respond(Promise.resolve(answer)), LibsyError);
    const done = (await run.next()).value;
    assert.ok(done instanceof Step.Done);
    assert.deepEqual(done.outcome.selected_model_ids, ["capable", "fast"]);
    assert.equal(done.outcome.metadata.evidence.source, fail ? "fail_open" : "decision_classifier");
  }
});

test("model calls accept normalized aggregates and response streams", async (t) => {
  const events = [
    { preservation: null, normalized: [{ MessageStart: { id: "answer-1", model: "upstream/fast" } }] },
    { preservation: null, normalized: [{ TextDelta: { index: 0, text: "hello" } }] },
    { preservation: null, normalized: [{ Usage: {
      input_tokens: 2, cached_input_tokens: null, cache_creation_input_tokens: null,
      output_tokens: 1, total_tokens: 3, reasoning_tokens: null,
    } }] },
    { preservation: null, normalized: [{ MessageStop: { reason: "end_turn" } }] },
  ];
  for (const isStream of [false, true]) {
    const { run, call } = await start(t);
    assert.deepEqual(call.models, ["fast"]);
    assert.equal(call.request.model, "fast");
    await call.respond(Promise.resolve(isStream
      ? new LlmResponse.Stream((async function* () { yield* events; })())
      : new LlmResponse.Agg(aggregate())));
    let isPolled = false;
    const duplicate = new LlmResponse.Stream({ async *[Symbol.asyncIterator]() { isPolled = true; } });
    await assert.rejects(call.respond(Promise.resolve(duplicate)), LibsyError);
    assert.equal(isPolled, false);
    const step = (await run.next()).value;
    assert.ok(step instanceof Step.Done);
    const outcome = step.outcome;
    if (isStream) {
      assert.ok(outcome.response instanceof LlmResponse.Stream);
      const replayed = await collect(outcome.response.stream);
      assert.deepEqual(replayed.map((event) => event.normalized), events.map((event) => event.normalized));
    } else {
      assert.ok(outcome.response instanceof LlmResponse.Agg);
      assert.equal(outcome.response.response.outputs[0].content[0].text, "hello");
    }
  }
});

test("response stream setup and read failures reach the routing iterator", { timeout: 5000 }, async (t) => {
  for (const isSetupFailure of [true, false]) {
    const { run, call } = await start(t);
    const source = isSetupFailure
      ? { [Symbol.asyncIterator]() { throw new Error("stream unavailable"); } }
      : { async *[Symbol.asyncIterator]() { throw new Error("stream unavailable"); } };
    const rejected = assert.rejects(run.next(), {
      name: "LibsyError", code: "ERR_ROUTING", message: /stream unavailable/,
    });
    await call.respond(Promise.resolve(new LlmResponse.Stream(source)));
    await rejected;
  }
});

test("non-recovering model responders reject with the original host failure", { timeout: 5000 }, async (t) => {
  for (const fail of [false, true]) {
    const { call } = await start(t);
    const cause = new Error("executor unavailable");
    await assert.rejects(fail ? call.fail(cause) : call.respond(Promise.reject(cause)), {
      name: "LibsyError", code: "ERR_ROUTING", message: /executor unavailable/,
    });
  }
});

test("concurrent model calls keep replies on their originating runs", { timeout: 5000 }, async (t) => {
  const pending = await Promise.all([start(t), start(t)]);
  const work = [Promise.withResolvers(), Promise.withResolvers()];
  const responding = pending.map(({ call }, index) => call.respond(work[index].promise));
  const outcomes = pending.map(({ run }) => run.next());
  for (const [index, text] of [[1, "second"], [0, "first"]]) {
    work[index].resolve(new LlmResponse.Agg(aggregate(text)));
    await responding[index];
    const step = (await outcomes[index]).value;
    assert.ok(step instanceof Step.Done);
    assert.equal(step.outcome.response.response.outputs[0].content[0].text, text);
  }
  for (const { run } of pending) assert.deepEqual(await run.next(), { done: true, value: undefined });
});

test("iterator return releases native waits and closes input readers without cancelling host promises", { timeout: 5000 }, async (t) => {
  const { run: concurrent, call: first } = await start(t);
  const p1 = concurrent.next();
  const p2 = concurrent.next();
  await first.respond(Promise.resolve(new LlmResponse.Agg(aggregate())));
  assert.ok((await p1).value instanceof Step.Done);
  assert.deepEqual(await p2, { done: true, value: undefined });

  const { run, call } = await start(t);
  const work = Promise.withResolvers();
  let isHostFinished = false;
  void work.promise.then(() => { isHostFinished = true; }, () => { isHostFinished = true; });
  const responding = call.respond(work.promise);
  await run.return();
  await responding;
  assert.equal(isHostFinished, false);
  work.reject(new Error("late host failure"));
  await Promise.resolve();
  assert.equal(isHostFinished, true);

  const { run: streamRun, call: streamCall } = await start(t);
  const entered = Promise.withResolvers();
  const returned = Promise.withResolvers();
  const pending = Promise.withResolvers();
  const source = {
    [Symbol.asyncIterator]() { return this; },
    next() { entered.resolve(); return pending.promise; },
    return() { returned.resolve(); return Promise.resolve({ done: true }); },
  };
  await streamCall.respond(Promise.resolve(new LlmResponse.Stream(source)));
  await entered.promise;
  const next = streamRun.next();
  await streamRun.return();
  assert.equal((await next).done, true);
  await returned.promise;
  pending.resolve({ done: true });
});

test("host-owned AbortController closes HTTP work while a concurrent route survives", { timeout: 5000 }, async (t) => {
  const entered = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const controller = new AbortController();
  const server = createServer((_req, res) => {
    res.once("close", closed.resolve);
    entered.resolve();
  });
  t.after(async () => {
    controller.abort();
    const closing = new Promise((resolve) => server.close(resolve));
    server.closeAllConnections();
    await closing;
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const cancelled = await start(t);
  const work = fetch(base, { signal: controller.signal })
    .then((response) => response.json()).then((response) => new LlmResponse.Agg(response));
  const responding = cancelled.call.respond(work).finally(async () => {
    controller.abort();
    await work.catch(() => {});
  });
  await entered.promise;
  const healthy = await start(t);
  const healthyNext = healthy.run.next();
  await cancelled.run.return();
  await responding;
  await closed.promise;
  await healthy.call.respond(Promise.resolve(new LlmResponse.Agg(aggregate("healthy"))));
  const done = (await healthyNext).value;
  assert.ok(done instanceof Step.Done);
  assert.equal(done.outcome.response.response.outputs[0].content[0].text, "healthy");
  assert.deepEqual(await healthy.run.next(), { done: true, value: undefined });
  assert.deepEqual(await cancelled.run.next(), { done: true, value: undefined });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { test } from "node:test";
import { AlgorithmFactory, ConfiguredAlgorithm, LibsyError, LlmResponse, Step } from "../dist/index.js";

const fixture = readFileSync(new URL("./fixtures/deployment.toml", import.meta.url), "utf8");
const deployment = `${fixture}
[routes.model]
id = "public/model"
type = "llm_classifier"
classifier_target = "fast"
strong_target = "capable"
weak_target = "fast"
base_threshold = 0.5
fail_open = false
[routes.advisor]
id = "public/advisor"
type = "advisor"
executor_target = "fast"
advisor_target = "capable"
gate_min_tool_results = 1
`;
const request = { model: "public/main", messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }] };
const aggregate = (text = "hello") => ({
  model: "upstream/fast", outputs: [{ role: "assistant", content: [{ type: "text", text }], stop_reason: "end_turn" }],
});
const answer = { answers: { route: { value: {
  type: "choice", data: { selected: "advantage", probabilities: { advantage: 0.9, no_advantage: 0.1 } },
} } } };

async function collect(source) {
  const result = [];
  for await (const item of source) result.push(item);
  return result;
}

test("factory returns logical targets, full outcomes, and header-driven subagent selection", async () => {
  const routes = AlgorithmFactory.fromToml(fixture).build();
  assert.deepEqual(Object.keys(routes), ["public/classify", "public/main"]);
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
    assert.match(outcome.metadata.outcome_id, /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert.deepEqual(outcome.metadata.evidence, headers ? { source: "fall_open" } : null);
  }
  assert.throws(() => AlgorithmFactory.fromToml("invalid"), { code: "ERR_CONFIG" });
  assert.throws(() => AlgorithmFactory.fromToml(fixture.replace('target = "fast"', 'target = "missing"')).build(), { code: "ERR_CONFIG" });
  assert.throws(() => routes["public/main"].runStream({ messages: [{ role: "invalid", content: [] }] }), { code: "ERR_INVALID_REQUEST" });
});

test("factory rejects configuration requiring Runner admission", () => {
  assert.throws(() => AlgorithmFactory.fromToml(`${fixture}\n[routes.parent.eligibility]\n`).build(), (error) => {
    assert.equal(error.code, "ERR_CONFIG");
    assert.match(error.message, /requires Runner admission/);
    return true;
  });
});

test("decision calls accept async replies and failures through the algorithm policy", async () => {
  for (const fail of [false, true]) {
    const route = AlgorithmFactory.fromToml(fixture).build()["public/classify"];
    let calls = 0;
    let outcome;
    for await (const step of route.runStream(request)) {
      if (step instanceof Step.CallDecision) {
        calls++;
        assert.equal(step.call.model, "upstream/judge");
        assert.equal(step.call.request.model, "upstream/judge");
        assert.deepEqual(Object.keys(step.call.request.questions), ["route"]);
        if (fail) await step.call.fail(new Error("judge unavailable"));
        else await step.call.respond(Promise.resolve(answer));
        await assert.rejects(step.call.respond(Promise.resolve(answer)), LibsyError);
      } else {
        assert.ok(step instanceof Step.Done);
        outcome = step.outcome;
      }
    }
    assert.equal(calls, 1);
    assert.deepEqual(outcome.selected_model_ids, ["capable", "fast"]);
    assert.equal(outcome.metadata.evidence.source, fail ? "fail_open" : "decision_classifier");
    assert.ok(outcome.metadata.algorithm);
  }
});

test("model calls accept normalized aggregates and preserve an existing normalized response stream", async () => {
  const routes = AlgorithmFactory.fromToml(deployment).build();
  for await (const step of routes["public/model"].runStream(request)) {
    if (step instanceof Step.CallModel) {
      assert.deepEqual(step.call.models, ["fast"]);
      assert.equal(step.call.request.model, "fast");
      await step.call.respond(Promise.resolve(new LlmResponse.Agg(aggregate(JSON.stringify({
        crux: "bounded task", primary_rule: "SUP-1", capability_boundary: "supported", p_solve: 0.9,
      })))));
      let isPolled = false;
      const duplicate = new LlmResponse.Stream({
        async *[Symbol.asyncIterator]() { isPolled = true; },
      });
      await assert.rejects(step.call.respond(Promise.resolve(duplicate)), LibsyError);
      assert.equal(isPolled, false);
    } else {
      assert.ok(step instanceof Step.Done);
      assert.equal(step.outcome.selected_model_ids[0], "fast");
    }
  }
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
    const route = AlgorithmFactory.fromToml(deployment).build()["public/advisor"];
    let outcome;
    for await (const step of route.runStream(request)) {
      if (step instanceof Step.CallModel) {
        const response = isStream
          ? new LlmResponse.Stream((async function* () { yield* events; })())
          : new LlmResponse.Agg(aggregate());
        await step.call.respond(Promise.resolve(response));
      } else outcome = step.outcome;
    }
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

test("non-recovering model responders reject with the original host failure", { timeout: 5000 }, async () => {
  const route = AlgorithmFactory.fromToml(deployment).build()["public/advisor"];
  for (const fail of [false, true]) {
    const run = route.runStream(request);
    try {
      const step = (await run.next()).value;
      assert.ok(step instanceof Step.CallModel);
      const cause = new Error("executor unavailable");
      const responding = fail ? step.call.fail(cause) : step.call.respond(Promise.reject(cause));
      await assert.rejects(responding, (error) => {
        assert.ok(error instanceof LibsyError);
        assert.equal(error.code, "ERR_ROUTING");
        assert.match(error.message, /executor unavailable/);
        assert.doesNotMatch(error.message, /promise was dropped/);
        return true;
      });
    } finally {
      await run.return();
    }
  }
});

test("concurrent model calls keep replies on their originating runs", { timeout: 5000 }, async () => {
  const route = AlgorithmFactory.fromToml(deployment).build()["public/advisor"];
  const runs = [route.runStream(request), route.runStream(request)];
  try {
    const calls = (await Promise.all(runs.map((run) => run.next()))).map((step) => step.value.call);
    assert.deepEqual(calls.map((call) => call.models), [["fast"], ["fast"]]);
    const work = [Promise.withResolvers(), Promise.withResolvers()];
    const responding = calls.map((call, index) => call.respond(work[index].promise));
    const outcomes = runs.map((run) => run.next());
    work[1].resolve(new LlmResponse.Agg(aggregate("second")));
    await responding[1];
    const second = (await outcomes[1]).value;
    assert.ok(second instanceof Step.Done);
    assert.equal(second.outcome.response.response.outputs[0].content[0].text, "second");
    work[0].resolve(new LlmResponse.Agg(aggregate("first")));
    await responding[0];
    const first = (await outcomes[0]).value;
    assert.ok(first instanceof Step.Done);
    assert.equal(first.outcome.response.response.outputs[0].content[0].text, "first");
    for (const run of runs) assert.deepEqual(await run.next(), { done: true, value: undefined });
  } finally {
    await Promise.all(runs.map((run) => run.return()));
  }
});

test("iterator return releases native waits and closes input readers without cancelling host promises", { timeout: 5000 }, async () => {
  const route = AlgorithmFactory.fromToml(deployment).build()["public/advisor"];
  const concurrent = route.runStream(request);
  const first = (await concurrent.next()).value;
  assert.ok(first instanceof Step.CallModel);
  const p1 = concurrent.next();
  const p2 = concurrent.next();
  await first.call.respond(Promise.resolve(new LlmResponse.Agg(aggregate())));
  assert.ok((await p1).value instanceof Step.Done);
  assert.deepEqual(await p2, { done: true, value: undefined });

  const run = route.runStream(request);
  const step = (await run.next()).value;
  assert.ok(step instanceof Step.CallModel);
  const work = Promise.withResolvers();
  let isHostFinished = false;
  void work.promise.then(() => { isHostFinished = true; }, () => { isHostFinished = true; });
  const responding = step.call.respond(work.promise);
  await run.return();
  await responding;
  assert.equal(isHostFinished, false);
  work.reject(new Error("late host failure"));
  await Promise.resolve();
  assert.equal(isHostFinished, true);

  const streamRun = route.runStream(request);
  const call = (await streamRun.next()).value.call;
  const entered = Promise.withResolvers();
  const returned = Promise.withResolvers();
  const pending = Promise.withResolvers();
  const source = {
    [Symbol.asyncIterator]() { return this; },
    next() { entered.resolve(); return pending.promise; },
    return() { returned.resolve(); return Promise.resolve({ done: true }); },
  };
  await call.respond(Promise.resolve(new LlmResponse.Stream(source)));
  await entered.promise;
  const next = streamRun.next();
  await streamRun.return();
  assert.equal((await next).done, true);
  await returned.promise;
  pending.resolve({ done: true });
});

test("host-owned AbortController closes HTTP work while a concurrent route survives", { timeout: 5000 }, async (t) => {
  const requests = Object.fromEntries(["control", "cancel", "healthy"].map((name) => [name, {
    entered: Promise.withResolvers(), closed: Promise.withResolvers(), controller: new AbortController(), count: 0,
  }]));
  const healthyRelease = Promise.withResolvers();
  const runs = [];
  let active = 0;
  const server = createServer((req, res) => {
    const state = requests[req.url.slice(1)];
    state.count++;
    active++;
    res.on("close", () => { active--; state.closed.resolve(); });
    req.resume();
    state.entered.resolve();
    if (req.url === "/healthy") {
      void healthyRelease.promise.then(() => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(aggregate("healthy")));
      });
    }
  });
  t.after(async () => {
    for (const state of Object.values(requests)) state.controller.abort();
    await Promise.all(runs.map((run) => run.return()));
    const closing = new Promise((resolve) => server.close(resolve));
    server.closeAllConnections();
    await closing;
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const route = AlgorithmFactory.fromToml(deployment).build()["public/advisor"];
  async function start(name) {
    const run = route.runStream(request);
    runs.push(run);
    const step = (await run.next()).value;
    assert.ok(step instanceof Step.CallModel);
    const work = fetch(`${base}/${name}`, { signal: requests[name].controller.signal })
      .then((response) => response.json()).then((response) => new LlmResponse.Agg(response));
    return { run, call: step.call, work };
  }

  const control = await start("control");
  const controlResponding = control.call.respond(control.work);
  await requests.control.entered.promise;
  await control.run.return();
  await controlResponding;
  assert.equal(active, 1);
  assert.equal(requests.control.controller.signal.aborted, false);
  requests.control.controller.abort();
  await control.work.catch(() => {});
  await requests.control.closed.promise;
  assert.equal(active, 0);

  const cancelled = await start("cancel");
  const responding = (async () => {
    try {
      await cancelled.call.respond(cancelled.work);
    } finally {
      requests.cancel.controller.abort();
      await cancelled.work.catch(() => {});
    }
  })();
  await requests.cancel.entered.promise;
  const healthy = await start("healthy");
  const healthyResponding = healthy.call.respond(healthy.work);
  void healthyResponding.catch(() => {});
  const healthyNext = healthy.run.next();
  await requests.healthy.entered.promise;
  assert.equal(active, 2);
  await cancelled.run.return();
  await responding;
  await requests.cancel.closed.promise;
  assert.equal(active, 1);
  assert.equal(requests.healthy.controller.signal.aborted, false);
  healthyRelease.resolve();
  await healthyResponding;
  const done = (await healthyNext).value;
  assert.ok(done instanceof Step.Done);
  assert.equal(done.outcome.response.response.outputs[0].content[0].text, "healthy");
  assert.deepEqual(await healthy.run.next(), { done: true, value: undefined });
  await requests.healthy.closed.promise;
  assert.equal(active, 0);
  assert.deepEqual(await cancelled.run.next(), { done: true, value: undefined });
  assert.deepEqual(Object.values(requests).map((state) => state.count), [1, 1, 1]);
});

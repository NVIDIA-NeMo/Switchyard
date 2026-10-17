// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { getEventListeners, once } from "node:events";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { test } from "node:test";
import { Runner } from "../dist/index.js";

const config = readFileSync(new URL("./classifier.toml", import.meta.url), "utf8");

async function setup(t, handler) {
  const calls = [];
  const http = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    calls.push({ url: req.url, body: JSON.parse(body) });
    handler(res);
  });
  http.listen(0, "127.0.0.1");
  await once(http, "listening");
  t.after(() => {
    http.closeAllConnections();
    return new Promise((resolve) => http.close(resolve));
  });
  const previous = process.env.SWITCHYARD_TEST_KEY;
  process.env.SWITCHYARD_TEST_KEY = "secret-sentinel";
  t.after(() => {
    if (previous === undefined) delete process.env.SWITCHYARD_TEST_KEY;
    else process.env.SWITCHYARD_TEST_KEY = previous;
  });
  const source = config.replace("http://127.0.0.1:1/classify", `http://127.0.0.1:${http.address().port}/classify`);
  return { runner: Runner.fromToml(source), calls };
}

test("selects the configured target through the classifier", async (t) => {
  const { runner, calls } = await setup(t, (res) => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ answers: { route: {
      type: "choice", choice: "advantage", probabilities: { advantage: 0.9, no_advantage: 0.1 },
    } } }));
  });
  assert.equal(calls.length, 0);
  assert.deepEqual(await runner.decide("switchyard/planner", "hard task"), {
    target: "complex", model: "gpt-5.6-sol",
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/classify");
  assert.deepEqual(calls[0].body.state.task[0].content, [{ type: "text", text: "hard task" }]);
});

test("reports safe configuration, route, and classifier errors", async (t) => {
  assert.throws(() => Runner.fromToml("secret-sentinel = ["), (error) => {
    assert.equal(error.code, "ERR_CONFIG");
    assert.match(error.message, /failed to parse TOML at line 1, column \d+/);
    assert.doesNotMatch(error.message, /secret-sentinel/);
    return true;
  });
  assert.throws(
    () => Runner.fromToml(config.replace('classify_trigger = "every_request"', 'classify_trigger = "new_session"')),
    { code: "ERR_CONFIG", message: /route planner:.*every_request/ },
  );
  const { runner } = await setup(t, (res) => { res.writeHead(401); res.end("secret-sentinel"); });
  await assert.rejects(runner.decide("unknown", "hello"), { code: "ERR_UNKNOWN_ROUTE" });
  await assert.rejects(runner.decide("switchyard/planner", [{ role: "invalid", content: [] }]), { code: "ERR_INVALID_REQUEST" });
  await assert.rejects(runner.decide("switchyard/planner", "hello"), (error) => {
    assert.equal(error.code, "ERR_ROUTING");
    assert.doesNotMatch(error.message, /secret-sentinel/);
    return true;
  });
});

test("abort cancels the native HTTP request and releases its listener", { timeout: 5000 }, async (t) => {
  const received = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const { runner } = await setup(t, (res) => {
    res.on("close", closed.resolve);
    received.resolve();
  });
  const controller = new AbortController();
  const rejected = assert.rejects(
    runner.decide("switchyard/planner", "hello", { signal: controller.signal }),
    { name: "AbortError", code: "ABORT_ERR" },
  );
  await received.promise;
  controller.abort();
  await rejected;
  await closed.promise;
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

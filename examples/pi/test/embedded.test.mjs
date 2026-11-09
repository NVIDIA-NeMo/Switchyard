// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { DllmEngine } from "@switchyard/libsy";
import extension from "../switchyard-router.ts";

function setup(t, dllm) {
  const dir = mkdtempSync(join(tmpdir(), "switchyard-embedded-"));
  const previous = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  t.after(() => {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previous;
    rmSync(dir, { recursive: true, force: true });
  });
  writeFileSync(join(dir, "switchyard-router.json"), JSON.stringify({
    dllm, capable_model: "test/capable", efficient_model: "test/efficient",
  }));
  const handlers = {};
  let model;
  extension({
    on: (event, handler) => { handlers[event] = handler; },
    registerVirtualModel: (value) => { model = value; },
  });
  const registry = {
    find: (provider, id) => ({ provider, id }),
    findOfType: () => { throw new Error("embedded decisions must bypass Pi classifiers"); },
    classify: () => { throw new Error("embedded decisions must bypass Pi classifiers"); },
  };
  return {
    start: () => handlers.session_start(),
    shutdown: () => handlers.session_shutdown(),
    route: (fields = {}) => model.route({
      reason: "user", messages: [{ role: "user", content: "What is 2 + 2?" }],
      signal: new AbortController().signal, ...fields,
    }, { modelRegistry: registry }),
  };
}

test("session startup loads dllm before routing and shutdown closes it", async (t) => {
  const config = { library: "engine.so", tokenizer_library: "tokenizer.so", weights: "weights", tokenizer: "tokenizer" };
  let finishLoading;
  const ready = new Promise((resolve) => { finishLoading = resolve; });
  const decide = t.mock.fn(async (request, { signal }) => {
    assert.equal(signal.aborted, false);
    assert.ok(request.questions.route);
    return { answers: { route: { value: { type: "choice", data: {
      selected: "efficient", probabilities: { capable: 0.1, efficient: 0.9 },
    } } } } };
  });
  const close = t.mock.fn(async () => {});
  const load = t.mock.method(DllmEngine, "load", async (options) => {
    assert.deepEqual(options, config);
    await ready;
    return { decide, close };
  });
  const host = setup(t, config);
  let started = false;
  const starting = host.start().then(() => { started = true; });
  await Promise.resolve();
  assert.equal(load.mock.callCount(), 1);
  assert.equal(started, false);
  finishLoading();
  await starting;
  const selected = await host.route();
  assert.deepEqual(selected.model, { provider: "test", id: "efficient" });
  assert.equal(decide.mock.callCount(), 1);
  assert.equal(load.mock.callCount(), 1);
  await host.shutdown();
  assert.equal(close.mock.callCount(), 1);
});

test("dllm startup failures surface before a routing request", async (t) => {
  t.mock.method(DllmEngine, "load", async () => { throw new Error("model load failed"); });
  const host = setup(t, {});
  await assert.rejects(host.start(), /model load failed/);
});

test("real local Deciffusion initializes and drives the Pi composite router", {
  skip: !process.env.SWITCHYARD_DLLM_OPTIONS,
}, async (t) => {
  const config = JSON.parse(readFileSync(process.env.SWITCHYARD_DLLM_OPTIONS, "utf8"));
  const host = setup(t, config);
  await host.start();
  try {
    const selected = await host.route();
    assert.deepEqual(selected.model, { provider: "test", id: "efficient" });
    assert.ok(selected.state);
  } finally {
    await host.shutdown();
  }
});

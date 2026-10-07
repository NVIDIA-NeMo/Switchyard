// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { Runner } from "@switchyard/runner";
import extension from "../switchyard-router.ts";

const context = { modelRegistry: { find: (provider, id) => ({ provider, id }) } };
const request = (fields = {}) => ({
  reason: "user", messages: [{ role: "user", content: "task" }],
  thinkingLevel: "high", signal: new AbortController().signal, ...fields,
});

function setup(t, decide = async () => ({ target: "complex", model: "gpt-5.6-sol" })) {
  const mock = t.mock.fn(decide);
  t.mock.method(Runner, "fromToml", () => ({ decide: mock }));
  let definition;
  extension({ registerVirtualModel: (value) => { definition = value; } });
  return { route: (req) => definition.route(req, context), mock };
}

test("keeps the planner until a successful edit, then keeps the implementation model", async (t) => {
  const { route, mock } = setup(t);
  const req = request();
  const planning = await route(req);
  assert.equal(planning.model.id, "gpt-5.6-sol");
  assert.equal(planning.thinkingLevel, "high");
  assert.deepEqual(planning.state, { phase: "planning", target: "complex" });
  assert.deepEqual(mock.mock.calls[0].arguments, ["switchyard/planner", "task", { signal: req.signal }]);

  const messages = [...req.messages, { role: "toolResult", toolName: "edit", isError: true }];
  assert.equal((await route(request({ state: planning.state, messages }))).model.id, "gpt-5.6-sol");
  messages[1].isError = false;
  const implementation = await route(request({ state: planning.state, messages }));
  assert.deepEqual(implementation.state, { phase: "implementation", target: "implementation" });
  const next = await route(request({ reason: "retry", state: implementation.state }));
  assert.equal(next.model.id, "gpt-5.6-luna");
  assert.equal(next.state, undefined);
  assert.equal(mock.mock.callCount(), 1);
});

test("uses the cheap model for direct requests and reuses an eligible previous planner", async (t) => {
  const { route, mock } = setup(t);
  assert.equal((await route(request({ reason: "direct" }))).model.id, "gpt-5.6-luna");
  const previous = { model: { provider: "openai-codex", id: "gpt-5.6-terra" } };
  assert.deepEqual((await route(request({ previous }))).state, { phase: "planning", target: "standard" });
  assert.equal(mock.mock.callCount(), 0);
});

test("routing failures use the default planner while cancellation propagates", async (t) => {
  let code = "ERR_ROUTING";
  const { route } = setup(t, async () => { throw Object.assign(new Error("test"), { code }); });
  assert.equal((await route(request())).model.id, "gpt-5.6-terra");
  code = "ABORT_ERR";
  await assert.rejects(route(request()), { code });
});

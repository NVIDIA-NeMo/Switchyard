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

function setup() {
  let definition;
  extension({ registerVirtualModel: (value) => { definition = value; } });
  return { definition, route: (req) => definition.route(req, context) };
}

function tool(id, name, args, text, isError = false) {
  return [
    { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }] },
    { role: "toolResult", toolCallId: id, toolName: name, content: [{ type: "text", text }], isError },
  ];
}

test("auto uses Luna, escalates to Sol on tool failure, and returns to Luna on recovery", async () => {
  const { route, definition } = setup();
  assert.deepEqual(definition.thinkingLevels, ["medium"]);
  const messages = request().messages;
  const initial = await route(request({ messages }));
  assert.equal(initial.model.id, "gpt-6-luna");
  assert.equal(initial.thinkingLevel, "medium");
  messages.push(...tool("failed", "bash", { command: "pytest" }, "out of memory", true));
  const escalated = await route(request({ reason: "continuation", messages }));
  assert.equal(escalated.model.id, "gpt-6.1-sol");
  for (let i = 0; i < 3; i++) messages.push(...tool(`write-${i}`, "write", { path: `file-${i}` }, "wrote file"));
  const recovered = await route(request({ reason: "retry", messages }));
  assert.equal(recovered.model.id, "gpt-6-luna");
});

test("passes text, tool arguments, results, and failure flags to Switchyard", async (t) => {
  const decide = t.mock.fn(async () => ({ target: "openrouter/vendor/model", model: "openrouter/vendor/model" }));
  t.mock.method(Runner, "fromToml", () => ({ decide }));
  const { route } = setup();
  const req = request({ messages: [
    { role: "system", content: "rules", sections: { extra: "more rules" } },
    { role: "user", content: [{ type: "text", text: "task" }, { type: "image", data: "image" }] },
    ...tool("call", "bash", { command: "pytest" }, "out of memory", true),
    { role: "assistant", content: [{ type: "thinking", thinking: "private" }] },
  ] });
  assert.deepEqual((await route(req)).model, { provider: "openrouter", id: "vendor/model" });
  assert.deepEqual(decide.mock.calls[0].arguments, ["switchyard/auto", [
    { role: "system", content: [{ type: "text", text: "rules" }, { type: "text", text: "more rules" }] },
    { role: "user", content: [{ type: "text", text: "task" }] },
    { role: "assistant", content: [{ type: "tool_call", id: "call", name: "bash", arguments: { command: "pytest" } }] },
    { role: "tool", content: [{ type: "tool_result", tool_call_id: "call", content: [{ type: "text", text: "out of memory" }], is_error: true }] },
    { role: "assistant", content: [] },
  ], { signal: req.signal }]);
});

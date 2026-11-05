// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { test } from "node:test";
import { Runner } from "../dist/index.js";

const request = { model: "coding", messages: [{ role: "user", content: [{ type: "text", text: "Summarize these notes." }] }] };
const stage = `type = "stage_router"
efficient_target = "fast"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.6
efficient_system_prompt = "Efficient role."
capable_system_prompt = "Capable role."`;
const judge = `type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
base_threshold = 0.5
fail_open = false`;

function config({ endpoint = "http://127.0.0.1:1", algorithm = stage, scope = "request", required = false } = {}) {
  const targets = [
    ["fast", "cloud/fast", []], ["capable", "cloud/capable", []],
    ["private_fast", "private/fast", ["private"]], ["private_capable", "private/capable", ["private"]],
    ["both_fast", "approved/fast", ["private", "customer_data"]],
    ["both_capable", "approved/capable", ["private", "customer_data"]],
    ["judge", "cloud/judge", []], ["approved_judge", "approved/judge", ["private", "customer_data"]],
  ].map(([name, id, labels]) => `[targets.${name}]
id = ${JSON.stringify(id)}
llm_client = "primary"
labels = ${JSON.stringify(labels)}
${name === "both_fast" ? 'system_prompt = "Use approved account data."' : ""}`).join("\n");
  return `schema_version = 1
[llm_clients.primary]
format = "openai_chat"
base_url = "${endpoint}/v1"
max_retries = 0
${targets}
[routes.coding]
id = "coding"
${algorithm}
[routes.coding.eligibility]
scope = "${scope}"
require_explicit_requirements = ${required}
target_alternatives = { fast = ["private_fast", "both_fast"], capable = ["private_capable", "both_capable"]${algorithm === judge ? ', judge = ["approved_judge"]' : ""} }
`;
}

test("Runner preserves Stage roles, approved fallback order and request rewrites", async () => {
  const runner = Runner.fromToml(config());
  for (const [labels, prefix] of [[[], ""], [["private"], "private_"], [["customer_data"], "both_"], [["private", "customer_data", "private"], "both_"]]) {
    const outcome = await runner.decide(request, { requiredLabels: labels });
    assert.deepEqual(outcome.selectedModelIds, [`${prefix}fast`, `${prefix}capable`]);
    assert.equal(outcome.request.model, `${prefix}fast`);
    assert.equal(outcome.metadata.algorithm, "stage_router");
    assert.equal(typeof outcome.metadata.outcomeId, "string");
    assert.equal(outcome.response, null);
    const fallback = outcome.candidateRequests[`${prefix}capable`];
    assert.equal(fallback.model, `${prefix}capable`);
    assert.equal(fallback.instructions[0].content[0].text, "Capable role.");
    assert.ok(!JSON.stringify(fallback).includes("Efficient role."));
    if (prefix === "both_") {
      assert.ok(outcome.request.instructions.some((instruction) => instruction.content.some((block) => block.text === "Use approved account data.")));
      assert.equal(outcome.fallbackRequest, null);
    }
  }
  const compacted = await runner.decide({
    ...request,
    messages: [{ role: "user", content: [{ type: "text", text: "[CONTEXT COMPACTION — REFERENCE ONLY] summary" }] }],
  }, { requiredLabels: [] });
  assert.equal(compacted.request.is_compaction, true);
  assert.equal(request.model, "coding");
  const noop = Runner.fromToml(`schema_version = 1
targets = {}
[routes.noop]
id = "noop"
type = "noop"`);
  for (const stream of [false, true]) {
    const outcome = await noop.decide({ model: "noop", stream });
    assert.equal(outcome.response.outputs[0].content[0].text, "OK");
  }
});

test("unspecified and explicitly empty requirements stay distinct across turns", async () => {
  const runner = Runner.fromToml(config({ scope: "task", required: true }));
  await assert.rejects(runner.decide(request, { taskId: "renewal" }), { code: "ERR_INVALID_REQUEST" });
  assert.equal((await runner.decide(request, { requiredLabels: [], taskId: "renewal" })).selectedModelIds[0], "fast");
  assert.equal((await runner.decide(request, { requiredLabels: ["private"], taskId: "renewal" })).selectedModelIds[0], "private_fast");
  assert.equal((await runner.decide(request, { requiredLabels: ["customer_data"], taskId: "renewal" })).selectedModelIds[0], "both_fast");
  assert.equal((await runner.decide(request, { requiredLabels: [], taskId: "renewal" })).selectedModelIds[0], "both_fast");
  await assert.rejects(runner.decide(request, { taskId: "renewal" }), { code: "ERR_INVALID_REQUEST" });
  assert.equal((await runner.decide(request, { requiredLabels: [], taskId: "other" })).selectedModelIds[0], "fast");
  for (const options of [{ requiredLabels: null }, { requiredLabels: [1] }, { requiredLabels: [""] }, { requiredLabels: [], taskId: "" }]) {
    await assert.rejects(runner.decide(request, options), { code: "ERR_INVALID_REQUEST" });
  }
});

test("Runner checks labels before any routing call and calls only the approved judge", async (t) => {
  const calls = [];
  const server = createServer(async (incoming, outgoing) => {
    const chunks = [];
    for await (const chunk of incoming) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    calls.push(body);
    outgoing.writeHead(200, { "content-type": "application/json" });
    outgoing.end(JSON.stringify({
      id: "fixture", object: "chat.completion", model: body.model,
      choices: [{ index: 0, message: { role: "assistant", content: JSON.stringify({ crux: "bounded task", primary_rule: "SUP-1", capability_boundary: "supported", p_solve: 0.9 }) }, finish_reason: "stop" }],
    }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise((resolve) => { server.closeAllConnections(); server.close(resolve); }));
  const runner = Runner.fromToml(config({ endpoint: `http://127.0.0.1:${server.address().port}`, algorithm: judge, required: true }));
  for (const options of [{}, { requiredLabels: ["unapproved"] }]) {
    await assert.rejects(runner.decide(request, options), { code: "ERR_INVALID_REQUEST" });
  }
  assert.equal(calls.length, 0);
  const outcome = await runner.decide(request, { requiredLabels: ["private", "customer_data"] });
  assert.equal(outcome.selectedModelIds[0], "both_fast");
  assert.deepEqual(calls.map((call) => call.model), ["approved/judge"]);
  assert.ok(!JSON.stringify(calls).includes("required_target_labels"));
});

test("provider-body fields cannot supply trusted requirements", async () => {
  const runner = Runner.fromToml(config({ required: true }));
  await assert.rejects(runner.decide({
    ...request,
    required_labels: [],
    metadata: { extra_metadata: { "switchyard.internal.required_target_labels": "[]" } },
  }), { code: "ERR_INVALID_REQUEST" });
  assert.throws(() => Runner.fromToml(config().replace('scope = "request"', 'scope = "invalid"')), { code: "ERR_CONFIG" });
});

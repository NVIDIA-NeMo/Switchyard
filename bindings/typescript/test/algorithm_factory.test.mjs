// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { test } from "node:test";
import { AlgorithmFactory } from "../dist/index.js";

const deployment = `
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.invalid/v1"
api_key_env = "SWITCHYARD_TS_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
[targets]
fast = { id = "upstream/fast", llm_client = "answers" }
capable = { id = "upstream/capable", llm_client = "answers" }
judge = { id = "upstream/judge", llm_client = "answers" }
[decision_clients.judge]
format = "openai_decisions"
endpoint = "https://example.invalid/v1/decisions"
api_key_env = "SWITCHYARD_TS_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
timeout_ms = 2000
[decision_targets.decision_judge]
id = "upstream/decision"
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
fail_open = false
classifier_target = "decision_judge"
strong_target = "capable"
weak_target = "fast"
[routes.classifier.decision]
cutoff = 0.7
candidates = { a = "capable", b = "fast" }
evidence = { description = "Compare both targets." }
[routes.escalation]
id = "public/escalate"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
escalation = { confirmations = 1 }
`;
const request = {
  model: "public/main",
  messages: [{ role: "user", content: [{ type: "text", text: "Fix this test." }] }],
};
const reply = (text) => ({
  outputs: [{ role: "assistant", content: [{ type: "text", text }], stop_reason: null }],
});
const routes = () => AlgorithmFactory.fromToml(deployment).build();

test("factory builds without credentials and preserves parent/subagent groups", async () => {
  assert.equal(process.env.SWITCHYARD_TS_FACTORY_TEST_KEY_THAT_IS_NOT_SET, undefined);
  const algorithms = routes();
  assert.ok(algorithms instanceof Map);
  assert.deepEqual([...algorithms.keys()], ["public/classify", "public/escalate", "public/main"]);
  const state = { host_marker: { string: "unchanged" } };
  const saved = structuredClone({ request, state });
  const [parent, child] = await Promise.all([
    algorithms.get("public/main").route(request, { state }),
    algorithms.get("public/main").route(request, { headers: { "X-Switchyard-Is-Subagent": "true" } }),
  ]);
  assert.deepEqual(parent.selectedModelIds, ["upstream/fast"]);
  assert.deepEqual(child.selectedModelIds, ["upstream/capable"]);
  assert.equal(parent.request.model, "upstream/fast");
  assert.deepEqual(parent.request.messages, request.messages);
  assert.deepEqual(parent.state, state);
  assert.equal(typeof parent.metadata.outcome_id, "string");
  assert.deepEqual({ request, state }, saved);
});

test("decision callbacks retain ordered fallbacks and the full outcome", async () => {
  const classifier = routes().get("public/classify");
  const outcome = await classifier.route(request, { decide: async (decision) => {
    assert.equal(decision.model, "upstream/decision");
    assert.deepEqual(decision.context.task, request.messages);
    assert.deepEqual(decision.context.evidence, { description: "Compare both targets." });
    return { answers: { route: { value: {
      type: "choice", data: { selected: "advantage", probabilities: { advantage: 0.9, no_advantage: 0.1 } },
    } } } };
  } });
  assert.deepEqual(outcome.selectedModelIds, ["upstream/capable", "upstream/fast"]);
  assert.equal(outcome.request.model, "upstream/capable");
  assert.equal(outcome.response, null);
  assert.equal(outcome.metadata.algorithm, "llm_task_classifier");
});

test("model callbacks return an answer produced during escalation", async () => {
  const calls = [];
  const answer = {
    ...reply("fixed"), id: "answer-1", model: "provider/reported-model", usage: { output_tokens: 2 },
  };
  const escalation = routes().get("public/escalate");
  const outcome = await escalation.route(request, { call: async (call) => {
    calls.push(call.models);
    assert.equal(call.request.model, call.models[0]);
    return {
      response: call.models[0] === "upstream/fast" ? answer : reply(JSON.stringify({
        escalate: false, category: "none", new_evidence: false, reason: "progressing",
      })),
      servedModel: call.models[0],
    };
  } });
  assert.deepEqual(calls, [["upstream/fast"], ["upstream/judge"]]);
  assert.deepEqual(outcome.selectedModelIds, ["upstream/fast"]);
  assert.equal(outcome.response.id, answer.id);
  assert.equal(outcome.response.model, answer.model);
  assert.deepEqual(outcome.response.outputs, answer.outputs);
  assert.equal(outcome.response.usage.output_tokens, 2);
  assert.deepEqual(outcome.metadata.evidence, { source: "escalation", verdict: "continue" });
});

test("missing callbacks, host failures, and model-call cancellation", { timeout: 5000 }, async () => {
  const algorithms = routes();
  const escalation = algorithms.get("public/escalate");
  await assert.rejects(escalation.route(request), { code: "ERR_CONFIG" });
  await assert.rejects(algorithms.get("public/classify").route(request), { code: "ERR_CONFIG" });
  const cause = new Error("provider unavailable");
  await assert.rejects(escalation.route(request, { call: async () => { throw cause; } }), {
    code: "ERR_ROUTING", cause,
  });
  const entered = Promise.withResolvers();
  const pending = Promise.withResolvers();
  const controller = new AbortController();
  const cancelled = assert.rejects(escalation.route(request, {
    signal: controller.signal,
    call: () => { entered.resolve(); return pending.promise; },
  }), { name: "AbortError", code: "ABORT_ERR" });
  await entered.promise;
  controller.abort();
  await cancelled;
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
  pending.resolve({ response: reply("late answer"), servedModel: "upstream/fast" });
});

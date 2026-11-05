// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { TranslationEngine, TranslationError, WireFormat } from "../dist/index.js";

const requests = {
  openai_chat: { model: "example", messages: [{ role: "user", content: "Hello" }], max_completion_tokens: 64 },
  openai_responses: { model: "example", input: "Hello", max_output_tokens: 64 },
  anthropic_messages: { model: "example", messages: [{ role: "user", content: "Hello" }], max_tokens: 64 },
};
const responses = {
  openai_chat: {
    id: "response-1", model: "example",
    choices: [{ message: { role: "assistant", content: "Hello" }, finish_reason: "stop" }],
  },
  openai_responses: {
    id: "response-1", model: "example", status: "completed",
    output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "Hello" }] }],
  },
  anthropic_messages: {
    id: "response-1", model: "example", type: "message", role: "assistant",
    content: [{ type: "text", text: "Hello" }], stop_reason: "end_turn",
  },
};

test("buffered translation and IR round trips cover the three Python formats", () => {
  const engine = new TranslationEngine();
  for (const format of Object.values(WireFormat)) {
    const body = { ...requests[format], provider_extension: { preserved: true } };
    const decoded = engine.decodeRequest(format, body);
    assert.deepEqual(Object.keys(decoded).sort(), ["diagnostics", "request"]);
    assert.deepEqual(decoded.diagnostics, []);
    const request = JSON.parse(JSON.stringify(decoded.request));
    request.model = "edited";
    assert.deepEqual(engine.encodeRequest(format, request), { body, diagnostics: [] });
    request.preservation.requests = {};
    assert.equal(engine.encodeRequest(format, request).body.model, "edited");

    const response = engine.decodeResponse(format, responses[format]);
    assert.deepEqual(Object.keys(response).sort(), ["diagnostics", "response"]);
    assert.deepEqual(response.diagnostics, []);
    for (const target of Object.values(WireFormat)) {
      const translated = engine.translateRequest(format, target, requests[format]);
      assert.deepEqual(translated.diagnostics, []);
      assert.equal(translated.body.model, "example");
      assert.equal(engine.decodeRequest(target, translated.body).request.messages[0].content[0].text, "Hello");
      for (const result of [
        engine.encodeResponse(target, JSON.parse(JSON.stringify(response.response))),
        engine.translateResponse(format, target, responses[format]),
      ]) {
        assert.deepEqual(Object.keys(result).sort(), ["body", "diagnostics"]);
        assert.deepEqual(result.diagnostics, []);
        assert.equal(result.body.model, "example");
        assert.equal(engine.decodeResponse(target, result.body).response.outputs[0].content[0].text, "Hello");
      }
    }
  }
});

test("response encoding restores request tool extensions only when supplied", () => {
  const engine = new TranslationEngine();
  const { request } = engine.decodeRequest(WireFormat.OPENAI_RESPONSES, {
    input: "Search documentation and list files",
    tools: [
      { type: "namespace", name: "mcp__docs", tools: [{ type: "function", name: "search", parameters: {} }] },
      { type: "custom", name: "exec", format: { type: "text" } },
    ],
  });
  const { response } = engine.decodeResponse(WireFormat.OPENAI_CHAT, {
    id: "response-1", model: "example", choices: [{ message: {
      role: "assistant", tool_calls: [
        { id: "call-1", type: "function", function: { name: "mcp__docs__search", arguments: '{"query":"streaming"}' } },
        { id: "call-2", type: "function", function: { name: "exec", arguments: '{"input":"ls -la"}' } },
      ],
    }, finish_reason: "tool_calls" }],
  });
  const plain = engine.encodeResponse(WireFormat.OPENAI_RESPONSES, response);
  assert.deepEqual(plain.body.output.map((call) => call.type), ["function_call", "function_call"]);
  assert.equal(plain.body.output[0].name, "mcp__docs__search");
  assert.equal(plain.body.output[0].namespace, undefined);
  for (const empty of [null, {}, { fields: {} }]) {
    assert.deepEqual(engine.encodeResponse(WireFormat.OPENAI_RESPONSES, response, empty), plain);
  }
  const restored = engine.encodeResponse(
    WireFormat.OPENAI_RESPONSES, response, JSON.parse(JSON.stringify(request.extensions)),
  );
  assert.deepEqual(restored.diagnostics, []);
  const [search, custom] = restored.body.output;
  assert.equal(search.name, "search");
  assert.equal(search.namespace, "mcp__docs");
  assert.equal(custom.type, "custom_tool_call");
  assert.equal(custom.name, "exec");
  assert.equal(custom.input, "ls -la");
  assert.equal(custom.arguments, undefined);
  assert.deepEqual(engine.encodeResponse(WireFormat.OPENAI_RESPONSES, response), plain);
});

test("diagnostics and structured translation errors match the Python boundary", () => {
  const engine = new TranslationEngine();
  assert.deepEqual(engine.translateRequest(WireFormat.OPENAI_CHAT, WireFormat.ANTHROPIC_MESSAGES, {
    model: "example", messages: [42],
  }).diagnostics, [{
    severity: "Warning", code: "lossy_conversion", message: "OpenAI message at index 0 is not an object",
    source: "openai_chat", target: "anthropic_messages", path: null,
  }]);
  for (const run of [
    () => engine.translateRequest("custom", WireFormat.OPENAI_CHAT, {}),
    () => engine.decodeResponse("custom", {}),
  ]) {
    assert.throws(run, (error) => {
      assert.ok(error instanceof TranslationError);
      assert.equal(error.kind, "Other");
      assert.deepEqual(error.details, { message: "no codec registered for custom" });
      return true;
    });
  }
  assert.throws(() => engine.translateRequest(WireFormat.OPENAI_CHAT, WireFormat.OPENAI_RESPONSES, {
    messages: {},
  }), (error) => {
    assert.ok(error instanceof TranslationError);
    assert.equal(error.message, "expected array at $.messages");
    assert.equal(error.kind, "InvalidType");
    assert.deepEqual(error.details, { path: "$.messages", expected: "array" });
    return true;
  });
  assert.throws(() => engine.decodeRequest(WireFormat.OPENAI_CHAT, "{}"), TypeError);
  assert.throws(() => engine.encodeRequest(WireFormat.OPENAI_CHAT, { messages: "not a list" }), { code: "ERR_INVALID_REQUEST" });
  assert.throws(() => engine.encodeResponse(WireFormat.OPENAI_CHAT, {}, { fields: 42 }), { code: "ERR_INVALID_REQUEST" });
});

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

test("buffered methods accept plain JSON objects across the three formats", () => {
  const engine = new TranslationEngine();
  const formats = Object.values(WireFormat);
  for (const [index, format] of formats.entries()) {
    const request = engine.decodeRequest(format, requests[format]);
    const response = engine.decodeResponse(format, responses[format]);
    assert.deepEqual([request.diagnostics, response.diagnostics], [[], []]);
    assert.deepEqual(engine.encodeRequest(format, JSON.parse(JSON.stringify(request.request))), {
      body: requests[format], diagnostics: [],
    });
    assert.deepEqual(engine.encodeResponse(format, JSON.parse(JSON.stringify(response.response))), {
      body: responses[format], diagnostics: [],
    });
    const target = formats[(index + 1) % formats.length];
    const translatedRequest = engine.translateRequest(format, target, requests[format]);
    const translatedResponse = engine.translateResponse(format, target, responses[format]);
    assert.deepEqual([translatedRequest.diagnostics, translatedResponse.diagnostics], [[], []]);
    assert.equal(engine.decodeRequest(target, translatedRequest.body).request.messages[0].content[0].text, "Hello");
    assert.equal(engine.decodeResponse(target, translatedResponse.body).response.outputs[0].content[0].text, "Hello");
  }
});

test("response encoding and translation restore request tool extensions only when supplied", () => {
  const engine = new TranslationEngine();
  const { request } = engine.decodeRequest(WireFormat.OPENAI_RESPONSES, {
    input: "Search documentation and list files",
    tools: [{ type: "namespace", name: "mcp__docs", tools: [{ type: "function", name: "search", parameters: {} }] }],
  });
  const wireResponse = {
    id: "response-1", model: "example", choices: [{ message: {
      role: "assistant", tool_calls: [
        { id: "call-1", type: "function", function: { name: "mcp__docs_sy_ns_search", arguments: '{"query":"streaming"}' } },
      ],
    }, finish_reason: "tool_calls" }],
  };
  const { response } = engine.decodeResponse(WireFormat.OPENAI_CHAT, wireResponse);
  for (const convert of [
    (...extensions) => engine.encodeResponse(WireFormat.OPENAI_RESPONSES, response, ...extensions),
    (...extensions) => engine.translateResponse(WireFormat.OPENAI_CHAT, WireFormat.OPENAI_RESPONSES, wireResponse, ...extensions),
  ]) {
    const plain = convert();
    assert.equal(plain.body.output[0].name, "mcp__docs_sy_ns_search");
    assert.equal(plain.body.output[0].namespace, undefined);
    const restored = convert(JSON.parse(JSON.stringify(request.extensions)));
    assert.deepEqual(restored.diagnostics, []);
    const [search] = restored.body.output;
    assert.equal(search.name, "search");
    assert.equal(search.namespace, "mcp__docs");
    for (const extensions of [undefined, null, {}]) assert.deepEqual(convert(extensions), plain);
    assert.deepEqual(convert(), plain);
  }
});

test("diagnostics and structured translation errors match the Python boundary", () => {
  const engine = new TranslationEngine();
  assert.deepEqual(engine.translateRequest(WireFormat.OPENAI_CHAT, WireFormat.ANTHROPIC_MESSAGES, {
    model: "example", messages: [42],
  }).diagnostics, [{
    severity: "Warning", code: "lossy_conversion", message: "OpenAI message at index 0 is not an object",
    source: "openai_chat", target: "anthropic_messages", path: null,
  }]);
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
});

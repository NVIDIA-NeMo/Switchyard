// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const vscode = require('vscode');
const { loadLibsy } = require('./router.js');
const { toSwitchyard } = require('./messages.js');

const FORMATS = [
  { label: 'OpenAI Chat Completions', value: 'openai_chat', example: 'https://api.example.com/v1/chat/completions' },
  { label: 'OpenAI Responses', value: 'openai_responses', example: 'https://api.example.com/v1/responses' },
  { label: 'Anthropic Messages', value: 'anthropic_messages', example: 'https://api.example.com/v1/messages' },
];
const MAX_OUTPUT_TOKENS = 8192;

function checkUrl(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    return 'Enter a full URL.';
  }
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && local)) return 'Use HTTPS, or HTTP on localhost.';
  return undefined;
}

function headers(format, apiKey) {
  const result = { 'Content-Type': 'application/json' };
  if (format === 'anthropic_messages') {
    result['anthropic-version'] = '2023-06-01';
    if (apiKey) result['x-api-key'] = apiKey;
  } else if (apiKey) {
    result.Authorization = `Bearer ${apiKey}`;
  }
  return result;
}

function toVscodePart(part) {
  if (part.type === 'text') return new vscode.LanguageModelTextPart(part.text);
  if (part.type === 'tool_call') {
    // Responses tool calls keep their arguments as a JSON string.
    const input = typeof part.arguments === 'string' ? JSON.parse(part.arguments) : part.arguments;
    return new vscode.LanguageModelToolCallPart(part.id, part.name, input);
  }
  return undefined;
}

// Wraps an endpoint in the same interface as a Copilot model:
// name, maxInputTokens, countTokens and sendRequest.
function createModel(config, apiKey) {
  return {
    id: `custom:${config.id}`,
    vendor: 'custom',
    name: config.name,
    maxInputTokens: config.maxInputTokens,

    // There is no tokenizer for custom endpoints. One token per byte overcounts on purpose.
    async countTokens(value) {
      return Buffer.byteLength(typeof value === 'string' ? value : JSON.stringify(value));
    },

    async sendRequest(messages, options = {}, token) {
      const { TranslationEngine } = await loadLibsy();
      const engine = new TranslationEngine();
      const request = {
        model: config.model,
        messages: toSwitchyard(messages),
        stream: false,
        output: { max_output_tokens: MAX_OUTPUT_TOKENS },
        tools: (options.tools ?? []).map(tool => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema ?? { type: 'object', properties: {} },
        })),
      };
      if (request.tools.length > 0) {
        const required = options.toolMode === vscode.LanguageModelChatToolMode.Required;
        request.tool_choice = { type: required ? 'required' : 'auto' };
      }

      const controller = new AbortController();
      const cancel = token?.onCancellationRequested(() => controller.abort());
      try {
        const response = await fetch(config.url, {
          method: 'POST',
          headers: headers(config.format, apiKey),
          body: JSON.stringify(engine.encodeRequest(config.format, request).body),
          signal: controller.signal,
        });
        if (!response.ok) {
          const reason = (await response.text()).slice(0, 200);
          throw new Error(`${config.name} returned HTTP ${response.status}: ${reason}`);
        }
        const decoded = engine.decodeResponse(config.format, await response.json());
        const parts = decoded.response.outputs
          .flatMap(output => output.content)
          .map(toVscodePart)
          .filter(Boolean);
        return { stream: (async function* () { yield* parts; })() };
      } finally {
        cancel?.dispose();
      }
    },
  };
}

// Sends one short request so that a wrong URL, model or key fails before saving.
async function checkModel(config, apiKey, token) {
  const model = createModel(config, apiKey);
  const response = await model.sendRequest([vscode.LanguageModelChatMessage.User('Reply with OK.')], {}, token);
  for await (const part of response.stream) {
    if (part instanceof vscode.LanguageModelTextPart && part.value.trim()) return;
  }
  throw new Error(`${config.name} returned no text.`);
}

module.exports = { FORMATS, MAX_OUTPUT_TOKENS, checkModel, checkUrl, createModel };

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const vscode = require('vscode');

const roles = {
  [vscode.LanguageModelChatMessageRole.User]: 'user',
  [vscode.LanguageModelChatMessageRole.Assistant]: 'assistant',
  // The System role exists at runtime but is not part of the stable API.
  3: 'system',
};

function isForwarded(part) {
  return part instanceof vscode.LanguageModelTextPart
    || part instanceof vscode.LanguageModelToolCallPart
    || part instanceof vscode.LanguageModelToolResultPart;
}

function toSwitchyardPart(part) {
  if (part instanceof vscode.LanguageModelTextPart) return [{ type: 'text', text: part.value }];
  if (part instanceof vscode.LanguageModelToolCallPart) {
    return [{ type: 'tool_call', id: part.callId, name: part.name, arguments: part.input }];
  }
  if (part instanceof vscode.LanguageModelToolResultPart) {
    return [{ type: 'tool_result', tool_call_id: part.callId, content: part.content.flatMap(toSwitchyardPart), is_error: part.isError ?? false }];
  }
  // Reasoning and Copilot metadata parts are model-specific and are dropped.
  return [];
}

// Converts VS Code chat messages to Switchyard's normalized messages.
function toSwitchyard(messages) {
  return messages
    .map(message => ({ role: roles[message.role], content: message.content.flatMap(toSwitchyardPart) }))
    .filter(message => message.content.length > 0);
}

// Prepares messages for a Copilot model. The stable API cannot send system
// messages, so they are sent as user messages.
function toCopilot(messages) {
  return messages
    .map(message => {
      const role = message.role === vscode.LanguageModelChatMessageRole.Assistant
        ? vscode.LanguageModelChatMessageRole.Assistant
        : vscode.LanguageModelChatMessageRole.User;
      return new vscode.LanguageModelChatMessage(role, message.content.filter(isForwarded), message.name);
    })
    .filter(message => message.content.length > 0);
}

module.exports = { toSwitchyard, toCopilot };

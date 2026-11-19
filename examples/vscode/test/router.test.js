// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { selectTier } = require('../src/router.js');

const user = { role: 'user', content: [{ type: 'text', text: 'Fix the failing test.' }] };

function tool(id, name, text) {
  return [
    { role: 'assistant', content: [{ type: 'tool_call', id, name, arguments: {} }] },
    { role: 'user', content: [{ type: 'tool_result', tool_call_id: id, content: [{ type: 'text', text }], is_error: false }] },
  ];
}

test('a new task starts on the efficient model', async () => {
  assert.equal(await selectTier(0.5, [user]), 'efficient');
});

test('a tool failure escalates and holds the capable model for two requests in the session', async () => {
  const messages = [user, ...tool('run', 'run_in_terminal', 'FATAL: out of memory')];
  assert.equal(await selectTier(0.5, messages, 'chat-1'), 'capable');

  for (let i = 0; i < 3; i++) messages.push(...tool(`write-${i}`, 'create_file', 'wrote file'));
  assert.equal(await selectTier(0.5, messages, 'chat-2'), 'efficient');
  for (const expected of ['capable', 'capable', 'efficient']) {
    assert.equal(await selectTier(0.5, messages, 'chat-1'), expected);
  }
});

test('the threshold accepts whole numbers', async () => {
  assert.equal(await selectTier(1, [user]), 'efficient');
});

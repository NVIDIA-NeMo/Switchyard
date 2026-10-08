// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
function editor(invoke = () => new Promise(() => {})) {
  const context = vm.createContext({
    window: {__TAURI__: {core: {invoke}}},
    document: {querySelector: () => ({}), querySelectorAll: () => []},
    setTimeout: () => {},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8'), context);
  return context;
}
// The classifier occupies position zero, so copying by position would replace the capable model.
test('algorithm switches preserve model roles instead of positions', () => {
  const context = editor();
  const result = vm.runInContext(`
    const route = {kind:'composite',choices:[{model:'judge'},{model:'capable'},{model:'efficient'}]};
    const draft = {algorithm:'stage_router',choices:route.choices,memory:new Map([['Judge',route.choices[0]],['Capable',route.choices[1]],['Efficient',route.choices[2]]])};
    [roleChoice(draft,route,{tier:'Capable'},0).model,roleChoice(draft,route,{tier:'Efficient'},1).model];
  `, context);
  assert.deepEqual(Array.from(result), ['capable', 'efficient']);
});
// Null tiers must not combine distinct random targets into one remembered model.
test('random roles without tiers keep distinct positional choices', () => {
  const context = editor();
  const result = vm.runInContext(`
    const route = {kind:'random',choices:[{model:'first'},{model:'second'},{model:'third'},{model:'fourth'}]};
    const draft = {algorithm:'random',choices:route.choices,memory:new Map()};
    [roleChoice(draft,route,{tier:null},2).model,roleChoice(draft,route,{tier:null},3).model];
  `, context);
  assert.deepEqual(Array.from(result), ['third', 'fourth']);
});
// A successful editor redraw must preserve the distinction between a saved route and a failed restart.
test('editor redraw retains the saved route and restart result', async () => {
  const context = editor((command, args) => command === 'snapshot'
    ? new Promise(() => {})
    : Promise.resolve({message: args.action.kind === 'editor' ? 'Single model.' : 'Saved the route. Restart failed.'}));
  const result = await vm.runInContext(`(async () => {
    busy = false;
    await run({kind:'apply'}, false);
    await run({kind:'editor'}, false);
    return status.textContent;
  })()`, context);
  assert.equal(result, 'Saved the route. Restart failed.');
});

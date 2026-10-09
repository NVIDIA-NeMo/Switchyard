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

test('reopening an edited route preserves untiered model choices', () => {
  const context = editor();
  const result = vm.runInContext(`
    const route = {kind:'passthrough',choices:[{model:'original'}]};
    const draft = {algorithm:'random',loadedAlgorithm:'random',choices:[{model:'edited-first'},{model:'edited-second'}],memory:new Map()};
    [roleChoice(draft,route,{tier:null},0).model,roleChoice(draft,route,{tier:null},1).model];
  `, context);
  assert.deepEqual(Array.from(result), ['edited-first', 'edited-second']);
});

// Model totals combine answer and classifier calls; session counts use only the chosen period.
test('usage totals retain cached tokens, routing overhead, and session dates', () => {
  const context = editor();
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/analytics.js'), 'utf8'), context);
  const result = vm.runInContext(`(() => {
    const row=modelRows({routed:{model:{requests:2,input:100,cached_input:40,output:20}},classifier:{model:{requests:1,input:10,cached_input:0,output:2}}})[0];
    const entries=[
      {ts:new Date(2026,8,1,12).toISOString(),session_id:'s',prompt_tokens:90,cached_tokens:30,completion_tokens:10},
      {ts:new Date(2026,8,28,12).toISOString(),session_id:'s',prompt_tokens:50,cached_tokens:20,completion_tokens:5},
      {ts:new Date(2026,8,29,0).toISOString(),session_id:null,prompt_tokens:1,cached_tokens:0,completion_tokens:1},
    ];
    return [row.requests,tokenTotal(row),sessionRows(entries,'today','2026-09-28')[0].requests,sessionRows(entries,'all','2026-09-28').length];
  })()`, context);
  assert.deepEqual(Array.from(result), [3,172,1,2]);
});

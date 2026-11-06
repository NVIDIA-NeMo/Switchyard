// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
function editor(invoke = () => new Promise(() => {})) {
  const elements = new Map();
  const element = selector => {
    if(!elements.has(selector))elements.set(selector,{dataset:{},append(){},querySelectorAll(){return [];},setAttribute(){},focus(){this.focused=true;}});
    return elements.get(selector);
  };
  const context = vm.createContext({
    window: {__TAURI__: {core: {invoke}}},
    document: {createElement:()=>({append(){}}),querySelector: element, querySelectorAll: () => []},
    setTimeout: () => {},structuredClone,
  });
  const source=fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8');
  // The fixture loads the real functions without starting the app's initial refresh.
  vm.runInContext(source.slice(0,source.indexOf("\ncontent.append(node('p','Loading Switchyard…'")), context);
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
// A successful editor redraw must preserve the distinction between a saved route and a failed restart.
test('editor redraw retains the saved route and restart result', async () => {
  const context = editor((command, args) => command === 'snapshot'
    ? new Promise(() => {})
    : Promise.resolve({message: args.action.kind === 'editor' ? 'Single model.' : 'Saved the route. Restart failed.',data:{warning:args.action.kind!=='editor'}}));
  const result = await vm.runInContext(`(async () => {
    busy = false;
    await run({kind:'apply'}, false);
    await run({kind:'editor'}, false);
    return status.textContent;
  })()`, context);
  assert.equal(result, 'Attention: Saved the route. Restart failed.');
});

// Navigation during a pending Usage refresh must not restore Usage's scroll position on Settings.
test('a pending refresh preserves the destination page scroll position', async () => {
  let release;
  const context=editor(()=>new Promise(resolve=>{release=resolve;}));
  vm.runInContext(`
    snapshot={};page='usage';scroller.scrollTop=200;
    scroller.scrollTo=(x,y)=>{scroller.scrollTop=y;};
    capturePageState=()=>({scroll:scroller.scrollTop});
    restorePageState=saved=>scroller.scrollTo(0,saved.scroll);
    render=()=>{};
    const pendingRefresh=refresh();
  `,context);
  await new Promise(setImmediate);
  vm.runInContext("navigate('settings')",context);
  release({errors:[]});
  await vm.runInContext('pendingRefresh',context);
  assert.equal(vm.runInContext('page',context),'settings');
  assert.equal(vm.runInContext('scroller.scrollTop',context),0);
});

test('route creation keeps pending model lists with their endpoint', async () => {
  const context=editor();
  vm.runInContext(`
    const controls=new Map();let load,resolveModels;
    node=()=>({dataset:{},isConnected:true,append(){},replaceChildren(){}});
    combobox=(options,value)=>({value,setOptions(options){this.options=options;}});
    field=(parent,label,control)=>{controls.set(label,control);return control;};
    modelField=(parent,label,client,value)=>{const control=combobox([],value);controls.set(label,control);return control;};
    actions=()=>({});button=(parent,label,callback)=>{load=callback;return {};};
    snapshot={settings:{config_file:'fixture'},clients:[{name:'A',host:'a'},{name:'B',host:'b'}],algorithms:[{kind:'passthrough',title:'Single',summary:'Single',roles:[{label:'Model'}]}]};
    readAction=()=>new Promise(resolve=>{resolveModels=resolve;});
    routeModels({append(){}});
    const pending=load();
    controls.get('Endpoint').value='B';controls.get('Endpoint').onchange();
    resolveModels({data:{models:['A/model'],source:'endpoint'}});
  `,context);
  await vm.runInContext('pending',context);
  const result=vm.runInContext(`[lists.get(modelListKey('A'))?.models,lists.has(modelListKey('B')),controls.get('Model').options]`,context);
  assert.deepEqual(JSON.parse(JSON.stringify(result)),[['A/model'],false,[]]);
});

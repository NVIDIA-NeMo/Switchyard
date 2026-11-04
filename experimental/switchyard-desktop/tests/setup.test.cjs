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
// Only the first snapshot selects setup; a later refresh must preserve the user's navigation.
test('first launch opens setup and retained files skip it after reinstall', async () => {
  for(const needsSetup of [true,false]){
    const context=editor(()=>Promise.resolve({needs_setup:needsSetup,errors:[]}));
    vm.runInContext(`render=()=>{};showPageErrors=()=>{};showFreshness=()=>{};capturePageState=()=>({});restorePageState=()=>{};`,context);
    await vm.runInContext('refresh()',context);
    assert.equal(vm.runInContext('page',context),needsSetup?'setup':'overview');
    vm.runInContext("page='settings'",context);
    await vm.runInContext('refresh()',context);
    assert.equal(vm.runInContext('page',context),'settings');
  }
});

// The real form handlers submit the selected files through the shared controller action.
test('setup submits selected files and Settings reuses the same form', async () => {
  const context=editor();
  const result=await vm.runInContext(`(async()=>{
    const controls=new Map(),handlers=new Map(),saved=[];
    node=()=>({append(){}});card=()=>({append(){}});actions=()=>({});
    input=value=>({value});combobox=(options,value)=>({value});
    field=(parent,label,control)=>{controls.set(label,control);return control;};
    button=()=>({});mutation=(parent,label,callback)=>{handlers.set(label,callback);return {};};
    fieldError=()=>true;confirmAction=(title,message,label,callback)=>callback();navigate=()=>{};
    run=async action=>{saved.push(action);return {};};
    snapshot={needs_setup:true,settings:{path:'/tmp/desktop.toml',server_url:'http://127.0.0.1:4123',config_file:'/tmp/default.toml',routing_log:'/tmp/routing.jsonl',launchd_label:'com.nvidia.switchyard.server',baseline_model:'baseline',refresh_seconds:30}};
    renderConfiguration(true);
    controls.get('Server config file').value='/tmp/chosen.toml';controls.get('Server config file').oninput();
    controls.get('Server configuration').value='existing';controls.get('Server configuration').onchange();
    await handlers.get('Save setup')();await Promise.resolve();
    renderConfiguration(false);
    return {saved,settingsHasSave:handlers.has('Save configuration')};
  })()`,context);
  const plain=JSON.parse(JSON.stringify(result));
  assert.equal(plain.saved.length,1);
  assert.equal(plain.saved[0].kind,'save_configuration');
  assert.equal(plain.saved[0].config.config_file,'/tmp/chosen.toml');
  assert.equal(plain.saved[0].create_server,false);
  assert.equal(plain.settingsHasSave,true);
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('newer native preferences and OS changes survive older appearance replies', async () => {
  // Deferred replies let this test deliver native changes before an older request completes.
  const listeners = {}, pending = [], errors = {textContent:''};
  const document = {documentElement:{dataset:{}},body:{dataset:{}},getElementById:()=>errors};
  const window = {
    addEventListener:(name,callback)=>{listeners[name]=callback;}, dispatchEvent:()=>{},
    __TAURI__:{core:{invoke:command=>command==='accessibility'?Promise.resolve({native_material:true,reduce_transparency:false}):new Promise(resolve=>pending.push(resolve))}},
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../ui/appearance.js'),'utf8'), {
    window, document, matchMedia:()=>({matches:true,addEventListener:()=>{}}),
    CustomEvent:class {constructor(name,value){this.detail=value?.detail;}},
  });
  listeners['switchyard-appearance']({detail:{mode:'canary',system:'Dark'}});
  pending.shift()({mode:'system',system:'Dark'});
  await new Promise(setImmediate);
  assert.equal(window.switchyardAppearance.mode,'canary');
  const selection = window.switchyardAppearance.choose('system');
  listeners['switchyard-system-theme']({detail:'Light'});
  pending.shift()({mode:'system',system:'Dark'});
  await selection;
  assert.equal(document.documentElement.dataset.theme,'light');
  const previous = window.switchyardAppearance.choose('light');
  listeners['switchyard-appearance']({detail:{mode:'canary',system:'Light'}});
  pending.shift()({mode:'light',system:'Light'});
  await previous;
  assert.equal(window.switchyardAppearance.mode,'canary');
});

test('native accessibility changes switch material exposure live, pause while hidden, and failures restore opaque chrome', async () => {
  let check, hidden=false, state={native_material:true,reduce_transparency:false,reduce_motion:false};
  const listeners={}, documentListeners={}, root={dataset:{}};
  const window={addEventListener:(event,fn)=>{listeners[event]=fn;},dispatchEvent:()=>{},
    __TAURI__:{core:{invoke:async command=>{if(command==='appearance')return {mode:'system',system:'Light'};if(state instanceof Error)throw state;return state;}}}};
  const document={documentElement:root,body:{dataset:{}},getElementById:()=>null,get hidden(){return hidden;},
    addEventListener:(event,fn)=>{documentListeners[event]=fn;}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../ui/appearance.js'),'utf8'),{
    window,document,matchMedia:()=>({matches:false,addEventListener:()=>{}}),
    setInterval:fn=>{check=fn;},clearInterval:()=>{check=undefined;},CustomEvent:class{},
  });
  await new Promise(setImmediate);
  assert.equal(root.dataset.nativeMaterial,'true');
  assert.equal(root.dataset.reduceTransparency,'false');
  state={native_material:true,reduce_transparency:true,reduce_motion:true};
  await check();
  assert.equal(root.dataset.reduceTransparency,'true');
  assert.equal(root.dataset.reduceMotion,'true');
  hidden=true;documentListeners.visibilitychange();
  assert.equal(check,undefined);
  state={native_material:true,reduce_transparency:false,reduce_motion:false};
  hidden=false;documentListeners.visibilitychange();
  await new Promise(setImmediate);
  assert.equal(root.dataset.reduceTransparency,'false');
  assert.equal(typeof check,'function');
  state=new Error('Native check failed');
  await listeners.focus();
  assert.equal(root.dataset.reduceTransparency,'true');
});

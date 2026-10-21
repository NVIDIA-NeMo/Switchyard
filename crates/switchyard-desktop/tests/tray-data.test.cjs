// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const context=vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/tray-usage.js'),'utf8'),context);
const now=Date.parse('2026-10-09T12:30:00Z');
function entry(age,model,session_id,tier='') {return {ts:new Date(now-age*3600000).toISOString(),model,session_id,tier,prompt_tokens:100,cached_tokens:60,cache_creation_tokens:20,completion_tokens:10};}
function usage(entries) {context.snapshot={entries,limited:true,skipped:2};context.now=now;return JSON.parse(vm.runInContext('JSON.stringify(buildTrayUsage(snapshot.entries,now))',context));}
test('rolling usage keeps exact bounds and counts cached input and cache writes once',()=>{
  const result=usage([entry(24,'old'),entry(24.1,'older'),entry(-1,'future'),entry(23.9,'first'),entry(0,'last'),entry(1,'judge','s','classifier')]);
  assert.deepEqual(result.models.map(m=>[m.id,m.tokens,m.input,m.cached,m.cache_write,m.output]),[['first',110,20,60,20,10],['last',110,20,60,20,10]]);
  assert.equal(result.rows[0].models.first.tokens,110);assert.equal(result.rows[23].models.last.calls,1);
  assert.equal(result.routingTokens,110);assert.equal(result.unknownSessionCalls,2);
});
test('switches follow chronological answer calls within each session',()=>{
  const result=usage([entry(1,'B','one'),entry(2,'C','two'),entry(3,'judge','one','classifier'),entry(4,'A','one'),entry(5,'B','two'),entry(6,'anonymous'),entry(7,'other',' '),entry(25,'Z','one')]);
  assert.deepEqual(result.switches.map(s=>[s.session_id,s.from,s.to]),[['one','Z','A'],['two','B','C'],['one','A','B']]);
  assert.deepEqual(result.rows.flatMap(b=>b.switches).map(s=>s.to),['A','C','B']);
  assert.equal(result.models.some(m=>m.id==='judge'),false);
  assert.equal(usage([]).rows.length,24);assert.equal(usage([]).switches.length,0);
});
test('timestamp ties preserve log order and raw session IDs remain distinct',()=>{
  const result=usage([entry(2,'A','session'),entry(2,'B','session '),entry(1,'C','session'),entry(1,'D','session'),{...entry(0,'invalid','session'),ts:'invalid' }]);
  assert.deepEqual(result.switches.map(s=>[s.session_id,s.from,s.to]),[['session','A','C'],['session','C','D']]);
  assert.equal(result.models.some(m=>m.id==='invalid'),false);
  assert.equal(result.models.reduce((sum,m)=>sum+m.calls,0),4);
});

test('the popup renders a production-shaped snapshot and its selected hour',async()=>{
  // The DOM fixture runs the popup's real scripts so mismatched usage fields fail during rendering.
  const nodes=new Map(),themes=['system','canary','light'].map(theme=>({dataset:{theme},setAttribute(){}}));
  function node(){return {textContent:'',hidden:false,children:[],style:{},dataset:{},classList:{toggle(){}},setAttribute(){},replaceChildren(...children){this.children=children;},append(...children){this.children.push(...children);}};}
  const document={
    getElementById(id){if(id==='chart-cursor')return null;if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},
    querySelectorAll(selector){return selector==='.themes button'?themes:[];},
    createElement:node,addEventListener(){},hasFocus:()=>false,
  };
  const snapshot={entries:[{ts:new Date().toISOString(),model:'fixture/model',session_id:'fixture/session',prompt_tokens:100,cached_tokens:60,completion_tokens:10}],metrics:{running:true},data_state:{health:{state:'running'},usage:{state:'available'},history:{state:'available'}},errors:[],limited:false,skipped:0};
  const popup=vm.createContext({document,window:{__TAURI__:{core:{invoke:async()=>snapshot}},switchyardAppearance:{mode:'system',choose(){}},addEventListener(){}},clearTimeout(){},setTimeout(){},modelColor:()=> '#abcdef'});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/tray-usage.js'),'utf8'),popup);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/tray-renderers.js'),'utf8'),popup);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ui/tray.js'),'utf8'),popup);
  popup.getComputedStyle=()=>({getPropertyValue:()=> '#abcdef'});
  await vm.runInContext('refresh()',popup);
  assert.equal(nodes.get('error').hidden,true,nodes.get('error').textContent);
  assert.equal(nodes.get('model-count').textContent,1);
  assert.match(nodes.get('hour-models').children[0].textContent,/fixture\/model · 110 tokens · 1 calls/);
  snapshot.data_state.history.state='unavailable';snapshot.entries=[];
  await vm.runInContext('refresh()',popup);
  assert.match(nodes.get('connection').textContent,/usage unavailable/);
  assert.equal(nodes.get('error').hidden,false);
  snapshot.data_state.history.state='available';snapshot.data_state.usage.state='unavailable';
  await vm.runInContext('refresh()',popup);
  assert.doesNotMatch(nodes.get('connection').textContent,/usage unavailable/);
  assert.equal(nodes.get('error').hidden,true);
});

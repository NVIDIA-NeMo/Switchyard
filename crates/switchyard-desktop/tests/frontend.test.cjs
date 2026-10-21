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

// Model totals combine answer and classifier calls; session counts use only the chosen period.
test('usage totals retain cached tokens, routing overhead, and session dates', () => {
  const context = editor();
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/analytics.js'), 'utf8'), context);
  const result = vm.runInContext(`(() => {
    const row=modelRows({routed:{model:{requests:2,input:100,cached_input:40,cache_write:5,output:20}},classifier:{model:{requests:1,input:10,cached_input:0,cache_write:0,output:2}}})[0];
    const entries=[
      {ts:new Date(2026,8,1,12).toISOString(),session_id:'s',prompt_tokens:90,cached_tokens:30,completion_tokens:10},
      {ts:new Date(2026,8,28,12).toISOString(),session_id:'s',prompt_tokens:50,cached_tokens:20,cache_creation_tokens:10,completion_tokens:5},
      {ts:new Date(2026,8,29,0).toISOString(),session_id:null,prompt_tokens:1,cached_tokens:0,completion_tokens:1},
    ];
    const today=sessionRows(entries,'today','2026-09-28')[0];
    return [row.requests,tokenTotal(row),today.requests,today.input,today.cache_write,sessionRows(entries,'all','2026-09-28').length];
  })()`, context);
  assert.deepEqual(Array.from(result), [3,177,1,20,10,2]);
});

// Background refresh must not recreate a dismissed action result.
test('action notifications stay dismissed through refresh', async () => {
  const context = editor(command => command === 'snapshot' ? new Promise(() => {}) : Promise.resolve({message:'Removed the route. Restarted Switchyard.'}));
  const result = await vm.runInContext(`(async () => {
    busy=false;
    await run({kind:'remove'},false);
    const shown=[notification.hidden,notification.dataset.tone,status.textContent];
    dismissNotification();
    return [...shown,notification.hidden,status.textContent];
  })()`, context);
  assert.deepEqual(Array.from(result), [false,'success','✓ Removed the route. Restarted Switchyard.',true,'']);
  const refreshing = editor(() => Promise.resolve({errors:['Server is unavailable.']}));
  const afterRefresh = await vm.runInContext(`(async () => {
    render=()=>{};
    busy=false;reading=false;
    notify('Removed the route.');
    dismissNotification();
    await refresh();
    return [notification.hidden,status.textContent,pageErrors.textContent];
  })()`, refreshing);
  assert.deepEqual(Array.from(afterRefresh), [true,'','Server is unavailable.']);
});

// A saved config with a failed restart needs a warning rather than a save error.
test('saved edits report restart warnings separately from action errors', async () => {
  const context = editor(() => Promise.resolve({message:'Removed the route. Switchyard could not restart.',data:{warning:true,details:['Backup: config.backup']}}));
  const warning = await vm.runInContext(`(async () => {
    busy=false;
    await run({kind:'remove'},false);
    return [notification.dataset.tone,document.querySelector('#notification-details').hidden,document.querySelector('#notification-detail-text').textContent];
  })()`,context);
  assert.deepEqual(Array.from(warning), ['warning',false,'Backup: config.backup']);
  const rejected = editor(() => Promise.reject('Invalid config.'));
  const failure = await vm.runInContext(`(async () => {
    busy=false;
    await run({kind:'apply'},false);
    return [notification.dataset.tone,status.textContent];
  })()`,rejected);
  assert.deepEqual(Array.from(failure), ['error','Error: Invalid config.']);
});

// A disabled Refresh button cannot receive focus after dismissal.
test('dismissal keeps keyboard focus reachable during an operation', () => {
  const context = editor();
  const result = vm.runInContext(`(() => {
    document.querySelector('#refresh').disabled=true;
    dismissNotification();
    return [document.querySelector('#refresh').focused||false,document.querySelector('#title').focused||false];
  })()`, context);
  assert.deepEqual(Array.from(result), [false,true]);
});

// Recovery must clear the page error while keeping the action notification dismissed.
test('route-page polling clears a recovered background error without recreating feedback', async () => {
  let calls=0;
  const context = editor(() => {calls++;return Promise.resolve({errors:[]});});
  const result = await vm.runInContext(`(async () => {
    page='routes';busy=false;reading=false;
    const elementsForPoll=document.querySelector;
    document.querySelector=selector=>selector==='[role="dialog"]'?null:elementsForPoll(selector);
    showPageErrors('Server is unavailable.');
    notify('Removed the route.');dismissNotification();
    await poll();
    return [pageErrors.hidden,pageErrors.textContent,notification.hidden,status.textContent];
  })()`, context);
  assert.deepEqual(Array.from(result), [true,'',true,'']);
});

// Cached input is part of prompt tokens; equal timestamps use stable log positions.
test('Usage sorts by time, model, and total tokens with deterministic ties', () => {
  const context = editor();
  const result = vm.runInContext(`(() => {
    const calls=[
      {id:'old',ts:'2026-10-08T10:00:00Z',model:'B',prompt_tokens:200,cached_tokens:0,completion_tokens:5},
      {id:'first',ts:'2026-10-09T10:00:00Z',model:'A',prompt_tokens:100,cached_tokens:100,completion_tokens:10},
      {id:'last',ts:'2026-10-09T10:00:00Z',model:'A',prompt_tokens:100,cached_tokens:100,completion_tokens:10},
      {id:'unknown',ts:'invalid',model:'C',prompt_tokens:0,cached_tokens:0,completion_tokens:0},
    ];
    return ['newest','oldest','model','tokens'].map(order=>sortedCalls(calls,order).map(e=>e.id));
  })()`, context);
  assert.deepEqual(JSON.parse(JSON.stringify(result)), [
    ['last','first','old','unknown'],['old','last','first','unknown'],
    ['last','first','old','unknown'],['old','last','first','unknown'],
  ]);
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

test('chart themes preserve recorded names that match retired color literals', () => {
  // This model name matched an old palette color and was rewritten by recursive color replacement.
  const context=editor();
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/analytics.js'), 'utf8'),context);
  const result=vm.runInContext(`(() => {
    getComputedStyle=()=>({getPropertyValue:name=>({'--muted':'#566173','--control':'#FFFCF5','--line':'#D4D0C5'}[name])});
    node=()=>({style:{},setAttribute(){}});
    let captured;
    window.echarts={init:()=>({setOption:options=>{captured=options;}})};
    ResizeObserver=class {observe(){}};
    mountChart({append(){}},{series:[{name:'#f2f4f8',data:[42]}]},200,'Usage');
    return [captured.series[0].name,captured.textStyle.color,captured.tooltip.backgroundColor,captured.tooltip.textStyle.color,captured.tooltip.renderMode];
  })()`,context);
  assert.deepEqual(Array.from(result),['#f2f4f8','#566173','#FFFCF5','#566173','richText']);
});

// A search must match every word, rank contiguous text above scattered letters, and match option details.
test('dropdown search requires every word and ranks contiguous matches first', () => {
  const context=vm.createContext({});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../ui/combobox.js'), 'utf8'), context);
  const result=vm.runInContext(`(() => {
    const options=[['a','gpt-6-luna','gateway'],['b','claude-sonnet-5-5','gateway'],['c','gpt-6.1-sol','openai']].map(comboboxOption);
    return ['sol','gpt sol','g6l','luna gateway','xyz'].map(query=>rankOptions(options,query).map(entry=>entry.option.value).join(''));
  })()`, context);
  assert.deepEqual(Array.from(result), ['c','c','ac','a','']);
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

// Settings must describe bundled rates when the snapshot has no price override.
test('pricing preferences distinguish bundled rates from a custom table', () => {
  const context=editor();
  const bundled=vm.runInContext(`
    const pricingText=[];
    card=()=>({append:value=>pricingText.push(value)});
    node=(tag,text)=>text;
    input=value=>({value});
    field=(parent,label,value)=>value;
    actions=()=>({});
    button=(parent,label)=>{pricingText.push(label);return {};};
    mutation=()=>{};
    snapshot={baseline_model:'gpt-6-sol',refresh_seconds:30,settings:{prices_file:null}};
    renderPreferences();
    pricingText.join('\\n');
  `,context);
  assert.match(bundled,/bundled rates/);
  assert.match(bundled,/About model prices/);
  assert.doesNotMatch(bundled,/prices\.json/);
  const override=vm.runInContext(`
    pricingText.length=0;
    snapshot.settings.prices_file='/custom/prices.json';
    renderPreferences();
    pricingText.join('\\n');
  `,context);
  assert.match(override,/\/custom\/prices\.json/);
  assert.match(override,/Open model price override/);
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const analyticsView = {period:'today',group:'model',metric:'tokens',search:'',model:null};
const chartInstances = [];
const tokenFields = ['requests','input','cached_input','cache_write','output'];
const chartColor = index => getComputedStyle(document.documentElement).getPropertyValue(`--model-${index+1}`).trim();
function chartTheme() {
  const style=getComputedStyle(document.documentElement);
  return {ink:style.getPropertyValue('--muted').trim(),panel:style.getPropertyValue('--control').trim(),line:style.getPropertyValue('--line').trim()};
}
const emptyTokens = () => ({requests:0,input:0,cached_input:0,cache_write:0,output:0});
const tokenTotal = row => row.input + row.cached_input + row.cache_write + row.output;
const compactNumber = value => new Intl.NumberFormat(undefined,{notation:'compact',maximumFractionDigits:1}).format(value);
const dollars = value => new Intl.NumberFormat(undefined,{style:'currency',currency:'USD'}).format(value);
function addTokens(target, source) { for(const key of tokenFields)target[key] += source[key]; return target; }
function modelRows(totals) {
  const rows = new Map();
  for(const group of [totals.routed,totals.classifier])for(const [id,tokens] of Object.entries(group)) {
    if(!rows.has(id))rows.set(id,{id,...emptyTokens()});
    addTokens(rows.get(id),tokens);
  }
  return [...rows.values()];
}
function disposeCharts() {
  for(const {chart,observer} of chartInstances) { observer.disconnect(); chart.dispose(); }
  chartInstances.length = 0;
}
// Canvas tooltips display recorded names without interpreting them as HTML.
function mountChart(parent, options, height, description, onClick) {
  const container=node('div',undefined,'usage-chart');container.style.height=`${height}px`;
  container.setAttribute('role','img');container.setAttribute('aria-label',description);parent.append(container);
  const colors=chartTheme();
  const chart=window.echarts.init(container,null,{renderer:'canvas'});
  // Recorded model names may match color literals, so theme colors apply only to presentation fields.
  chart.setOption({animation:false,backgroundColor:'transparent',textStyle:{fontFamily:'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',color:colors.ink},
    aria:{enabled:true},tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:colors.panel,textStyle:{color:colors.ink},borderColor:colors.line},...options});
  if(onClick)chart.on('click',onClick);
  const observer=new ResizeObserver(()=>chart.resize());observer.observe(container);chartInstances.push({chart,observer});
}
// Local midnight boundaries match the Rust reader's calendar-day totals.
function inPeriod(timestamp, period, today) {
  if(period==='all')return true;
  const date=new Date(timestamp),start=new Date(`${today}T00:00:00`),end=new Date(start);
  end.setDate(end.getDate()+1);
  if(period==='week')start.setDate(start.getDate()-6);
  return date>=start&&date<end;
}
function sessionRows(entries, period, today) {
  const rows=new Map();
  for(const entry of entries) {
    if(!inPeriod(entry.ts,period,today))continue;
    const id=entry.session_id||'';
    if(!rows.has(id))rows.set(id,{id,...emptyTokens()});
    const writes=entry.cache_creation_tokens||0;
    addTokens(rows.get(id),{requests:1,input:Math.max(0,entry.prompt_tokens-entry.cached_tokens-writes),cached_input:entry.cached_tokens,cache_write:writes,output:entry.completion_tokens});
  }
  return [...rows.values()];
}
function stableColor(id){let value=0;for(const c of id)value=(value*31+c.charCodeAt(0))>>>0;return chartColor(value%6);}
function openUsage(group, id) {
  usageView.page=0;usageView.session='';usageView.search='';
  usageView.exact={group,id,period:analyticsView.period};usageView.period=analyticsView.period;
  navigate('usage');
}
function renderOverview() {
  const colors=chartTheme();
  const metrics=snapshot.metrics,line=node('div',undefined,'connection-line');
  if(metrics.running)line.append(node('span','Connected','connection-state connected'));
  line.append(node('span',metrics.server_url,'connection-address'));
  button(line,`Manage ${snapshot.routes.length} routes`,()=>navigate('routes'));content.append(line);renderOnboarding();
  if(usageUnavailable()){const c=card('Usage unavailable');c.append(node('p','The routing log could not be read. Totals and costs are unavailable.'));button(actions(c),'Retry',()=>refresh());button(actions(c),'Open settings',()=>navigate('settings'));return;}
  const controls=node('div',undefined,'analytics-controls');content.append(controls);
  const period=field(controls,'Period',combobox([['all','All retained history'],['week','Seven local calendar days'],['today','Today']],analyticsView.period));
  const totals=snapshot.analytics[analyticsView.period],models=modelRows(totals),combined=models.reduce(addTokens,emptyTokens());
  const overhead=Object.values(totals.classifier).reduce(addTokens,emptyTokens());
  const estimate=snapshot.estimates[analyticsView.period],baseline=snapshot.baseline_model;
  const saved=estimate.actual!==null&&estimate.baseline!==null&&!estimate.uncounted_answers?estimate.baseline-estimate.actual:null;
  const unknown=estimate.unknown_calls?`${estimate.unknown_calls.toLocaleString()} ${estimate.unknown_calls===1?'call has':'calls have'} no recorded cost`:'';
  const metricsGrid=node('div',undefined,'usage-metrics');content.append(metricsGrid);
  for(const [label,value,detail] of [
    ['Recorded tokens',compactNumber(tokenTotal(combined)),`Model calls: ${combined.requests.toLocaleString()} · includes cached reads`],
    ['Routing overhead',compactNumber(tokenTotal(overhead)),`Classifier and judge calls: ${overhead.requests.toLocaleString()}`],
    ['Cost',estimate.actual!==null?dollars(estimate.actual):unknown?`${dollars(estimate.known)} or more`:'—',unknown||'Recorded by the server · includes routing overhead'],
    [saved!==null&&saved<0?'Extra cost':'Savings',saved!==null?dollars(Math.abs(saved)):'—',saved!==null?`vs ${baseline}`:estimate.baseline===null?`Add ${baseline} to the model prices`:unknown?'Needs a cost for every call':'Needs token counts for every answer'],
  ]) {
    const panel=node('div',undefined,'usage-metric');panel.append(node('p',label),node('strong',value),node('small',detail));metricsGrid.append(panel);
  }
  const costNote=node('details',undefined,'estimate-note');costNote.append(node('summary','How costs work'));
  costNote.append(node('p','The server records each call’s cost when the call finishes: the charge the provider reports, or an estimate from the model prices. Savings compare that cost with the same answer tokens at the baseline model’s rates. Classifier and judge calls add cost but have no baseline, so they reduce savings. These figures are not subscription charges.'));
  if(saved!==null)costNote.append(node('p',`Baseline: ${baseline} · ${dollars(estimate.baseline)}. ${estimate.baseline>0?`${(Math.abs(saved)/estimate.baseline*100).toFixed(1)}% ${saved<0?'extra cost':'savings'}.`:''}`));
  else {const missing=node('p',estimate.baseline===null?`The model prices have no rates for ${baseline}.`:unknown?`${unknown}. Calls recorded before the server priced calls, or for models without rates, have none.`:`${estimate.uncounted_answers.toLocaleString()} ${estimate.uncounted_answers===1?'answer has':'answers have'} no token counts, so the baseline cannot price ${estimate.uncounted_answers===1?'it':'them'}.`,'warning-note');content.append(missing);button(actions(missing),snapshot.settings?.prices_file?'Open model price override':'About model prices',()=>run({kind:'open_prices'},false));}
  content.append(costNote);
  const comparison=card(analyticsView.model?`Usage for ${analyticsView.model}`:'Usage by model');comparison.append(node('p',`Period: ${analyticsView.period==='all'?'All retained history':analyticsView.period==='today'?snapshot.today:`${new Date(new Date(`${snapshot.today}T00:00:00`).setDate(new Date(`${snapshot.today}T00:00:00`).getDate()-6)).toLocaleDateString()} – ${snapshot.today}`} · ${Intl.DateTimeFormat().resolvedOptions().timeZone}.`,'muted'));
  const filters=node('div',undefined,'analytics-controls');comparison.append(filters);
  const group=field(filters,'Compare',combobox([['model','Models'],['route','Routes'],['session','Sessions']],analyticsView.group));
  const metric=field(filters,'Measure',combobox([['tokens','Tokens'],['requests','Model calls']],analyticsView.metric));
  const search=field(filters,'Filter names',input(analyticsView.search));search.placeholder='Find a model, route, or session';
  button(actions(comparison),'Clear filters',()=>{analyticsView.search='';analyticsView.model=null;analyticsView.group='model';render();});
  const chartParent=node('div'),tableParent=node('div',undefined,'analytics-table');comparison.append(chartParent,tableParent);
  const labelFor=row=>row.id|| (group.value==='session'?'Session not recorded':'Route not recorded');
  const draw=()=>{
    analyticsView.group=group.value;analyticsView.metric=metric.value;analyticsView.search=search.value;
    comparison.querySelector('h3').textContent=analyticsView.model&&group.value==='model'?`Usage for ${analyticsView.model}`:`Usage by ${group.value}`;
    // A chart owns its resize observer only while its container is visible.
    disposeCharts();chartParent.replaceChildren();tableParent.replaceChildren();
    let rows=group.value==='model'?models:group.value==='route'?Object.entries(totals.routes).map(([id,tokens])=>({id,...tokens})):sessionRows(snapshot.entries,period.value,snapshot.today);
    const value=row=>metric.value==='tokens'?tokenTotal(row):row.requests;
    rows=rows.filter(row=>(!analyticsView.model||group.value!=='model'||row.id===analyticsView.model)&&labelFor(row).toLowerCase().includes(search.value.toLowerCase())).sort((a,b)=>value(b)-value(a)||a.id.localeCompare(b.id));
    chartParent.append(node('p',group.value==='session'?`Session comparison uses ${snapshot.entries.length.toLocaleString()} recent calls${snapshot.limited?' · limited to 5,000 records / 8 MiB':''}.`:'Totals cover completed calls in the retained log, including routing overhead.','muted'));
    if(rows.length) {
      const shown=rows.slice(0,10),names=shown.map(labelFor);
      if(rows.length>10)chartParent.append(node('p',`${rows.length-10} other groups: ${compactNumber(rows.slice(10).reduce((sum,row)=>sum+value(row),0))} ${metric.value==='tokens'?'tokens':'model calls'}. See all groups in the table.`,'muted'));
      chartParent.append(node('small',`Showing ${shown.length} of ${rows.length} ${group.value==='model'?'models':group.value==='route'?'routes':'sessions'} · exact values below`));
      mountChart(chartParent,{
        grid:{left:12,right:24,top:36,bottom:24,containLabel:true},
        legend:{top:0,textStyle:{color:colors.ink},itemWidth:12,itemHeight:8},
        xAxis:{type:'value',axisLabel:{formatter:compactNumber,color:colors.ink},splitLine:{lineStyle:{color:colors.line}}},
        yAxis:{type:'category',inverse:true,data:names,axisLabel:{color:colors.ink,width:160,overflow:'truncate'},axisTick:{show:false},axisLine:{show:false}},
        series:metric.value==='tokens'?['input','cached_input','cache_write','output'].map((key,i)=>({name:['Uncached input','Cached input','Cache writes','Output'][i],type:'bar',stack:'tokens',barMaxWidth:18,itemStyle:{color:chartColor(i)},data:shown.map(row=>row[key])})):[{name:'Model calls',type:'bar',barMaxWidth:18,itemStyle:{color:chartColor(0)},data:shown.map(row=>row.requests)}],
      },Math.max(200,shown.length*38+80),'Usage comparison. Exact counts and links are in the table below.',event=>openUsage(group.value,shown[event.dataIndex].id));
    } else chartParent.append(node('p',search.value?'No names match this filter.':'No model calls recorded for this period.','empty-state'));
    const scroll=node('div',undefined,'usage-table-scroll'),table=node('table',undefined,'usage-totals-table'),head=node('tr'),thead=node('thead'),body=node('tbody');
    for(const name of [group.value==='model'?'Model':group.value==='route'?'Route':'Session','Calls','Uncached input','Cached input','Cache writes','Output','Total tokens',...(group.value==='model'?['Answer tokens','Overhead tokens']:[])]) { const th=node('th',name);th.scope='col';head.append(th); }
    scroll.tabIndex=0;scroll.setAttribute('role','region');scroll.setAttribute('aria-label','Usage comparison table');thead.append(head);table.append(thead,body);scroll.append(table);tableParent.append(scroll);
    for(const row of rows) {
      const tr=node('tr'),name=node('td');button(name,labelFor(row),()=>openUsage(group.value,row.id));tr.append(name);
      for(const count of [row.requests,row.input,row.cached_input,row.cache_write,row.output,tokenTotal(row),...(group.value==='model'?[tokenTotal(totals.routed[row.id]||emptyTokens()),tokenTotal(totals.classifier[row.id]||emptyTokens())]:[])])tr.append(node('td',count.toLocaleString()));body.append(tr);
    }
    renderDaily(comparison);
  };
  period.onchange=()=>{analyticsView.period=period.value;analyticsView.model=null;render();};group.onchange=()=>{analyticsView.model=null;draw();};metric.onchange=draw;search.oninput=()=>{analyticsView.model=null;draw();};draw();
  content.append(node('p',`Aggregate totals exclude ${snapshot.analytics.skipped||0} unreadable records. Incomplete log data can undercount usage.`,'coverage-note'));
  content.append(node('p','All retained history resets if the routing log is replaced or truncated. Session and turn details use recent history.','coverage-note'));
}
function renderDaily(parent) {
  const colors=chartTheme();
  parent.querySelector('.daily-usage')?.remove();
  const panel=node('div',undefined,'daily-usage');parent.append(panel);panel.append(node('h3','Daily usage by model'),node('p',`This chart shows tokens for the seven local calendar days through ${snapshot.today}, including cached reads and routing overhead.`,'muted'));
  const end=new Date(`${snapshot.today}T00:00:00`),days=[];
  for(let offset=6;offset>=0;offset--) {const date=new Date(end);date.setDate(date.getDate()-offset);days.push(`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`);}
  const models=modelRows(snapshot.analytics.week).sort((a,b)=>tokenTotal(b)-tokenTotal(a)||a.id.localeCompare(b.id));
  if(!models.length){panel.append(node('p','No model calls recorded in the past 7 days.','empty-state'));return;}
  const shown=models.slice(0,5),ids=new Set(shown.map(model=>model.id));
  const daily=days.map(day=>new Map(modelRows(snapshot.analytics.days[day]||{routed:{},classifier:{}}).map(row=>[row.id,tokenTotal(row)])));
  const series=shown.map((model,i)=>({name:model.id,type:'bar',stack:'models',barMaxWidth:40,itemStyle:{color:stableColor(model.id)},data:daily.map(totals=>totals.get(model.id)||0)}));
  if(models.length>5)series.push({name:'Other models',type:'bar',stack:'models',itemStyle:{color:chartColor(5)},data:daily.map(totals=>[...totals].reduce((sum,[id,count])=>sum+(ids.has(id)?0:count),0))});
  mountChart(panel,{grid:{left:12,right:20,top:20,bottom:28,containLabel:true},
    xAxis:{type:'category',data:days.map(day=>new Date(`${day}T12:00:00`).toLocaleDateString(undefined,{month:'short',day:'numeric'})),axisLabel:{color:colors.ink},axisTick:{show:false}},
    yAxis:{type:'value',axisLabel:{color:colors.ink,formatter:compactNumber},splitLine:{lineStyle:{color:colors.line}}},series},220,'Daily recorded tokens. Exact daily totals follow.');
  const key=node('div',undefined,'chart-key');series.forEach((item,i)=>{const label=node('span'),swatch=node('i');swatch.style.background=item.itemStyle.color;label.append(swatch,document.createTextNode(item.name));key.append(label);});panel.append(key);
  const details=node('details');details.append(node('summary','Exact daily totals'));for(const [index,day] of days.entries())details.append(node('p',`${day}: ${series.map(item=>`${item.name} ${item.data[index].toLocaleString()}`).join(' · ')}`,'summary-row'));panel.append(details);
}

function renderOnboarding() {
  if(snapshot.entries.length || sessionStorage.getItem('switchyard-checklist-dismissed'))return;
  const c=card('Get started');c.append(node('p','Complete these steps to see your first recorded call.'));
  const steps=[['Server ready',snapshot.metrics.running,'settings'],['Choose a route',snapshot.routes.length>0,'routes'],['Connect a harness',connectedTools.size>0,'install'],['Open a session in Terminal',sessionOpened,'sessions'],['Find the first recorded call',false,'usage']];
  for(const [label,done,target] of steps)button(c,`${done?'✓ ':''}${label}`,()=>navigate(target));button(actions(c),'Dismiss checklist',()=>{sessionStorage.setItem('switchyard-checklist-dismissed','1');c.remove();});
}

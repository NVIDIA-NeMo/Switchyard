// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const analyticsView = {period:'all',group:'model',metric:'tokens',search:''};
const chartInstances = [];
const tokenFields = ['requests','input','cached_input','output'];
const chartColors = ['#90bfff','#73d7ba','#c5a4f5','#e9c47b','#ffacb4','#8dd4ec'];
const emptyTokens = () => ({requests:0,input:0,cached_input:0,output:0});
const tokenTotal = row => row.input + row.cached_input + row.output;
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
  const chart=window.echarts.init(container,null,{renderer:'canvas'});
  chart.setOption({animation:false,backgroundColor:'transparent',textStyle:{fontFamily:'-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',color:'#d1d7e0'},
    aria:{enabled:true,decal:{show:true}},tooltip:{trigger:'axis',renderMode:'richText',confine:true,backgroundColor:'#252b33',textStyle:{color:'#f2f4f8'},borderColor:'#8491a2'},...options});
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
    addTokens(rows.get(id),{requests:1,input:Math.max(0,entry.prompt_tokens-entry.cached_tokens),cached_input:entry.cached_tokens,output:entry.completion_tokens});
  }
  return [...rows.values()];
}
function openUsage(group, id) {
  usageView.session='';usageView.search='';
  usageView.exact={group,id,period:analyticsView.period};
  page='usage';render();
}
function renderOverview() {
  const metrics=snapshot.metrics,line=node('div',undefined,'connection-line');
  line.append(node('span',metrics.running?'Connected':'Not responding',metrics.running?'connection-state connected':'connection-state'),node('span',metrics.server_url,'connection-address'));
  button(line,`Manage ${snapshot.routes.length} routes`,()=>{page='routes';render();});content.append(line);
  const controls=node('div',undefined,'analytics-controls');content.append(controls);
  const period=field(controls,'Period',select([['all','All retained history'],['week','Past 7 days'],['today','Today']],analyticsView.period));
  const totals=snapshot.analytics[analyticsView.period],models=modelRows(totals),combined=models.reduce(addTokens,emptyTokens());
  const overhead=Object.values(totals.classifier).reduce(addTokens,emptyTokens());
  const estimate=snapshot.estimates[analyticsView.period],cost=estimate.cost;
  const metricsGrid=node('div',undefined,'usage-metrics');content.append(metricsGrid);
  for(const [label,value,detail] of [
    ['Recorded tokens',compactNumber(tokenTotal(combined)),`Model calls: ${combined.requests.toLocaleString()} · includes cached reads`],
    ['Routing overhead',compactNumber(tokenTotal(overhead)),`Classifier and judge calls: ${overhead.requests.toLocaleString()}`],
    ['Estimated cost',cost?dollars(cost.actual):'—',cost?'Includes routing overhead':'Prices required'],
    ['Estimated savings',cost?dollars(cost.baseline-cost.actual):'—',cost?`vs ${snapshot.baseline_model}`:'Set baseline and model prices'],
  ]) {
    const panel=node('div',undefined,'usage-metric');panel.append(node('p',label),node('strong',value),node('small',detail));metricsGrid.append(panel);
  }
  const costNote=node('details',undefined,'estimate-note');costNote.append(node('summary','How cost estimates work'));
  costNote.append(node('p','Estimates apply your configured per-token prices to recorded usage. The baseline prices the same answer tokens at one model’s rates. Routing overhead reduces savings. These are not subscription charges or measured token savings.'));
  if(cost)costNote.append(node('p',`Baseline: ${snapshot.baseline_model} · ${dollars(cost.baseline)}. ${cost.baseline>0?`${((cost.baseline-cost.actual)/cost.baseline*100).toFixed(1)}% estimated savings.`:''}`));
  else costNote.append(node('p',`Add prices in menubar.toml for: ${estimate.missing.join(', ')||'the baseline model'}.`));
  content.append(costNote);
  const comparison=card('Usage by model');
  const filters=node('div',undefined,'analytics-controls');comparison.append(filters);
  const group=field(filters,'Compare',select([['model','Models'],['route','Routes'],['session','Sessions']],analyticsView.group));
  const metric=field(filters,'Measure',select([['tokens','Tokens'],['requests','Model calls']],analyticsView.metric));
  const search=field(filters,'Filter names',input(analyticsView.search));search.placeholder='Find a model, route, or session';
  const chartParent=node('div'),tableParent=node('div',undefined,'analytics-table');comparison.append(chartParent,tableParent);
  const labelFor=row=>row.id|| (group.value==='session'?'Session not recorded':'Route not recorded');
  const draw=()=>{
    analyticsView.group=group.value;analyticsView.metric=metric.value;analyticsView.search=search.value;
    comparison.querySelector('h3').textContent=`Usage by ${group.value}`;
    // A chart owns its resize observer only while its container is visible.
    disposeCharts();chartParent.replaceChildren();tableParent.replaceChildren();
    let rows=group.value==='model'?models:group.value==='route'?Object.entries(totals.routes).map(([id,tokens])=>({id,...tokens})):sessionRows(snapshot.entries,period.value,snapshot.today);
    const value=row=>metric.value==='tokens'?tokenTotal(row):row.requests;
    rows=rows.filter(row=>labelFor(row).toLowerCase().includes(search.value.toLowerCase())).sort((a,b)=>value(b)-value(a)||a.id.localeCompare(b.id));
    chartParent.append(node('p',group.value==='session'?`Session comparison uses ${snapshot.entries.length.toLocaleString()} recent calls${snapshot.limited?' · limited to 5,000 records / 8 MiB':''}.`:'Totals cover completed calls in the retained log, including routing overhead.','muted'));
    if(rows.length) {
      const shown=rows.slice(0,10),names=shown.map(labelFor);
      chartParent.append(node('small',`Showing ${shown.length} of ${rows.length} ${group.value==='model'?'models':group.value==='route'?'routes':'sessions'} · exact values below`));
      mountChart(chartParent,{
        grid:{left:12,right:24,top:36,bottom:24,containLabel:true},
        legend:{top:0,textStyle:{color:'#d1d7e0'},itemWidth:12,itemHeight:8},
        xAxis:{type:'value',axisLabel:{formatter:compactNumber,color:'#d1d7e0'},splitLine:{lineStyle:{color:'#454d59'}}},
        yAxis:{type:'category',inverse:true,data:names,axisLabel:{color:'#f2f4f8',width:160,overflow:'truncate'},axisTick:{show:false},axisLine:{show:false}},
        series:metric.value==='tokens'?['input','cached_input','output'].map((key,i)=>({name:['Input','Cached input','Output'][i],type:'bar',stack:'tokens',barMaxWidth:18,itemStyle:{color:chartColors[i]},data:shown.map(row=>row[key])})):[{name:'Model calls',type:'bar',barMaxWidth:18,itemStyle:{color:chartColors[0]},data:shown.map(row=>row.requests)}],
      },Math.max(200,shown.length*38+80),'Usage comparison. Exact counts and links are in the table below.',event=>openUsage(group.value,shown[event.dataIndex].id));
    } else chartParent.append(node('p',search.value?'No names match this filter.':'No model calls recorded for this period.','empty-state'));
    const scroll=node('div',undefined,'usage-table-scroll'),table=node('table',undefined,'usage-totals-table'),head=node('tr'),thead=node('thead'),body=node('tbody');
    for(const name of [group.value==='model'?'Model':group.value==='route'?'Route':'Session','Calls','Input','Cached','Output','Total tokens']) { const th=node('th',name);th.scope='col';head.append(th); }
    thead.append(head);table.append(thead,body);scroll.append(table);tableParent.append(scroll);
    for(const row of rows) {
      const tr=node('tr'),name=node('td');button(name,labelFor(row),()=>openUsage(group.value,row.id));tr.append(name);
      for(const count of [row.requests,row.input,row.cached_input,row.output,tokenTotal(row)])tr.append(node('td',count.toLocaleString()));body.append(tr);
    }
    renderDaily(comparison);
  };
  period.onchange=()=>{analyticsView.period=period.value;render();};group.onchange=draw;metric.onchange=draw;search.oninput=draw;draw();
  content.append(node('p','All retained history resets if the routing log is replaced or truncated. Session and turn details use recent history.','coverage-note'));
}
function renderDaily(parent) {
  parent.querySelector('.daily-usage')?.remove();
  const panel=node('div',undefined,'daily-usage');parent.append(panel);panel.append(node('h3','Daily usage by model'),node('p','Past 7 local calendar days · tokens, including cached reads and routing overhead','muted'));
  const end=new Date(`${snapshot.today}T00:00:00`),days=[];
  for(let offset=6;offset>=0;offset--) {const date=new Date(end);date.setDate(date.getDate()-offset);days.push(`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`);}
  const models=modelRows(snapshot.analytics.week).sort((a,b)=>tokenTotal(b)-tokenTotal(a)||a.id.localeCompare(b.id));
  if(!models.length){panel.append(node('p','No model calls recorded in the past 7 days.','empty-state'));return;}
  const series=models.slice(0,5).map((model,i)=>({name:model.id,type:'bar',stack:'models',barMaxWidth:40,itemStyle:{color:chartColors[i]},data:days.map(day=>tokenTotal(modelRows(snapshot.analytics.days[day]||{routed:{},classifier:{}}).find(row=>row.id===model.id)||emptyTokens()))}));
  if(models.length>5)series.push({name:'Other models',type:'bar',stack:'models',itemStyle:{color:chartColors[5]},data:days.map(day=>modelRows(snapshot.analytics.days[day]||{routed:{},classifier:{}}).filter(row=>!models.slice(0,5).some(model=>model.id===row.id)).reduce((sum,row)=>sum+tokenTotal(row),0))});
  mountChart(panel,{grid:{left:12,right:20,top:20,bottom:28,containLabel:true},
    xAxis:{type:'category',data:days.map(day=>new Date(`${day}T12:00:00`).toLocaleDateString(undefined,{month:'short',day:'numeric'})),axisLabel:{color:'#d1d7e0'},axisTick:{show:false}},
    yAxis:{type:'value',axisLabel:{color:'#d1d7e0',formatter:compactNumber},splitLine:{lineStyle:{color:'#454d59'}}},series},220,'Daily recorded tokens. Exact daily totals follow.');
  const key=node('div',undefined,'chart-key');series.forEach((item,i)=>{const label=node('span'),swatch=node('i');swatch.style.background=chartColors[i];label.append(swatch,document.createTextNode(item.name));key.append(label);});panel.append(key);
  const daily=node('details');daily.append(node('summary','Exact daily totals'));for(const [index,day] of days.entries())daily.append(node('p',`${day}: ${series.map(item=>`${item.name} ${item.data[index].toLocaleString()}`).join(' · ')}`,'summary-row'));panel.append(daily);
}

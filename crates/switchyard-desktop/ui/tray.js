// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const invoke=window.__TAURI__.core.invoke,$=id=>document.getElementById(id);
const compact=value=>new Intl.NumberFormat(undefined,{notation:'compact',maximumFractionDigits:1}).format(value);
const themeButtons=[...document.querySelectorAll('.themes button')];
let disposeChart=()=>{},chart,usage,selectedHour=23,selectedModel=null,hoveredModel=null;
let snapshot,refreshing=false,refreshTimer;
function clearChart() {disposeChart();disposeChart=()=>{};chart=undefined;}
function message(text) {$('error').textContent=text;$('error').hidden=!text;}
function inspectHour(index) {
  if(!usage)return;
  selectedHour=Math.max(0,Math.min(23,index));const row=usage.rows[selectedHour];
  const end=new Date(row.end),start=new Date(row.start),time=date=>date.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});
  $('hour-readout').textContent=`${start.toLocaleDateString(undefined,{month:'short',day:'numeric'})} · ${time(start)}–${time(end)}`;
  const activeModel=hoveredModel||selectedModel;
  const entries=Object.entries(row.models).filter(([id])=>!activeModel||id===activeModel);
  $('hour-models').replaceChildren(...entries.map(([id,value])=>{const line=document.createElement('p');line.className='hour-model';line.textContent=`${id} · ${value.tokens.toLocaleString()} tokens · ${value.calls} calls`;line.style.borderColor=modelColor(id);return line;}));
  if(!entries.length)$('hour-models').textContent='No recorded calls match this selection.';
  const changes=row.switches.filter(change=>!activeModel||change.from===activeModel||change.to===activeModel);
  const completionTime=ts=>new Date(ts).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  $('switch-readout').textContent=changes.length?changes.map(change=>`${completionTime(change.ts)} ${change.from} → ${change.to} · session ${change.session_id}`).join('\n'):'No recorded session model changes in this hour.';
  [...$('hour-buttons').children].forEach((button,i)=>button.setAttribute('aria-pressed',String(i===selectedHour)));
  chart?.highlight({model:activeModel,hour:selectedHour});
  for(const marker of document.querySelectorAll('.switch-marker'))marker.classList.toggle('selected',Number(marker.dataset.hour)===selectedHour);
  const cursor=$('chart-cursor');if(cursor)cursor.style.left=`${hourPosition(selectedHour)}px`;
}
function selectModel(id) {
  selectedModel=id;hoveredModel=null;$('all-models').setAttribute('aria-pressed',String(!id));for(const button of $('model-list').children)button.setAttribute('aria-pressed',String(button.dataset.model===id));inspectHour(selectedHour);
}
function render() {
  usage=buildTrayUsage(snapshot.entries,Date.now());
  configureModelColors(usage.models.map(row=>row.id));
  hoveredModel=null;
  if(selectedModel&&!usage.models.some(row=>row.id===selectedModel))selectedModel=null;
  $('all-models').setAttribute('aria-pressed',String(!selectedModel));
  const health=snapshot.data_state?.health;
  const healthState=health?.state||(snapshot.metrics.running?'running':'unreachable');
  const usageState=snapshot.data_state?.usage?.state;
  $('connection').textContent=`Server ${healthState}${usageState==='unavailable'?' · usage unavailable':usageState==='stale'?' · usage stale':''}`;
  $('connection').title=health?.checked_at?`Checked ${health.checked_at}. ${health.reason||''}`:health?.reason||'';
  $('connection').classList.toggle('running',healthState==='running');
  $('model-count').textContent=usage.models.length;$('calls').textContent=`${usage.models.reduce((sum,row)=>sum+row.calls,0).toLocaleString()} calls`;
  $('switch-count').textContent=usage.switches.length;$('overhead').textContent=`Routing: ${compact(usage.routingTokens)} tokens`;
  message([...(snapshot.errors||[]),...(usageState==='unavailable'?['Usage is unavailable. Counts below may contain retained data.']:[])].join('\n'));
  $('coverage-summary').textContent=snapshot.limited?'Log limit reached · usage may be incomplete':'Recorded calls · completion times';
  $('coverage').textContent=`${snapshot.limited?'Log limit reached; this view may omit calls. ':'Retained calls only. '}${snapshot.skipped?`${snapshot.skipped} unreadable log records skipped. `:''}Tokens include input and output; cached input is counted once. Times show call completion. Model changes compare consecutive completed answers in the same session. Overlapping calls may complete out of order. Routing tokens are shown separately.${usage.unknownSessionCalls?` ${usage.unknownSessionCalls} calls have no session ID and cannot show model changes.`:''}`;
  $('hour-buttons').replaceChildren(...usage.rows.map((row,index)=>{const button=document.createElement('button');button.textContent=index%6===0?new Date(row.start).toLocaleTimeString(undefined,{hour:'2-digit',hour12:false}):'·';button.title=new Date(row.start).toLocaleString();button.setAttribute('aria-label',`Inspect ${button.title}`);button.onclick=()=>inspectHour(index);return button;}));
  $('model-list').replaceChildren(...usage.models.map(row=>{const button=document.createElement('button');button.className='model-row';button.dataset.model=row.id;button.setAttribute('aria-pressed',String(row.id===selectedModel));const dot=document.createElement('i'),name=document.createElement('strong'),count=document.createElement('span');dot.style.background=modelColor(row.id);name.textContent=row.id;name.title=row.id;count.textContent=`${compact(row.tokens)} · ${row.calls} calls`;button.title=`${row.tokens.toLocaleString()} tokens · ${row.calls} calls`;button.append(dot,name,count);button.onclick=()=>selectModel(row.id);button.onpointerenter=()=>{hoveredModel=row.id;inspectHour(selectedHour);};button.onpointerleave=()=>{hoveredModel=null;inspectHour(selectedHour);};return button;}));
  if(!usage.models.length)$('model-list').textContent='No model calls recorded in the past 24 hours.';
  renderChart();inspectHour(selectedHour);
}
function renderChart() {
  clearChart();$('chart').replaceChildren();if(!document.hasFocus())return;
  const css=getComputedStyle(document.body),colors={};for(const name of ['accent','amber','muted','line'])colors[name]=css.getPropertyValue(`--${name}`).trim();
  try {
    chart=renderTrayChart($('chart'),usage,colors,cleanup=>{disposeChart=cleanup;});
    const overlay=document.createElement('div');overlay.className='chart-overlay';
    const cursor=document.createElement('i');cursor.id='chart-cursor';overlay.append(cursor);
    for(const [index,hour] of usage.rows.entries()){if(!hour.switches.length)continue;const marker=document.createElement('button');marker.className='switch-marker';marker.dataset.hour=index;marker.style.left=`${hourPosition(index)}px`;marker.title=`${hour.switches.length} recorded model changes · ${new Date(hour.start).toLocaleTimeString()}`;marker.setAttribute('aria-label',marker.title);marker.onclick=event=>{event.stopPropagation();inspectHour(index);};overlay.append(marker);}
    $('chart').append(overlay);$('renderer-note').textContent=trayChartNote;
  }catch(error){clearChart();$('chart').replaceChildren();const note=document.createElement('p');note.className='chart-error';note.textContent=`Could not draw usage: ${error.message}`;$('chart').append(note);$('renderer-note').textContent='Reopen the preview to try again.';}
}
function hourPosition(index) {const layout=chartLayout($('chart'),usage);return layout.left+(index+.5)*layout.step;}
function hourAt(clientX) {const bounds=$('chart').getBoundingClientRect(),layout=chartLayout($('chart'),usage);return Math.max(0,Math.min(23,Math.floor((clientX-bounds.left-layout.left)/layout.step)));}
$('chart').onpointermove=event=>{if(!usage)return;const index=hourAt(event.clientX);if(index!==selectedHour)inspectHour(index);};
$('chart').onclick=event=>{if(usage)inspectHour(hourAt(event.clientX));};
$('all-models').onclick=()=>selectModel(null);
async function refresh() {
  clearTimeout(refreshTimer);if(refreshing)return;refreshing=true;
  try {snapshot=await invoke('snapshot');render();}catch(error){message(`${snapshot?'Displayed counts are stale. ':''}Could not load usage: ${error}`);$('connection').textContent=snapshot?'Stale · last refresh failed':'Unavailable · refresh failed';$('connection').classList.remove('running');}
  finally{refreshing=false;if(document.hasFocus())refreshTimer=setTimeout(refresh,(snapshot?.refresh_seconds||1)*1000);}
}
async function command(name,args){try{await invoke(name,args);}catch(error){message(`Could not open or close the window: ${error}`);}}
function updateAppearance() {
  const mode=window.switchyardAppearance.mode;
  for(const choice of themeButtons)choice.setAttribute('aria-pressed',String(choice.dataset.theme===mode));
  $('theme-position').textContent=`${themeButtons.findIndex(button=>button.dataset.theme===mode)+1} / ${themeButtons.length}`;
  if(snapshot){render();}
}
function selectTheme(index){const choice=themeButtons[(index+themeButtons.length)%themeButtons.length];window.switchyardAppearance.choose(choice.dataset.theme);}
themeButtons.forEach((button,index)=>{button.onclick=()=>selectTheme(index);});
for(const [id,direction] of [['previous-theme',-1],['next-theme',1]])$(id).onclick=()=>selectTheme(themeButtons.findIndex(button=>button.dataset.theme===window.switchyardAppearance.mode)+direction);
window.addEventListener('switchyard-theme',updateAppearance);
for(const button of document.querySelectorAll('[data-page]'))button.onclick=()=>command('tray_page',{page:button.dataset.page});
$('close').onclick=()=>command('dismiss_tray');document.addEventListener('keydown',event=>{if(event.key==='Escape')command('dismiss_tray');});
window.addEventListener('resize',()=>{if(snapshot&&document.hasFocus()){renderChart();inspectHour(selectedHour);}});
window.addEventListener('switchyard-tray-open',refresh);window.addEventListener('focus',refresh);
window.addEventListener('blur',()=>{clearTimeout(refreshTimer);clearChart();});
updateAppearance();
if(document.hasFocus())refresh();

let measuredHeight = 0;
function sizePopup() {
  const header = document.querySelector('header'), main = document.querySelector('main'), footer = document.querySelector('footer');
  const height = Math.ceil(header.getBoundingClientRect().height + main.scrollHeight + footer.getBoundingClientRect().height);
  if (height !== measuredHeight) {measuredHeight = height; command('tray_size', {height});}
}
if (typeof ResizeObserver === 'function') {
  const observer = new ResizeObserver(sizePopup);
  observer.observe(document.querySelector('main'));
  observer.observe(document.querySelector('footer'));
  const changes = new MutationObserver(() => requestAnimationFrame(sizePopup));
  changes.observe(document.querySelector('main'), {childList:true,subtree:true,characterData:true});
  document.querySelector('details').addEventListener('toggle', sizePopup);
}
window.addEventListener('switchyard-tray-open', sizePopup);

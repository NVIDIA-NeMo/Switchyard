// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const invoke=window.__TAURI__.core.invoke,$=id=>document.getElementById(id);
let refreshing=false,refreshTimer,running=false;
function node(tag,text,className) {const element=document.createElement(tag);if(text!==undefined)element.textContent=text;if(className)element.className=className;return element;}
function message(text) {$('error').textContent=text;$('error').hidden=!text;}
function show(id,text) {$(id).textContent=text||'';$(id).hidden=!text;}
// The tray menu shows the same overview as text rows; the popup groups it into sections.
function render(view) {
  const overview=view.overview;
  message(view.errors.join('\n'));
  $('connection').textContent=overview?(overview.running?'Server running':'Server not responding'):'Usage unavailable';
  $('connection').classList.toggle('running',Boolean(overview?.running));
  show('server-url',overview?.server_url);
  show('empty',overview?.empty);
  $('periods').replaceChildren(...(overview?.periods||[]).map(period=>{
    const section=node('section',undefined,'period');
    section.append(node('h2',period.name),node('p',period.traffic),node('p',period.spent,'detail'));
    if(period.saved)section.append(node('p',period.saved,'detail saved'));
    return section;
  }));
  const models=overview?.models||[];
  $('models').hidden=!models.length;
  $('model-list').replaceChildren(...models.map(share=>{
    const row=node('li'),name=node('span',share.model,'model'),track=node('span',undefined,'track'),bar=node('i');
    name.title=share.model;bar.style.width=`${share.percent}%`;track.append(bar);
    row.append(name,node('span',`${share.percent.toFixed(0)}%`,'share'),track);
    return row;
  }));
  show('note',overview?.note);
}
async function refresh() {
  clearTimeout(refreshTimer);if(refreshing)return;refreshing=true;let seconds=5;
  try {const view=await invoke('tray_overview');seconds=view.refresh_seconds;render(view);}
  catch(error){message(`Could not read usage: ${error}`);}
  finally{refreshing=false;if(document.hasFocus())refreshTimer=setTimeout(refresh,seconds*1000);}
}
async function command(name,args) {try{await invoke(name,args);}catch(error){message(`Could not open or close the window: ${error}`);}}
const actionButtons=[...document.querySelectorAll('[data-action]')];
for(const button of actionButtons)button.onclick=async()=>{
  if(running)return;running=true;actionButtons.forEach(other=>{other.disabled=true;});
  $('status').textContent=button.dataset.action==='restart'?'Restarting the server…':'Opening the server config…';
  try {const reply=await invoke('action',{action:{kind:button.dataset.action}});$('status').textContent=reply.message;}
  catch(error){$('status').textContent=String(error);}
  finally{running=false;actionButtons.forEach(other=>{other.disabled=false;});refresh();}
};
for(const button of document.querySelectorAll('[data-page]'))button.onclick=()=>command('tray_page',{page:button.dataset.page});
$('quit').onclick=()=>command('quit_app');
// Rows highlight like menu items: under the pointer or from the arrow keys, and never left
// over from the last time the popup was open.
const rows=[...document.querySelectorAll('nav button')];
function highlight(row) {rows.forEach(other=>other.classList.toggle('active',other===row));if(row)row.focus({preventScroll:true});else document.activeElement?.blur();}
for(const row of rows){row.onpointerenter=()=>highlight(row);row.onpointerleave=()=>highlight(null);}
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'){command('dismiss_tray');return;}
  if(event.key!=='ArrowDown'&&event.key!=='ArrowUp')return;
  event.preventDefault();
  const enabled=rows.filter(row=>!row.disabled),current=enabled.indexOf(document.activeElement);
  const next=event.key==='ArrowDown'?(current+1)%enabled.length:(current<=0?enabled.length:current)-1;
  highlight(enabled[next]);
});
window.addEventListener('switchyard-tray-open',()=>{highlight(null);refresh();});window.addEventListener('focus',refresh);
window.addEventListener('blur',()=>{clearTimeout(refreshTimer);highlight(null);});
// Loading once at startup gives the first opening its final height instead of a resize.
refresh();

// The native panel takes its height from the content, up to the limit that tray_size applies.
let measuredHeight=0;
function sizePopup() {
  const style=getComputedStyle($('popup')),padding=parseFloat(style.paddingTop)+parseFloat(style.paddingBottom);
  const height=Math.ceil(padding+document.querySelector('header').offsetHeight+document.querySelector('main').scrollHeight+document.querySelector('footer').offsetHeight);
  if(height!==measuredHeight){measuredHeight=height;command('tray_size',{height});}
}
if(typeof ResizeObserver==='function') {
  const observer=new ResizeObserver(sizePopup);
  for(const tag of ['header','main','footer'])observer.observe(document.querySelector(tag));
  new MutationObserver(()=>requestAnimationFrame(sizePopup)).observe(document.body,{childList:true,subtree:true,characterData:true});
}
window.addEventListener('switchyard-tray-open',sizePopup);

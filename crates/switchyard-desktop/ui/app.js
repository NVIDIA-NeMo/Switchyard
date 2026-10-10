// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const invoke = window.__TAURI__.core.invoke;
const content = document.querySelector('#content');
const status = document.querySelector('#status');
const notification = document.querySelector('#notification');
const pageErrors = document.querySelector('#page-errors');
// Each action replaces the previous notification so dismissed results stay dismissed.
let notificationReturn;
function notify(message, tone = 'success', details = []) {
  if(notification.hidden&&document.activeElement!==document.body)notificationReturn=document.activeElement;
  notification.dataset.tone = tone;
  const label = {success:'✓',progress:'In progress:',warning:'Attention:',error:'Error:'}[tone];
  status.textContent = message ? `${label} ${message}` : '';
  const disclosure = document.querySelector('#notification-details');
  disclosure.open = false;
  disclosure.hidden = !details.length;
  document.querySelector('#notification-detail-text').textContent = details.join('\n');
  notification.hidden = !message;
}
// Background failures remain on the page without replacing the last action result.
function showPageErrors(message) {
  pageErrors.textContent = message;
  pageErrors.hidden = !message;
}
// Dismissal returns keyboard focus to an available element, even during an operation.
function dismissNotification(event) {
  notification.hidden = true;
  status.textContent = '';
  const refreshButton = document.querySelector('#refresh');
  if(!event || event.detail===0) (notificationReturn?.isConnected && !notificationReturn.disabled ? notificationReturn : refreshButton.disabled ? document.querySelector('#title') : refreshButton).focus();
}

let snapshot, page = 'overview', busy = false, reading = false, lastRefresh = null, stale = false;
const sessionView = {tool:'',account:'',route:'',project:''};
const accountView = {tool:'codex_cli',name:''};
const activity = [];
const connectedTools=new Set();
let sessionOpened=false;
let preferencesDraft;
const pageNames = {overview:'Overview',routes:'Routes',install:'Connect tools',usage:'Usage',sessions:'Sessions',settings:'Settings'};
let operationQueue = Promise.resolve(),ipcQueue=Promise.resolve();
// The native controller accepts one operation at a time; queued reads leave the page usable.
function request(command,args){const captured=args===undefined?undefined:structuredClone(args);const pending=ipcQueue.then(()=>invoke(command,captured));ipcQueue=pending.catch(()=>{});return pending;}
const drafts = new Map(), lists = new Map();
const usageView = {session:'',search:'',exact:null,period:'all',role:'all',turn:'',sort:'newest'};
const routeView = {selected:'',search:''};
const installView = {tool:'',account:'',route:'',file:'',mode:'default'};
function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
function card(title) { const c = node('div', undefined, 'card'); c.append(node('h3', title)); content.append(c); return c; }
function button(parent, text, fn, primary = false) { const b = node('button', text, primary ? 'primary' : ''); b.onclick = fn; parent.append(b); return b; }
// Wry's macOS delegate has no JavaScript confirm handler, so this dialog handles confirmation.
function confirmAction(title, message, label, action, onCancel) {
  const previous=document.activeElement,overlay=node('div',undefined,'modal-backdrop'),dialog=node('div',undefined,'confirm-dialog');
  dialog.setAttribute('role','dialog');dialog.setAttribute('aria-modal','true');dialog.setAttribute('aria-labelledby','confirmation-title');dialog.setAttribute('aria-describedby','confirmation-description');
  const heading=node('h3',title);heading.id='confirmation-title';heading.tabIndex=-1;
  const description=node('p',message,'confirmation-description');description.id='confirmation-description';
  dialog.append(heading,description);overlay.append(dialog);document.body.append(overlay);
  const background=[...document.body.children].filter(element=>element!==overlay).map(element=>[element,element.inert]);
  background.forEach(([element])=>{element.inert=true;});
  const close=(cancelled=true)=>{overlay.remove();background.forEach(([element,inert])=>{element.inert=inert;});if(previous?.isConnected)previous.focus();if(cancelled)onCancel?.();};
  const controls=actions(dialog),cancel=button(controls,'Cancel',close),accept=button(controls,label,()=>{close(false);action();});accept.classList.add('danger');
  dialog.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();close();}else if(event.key==='Tab'){const items=[cancel,accept],index=items.indexOf(document.activeElement);event.preventDefault();items[index<0?(event.shiftKey?items.length-1:0):(index+(event.shiftKey?-1:1)+items.length)%items.length].focus();}};
  if(message.length>300)heading.focus();else cancel.focus();
}
function field(parent, title, control) { control.setAttribute('aria-label',title); const l = node('label', title); l.append(control); parent.append(l); return control; }
function select(options, chosen) { const s = node('select'); if(chosen && !options.some(([value])=>value===chosen))options=[...options,[chosen,`${chosen} · unavailable in current data`]]; for (const [value, label] of options) { const o = node('option', label); o.value = value; s.append(o); } if (chosen !== undefined) s.value = chosen; return s; }
function input(value = '', type = 'text') { const i = node('input'); i.type = type; i.value = value; return i; }
function text(parent, value) { parent.append(node('pre', value)); }
function actions(parent) { const a = node('div', undefined, 'actions'); parent.append(a); return a; }
let lockedFields=[];
function setBusy(value) {
  busy=value;
  if(value){lockedFields=[...document.activeElement?.closest?.('.card')?.querySelectorAll('input,select')||[]].map(control=>[control,control.disabled]);lockedFields.forEach(([control])=>{control.disabled=true;});}
  else{for(const [control,disabled] of lockedFields)control.disabled=disabled||control.dataset.blocked==='true';lockedFields=[];}
  document.querySelectorAll('button[data-mutation]').forEach(n=>{n.disabled=value || n.dataset.blocked==='true';});
  document.querySelector('#content').setAttribute?.('aria-busy',String(value));
}
// Reads and writes use the controller queue; only writes lock mutation controls.
function enqueue(operation) {
  operationQueue=operationQueue.then(operation).catch(error=>notify(String(error),'error'));
  return operationQueue;
}
async function readAction(action) {
  return request('action',{action});
}
async function run(action, refreshAfter = true) {
  if(['editor','preview_install','preview_restore','preview_launch','preview_update','preview_route','preview_account_archive','preview_session_cleanup','preview_route_change'].includes(action.kind)) {
    try { return await readAction(action); } catch(error) { showPageErrors(String(error)); return; }
  }
  if(busy)return;
  const progress={apply:'Saving the route and restarting Switchyard…',remove:'Removing the route and restarting Switchyard…',restart:'Restarting Switchyard…',install:'Applying routing settings…',restore:'Restoring settings…',launch:'Opening Terminal…',models:'Loading models…'};
  const previous=document.activeElement;
  setBusy(true);notificationReturn=previous?.tagName==='BODY'?document.querySelector('#title'):previous;notify(progress[action.kind]||'Working…','progress');
  try {
    const result=await request('action',{action});
    notify(result.message,result.data?.warning?'warning':'success',result.data?.details||[]);
    if(action.kind==='install')connectedTools.add(action.tool);if(action.kind==='launch')sessionOpened=true;
    activity.unshift({message:result.message,time:new Date().toLocaleTimeString()});activity.splice(8);
    if(refreshAfter)await refresh(true);
    return result;
  } catch(error) {notify(String(error),'error');}
  finally {setBusy(false);if(previous?.isConnected&&!previous.disabled)previous.focus({preventScroll:true});else document.querySelector('#title').focus();}
}
function capturePageState() {
  const focused=document.activeElement;
  return {label:focused?.getAttribute?.('aria-label'),id:focused?.id,start:focused?.selectionStart,end:focused?.selectionEnd,
    open:[...content.querySelectorAll('details[open]')].map(n=>n.querySelector('summary')?.textContent),scroll:window.scrollY||0,
    regions:[...content.querySelectorAll('.scroll,.usage-table-scroll,.route-picker')].map(n=>[n.scrollTop,n.scrollLeft])};
}
function restorePageState(saved) {
  content.querySelectorAll('details').forEach(n=>{n.open=saved.open.includes(n.querySelector('summary')?.textContent);});
  const control=[...content.querySelectorAll('input,select,button')].find(n=>saved.id?n.id===saved.id:saved.label&&n.getAttribute('aria-label')===saved.label);
  if(control && !control.disabled){control.focus({preventScroll:true});if(saved.start!==null && saved.start!==undefined && control.setSelectionRange && !['number','date'].includes(control.type))control.setSelectionRange(saved.start,saved.end);}
  content.querySelectorAll('.scroll,.usage-table-scroll,.route-picker').forEach((n,i)=>{if(saved.regions[i])[n.scrollTop,n.scrollLeft]=saved.regions[i];});
  window.scrollTo?.(0,saved.scroll);
}
async function refresh(force = true) {
  if(reading)return;
  reading=true;
  const saved=snapshot?capturePageState():null;
  document.querySelector('#refresh').setAttribute?.('aria-busy','true');
  try {
    const latest=await request('snapshot');snapshot=latest;lastRefresh=new Date();stale=false;
    if(force || ['overview','usage','settings'].includes(page)) {render();if(saved)restorePageState(saved);}
    showPageErrors((snapshot.errors||[]).join('\n'));showFreshness();
  } catch(error) {stale=true;showPageErrors(`Could not refresh the page: ${error}`);showFreshness();if(!snapshot)renderRecovery(String(error));}
  finally {reading=false;document.querySelector('#refresh').setAttribute?.('aria-busy','false');}
}
function showFreshness() {
  const health=document.querySelector('#server-state');if(health&&snapshot?.metrics){health.replaceChildren();const state=snapshot.data_state?.health?.state||(snapshot.metrics.running?'running':'unreachable');health.hidden=state==='running'&&!stale;if(!health.hidden){health.append(node('span',stale?'Server status is stale. Check again before relying on it.':`Server ${state}. ${snapshot.data_state?.health?.reason||snapshot.metrics.server_url}`));button(health,'Check again',()=>refresh(false));mutation(health,'Retry restart',()=>run({kind:'restart'}));button(health,'Open recovery',()=>navigate('settings'));}}
  const label=document.querySelector('#freshness');
  if(label)label.textContent=lastRefresh?`${stale?'Stale · last successful check':'Checked'} ${lastRefresh.toLocaleTimeString()}`:'Loading…';
  const connection=document.querySelector('.connection-state');if(stale&&connection){connection.textContent='Unavailable · last check failed';connection.classList.remove('connected');}
}
function navigate(next) {page=next;render();document.querySelector('#title').focus();window.scrollTo?.(0,0);}
function renderRecovery(error) {
  content.replaceChildren();const c=card('App settings need attention');c.append(node('p',error));
  button(actions(c),'Open app settings',()=>run({kind:'open_settings'},false));button(actions(c),'Retry',()=>refresh());
}
function blocked(control, value) {control.dataset.blocked=String(value);control.disabled=(control.dataset.mutation==='true'&&busy)||value;}
function mutation(parent,label,fn,primary=false) {const b=button(parent,label,fn,primary);b.dataset.mutation='true';b.disabled=busy;return b;}
function fieldError(control,message) {
  const id=`error-${control.getAttribute('aria-label').replace(/[^a-z0-9]/gi,'-')}`;
  let note=control.parentElement.querySelector('.field-error');if(!note){note=node('small',undefined,'field-error');note.id=id;control.parentElement.append(note);}
  note.textContent=message;note.hidden=!message;control.setAttribute('aria-describedby',id);control.setAttribute('aria-invalid',String(Boolean(message)));
  return !message;
}
function routeMatches(route, query) { return [route.id,route.label,snapshot.algorithms.find(a=>a.kind===route.kind)?.title,...route.choices.flatMap(c=>[c.model,c.client,snapshot.clients.find(e=>e.name===c.client)?.host])].join(' ').toLowerCase().includes(query.toLowerCase()); }
function routeField(c,chosen='',tool='') {
  const wrapper=node('div');c.append(wrapper);
  const route=field(wrapper,'Route',select(snapshot.routes.map(r=>[r.key,`${r.id} · ${snapshot.algorithms.find(a=>a.kind===r.kind)?.title||r.kind}${tool&&r.compatibility?.[tool]?' · unavailable':''}`]),chosen||undefined));
  if(snapshot.routes.length>8){
    const search=field(wrapper,'Find route',input());search.placeholder='Name, model, or endpoint';
    search.oninput=()=>{const previous=route.value;route.replaceChildren();for(const r of snapshot.routes.filter(r=>routeMatches(r,search.value))){const option=node('option',r.id);option.value=r.key;route.append(option);}if([...route.options].some(o=>o.value===previous))route.value=previous;else {const placeholder=node('option','Choose a matching route');placeholder.value='';route.prepend(placeholder);route.value='';route.onchange?.();}};
  }
  return route;
}
function routeArgs(s) { const r=snapshot.routes.find(r=>r.key===s.value); if (!r) throw new Error('Choose a route first.'); return {route:r.key,id:r.id}; }
let generation;
function render() {
  if (!snapshot) return;
  generation=snapshot.generation;
  disposeCharts();
  content.replaceChildren();
  document.querySelector('#title').textContent=pageNames[page];
  document.querySelectorAll('[data-page]').forEach(b=>{b.classList.toggle('active',b.dataset.page===page);if(b.dataset.page===page)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  if (page==='overview') {
    renderOverview();
  } else if (page==='routes') {
    renderRouteBrowser();

  } else if (page==='install') {
    const intro=card('Connect your coding tools');
    intro.append(node('p','Choose a Switchyard route for each coding tool, review the settings below, then apply routing settings.'));
    const help=node('details');help.append(node('summary','How accounts and settings work'));help.append(node('p','Subscription routes use the coding tool’s login. API routes use credentials configured on the Switchyard server. Keys entered in Routes only load model lists.'));help.append(node('p','Installation backs up the original settings. Shell variables and project settings may override these defaults.'));intro.append(help);
    const tool=field(intro,'Coding tool',select(snapshot.tools.map(t=>[t.tool,`${t.label} · ${t.available?'Detected':'Not detected'}`]),installView.tool||snapshot.tools[0].tool));
    installView.tool=tool.value;
    const panel=node('div');content.append(panel);
    const show=()=>{installView.tool=tool.value;installView.account='';installView.file='';installView.mode='default';panel.replaceChildren();renderInstall(snapshot.tools.find(t=>t.tool===tool.value),panel);};
    tool.onchange=show;renderInstall(snapshot.tools.find(t=>t.tool===tool.value),panel);
    renderAccounts();
  } else if(page==='usage')renderUsage();
  else if(page==='sessions'){renderSessions();renderAccounts();}
  else if(page==='settings')renderSettings();
  showFreshness();
}
function usageUnavailable() {return snapshot.data_state?.usage?.state==='unavailable'||snapshot.data_state?.history?.state==='unavailable'||snapshot.usage_status==='unavailable'||snapshot.history_status==='unavailable'||(snapshot.errors||[]).some(e=>/read.*(usage|history|routing log)|usage.*read/i.test(e));}
// The final tie-breaker keeps calls with later log positions first across refreshes.
function sortedCalls(entries, order) {
  const compareText=(a,b)=>a<b?-1:a>b?1:0;
  return entries.map((entry,index)=>({entry,index,time:Date.parse(entry.ts)})).sort((a,b)=>{
    const validA=Number.isFinite(a.time),validB=Number.isFinite(b.time);
    let time=0;
    if(validA!==validB)time=validA?-1:1;
    else if(validA)time=order==='oldest'?a.time-b.time:b.time-a.time;
    const model=compareText(a.entry.model||'',b.entry.model||'');
    const tokens=(b.entry.prompt_tokens+b.entry.completion_tokens)-(a.entry.prompt_tokens+a.entry.completion_tokens);
    let primary=0;
    if(order==='model')primary=model;
    else if(order==='tokens')primary=tokens;
    return primary||time||model||b.index-a.index;
  }).map(row=>row.entry);
}
function renderUsage() {
  const c=card('Model calls');c.append(node('p','Each row shows a completed model call. A turn may include several calls. Choose the order below.'));
  const entries=snapshot.entries,dates=entries.map(e=>new Date(e.ts)).filter(d=>!Number.isNaN(d.getTime())).sort((a,b)=>a-b);
  c.append(node('p',`${entries.length} retained calls${snapshot.limited?' · limited to 8 MiB / 5,000 records':''} · ${snapshot.skipped} unreadable recent records. ${dates.length?`${dates[0].toLocaleString()} – ${dates.at(-1).toLocaleString()}`:'No retained timestamps.'}`,'muted'));
  if(usageView.exact){const pill=node('div',undefined,'usage-filter');pill.append(node('span',`${usageView.exact.group}: ${usageView.exact.id||'Not recorded'}`));button(pill,'Clear comparison',()=>{usageView.exact=null;render();});c.append(pill);}
  const filters=node('div',undefined,'field-grid');c.append(filters);
  const period=field(filters,'Usage period',select([['all','All recent history'],['week','Seven local calendar days'],['today','Today']],usageView.period));
  const session=field(filters,'Session',select([['','All sessions'],['__missing','Session not recorded'],...snapshot.sessions.map(s=>[s,s])],usageView.session));
  const search=field(filters,'Search turn, model, route, or session',input(usageView.search));
  const role=field(filters,'Call role',select([['all','All calls'],['answer','Answer calls'],['overhead','Routing overhead']],usageView.role));
  const sort=field(filters,'Sort calls',select([['newest','Time: newest first'],['oldest','Time: oldest first'],['model','Model: A to Z'],['tokens','Total tokens: highest first']],usageView.sort));
  const tally=node('p',undefined,'muted');c.append(tally);
  const a=actions(c);button(a,'Clear all filters',()=>{Object.assign(usageView,{session:'',search:'',exact:null,role:'all',turn:'',period:'all'});render();});
  button(a,'Copy filtered calls',()=>copyText(JSON.stringify(filteredCalls(),null,2)));
  const container=node('div',undefined,'scroll');container.tabIndex=0;container.setAttribute('role','region');container.setAttribute('aria-label','Model calls table');c.append(container);
  function filteredCalls(){return sortedCalls(snapshot.entries,usageView.sort).filter(e=>{
    const exact=usageView.exact;if(exact&&(e[{model:'model',route:'route_id',session:'session_id'}[exact.group]]||'')!==exact.id)return false;
    if(!inPeriod(e.ts,usageView.period,snapshot.today))return false;
    if(usageView.session==='__missing'?Boolean(e.session_id):usageView.session&&e.session_id!==usageView.session)return false;
    if(usageView.turn&&e.turn_id!==usageView.turn)return false;
    if(usageView.role!=='all'&&(usageView.role==='overhead')!==(e.tier==='classifier'))return false;
    return [e.model,e.turn_id,e.route_id,e.session_id].join(' ').toLowerCase().includes(usageView.search.toLowerCase());
  });}
  const draw=()=>{
    usageView.session=session.value;usageView.search=search.value;usageView.period=period.value;usageView.role=role.value;usageView.sort=sort.value;
    container.replaceChildren();const rows=filteredCalls();tally.textContent=`${rows.length} matching of ${entries.length} retained calls${usageView.turn?` · Turn: ${usageView.turn}`:''}`;
    if(usageUnavailable()){container.append(node('p','Usage unavailable. Check the routing log in Settings and retry.','empty-state'));button(container,'Retry',()=>refresh());return;}
    const table=node('table'),thead=node('thead'),head=node('tr'),tbody=node('tbody');
    for(const label of ['Time','Model / role','Session / turn','Route','Uncached input','Cached input','Output']){const th=node('th',label);th.scope='col';head.append(th);}thead.append(head);table.append(thead,tbody);
    for(const e of rows){const row=node('tr'),time=node('td'),date=new Date(e.ts);time.title=e.ts;time.textContent=Number.isNaN(date.getTime())?e.ts:date.toLocaleString();
      const model=node('td');model.append(node('span',e.model,'table-primary'),node('small',e.tier==='classifier'?'Routing overhead':'Answer call','table-secondary'));
      const ids=node('td');if(e.session_id)button(ids,e.session_id,()=>{usageView.session=e.session_id;render();});else ids.append(node('span','Session not recorded'));if(e.turn_id){button(ids,e.turn_id,()=>{usageView.turn=e.turn_id;draw();});button(ids,'Copy turn ID',()=>copyText(e.turn_id));}else ids.append(node('small','Turn not recorded'));
      row.append(time,model,ids,node('td',e.route_id||'Route not recorded'));for(const value of [Math.max(0,e.prompt_tokens-e.cached_tokens),e.cached_tokens,e.completion_tokens])row.append(node('td',value.toLocaleString()));tbody.append(row);
    }container.append(table);
    if(!rows.length)container.append(node('p',entries.length?'No calls match these filters. A selected session may be outside recent history.':'No calls recorded yet. Start a session, then return here.','empty-state'));
    if(usageView.exact&&snapshot.limited)container.prepend(node('p','Comparison totals cover the full log; this table only includes the retained range shown above.','coverage-note'));
  };for(const control of [period,session,role,sort])control.onchange=draw;search.oninput=draw;draw();
}
async function choosePath(control,directory,changed){try{const result=await request('choose_path',{directory});const picked=typeof result==='string'?result:result?.data?.path||result?.path;if(picked&&control.isConnected){control.value=picked;changed();}}catch(error){notify(`Could not choose a ${directory?'folder':'file'}: ${error}`,'error');}}
async function copyText(value) {try{await navigator.clipboard.writeText(value);notify('Copied.');}catch{notify('Copy failed. Select and copy the text manually.','error');}}
function sessionPreviewText(data){return [`Repository: ${data.repository}`,`Starting folder: ${data.relative_directory?`${data.relative_directory} within the new worktree`:'New worktree root'}`,`Branch: ${data.branch||'Detached HEAD'}`,`Revision: ${data.revision}`,`Route: ${data.route}`,`Account: ${data.account_directory||'Current login'}`,data.dirty?'Your checkout has uncommitted changes.':'',data.notice,data.settings_notice].filter(Boolean).join('\n\n');}
function renderSessions() {
  const c=card('New session');c.append(node('p','Create a Git worktree from committed HEAD and open the coding tool in Terminal. Uncommitted changes stay in the original checkout.'));
  const tools=snapshot.tools.filter(t=>['codex_cli','claude','pi'].includes(t.tool));
  const tool=field(c,'Session coding tool',select(tools.map(t=>[t.tool,`${t.label}${t.available?'':' · not detected'}`]),sessionView.tool||tools[0]?.tool));sessionView.tool=tool.value;
  const account=field(c,'Session account',select([['','Current login'],...(tools.find(t=>t.tool===tool.value)?.accounts||[]).map(n=>[n,n])],sessionView.account));
  const route=routeField(c,sessionView.route,tool.value);
  const project=field(c,'Project folder (absolute path)',input(sessionView.project));project.placeholder='/Users/you/project';
  const info=node('p',undefined,'muted');c.append(info);const preview=node('div');preview.setAttribute('role','status');preview.setAttribute('aria-live','polite');c.append(preview);let plan;
  const a=actions(c),review=button(a,'Review session',async()=>{const reviewed=args();preview.setAttribute('aria-busy','true');preview.textContent='Checking the project and session settings…';try{const r=await readAction({kind:'preview_launch',...reviewed});if(JSON.stringify(reviewed)!==JSON.stringify(args())){preview.textContent='Session inputs changed. Review them again.';return;}plan={reply:r,args:structuredClone(reviewed)};preview.replaceChildren(node('pre',sessionPreviewText(r.data)));r.message=sessionPreviewText(r.data);blocked(launch,false);}catch(error){fieldError(project,String(error));preview.textContent='Session preview unavailable. Correct the project or selected settings and retry.';}finally{preview.setAttribute('aria-busy','false');}}),launch=mutation(a,'Open Terminal…',()=>{if(!plan)return;const reviewed=structuredClone(plan.args),reply=plan.reply;confirmAction('Start this session?',reply.message,'Open Terminal',()=>run({kind:'launch',...reviewed,preview_token:reply.data.preview_token}));},true);
  const args=()=>({tool:tool.value,account:account.value||null,...routeArgs(route),project:project.value});
  const changed=()=>{Object.assign(sessionView,{tool:tool.value,account:account.value,route:route.value,project:project.value});plan=null;preview.replaceChildren();blocked(launch,true);const t=tools.find(t=>t.tool===tool.value);const incompatible=snapshot.routes.find(r=>r.key===route.value)?.compatibility?.[tool.value];const invalid=!project.value.startsWith('/')?'Enter an absolute project folder. Git validation runs during Review.':'';fieldError(project,invalid);blocked(review,!t?.available||!route.value||Boolean(invalid)||Boolean(incompatible));if(incompatible)preview.append(node('p',incompatible,'field-error'));info.textContent=account.value?`Saved account: ${account.value}. ${t?.account_details?.find(a=>a.name===account.value)?.path||''}. Login status is unknown until the coding tool confirms it.`:t?.status||'Select a coding tool.';if(tool.value==='pi')info.append(node('p','Pi uses isolated session settings. Personal settings and extensions are not copied.'));};
  button(a,'Choose project folder…',()=>choosePath(project,true,changed));
  tool.onchange=()=>{sessionView.account='';sessionView.tool=tool.value;render();};account.onchange=changed;route.onchange=changed;project.oninput=changed;changed();
  if(!snapshot.routes.length){c.append(node('p','Add a route before starting a session.'));button(a,'Choose a route',()=>navigate('routes'));}
  c.append(node('p','Codex app workspaces are managed in the Codex app. Supported command line tools appear above.','muted'));
  const inventory=card('Recorded sessions');inventory.append(node('p','Recorded calls do not establish that a process is running.'));
  if(snapshot.session_inventory_error)inventory.append(node('p',snapshot.session_inventory_error,'field-error'));
  for(const item of snapshot.session_inventory||[]){const row=node('div',undefined,'route-row'),last=snapshot.entries.filter(e=>e.session_id===item.id).at(-1);row.append(node('h4',item.id),node('p',[item.repository,item.branch,item.route,item.worktree,item.created_at,last?`Last recorded call: ${last.ts}`:'No recent recorded call'].filter(Boolean).join(' · ')));button(actions(row),'Copy path',()=>copyText(item.worktree));button(actions(row),'Open folder',()=>run({kind:'open_session',id:item.id},false));mutation(actions(row),'Review cleanup…',async()=>{const result=await run({kind:'preview_session_cleanup',id:item.id},false);if(!result)return;const data=result.data;confirmAction('Remove session worktree?',`${data.cleanup_notice}\n\nDirectory: ${data.worktree}\nBranch: ${data.current_branch}\nRevision: ${data.current_revision}\n${data.dirty?'This worktree has uncommitted changes. Cleanup will refuse to remove it.':'The worktree has no uncommitted changes.'}`,'Remove session',()=>run({kind:'remove_session',id:item.id,preview_token:data.preview_token}));});inventory.append(row);}
  if(!(snapshot.session_inventory||[]).length){inventory.append(node('p','Switchyard has not recorded any managed sessions.'));for(const id of snapshot.sessions){button(actions(inventory),id,()=>{usageView.session=id;navigate('usage');});}}
}
function updatePreviewText(data){return [`Source checkout: ${data.source}`,`Branch: ${data.branch||'Detached HEAD'}`,`Revision: ${data.revision}`,`Local changes: ${data.changes||'None'}`,Array.isArray(data.plan)?data.plan.join('\n'):data.plan].filter(Boolean).join('\n\n');}
function renderSettings() {
  const c=card('Server and app');const paths=snapshot.settings||{};
  for(const [label,value] of Object.entries(paths))if(typeof value==='string')c.append(node('p',`${label.replaceAll('_',' ')}: ${value}`));
  if(snapshot.version)c.append(node('p',`Installed version: ${snapshot.version}`));
  const a=actions(c);button(a,'Copy diagnostics',()=>copyText(JSON.stringify({app:'Switchyard desktop',version:snapshot.version||'Not reported',last_check:lastRefresh?.toISOString(),stale,server:snapshot.metrics,paths:Object.fromEntries(Object.entries(snapshot.settings||{}).filter(([key,value])=>typeof value==='string'&&['path','config_file','routing_log'].includes(key))),routes:snapshot.routes.map(r=>r.id),tools:snapshot.tools.map(t=>({label:t.label,available:t.available})),update:snapshot.update_status?.state||'Unknown'},null,2)));for(const [label,kind] of [['Open server config','open_config'],['Open app settings','open_settings'],['Check config','check_config'],['Restart and check health','restart'],['Open server logs','open_logs']])mutation(a,label,()=>run({kind}));
  c.append(node('p','Set baseline model, model prices, server address, and refresh interval in app settings. Prices use dollars per million tokens.'));
  const update=card('Update from source');update.append(node('p','Review the source checkout and service restarts before building.'));
  if(snapshot.update_status){const state=snapshot.update_status;update.append(node('p',`${state.state||'No update running'}${state.stage?` · ${state.stage}`:''}: ${state.message||''}`));}
  button(actions(update),'Review update…',async()=>{const result=await run({kind:'preview_update'},false);if(!result)return;const unsaved=[...drafts.values()].some(d=>d.dirty);confirmAction('Build and reinstall?',`${updatePreviewText(result.data)}${unsaved?'\n\nYou have unsaved route edits. Cancel and save or discard them before updating.':''}`,'Build and reinstall',()=>{if(unsaved){notify('Save or discard route drafts before updating.','warning');return;}run({kind:'update',preview_token:result.data.preview_token});});});
  button(actions(update),'Open update log',()=>run({kind:'open_update_log'},false));
  renderPreferences();
  const display=card('Display');const zoom=field(display,'Text size',select([['100','100%'],['125','125%'],['150','150%'],['200','200%']],String(Math.round(Number(document.documentElement.style.zoom||1)*100))));zoom.onchange=()=>{document.documentElement.style.zoom=String(Number(zoom.value)/100);document.documentElement.style.setProperty('--ui-scale',String(Number(zoom.value)/100));document.body.classList.toggle('enlarged',Number(zoom.value)>=150);};
  const recent=card('Recent actions');for(const item of activity)recent.append(node('p',`${item.time} · ${item.message}`));if(!activity.length)recent.append(node('p','Actions from this app session appear here.'));
  c.append(node('p','Closing the window keeps the app in the menu bar and the server running. Quit Switchyard exits the app; Restart controls the server.'));text(c,'Terminal: switchyard-desktop --tui');
}

function renderPreferences() {
  const c=card('Usage estimates and refresh');
  if(!preferencesDraft)preferencesDraft={baseline:snapshot.baseline_model,refresh:String(snapshot.refresh_seconds),prices:Object.entries(snapshot.settings?.prices||{}).map(([model,p])=>({model,input:String(p.input_per_mtok),cached:p.cached_input_per_mtok===null||p.cached_input_per_mtok===undefined?'':String(p.cached_input_per_mtok),output:String(p.output_per_mtok)}))};
  const draft=preferencesDraft;
  const baseline=field(c,'Baseline model ID',input(draft.baseline)),refreshInterval=field(c,'Refresh interval in seconds',input(draft.refresh,'number'));refreshInterval.min='1';refreshInterval.step='1';
  c.append(node('p','Rates are dollars per million tokens. Leave cached input empty to use the input rate. Estimates need prices for every recorded model and the baseline.'));
  const rows=node('div');c.append(rows);
  const draw=()=>{rows.replaceChildren();for(const [index,price] of draft.prices.entries()){const row=node('div',undefined,'price-row field-grid');rows.append(row);for(const [key,label] of [['model','Price model ID'],['input','Uncached input rate'],['cached','Cached input rate'],['output','Output rate']]){const control=field(row,`${label} ${index+1}`,input(price[key],key==='model'?'text':'number'));if(key!=='model'){control.min='0';control.step='any';}control.oninput=()=>{price[key]=control.value;fieldError(control,'');};}button(actions(row),'Remove price',()=>{draft.prices.splice(index,1);draw();});}};draw();
  button(actions(c),'Add model price',()=>{draft.prices.push({model:'',input:'',cached:'',output:''});draw();});
  baseline.oninput=()=>{draft.baseline=baseline.value;fieldError(baseline,'');};refreshInterval.oninput=()=>{draft.refresh=refreshInterval.value;fieldError(refreshInterval,'');};
  mutation(actions(c),'Save app settings',async()=>{
    let valid=fieldError(baseline,!draft.baseline.trim()||/[\n\r]/.test(draft.baseline)?'Enter a baseline model ID without line breaks.':'');
    valid=fieldError(refreshInterval,!Number.isSafeInteger(Number(draft.refresh))||Number(draft.refresh)<1?'Enter a whole number of at least one second.':'')&&valid;
    const prices={},seen=new Set();for(const [index,price] of draft.prices.entries()){const controls=rows.children[index].querySelectorAll('input');const modelError=!price.model.trim()?'Enter a model ID.':seen.has(price.model)?'This model already has a price.':'';valid=fieldError(controls[0],modelError)&&valid;seen.add(price.model);
      for(const [offset,key] of [[1,'input'],[2,'cached'],[3,'output']]){const missing=key!=='cached'&&!price[key].trim();const invalid=missing||price[key]!==''&&(!Number.isFinite(Number(price[key]))||Number(price[key])<0);valid=fieldError(controls[offset],invalid?'Enter a finite rate of zero or more.':'')&&valid;}
      prices[price.model]={input_per_mtok:Number(price.input),cached_input_per_mtok:price.cached===''?null:Number(price.cached),output_per_mtok:Number(price.output)};
    }
    if(!valid){c.querySelector('[aria-invalid="true"]')?.focus();return;}
    const result=await run({kind:'save_preferences',baseline_model:draft.baseline,refresh_seconds:Number(draft.refresh),prices});if(result){preferencesDraft=undefined;render();}
  },true);
}

function draftKey(route) {return `${snapshot.settings?.config_file||generation}:${route.key}`;}
function routeSummary(route){return route.choices.map(c=>`${c.client} / ${c.model}`).join(' → ');}
function renderRouteBrowser() {
  const c=card('Route library'),a=actions(c);button(a,'Create route…',()=>routeIdentity('create_route'));
  if([...drafts.keys()].some(k=>!k.startsWith(`${snapshot.settings?.config_file||generation}:`)))c.append(node('p','Drafts from another config file are retained in this app session. Return to that config to finish or discard them.','warning-note'));
  if(!snapshot.routes.length){c.append(node('p',(snapshot.errors||[]).length?'Routes could not be loaded. Open the server config to fix the error.':'No routes configured. Create a route or open the server config.'));button(a,'Open server config',()=>run({kind:'open_config'},false));return;}
  const search=field(c,'Find a route by name, routing method, endpoint, or model',input(routeView.search));
  const count=node('p',undefined,'muted'),layout=node('div',undefined,'route-workspace'),list=node('div',undefined,'route-picker'),detail=node('div',undefined,'route-detail');
  c.append(count,layout);layout.append(list,detail);
  const draw=()=>{
    routeView.search=search.value;const routes=snapshot.routes.filter(r=>routeMatches(r,search.value));count.textContent=`${routes.length} of ${snapshot.routes.length} routes`;list.replaceChildren();detail.replaceChildren();
    if(!routeView.selected)routeView.selected=routes[0]?.key||'';
    for(const route of routes){const b=button(list,'',()=>{routeView.selected=route.key;draw();list.querySelector(`[data-route="${CSS.escape(route.key)}"]`)?.focus();});b.dataset.route=route.key;b.classList.toggle('selected',route.key===routeView.selected);b.setAttribute('aria-pressed',String(route.key===routeView.selected));b.append(node('span',route.id,'route-name'),node('small',snapshot.algorithms.find(a=>a.kind===route.kind)?.title||route.kind),node('small',routeSummary(route)));const badge=node('small','Unsaved edits','draft-badge');badge.hidden=!drafts.get(draftKey(route))?.dirty;b.append(badge);}
    if(!routes.length){list.append(node('p','No routes match your search.','empty-state'));button(list,'Clear search',()=>{search.value='';draw();});return;}
    const route=routes.find(r=>r.key===routeView.selected);
    if(route)renderRoute(route,detail);else detail.append(node('p','Selected route is outside this filter. Clear the search to return to it.','empty-state'));
  };search.oninput=draw;draw();
}
function routeIdentity(kind,route) {
  const overlay=node('div',undefined,'modal-backdrop'),form=node('form',undefined,'confirm-dialog');form.setAttribute('role','dialog');form.setAttribute('aria-modal','true');form.setAttribute('aria-label',kind==='create_route'?'Create route':kind==='rename_route'?'Rename route':'Duplicate route');overlay.append(form);document.body.append(overlay);
  const siblings=[...document.body.children].filter(n=>n!==overlay).map(n=>[n,n.inert]);siblings.forEach(([n])=>{n.inert=true;});const previous=document.activeElement;
  const close=()=>{overlay.remove();siblings.forEach(([n,inert])=>{n.inert=inert;});if(previous?.isConnected)previous.focus();else document.querySelector('#title').focus();};
  form.append(node('h3',kind==='create_route'?'Create route':kind==='rename_route'?'Rename route':'Duplicate route'));
  const id=field(form,'Public model ID',input(kind==='rename_route'?route.id:'')),name=kind==='rename_route'?null:field(form,'Config route name',input());form.append(node('small','The public model ID is the name coding tools request. The config name uses 1–64 letters, digits, hyphens, or underscores.'));let client,model;
  if(kind==='create_route'){client=field(form,'Endpoint',select(snapshot.clients.map(c=>[c.name,`${c.name} (${c.host})`])));model=field(form,'Model ID',input());}
  const a=actions(form);button(a,'Cancel',close).type='button';const save=button(a,'Review changes…',()=>{},true);save.type='submit';
  form.onsubmit=async event=>{
    event.preventDefault();let valid=fieldError(id,!id.value.trim()||/[\n\r]/.test(id.value)?'Enter a public model ID without line breaks.':snapshot.routes.some(r=>r.id===id.value&&!(kind==='rename_route'&&r.key===route?.key))?'That public model ID already exists.':'');
    if(name)valid=fieldError(name,!/^[A-Za-z0-9_-]{1,64}$/.test(name.value)?'Use 1–64 letters, digits, hyphens, or underscores.':snapshot.routes.some(r=>r.key===name.value)?'That config name already exists.':'')&&valid;
    if(model)valid=fieldError(model,!model.value.trim()?'Enter a model ID.':'')&&valid;if(!valid){form.querySelector('[aria-invalid="true"]')?.focus();return;}
    const args={generation,operation:kind.replace('_route',''),route:route?.key||'',name:name?.value||'',id:id.value,algorithm:client?'passthrough':null,choices:client?[{client:client.value,model:model.value}]:[]};
    save.disabled=true;const preview=await run({kind:'preview_route_change',...args},false);save.disabled=false;if(!preview)return;
    if(id.value!==args.id||(name&&name.value!==args.name)||(model&&model.value!==args.choices[0].model)||(client&&client.value!==args.choices[0].client)){notify('The form changed during review. Review the current values again.','warning');return;}
    confirmAction('Save this route change?', [`Changed file: ${preview.data.file}`,...preview.data.notes,'Switchyard validates the config, saves a backup, and restarts the server.'].join('\n\n'),'Save and restart',async()=>{form.querySelectorAll('input,select,button').forEach(control=>{control.disabled=true;});const result=await run({kind:'route_change',...args,preview_token:preview.data.preview_token});if(result)close();else form.querySelectorAll('input,select,button').forEach(control=>{control.disabled=false;});});
  };

  form.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();close();}if(event.key==='Tab'){const controls=[...form.querySelectorAll('input,select,button')],i=controls.indexOf(document.activeElement);if(event.shiftKey&&i===0){event.preventDefault();controls.at(-1).focus();}else if(!event.shiftKey&&i===controls.length-1){event.preventDefault();controls[0].focus();}}};id.focus();
}
// A preview belongs to its exact inputs; obsolete replies never enable Apply.
function renderInstall(tool,parent) {
  const c=card(tool.label);parent.append(c);c.dataset.tool=tool.tool;
  c.append(node('p',tool.available?'Coding tool detected.':'Coding tool not detected. You can prepare settings and install the tool later.','muted'));
  c.append(node('p',{codex_cli:'Choose the Codex CLI profile file to change.',codex_app:'Detected user settings change defaults shared by the Codex app and CLI. Custom files must be loaded by the coding tool.',claude:'Start a new Claude Code session after changing its settings.',pi:'Pi uses models.json and settings.json together. Model capabilities and prices are presets; check them for your chosen route.'}[tool.tool]));
  const fields=node('div',undefined,'field-grid');c.append(fields);
  const mode=field(fields,'Destination',select([['default','Detected settings'],['account','Saved account'],['custom','Custom file']],installView.mode));
  const account=field(fields,'Saved account',select(tool.accounts.map(name=>[name,name]),installView.account));
  const file=field(fields,tool.tool==='pi'?'Custom models.json path':'Custom settings file path',input(installView.file));file.placeholder=tool.files?.[0]||'/absolute/path/to/settings';
  const chooseFile=button(c,'Choose settings file…',()=>choosePath(file,false,update));
  const route=routeField(fields,installView.route,tool.tool);
  const comparison=node('div',undefined,'install-diff'),authentication=node('p',undefined,'install-auth'),progress=node('p',undefined,'install-progress');progress.setAttribute('role','status');progress.setAttribute('aria-live','polite');c.append(comparison,authentication,progress);
  const target=()=>({tool:tool.tool,account:mode.value==='account'?account.value||null:null,settings_file:mode.value==='custom'?file.value||null:null});
  let plan,revision=0;const a=actions(c);
  const install=mutation(a,'Apply routing…',()=>{if(!plan)return;const selected=plan;confirmAction(`Apply routing to ${tool.label}?`,`${selected.data.proposed}\n\n${selected.data.authentication}\n\nFiles: ${[...new Set(selected.data.changes.map(change=>change.file))].join(', ')}. Switchyard backs up the originals and preserves unrelated settings.`, 'Apply routing',async()=>{if(!c.isConnected||selected!==plan)return;progress.textContent='Saving settings…';const result=await run(selected.action,false);progress.textContent='';if(result)update();});},true);
  const restore=mutation(a,'Review restore…',async()=>{
    blocked(install,true);plan=null;progress.textContent='Checking backup…';
    const result=await run({kind:'preview_restore',...target()},false);progress.textContent='';if(!result)return;
    const selectedTarget=target(),data=result.data;
    confirmAction('Restore coding-tool settings?',[data.warning,...(data.files||[]).map(file=>`${file.action}: ${file.file}${file.backup?`\nOriginal backup: ${file.backup}`:''}`)].join('\n\n'),'Restore settings',async()=>{progress.textContent='Restoring settings…';await run({kind:'restore',...selectedTarget,preview_token:data.preview_token},false);progress.textContent='';update();},update);
  });
  const update=()=>{
    Object.assign(installView,{mode:mode.value,account:account.value,route:route.value,file:file.value});
    account.parentElement.hidden=mode.value!=='account';file.parentElement.hidden=mode.value!=='custom';chooseFile.hidden=mode.value!=='custom';
    const requested=++revision;plan=null;blocked(install,true);install.textContent='Checking settings…';comparison.setAttribute('aria-busy','true');comparison.replaceChildren(node('p','Loading settings preview…'));authentication.textContent='';progress.textContent='Checking preview…';
    const invalid=mode.value==='custom'&&!file.value.startsWith('/');fieldError(file,invalid?'Enter an absolute settings file path.':'');
    if(invalid||mode.value==='account'&&!account.value){comparison.replaceChildren(node('p',invalid?'Enter a settings file path.':'No saved account is available. Sign in below or choose Detected settings.'));install.textContent='Preview required';comparison.setAttribute('aria-busy','false');progress.textContent='';blocked(restore,true);return;}
    blocked(restore,true);
    const incompatible=snapshot.routes.find(r=>r.key===route.value)?.compatibility?.[tool.tool];
    if(incompatible){comparison.replaceChildren(node('p',incompatible,'field-error'));install.textContent='Choose a compatible route';progress.textContent='Preview unavailable for this tool.';comparison.setAttribute('aria-busy','false');return;}
    if(!route.value){comparison.replaceChildren(node('p','Choose a route to preview its settings.'));button(comparison,'Choose a route',()=>navigate('routes'));install.textContent='Choose a route';comparison.setAttribute('aria-busy','false');progress.textContent='';return;}
    const requestedTarget=target(),requestedRoute=routeArgs(route);
    enqueue(async()=>{
      if(!c.isConnected||requested!==revision)return;
      try{
        let backupReady=false;try{await readAction({kind:'preview_restore',...requestedTarget});backupReady=true;}catch(error){restore.title=String(error);}
        if(!c.isConnected||requested!==revision)return;blocked(restore,!backupReady);
        const result=await readAction({kind:'preview_install',...requestedTarget,...requestedRoute});if(!c.isConnected||requested!==revision)return;
        comparison.replaceChildren();const summaries=node('div',undefined,'install-summary');comparison.append(summaries);
        for(const [title,value] of [['Current settings',result.data.current],['After applying routing',result.data.proposed]])if(value){const summary=node('div');summary.append(node('h4',title),node('pre',value));summaries.append(summary);}
        for(const file of result.data.file_actions||[])summaries.append(node('p',`${file.action==='create'?'Will create':file.action==='preserve'?'Will preserve':'Will update'}: ${file.file}`));
        authentication.textContent=result.data.authentication||'';
        if(result.data.error){comparison.append(node('p',result.data.error,'field-error'));install.textContent='Preview unavailable';progress.textContent='Preview unavailable. Review the error.';blocked(restore,!backupReady);return;}
        const grouped=new Map();for(const change of result.data.changes){if(!grouped.has(change.file))grouped.set(change.file,[]);grouped.get(change.file).push(change);}
        for(const [path,changes] of grouped){const details=node('details');details.append(node('summary',`${path} · ${changes.length} changed settings`));for(const change of changes)details.append(node('pre',change.key,'diff-heading'),node('pre',`− ${change.before}`,'diff-remove'),node('pre',`+ ${change.after}`,'diff-add'));comparison.append(details);}
        if(!result.data.changes.length)comparison.append(node('p','Settings already match this route. Saved settings do not confirm that the server is reachable or that the tool has loaded them.'));
        const ready=result.data.changes.length>0;if(ready)plan={data:result.data,action:{kind:'install',...requestedTarget,...requestedRoute,preview_token:result.data.preview_token}};
        install.textContent=ready?'Apply routing…':'Routing already applied';blocked(install,!ready);blocked(restore,!backupReady);progress.textContent=ready?'The preview shows the files and authentication above. Review them before applying routing.':'Routing already applied.';
      }catch(error){if(c.isConnected&&requested===revision){install.textContent='Preview unavailable';comparison.replaceChildren(node('p',String(error),'field-error'));progress.textContent='Preview unavailable.';button(comparison,'Retry preview',update);}}
      finally{if(c.isConnected&&requested===revision)comparison.setAttribute('aria-busy','false');}
    });
  };
  for(const control of [mode,account,route])control.onchange=update;
  file.oninput=()=>{installView.file=file.value;revision++;plan=null;blocked(install,true);blocked(restore,true);install.textContent='Preview required';progress.textContent='Finish entering the file path, then press Enter or leave the field to preview.';};file.onchange=update;file.onkeydown=event=>{if(event.key==='Enter')update();};update();
}
function renderAccounts() {
  const c=card('Subscription accounts');c.append(node('p','Account folders keep separate coding-tool settings. Folder existence does not confirm a completed login.'));
  for(const tool of snapshot.tools.filter(t=>['codex_cli','claude'].includes(t.tool))){
    if(tool.discovery_error)c.append(node('p',`${tool.label}: ${tool.discovery_error}`,'field-error'));
    for(const item of tool.account_details||tool.accounts.map(name=>({name,status:'Login status unknown'}))){const row=node('div',undefined,'route-row');row.append(node('h4',`${tool.label} · ${item.name}`),node('p',item.status||'Login status unknown'),node('pre',item.path||''));if(item.path){button(actions(row),'Copy account path',()=>copyText(item.path));button(actions(row),'Open folder',()=>run({kind:'open_account',tool:tool.tool,name:item.name},false));}mutation(actions(row),'Sign in again…',()=>run({kind:'login_account',tool:tool.tool,name:item.name}));mutation(actions(row),'Review archive…',async()=>{const result=await run({kind:'preview_account_archive',tool:tool.tool,name:item.name},false);if(result)confirmAction('Archive this account folder?',`${result.data.notice}\n\nAccount: ${result.data.name}\nDirectory: ${result.data.path}`,'Archive account',()=>run({kind:'archive_account',tool:tool.tool,name:item.name,preview_token:result.data?.preview_token}));});c.append(row);}
  }
  const disclosure=node('details');disclosure.append(node('summary','Sign in to another subscription account…'));c.append(disclosure);
  disclosure.append(node('p','Finish login in the Terminal window. API routes use credentials configured on the server. The Codex app manages its own account.'));
  const tool=field(disclosure,'Account coding tool',select(snapshot.tools.filter(t=>['codex_cli','claude'].includes(t.tool)).map(t=>[t.tool,t.label]),accountView.tool));
  const name=field(disclosure,'Name for the separate account',input(accountView.name));disclosure.append(node('small','Use 1–64 ASCII letters, digits, hyphens, or underscores.'));
  const add=mutation(actions(disclosure),'Open login in Terminal…',async()=>{const result=await run({kind:'add_account',tool:tool.value,name:name.value});if(result){accountView.name='';render();}});
  const changed=()=>{accountView.name=name.value;accountView.tool=tool.value;const error=!/^[A-Za-z0-9_-]{1,64}$/.test(name.value)?'Use 1–64 letters, digits, hyphens, or underscores.':snapshot.tools.find(t=>t.tool===tool.value)?.accounts.includes(name.value)?'That account name already exists. Choose another name.':'';fieldError(name,error);blocked(add,Boolean(error)||!snapshot.tools.find(t=>t.tool===tool.value)?.available);};name.oninput=changed;tool.onchange=changed;changed();
}
// Every algorithm retains its complete choices until the user saves or discards the draft.
function modelListKey(client){return `${snapshot.settings?.config_file||generation}:${snapshot.clients.find(endpoint=>endpoint.name===client)?.host||''}:${client}`;}
function roleChoice(draft,route,role,index) {
  return draft.byAlgorithm?.get(draft.algorithm)?.[index] || (role.tier && draft.memory.get(role.tier)) ||
    ((draft.algorithm===route.kind||draft.loadedAlgorithm===draft.algorithm)&&draft.choices[index]) ||
    {client:route.choices[0]?.client||snapshot.clients[0]?.name||'',model:''};
}
function draftDirty(draft,route){return draft.algorithm!==route.kind||JSON.stringify(draft.choices)!==JSON.stringify(route.choices);}
function renderRoute(route,parent) {
  const c=card(route.id);parent.append(c);const key=draftKey(route),loadedGeneration=generation;
  if(route.generated){c.append(node('p',`Generated by ${route.generated}. Edit the generator’s source to keep changes across regeneration; generated output may replace edits here.`,'warning-note'));button(actions(c),'Open server config',()=>run({kind:'open_config'},false));}
  const remove=()=>confirmAction(`Delete route “${route.id}”?`,`This removes the public model ID from the server. Targets and coding-tool settings are kept. Any tool using ${route.id} must choose another route. ${route.dependents?.length?`Known settings: ${route.dependents.map(d=>typeof d==='string'?d:`${d.label}${d.account?` (${d.account})`:''}: ${d.files.join(', ')}`).join('; ')}.`:''}`,'Delete route',()=>run({kind:'remove',generation:loadedGeneration,route:route.key,id:route.id,revision:drafts.get(key)?.revision||route.revision}));
  if(!route.editable){c.append(node('p','This routing method has no visual editor. Edit its fields in the server config.'));button(actions(c),'Open server config',()=>run({kind:'open_config'},false));mutation(actions(c),'Delete route…',remove).classList.add('danger');return;}
  let draft=drafts.get(key);
  if(!draft){draft={algorithm:route.kind,choices:route.choices.map(c=>({...c})),memory:new Map(),byAlgorithm:new Map([[route.kind,route.choices.map(c=>({...c}))]]),revision:route.revision,generation:loadedGeneration,dirty:false};route.tiers?.forEach((tier,index)=>{if(tier)draft.memory.set(tier,{...route.choices[index]});});drafts.set(key,draft);}
  const algorithm=field(c,'Routing algorithm',select(snapshot.algorithms.map(a=>[a.kind,a.title]),draft.algorithm));
  const description=node('p',undefined,'description'),roles=node('div',undefined,'role-fields'),editorState=node('p',undefined,'muted');editorState.setAttribute('role','status');c.append(description,roles,editorState);
  const a=actions(c);let revision=0;
  if(draft.generation!==loadedGeneration){c.append(node('p','This draft was retained while another server config was selected. Review it against the current saved route before saving.','warning-note'));button(a,'Review restored draft…',()=>confirmAction('Use the current saved route as the baseline?',`Current saved route: ${route.kind} · ${routeSummary(route)}\n\nRetained draft: ${draft.algorithm} · ${draft.choices.map(choice=>`${choice.client} / ${choice.model}`).join(' → ')}\n\nThe next Save review will show settings removed or retained from the current file.`,'Use current baseline',()=>{draft.revision=route.revision;draft.generation=loadedGeneration;render();}));}
  const updateDirty=()=>{draft.dirty=draftDirty(draft,route);editorState.textContent=draft.dirty?'Unsaved edits':'Saved route';blocked(save,!draft.dirty||draft.generation!==loadedGeneration);blocked(revert,!draft.dirty);const badge=document.querySelector(`[data-route="${CSS.escape(route.key)}"] .draft-badge`);if(badge)badge.hidden=!draft.dirty;};
  const saveChoices=()=>{roles.querySelectorAll('[data-role]').forEach(row=>{const controls=row.querySelectorAll('select,input');const choice={client:controls[0].value,model:controls[1].value};draft.choices[Number(row.dataset.role)]=choice;if(row.dataset.tier)draft.memory.set(row.dataset.tier,choice);});draft.byAlgorithm.set(draft.algorithm,draft.choices.map(c=>({...c})));updateDirty();};
  const save=mutation(a,'Review and save…',async()=>{
    saveChoices();let valid=true;roles.querySelectorAll('[data-role] input[type="text"]').forEach(control=>{valid=fieldError(control,!control.value.trim()?'Enter a model ID. Manual IDs are allowed.':'')&&valid;});if(!valid)return;
    const args={generation:loadedGeneration,route:route.key,algorithm:draft.algorithm,choices:draft.choices.map(choice=>({...choice})),revision:draft.revision};
    const preview=await run({kind:'preview_route',...args},false);if(!preview)return;
    if(draft.algorithm!==args.algorithm||JSON.stringify(draft.choices)!==JSON.stringify(args.choices)){notify('The route draft changed during review. Review the current edits again.','warning');return;}
    confirmAction(`Save route “${route.id}”?`,[`Changed file: ${preview.data.file}.`,...preview.data.notes,'Switchyard validates the config, saves a backup, and restarts the server.'].join('\n\n'),'Save and restart',async()=>{const result=await run({kind:'apply',...args});if(result){if(draft.algorithm===args.algorithm&&JSON.stringify(draft.choices)===JSON.stringify(args.choices))drafts.delete(key);else{draft.revision=undefined;notify('Saved the reviewed route. Your newer edits remain unsaved.','warning',result.data?.details||[]);}render();document.querySelector('#title').focus();}});
  },true);
  const revert=mutation(a,'Discard unsaved edits…',()=>{if(!draft.dirty)return;confirmAction('Discard route edits?','This returns the route editor to the saved configuration.','Discard edits',()=>{drafts.delete(key);refresh();});});
  button(a,'Reload saved route…',()=>confirmAction('Reload the saved route?','This discards the current route draft and reads the latest server config.','Reload route',()=>{drafts.delete(key);refresh();}));
  mutation(a,'Duplicate…',()=>routeIdentity('duplicate_route',route));mutation(a,'Rename…',()=>routeIdentity('rename_route',route));mutation(a,'Delete route…',remove).classList.add('danger');
  const draw=async()=>{
    const current=++revision,chosen=draft.algorithm;editorState.textContent='Loading routing roles…';roles.setAttribute('aria-busy','true');
    try{const result=await readAction({kind:'editor',generation:loadedGeneration,route:route.key,algorithm:chosen});if(!c.isConnected||current!==revision)return;
      description.textContent=result.message;if(!draft.revision)draft.revision=result.data.revision;roles.replaceChildren();
      let roleList=result.data.roles;
      if(chosen==='random'&&draft.byAlgorithm.has(chosen)){const template=roleList[0];roleList=draft.byAlgorithm.get(chosen).map((_,i)=>({...template,label:`Model ${i+1}`,tier:null}));}
      draft.choices=roleList.map((role,index)=>({...roleChoice(draft,route,role,index)}));
      roleList.forEach((role,index)=>{
        const choice=draft.choices[index],row=node('div');row.dataset.role=index;row.dataset.tier=role.tier||'';
        const client=field(row,`${role.label} endpoint`,select(snapshot.clients.map(c=>[c.name,`${c.name} (${c.host})`]),choice.client));
        const model=field(row,role.label==='Model'?'Model':`${role.label} model`,input(choice.model));row.append(node('small',role.hint));
        const datalist=node('datalist');datalist.id=`models-${loadedGeneration}-${route.key}-${index}`;model.setAttribute('list',datalist.id);row.append(datalist);
        const guidance=node('div'),listStatus=node('small',undefined,'model-list-status');row.append(guidance,listStatus);
        const fill=()=>{const cached=lists.get(modelListKey(client.value)),values=cached?.models||snapshot.clients.find(c=>c.name===client.value)?.models||[];datalist.replaceChildren();for(const value of values){const option=node('option');option.value=value;datalist.append(option);}listStatus.textContent=`${values.length} suggestions · ${cached?`${cached.source==='cache'?'Switchyard cached suggestions':cached.source==='local'?'Codex cached suggestions':cached.source==='config'?'Configured targets':'Endpoint suggestions'} · ${Number.isFinite(cached.fetchedAt)&&cached.fetchedAt>0?`fetched ${new Date(cached.fetchedAt*1000).toLocaleString()}`:'fetch time unavailable'}${cached.warning?` · refresh failed at ${cached.time}`:''}`:'configured targets'}${values.length?'':' · enter a model ID manually'}.`;
        };
        const endpointGuidance=()=>{guidance.replaceChildren();const endpoint=snapshot.clients.find(e=>e.name===client.value);
          if(endpoint?.note)guidance.append(node('p',endpoint.note,'warning-note'));
          if(endpoint?.accepts_key){const details=node('details');details.append(node('summary','Key for model suggestions'));const keyInput=field(details,'Model-list key (stored in macOS Keychain)',input('','password'));keyInput.autocomplete='off';details.append(node('small','This key loads model suggestions. Configure server credentials in the server config.'));
            mutation(actions(details),'Save key and load models',async()=>{const listKey=modelListKey(client.value);const result=await run({kind:'models',client:client.value,refresh:true,key:keyInput.value},false);if(result){keyInput.value='';lists.set(listKey,{models:result.data.models,time:new Date().toLocaleTimeString(),warning:Boolean(result.data.warning&&['cache','config'].includes(result.data.source)),source:result.data.source||'model-list result',fetchedAt:result.data.fetched_at});fill();}});guidance.append(details);}
        };
        client.onchange=()=>{fill();endpointGuidance();saveChoices();if(model.value)guidance.append(node('p','The endpoint changed. Check that it supports the retained model ID before saving.','warning-note'));};model.oninput=()=>{fieldError(model,'');saveChoices();};fill();endpointGuidance();
        mutation(actions(row),'Refresh suggestions',async()=>{const listKey=modelListKey(client.value);listStatus.textContent='Loading suggestions…';const result=await run({kind:'models',client:client.value,refresh:true,key:null},false);if(result){lists.set(listKey,{models:result.data.models,time:new Date().toLocaleTimeString(),warning:Boolean(result.data.warning&&['cache','config'].includes(result.data.source)),source:result.data.source||'model-list result',fetchedAt:result.data.fetched_at});fill();}else listStatus.textContent='Suggestions unavailable. Retry or enter a model ID manually.';});
        if(!choice.model)row.append(node('p','New role: choose its endpoint and model before saving.','warning-note'));
        if(chosen==='random')mutation(actions(row),'Remove model',()=>{saveChoices();if(draft.choices.length<=2){notify('Random routing needs at least two models.','warning');return;}draft.choices.splice(index,1);draft.byAlgorithm.set(chosen,draft.choices.map(c=>({...c})));draw();});
        roles.append(row);
      });draft.loadedAlgorithm=chosen;draft.byAlgorithm.set(chosen,draft.choices.map(c=>({...c})));
      if(chosen==='random')mutation(roles,'Add model',()=>{saveChoices();draft.choices.push({client:snapshot.clients[0]?.name||'',model:''});draft.byAlgorithm.set(chosen,draft.choices.map(c=>({...c})));draw();});
      updateDirty();
    }catch(error){if(c.isConnected&&current===revision){editorState.textContent=`Could not load the editor: ${error}`;button(roles,'Retry editor',draw);blocked(save,true);}}
    finally{roles.setAttribute('aria-busy','false');}
  };
  algorithm.onchange=()=>{saveChoices();draft.algorithm=algorithm.value;draft.choices=draft.byAlgorithm.get(draft.algorithm)?.map(c=>({...c}))||[];draw();};updateDirty();enqueue(draw);
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>navigate(b.dataset.page));
document.querySelector('#refresh').onclick=refresh;
document.querySelector('#dismiss-notification').onclick=dismissNotification;
document.querySelector('#dismiss-notification').onmousedown=event=>event.preventDefault();
window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(page==='overview'&&snapshot){const saved=capturePageState();render();restorePageState(saved);}});
window.addEventListener?.('switchyard-notice',event=>notify(String(event.detail),'warning'));
window.addEventListener?.('switchyard-page',event=>{
  const next=typeof event.detail==='string'?event.detail:event.detail?.page;
  if(!Object.hasOwn(pageNames,next))return;
  if(next==='overview')Object.assign(analyticsView,{period:'all',group:'model',search:'',model:typeof event.detail==='object'?event.detail.model||null:null});
  navigate(next);if(!snapshot)refresh();
});
async function poll() {
  if(!document.querySelector('[role="dialog"]')&&!busy)await refresh(false);
  setTimeout(poll,(snapshot?.refresh_seconds||30)*1000);
}
content.append(node('p','Loading Switchyard…','empty-state'));
refresh().finally(()=>setTimeout(poll,(snapshot?.refresh_seconds||30)*1000));

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const invoke = window.__TAURI__.core.invoke;
const content = document.querySelector('#content');
const status = document.querySelector('#status');
let snapshot, page = 'overview', busy = false;
let operationQueue = Promise.resolve();
const drafts = new Map(), lists = new Map();
const usageView = {session:'',search:''};
const routeView = {selected:'',search:''};
const installView = {tool:'',account:'',route:'',file:''};
function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
function card(title) { const c = node('div', undefined, 'card'); c.append(node('h3', title)); content.append(c); return c; }
function button(parent, text, fn, primary = false) { const b = node('button', text, primary ? 'primary' : ''); b.onclick = fn; parent.append(b); return b; }
// Wry's macOS delegate has no JavaScript confirm handler, so this dialog handles confirmation.
function confirmAction(title, message, label, action) {
  const previous=document.activeElement,overlay=node('div',undefined,'modal-backdrop'),dialog=node('div',undefined,'confirm-dialog');
  dialog.setAttribute('role','dialog');dialog.setAttribute('aria-modal','true');dialog.setAttribute('aria-label',title);
  dialog.append(node('h3',title),node('p',message));overlay.append(dialog);document.body.append(overlay);
  const background=[...document.body.children].filter(element=>element!==overlay).map(element=>[element,element.inert]);
  background.forEach(([element])=>{element.inert=true;});
  const close=()=>{overlay.remove();background.forEach(([element,inert])=>{element.inert=inert;});if(previous?.isConnected)previous.focus();};
  const controls=actions(dialog),cancel=button(controls,'Cancel',close),accept=button(controls,label,()=>{close();action();});accept.classList.add('danger');
  dialog.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();close();}else if(event.key==='Tab'){event.preventDefault();(document.activeElement===cancel?accept:cancel).focus();}};
  cancel.focus();
}
function field(parent, title, control) { control.setAttribute('aria-label',title); const l = node('label', title); l.append(control); parent.append(l); return control; }
function select(options, chosen) { const s = node('select'); for (const [value, label] of options) { const o = node('option', label); o.value = value; s.append(o); } if (chosen !== undefined) s.value = chosen; return s; }
function input(value = '', type = 'text') { const i = node('input'); i.type = type; i.value = value; return i; }
function text(parent, value) { parent.append(node('pre', value)); }
function actions(parent) { const a = node('div', undefined, 'actions'); parent.append(a); return a; }
function setBusy(value) { busy = value; document.querySelectorAll('button,input,select').forEach(n => n.disabled = value || n.dataset.blocked === 'true'); }
// Route editors and install previews share a queue because the controller runs one operation at a time.
function enqueue(operation) {
  operationQueue = operationQueue.then(async () => {
    while (busy || document.querySelector('[role="dialog"]')) await new Promise(resolve => setTimeout(resolve, 30));
    await operation();
  }).catch(error => { status.textContent = String(error); });
}
// Successful editor loading keeps the saved-route and restart result visible.
async function run(action, refresh = true) {
  if (busy) return;
  setBusy(true); if(!['editor','preview_install'].includes(action.kind))status.textContent = 'Working…';
  try { const result = await invoke('action', {action}); if(!['editor','preview_install'].includes(action.kind))status.textContent = result.message; if (refresh) { snapshot = await invoke('snapshot'); render(); } return result; }
  catch (error) { status.textContent = String(error); }
  finally { setBusy(false); }
}
async function refresh() {
  if (busy) return;
  setBusy(true);
  try { snapshot = await invoke('snapshot'); render(); status.textContent = snapshot.errors.join('\n'); }
  catch(error) { status.textContent = String(error); }
  finally { setBusy(false); }
}
function toolFields(c) {
  const tool = field(c, 'Coding tool', select(snapshot.tools.map(t => [t.tool, `${t.label}${t.available ? '' : ' (binary not found)'}`])));
  const account = field(c, 'Account', select([['','Current login']]));
  const info = node('pre'); c.append(info);
  const update = () => { const t = snapshot.tools.find(t=>t.tool===tool.value); account.replaceChildren(); for (const name of ['',...t.accounts]) { const o = node('option',name || 'Current login'); o.value=name; account.append(o); } info.textContent=t.status; };
  tool.onchange=update; update();
  return {tool,account,get:()=>({tool:tool.value,account:account.value || null})};
}
function routeMatches(route, query) { return `${route.label} ${snapshot.algorithms.find(a=>a.kind===route.kind)?.title||''}`.toLowerCase().includes(query.toLowerCase()); }
function routeField(c) {
  const wrapper=node('div');c.append(wrapper);
  const route=field(wrapper,'Route',select(snapshot.routes.map(r=>[r.key,`${r.id} · ${snapshot.algorithms.find(a=>a.kind===r.kind)?.title||r.kind}`])));
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
  if (generation !== snapshot.generation) { drafts.clear(); lists.clear(); generation = snapshot.generation; }
  content.replaceChildren();
  document.querySelector('#title').textContent = {overview:'Overview',routes:'Routes',install:'Install',usage:'Usage',sessions:'Sessions',settings:'Settings'}[page];
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  if (page==='overview') {
    const metrics=snapshot.metrics;
    const connection=card('Connection');
    connection.append(node('span',metrics.running?'Connected':'Not responding',metrics.running?'badge connected':'badge'));
    connection.append(node('p',metrics.server_url,'endpoint-address'));
    const totals=node('div',undefined,'metric-grid');content.append(totals);
    for(const [label,period] of [['Today',metrics.today],['Past 7 days',metrics.week]]) {
      const panel=node('div',undefined,'card metric');panel.append(node('h3',label),node('strong',period.requests.toLocaleString()),node('span',' model calls'),node('p',`${period.tokens.toLocaleString()} tokens, including routing overhead`,'muted'));totals.append(panel);
    }
    renderActivity(card('Recent activity'));
    const notes=card('Cost estimates');
    let period='';for(const line of snapshot.summary){if(line.startsWith('Today —'))period='Today';if(line.startsWith('This week —'))period='Past 7 days';if(line.includes('Saved'))notes.append(node('p',`${period}: ${line.trim()}`));else if(line.includes('prices')||line.includes('Savings hidden'))notes.append(node('p',line.trim()));}
    if(notes.children.length===1)notes.append(node('p','No cost estimate is available yet.','muted'));
    const breakdown=node('details');breakdown.append(node('summary','Model breakdown and full usage summary'));for(const line of snapshot.summary)if(line.trim())breakdown.append(node('p',line.trim(),'summary-row'));notes.append(breakdown);
    const c=card('Routes');c.classList.add('routing-board');
    const heading=node('div',undefined,'board-head');
    for(const title of ['Route','Routing','Models'])heading.append(node('span',title));
    c.append(heading);
    const list=node('div',undefined,'route-list');
    for(const route of snapshot.routes.slice(0,6)) {
      const row=node('div',undefined,'route-row board-row');
      const models=node('div',undefined,'board-models');
      for(const choice of route.choices)models.append(node('p',`${choice.model} · ${choice.client}`));
      row.append(node('h4',route.id),node('span',snapshot.algorithms.find(a=>a.kind===route.kind)?.title || route.kind,'board-routing'),models);
      list.append(row);
    }
    if(!snapshot.routes.length)list.append(node('p','No routes are configured. Open the server config in Settings to add one.','empty-state'));
    c.append(list);button(actions(c),`Manage ${snapshot.routes.length} routes →`,()=>{page='routes';render();});
  } else if (page==='routes') {
    renderRouteBrowser();
    if (!snapshot.routes.length) text(card('No routes loaded'),'Check the server config in Settings, then refresh.');
  } else if (page==='install') {
    const intro=card('Connect your coding tools');
    intro.append(node('p','Choose a Switchyard route for each coding tool, review the settings below, then install the route.'));
    const help=node('details');help.append(node('summary','How accounts and settings work'));help.append(node('p','Subscription routes use the coding tool’s login. API routes use credentials configured on the Switchyard server. Keys entered in Routes only load model lists.'));help.append(node('p','Installation backs up the original settings. Shell variables and project settings may override these defaults.'));intro.append(help);
    const tool=field(intro,'Coding tool',select(snapshot.tools.map(t=>[t.tool,`${t.label} · ${t.available?'Detected':'Binary not found'}`]),installView.tool||snapshot.tools[0].tool));
    installView.tool=tool.value;
    const panel=node('div');content.append(panel);
    const show=()=>{installView.tool=tool.value;installView.account='';installView.file='';panel.replaceChildren();renderInstall(snapshot.tools.find(t=>t.tool===tool.value),panel);};
    tool.onchange=show;renderInstall(snapshot.tools.find(t=>t.tool===tool.value),panel);
    renderAccounts();
  } else if (page==='usage') {
    const c=card('Model calls');
    c.append(node('p','Each row shows a completed model call. A turn may include several calls. Input tokens include cached reads.'));
    c.append(node('p',`${snapshot.entries.length} recent calls${snapshot.limited?' (limited to 8 MiB / 5,000 records)':''}; Switchyard skipped ${snapshot.skipped} unreadable records.`,'muted'));
    const filters=node('div',undefined,'field-grid');c.append(filters);
    const session=field(filters,'Session',select([['','All sessions'],...snapshot.sessions.map(s=>[s,s])],usageView.session));
    const search=field(filters,'Search turn, model, or route',input(usageView.search));
    const container=node('div',undefined,'scroll'); c.append(container);
    const draw=()=>{
      container.replaceChildren(); const table=node('table'),head=node('tr');
      for(const name of ['Time','Model / role','Session / turn','Route','Input','Cached','Output'])head.append(node('th',name));
      const thead=node('thead');thead.append(head);table.append(thead);const tbody=node('tbody');table.append(tbody);
      const query=search.value.toLowerCase();
      for(const e of [...snapshot.entries].reverse()) {
        if(session.value && e.session_id!==session.value)continue;
        if(query && ![e.model,e.turn_id,e.route_id].join(' ').toLowerCase().includes(query))continue;
        const row=node('tr');
        const time=node('td'),date=new Date(e.ts);time.title=e.ts;
        if(Number.isNaN(date.getTime()))time.textContent=e.ts;
        else time.append(node('span',date.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'}),'table-primary'),node('span',date.toLocaleTimeString(),'table-secondary'));
        const model=node('td');model.append(node('span',e.model,'table-primary'));
        if(e.tier==='classifier')model.append(node('span','Routing overhead','table-secondary'));
        const ids=node('td');ids.append(node('span',e.session_id||'Session not recorded','table-primary'),node('span',e.turn_id||'Turn not recorded','table-secondary'));
        row.append(time,model,ids,node('td',e.route_id));
        for(const value of [e.prompt_tokens,e.cached_tokens,e.completion_tokens])row.append(node('td',value.toLocaleString()));
        tbody.append(row);
      }
      container.append(table);
      if(!tbody.children.length)container.append(node('p',snapshot.entries.length?'No calls match these filters.':'Switchyard has not recorded any model calls yet.','empty-state'));
    };session.onchange=()=>{usageView.session=session.value;draw();};search.oninput=()=>{usageView.search=search.value;draw();};draw();
  } else if(page==='sessions') {
    const c=card('New session');
    c.append(node('p','Launch a coding tool in a separate Git worktree and Terminal window. Your account’s usage limits apply.'));
    const tool=toolFields(c),route=routeField(c),project=field(c,'Project folder (absolute path)',input());
    const launch=button(actions(c),'Launch session',()=>run({kind:'launch',...tool.get(),...routeArgs(route),project:project.value}),true);
    const selected=()=>{launch.dataset.blocked=String(!route.value);launch.disabled=busy||!route.value;};route.onchange=selected;selected();
    renderAccounts();
  } else if(page==='settings') {
    const c=card('Application');
    button(actions(c),'Open server config',()=>run({kind:'open_config'}));
    button(actions(c),'Restart server',()=>run({kind:'restart'}));
    button(actions(c),'Update from source…',()=>run({kind:'update'}));
    c.append(node('p','Updates build from the source checkout used to install this app. Your settings, usage history, and accounts are preserved.'));
    text(c,'Terminal: switchyard-menubar --tui');
    c.append(node('p','Closing the window keeps Switchyard running. To exit, choose Quit Switchyard from the menu bar after any operation finishes.'));
  }
}
function renderActivity(c) {
  c.append(node('p','Model calls per hour · past 24 hours · recent recorded calls','muted'));
  const buckets=snapshot.activity;
  const graph=node('div',undefined,'activity-graph');graph.setAttribute('role','img');graph.setAttribute('aria-label',`Hourly model calls, oldest first: ${buckets.join(', ')}`);
  const max=Math.max(1,...buckets);
  buckets.forEach((count,index)=>{const bar=node('div',undefined,'activity-bar');bar.style.height=`${Math.max(2,count/max*100)}%`;bar.title=`${24-index}–${23-index} hours ago · ${count} calls`;graph.append(bar);});c.append(graph);
  const axis=node('div',undefined,'graph-axis');axis.append(node('span','24h ago'),node('span','Now'));c.append(axis);
  c.append(node('small',`${buckets.reduce((a,b)=>a+b,0)} recorded calls${snapshot.limited?' · recent history is limited':''}`));
}
function renderRouteBrowser() {
  const c=card('Route library');
  const search=field(c,'Find a route by name, routing method, endpoint, or model',input(routeView.search));
  const count=node('p',undefined,'muted'),layout=node('div',undefined,'route-workspace'),list=node('div',undefined,'route-picker'),detail=node('div',undefined,'route-detail');
  c.append(count,layout);layout.append(list,detail);
  const draw=()=>{
    routeView.search=search.value;
    const routes=snapshot.routes.filter(r=>routeMatches(r,search.value));
    count.textContent=`${routes.length} of ${snapshot.routes.length} routes`;
    list.replaceChildren();
    for(const route of routes){const b=button(list,route.id,()=>{routeView.selected=route.key;draw();});b.classList.toggle('selected',route.key===routeView.selected);b.setAttribute('aria-pressed',String(route.key===routeView.selected));b.append(node('small',snapshot.algorithms.find(a=>a.kind===route.kind)?.title||route.kind));if(drafts.get(route.key)?.dirty)b.append(node('small','Unsaved edits'));}
    detail.replaceChildren();
    if(!routes.length){list.append(node('p','No routes match your search.','empty-state'));return;}
    const route=routes.find(r=>r.key===routeView.selected)||routes[0];routeView.selected=route.key;
    list.querySelectorAll('button').forEach((b,i)=>{const selected=routes[i].key===route.key;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});
    renderRoute(route,detail);
  };search.oninput=draw;draw();
}
// renderInstall checks each reply's revision before updating the preview or Install button.
function renderInstall(tool, parent) {
  const c=card(tool.label);parent.append(c);c.dataset.tool=tool.tool;
  c.append(node('p',tool.available?'Coding tool detected.':'Binary not found. You can configure settings before installing the tool.','muted'));
  const fields=node('div',undefined,'field-grid');c.append(fields);
  const account=field(fields,'Settings location',select([['','Detected user settings'],...tool.accounts.map(name=>[name,`Account: ${name}`])],installView.account));
  const route=routeField(fields);if(installView.route&&snapshot.routes.some(r=>r.key===installView.route))route.value=installView.route;
  const file=field(c,tool.tool==='pi'?'Custom models.json path (optional)':'Custom settings file path (optional)',input(installView.file));
  file.placeholder=tool.files?.[0]||'/absolute/path/to/settings';
  c.append(node('small','Use an absolute file path for settings in another location. Leave empty to use the selected settings location. Pi also uses settings.json beside models.json.'));
  const comparison=node('div',undefined,'install-diff');c.append(comparison);
  const authentication=node('p',undefined,'install-auth');c.append(authentication);
  const target=()=>({tool:tool.tool,account:account.value||null,settings_file:file.value||null});
  const a=actions(c),install=button(a,`Install into ${tool.label} →`,()=>run({kind:'install',...target(),...routeArgs(route)}),true);
  button(a,'Restore backed-up settings…',()=>confirmAction('Restore coding-tool settings?','Restore the settings saved before the first installation. This changes coding tool settings and keeps your Switchyard routes.','Restore settings',()=>run({kind:'restore',...target()})));
  let revision=0;
  const update=()=>{
    installView.account=account.value;installView.route=route.value;installView.file=file.value;
    const requested=++revision;install.dataset.blocked='true';install.disabled=true;comparison.replaceChildren(node('p','Loading settings preview…'));authentication.textContent='';
    enqueue(async()=>{
      if(!c.isConnected||requested!==revision)return;
      if(!route.value){comparison.replaceChildren(node('p',snapshot.routes.length?'Choose a route to preview its settings.':'Add a route in the server config first.'));return;}
      setBusy(true);
      try {
        const result=await invoke('action',{action:{kind:'preview_install',...target(),...routeArgs(route)}});
        if(!c.isConnected||requested!==revision)return;
        comparison.replaceChildren(node('h4','Settings diff · − current / + proposed'));
        if(result.data.error)comparison.append(node('p',result.data.error));
        else {for(const change of result.data.changes){comparison.append(node('pre',`${change.file} · ${change.key}`,'diff-heading'));comparison.append(node('pre',`− ${change.before}`,'diff-remove'),node('pre',`+ ${change.after}`,'diff-add'));}if(!result.data.changes.length)comparison.append(node('p','These settings already match the selected route.'));}
        if(result.data.current){const current=node('details');current.append(node('summary','Current settings and file locations'),node('pre',result.data.current));comparison.append(current);}
        authentication.textContent=result.data.authentication||'';
        install.dataset.blocked=String(Boolean(result.data.error));
      } catch(error) { if(c.isConnected&&requested===revision)comparison.replaceChildren(node('p',String(error))); }
      finally { setBusy(false); }
    });
  };
  account.onchange=update;route.onchange=update;file.oninput=()=>{installView.file=file.value;revision++;install.dataset.blocked='true';install.disabled=true;comparison.replaceChildren(node('p','Finish entering the file path, then press Enter or leave the field to preview.'));};file.onchange=update;file.onkeydown=event=>{if(event.key==='Enter')update();};update();
}
function renderAccounts() {
  const c=card('Subscription accounts');const disclosure=node('details');disclosure.append(node('summary','Sign in to another subscription account…'));c.append(disclosure);const parent=disclosure;
  parent.append(node('p','An additional account is optional. Use your existing login unless you want a separate subscription account. API routes use the server’s credentials.'));
  parent.append(node('p','Switchyard opens the Codex CLI or Claude Code login in Terminal and saves the account in a separate settings folder. It does not add an API key or change the account in the Codex app. Select the folder when installing a route or launching a new session.'));
  const tool=field(parent,'Coding tool',select(snapshot.tools.filter(t=>['codex_cli','claude'].includes(t.tool)).map(t=>[t.tool,t.label])));
  const name=field(parent,'Name for the separate account',input());
  button(actions(parent),'Sign in to another account…',()=>run({kind:'add_account',tool:tool.value,name:name.value}));
}
// Tiered roles keep their models across algorithms; roles without tiers keep their positions.
function roleChoice(draft, route, role, index) {
  return (role.tier && draft.memory.get(role.tier)) ||
    ((draft.algorithm === route.kind || draft.loadedAlgorithm === draft.algorithm) && draft.choices[index]) ||
    route.choices[0] || {client:snapshot.clients[0]?.name || '',model:''};
}
function renderRoute(route, parent) {
  const c=card(route.id);parent.append(c);
  if(route.generated)c.append(node('p',`This route belongs to a generated block: ${route.generated}. Its owner may replace these edits.`));
  const remove=()=>confirmAction(`Delete route “${route.id}”?`,'Remove this route from the server config. Targets and coding tool settings are kept. The config will be backed up before saving.','Delete route',()=>run({kind:'remove',generation,route:route.key,id:route.id}));
  if(!route.editable){c.append(node('p','Edit this route in the server config.'));button(actions(c),'Delete route…',remove).classList.add('danger');return;}
  let draft=drafts.get(route.key);
  if(!draft){draft={algorithm:snapshot.algorithms.some(a=>a.kind===route.kind)?route.kind:snapshot.algorithms[0].kind,choices:route.choices.map(c=>({...c})),memory:new Map()}; route.tiers?.forEach((tier,index)=>{if(tier)draft.memory.set(tier, {...route.choices[index]});});drafts.set(route.key,draft);}
  const algorithm=field(c,'Routing algorithm',select(snapshot.algorithms.map(a=>[a.kind,a.title]),draft.algorithm));
  const description=node('p',undefined,'description'),roles=node('div',undefined,'role-fields'); c.append(description,roles);
  const editState=node('p',draft.dirty?'Unsaved edits':'Saved route','muted');c.append(editState);
  const saveChoices=()=>{draft.dirty=true;editState.textContent='Unsaved edits';roles.querySelectorAll('[data-role]').forEach(row=>{const selects=row.querySelectorAll('select,input'); const choice={client:selects[0].value,model:selects[1].value};draft.choices[Number(row.dataset.role)]=choice;if(row.dataset.tier)draft.memory.set(row.dataset.tier,choice);});};
  const draw=async()=>{
    const result=await run({kind:'editor',generation,route:route.key,algorithm:draft.algorithm},false); if(!result)return;
    description.textContent=result.message; roles.replaceChildren();
    result.data.roles.forEach((role,index)=>{
      const choice=roleChoice(draft, route, role, index);
      draft.choices[index]={...choice};const row=node('div');row.dataset.role=index;row.dataset.tier=role.tier || '';
      const client=field(row,`${role.label} endpoint`,select(snapshot.clients.map(c=>[c.name,`${c.name} (${c.host})`]),choice.client));
      const model=field(row,role.label==='Model'?'Model':`${role.label} model`,input(choice.model));row.append(node('small',role.hint));
      const datalist=node('datalist');datalist.id=`models-${route.key}-${index}`;model.setAttribute('list',datalist.id);row.append(datalist);
      const fill=()=>{const values=lists.get(client.value)||snapshot.clients.find(c=>c.name===client.value)?.models||[];datalist.replaceChildren();for(const value of values){const o=node('option');o.value=value;datalist.append(o);}};
      client.onchange=()=>{fill();saveChoices();};model.oninput=saveChoices;fill();
      const a=actions(row);button(a,'Refresh models',async()=>{const r=await run({kind:'models',client:client.value,refresh:true,key:null},false);if(r){lists.set(client.value,r.data.models);fill();}});
      const details=node('details');details.append(node('summary','API key for this endpoint')); const key=field(details,'Save in macOS Keychain',input('','password'));key.autocomplete='off';
      button(actions(details),'Save key and load models',async()=>{const value=key.value;key.value='';const r=await run({kind:'models',client:client.value,refresh:true,key:value},false);if(r){lists.set(client.value,r.data.models);fill();}});const endpoint=snapshot.clients.find(c=>c.name===client.value);
      if(endpoint?.accepts_key)row.append(details);
      if(endpoint?.note)row.append(node('p',endpoint.note));
      roles.append(row);
    });draft.choices.length=result.data.roles.length;draft.loadedAlgorithm=draft.algorithm;
  };
  algorithm.onchange=()=>{saveChoices();draft.algorithm=algorithm.value;draw();};
  const a=actions(c);
  button(a,'Save and restart',async()=>{saveChoices();const result=await run({kind:'apply',generation,route:route.key,algorithm:draft.algorithm,choices:draft.choices});if(result){drafts.delete(route.key);render();}},true);
  button(a,'Revert unsaved edits',()=>{drafts.delete(route.key);render();});
  button(a,'Delete route…',remove).classList.add('danger');
  // Each route waits for the shared operation to finish before loading its editor roles.
  enqueue(async()=>{if(page==='routes'&&c.isConnected)await draw();});
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>{page=b.dataset.page;render();});
document.querySelector('#refresh').onclick=refresh;
window.addEventListener?.('switchyard-page', event=>{
  if(!['overview','routes','install','usage','sessions','settings'].includes(event.detail))return;
  page=event.detail;render();
  if(!snapshot)refresh();
});
async function poll() {
  if(document.querySelector('[role="dialog"]')){setTimeout(poll,(snapshot?.refresh_seconds||30)*1000);return;}
  if(page==='overview'||page==='usage')await refresh();
  else if(!busy) {
    setBusy(true);
    try { await invoke('snapshot'); } catch(error) { status.textContent=String(error); }
    finally { setBusy(false); }
  }
  setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000);
}
refresh().finally(()=>setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000));

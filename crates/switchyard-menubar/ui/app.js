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
function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
function card(title) { const c = node('div', undefined, 'card'); c.append(node('h3', title)); content.append(c); return c; }
function button(parent, text, fn, primary = false) { const b = node('button', text, primary ? 'primary' : ''); b.onclick = fn; parent.append(b); return b; }
function field(parent, title, control) { const l = node('label', title); l.append(control); parent.append(l); return control; }
function select(options, chosen) { const s = node('select'); for (const [value, label] of options) { const o = node('option', label); o.value = value; s.append(o); } if (chosen !== undefined) s.value = chosen; return s; }
function input(value = '', type = 'text') { const i = node('input'); i.type = type; i.value = value; return i; }
function text(parent, value) { parent.append(node('pre', value)); }
function actions(parent) { const a = node('div', undefined, 'actions'); parent.append(a); return a; }
function setBusy(value) { busy = value; document.querySelectorAll('button,input,select').forEach(n => n.disabled = value || n.dataset.blocked === 'true'); }
// Route editors and install previews share a queue because the controller runs one operation at a time.
function enqueue(operation) {
  operationQueue = operationQueue.then(async () => {
    while (busy) await new Promise(resolve => setTimeout(resolve, 30));
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
function routeField(c) { return field(c, 'Route', select(snapshot.routes.map(r=>[r.key,r.label]))); }
function routeArgs(s) { const r=snapshot.routes.find(r=>r.key===s.value); if (!r) throw new Error('Choose a route first.'); return {route:r.key,id:r.id}; }
let generation;
function render() {
  if (!snapshot) return;
  if (generation !== snapshot.generation) { drafts.clear(); lists.clear(); generation = snapshot.generation; }
  content.replaceChildren();
  document.querySelector('#title').textContent = {overview:'Overview',routes:'Routes',install:'Install',usage:'Usage',sessions:'Sessions',settings:'Settings'}[page];
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  if (page==='overview') {
    const summary=card('Server and usage'),rows=node('div',undefined,'summary-list');
    for(const line of snapshot.summary)if(line.trim())rows.append(node('p',line.trim(),'summary-row'));
    summary.append(rows);
    const c=card('Routes');c.classList.add('routing-board');
    const heading=node('div',undefined,'board-head');
    for(const title of ['Route','Routing','Models'])heading.append(node('span',title));
    c.append(heading);
    const list=node('div',undefined,'route-list');
    for(const route of snapshot.routes) {
      const row=node('div',undefined,'route-row board-row');
      const models=node('div',undefined,'board-models');
      for(const choice of route.choices)models.append(node('p',`${choice.model} · ${choice.client}`));
      row.append(node('h4',route.id),node('span',snapshot.algorithms.find(a=>a.kind===route.kind)?.title || route.kind,'board-routing'),models);
      list.append(row);
    }
    if(!snapshot.routes.length)list.append(node('p','No routes are configured. Open the server config in Settings to add one.','empty-state'));
    c.append(list);
  } else if (page==='routes') {
    for (const route of snapshot.routes) renderRoute(route);
    if (!snapshot.routes.length) text(card('No routes loaded'),'Check the server config in Settings, then refresh.');
  } else if (page==='install') {
    const intro=card('Connect your coding tools');
    intro.append(node('p','Choose a Switchyard route for each coding tool, review the settings below, then install the route.'));
    intro.append(node('p','Subscription routes reuse the coding tool’s login. API routes use credentials configured on the Switchyard server. Configure API credentials on the server. Keys entered in Routes only load model lists.'));
    intro.append(node('p','Installation backs up the original user settings. Shell variables and project settings may override these defaults.','muted'));
    for(const tool of snapshot.tools) renderInstall(tool);
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
    button(actions(c),'Launch session',()=>run({kind:'launch',...tool.get(),...routeArgs(route),project:project.value}),true);
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
// Each card previews the selected account and route before the user changes any settings.
function renderInstall(tool) {
  const c=card(tool.label);c.dataset.tool=tool.tool;
  if(!tool.available)c.append(node('p','The coding tool was not found on this machine. You can configure it now and install the tool later.','muted'));
  const fields=node('div',undefined,'field-grid');c.append(fields);
  const account=field(fields,'Settings folder',select([['','Existing user settings'],...tool.accounts.map(name=>[name,`Separate account: ${name}`])]));
  const route=routeField(fields);
  const comparison=node('div',undefined,'install-comparison');c.append(comparison);
  const before=node('div'),after=node('div');
  before.append(node('h4','Current settings'));after.append(node('h4','After installation'));
  const current=node('pre',tool.status),proposed=node('pre','Loading preview…');before.append(current);after.append(proposed);comparison.append(before,after);
  const routing=node('p',undefined,'muted'),authentication=node('p',undefined,'install-auth');c.append(routing,authentication);
  const a=actions(c),install=button(a,'Install route',()=>run({kind:'install',tool:tool.tool,account:account.value||null,...routeArgs(route)}),true);
  button(a,'Restore original settings',()=>run({kind:'restore',tool:tool.tool,account:account.value||null}));
  let revision=0;
  const update=()=>{
    const requested=++revision;install.dataset.blocked='true';install.disabled=true;
    proposed.textContent='Loading preview…';authentication.textContent='';
    const chosen=snapshot.routes.find(r=>r.key===route.value);
    routing.textContent=chosen?`${snapshot.algorithms.find(a=>a.kind===chosen.kind)?.title || chosen.kind}: ${chosen.choices.map(c=>c.model).join(' → ')}`:'';
    enqueue(async()=>{
      if(!c.isConnected||requested!==revision)return;
      if(!route.value){proposed.textContent='Add a route in the server config first.';return;}
      current.textContent='Loading preview…';
      setBusy(true);
      try {
        const result=await invoke('action',{action:{kind:'preview_install',tool:tool.tool,account:account.value||null,...routeArgs(route)}});
        if(!c.isConnected||requested!==revision)return;
        current.textContent=result.data.current;
        proposed.textContent=result.data.error||result.data.proposed;
        authentication.textContent=result.data.authentication||'';
        install.dataset.blocked=String(Boolean(result.data.error));
      } catch(error) { if(c.isConnected&&requested===revision){current.textContent=String(error);proposed.textContent=String(error);} }
      finally { setBusy(false); }
    });
  };
  account.onchange=update;route.onchange=update;update();
}
function renderAccounts() {
  const c=card('Use another subscription account');
  c.append(node('p','An additional account is optional. Use your existing login unless you want a separate subscription account. API routes use the server’s credentials.'));
  c.append(node('p','Switchyard opens the Codex CLI or Claude Code login in Terminal and saves the account in a separate settings folder. It does not add an API key or change the account in the Codex app. Select the folder when installing a route or launching a new session.'));
  const tool=field(c,'Coding tool',select(snapshot.tools.filter(t=>['codex_cli','claude'].includes(t.tool)).map(t=>[t.tool,t.label])));
  const name=field(c,'Name for the separate account',input());
  button(actions(c),'Sign in to another account…',()=>run({kind:'add_account',tool:tool.value,name:name.value}));
}
// Tiered roles keep their models across algorithms; roles without tiers keep their positions.
function roleChoice(draft, route, role, index) {
  return (role.tier && draft.memory.get(role.tier)) ||
    (draft.algorithm === route.kind && draft.choices[index]) ||
    route.choices[0] || {client:snapshot.clients[0]?.name || '',model:''};
}
function renderRoute(route) {
  const c=card(route.id);
  if(route.generated)c.append(node('p',`This route belongs to a generated block: ${route.generated}. Its owner may replace these edits.`));
  if(!route.editable){c.append(node('p','Edit this route in the server config.'));return;}
  let draft=drafts.get(route.key);
  if(!draft){draft={algorithm:snapshot.algorithms.some(a=>a.kind===route.kind)?route.kind:snapshot.algorithms[0].kind,choices:route.choices.map(c=>({...c})),memory:new Map()}; route.tiers?.forEach((tier,index)=>{if(tier)draft.memory.set(tier, {...route.choices[index]});});drafts.set(route.key,draft);}
  const algorithm=field(c,'Routing algorithm',select(snapshot.algorithms.map(a=>[a.kind,a.title]),draft.algorithm));
  const description=node('p',undefined,'description'),roles=node('div',undefined,'role-fields'); c.append(description,roles);
  const saveChoices=()=>{roles.querySelectorAll('[data-role]').forEach(row=>{const selects=row.querySelectorAll('select,input'); const choice={client:selects[0].value,model:selects[1].value};draft.choices[Number(row.dataset.role)]=choice;if(row.dataset.tier)draft.memory.set(row.dataset.tier,choice);});};
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
    });draft.choices.length=result.data.roles.length;
  };
  algorithm.onchange=()=>{saveChoices();draft.algorithm=algorithm.value;draw();};
  const a=actions(c);
  button(a,'Save and restart',async()=>{saveChoices();const result=await run({kind:'apply',generation,route:route.key,algorithm:draft.algorithm,choices:draft.choices});if(result){drafts.delete(route.key);render();}},true);
  button(a,'Discard changes',()=>{drafts.delete(route.key);render();});
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
  if(page==='overview'||page==='usage')await refresh();
  else if(!busy) {
    setBusy(true);
    try { await invoke('snapshot'); } catch(error) { status.textContent=String(error); }
    finally { setBusy(false); }
  }
  setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000);
}
refresh().finally(()=>setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000));

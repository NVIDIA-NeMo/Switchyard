// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const invoke = window.__TAURI__.core.invoke;
const content = document.querySelector('#content');
const status = document.querySelector('#status');
let snapshot, page = 'overview', busy = false;
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
function setBusy(value) { busy = value; document.querySelectorAll('button,input,select').forEach(n => n.disabled = value); }
// Successful editor loading keeps the saved-route and restart result visible.
async function run(action, refresh = true) {
  if (busy) return;
  setBusy(true); if(action.kind!=='editor')status.textContent = 'Working…';
  try { const result = await invoke('action', {action}); if(action.kind!=='editor')status.textContent = result.message; if (refresh) { snapshot = await invoke('snapshot'); render(); } return result; }
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
function routeField(c) { return field(c, 'Route · algorithm · endpoint / model', select(snapshot.routes.map(r=>[r.key,r.label]))); }
function routeArgs(s) { const r=snapshot.routes.find(r=>r.key===s.value); if (!r) throw new Error('Choose a route first.'); return {route:r.key,id:r.id}; }
let generation;
function render() {
  if (!snapshot) return;
  if (generation !== snapshot.generation) { drafts.clear(); lists.clear(); generation = snapshot.generation; }
  content.replaceChildren();
  document.querySelector('#title').textContent = {overview:'Overview',routes:'Routes',install:'Install…',usage:'Usage',sessions:'Sessions',settings:'Settings'}[page];
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===page));
  if (page==='overview') {
    text(card('Server and savings'),snapshot.summary.join('\n'));
    const c=card('Available routes'); for(const r of snapshot.routes)c.append(node('p',r.label));
  } else if (page==='routes') {
    for (const route of snapshot.routes) renderRoute(route);
    if (!snapshot.routes.length) text(card('No routes loaded'),'Check the server config in Settings, then refresh.');
  } else if (page==='install') {
    const c=card('Install or update a coding tool');
    c.append(node('p','Inspect the current model and endpoint, then install a route. Switchyard backs up settings before changing them. Caller login routes require a matching coding tool.'));
    const tool=toolFields(c),route=routeField(c),a=actions(c);
    button(a,'Install / update',()=>run({kind:'install',...tool.get(),...routeArgs(route)}),true);
    button(a,'Restore previous settings',()=>run({kind:'restore',...tool.get()}));
    renderAccounts();
  } else if (page==='usage') {
    const c=card('Model calls by session and turn');
    c.append(node('p','Each row is one completed model call. A user turn can contain several calls. Classifier calls are routing overhead. Input includes cached reads.'));
    c.append(node('p',`${snapshot.entries.length} recent calls${snapshot.limited?' (limited to 8 MiB / 5,000 records)':''}; Switchyard skipped ${snapshot.skipped} unreadable records.`,'muted'));
    const session=field(c,'Session',select([['','All sessions'],...snapshot.sessions.map(s=>[s,s])],usageView.session));
    const search=field(c,'Filter turn, model, or route',input(usageView.search));
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
        for(const value of [e.ts,`${e.model}${e.tier==='classifier'?' (routing overhead)':''}`,`${e.session_id||'Not recorded'} / ${e.turn_id||'Not recorded'}`,e.route_id,e.prompt_tokens,e.cached_tokens,e.completion_tokens])row.append(node('td',String(value)));
        tbody.append(row);
      }
      container.append(table);
    };session.onchange=()=>{usageView.session=session.value;draw();};search.oninput=()=>{usageView.search=search.value;draw();};draw();
  } else if(page==='sessions') {
    const c=card('Start a separate worktree session');
    c.append(node('p','Choose a route and login for one coding tool. Each launch gets its own Git worktree and Terminal window. Existing subscription limits still apply.'));
    const tool=toolFields(c),route=routeField(c),project=field(c,'Git project (absolute path)',input());
    button(actions(c),'Launch session',()=>run({kind:'launch',...tool.get(),...routeArgs(route),project:project.value}),true);
    renderAccounts();
  } else if(page==='settings') {
    const c=card('Manage the local app');
    button(actions(c),'Open server config',()=>run({kind:'open_config'}));
    button(actions(c),'Restart server',()=>run({kind:'restart'}));
    button(actions(c),'Update from source…',()=>run({kind:'update'}));
    c.append(node('p','Update rebuilds and reinstalls from the source checkout used for this installation. Routing settings, prices, log history, and native logins stay saved.'));
    c.append(node('p','Terminal app: switchyard-menubar --tui. Close this window to keep Switchyard in the tray; use Quit Switchyard in the tray to exit. If an operation is queued or running, try Quit again after it finishes.'));
  }
}
function renderAccounts() {
  const c=card('Add a native login'); const tool=toolFields(c),name=field(c,'Account name',input());
  button(actions(c),'Add account / open login',()=>run({kind:'add_account',tool:tool.tool.value,name:name.value}));
  c.append(node('p','Switchyard opens the coding tool’s own login flow. It keeps each login in a separate configuration directory and does not copy or rotate tokens.'));
}
// Tiered roles keep their models across algorithms; roles without tiers keep their positions.
function roleChoice(draft, route, role, index) {
  return (role.tier && draft.memory.get(role.tier)) ||
    (draft.algorithm === route.kind && draft.choices[index]) ||
    route.choices[0] || {client:snapshot.clients[0]?.name || '',model:''};
}
function renderRoute(route) {
  const c=card(route.label);
  if(route.generated)c.append(node('p',`This route belongs to a generated block: ${route.generated}. Its owner may replace these edits.`));
  if(!route.editable){c.append(node('p','Edit this route in the server config.'));return;}
  let draft=drafts.get(route.key);
  if(!draft){draft={algorithm:snapshot.algorithms.some(a=>a.kind===route.kind)?route.kind:snapshot.algorithms[0].kind,choices:route.choices.map(c=>({...c})),memory:new Map()}; route.tiers?.forEach((tier,index)=>{if(tier)draft.memory.set(tier, {...route.choices[index]});});drafts.set(route.key,draft);}
  const algorithm=field(c,'Routing algorithm',select(snapshot.algorithms.map(a=>[a.kind,a.title]),draft.algorithm));
  const description=node('p'),roles=node('div'); c.append(description,roles);
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
  button(a,'Apply route and restart',async()=>{saveChoices();const result=await run({kind:'apply',generation,route:route.key,algorithm:draft.algorithm,choices:draft.choices});if(result){drafts.delete(route.key);render();}},true);
  button(a,'Revert draft',()=>{drafts.delete(route.key);render();});
  // Each route waits for the shared operation to finish before loading its editor roles.
  const setup=async()=>{while(busy)await new Promise(resolve=>setTimeout(resolve,30));if(page==='routes'&&c.isConnected)await draw();};setup();
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>{page=b.dataset.page;render();});
document.querySelector('#refresh').onclick=refresh;
async function poll() {
  if(page==='overview'||page==='usage')await refresh();
  setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000);
}
refresh().finally(()=>setTimeout(poll,(snapshot?.refresh_seconds || 30)*1000));

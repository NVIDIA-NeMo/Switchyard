// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
function buildTrayUsage(entries, now=Date.now()) {
  const start=now-24*3600000, models=new Map(), previous=new Map(), switches=[];
  const buckets=Array.from({length:24},(_,i)=>({start:start+i*3600000,end:start+(i+1)*3600000,label:new Date(start+i*3600000).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}),models:Object.create(null),switches:[]}));
  let routingTokens=0,unassignedCalls=0;
  // Completion times establish order; the original log order breaks timestamp ties.
  const ordered=(entries||[]).map(entry=>({entry,time:Date.parse(entry.ts)})).filter(({time})=>Number.isFinite(time)&&time<=now).sort((a,b)=>a.time-b.time);
  for(const {entry,time} of ordered) {
    const inWindow=time>start, count=(entry.prompt_tokens||0)+(entry.completion_tokens||0);
    if(entry.tier==='classifier') {if(inWindow)routingTokens+=count;continue;}
    const session=entry.session_id?.trim()?entry.session_id:null, prior=session?previous.get(session):undefined;
    const hour=Math.min(23,Math.floor((time-start)/3600000));
    if(inWindow) {
      const tokens={tokens:count,calls:1,input:Math.max(0,(entry.prompt_tokens||0)-(entry.cached_tokens||0)-(entry.cache_creation_tokens||0)),cached:entry.cached_tokens||0,cache_write:entry.cache_creation_tokens||0,output:entry.completion_tokens||0};
      const model=models.get(entry.model)||{id:entry.model,tokens:0,calls:0,input:0,cached:0,cache_write:0,output:0};
      const bucket=buckets[hour].models[entry.model]||{tokens:0,calls:0,input:0,cached:0,cache_write:0,output:0};
      for(const key of Object.keys(tokens)){model[key]+=tokens[key];bucket[key]+=tokens[key];}
      models.set(entry.model,model);buckets[hour].models[entry.model]=bucket;
      if(!session)unassignedCalls++;
      if(prior&&prior!==entry.model) {
        const change={ts:time,from:prior,to:entry.model,session_id:session,turn_id:entry.turn_id,hour};
        switches.push(change);buckets[hour].switches.push(change);
      }
    }
    // Answers before the window can establish the previous model for a session.
    if(session)previous.set(session,entry.model);
  }
  return {start,end:now,rows:buckets,models:[...models.values()].sort((a,b)=>b.tokens-a.tokens||a.id.localeCompare(b.id)),switches,routingTokens,unknownSessionCalls:unassignedCalls};
}

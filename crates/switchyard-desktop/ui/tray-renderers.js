// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const trayChartNote='Brightness shows tokens. ◆ marks session model changes.';
function svgElement(tag,attrs,text) {
  const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);
  if(text!==undefined)node.textContent=text;return node;
}
const trayPalette=['#70ddd1','#a9baff','#f6c178','#ee95c0','#8bcee9','#c8db8c','#c59af0','#ed9d7f'];
let trayModelColors=new Map();
function configureModelColors(models) {
  trayModelColors=new Map([...models].sort().map((id,index)=>{
    if(index<trayPalette.length)return [id,getComputedStyle(document.documentElement).getPropertyValue(`--model-${index+1}`).trim()||trayPalette[index]];
    const hue=(index*137.508)%360,channel=offset=>{const k=(offset+hue/30)%12;return Math.round(255*(.72-.22*Math.max(-1,Math.min(k-3,9-k,1)))).toString(16).padStart(2,'0');};
    return [id,`#${channel(0)}${channel(8)}${channel(4)}`];
  }));
}
function modelColor(id) {return trayModelColors.get(id)||trayPalette[0];}
function chartLayout(host,usage) {
  const w=host.clientWidth,h=host.clientHeight,left=28,right=w-4,bottom=h-20,top=8,step=(right-left)/24;
  return {w,h,left,right,bottom,top,step,max:Math.max(1,...usage.rows.map(hour=>Object.values(hour.models).reduce((sum,row)=>sum+row.tokens,0)))};
}
const opacity=(model,index,selection)=>(selection.model&&model!==selection.model ? .16 : 1)*(selection.hour!==null&&index!==selection.hour ? .4 : 1);
// The caller retains cleanup before drawing, including when initialization fails.
function renderTrayChart(host,usage,c,retainCleanup) {
    const p=chartLayout(host,usage),svg=svgElement('svg',{width:p.w,height:p.h,class:'station-board'}),cells=[];retainCleanup(()=>svg.remove());host.append(svg);
    const lane=(p.bottom-p.top)/Math.max(1,usage.models.length);
    usage.models.forEach(({id:model},row)=>{svg.append(svgElement('circle',{cx:p.left-8,cy:p.top+(row+.5)*lane,r:3,fill:modelColor(model)}));usage.rows.forEach((hour,index)=>{const tokens=hour.models[model]?.tokens||0,rect=svgElement('rect',{x:p.left+index*p.step+1,y:p.top+row*lane+1,width:Math.max(1,p.step-2),height:Math.max(2,lane-3),class:tokens?'usage-cell':'empty-cell',rx:1,fill:tokens?modelColor(model):c.line,'fill-opacity':tokens?.25+.75*Math.sqrt(tokens/p.max):.2});svg.append(rect);cells.push({rect,model,index,alpha:tokens?.25+.75*Math.sqrt(tokens/p.max):.2});});});
    return {highlight(selection){for(const cell of cells){cell.rect.setAttribute('fill-opacity',cell.alpha*opacity(cell.model,cell.index,selection));cell.rect.setAttribute('stroke',cell.index===selection.hour?c.muted:'none');}}};
}

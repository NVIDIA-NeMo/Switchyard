// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
window.switchyardAppearance = (() => {
  const options = [['system','System'],['canary','Dark'],['light','Light']];
  const media = matchMedia('(prefers-color-scheme: dark)');
  let mode = 'system', system = media.matches ? 'dark' : 'light', lastError = '';
  // OS changes must not discard a pending reply that contains a newer saved preference.
  let preferenceRevision = 0, systemRevision = 0;
  function apply(value = {}) {
    const previousMode = mode;
    if (options.some(([id]) => id === value.mode)) mode = value.mode;
    if (['dark','light'].includes(value.system?.toLowerCase())) system = value.system.toLowerCase();
    const theme = mode === 'system' ? (system === 'dark' ? 'canary' : 'light') : mode;
    const changed = previousMode !== mode || document.documentElement.dataset.theme !== theme;
    document.documentElement.dataset.theme = theme;
    if (document.body) document.body.dataset.theme = theme;
    if (changed) window.dispatchEvent(new CustomEvent('switchyard-theme', {detail:{mode,theme}}));
  }
  function applyReply(value, revision, appearanceRevision) {
    apply({mode: revision === preferenceRevision ? value.mode : undefined,
      system: appearanceRevision === systemRevision ? value.system : undefined});
  }
  async function load() {
    const revision = preferenceRevision, appearanceRevision = systemRevision;
    try {
      const value = await window.__TAURI__.core.invoke('appearance');
      applyReply(value, revision, appearanceRevision);
    }
    catch (error) { report(error); }
  }
  function report(error) {
    const target = document.getElementById('appearance-error');
    lastError = `Switchyard could not update the appearance: ${error}`;
    if (target) target.textContent = lastError;
  }
  async function choose(selected) {
    if (!options.some(([id]) => id === selected)) return;
    const revision = ++preferenceRevision, appearanceRevision = systemRevision;
    try {
      const value = await window.__TAURI__.core.invoke('appearance', {mode:selected});
      applyReply(value, revision, appearanceRevision);
      const target = document.getElementById('appearance-error');
      lastError = '';
      if (target) target.textContent = '';
    } catch (error) { report(error); }
  }
  window.addEventListener('switchyard-appearance', event => {preferenceRevision++; systemRevision++; apply(event.detail);});
  window.addEventListener('switchyard-system-theme', event => {systemRevision++; apply({system:event.detail});});
  media.addEventListener('change', event => {systemRevision++; apply({system:event.matches ? 'dark' : 'light'});});
  window.addEventListener('focus', load);
  // Opaque chrome is the safe default until the native accessibility check succeeds.
  document.documentElement.dataset.reduceTransparency = 'true';
  let accessibilityPending = false;
  async function accessibility() {
    if (accessibilityPending) return;
    accessibilityPending = true;
    try {
      const value = await window.__TAURI__.core.invoke('accessibility');
      document.documentElement.dataset.nativeMaterial = String(value.native_material === true);
      document.documentElement.dataset.reduceTransparency = String(value.reduce_transparency !== false);
      document.documentElement.dataset.reduceMotion = String(value.reduce_motion === true);
    } catch {
      document.documentElement.dataset.reduceTransparency = 'true';
    } finally { accessibilityPending = false; }
  }
  window.addEventListener('focus', accessibility);
  window.addEventListener('switchyard-tray-open', accessibility);
  // macOS sends accessibility updates outside the focused webview, so a visible window checks
  // every second. A hidden window stops checking and checks once when it becomes visible.
  let poll;
  function watch() {
    clearInterval(poll);
    poll = document.hidden ? undefined : setInterval(accessibility, 1000);
  }
  if (typeof setInterval === 'function') {
    document.addEventListener('visibilitychange', () => { watch(); if (!document.hidden) accessibility(); });
    watch();
  }
  apply();
  load();
  accessibility();
  return {options, choose, get mode(){return mode;}, get error(){return lastError;}};
})();

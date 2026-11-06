// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
// A combobox is a text field with a filtered option list. It replaces native <select> menus so
// long lists of routes, endpoints, and models can be searched. A strict combobox keeps a value
// from its list, like a <select>. A free combobox keeps any typed text and offers the list as
// suggestions, which model ID fields need because endpoints accept unlisted IDs.

// This function returns a score and the matched character positions when every query
// character appears in `text` in order. Substring matches rank above scattered matches.
function fuzzyMatch(query, text) {
  const q = query.toLowerCase(), t = text.toLowerCase();
  const boundary = index => index === 0 || /[^a-z0-9]/.test(t[index - 1]);
  const at = t.indexOf(q);
  if (at >= 0) return {score: 1000 + (boundary(at) ? 300 : 0) - at - t.length / 100, positions: Array.from(q, (_, i) => at + i)};
  let best = null;
  for (let start = t.indexOf(q[0]); start >= 0; start = t.indexOf(q[0], start + 1)) {
    const positions = [start];
    for (let i = 1, j = start + 1; i < q.length; i++, j++) {
      j = t.indexOf(q[i], j);
      if (j < 0) break;
      positions.push(j);
    }
    // A later start leaves fewer characters, so it cannot match when this one failed.
    if (positions.length !== q.length) break;
    let score = 500 - start / 10;
    positions.forEach((p, i) => {
      if (i) score += p === positions[i - 1] + 1 ? 15 : -Math.min(p - positions[i - 1], 10);
      if (boundary(p)) score += 10;
    });
    if (!best || score > best.score) best = {score, positions};
  }
  return best;
}

// Every whitespace-separated token must match the label or the option's other search text.
// Label matches rank higher and are the only ones highlighted.
function rankOptions(options, query) {
  const tokens = query.trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) return options.map(option => ({option, positions: []}));
  const ranked = [];
  options.forEach((option, index) => {
    let score = 0;
    const positions = new Set();
    for (const token of tokens) {
      const label = fuzzyMatch(token, option.label), other = fuzzyMatch(token, option.search);
      if (!label && !other) return;
      if (label && (!other || label.score * 1.5 >= other.score)) {
        score += label.score * 1.5;
        label.positions.forEach(p => positions.add(p));
      } else score += other.score;
    }
    ranked.push({option, positions: [...positions], score, index});
  });
  return ranked.sort((a, b) => b.score - a.score || a.index - b.index);
}

// Options are [value, label, detail, keywords] arrays. Only value is required.
function comboboxOption(option) {
  const [value, label = value, detail = '', keywords = ''] = option;
  return {value: String(value), label: String(label), detail: detail || '', search: [value, detail, keywords].filter(Boolean).join(' ')};
}

let comboboxCount = 0;
// This function returns the text input. Its `host` property holds the element to insert into
// the page, and `setOptions` replaces the list. A strict combobox reports the chosen option's
// value through `value` and fires `change` only when that value changes.
function combobox(options, chosen, {free = false, mono = false, placeholder = '', limit = 200} = {}) {
  const id = `combobox-${++comboboxCount}`;
  // The strict field replaces `value`, so it reads and writes its visible text through the prototype.
  const nativeInputValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
  const host = document.createElement('div');
  host.className = `combobox${mono ? ' mono' : ''}${free ? ' free' : ''}`;
  const input = document.createElement('input');
  Object.assign(input, {type: 'text', autocomplete: 'off', spellcheck: false, placeholder});
  for (const [name, value] of [['role', 'combobox'], ['aria-autocomplete', 'list'], ['aria-expanded', 'false'], ['aria-controls', `${id}-list`], ['autocapitalize', 'off']]) input.setAttribute(name, value);
  const toggle = document.createElement('button');
  Object.assign(toggle, {type: 'button', tabIndex: -1, className: 'combobox-toggle'});
  toggle.setAttribute('aria-hidden', 'true');
  const list = document.createElement('div');
  Object.assign(list, {id: `${id}-list`, className: 'combobox-list', hidden: true});
  list.setAttribute('role', 'listbox');
  host.append(input, toggle, list);

  let source = [], items = [], selected = '', shown = [], active = -1, query = '';
  const typed = () => nativeInputValue.get.call(input);
  const current = () => free ? typed() : selected;
  const show = value => nativeInputValue.set.call(input, free ? value : items.find(item => item.value === value)?.label ?? value);
  function setOptions(next) {
    source = next;
    items = next.map(comboboxOption);
    if (!free && selected && !items.some(item => item.value === selected)) items.push({value: selected, label: selected, detail: 'Unavailable in current data', search: selected});
    if (!list.hidden) render();
  }
  function note(text) {
    const row = document.createElement('p');
    row.className = 'combobox-note';
    row.textContent = text;
    list.append(row);
  }
  function activate(index, scroll = true) {
    list.querySelector('.active')?.classList.remove('active');
    active = index;
    const row = index >= 0 ? list.querySelector(`#${id}-option-${index}`) : null;
    if (!row) { input.removeAttribute('aria-activedescendant'); return; }
    row.classList.add('active');
    input.setAttribute('aria-activedescendant', row.id);
    if (scroll) row.scrollIntoView?.({block: 'nearest'});
  }
  function render() {
    const ranked = rankOptions(items, query);
    shown = ranked.slice(0, limit);
    list.replaceChildren();
    const custom = free && query.trim() && !items.some(item => item.value === query.trim());
    if (custom) note(`Uses “${query.trim()}” as typed, or choose a suggestion.`);
    shown.forEach(({option, positions}, index) => {
      const row = document.createElement('div');
      row.id = `${id}-option-${index}`;
      row.className = 'combobox-option';
      row.setAttribute('role', 'option');
      row.setAttribute('aria-selected', String(option.value === current()));
      const label = document.createElement('span');
      label.className = 'combobox-label';
      const marked = new Set(positions);
      let run = '', highlighted = false;
      const flush = () => { if (!run) return; if (highlighted) { const mark = document.createElement('mark'); mark.textContent = run; label.append(mark); } else label.append(run); run = ''; };
      [...option.label].forEach((character, index) => { if (marked.has(index) !== highlighted) { flush(); highlighted = !highlighted; } run += character; });
      flush();
      row.append(label);
      if (option.detail) { const detail = document.createElement('small'); detail.textContent = option.detail; row.append(detail); }
      row.onmousedown = event => event.preventDefault();
      row.onclick = () => commit(option.value);
      row.onmousemove = () => { if (active !== index) activate(index, false); };
      list.append(row);
    });
    if (!shown.length && !custom) note(items.length ? 'No matches. Try fewer letters.' : free ? 'No suggestions loaded. Type an ID.' : 'No options available.');
    if (ranked.length > shown.length) note(`Showing ${shown.length} of ${ranked.length}. Keep typing to narrow the list.`);
    // Typed text in a free field stays the value until the user picks a suggestion.
    const selectedIndex = shown.findIndex(entry => entry.option.value === current());
    activate(query ? (free || !shown.length ? -1 : 0) : selectedIndex, false);
  }
  // The list opens below the field unless the nearest scrolling container has more room above.
  function place() {
    const box = input.getBoundingClientRect();
    let top = 0, bottom = window.innerHeight;
    for (let parent = host.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
      if (!['auto', 'scroll'].includes(getComputedStyle(parent).overflowY)) continue;
      const bounds = parent.getBoundingClientRect();
      top = Math.max(top, bounds.top);
      bottom = Math.min(bottom, bounds.bottom);
      break;
    }
    const below = bottom - box.bottom - 12, above = box.top - top - 12, upward = below < 200 && above > below;
    host.dataset.placement = upward ? 'top' : 'bottom';
    list.style.maxHeight = `${Math.round(Math.max(120, Math.min(320, upward ? above : below)))}px`;
  }
  function open() {
    if (input.disabled || (free && !items.length)) return;
    if (list.hidden) {
      list.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      host.dataset.open = 'true';
    }
    render();
    place();
    if (active >= 0) list.querySelector(`#${id}-option-${active}`)?.scrollIntoView?.({block: 'nearest'});
  }
  function close() {
    list.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    delete host.dataset.open;
    input.removeAttribute('aria-activedescendant');
    query = '';
    active = -1;
    // A focused strict field keeps its label selected so typing starts a new search.
    if (!free && document.activeElement === input) input.select();
  }
  function revert() {
    if (!free) show(selected);
    close();
  }
  function commit(value) {
    const changed = value !== current();
    if (!free) selected = value;
    show(value);
    close();
    if (!changed) return;
    if (free) input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }

  // A strict field's typed text is only a search query, so the browser's own change event is
  // not a selection. The capture listener runs before any `onchange` handler and stops it.
  if (!free) input.addEventListener('change', event => { if (event.isTrusted) event.stopImmediatePropagation(); }, true);
  input.addEventListener('input', event => { if (!event.isTrusted) return; query = typed(); open(); });
  input.addEventListener('focus', () => { if (!free) input.select(); });
  // Reopening a strict field selects its label so typing starts a new search.
  input.addEventListener('click', () => { if (!list.hidden) return; if (!free) input.select(); open(); });
  input.addEventListener('keydown', event => {
    const isOpen = !list.hidden;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (!isOpen) { open(); return; }
      if (!shown.length) return;
      const step = event.key === 'ArrowDown' ? 1 : -1;
      activate(active < 0 ? (step > 0 ? 0 : shown.length - 1) : Math.max(0, Math.min(shown.length - 1, active + step)));
    } else if (event.key === 'Enter' && isOpen) {
      if (active >= 0) { event.preventDefault(); commit(shown[active].option.value); }
      else if (!free) { event.preventDefault(); revert(); }
      else close();
    } else if (event.key === 'Escape' && isOpen) {
      // The list closes first; a second Escape reaches the surrounding dialog.
      event.preventDefault();
      event.stopPropagation();
      revert();
    } else if (event.key === 'Tab' && isOpen) {
      if (!free && query && active >= 0) commit(shown[active].option.value);
      else revert();
    }
  });
  host.addEventListener('focusout', event => { if (!host.contains(event.relatedTarget)) revert(); });
  toggle.onmousedown = event => {
    event.preventDefault();
    if (input.disabled) return;
    if (list.hidden) { input.focus(); open(); } else revert();
  };

  if (!free) selected = chosen !== undefined ? String(chosen) : options.length ? String(options[0][0]) : '';
  setOptions(options);
  show(free ? String(chosen ?? '') : selected);
  if (!free) Object.defineProperty(input, 'value', {
    configurable: true,
    get: () => selected,
    set: value => { selected = value == null ? '' : String(value); setOptions(source); show(selected); },
  });
  input.host = host;
  input.setOptions = setOptions;
  return input;
}

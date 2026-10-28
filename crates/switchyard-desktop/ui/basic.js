// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const invoke = window.__TAURI__.core.invoke;
const message = document.getElementById('message');
let timer;
let busy = false;

/** Missing usage stays unavailable so a failed read cannot appear as zero. */
function counts(label, value) {
  return value
    ? `${label}: ${value.requests.toLocaleString()} requests · ${value.tokens.toLocaleString()} tokens`
    : `${label}: usage unavailable`;
}

/** Each refresh updates the window and schedules the next check, including after errors. */
async function refresh() {
  if (busy) return;
  busy = true;
  clearTimeout(timer);
  let seconds = 30;
  try {
    const view = await invoke('snapshot');
    document.getElementById('server').textContent = view.health.state === 'running'
      ? 'Server is running' : `Server is not responding: ${view.health.reason}`;
    document.getElementById('server-url').textContent = view.server_url;
    document.getElementById('today').textContent = counts('Today', view.today);
    document.getElementById('week').textContent = counts('Last 7 days', view.week);
    message.textContent = view.errors.join('\n');
    seconds = view.refresh_seconds;
  } catch (error) {
    document.getElementById('server').textContent = 'Server status unavailable';
    document.getElementById('server-url').textContent = '';
    document.getElementById('today').textContent = counts('Today', null);
    document.getElementById('week').textContent = counts('Last 7 days', null);
    message.textContent = String(error);
  } finally {
    busy = false;
    timer = setTimeout(refresh, Math.min(seconds * 1000, 2147483647));
  }
}

document.getElementById('refresh').addEventListener('click', refresh);
for (const [id, kind] of [['open-config', 'config'], ['open-logs', 'logs']]) {
  document.getElementById(id).addEventListener('click', async () => {
    try {
      await invoke('open_file', { kind });
      message.textContent = '';
    } catch (error) {
      message.textContent = String(error);
    }
  });
}
document.getElementById('quit').addEventListener('click', () => invoke('quit_app'));
refresh();

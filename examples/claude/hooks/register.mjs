// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Claude keeps the full prompt. Switchyard receives visible text and tool activity.
export function routingMessages(transcript) {
  const calls = new Set();
  return transcript.flatMap((message) => {
    const content = [];
    for (const result of message.toolResults ?? []) {
      // The transcript API can truncate older messages. Drop orphaned results.
      if (calls.has(result.tool_use_id)) {
        content.push({ type: "tool_result", tool_call_id: result.tool_use_id,
          content: [{ type: "text", text: result.text }], is_error: result.isError });
      }
    }
    if (message.text) content.push({ type: "text", text: message.text });
    for (const call of message.toolUses ?? []) {
      calls.add(call.tool_use_id);
      content.push({ type: "tool_call", id: call.tool_use_id, name: call.tool, arguments: call.input });
    }
    return content.length ? [{ role: message.role, content }] : [];
  });
}

function selectedModel(body) {
  const model = body.model;
  if (typeof model !== "string" || !/^claude-[\w.-]+$/.test(model)) {
    throw new Error("Switchyard did not select a native Claude model");
  }
  return model;
}

const DECISION_MS = 3000;
const CLASSIFIER_MS = 10000;
const ROUTING_MS = 15000;
const WAITING = "Switchyard · waiting for a request";

// Races a promise against a deadline on the session clock.
function within($, ms, promise, message) {
  return new Promise((resolve, reject) => {
    const timer = $.clock.after(ms, () => reject(new Error(message)));
    promise.then(resolve, reject).finally(() => timer.cancel());
  });
}

// Controls follow the active Claude session. The Runner holds routing state by session ID.
let sessionId;
let enabled = true;
let status = WAITING;
let generation = 0;
let nativeModel;
const preferenceKey = "switchyard.enabled:";
let preferenceWrites = Promise.resolve();

function savePreference($, session, value) {
  const saved = preferenceWrites.then(() => $.store.set(preferenceKey + session, value));
  preferenceWrites = saved.catch(() => {});
  return saved;
}

// One helper and Runner for this plugin runtime, including while routing is off.
let life;

// Only the helper's own directories are removed, whatever the helper printed.
async function removeDir($, dir) {
  if (typeof dir === "string" && /\/switchyard-claude-[\w-]+$/.test(dir)) {
    await $.process.run(["rm", "-rf", dir], { timeoutMs: 1000 }).catch(() => {});
  }
}

// Ends a helper and fails what waits on it. Removing its directory also ends a
// helper that survives the kill.
async function close(l, $, error) {
  if (l.closed) return;
  l.closed = true;
  if (life === l) life = undefined;
  for (const { abort } of l.classifications.values()) abort.abort();
  for (const { reject } of l.pending.values()) reject(error);
  l.pending.clear();
  l.rejectReady(error);
  // Not awaited: the stream may hold this until its next piece arrives.
  try { l.stream.return?.(undefined)?.catch?.(() => {}); } catch {}
  await removeDir($, l.dir);
}

function handleLine(l, line) {
  const message = JSON.parse(line);
  if (message.dir !== undefined) {
    l.dir = message.dir;
    l.resolveReady();
  } else if (message.classifier !== undefined) {
    l.pending.get(message.id)?.classify(message);
  } else if (l.pending.has(message.id)) {
    const { resolve, reject } = l.pending.get(message.id);
    l.pending.delete(message.id);
    // The helper's own errors leave its state intact.
    if (message.error !== undefined) reject(Object.assign(new Error(message.error), { fromHelper: true }));
    else resolve(message);
  }
}

function launch($, config) {
  $.ui.status("Switchyard · starting Runner with fresh routing state");
  const l = { pending: new Map(), classifications: new Map(), writes: Promise.resolve(), nextId: 0, closed: false, dir: undefined, stderr: "" };
  // The session owner can cancel a request even after its turn's dispatch is gone.
  l.cancelRequest = (id) => $.fs.write(`${l.dir}/cancel-${id}`, "").catch(() => {});
  l.ready = new Promise((resolve, reject) => { l.resolveReady = resolve; l.rejectReady = reject; });
  l.ready.catch(() => {});
  // Mods cannot load compiled code, and a child's stdin closes after one write. The
  // bundled helper keeps one Runner alive and reads requests from files instead.
  l.stream = $.process.spawn({ argv: [`${$.plugin.root}/bin/switchyard-claude`, config] });
  l.stream.result?.catch(() => {});
  life = l;
  void (async () => {
    let buffer = "";
    try {
      for await (const { stream, text } of l.stream) {
        if (l.closed) {
          // A helper that announces itself after being closed still gets cleaned up.
          try { await removeDir($, JSON.parse(text.split("\n")[0]).dir); } catch {}
          break;
        }
        if (stream === "stderr") { l.stderr = (l.stderr + text).slice(-500); continue; }
        buffer += text;
        for (let end; (end = buffer.indexOf("\n")) >= 0;) {
          handleLine(l, buffer.slice(0, end));
          buffer = buffer.slice(end + 1);
        }
      }
      await close(l, $, new Error(l.stderr.trim() || "Switchyard helper exited"));
    } catch (error) {
      await close(l, $, error);
    }
  })();
  return l;
}

// Work belongs to the generation and session it began in. After any await it must
// stop if either changed, even when the lifecycle that changed them is not its own.
const isStale = (g, session) => g !== generation || session !== sessionId;
const staleError = () => Object.assign(new Error("Switchyard routing stopped"), { stale: true });

// Claude ties a spawned child to the dispatch that spawned it and kills it when that
// dispatch is abandoned, such as an interrupted turn. The loop started by session.start
// outlives every turn, so every helper is spawned from it. Callers queue a job and wait.
let jobs = [];
let wakeController;
let controllerStarted = false;

async function controller($) {
  for (;;) {
    for (let job; (job = jobs.shift());) {
      try {
        if (isStale(job.g, job.session)) throw staleError();
        job.resolve(life ?? launch($, job.config));
      } catch (error) {
        job.reject(error);
      }
    }
    await new Promise((resolve) => { wakeController = resolve; });
  }
}

function startHelper(g, session, config) {
  if (!controllerStarted) return Promise.reject(new Error("Switchyard helper owner not started"));
  return new Promise((resolve, reject) => {
    jobs.push({ g, session, config, resolve, reject });
    wakeController?.();
  });
}

// The live helper, or a new one. Nothing starts if routing stopped or moved on.
async function acquire($, g, session) {
  if (isStale(g, session)) throw staleError();
  if (life) return life;
  const config = (await $.env.get("SWITCHYARD_CONFIG")) || `${$.plugin.root}/switchyard.toml`;
  if (isStale(g, session)) throw staleError();
  return life ?? startHelper(g, session, config);
}

async function decide($, messages, g, session, turn, signal) {
  const l = await acquire($, g, session);
  let id;
  const cancel = () => l.pending.get(id)?.cancel();
  try {
    await within($, DECISION_MS, l.ready, "Switchyard helper did not start");
    if (isStale(g, session)) throw staleError();
    id = ++l.nextId;
    const answer = new Promise((resolve, reject) => l.pending.set(id, {
      resolve, reject,
      cancel: () => {
        l.classifications.get(session)?.abort.abort();
        // Release a pending judge without throwing away this Runner's other sessions.
        void l.cancelRequest(id);
        reject(staleError());
      },
      classify: async ({ classifier, call, cache_turn }) => {
        try {
          if (l.closed || isStale(g, session) || signal.aborted) throw staleError();
          let cached = l.classifications.get(session);
          // Retries of a user turn reuse its answer (including failure); tool
          // continuations normally never reach the classifier at all.
          if (!cache_turn || !cached || cached.turn !== turn) {
            const abort = new AbortController();
            const result = $.model.complete({ ...classifier, effort: "low", timeoutMs: CLASSIFIER_MS }, { signal: abort.signal })
              .then((r) => r.isAnswered ? { text: r.text } : { error: r.reason })
              .catch(() => ({ error: "Classifier unavailable" }));
            cached = { turn, result, abort };
            l.classifications.set(session, cached);
          }
          const reply = await cached.result;
          if (l.closed || isStale(g, session) || signal.aborted) throw staleError();
          await $.fs.write(`${l.dir}/judge-${id}-${call}.json`, JSON.stringify(reply));
        } catch (error) {
          if (!error.stale) l.pending.get(id)?.reject(Object.assign(error, { fromHelper: true }));
          l.pending.get(id)?.cancel();
        }
      },
    }));
    answer.catch(() => {});
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) cancel();
    // Publish in ID order, even if filesystem calls from two hooks complete out of order.
    const write = l.writes.then(() => {
      if (l.closed || isStale(g, session) || signal.aborted) throw staleError();
      return $.fs.write(`${l.dir}/req-${id}.json`, `${JSON.stringify({ session_id: session, messages })}\n`);
    });
    l.writes = write.catch(() => {});
    const [, decision] = await within($, ROUTING_MS, Promise.all([write, answer]), "Switchyard decision timed out");
    return decision;
  } catch (error) {
    // A silent or broken helper is replaced by a fresh one on the next request. A stale
    // request leaves the helper alone: it is gone, or belongs to newer work.
    if (!error.fromHelper && !error.stale && !isStale(g, session)) await close(l, $, error);
    throw error;
  } finally {
    signal.removeEventListener("abort", cancel);
    l.pending.delete(id);
  }
}

function invalidate() {
  generation += 1;
  // Discard waiting decisions without destroying the Runner's session state.
  for (const { cancel } of life?.pending.values() ?? []) cancel();
  life?.pending.clear();
}

function stop($) {
  invalidate();
  return life ? close(life, $, new Error("Switchyard routing stopped")) : Promise.resolve();
}

function disable($, model) {
  enabled = false;
  status = `Switchyard off · Claude selected ${model}; /switchyard on to resume`;
  invalidate();
  return savePreference($, sessionId, false);
}

// Claude does not fire session.start after /clear, /resume or /branch, so every
// hook checks that the session is the one this state belongs to.
async function sync($) {
  const id = await $.session.id();
  if (id === sessionId) return true;
  const started = generation;
  const model = await $.session.model();
  const preference = await $.store.get(preferenceKey + id);
  if (id !== await $.session.id()) return false;
  // Another hook may have synced meanwhile; resetting again would undo its /switchyard off.
  if (id === sessionId) return true;
  if (started !== generation) return false;
  invalidate();
  sessionId = id;
  enabled = preference !== false;
  status = enabled ? WAITING : "Switchyard off";
  nativeModel = model;
  return true;
}

export function register(on) {
  on("session.start", async ($, e, next) => {
    if (!controllerStarted) {
      controllerStarted = true;
      void controller($);
    }
    await $.command.register({ name: "switchyard", description: "Switchyard routing: on, off, or status" });
    if (await sync($)) $.ui.status(status);
    return next(e);
  });

  on("session.end", async ($, e, next) => {
    if (["clear", "resume"].includes(e.reason)) invalidate();
    else await stop($);
    sessionId = undefined;
    enabled = true;
    status = WAITING;
    $.ui.status(status);
    return next(e);
  });

  on("command.run", { command: "switchyard" }, async ($, e) => {
    const action = e.args.trim() || "status";
    if (action !== "on" && action !== "off" && action !== "status") {
      return { text: "Usage: /switchyard [on|off|status]" };
    }
    if (!await sync($)) return { text: status };
    if (action === "off") {
      enabled = false;
      status = "Switchyard off";
      invalidate();
      await savePreference($, sessionId, false);
    } else if (action === "on") {
      if (!enabled) {
        const started = generation;
        const session = sessionId;
        const model = await $.session.model();
        if (isStale(started, session)) return { text: status };
        // Set together after the await, so no turn sees routing on with a stale model.
        if (!enabled) {
          enabled = true;
          generation += 1;
          nativeModel = model;
          status = WAITING;
          const enabledAt = generation;
          await savePreference($, session, true);
          if (isStale(enabledAt, session)) return { text: status };
        }
      }
      // Start the helper now so a failure shows in the status. It stays if running.
      const started = generation;
      const session = sessionId;
      let l;
      try {
        l = await acquire($, started, session);
        await within($, DECISION_MS, l.ready, "Switchyard helper did not start");
      } catch (error) {
        if (!isStale(started, session)) {
          status = `Switchyard unavailable · ${error.message}`;
          // A helper that missed the deadline is replaced by the next request.
          if (l) await close(l, $, error);
        }
      }
    }
    $.ui.status(status);
    return { text: status };
  });

  on("command.run", { command: "model" }, async ($, e, next) => {
    if (!await sync($)) return next(e);
    const started = generation;
    const session = sessionId;
    const before = await $.session.model();
    const result = await next(e);
    const after = await $.session.model();
    const id = await $.session.id();
    if (id === session && !isStale(started, session) && (e.args.trim() || before !== after)) {
      await disable($, after);
      $.ui.status(status);
    }
    return result;
  });

  on("turn.step", async function* ($, e, next) {
    // session.messages() describes the main conversation, not a subagent's work.
    if (e.agentId || next.signal.aborted) return yield* next(e);
    let request = e;
    let started = generation;
    let session = sessionId;
    try {
      if (!await sync($)) throw staleError();
      started = generation;
      session = sessionId;
      if (enabled) {
        // Covers picker changes after command.run returns, SDK changes, and
        // native fallbacks without using managed-only classic hooks.
        const current = await $.session.model();
        if (isStale(started, session)) throw staleError();
        if (nativeModel !== undefined && nativeModel !== current) {
          await disable($, current);
          $.ui.status(status);
        }
        nativeModel = current;
        if (enabled && !isStale(started, session)) {
          const messages = routingMessages(await $.session.messages());
          if (session !== await $.session.id() || next.signal.aborted) throw staleError();
          if (!messages.length) throw new Error("No routing context available");
          const decision = await decide($, messages, started, session, e.turnId, next.signal);
          const model = selectedModel(decision);
          // Read the session as it is now: a picker or SDK change needs no hook of ours.
          const [id, now] = [await $.session.id(), await $.session.model()];
          if (id === session && now !== current && enabled && !isStale(started, session)) {
            await disable($, now);
            $.ui.status(status);
          } else if (id === session && enabled && !isStale(started, session) && !next.signal.aborted) {
            request = { ...e, model };
            const fallback = decision.evidence?.source === "fail_open" ? " · classifier unavailable; stage routing" : "";
            status = `Switchyard ${decision.algorithm === "composite" ? "Composite" : "Auto"} · ${model}${fallback}`;
            $.ui.status(status);
          }
        }
      }
    } catch (error) {
      if (!error.stale && !isStale(started, session) && !next.signal.aborted) {
        status = `Switchyard unavailable · keeping ${e.model}: ${error.message}`;
        $.ui.status(status);
      }
    }
    // Keep completion failures outside the routing catch: never replay a stream.
    return yield* next(request);
  });
}

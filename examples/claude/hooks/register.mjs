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

const prefix = "switchyard.session.v1:";
let current;
let generation = 0;
let writes = Promise.resolve();
let steps = Promise.resolve();

// Controls and commits share a short lock; model streaming never blocks controls.
function locked(fn) {
  const work = writes.then(fn);
  writes = work.catch(() => {});
  return work;
}

async function context($) {
  const id = await $.session.id();
  if (current?.id === id) return current;
  generation += 1;
  const saved = await $.store.get(prefix + id);
  if (saved !== undefined && (saved.version !== 1 || typeof saved.enabled !== "boolean")) {
    throw new Error("Invalid saved routing state; /switchyard reset to clear it");
  }
  current = { id, data: saved ?? {
    version: 1, enabled: true, nativeModel: await $.session.model(), checkpoint: null,
    status: "Switchyard Auto · waiting for a request",
  } };
  return current;
}

async function save($, ctx) {
  ctx.data.updatedAt = await $.clock.now();
  await $.store.set(prefix + ctx.id, ctx.data);
}

async function disable($, ctx, model) {
  generation += 1;
  ctx.data = { ...ctx.data, enabled: false, nativeModel: model,
    status: `Switchyard off · Claude selected ${model}; /switchyard on to resume` };
  await save($, ctx);
  $.ui.status(ctx.data.status);
}

export function register(on) {
  on("session.start", async ($, e, next) => {
    await $.command.register({ name: "switchyard", description: "Switchyard routing: on, off, status, or reset" });
    await locked(async () => {
      const ctx = await context($);
      $.ui.status(ctx.data.status);
      // Bound persistent history without expiring an actively resumed session.
      const entries = [];
      for (const key of await $.store.keys()) {
        if (key.startsWith(prefix) && key !== prefix + ctx.id) {
          entries.push([key, (await $.store.get(key))?.updatedAt ?? 0]);
        }
      }
      entries.sort((a, b) => b[1] - a[1]);
      for (const [key] of entries.slice(127)) await $.store.delete(key);
    });
    return next(e);
  });

  on("session.end", async ($, e, next) => {
    generation += 1;
    await locked(async () => { generation += 1; current = undefined; });
    if (e.reason === "clear" || e.reason === "resume") {
      $.ui.status("Switchyard · session changed; routing state loads on the next request");
    }
    return next(e);
  });

  on("command.run", { command: "switchyard" }, async ($, e) => {
    const action = e.args.trim() || "status";
    if (!["on", "off", "status", "reset"].includes(action)) {
      return { text: "Usage: /switchyard [on|off|status|reset]" };
    }
    if (action !== "status") generation += 1;
    return locked(async () => {
      if (action !== "status") generation += 1;
      if (action === "reset") {
        await $.store.delete(prefix + await $.session.id());
        current = undefined;
      }
      const ctx = await context($);
      if (action === "on" || action === "off") {
        ctx.data = { ...ctx.data, enabled: action === "on", nativeModel: await $.session.model(),
          status: action === "on" ? "Switchyard Auto · waiting for a request" : "Switchyard off" };
      }
      if (action !== "status") await save($, ctx);
      $.ui.status(ctx.data.status);
      return { text: ctx.data.status };
    });
  });

  on("command.run", { command: "model" }, async ($, e, next) => {
    const before = await $.session.model();
    const result = await next(e);
    const after = await $.session.model();
    if (e.args.trim() || before !== after) {
      generation += 1;
      await locked(async () => disable($, await context($), after));
    }
    return result;
  });

  on("turn.step", async function* ($, e, next) {
    // session.messages() describes the main conversation, not a subagent's work.
    if (e.agentId || next.signal.aborted) return yield* next(e);
    // Serialize main steps, including duplicate dispatches, without blocking on/off.
    const previous = steps;
    let release;
    steps = new Promise(resolve => { release = resolve; });
    await previous;
    try {
      let ctx;
      let started;
      let decision;
      let request = e;
      try {
        await locked(async () => {
          ctx = await context($);
          const native = await $.session.model();
          if (ctx.data.enabled && ctx.data.nativeModel !== native) await disable($, ctx, native);
          started = generation;
        });
        if (ctx.data.enabled && !next.signal.aborted) {
          const config = (await $.env.get("SWITCHYARD_CONFIG")) || `${$.plugin.root}/switchyard.toml`;
          const messages = routingMessages(await $.session.messages());
          if (!messages.length) throw new Error("No routing context available");
          const decisionId = `${e.turnId}:${e.index}`;
          const result = await $.process.run([`${$.plugin.root}/bin/switchyard-claude`, config], {
            stdin: JSON.stringify({ version: 1, session_id: ctx.id, decision_id: decisionId,
              messages, checkpoint: ctx.data.checkpoint }), timeoutMs: 3000,
          });
          if (result.exitCode !== 0) throw new Error(result.stderr.trim() || "Switchyard helper failed");
          decision = JSON.parse(result.stdout);
          const model = selectedModel(decision);
          if (decision.version !== 1 || decision.session_id !== ctx.id || decision.decision_id !== decisionId
              || decision.checkpoint?.version !== 1 || decision.checkpoint.session_id !== ctx.id) {
            throw new Error("Switchyard returned an invalid session checkpoint");
          }
          if (started === generation && ctx.id === await $.session.id() && !next.signal.aborted) {
            request = { ...e, model };
            ctx.data.status = `Switchyard Auto · ${model}${decision.state_reset ? " · configuration changed; state reset" : ""}`;
            $.ui.status(ctx.data.status);
          } else {
            decision = undefined;
          }
        }
      } catch (error) {
        decision = undefined;
        if ((started === undefined || started === generation) && !next.signal.aborted) {
          const status = `Switchyard unavailable · keeping ${e.model}: ${error.message}`;
          if (ctx) ctx.data.status = status;
          $.ui.status(status);
        }
      }
      // A failed or abandoned completion never commits its tentative checkpoint.
      // Keep this outside the routing catch: never replay a completion stream.
      const result = yield* next(request);
      if (decision && !next.signal.aborted) {
        await locked(async () => {
          if (started !== generation || ctx.id !== await $.session.id() || next.signal.aborted) return;
          const data = { ...ctx.data, checkpoint: decision.checkpoint, status: `Switchyard Auto · ${request.model}` };
          try {
            await save($, { id: ctx.id, data });
            ctx.data = data;
          } catch (error) {
            $.ui.status(`Switchyard state not saved: ${error.message}; /switchyard reset to retry`);
          }
        });
      }
      return result;
    } finally {
      release();
    }
  });
}

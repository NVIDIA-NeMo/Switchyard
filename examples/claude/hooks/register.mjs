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

export function register(on) {
  let enabled = true;
  let status = "Switchyard Auto · waiting for a request";
  let generation = 0;
  let nativeModel;

  function disable(model) {
    enabled = false;
    generation += 1;
    status = `Switchyard off · Claude selected ${model}; /switchyard on to resume`;
  }

  on("session.start", async ($, e, next) => {
    await $.command.register({ name: "switchyard", description: "Switchyard routing: on, off, or status" });
    nativeModel = await $.session.model();
    $.ui.status(status);
    return next(e);
  });

  on("command.run", { command: "switchyard" }, async ($, e) => {
    const action = e.args.trim() || "status";
    if (action === "on" || action === "off") {
      enabled = action === "on";
      generation += 1;
      nativeModel = await $.session.model();
      status = enabled ? "Switchyard Auto · waiting for a request" : "Switchyard off";
      $.ui.status(status);
    } else if (action !== "status") {
      return { text: "Usage: /switchyard [on|off|status]" };
    }
    return { text: status };
  });

  on("command.run", { command: "model" }, async ($, e, next) => {
    const before = await $.session.model();
    const result = await next(e);
    const after = await $.session.model();
    if (e.args.trim() || before !== after) {
      disable(after);
      $.ui.status(status);
    }
    return result;
  });

  on("turn.step", async function* ($, e, next) {
    // session.messages() describes the main conversation, not a subagent's work.
    if (!enabled || e.agentId || next.signal.aborted) return yield* next(e);
    const started = generation;
    let request = e;
    try {
      // Covers picker changes after command.run returns, SDK changes, and
      // native fallbacks without using managed-only classic hooks.
      const current = await $.session.model();
      if (nativeModel !== undefined && nativeModel !== current) {
        disable(current);
        $.ui.status(status);
      }
      nativeModel = current;
      if (enabled) {
        const config = (await $.env.get("SWITCHYARD_CONFIG")) || `${$.plugin.root}/switchyard.toml`;
        const messages = routingMessages(await $.session.messages());
        if (!messages.length) throw new Error("No routing context available");
        // Mods cannot load compiled code. The bundled helper calls Switchyard's
        // runner in a short-lived process, with a host-enforced deadline.
        const result = await $.process.run([`${$.plugin.root}/bin/switchyard-claude`, config], {
          stdin: JSON.stringify(messages), timeoutMs: 3000,
        });
        if (result.exitCode !== 0) throw new Error(result.stderr.trim() || "Switchyard helper failed");
        const model = selectedModel(JSON.parse(result.stdout));
        if (enabled && started === generation && !next.signal.aborted) {
          request = { ...e, model };
          status = `Switchyard Auto · ${model}`;
          $.ui.status(status);
        }
      }
    } catch (error) {
      if (started === generation && !next.signal.aborted) {
        status = `Switchyard unavailable · keeping ${e.model}: ${error.message}`;
        $.ui.status(status);
      }
    }
    // Keep completion failures outside the routing catch: never replay a stream.
    return yield* next(request);
  });
}

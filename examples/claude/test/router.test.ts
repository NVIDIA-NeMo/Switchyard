// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, mock, test } from "claude-code/testing";

const input = { turnId: "turn", index: 0, model: "claude-sonnet-4-6", effort: "high", messageCount: 1 };
const prompt = { role: "user", text: "Fix the tests", toolUses: [] };
const selection = (model = "claude-opus-5-5") => ({ target: "capable", model });

// Stands in for the native helper: same protocol, none of the routing.
function setup(on, options = {}) {
  const sent = [];
  const statuses = [];
  const helpers = [];
  const removed = [];
  const classifierCalls = [];
  const clock = mock.clock(on);
  const stored = new Map(Object.entries(options.store ?? {}));
  on("store.get", ($, e) => ({ value: stored.get(e.key) }));
  on("store.set", async ($, e) => {
    await options.storeSet?.(e, clock);
    stored.set(e.key, e.value);
    return { value: null };
  });
  mock.env(on, { SWITCHYARD_CONFIG: "/tmp/switchyard-test.toml" });
  on("ui.status", ($, e) => { statuses.push(e.text); return { value: null }; });
  let nativeModel = input.model;
  let session = "session-a";
  // A test sets a gate to hold the next read of the model or transcript until it opens it.
  const gates = {};
  const open = async (name) => { await gates[name]?.promise; };
  const gate = (name) => {
    let release;
    gates[name] = { promise: new Promise((resolve) => { release = resolve; }) };
    return release;
  };
  on("session.model", async () => { const value = nativeModel; await open("model"); return { value }; });
  on("session.id", () => ({ value: session }));
  on("session.messages", async () => { await open("messages"); return { value: options.messages ?? [prompt] }; });
  on("model.complete", async ($, e) => {
    classifierCalls.push(e);
    await open("classifier");
    return { value: options.classifierResult ?? { isAnswered: true, text: '{"p_solve":0.9}', usage: null } };
  });
  on("process.spawn", async function* ($, e) {
    const helper = { argv: e.argv, dir: `/tmp/switchyard-claude-${helpers.length}`, requests: [], alive: true, queue: [], wake: undefined };
    helper.emit = (line) => { helper.queue.push(line); helper.wake?.(); };
    helper.exit = () => { helper.alive = false; helper.wake?.(); };
    helpers.push(helper);
    try {
      if (options.startup !== "silent") yield { stream: "stdout", text: `${JSON.stringify({ dir: helper.dir })}\n` };
      while (helper.alive || helper.queue.length) {
        if (helper.queue.length) yield helper.queue.shift();
        else await new Promise((resolve) => { helper.wake = resolve; });
      }
    } finally {
      // Also what a stream closed early ends with: the kit wants a result either way.
      helper.alive = false;
      // eslint-disable-next-line no-unsafe-finally
      return { value: { code: null, signal: "SIGTERM" } };
    }
  });
  on("fs.write", async ($, e) => {
    await open("write");
    await options.write?.(e, clock);
    const helper = helpers.find((h) => e.path.startsWith(`${h.dir}/`));
    if (/\/cancel-/.test(e.path)) return { value: undefined };
    const judge = /judge-(\d+)-(\d+)\.json$/.exec(e.path);
    if (judge) {
      helper.replies ??= [];
      helper.replies.push(JSON.parse(e.text));
      helper.emit({ stream: "stdout", text: `${JSON.stringify({ id: Number(judge[1]), ...selection() })}\n` });
      return { value: undefined };
    }
    const id = Number(/req-(\d+)\.json$/.exec(e.path)?.[1]);
    const body = JSON.parse(e.text);
    helper.requests.push({ id, ...body });
    const answer = options.answer?.(helper, id, body) ?? (options.classifier ? {
      id, call: 0, cache_turn: true, classifier: { model: "claude-sonnet-5-5", prompt: "Classify", system: "Return JSON", maxTokens: 384 },
    } : { id, ...selection() });
    if (answer !== "silent") helper.emit({ stream: "stdout", text: `${typeof answer === "string" ? answer : JSON.stringify(answer)}\n` });
    return { value: undefined };
  });
  // The mod removes the helper's directory, which also ends a helper that outlived its kill.
  on("process.run", ($, e) => {
    removed.push(e.argv.at(-1));
    helpers.find((h) => h.dir === e.argv.at(-1))?.exit();
    return { value: { exitCode: 0, stdout: "", stderr: "", isStdoutTruncated: false, isStderrTruncated: false } };
  });
  on("turn.step", async function* ($, e) {
    sent.push(e);
    yield { kind: "text", index: 0, text: "hello" };
    if (options.failCompletion) throw new Error("completion failed");
    return { turnId: e.turnId, index: e.index, answer: "hello", toolUses: [], stopReason: "end_turn", usage: null };
  });
  on("command.run", { command: "model" }, async () => {
    await open("modelCommand");
    return { text: "model selected" };
  });
  on("session.start", ($, e) => ({ cwd: e.cwd }));
  on("session.end", ($, e) => ({ sessionId: e.sessionId }));
  on("command.register", ($, e) => ({ value: { command: e.name } }));
  const alive = () => helpers.filter((h) => h.alive).length;
  return {
    sent, clock, statuses, helpers, removed, alive, gate, stored, classifierCalls,
    setModel: (model) => { nativeModel = model; },
    setSession: (id) => { session = id; },
  };
}

async function drain(stream) {
  const chunks = [];
  let step = await stream.next();
  while (!step.done) {
    chunks.push(step.value);
    step = await stream.next();
  }
  return { chunks, result: step.value };
}


// Claude fires session.start once per load, and the mod spawns every helper from the loop it starts.
const booted = new WeakSet();
const boot = async ($) => {
  if (booted.has($)) return;
  booted.add($);
  await $.session.start({ cwd: "/tmp", surface: "terminal", isInteractive: true });
};
const step = async ($, e = input) => { await boot($); return drain($.turn.step(e)); };
const ask = async ($, args) => { await boot($); return $.command.run({ command: "switchyard", args }); };
const stale = (helper, id) => helper.emit({ stream: "stdout", text: `${JSON.stringify({ id, ...selection("claude-haiku-4-5") })}\n` });

test("selects a native model, preserves effort, and passes through the stream", async ($, on) => {
  const { sent, helpers } = setup(on);
  const output = await step($);
  expect(sent).toEqual([{ ...input, model: "claude-opus-5-5" }]);
  expect(output.chunks).toMatchObject([{ kind: "text", text: "hello" }]);
  expect(output.result.answer).toBe("hello");
  expect(helpers[0].argv[0]).toMatch(/\/bin\/switchyard-claude$/);
  expect(helpers[0].argv.slice(1)).toEqual(["/tmp/switchyard-test.toml"]);
  expect(helpers[0].requests).toEqual([
    { id: 1, session_id: "session-a", messages: [{ role: "user", content: [{ type: "text", text: "Fix the tests" }] }] },
  ]);
});

test("one helper serves every decision of a session", async ($, on) => {
  const { helpers, alive } = setup(on);
  for (let i = 0; i < 3; i++) await step($);
  expect(helpers.length).toBe(1);
  expect(alive()).toBe(1);
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2, 3]);
});

test("passes paired tool calls and failures once, including continuation requests", async ($, on) => {
  const { sent, helpers } = setup(on, { messages: [
    { role: "user", text: "", toolResults: [{ tool_use_id: "old", text: "truncated", isError: true }] },
    prompt,
    { role: "assistant", text: "Checking", toolUses: [
      { tool_use_id: "call", tool: "Bash", input: { command: "pytest" }, text: "out of memory", isError: true },
    ] },
    { role: "user", text: "", toolUses: [], toolResults: [
      { tool_use_id: "call", text: "out of memory", isError: true },
    ] },
    { role: "assistant", text: "", toolUses: [] },
  ] });
  await step($, { ...input, index: 1, messageCount: 5 });
  const { messages } = helpers[0].requests[0];
  expect(messages.length).toBe(3);
  expect(messages[1]).toEqual({ role: "assistant", content: [
    { type: "text", text: "Checking" },
    { type: "tool_call", id: "call", name: "Bash", arguments: { command: "pytest" } },
  ] });
  expect(messages[2]).toEqual({ role: "user", content: [
    { type: "tool_result", tool_call_id: "call", content: [{ type: "text", text: "out of memory" }], is_error: true },
  ] });
  expect(sent[0].model).toBe("claude-opus-5-5");
});

test("does not route a subagent using the parent's transcript", async ($, on) => {
  const { sent, helpers } = setup(on);
  const agent = { ...input, agentId: "worker" };
  await step($, agent);
  expect(sent).toEqual([agent]);
  expect(helpers.length).toBe(0);
});

test("off pauses routing and on reuses the same helper", async ($, on) => {
  const { sent, helpers, removed, alive } = setup(on);
  await step($);
  expect((await ask($, "off")).text).toBe("Switchyard off");
  expect(helpers[0].alive).toBe(true);
  expect(removed).toEqual([]);
  await ask($, "off");
  await step($);
  expect(helpers[0].requests.length).toBe(1);
  await ask($, "on");
  await ask($, "on");
  await step($);
  expect(helpers.length).toBe(1);
  expect(alive()).toBe(1);
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2]);
  expect(sent.map((e) => e.model)).toEqual(["claude-opus-5-5", input.model, "claude-opus-5-5"]);
});

test("a redundant on keeps the helper and its place", async ($, on) => {
  const { helpers } = setup(on);
  await step($);
  await ask($, "on");
  await step($);
  expect(helpers.length).toBe(1);
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2]);
});

test("/model selection pauses routing and retains its helper", async ($, on) => {
  const { sent, helpers, alive } = setup(on);
  await step($);
  await $.command.run({ command: "model", args: "sonnet" });
  expect(alive()).toBe(1);
  await step($);
  expect(sent[1].model).toBe(input.model);
  expect(helpers.length).toBe(1);
  expect((await ask($, "status")).text).toContain("Switchyard off");
});

test("opening and dismissing /model keeps routing and the helper", async ($, on) => {
  const { sent, statuses, helpers } = setup(on);
  await step($);
  await $.command.run({ command: "model", args: "" });
  await step($);
  expect(sent[1].model).toBe("claude-opus-5-5");
  expect(statuses.at(-1)).toContain("claude-opus-5-5");
  expect(helpers.length).toBe(1);
});

test("a native model change between requests ends routing", async ($, on) => {
  const { sent, helpers, statuses, setModel, alive } = setup(on);
  await step($);
  setModel("claude-haiku-4-5");
  const changed = { ...input, model: "claude-haiku-4-5" };
  await step($, changed);
  expect(sent[1]).toEqual(changed);
  expect(helpers.length).toBe(1);
  expect(alive()).toBe(1);
  expect(statuses.at(-1)).toContain("Switchyard off");
});

for (const transition of ["clear", "resume"]) {
  test(`${transition} keeps the Runner and routes the new session separately`, async ($, on) => {
    const { sent, helpers, alive, setSession, statuses } = setup(on);
    await step($);
    await ask($, "off");
    await $.session.end({ reason: transition, sessionId: "session-a", resume: { id: "session-a" } });
    expect(statuses.at(-1)).toBe("Switchyard · waiting for a request");
    setSession("session-b");
    // session.start does not fire here, so the next request notices the new session.
    await step($);
    expect(sent.at(-1).model).toBe("claude-opus-5-5");
    expect(helpers.length).toBe(1);
    expect(helpers[0].requests.at(-1).session_id).toBe("session-b");
    expect(alive()).toBe(1);
  });
}

test("a session switch without session.end uses the same helper with a new ID", async ($, on) => {
  const { helpers, alive, setSession } = setup(on);
  await step($);
  await ask($, "off");
  setSession("session-b");
  expect((await ask($, "status")).text).toBe("Switchyard · waiting for a request");
  await step($);
  expect(helpers.map((h) => h.alive)).toEqual([true]);
  expect(helpers[0].requests.map((r) => r.session_id)).toEqual(["session-a", "session-b"]);
  expect(alive()).toBe(1);
});

test("session end stops the helper", async ($, on) => {
  const { alive } = setup(on);
  await step($);
  await $.session.end({ reason: "other", sessionId: "session-a", resume: { id: "session-a" } });
  expect(alive()).toBe(0);
});

test("an answer that arrives after off is discarded", async ($, on) => {
  const { sent, clock, helpers, statuses } = setup(on, { answer: () => "silent" });
  const pending = step($);
  await clock.settle();
  await ask($, "off");
  stale(helpers[0], 1);
  await clock.advance(10);
  await pending;
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toBe("Switchyard off");
});

test("an answer from the previous session cannot change the next one", async ($, on) => {
  const { sent, clock, helpers, setSession, statuses } = setup(on, { answer: () => "silent" });
  const pending = step($);
  await clock.settle();
  setSession("session-b");
  // The new session invalidates old answers while the Runner keeps both sessions.
  const next = step($);
  await clock.settle();
  stale(helpers[0], 1);
  helpers[0].emit({ stream: "stdout", text: `${JSON.stringify({ id: 2, ...selection("claude-sonnet-5-5") })}\n` });
  await clock.advance(10);
  await Promise.all([pending, next]);
  expect(sent.map((e) => e.model)).toEqual([input.model, "claude-sonnet-5-5"]);
  expect(statuses.at(-1)).toBe("Switchyard Auto · claude-sonnet-5-5");
  expect(helpers[0].alive).toBe(true);
});

test("a decision that times out ends the helper and the next request starts a fresh one", async ($, on) => {
  const { sent, clock, helpers, removed, statuses } = setup(on, { answer: (h) => (h === undefined ? undefined : "silent") });
  const pending = step($);
  await clock.settle();
  await clock.advance(14999);
  expect(helpers[0].alive).toBe(true);
  await clock.advance(1);
  await pending;
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("Switchyard unavailable");
  expect(helpers[0].alive).toBe(false);
  expect(removed).toEqual([helpers[0].dir]);
  expect((await ask($, "status")).text).toContain("timed out");
});

test("a helper that never starts fails within the deadline", async ($, on) => {
  const { sent, clock, statuses, helpers, removed } = setup(on, { startup: "silent" });
  const pending = step($);
  await clock.settle();
  await clock.advance(3000);
  await pending;
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("did not start");
  expect(helpers.length).toBe(1);
  // The helper announces itself after the deadline: it is cleaned up, not left running.
  helpers[0].emit({ stream: "stdout", text: `${JSON.stringify({ dir: helpers[0].dir })}\n` });
  await clock.settle();
  expect(removed).toEqual([helpers[0].dir]);
  expect(helpers[0].alive).toBe(false);
});

test("a helper that exits is replaced on the next request", async ($, on) => {
  const { sent, helpers, statuses, clock } = setup(on);
  await step($);
  helpers[0].emit({ stream: "stderr", text: "boom\n" });
  helpers[0].exit();
  await clock.settle();
  await step($);
  expect(helpers.length).toBe(2);
  expect(sent.map((e) => e.model)).toEqual(["claude-opus-5-5", "claude-opus-5-5"]);
  expect(statuses.at(-1)).toContain("claude-opus-5-5");
});

test("an exit during a decision reports the helper's error", async ($, on) => {
  const { sent, statuses } = setup(on, { answer: (h) => { h.emit({ stream: "stderr", text: "Invalid config\n" }); h.exit(); return "silent"; } });
  await step($);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("Invalid config");
});

test("a malformed helper line ends the helper", async ($, on) => {
  const { sent, helpers } = setup(on, { answer: () => "{" });
  await step($);
  expect(sent).toEqual([input]);
  expect(helpers[0].alive).toBe(false);
});

test("a helper's own error keeps its state and the helper", async ($, on) => {
  const { sent, helpers, statuses } = setup(on, { answer: (h, id) => (id === 1 ? { id, error: "Routing must only select a model" } : undefined) });
  await step($);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("Routing must only select a model");
  await step($);
  expect(helpers.length).toBe(1);
  expect(sent[1].model).toBe("claude-opus-5-5");
});

for (const [name, answer] of [
  ["missing target", (h, id) => ({ id })],
  ["unresolved alias", (h, id) => ({ id, ...selection("sonnet") })],
  ["non-Claude target", (h, id) => ({ id, ...selection("gpt-test") })],
]) {
  test(`keeps Claude's model on ${name}`, async ($, on) => {
    const { sent, statuses } = setup(on, { answer });
    await step($);
    expect(sent).toEqual([input]);
    expect(statuses.at(-1)).toContain("Switchyard unavailable");
    expect((await ask($, "status")).text).toContain("keeping");
  });
}

test("empty context passes through once", async ($, on) => {
  const { sent, helpers } = setup(on, { messages: [] });
  await step($);
  expect(sent).toEqual([input]);
  expect(helpers.length).toBe(0);
});

test("completion errors do not retry or substitute another model", async ($, on) => {
  const { sent } = setup(on, { failCompletion: true });
  let failure = "";
  try { await step($); } catch (error) { failure = error.message; }
  // The kit skips a throwing engine stub and reports a missing implementation.
  expect(failure).toContain("turn.step");
  expect(sent.length).toBe(1);
});

test("startup exposes enabled status and registers controls", async ($, on) => {
  const { statuses, helpers } = setup(on);
  await $.session.start({ cwd: "/tmp", surface: "terminal", isInteractive: true });
  expect(statuses.at(-1)).toBe("Switchyard · waiting for a request");
  expect((await ask($, "")).text).toBe(statuses.at(-1));
  expect(helpers.length).toBe(0);
});

test("off while the transcript is read starts no helper", async ($, on) => {
  const { sent, clock, helpers, gate, alive } = setup(on);
  const release = gate("messages");
  const pending = step($);
  await clock.settle();
  await ask($, "off");
  release();
  await pending;
  expect(helpers.length).toBe(0);
  expect(alive()).toBe(0);
  expect(sent).toEqual([input]);
  expect((await ask($, "status")).text).toBe("Switchyard off");
});

test("off then on while old work reads the transcript leaves the new helper alone", async ($, on) => {
  const { sent, clock, helpers, gate, statuses } = setup(on);
  const release = gate("messages");
  const pending = step($);
  await clock.settle();
  await ask($, "off");
  await ask($, "on");
  expect(helpers.length).toBe(1);
  release();
  await pending;
  expect(helpers[0].requests).toEqual([]);
  expect(helpers[0].alive).toBe(true);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).not.toContain("unavailable");
});

test("a session change with no lifecycle hook discards the answer", async ($, on) => {
  const { sent, clock, helpers, setSession, statuses } = setup(on, { answer: () => "silent" });
  const pending = step($);
  await clock.settle();
  setSession("session-b");
  helpers[0].emit({ stream: "stdout", text: `${JSON.stringify({ id: 1, ...selection() })}\n` });
  await pending;
  expect(sent).toEqual([input]);
  expect(statuses).not.toContain("Switchyard Auto · claude-opus-5-5");
});

test("a native model change during a decision is honored", async ($, on) => {
  const { sent, helpers, setModel, statuses, alive } = setup(on, { answer: (h, id) => { setModel("claude-haiku-4-5"); return { id, ...selection() }; } });
  await step($);
  expect(sent).toEqual([input]);
  expect(alive()).toBe(1);
  expect(helpers.length).toBe(1);
  expect(statuses.at(-1)).toContain("Switchyard off · Claude selected claude-haiku-4-5");
});

test("on does not enable routing until the model is read", async ($, on) => {
  const { sent, clock, helpers, gate, setModel } = setup(on);
  await ask($, "off");
  setModel("claude-haiku-4-5");
  const haiku = { ...input, model: "claude-haiku-4-5" };
  const release = gate("model");
  const enabling = ask($, "on");
  await clock.settle();
  await step($, haiku);
  expect(sent).toEqual([haiku]);
  release();
  await enabling;
  await step($, haiku);
  expect(sent.at(-1).model).toBe("claude-opus-5-5");
  expect(helpers.length).toBe(1);
});

test("off cancels an earlier on while its model read is pending", async ($, on) => {
  const { sent, clock, helpers, alive, gate, statuses } = setup(on);
  await ask($, "off");
  const release = gate("model");
  const enabling = ask($, "on");
  await clock.settle();
  await ask($, "off");
  release();
  await enabling;
  await step($);
  expect(helpers.length).toBe(0);
  expect(alive()).toBe(0);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toBe("Switchyard off");
});

// The kit cannot model the host killing a child on stream.return(), so these tests cover
// the mod's own part: no requests to an unready helper, and removal of a directory that
// is announced after the helper was stopped.
test("off before the helper is ready keeps it without submitting stale work", async ($, on) => {
  const { sent, clock, helpers, removed } = setup(on, { startup: "silent" });
  const pending = step($);
  await clock.settle();
  await ask($, "off");
  helpers[0].emit({ stream: "stdout", text: `${JSON.stringify({ dir: helpers[0].dir })}\n` });
  await clock.settle();
  await pending;
  expect(sent).toEqual([input]);
  expect(helpers[0].requests).toEqual([]);
  expect(removed).toEqual([]);
  expect(helpers[0].alive).toBe(true);
  await ask($, "on");
  await step($);
  expect(helpers.length).toBe(1);
  expect(helpers[0].requests.length).toBe(1);
});

test("an explicit on that misses the startup deadline replaces the helper", async ($, on) => {
  const { clock, helpers, statuses } = setup(on, { startup: "silent" });
  await ask($, "off");
  const enabling = ask($, "on");
  await clock.settle();
  await clock.advance(3000);
  await enabling;
  expect(statuses.at(-1)).toContain("did not start");
  expect(helpers.length).toBe(1);
  const retry = ask($, "on");
  await clock.settle();
  expect(helpers.length).toBe(2);
  await clock.advance(3000);
  await retry;
});

test("a helper is never spawned from a turn, which Claude can interrupt", async ($, on) => {
  const { sent, helpers, statuses } = setup(on);
  // No session.start has run, so nothing owns helpers beyond a single dispatch.
  await drain($.turn.step(input));
  expect(helpers.length).toBe(0);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("Switchyard unavailable");
});


test("resume restores controls and reuses the same Runner", async ($, on) => {
  const { helpers, setSession } = setup(on);
  await step($);
  await ask($, "off");
  await $.session.end({ reason: "clear", sessionId: "session-a", resume: {} });
  setSession("session-b");
  await step($);
  await $.session.end({ reason: "resume", sessionId: "session-b", resume: {} });
  setSession("session-a");
  expect((await ask($, "status")).text).toBe("Switchyard off");
  await step($);
  expect(helpers[0].requests.length).toBe(2);
  await ask($, "on");
  await step($);
  expect(helpers.length).toBe(1);
  expect(helpers[0].requests.map((r) => r.session_id)).toEqual(["session-a", "session-b", "session-a"]);
});

test("reload restores off without starting a helper", async ($, on) => {
  const { helpers, sent } = setup(on, { store: { "switchyard.enabled:session-a": false } });
  await step($);
  expect(helpers.length).toBe(0);
  expect(sent).toEqual([input]);
  expect((await ask($, "status")).text).toBe("Switchyard off");
});

test("a session change during initialization cannot submit another session's history", async ($, on) => {
  const { clock, helpers, gate, setSession } = setup(on);
  await boot($);
  setSession("session-b");
  const releaseOld = gate("model");
  const old = step($);
  await clock.settle();
  setSession("session-c");
  const releaseNew = gate("model");
  releaseNew();
  await ask($, "off");
  releaseOld();
  await old;
  expect(helpers.length).toBe(0);
  expect((await ask($, "status")).text).toBe("Switchyard off");
});

test("a late model command cannot disable a different session", async ($, on) => {
  const { clock, gate, setSession } = setup(on);
  await boot($);
  const release = gate("modelCommand");
  const old = $.command.run({ command: "model", args: "sonnet" });
  await clock.settle();
  setSession("session-b");
  await ask($, "status");
  release();
  await old;
  expect((await ask($, "status")).text).toBe("Switchyard · waiting for a request");
});

test("a delayed on preference cannot overwrite off or start a helper afterward", async ($, on) => {
  const { clock, helpers, stored } = setup(on, { storeSet: async (e, clock) => {
    if (e.value) await clock.sleep(100);
  } });
  await ask($, "off");
  const enabling = ask($, "on");
  await clock.settle();
  const disabling = ask($, "off");
  await clock.settle();
  await clock.advance(100);
  await Promise.all([enabling, disabling]);
  expect(helpers.length).toBe(0);
  expect(stored.get("switchyard.enabled:session-a")).toBe(false);
});

test("the decision timeout also bounds writing the request", async ($, on) => {
  const { clock, sent, helpers, gate } = setup(on);
  const release = gate("write");
  const pending = step($);
  await clock.settle();
  await clock.advance(15000);
  const timedOut = sent.length === 1;
  release();
  await clock.settle();
  await pending;
  expect(timedOut).toBe(true);
  expect(sent).toEqual([input]);
  expect(helpers[0].alive).toBe(false);
});

test("concurrent submissions publish request files in decision order", async ($, on) => {
  const { clock, helpers } = setup(on, { write: async (e, clock) => {
    if (e.path.endsWith("req-1.json")) await clock.sleep(100);
  } });
  const first = step($);
  await clock.settle();
  const second = step($, { ...input, index: 1 });
  await clock.settle();
  await clock.advance(100);
  await Promise.all([first, second]);
  expect(helpers.length).toBe(1);
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2]);
});

test("a session switch while reading history cannot update the previous session's Runner state", async ($, on) => {
  const { clock, helpers, gate, setSession } = setup(on);
  const release = gate("messages");
  const pending = step($);
  await clock.settle();
  setSession("session-b");
  release();
  await pending;
  expect(helpers.length).toBe(0);
});


test("native classifier uses low-effort Sonnet and reuses a user-turn verdict", async ($, on) => {
  const { classifierCalls, helpers } = setup(on, { classifier: true });
  await step($);
  await step($); // Retry of the same user turn.
  expect(classifierCalls.length).toBe(1);
  expect(classifierCalls[0]).toMatchObject({ model: "claude-sonnet-5-5", effort: "low", maxTokens: 384, timeoutMs: 10000 });
  await step($, { ...input, turnId: "next-user-turn" });
  expect(classifierCalls.length).toBe(2);
  expect(helpers.length).toBe(1);
});

test("classifier failure is returned to Runner and cached for that turn", async ($, on) => {
  const { classifierCalls, helpers } = setup(on, { classifier: true, classifierResult: { isAnswered: false, reason: "aborted", usage: null } });
  await step($);
  await step($);
  expect(classifierCalls.length).toBe(1);
  expect(helpers[0].replies).toEqual([{ error: "aborted" }, { error: "aborted" }]);
  expect(helpers[0].alive).toBe(true);
});

test("off during classification discards the verdict and keeps the Runner", async ($, on) => {
  const { helpers, sent, gate, clock } = setup(on, { classifier: true });
  const release = gate("classifier");
  const pending = step($);
  await clock.settle();
  await ask($, "off");
  release();
  await pending;
  expect(sent).toEqual([input]);
  expect(helpers[0].replies ?? []).toEqual([]);
  expect(helpers[0].alive).toBe(true);
  await ask($, "on");
  await step($, { ...input, turnId: "new-turn" });
  expect(helpers.length).toBe(1);
});

test("classifier verdicts are isolated by session", async ($, on) => {
  const { classifierCalls, helpers, setSession } = setup(on, { classifier: true });
  await step($);
  setSession("session-b");
  await step($);
  expect(classifierCalls.length).toBe(2);
  expect(helpers.length).toBe(1);
});

test("classifier failure is visible alongside the selected stage model", async ($, on) => {
  const { statuses } = setup(on, { answer: (h, id) => ({ id, ...selection(), evidence: { source: "fail_open" } }) });
  await step($);
  expect(statuses.at(-1)).toContain("classifier unavailable; stage routing");
});

test("a failed classifier reply write reports the error and retains the Runner", async ($, on) => {
  const { helpers, statuses, sent } = setup(on, { classifier: true, write: (e) => {
    if (/\/judge-/.test(e.path)) throw new Error("judge write failed");
  } });
  await step($);
  expect(helpers[0].alive).toBe(true);
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("Switchyard unavailable");
  expect(statuses.at(-1)).toContain("fs.write");
});

test("per-request classifiers do not reuse a composite user-turn verdict", async ($, on) => {
  const { classifierCalls } = setup(on, { answer: (h, id) => ({
    id, call: 0, cache_turn: false, classifier: { model: "claude-sonnet-5-5", prompt: "Classify", maxTokens: 384 },
  }) });
  await step($);
  await step($);
  expect(classifierCalls.length).toBe(2);
});

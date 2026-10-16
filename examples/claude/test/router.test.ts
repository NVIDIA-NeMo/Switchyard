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
  const classifierCalls = [];
  const clock = mock.clock(on);
  const stored = new Map();
  on("store.get", ($, e) => ({ value: stored.get(e.key) }));
  on("store.set", ($, e) => {
    stored.set(e.key, e.value);
    return { value: null };
  });
  mock.env(on, { SWITCHYARD_CONFIG: "/tmp/switchyard-test.toml" });
  on("ui.status", ($, e) => { statuses.push(e.text); return { value: null }; });
  let nativeModel = input.model;
  let session = "session-a";
  on("session.model", () => ({ value: nativeModel }));
  on("session.id", () => ({ value: session }));
  on("session.messages", () => ({ value: options.messages ?? [prompt] }));
  on("model.complete", async ($, e) => {
    classifierCalls.push(e);
    await options.classifierGate;
    return { value: options.classifierResult ?? { isAnswered: true, text: '{"decision":{"target":"efficient"}}', usage: null } };
  });
  on("process.spawn", async function* ($, e) {
    const helper = { argv: e.argv, dir: `/tmp/switchyard-claude-${helpers.length}`, requests: [], alive: true, queue: [], wake: undefined };
    helper.emit = (line) => { helper.queue.push(line); helper.wake?.(); };
    helper.exit = () => { helper.alive = false; helper.wake?.(); };
    helpers.push(helper);
    try {
      yield { stream: "stdout", text: `${JSON.stringify({ dir: helper.dir })}\n` };
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
    if (answer !== "silent") helper.emit({ stream: "stdout", text: `${JSON.stringify(answer)}\n` });
    return { value: undefined };
  });
  // The mod removes the helper's directory, which also ends a helper that outlived its kill.
  on("process.run", ($, e) => {
    helpers.find((h) => h.dir === e.argv.at(-1))?.exit();
    return { value: { exitCode: 0, stdout: "", stderr: "", isStdoutTruncated: false, isStderrTruncated: false } };
  });
  on("turn.step", async function* ($, e) {
    sent.push(e);
    yield { kind: "text", index: 0, text: "hello" };
    return { turnId: e.turnId, index: e.index, answer: "hello", toolUses: [], stopReason: "end_turn", usage: null };
  });
  on("command.run", { command: "model" }, () => ({ text: "model selected" }));
  on("session.start", ($, e) => ({ cwd: e.cwd }));
  on("session.end", ($, e) => ({ sessionId: e.sessionId }));
  on("command.register", ($, e) => ({ value: { command: e.name } }));
  return {
    sent, clock, statuses, helpers, classifierCalls,
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

test("controls pause routing without replacing the Runner; exit stops it", async ($, on) => {
  const { sent, helpers, setModel } = setup(on);
  await step($);
  await ask($, "off");
  await step($);
  expect(sent.at(-1)).toEqual(input);
  expect(helpers[0].requests.length).toBe(1);
  await ask($, "on");
  await step($);
  expect(sent.at(-1).model).toBe("claude-opus-5-5");
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2]);
  await $.command.run({ command: "model", args: "sonnet" });
  await step($);
  expect(sent.at(-1)).toEqual(input);
  await ask($, "on");
  setModel("claude-haiku-4-5");
  const manual = { ...input, model: "claude-haiku-4-5" };
  await step($, manual);
  expect(sent.at(-1)).toEqual(manual);
  expect(helpers.length).toBe(1);
  expect(helpers[0].alive).toBe(true);
  await $.session.end({ reason: "other", sessionId: "session-a", resume: {} });
  expect(helpers[0].alive).toBe(false);
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

test("a timed-out decision keeps Claude's model and replaces the helper", async ($, on) => {
  const { sent, clock, helpers, statuses } = setup(on, {
    answer: (h) => h.dir.endsWith("-0") ? "silent" : undefined,
  });
  const pending = step($);
  await clock.settle();
  await clock.advance(15000);
  await pending;
  expect(sent).toEqual([input]);
  expect(statuses.at(-1)).toContain("timed out");
  expect(helpers[0].alive).toBe(false);
  await step($);
  expect(helpers.length).toBe(2);
  expect(sent.at(-1).model).toBe("claude-opus-5-5");
});

test("subagents pass through and helpers require a session controller", async ($, on) => {
  const { sent, helpers } = setup(on);
  await drain($.turn.step(input));
  const agent = { ...input, agentId: "worker" };
  await step($, agent);
  expect(sent).toEqual([input, agent]);
  expect(helpers.length).toBe(0);
});

for (const [name, result, reply] of [
  ["verdict", { isAnswered: true, text: '{"decision":{"target":"efficient"}}', usage: null }, { text: '{"decision":{"target":"efficient"}}' }],
  ["failure", { isAnswered: false, reason: "aborted", usage: null }, { error: "aborted" }],
]) {
  test(`classifier ${name} is cached per user turn and isolated by session`, async ($, on) => {
    const { classifierCalls, helpers, setSession } = setup(on, { classifier: true, classifierResult: result });
    await step($);
    await step($);
    expect(classifierCalls.length).toBe(1);
    expect(classifierCalls[0]).toMatchObject({ model: "claude-sonnet-5-5", effort: "low", maxTokens: 384, timeoutMs: 10000 });
    expect(helpers[0].replies).toEqual([reply, reply]);
    await step($, { ...input, turnId: "next-user-turn" });
    expect(classifierCalls.length).toBe(2);
    setSession("session-b");
    await step($, { ...input, turnId: "next-user-turn" });
    expect(classifierCalls.length).toBe(3);
    expect(helpers.length).toBe(1);
    expect(helpers[0].alive).toBe(true);
  });
}

test("off during classification discards the verdict and keeps the Runner", async ($, on) => {
  let release;
  const classifierGate = new Promise((resolve) => { release = resolve; });
  const { helpers, sent, clock } = setup(on, { classifier: true, classifierGate });
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

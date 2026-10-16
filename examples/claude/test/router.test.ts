// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, mock, test } from "claude-code/testing";

const input = { turnId: "turn", index: 0, model: "claude-sonnet-4-6", effort: "high", messageCount: 1 };
const prompt = { role: "user", text: "Fix the tests", toolUses: [] };
const selection = (model = "claude-opus-5-5") => ({ target: "capable", model });

// Stands in for the native helper: same protocol, none of the routing.
function setup(on, classifier = false) {
  const sent = [];
  const helpers = [];
  const classifierCalls = [];
  mock.clock(on);
  const stored = new Map();
  on("store.get", ($, e) => ({ value: stored.get(e.key) }));
  on("store.set", ($, e) => {
    stored.set(e.key, e.value);
    return { value: null };
  });
  mock.env(on, { SWITCHYARD_CONFIG: "/tmp/switchyard-test.toml" });
  on("ui.status", () => ({ value: null }));
  on("session.model", () => ({ value: input.model }));
  on("session.id", () => ({ value: "session-a" }));
  on("session.messages", () => ({ value: [prompt] }));
  on("model.complete", ($, e) => {
    classifierCalls.push(e);
    return { value: { isAnswered: true, text: '{"decision":{"target":"efficient"}}', usage: null } };
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
  on("fs.write", ($, e) => {
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
    const answer = classifier ? {
      id, call: 0, cache_turn: true, classifier: { model: "claude-sonnet-5-5", prompt: "Classify", system: "Return JSON", maxTokens: 384 },
    } : { id, ...selection() };
    helper.emit({ stream: "stdout", text: `${JSON.stringify(answer)}\n` });
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
  on("session.start", ($, e) => ({ cwd: e.cwd }));
  on("session.end", ($, e) => ({ sessionId: e.sessionId }));
  on("command.register", ($, e) => ({ value: { command: e.name } }));
  return { sent, helpers, classifierCalls };
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

test("off/on reuses the Runner and session exit stops it", async ($, on) => {
  const { sent, helpers } = setup(on);
  await ask($, "on");
  expect(helpers.length).toBe(0);
  await step($);
  await ask($, "off");
  await step($);
  expect(sent.at(-1)).toEqual(input);
  expect(helpers[0].requests.length).toBe(1);
  await ask($, "on");
  await step($);
  expect(sent.at(-1).model).toBe("claude-opus-5-5");
  expect(helpers.length).toBe(1);
  expect(helpers[0].requests.map((r) => r.id)).toEqual([1, 2]);
  await $.session.end({ reason: "other", sessionId: "session-a", resume: {} });
  expect(helpers[0].alive).toBe(false);
});

test("calls Claude's classifier once per user turn", async ($, on) => {
  const { sent, classifierCalls, helpers } = setup(on, true);
  await step($);
  await step($, { ...input, index: 1 });
  expect(classifierCalls.length).toBe(1);
  expect(classifierCalls[0]).toMatchObject({ model: "claude-sonnet-5-5", effort: "low", maxTokens: 384, timeoutMs: 10000 });
  expect(helpers[0].replies).toEqual([
    { text: '{"decision":{"target":"efficient"}}' },
    { text: '{"decision":{"target":"efficient"}}' },
  ]);
  expect(sent.map((e) => e.model)).toEqual(["claude-opus-5-5", "claude-opus-5-5"]);
  await step($, { ...input, turnId: "next-user-turn" });
  expect(classifierCalls.length).toBe(2);
});

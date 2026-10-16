// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, mock, test } from "claude-code/testing";

const input = { turnId: "turn", index: 0, model: "claude-sonnet-4-6", effort: "high", messageCount: 1 };
const prompt = { role: "user", text: "Fix the tests", toolUses: [] };
const selection = (model = "claude-opus-5-5") => ({ target: "capable", model });
const response = (body = selection()) => ({ exitCode: 0, stdout: JSON.stringify(body), stderr: "" });

function helperResponse(e) {
  const body = JSON.parse(e.init.stdin);
  return response({ ...selection(), version: 1, session_id: body.session_id,
    decision_id: body.decision_id, checkpoint: { version: 1, session_id: body.session_id,
      count: (body.checkpoint?.count ?? 0) + 1 } });
}

function setup(on, options = {}) {
  const sent = [];
  const requests = [];
  const statuses = [];
  const clock = mock.clock(on);
  const stored = new Map(Object.entries(options.store ?? {}));
  on("store.get", ($, e) => ({ value: stored.get(e.key) }));
  on("store.keys", () => ({ value: [...stored.keys()] }));
  on("store.delete", ($, e) => { stored.delete(e.key); return { value: null }; });
  on("store.set", ($, e) => {
    if (options.storeFailure) return { deny: "disk full" };
    stored.set(e.key, JSON.parse(JSON.stringify(e.value)));
    return { value: null };
  });
  let sessionId = "session-a";
  let messages = options.messages ?? [prompt];
  on("session.id", () => ({ value: sessionId }));
  mock.env(on, { SWITCHYARD_CONFIG: "/tmp/switchyard-test.toml" });
  on("ui.status", ($, e) => { statuses.push(e.text); return { value: null }; });
  let nativeModel = input.model;
  on("session.model", () => ({ value: nativeModel }));
  on("session.messages", () => ({ value: messages }));
  on("process.run", async ($, e) => {
    requests.push(e);
    if (options.process) return options.process(e, clock);
    return { value: helperResponse(e) };
  });
  on("turn.step", async function* ($, e) {
    sent.push(e);
    yield { kind: "text", index: 0, text: "hello" };
    if (options.completion) await options.completion(e, clock);
    if (options.failCompletion) throw new Error("completion failed");
    return { turnId: e.turnId, index: e.index, answer: "hello", toolUses: [], stopReason: "end_turn", usage: null };
  });
  on("command.run", { command: "model" }, () => ({ text: "model selected" }));
  on("session.start", ($, e) => ({ cwd: e.cwd }));
  on("session.end", ($, e) => ({ sessionId: e.sessionId }));
  on("command.register", ($, e) => ({ value: { command: e.name } }));
  return { sent, requests, clock, statuses, readStore: (key) => stored.get(key), setSession: (id) => { sessionId = id; },
    setMessages: (value) => { messages = value; }, setModel: (model) => { nativeModel = model; } };
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

test("selects a native model, preserves effort, and passes through the stream", async ($, on) => {
  const { sent, requests, readStore } = setup(on);
  const output = await drain($.turn.step(input));
  expect(sent).toEqual([{ ...input, model: "claude-opus-5-5" }]);
  expect(output.chunks).toMatchObject([{ kind: "text", text: "hello" }]);
  expect(output.result.answer).toBe("hello");
  expect(requests[0].argv[0]).toMatch(/\/bin\/switchyard-claude$/);
  expect(requests[0].argv[1]).toBe("/tmp/switchyard-test.toml");
  expect(requests[0].init.timeoutMs).toBe(3000);
  expect(JSON.parse(requests[0].init.stdin).messages).toEqual([
    { role: "user", content: [{ type: "text", text: "Fix the tests" }] },
  ]);
});

test("passes paired tool calls and failures once, including continuation requests", async ($, on) => {
  const { sent, requests, readStore } = setup(on, { messages: [
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
  await drain($.turn.step({ ...input, index: 1, messageCount: 5 }));
  const messages = JSON.parse(requests[0].init.stdin).messages;
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
  const { sent, requests, readStore } = setup(on);
  const agent = { ...input, agentId: "worker" };
  await drain($.turn.step(agent));
  expect(sent).toEqual([agent]);
  expect(requests.length).toBe(0);
});

test("off and /model keep native selection until routing is enabled again", async ($, on) => {
  const { sent, requests, readStore } = setup(on);
  await $.command.run({ command: "switchyard", args: "off" });
  await drain($.turn.step(input));
  await $.command.run({ command: "switchyard", args: "on" });
  await drain($.turn.step(input));
  await $.command.run({ command: "model", args: "sonnet" });
  await drain($.turn.step(input));
  expect(sent.map((e) => e.model)).toEqual([input.model, "claude-opus-5-5", input.model]);
  expect(requests.length).toBe(1);
  expect((await $.command.run({ command: "switchyard", args: "status" })).text).toContain("Switchyard off");
});

for (const [name, process] of [
  ["helper error", () => ({ value: { exitCode: 1, stdout: "", stderr: "Invalid config" } })],
  ["process refusal", () => ({ deny: "unavailable" })],
  ["deadline", () => ({ deny: "process timed out" })],
  ["malformed JSON", () => ({ value: { ...response(), stdout: "{" } })],
  ["missing target", () => ({ value: response({}) })],
  ["unresolved alias", () => ({ value: response(selection("sonnet")) })],
  ["non-Claude target", () => ({ value: response(selection("gpt-test")) })],
]) {
  test(`keeps Claude's model on ${name}`, async ($, on) => {
    const { sent, statuses, readStore } = setup(on, { process });
    await drain($.turn.step(input));
    expect(sent).toEqual([input]);
    expect(statuses.at(-1)).toContain("Switchyard unavailable");
    expect((await $.command.run({ command: "switchyard", args: "status" })).text).toContain("keeping");
  });
}

test("empty context passes through once", async ($, on) => {
  const { sent, requests, readStore } = setup(on, { messages: [] });
  await drain($.turn.step(input));
  expect(sent).toEqual([input]);
  expect(requests.length).toBe(0);
});

test("completion errors do not retry or substitute another model", async ($, on) => {
  const { sent, readStore } = setup(on, { failCompletion: true });
  let failure = "";
  try { await drain($.turn.step(input)); } catch (error) { failure = error.message; }
  // The kit skips a throwing engine stub and reports a missing implementation.
  expect(failure).toContain("turn.step");
  expect(sent.length).toBe(1);
});

test("a manual override during a decision discards its answer", async ($, on) => {
  const { sent, clock, readStore } = setup(on, { process: async (e, clock) => {
    await clock.sleep(100);
    return { value: helperResponse(e) };
  } });
  const step = drain($.turn.step(input));
  await clock.settle();
  await $.command.run({ command: "switchyard", args: "off" });
  await clock.advance(100);
  await step;
  expect(sent).toEqual([input]);
  expect((await $.command.run({ command: "switchyard", args: "status" })).text).toBe("Switchyard off");
});

test("a native model change between requests disables routing", async ($, on) => {
  const { sent, requests, statuses, setModel, readStore } = setup(on);
  await drain($.turn.step(input));
  setModel("claude-haiku-4-5");
  const changed = { ...input, model: "claude-haiku-4-5" };
  await drain($.turn.step(changed));
  expect(sent[1]).toEqual(changed);
  expect(requests.length).toBe(1);
  expect(statuses.at(-1)).toContain("Switchyard off");
});

test("opening and dismissing /model does not disable routing", async ($, on) => {
  const { sent, statuses, readStore } = setup(on);
  await $.command.run({ command: "model", args: "" });
  await drain($.turn.step(input));
  expect(sent[0].model).toBe("claude-opus-5-5");
  expect(statuses.at(-1)).toContain("claude-opus-5-5");
});

test("startup exposes enabled status and registers controls", async ($, on) => {
  const { statuses, readStore } = setup(on);
  await $.session.start({ cwd: "/tmp", surface: "terminal", isInteractive: true });
  expect(statuses.at(-1)).toBe("Switchyard Auto · waiting for a request");
  expect((await $.command.run({ command: "switchyard", args: "" })).text).toBe(statuses.at(-1));
});

const sessionKey = (id = "session-a") => `switchyard.session.v1:${id}`;
const savedState = (enabled = true, count = 4) => ({
  version: 1, enabled, nativeModel: input.model, checkpoint: { version: 1, session_id: "session-a", count },
  status: enabled ? "Switchyard Auto · claude-opus-5-5" : "Switchyard off", updatedAt: 0,
});

test("completed requests carry state forward and off/on preserves it", async ($, on) => {
  const { requests, readStore } = setup(on);
  await drain($.turn.step(input));
  expect((readStore(sessionKey())).checkpoint.count).toBe(1);
  await $.command.run({ command: "switchyard", args: "off" });
  await drain($.turn.step({ ...input, index: 1 }));
  await $.command.run({ command: "switchyard", args: "on" });
  await drain($.turn.step({ ...input, index: 2 }));
  expect(requests.length).toBe(2);
  expect(JSON.parse(requests[1].init.stdin).checkpoint.count).toBe(1);
  expect((readStore(sessionKey())).checkpoint.count).toBe(2);
});

test("a new module restores the saved session checkpoint", async ($, on) => {
  const { requests, readStore } = setup(on, { store: { [sessionKey()]: savedState() } });
  await $.session.start({ cwd: "/tmp", surface: "terminal", isInteractive: true });
  await drain($.turn.step(input));
  expect(JSON.parse(requests[0].init.stdin).checkpoint.count).toBe(4);
  expect((readStore(sessionKey())).checkpoint.count).toBe(5);
});

test("clear starts enabled and resume restores that session's controls and checkpoint", async ($, on) => {
  const { requests, setSession, readStore } = setup(on);
  await drain($.turn.step(input));
  await $.command.run({ command: "switchyard", args: "off" });
  await $.session.end({ reason: "clear", sessionId: "session-a", resume: {} });
  setSession("session-b");
  await drain($.turn.step({ ...input, turnId: "new-turn" }));
  expect(JSON.parse(requests[1].init.stdin).checkpoint).toBe(null);
  expect(JSON.parse(requests[1].init.stdin).session_id).toBe("session-b");
  await $.session.end({ reason: "resume", sessionId: "session-b", resume: {} });
  setSession("session-a");
  await drain($.turn.step(input));
  expect(requests.length).toBe(2);
  expect((await $.command.run({ command: "switchyard", args: "status" })).text).toBe("Switchyard off");
  await $.command.run({ command: "switchyard", args: "on" });
  await drain($.turn.step({ ...input, index: 1 }));
  expect(JSON.parse(requests[2].init.stdin).checkpoint.count).toBe(1);
});

test("compaction rereads current messages and retains routing state", async ($, on) => {
  const { requests, setMessages, readStore } = setup(on);
  await drain($.turn.step(input));
  setMessages([{ role: "user", text: "Summary of the compacted conversation" }]);
  await drain($.turn.step({ ...input, index: 1 }));
  const body = JSON.parse(requests[1].init.stdin);
  expect(body.checkpoint.count).toBe(1);
  expect(body.messages[0].content[0].text).toBe("Summary of the compacted conversation");
});

test("abandoning a completion stream leaves the previous checkpoint untouched", async ($, on) => {
  const { readStore } = setup(on, { store: { [sessionKey()]: savedState() } });
  const stream = $.turn.step(input);
  await stream.next();
  await stream.return();
  expect((readStore(sessionKey())).checkpoint.count).toBe(4);
});

test("a completion error leaves the checkpoint untouched", async ($, on) => {
  const { readStore } = setup(on, { failCompletion: true, store: { [sessionKey()]: savedState() } });
  try { await drain($.turn.step(input)); } catch (_) {}
  expect((readStore(sessionKey())).checkpoint.count).toBe(4);
});

test("off during streaming invalidates a pending state commit", async ($, on) => {
  const { clock, readStore } = setup(on, { completion: async (e, clock) => { await clock.sleep(100); } });
  const step = drain($.turn.step(input));
  await clock.settle();
  await $.command.run({ command: "switchyard", args: "off" });
  await clock.advance(100);
  await step;
  const stored = readStore(sessionKey());
  expect(stored.enabled).toBe(false);
  expect(stored.checkpoint).toBe(null);
});

test("a session change discards an old helper decision", async ($, on) => {
  const { clock, sent, setSession, readStore } = setup(on, { process: async (e, clock) => {
    await clock.sleep(100);
    return { value: helperResponse(e) };
  } });
  const step = drain($.turn.step(input));
  await clock.settle();
  await $.session.end({ reason: "clear", sessionId: "session-a", resume: {} });
  setSession("session-b");
  await clock.advance(100);
  await step;
  expect(sent[0]).toEqual(input);
  expect(readStore(sessionKey())).toBe(undefined);
  expect(readStore(sessionKey("session-b"))).toBe(undefined);
});

test("subagent steps do not consume the parent checkpoint", async ($, on) => {
  const { requests, readStore } = setup(on, { store: { [sessionKey()]: savedState() } });
  await drain($.turn.step({ ...input, agentId: "child" }));
  expect(requests.length).toBe(0);
  expect((readStore(sessionKey())).checkpoint.count).toBe(4);
});

test("reset recovers corrupt storage without losing Claude's request", async ($, on) => {
  const { sent, requests, readStore } = setup(on, { store: { [sessionKey()]: { version: 99 } } });
  await drain($.turn.step(input));
  expect(sent[0]).toEqual(input);
  expect(requests.length).toBe(0);
  await $.command.run({ command: "switchyard", args: "reset" });
  await drain($.turn.step({ ...input, index: 1 }));
  expect(JSON.parse(requests[0].init.stdin).checkpoint).toBe(null);
});

test("storage failure does not replay or fail a completed model response", async ($, on) => {
  const { sent, statuses, readStore } = setup(on, { storeFailure: true });

  const result = await drain($.turn.step(input));
  expect(result.result.answer).toBe("hello");
  expect(sent.length).toBe(1);
  expect(statuses.at(-1)).toContain("state not saved");
});

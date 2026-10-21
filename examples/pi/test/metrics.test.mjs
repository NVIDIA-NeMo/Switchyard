// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { test } from "node:test";
import { Metrics } from "@switchyard/libsy";
import { registerMetrics } from "../metrics.ts";

function usage(input, output, cacheRead = 0, cacheWrite = 0, reasoning) {
  return { input, output, cacheRead, cacheWrite, reasoning, totalTokens: input + output + cacheRead + cacheWrite,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
}

function setup(t) {
  let sessionId = randomUUID(), now = 0;
  t.mock.method(performance, "now", () => now);
  const handlers = new Map(), sessions = new Map([[sessionId, []]]);
  const ctx = {
    hasUI: true,
    sessionManager: { getSessionId: () => sessionId, getEntries: () => sessions.get(sessionId) },
    ui: { setStatus: t.mock.fn() },
  };
  const pi = {
    on: (name, handler) => handlers.set(name, handler),
    appendEntry: (customType, data) => sessions.get(sessionId).push({ type: "custom", customType, data }),
  };
  let measurements = registerMetrics(pi);
  const emit = (type, data = {}) => handlers.get(type)?.({ type, ...data }, ctx);
  return {
    ctx, emit, sessions,
    begin: (reason = "user") => measurements.begin(reason, ctx),
    time: (value) => { now = value; },
    status: () => ctx.ui.setStatus.mock.calls.at(-1).arguments[1],
    entries: () => sessions.get(sessionId),
    reload: () => { measurements = registerMetrics(pi); emit("session_start"); },
    switchSession: (id = randomUUID(), entries = []) => {
      sessionId = id;
      if (!sessions.has(id)) sessions.set(id, entries);
      emit("session_start");
      return id;
    },
  };
}

const assistant = (reportedUsage, stopReason = "stop") => ({
  role: "assistant", provider: "provider", model: "answer", usage: reportedUsage, stopReason,
});
const update = (fixture, type, data = {}) => fixture.emit("message_update", {
  message: { role: "assistant" }, assistantMessageEvent: { type, ...data },
});

// This exercises the native collector; the fixture replaces only Pi and the clock.
test("status shows session totals and last request including classifier usage", async (t) => {
  const fixture = setup(t);
  const first = fixture.begin();
  assert.deepEqual(first.context, { session_id: fixture.ctx.sessionManager.getSessionId(), agent_id: "root",
    agent_turn_idx: 0, user_turn_idx: 0 });
  await first.classify("provider/judge", async () => {
    fixture.time(20);
    return { stopReason: "stop", usage: usage(5, 2, 3) };
  });
  fixture.time(30);
  update(fixture, "thinking_start");
  update(fixture, "thinking_delta", { delta: "" });
  fixture.time(50);
  update(fixture, "thinking_delta", { delta: "thinking" });
  fixture.time(70);
  update(fixture, "text_delta", { delta: "answer" });
  fixture.time(100);
  fixture.emit("message_end", { message: assistant(usage(100, 20, 30, 0, 5)) });
  assert.equal(fixture.status(), "SY session 160 tok TTFT avg 50ms req 100ms | last 160 tok TTFT 50ms req 100ms");
  assert.ok(fixture.status().length <= 80);
  const records = fixture.entries().map((entry) => entry.data);
  assert.equal(records.length, 2);
  assert.equal(records[1].usage.reasoning_tokens, 5);
  assert.equal(records[1].usage.output_tokens, 20);
  const native = new Metrics();
  const snapshot = native.stats(first.context);
  const counter = (name) => snapshot.series.filter((series) => series.name === name)
    .reduce((sum, series) => sum + series.value.value, 0);
  assert.equal(counter("switchyard_llm_call_prompt_tokens_total"), 138);
  assert.equal(counter("switchyard_llm_call_completion_tokens_total"), 22);

  // A tool-only continuation has its own request index, under the same user turn.
  fixture.time(100);
  const continued = fixture.begin("continuation");
  assert.equal(continued.context.agent_turn_idx, 1);
  assert.equal(continued.context.user_turn_idx, 0);
  fixture.time(140);
  update(fixture, "toolcall_start", { contentIndex: 0,
    partial: { content: [{ type: "toolCall", id: "read", name: "read", arguments: {} }] } });
  fixture.time(160);
  fixture.emit("message_end", { message: assistant(usage(5, 2), "toolUse") });
  assert.equal(fixture.status(), "SY session 167 tok TTFT avg 45ms req 160ms | last 7 tok TTFT 40ms req 60ms");
  // Tool execution time is outside the completed request's latency.
  fixture.time(1160);
  fixture.emit("message_end", { message: { role: "toolResult" } });
  assert.match(fixture.status(), /req 160ms \| last 7 tok TTFT 40ms req 60ms$/);

  const savedStatus = fixture.status();
  fixture.reload();
  assert.equal(fixture.status(), savedStatus);
  assert.equal(fixture.entries().length, 3);
  fixture.reload();
  assert.equal(fixture.status(), savedStatus);
  assert.equal(fixture.begin().context.user_turn_idx, 1);
});

test("failures retain reported tokens and leave missing TTFT unknown", async (t) => {
  const fixture = setup(t);
  const request = fixture.begin();
  fixture.time(10);
  const result = await request.classify("provider/judge", async () => ({ stopReason: "error", usage: usage(5, 0) }));
  assert.equal(result.stopReason, "error");
  request.fail();
  assert.equal(fixture.status(), "SY session 5 tok TTFT avg — req 10ms | last 5 tok TTFT — req 10ms");
  fixture.emit("message_end", { message: assistant(usage(0, 0), "error") });
  assert.equal(fixture.entries().length, 2);
  assert.equal(fixture.entries()[1].data.is_success, false);

  fixture.time(20);
  const retry = fixture.begin("retry");
  assert.equal(retry.context.agent_turn_idx, 1);
  assert.equal(retry.context.user_turn_idx, 0);
  fixture.time(30);
  fixture.emit("message_end", { message: assistant(usage(8, 2), "aborted") });
  assert.equal(fixture.status(), "SY session 15 tok TTFT avg — req 20ms | last 10 tok TTFT — req 10ms");
  assert.equal(fixture.entries().at(-1).data.is_success, false);
});

test("new and forked sessions start fresh, and resume restores once", (t) => {
  const fixture = setup(t);
  const first = fixture.begin();
  fixture.time(25);
  fixture.emit("message_end", { message: assistant(usage(10, 2)) });
  const saved = fixture.status(), parentRecords = fixture.entries();
  const child = fixture.switchSession(undefined, structuredClone(parentRecords));
  assert.equal(fixture.status(), "SY session — tok TTFT avg — req — | last — tok TTFT — req —");
  assert.equal(fixture.begin().context.agent_turn_idx, 0);
  fixture.switchSession(first.context.session_id);
  assert.equal(fixture.status(), saved);
  assert.equal(fixture.begin().context.agent_turn_idx, 1);
  assert.equal(fixture.sessions.get(child).length, 1);
});

test("headless use persists metrics and direct helper calls leave the active request intact", (t) => {
  const fixture = setup(t);
  fixture.ctx.hasUI = false;
  const first = fixture.begin();
  assert.equal(fixture.begin("direct"), undefined);
  fixture.time(10);
  fixture.emit("message_end", { message: assistant(undefined) });
  assert.equal(fixture.ctx.ui.setStatus.mock.callCount(), 0);
  assert.equal(fixture.entries().length, 1);
  const snapshot = new Metrics().stats(first.context);
  assert.ok(!snapshot.series.some((series) => series.name === "switchyard_llm_call_total_tokens_total"));
  assert.ok(snapshot.series.some((series) => series.name === "switchyard_inference_duration_ms"));
});

test("task predictions survive reload and resume, reset on forks, and reject late session updates", (t) => {
  const fixture = setup(t);
  const active = fixture.begin();
  const parent = active.context.session_id;
  const forecast = { predicted_tokens: 3000, expected_upper_tokens: 4000, selected_interval: [2000, 4000],
    selected_choice: "b002", finite_budget_probability: 0.8, impossible_probability: 0.2 };
  active.prediction(forecast, 3000);
  assert.match(fixture.status(), /^SY remaining ~1\.0k tok ETA [^|]+ \|/);
  fixture.time(10);
  fixture.emit("message_end", { message: assistant(usage(10, 2)) });
  const saved = fixture.status();
  assert.match(saved, /^SY remaining ~988 tok ETA [^|]+ \|/);
  fixture.emit("turn_end");
  fixture.emit("agent_end");
  assert.equal(fixture.status(), saved);
  fixture.reload();
  assert.equal(fixture.status(), saved);
  const copy = structuredClone(fixture.entries());
  const pending = fixture.begin("continuation");
  fixture.switchSession(undefined, copy);
  pending.prediction(null);
  assert.ok(!fixture.status().includes("remaining"));
  assert.equal(fixture.entries().length, copy.length);
  fixture.switchSession(parent);
  assert.equal(fixture.status(), saved);
  fixture.begin("continuation").prediction(null);
  assert.match(fixture.status(), /^SY remaining unavailable/);
  fixture.reload();
  assert.match(fixture.status(), /^SY remaining unavailable/);
  fixture.emit("agent_settled");
  assert.ok(!fixture.status().includes("remaining"));
  fixture.reload();
  assert.ok(!fixture.status().includes("remaining"));
  const pendingAfterClear = fixture.begin("user");
  pendingAfterClear.prediction(forecast, 3900);
  assert.match(fixture.status(), /^SY remaining ~100 tok ETA [^|]+ \|/);
  fixture.emit("agent_settled");
  pendingAfterClear.prediction(forecast, 0);
  assert.ok(!fixture.status().includes("remaining"));
});

test("remaining usage counts cache writes and output once, excludes cache reads, and floors at zero", (t) => {
  const fixture = setup(t);
  const forecast = { predicted_tokens: 3000, expected_upper_tokens: 4000, selected_interval: [2000, 4000],
    selected_choice: "b002", finite_budget_probability: 0.8, impossible_probability: 0.2 };
  fixture.begin().prediction(forecast, 3000);
  fixture.emit("message_end", { message: assistant(usage(100, 20, 9000, 30, 10), "toolUse") });
  assert.match(fixture.status(), /^SY remaining ~850 tok ETA [^|]+ \|/);
  assert.ok(!fixture.status().includes("impossible"));
  fixture.reload();
  assert.match(fixture.status(), /^SY remaining ~850 tok ETA [^|]+ \|/);
  fixture.begin("continuation").prediction(forecast, 4500);
  assert.match(fixture.status(), /^SY remaining ~0 tok ETA [^|]+ \|/);
});

test("ETA uses session timing, survives reload, resets on forks, and clears when the user turn settles", (t) => {
  const fixture = setup(t);
  const parent = fixture.ctx.sessionManager.getSessionId();
  const forecast = (total) => ({ predicted_tokens: total - 100, expected_upper_tokens: total,
    selected_interval: [0, total], selected_choice: "b002", finite_budget_probability: 1, impossible_probability: 0 });
  fixture.begin().prediction(forecast(2500), 0);
  assert.match(fixture.status(), /^SY remaining ~2\.5k tok ETA — \|/);
  fixture.time(2000);
  update(fixture, "text_delta", { delta: "checking parser" });
  fixture.time(6000);
  fixture.emit("message_end", { message: assistant(usage(700, 200, 10000, 100, 50), "toolUse") });
  assert.match(fixture.status(), /^SY remaining ~1\.5k tok ETA ~9s \|/);
  const saved = fixture.status();
  fixture.time(16000);
  fixture.emit("message_end", { message: { role: "toolResult" } });
  assert.equal(fixture.status(), saved);
  fixture.reload();
  assert.equal(fixture.status(), saved);
  fixture.reload();
  assert.equal(fixture.status(), saved);
  const entries = structuredClone(fixture.entries());
  fixture.switchSession(undefined, entries);
  fixture.begin().prediction(forecast(2500), 1000);
  assert.match(fixture.status(), /^SY remaining ~1\.5k tok ETA — \|/);
  fixture.switchSession(parent);
  assert.equal(fixture.status(), saved);
  fixture.begin("continuation").prediction(forecast(4500), 1000);
  assert.match(fixture.status(), /^SY remaining ~3\.5k tok ETA ~21s \|/);
  fixture.time(20000);
  update(fixture, "thinking_delta", { delta: "checking tests" });
  fixture.time(28000);
  fixture.emit("message_end", { message: assistant(usage(1600, 400, 10000, 0, 100)) });
  assert.match(fixture.status(), /^SY remaining ~1\.5k tok ETA ~9s \|/);
  fixture.emit("agent_settled");
  assert.ok(!fixture.status().includes("ETA"));
  fixture.reload();
  assert.ok(!fixture.status().includes("ETA"));
  fixture.begin().prediction(forecast(3000), 3000);
  assert.match(fixture.status(), /^SY remaining ~0 tok ETA ~0s \|/);
});

test("ETA includes sequential, overlapping and nested tool wall time and restores it once", (t) => {
  const fixture = setup(t);
  const parent = fixture.ctx.sessionManager.getSessionId();
  const forecast = (total) => ({ predicted_tokens: total - 100, expected_upper_tokens: total,
    selected_interval: [0, total], selected_choice: "b002", finite_budget_probability: 1, impossible_probability: 0 });
  const start = (id, time, parentToolCallId) => {
    fixture.time(time);
    fixture.emit("tool_execution_start", { toolCallId: id, toolName: "bash", parentToolCallId });
  };
  const end = (id, time, parentToolCallId, isError = false) => {
    fixture.time(time);
    fixture.emit("tool_execution_end", { toolCallId: id, toolName: "bash", parentToolCallId, isError });
  };
  fixture.begin().prediction(forecast(2500), 0);
  fixture.time(2000);
  update(fixture, "text_delta", { delta: "running tools" });
  fixture.time(6000);
  fixture.emit("message_end", { message: assistant(usage(700, 200, 0, 100), "toolUse") });
  assert.match(fixture.status(), /ETA ~9s/);
  start("a", 6000);
  start("b", 8000);
  end("a", 10000);
  assert.match(fixture.status(), /ETA ~9s/);
  end("b", 14000, undefined, true);
  assert.match(fixture.status(), /ETA ~21s/);
  start("c", 15000);
  end("c", 17000);
  assert.match(fixture.status(), /ETA ~24s/);
  start("parent", 18000);
  start("child", 19000, "parent");
  end("child", 20000, "parent");
  end("parent", 22000);
  assert.match(fixture.status(), /ETA ~30s/);
  const toolRecords = fixture.entries().filter((entry) => entry.customType === "switchyard.tools.v1");
  assert.deepEqual(toolRecords.map((entry) => entry.data.duration_ms), [8000, 2000, 4000]);
  assert.equal(fixture.entries().find((entry) => entry.customType === "switchyard.metrics.v1").data.duration_ms, 6000);
  const saved = fixture.status();
  end("parent", 23000);
  assert.equal(fixture.status(), saved);
  fixture.reload();
  assert.equal(fixture.status(), saved);
  fixture.reload();
  assert.equal(fixture.status(), saved);
  fixture.switchSession(undefined, structuredClone(fixture.entries()));
  fixture.begin().prediction(forecast(2500), 1000);
  assert.match(fixture.status(), /ETA —/);
  end("parent", 23000);
  fixture.switchSession(parent);
  assert.equal(fixture.status(), saved);
  fixture.begin("continuation").prediction(forecast(3500), 1000);
  fixture.time(25000);
  update(fixture, "text_delta", { delta: "done" });
  fixture.time(29000);
  fixture.emit("message_end", { message: assistant(usage(800, 200)) });
  assert.match(fixture.status(), /ETA ~20s/);
  fixture.emit("agent_settled");
  assert.ok(!fixture.status().includes("ETA"));
  fixture.reload();
  assert.ok(!fixture.status().includes("ETA"));
});

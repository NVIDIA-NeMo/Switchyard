// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import { Metrics, type CallMetrics, type MetricContext, type MetricsSnapshot, type TokenUsage, type TaskPrediction } from "@switchyard/libsy";
import type { ClassifierResult, Usage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { CompletionTiming, completionDuration } from "./completion-time.ts";

const ENTRY_TYPE = "switchyard.metrics.v1";
const STATUS_KEY = "switchyard";
const PREDICTION_ENTRY = "switchyard.prediction.v1";
const TOOL_ENTRY = "switchyard.tools.v1";

export function tokenUsage(usage: Usage | undefined): TokenUsage | undefined {
  if (!usage) return undefined;
  return {
    input_tokens: usage.input, output_tokens: usage.output,
    cached_input_tokens: usage.cacheRead, cache_creation_input_tokens: usage.cacheWrite,
    total_tokens: usage.totalTokens, reasoning_tokens: usage.reasoning,
  };
}

function total(snapshot: MetricsSnapshot, name: string): number | undefined {
  const values = snapshot.series.flatMap((series) => series.name === name && series.value.type === "counter" ? [series.value.value] : []);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : undefined;
}

function timing(snapshot: MetricsSnapshot, name: string, average = false): number | undefined {
  let sum = 0, count = 0;
  for (const series of snapshot.series) {
    if (series.name === name && series.value.type === "histogram") {
      sum += series.value.sum;
      count += series.value.count;
    }
  }
  return count ? sum / (average ? count : 1) : undefined;
}

function tokens(value: number | undefined): string {
  if (value === undefined) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return String(value);
}

function duration(value: number | undefined): string {
  if (value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`;
}

interface Session {
  nextAgentTurn: number;
  userTurn: number;
  lastTurn?: number;
  prediction?: TaskPrediction | null;
  tokensUsed: number;
  completion: CompletionTiming;
}

interface Active {
  context: MetricContext;
  started: number;
  firstToken?: number;
}

interface ToolWindow {
  context: MetricContext;
  running: Set<string>;
  started: number;
}

/** Feed Pi-owned calls into Switchyard and read status data from its metric snapshots. */
export function registerMetrics(pi: ExtensionAPI) {
  let collector: Metrics | undefined;
  const metrics = () => collector ??= new Metrics();
  const sessions = new Map<string, Session>();
  let active: Active | undefined;
  let tools: ToolWindow | undefined;

  function session(ctx: ExtensionContext): Session {
    const id = ctx.sessionManager.getSessionId();
    let state = sessions.get(id);
    if (state) return state;
    state = { nextAgentTurn: 0, userTurn: -1, tokensUsed: 0, completion: new CompletionTiming() };
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type === "custom" && entry.customType === TOOL_ENTRY) {
        const saved = entry.data as { context: MetricContext; duration_ms: number };
        if (saved.context.session_id === id) state.completion.recordTools(saved.context, saved.duration_ms);
        continue;
      }
      if (entry.type === "custom" && entry.customType === PREDICTION_ENTRY) {
        const saved = entry.data as { session_id: string; prediction: TaskPrediction | null; tokens_used_so_far?: number; cleared?: boolean };
        if (saved.session_id === id) {
          state.prediction = saved.cleared ? undefined : saved.prediction;
          state.tokensUsed = saved.tokens_used_so_far ?? state.tokensUsed;
        }
        continue;
      }
      if (entry.type !== "custom" || entry.customType !== ENTRY_TYPE) continue;
      const record = entry.data as CallMetrics;
      if (record.context.session_id !== id) continue;
      metrics().record(record);
      state.completion.record(record);
      if (record.call_kind === "answer" && record.usage) {
        state.tokensUsed += (record.usage.input_tokens ?? 0) + (record.usage.cache_creation_input_tokens ?? 0) + (record.usage.output_tokens ?? 0);
      }
      state.nextAgentTurn = Math.max(state.nextAgentTurn, record.context.agent_turn_idx + 1);
      state.userTurn = Math.max(state.userTurn, record.context.user_turn_idx);
      if (record.call_kind !== "decision") {
        state.lastTurn = Math.max(state.lastTurn ?? -1, record.context.agent_turn_idx);
      }
    }
    sessions.set(id, state);
    return state;
  }

  function render(ctx: ExtensionContext) {
    if (!ctx.hasUI) return;
    const state = session(ctx);
    const filter = { session_id: ctx.sessionManager.getSessionId(), agent_id: "root" };
    const all = metrics().stats(filter);
    const last = state.lastTurn === undefined ? { series: [] } : metrics().stats({ ...filter, agent_turn_idx: state.lastTurn });
    const count = (snapshot: MetricsSnapshot) => tokens(total(snapshot, "switchyard_llm_call_total_tokens_total"));
    const ttft = (snapshot: MetricsSnapshot) => duration(timing(snapshot, "switchyard_inference_ttft_ms", true));
    const latency = (snapshot: MetricsSnapshot) => duration(timing(snapshot, "switchyard_inference_duration_ms"));
    const forecast = state.prediction;
    const remaining = forecast?.expected_upper_tokens == null ? undefined : Math.max(0, forecast.expected_upper_tokens - state.tokensUsed);
    const task = forecast === undefined ? "" : remaining === undefined ? "remaining unavailable | "
      : `remaining ~${tokens(Math.round(remaining))} tok ETA ${completionDuration(state.completion.estimateMs(remaining))} | `;
    const stats = forecast === undefined
      ? `session ${count(all)} tok TTFT avg ${ttft(all)} req ${latency(all)} | last ${count(last)} tok TTFT ${ttft(last)} req ${latency(last)}`
      : `session ${count(all)}/${ttft(all)}/${latency(all)} | last ${count(last)}/${ttft(last)}/${latency(last)}`;
    ctx.ui.setStatus(STATUS_KEY, `SY ${task}${stats}`);
  }

  function record(ctx: ExtensionContext, observation: Omit<CallMetrics, "id">) {
    const saved = { id: randomUUID(), ...observation };
    metrics().record(saved);
    // A cancelled request can finish after Pi switches to another session.
    if (ctx.sessionManager.getSessionId() !== observation.context.session_id) return;
    pi.appendEntry(ENTRY_TYPE, saved);
    session(ctx).completion.record(saved);
    if (saved.call_kind !== "decision") session(ctx).lastTurn = saved.context.agent_turn_idx;
    render(ctx);
  }

  pi.on("session_start", (_, ctx) => {
    active = undefined;
    tools = undefined;
    session(ctx);
    render(ctx);
  });
  pi.on("agent_settled", (_, ctx) => {
    active = undefined;
    tools = undefined;
    const state = session(ctx);
    if (state.prediction === undefined) return;
    state.prediction = undefined;
    pi.appendEntry(PREDICTION_ENTRY, { session_id: ctx.sessionManager.getSessionId(), prediction: null, cleared: true });
    render(ctx);
  });
  pi.on("tool_execution_start", (event, ctx) => {
    if (!tools || tools.context.session_id !== ctx.sessionManager.getSessionId()) return;
    if (tools.running.size === 0) tools.started = performance.now();
    tools.running.add(event.toolCallId);
  });
  pi.on("tool_execution_end", (event, ctx) => {
    if (!tools || tools.context.session_id !== ctx.sessionManager.getSessionId()
      || !tools.running.delete(event.toolCallId) || tools.running.size > 0) return;
    const saved = { context: tools.context, duration_ms: performance.now() - tools.started };
    session(ctx).completion.recordTools(saved.context, saved.duration_ms);
    pi.appendEntry(TOOL_ENTRY, saved);
    render(ctx);
  });
  pi.on("message_update", (event, ctx) => {
    if (!active || active.firstToken !== undefined || event.message.role !== "assistant"
      || active.context.session_id !== ctx.sessionManager.getSessionId()) return;
    const update = event.assistantMessageEvent;
    const tool = update.type === "toolcall_start" ? update.partial.content[update.contentIndex] : undefined;
    const hasContent = (update.type === "text_delta" || update.type === "thinking_delta" || update.type === "toolcall_delta")
      ? update.delta.length > 0
      : (tool?.type === "toolCall" && (tool.name.length > 0 || Object.keys(tool.arguments).length > 0))
        || update.type === "toolcall_end"
        || ((update.type === "text_end" || update.type === "thinking_end") && update.content.length > 0);
    if (hasContent) active.firstToken = performance.now();
  });
  pi.on("message_end", (event, ctx) => {
    if (!active || event.message.role !== "assistant"
      || active.context.session_id !== ctx.sessionManager.getSessionId()) return;
    const current = active;
    active = undefined;
    const usage = event.message.usage;
    if (usage) session(ctx).tokensUsed += usage.input + usage.cacheWrite + usage.output;
    record(ctx, {
      context: current.context, model: `${event.message.provider}/${event.message.model}`, call_kind: "answer",
      is_success: event.message.stopReason !== "error" && event.message.stopReason !== "aborted",
      duration_ms: performance.now() - current.started,
      ...(current.firstToken === undefined ? {} : { ttft_ms: current.firstToken - current.started }),
      usage: tokenUsage(event.message.usage),
    });
    tools = event.message.stopReason === "toolUse" ? { context: current.context, running: new Set(), started: 0 } : undefined;
  });

  return {
    begin(reason: "user" | "continuation" | "retry" | "direct", ctx: ExtensionContext) {
      // Direct helper requests do not emit the agent message events used to finish measurements.
      if (reason === "direct") return undefined;
      tools = undefined;
      metrics();
      const state = session(ctx);
      if (reason === "user" || state.userTurn < 0) state.userTurn++;
      const current: Active = {
        started: performance.now(),
        context: { session_id: ctx.sessionManager.getSessionId(), agent_id: "root",
          agent_turn_idx: state.nextAgentTurn++, user_turn_idx: state.userTurn },
      };
      active = current;
      return {
        context: current.context,
        prediction(value: TaskPrediction | null, tokensUsed?: number) {
          if (ctx.sessionManager.getSessionId() !== current.context.session_id || active !== current) return;
          session(ctx).prediction = value;
          if (tokensUsed !== undefined) session(ctx).tokensUsed = tokensUsed;
          pi.appendEntry(PREDICTION_ENTRY, { session_id: current.context.session_id, prediction: value, tokens_used_so_far: session(ctx).tokensUsed });
          render(ctx);
        },
        async classify(model: string, call: () => Promise<ClassifierResult>) {
          const started = performance.now();
          let result: ClassifierResult | undefined;
          try { return result = await call(); }
          finally {
            record(ctx, { context: current.context, model, call_kind: "decision",
              is_success: result?.stopReason === "stop", duration_ms: performance.now() - started,
              usage: tokenUsage(result?.usage) });
          }
        },
        fail() {
          if (active !== current) return;
          active = undefined;
          record(ctx, { context: current.context, model: "switchyard/auto", call_kind: "request",
            is_success: false, duration_ms: performance.now() - current.started });
        },
      };
    },
  };
}

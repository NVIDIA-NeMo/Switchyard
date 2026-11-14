// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { taskPrediction, type TaskPrediction } from "@switchyard/libsy";
import type { Message, ModelCost, ModelThinkingLevel } from "@earendil-works/pi-ai";
import { getAgentDir, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { registerMetrics } from "./metrics.ts";
import { completionDuration } from "./completion-time.ts";
import { totalTokenPredictionRequest } from "./prediction-total.ts";
import { classifierContext, decisionResponse, modelIdentity } from "./forecast-classifier.ts";
import { ForecastWorker, observedTokens, type HorizonForecast } from "./forecast-worker.ts";
import { NextCallMAE, ERROR_ENTRY, errorStatus, type ErrorSample } from "./forecast-error.ts";
import { scoreRemainingForecasts, REMAINING_ERROR_ENTRY,
  type RemainingSettlement } from "./remaining-error.ts";

const ENTRY = "switchyard.compare.forecast.v2";
interface Pair {
  remaining_error_version?: 1;
  session_id: string;
  agent_turn_idx: number;
  lightgbm: HorizonForecast | null;
  greg: TaskPrediction | null;
  full_used?: number;
  token_basis: "full_input_plus_output";
  thinking_level: ModelThinkingLevel;
  model_cost?: ModelCost;
  errors: { lightgbm?: string; greg?: string };
}
const count = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n));

export default function (pi: ExtensionAPI) {
  const metrics = registerMetrics(pi, { status: false });
  let worker: ForecastWorker | undefined;
  const latest = new Map<string, Pair>();
  const errors = new Map<string, NextCallMAE>();
  const messages = (ctx: ExtensionContext): Message[] => ctx.sessionManager.getBranch().flatMap(entry =>
    entry.type === "message" && ["assistant", "user", "toolResult"].includes(entry.message.role) ? [entry.message as Message] : []);
  function tracker(ctx: ExtensionContext) {
    const id = ctx.sessionManager.getSessionId();
    if (!errors.has(id)) errors.set(id, new NextCallMAE(id, ctx.sessionManager.getEntries().flatMap(entry =>
      entry.type === "custom" && entry.customType === ERROR_ENTRY ? [entry.data as ErrorSample] : [])));
    return errors.get(id)!;
  }
  function render(ctx: ExtensionContext) {
    if (!ctx.hasUI) return;
    // The two widget rows are the entire comparison display.
    ctx.ui.setStatus("switchyard", undefined);
    const pair = latest.get(ctx.sessionManager.getSessionId());
    if (!pair) {
      ctx.ui.setWidget("switchyard-forecast-compare", ["LightGBM · waiting for forecasts…", "Jev · waiting for forecast…"], { placement: "belowEditor" });
      return;
    }
    const history = messages(ctx);
    const branch = ctx.sessionManager.getBranch();
    const lastState = branch.findLast(entry => entry.type === "custom"
      && (entry.customType === ENTRY || entry.customType === REMAINING_ERROR_ENTRY)
      && (entry.data as { session_id: string }).session_id === pair.session_id);
    // A completed turn has no future usage. Keep its original forecasts for error scoring.
    const completed = lastState?.type === "custom" && lastState.customType === REMAINING_ERROR_ENTRY
      && (lastState.data as RemainingSettlement).completed;
    const fullNow = observedTokens(history);
    const lgbmRemaining = completed ? 0 : pair.lightgbm && fullNow !== undefined && pair.full_used !== undefined
      ? Math.max(0, pair.lightgbm.remaining_task.tokens - (fullNow - pair.full_used)) : undefined;
    // Use the distribution's point estimate (band midpoints), not its upper-bound buffer.
    const gregRemaining = completed ? 0 : pair.greg?.predicted_tokens == null || fullNow === undefined ? undefined
      : Math.max(0, pair.greg.predicted_tokens - fullNow);
    const savedErrors = branch.flatMap(entry => entry.type === "custom"
      && entry.customType === REMAINING_ERROR_ENTRY ? [(entry.data as RemainingSettlement)] : [])
      .filter(record => record.session_id === pair.session_id).flatMap(record => record.samples);
    const formatError = (stats: ReturnType<NextCallMAE["summary"]>) => errorStatus(stats)
      .replace(/^ \| /, "").replace(/ \(n=\d+\)/, "");
    const row = (label: string, predictor: "lightgbm" | "greg", remaining: number | undefined, next?: string) => {
      const stats = new NextCallMAE(pair.session_id, savedErrors.filter(sample => sample.predictor === predictor)).summary();
      const err = formatError(stats);
      const estimate = remaining === undefined ? undefined : metrics.projectRemaining(ctx, remaining, "full", pair.model_cost);
      const cost = completed ? "$0.0000" : estimate?.usd === undefined ? "$—" : `~$${estimate.usd.toFixed(4)}`;
      const eta = completed ? "0s" : completionDuration(estimate?.eta_ms);
      const prefix = `${label} · ${cost} · ETA ${eta} · ${next ? `${next} | ` : ""}remaining`;
      if (remaining === undefined) return `${prefix} unavailable (${err})`;
      return `${prefix} ${completed ? "" : "~"}${count(remaining)} tok (${err})`;
    };
    const next = completed ? `next — (${formatError(tracker(ctx).summary())})`
      : pair.lightgbm ? `next ~${count(pair.lightgbm.next_call.tokens)} tok (${formatError(tracker(ctx).summary())})` : "next unavailable";
    const lines = [row("LightGBM", "lightgbm", lgbmRemaining, next), row("Jev", "greg", gregRemaining)];
    ctx.ui.setWidget("switchyard-forecast-compare", lines, { placement: "belowEditor" });
  }
  pi.on("session_start", (_, ctx) => {
    const saved = ctx.sessionManager.getBranch().filter(entry => entry.type === "custom" && entry.customType === ENTRY).at(-1);
    if (saved?.type === "custom") latest.set(ctx.sessionManager.getSessionId(), saved.data as Pair);
    render(ctx);
  });
  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    const sample = tracker(ctx).finish(event.message.usage, !["error", "aborted"].includes(event.message.stopReason));
    if (sample) pi.appendEntry(ERROR_ENTRY, sample);
    render(ctx);
  });
  pi.on("tool_execution_end", (_, ctx) => render(ctx));
  pi.on("agent_settled", (_, ctx) => {
    const id = ctx.sessionManager.getSessionId();
    const branch = ctx.sessionManager.getBranch();
    const boundary = branch.findLastIndex(entry => entry.type === "custom" && entry.customType === REMAINING_ERROR_ENTRY
      && (entry.data as RemainingSettlement).session_id === id);
    const pending = branch.slice(boundary + 1).flatMap(entry => entry.type === "custom" && entry.customType === ENTRY
      && (entry.data as Pair).session_id === id && (entry.data as Pair).remaining_error_version === 1 ? [entry.data as Pair] : []);
    if (pending.length) {
      const history = messages(ctx);
      const lastForecast = branch.findLastIndex(entry => entry.type === "custom" && entry.customType === ENTRY
        && (entry.data as Pair).session_id === id);
      const finalEntry = branch.slice(lastForecast + 1).findLast(entry => entry.type === "message" && entry.message.role === "assistant");
      // A normal final response defines this measured horizon. Aborts/errors close it without scores.
      const completed = finalEntry?.type === "message" && finalEntry.message.role === "assistant" && finalEntry.message.stopReason === "stop";
      const record: RemainingSettlement = { session_id: id, completed,
        samples: completed ? scoreRemainingForecasts(pending, observedTokens(history)) : [] };
      pi.appendEntry(REMAINING_ERROR_ENTRY, record);
    }
    render(ctx);
  });
  pi.on("session_shutdown", () => worker?.close());
  pi.registerVirtualModel({
    provider: "switchyard", id: "compare", name: "LightGBM + Greg forecasts (passthrough)",
    thinkingLevels: ["medium", "high", "max"],
    async route(request, ctx) {
      const measurement = metrics.begin(request.reason, ctx);
      try {
        const thinkingLevel = request.thinkingLevel ?? "medium";
        if (thinkingLevel !== "medium" && thinkingLevel !== "high" && thinkingLevel !== "max") throw new Error("Select --thinking medium, --thinking high, or --thinking max");
        const config = JSON.parse(readFileSync(process.env.SY_FORECAST_CONFIG ?? join(getAgentDir(), "switchyard-forecast.json"), "utf8"));
        const identity = modelIdentity(config.completion_model);
        const model = ctx.modelRegistry.find(identity.provider, identity.id);
        if (!model) throw new Error(`Unknown completion model: ${identity.reference}`);
        measurement?.modelCost(model.cost);
        if (measurement) {
          const id = ctx.sessionManager.getSessionId();
          tracker(ctx).begin("", undefined);
          const branch = messages(ctx);
          const history = branch.length ? branch : request.messages;
          if (ctx.hasUI) {
            ctx.ui.setStatus("switchyard", undefined);
            ctx.ui.setWidget("switchyard-forecast-compare", ["LightGBM · predicting…", "Jev · predicting…"], { placement: "belowEditor" });
          }
          // Both forecasts use this prefix and the same full-input-plus-output token target.
          const results = await Promise.allSettled([
            (async () => {
              worker ??= new ForecastWorker({ python: config.python, command: config.worker_command, researchRoot: config.research_root,
                tokenCastRoot: config.tokencast_root });
              return worker.predict({ session_id: id, cwd: ctx.cwd, model: identity.reference, thinking_level: thinkingLevel,
                messages: history, request_messages: request.messages }, request.signal);
            })(),
            (async () => {
              const classifier = modelIdentity(config.decision_model);
              const judge = ctx.modelRegistry.findOfType("classifier", classifier.provider, classifier.id);
              if (!judge) throw new Error(`Unknown classifier: ${classifier.reference}`);
              const prediction = totalTokenPredictionRequest(classifier.reference, history, identity.reference, thinkingLevel);
              const result = await measurement.classify(classifier.reference,
                () => ctx.modelRegistry.classify(judge, classifierContext(prediction), { signal: request.signal }), judge.cost);
              return taskPrediction(decisionResponse(prediction, result));
            })(),
          ]);
          if (request.signal?.aborted) throw Object.assign(new Error("Forecast cancelled"), { name: "AbortError" });
          if (ctx.sessionManager.getSessionId() === id) {
            const [lgbm, greg] = results;
            const pair: Pair = { remaining_error_version: 1, session_id: id, agent_turn_idx: measurement.context.agent_turn_idx,
              lightgbm: lgbm.status === "fulfilled" ? lgbm.value : null,
              greg: greg.status === "fulfilled" ? greg.value : null,
              full_used: observedTokens(history), token_basis: "full_input_plus_output", thinking_level: thinkingLevel, model_cost: model.cost,
              errors: { lightgbm: lgbm.status === "rejected" ? String(lgbm.reason?.message ?? lgbm.reason) : undefined,
                greg: greg.status === "rejected" ? String(greg.reason?.message ?? greg.reason) : undefined } };
            latest.set(id, pair);
            pi.appendEntry(ENTRY, pair);
            tracker(ctx).begin(`${id}:${pair.agent_turn_idx}`, pair.lightgbm?.next_call.tokens);
            render(ctx);
          }
        }
        measurement?.answerStarted();
        return { model, thinkingLevel };
      } catch (error) { measurement?.fail(); throw error; }
    },
  });
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { estimateCost, type CallMetrics } from "@switchyard/libsy";

export type Predictor = "lightgbm" | "greg";
type Dollars = number | null;
type Entries = readonly { type: string; customType?: string; data?: unknown }[];

export interface ObservedSpend {
  completion_usd: Dollars;
  prediction_api_usd: Dollars;
}
export interface CostPrediction {
  remaining_usd: Dollars;
  next_call_usd: Dollars;
  total_usd: Dollars; // Observed task spend plus the current remaining forecast.
  session_total_usd: Dollars;
  initial_total_usd: Dollars; // C0, frozen only at the first forecast of this task.
  first_priced_total_usd: Dollars;
  first_priced_agent_turn_idx: number | null;
  revision_usd: Dollars; // Change to the projected total since the previous forecast.
  implied_previous_call_usd: Dollars; // Previous remaining minus current remaining, not an observed charge.
  actual_since_previous_forecast_usd: Dollars;
}
export interface CostFrame {
  version: 1;
  currency: "USD";
  basis: "catalog_priced_acting_model_calls";
  task_start_agent_turn_idx: number;
  agent_turn_idx: number;
  spent_usd: Dollars;
  session_spent_usd: Dollars;
  prediction_api_spent_usd: Dollars;
  prediction_api_call_usd: Dollars;
  predictors: Record<Predictor, CostPrediction>;
}

const add = (a: Dollars, b: Dollars): Dollars => a === null || b === null ? null : a + b;
const subtract = (a: Dollars, b: Dollars): Dollars => a === null || b === null ? null : a - b;

/** Price recorded acting calls and forecasting API calls separately; unknown charges stay unknown. */
export function observedSpend(entries: Entries, sessionId: string, fromAgentTurn = 0): ObservedSpend {
  let completion: Dollars = 0, prediction: Dollars = 0;
  const seen = new Set<string>();
  for (const entry of entries) {
    if (entry.type !== "custom" || entry.customType !== "switchyard.metrics.v1") continue;
    const call = entry.data as CallMetrics;
    if (call.context.session_id !== sessionId || call.context.agent_turn_idx < fromAgentTurn || seen.has(call.id)) continue;
    seen.add(call.id);
    if (call.call_kind !== "answer" && call.call_kind !== "decision") continue;
    const charge = call.usage && call.rates ? estimateCost(call.usage, call.rates)?.usd ?? null : null;
    if (call.call_kind === "answer") completion = add(completion, charge);
    else prediction = add(prediction, charge);
  }
  return { completion_usd: completion, prediction_api_usd: prediction };
}

/** Freeze each cost forecast at its pre-call boundary, including the original C0. */
export function costFrame(input: {
  taskStart: number; agentTurn: number; spent: ObservedSpend; sessionSpent: Dollars;
  predictionApiCall: Dollars; remaining: Record<Predictor, Dollars>; nextCall: Dollars; previous: CostFrame[];
}): CostFrame {
  const first = input.previous[0], previous = input.previous.at(-1);
  const prediction = (name: Predictor): CostPrediction => {
    const total = add(input.spent.completion_usd, input.remaining[name]);
    const prior = previous?.predictors[name];
    const initial = first ? first.predictors[name].initial_total_usd
      : input.agentTurn === input.taskStart ? total : null;
    return {
      remaining_usd: input.remaining[name], next_call_usd: name === "lightgbm" ? input.nextCall : null,
      total_usd: total, session_total_usd: add(input.sessionSpent, input.remaining[name]), initial_total_usd: initial,
      first_priced_total_usd: prior?.first_priced_total_usd ?? total,
      first_priced_agent_turn_idx: prior?.first_priced_agent_turn_idx ?? (total === null ? null : input.agentTurn),
      revision_usd: prior ? subtract(total, prior.total_usd) : null,
      implied_previous_call_usd: prior ? subtract(prior.remaining_usd, input.remaining[name]) : null,
      actual_since_previous_forecast_usd: previous ? subtract(input.spent.completion_usd, previous.spent_usd) : null,
    };
  };
  return {
    version: 1, currency: "USD", basis: "catalog_priced_acting_model_calls",
    task_start_agent_turn_idx: input.taskStart, agent_turn_idx: input.agentTurn,
    spent_usd: input.spent.completion_usd, session_spent_usd: input.sessionSpent,
    prediction_api_spent_usd: input.spent.prediction_api_usd, prediction_api_call_usd: input.predictionApiCall,
    predictors: { lightgbm: prediction("lightgbm"), greg: prediction("greg") },
  };
}

/** Consume observed dollars, not future forecasts; terminal errors compare against the frozen baseline. */
export function costView(frame: CostFrame, name: Predictor, spentNow: Dollars, completed: boolean) {
  const estimate = frame.predictors[name];
  const used = subtract(spentNow, frame.spent_usd);
  const remaining = completed ? 0 : used === null || estimate.remaining_usd === null || used < 0
    ? null : Math.max(0, estimate.remaining_usd - used);
  return {
    spent_usd: spentNow, remaining_usd: remaining, total_usd: add(spentNow, remaining),
    session_total_usd: add(add(frame.session_spent_usd, used), remaining),
    spent_since_forecast_usd: used,
    terminal_revision_usd: completed ? subtract(spentNow, estimate.total_usd) : null,
    initial_total_usd: estimate.initial_total_usd,
    actual_minus_initial_usd: completed ? subtract(spentNow, estimate.initial_total_usd) : null,
    first_priced_total_usd: estimate.first_priced_total_usd,
    first_priced_agent_turn_idx: estimate.first_priced_agent_turn_idx,
    actual_minus_first_priced_usd: completed ? subtract(spentNow, estimate.first_priced_total_usd) : null,
  };
}

export function finalCost(frame: CostFrame, spend: ObservedSpend) {
  return { actual_usd: spend.completion_usd, prediction_api_spent_usd: spend.prediction_api_usd,
    lightgbm: costView(frame, "lightgbm", spend.completion_usd, true),
    greg: costView(frame, "greg", spend.completion_usd, true) };
}
export type CostOutcome = ReturnType<typeof finalCost>;

export function formatDollars(value: Dollars, estimated = false) {
  return value === null ? "$—" : `${estimated ? "~" : ""}$${value.toFixed(4)}`;
}
export function formatCostError(view: ReturnType<typeof costView>) {
  const initial = view.actual_minus_initial_usd;
  const error = initial ?? view.actual_minus_first_priced_usd;
  if (error === null) return "C0 err —";
  return `${initial === null ? "first err" : "C0 err"} ${error < 0 ? "−" : "+"}$${Math.abs(error).toFixed(4)}`;
}

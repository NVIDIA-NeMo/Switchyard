// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ErrorSample } from "./forecast-error.ts";
import type { CostOutcome } from "./cost-forecast.ts";

export const REMAINING_ERROR_ENTRY = "switchyard.compare.remaining-errors.v2";
export interface RemainingForecast {
  session_id: string;
  agent_turn_idx: number;
  lightgbm: { remaining_task: { tokens: number } } | null;
  greg: { predicted_tokens?: number | null } | null;
  full_used?: number;
}
export interface RemainingError extends ErrorSample {
  predictor: "lightgbm" | "greg";
}
export interface RemainingSettlement {
  session_id: string;
  completed: boolean;
  samples: RemainingError[];
  cost_outcome?: CostOutcome;
}

/** Score original pre-call forecasts against usage from their cutoff to the final response. */
export function scoreRemainingForecasts(forecasts: RemainingForecast[], fullTotal: number | undefined): RemainingError[] {
  const samples = new Map<string, RemainingError>();
  const add = (forecast: RemainingForecast, predictor: RemainingError["predictor"],
    predicted: number | null | undefined, cutoff: number | undefined, total: number | undefined) => {
    if (predicted == null || cutoff === undefined || total === undefined
      || ![predicted, cutoff, total].every(n => Number.isFinite(n) && n >= 0) || total < cutoff) return;
    const actual = total - cutoff;
    const sample: RemainingError = {
      predictor, session_id: forecast.session_id, sample_id: `${forecast.session_id}:${forecast.agent_turn_idx}:${predictor}`,
      predicted_tokens: predicted, actual_tokens: actual,
      signed_error: predicted - actual, absolute_error: Math.abs(predicted - actual),
    };
    samples.set(sample.sample_id, sample);
  };
  for (const forecast of forecasts) {
    add(forecast, "lightgbm", forecast.lightgbm?.remaining_task.tokens, forecast.full_used, fullTotal);
    const total = forecast.greg?.predicted_tokens;
    const predicted = total == null || forecast.full_used === undefined ? undefined
      : Math.max(0, total - forecast.full_used);
    add(forecast, "greg", predicted, forecast.full_used, fullTotal);
  }
  return [...samples.values()];
}

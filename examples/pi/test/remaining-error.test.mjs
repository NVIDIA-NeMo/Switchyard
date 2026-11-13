// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { scoreRemainingForecasts } from "../remaining-error.ts";
import { NextCallMAE } from "../forecast-error.ts";

const forecast = (overrides = {}) => ({ session_id: "s", agent_turn_idx: 1,
  lightgbm: { remaining_task: { tokens: 1200 } }, greg: { predicted_tokens: 1000 },
  full_used: 400, ...overrides });

test("scores remaining tokens from each forecast cutoff using the same full-token definition for both predictors", () => {
  const samples = scoreRemainingForecasts([forecast()], 1200);
  assert.equal(samples.length, 2);
  assert.deepEqual(samples.map(s => [s.predictor, s.predicted_tokens, s.actual_tokens, s.signed_error]),
    [["lightgbm", 1200, 800, 400], ["greg", 600, 800, -200]]);
  const greg = new NextCallMAE("s", samples.filter(s => s.predictor === "greg")).summary();
  assert.equal(greg.meanPercentageError, -25);
  assert.equal(greg.mae, 200);
});

test("averages independent forecast cutoffs, deduplicates records, and excludes undefined percentages", () => {
  const first = forecast();
  const second = forecast({ agent_turn_idx: 2, full_used: 800,
    lightgbm: { remaining_task: { tokens: 200 } }, greg: { predicted_tokens: 1400 } });
  const samples = scoreRemainingForecasts([first, first, second], 1200);
  assert.equal(samples.length, 4);
  assert.equal(new NextCallMAE("s", samples.filter(s => s.predictor === "lightgbm")).summary().meanPercentageError, 0);
  assert.equal(new NextCallMAE("s", samples.filter(s => s.predictor === "greg")).summary().meanPercentageError, 12.5);
  const zero = scoreRemainingForecasts([forecast()], 400);
  assert.equal(new NextCallMAE("s", zero).summary().percentageCount, 0);
});

test("missing usage, impossible forecasts and invalid cutoffs stay unavailable", () => {
  assert.equal(scoreRemainingForecasts([forecast()], undefined).length, 0);
  assert.equal(scoreRemainingForecasts([forecast()], 399).length, 0);
  assert.equal(scoreRemainingForecasts([forecast({ lightgbm: null, greg: { predicted_tokens: null } })], 1200).length, 0);
  assert.equal(scoreRemainingForecasts([forecast({ full_used: undefined })], 1200).length, 0);
});

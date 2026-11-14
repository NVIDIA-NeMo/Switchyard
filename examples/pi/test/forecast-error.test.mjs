// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { NextCallMAE, errorStatus } from "../forecast-error.ts";

test("MAE pairs each old forecast with actual input/cache/output once", () => {
  const tracker = new NextCallMAE("s");
  tracker.begin("s:0", 100);
  const first = tracker.finish({ input: 20, cacheRead: 30, cacheWrite: 10, output: 20, reasoning: 5 }, true);
  assert.equal(first.actual_tokens, 80);
  assert.equal(first.absolute_error, 20);
  assert.equal(tracker.finish({ input: 20, cacheRead: 30, cacheWrite: 10, output: 20 }, true), undefined);
  tracker.begin("s:1", 200);
  tracker.finish({ input: 100, cacheRead: 0, cacheWrite: 0, output: 50 }, true);
  assert.equal(tracker.summary().mae, 35);
  assert.ok(Math.abs(tracker.summary().meanPercentageError - (25 + 100/3) / 2) < 1e-10);
  assert.match(errorStatus(tracker.summary()), /avg err \+29.2% \(n=2\)/);
});

test("missing forecasts, invalid usage and failed calls never count as zero error", () => {
  const tracker = new NextCallMAE("s");
  const usage = { input: 10, cacheRead: 0, cacheWrite: 0, output: 2 };
  tracker.begin("s:0", undefined);
  tracker.finish(usage, true);
  tracker.begin("s:1", 100);
  tracker.finish(undefined, true);
  tracker.begin("s:2", 100);
  tracker.finish(usage, false);
  tracker.begin("s:3", 100);
  tracker.finish({ ...usage, input: NaN }, true);
  assert.equal(tracker.summary().count, 0);
});

test("resume deduplicates saved records and a new session starts fresh", () => {
  const sample = { sample_id: "s:0", session_id: "s", predicted_tokens: 100, actual_tokens: 80, absolute_error: 20 };
  const tracker = new NextCallMAE("s", [sample, sample]);
  assert.equal(tracker.summary().count, 1);
  tracker.begin("s:0", 100);
  assert.equal(tracker.finish({ input: 70, output: 10, cacheRead: 0, cacheWrite: 0 }, true), undefined);
  tracker.begin("s:1", 200);
  tracker.finish({ input: 100, output: 50, cacheRead: 0, cacheWrite: 0 }, true);
  assert.equal(tracker.summary().mae, 35);
  assert.equal(new NextCallMAE("fork", [sample]).summary().count, 0);
});

test("signed error distinguishes underprediction and restores old records", () => {
  const tracker = new NextCallMAE("s");
  tracker.begin("s:0", 50);
  const sample = tracker.finish({ input: 100, output: 10, cacheRead: 0, cacheWrite: 0 }, true);
  assert.equal(sample.signed_error, -60);
  assert.match(errorStatus(tracker.summary()), /avg err -54.5%/);
  const old = { ...sample }; delete old.signed_error;
  assert.match(errorStatus(new NextCallMAE("s", [old]).summary()), /avg err -54.5%/);
});

test("percentage error excludes zero actual tokens and survives resume", () => {
  const tracker = new NextCallMAE("s");
  tracker.begin("s:0", 50);
  tracker.finish({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, true);
  assert.equal(tracker.summary().percentageCount, 0);
  assert.match(errorStatus(tracker.summary()), /avg err — \(n=0\)/);
  const saved = { sample_id: "s:1", session_id: "s", predicted_tokens: 150, actual_tokens: 100, absolute_error: 50 };
  assert.match(errorStatus(new NextCallMAE("s", [saved]).summary()), /avg err \+50.0% \(n=1\)/);
});

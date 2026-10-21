// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { taskPrediction, taskPredictionRequest } from "../dist/index.js";

test("native task predictor matches the Python task-completion fixture", () => {
  const fixture = JSON.parse(readFileSync(new URL("../../../crates/switchyard-token-prediction/tests/fixtures/python.json", import.meta.url), "utf8"));
  const request = taskPredictionRequest("jev-latest", fixture.request.state.task_history, [
    { uncached_input_tokens: 800, output_tokens: 100, reasoning_tokens: 40 },
    { uncached_input_tokens: 300, output_tokens: 200, reasoning_tokens: 50 },
  ]);
  assert.deepEqual(request.context, fixture.request.state);
  const options = request.questions.minimum_likely_budget.kind.data.options;
  assert.equal(options.length, 81);
  assert.deepEqual(Object.fromEntries(options.map(({ id, description }) => [id, description])),
    fixture.request.questions.minimum_likely_budget.criteria);
  const answer = fixture.response.answers.minimum_likely_budget;
  const response = { answers: { minimum_likely_budget: { value: { type: "choice", data: {
    selected: answer.choice, probabilities: answer.probabilities,
  } } } } };
  const prediction = taskPrediction(response);
  assert.equal(prediction.predicted_tokens, fixture.prediction.predicted_tokens);
  assert.equal(prediction.impossible_probability, fixture.prediction.impossible_probability);
  assert.deepEqual(prediction.selected_interval, fixture.prediction.selected_interval);
  assert.throws(() => taskPrediction({ answers: {} }), { code: "ERR_INVALID_REQUEST" });
  assert.throws(() => taskPredictionRequest("jev", [], [{ uncached_input_tokens: -1, output_tokens: 0, reasoning_tokens: 0 }]), { code: "ERR_INVALID_REQUEST" });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { DllmEngine } from "../dist/index.js";

test("failed library loads release the process engine lease", async () => {
  const options = {
    library: "/nonexistent/switchyard-dllm.so", tokenizer_library: "/nonexistent/tokenizer.so",
    weights: "/nonexistent/weights", tokenizer: "/nonexistent/tokenizer",
  };
  for (let i = 0; i < 2; i++) {
    await assert.rejects(DllmEngine.load(options), /switchyard-dllm.so/);
  }
});

test("native engine serializes decisions, survives invalid input, and releases GPU resources", {
  skip: !process.env.SWITCHYARD_DLLM_OPTIONS,
}, async () => {
  const options = JSON.parse(readFileSync(process.env.SWITCHYARD_DLLM_OPTIONS, "utf8"));
  await assert.rejects(DllmEngine.load({ ...options, tokenizer: "/nonexistent/tokenizer" }), /dllm tokenizer: OpenTextFileFailed/);
  const request = {
    model: "dllm/deciffusion", input: [{ role: "user", content: [{ type: "text", text: "What is 2 + 2?" }] }],
    questions: { route: {
      instructions: "Select the model tier needed to answer the user's request.",
      kind: { type: "choice", data: { options: [
        { id: "capable", description: "Complex reasoning, difficult coding, or expert analysis." },
        { id: "efficient", description: "Simple questions and routine tasks." },
      ] } },
    } },
  };
  const engine = await DllmEngine.load(options);
  try {
    await assert.rejects(DllmEngine.load(options), /already loaded/);
    await assert.rejects(engine.decide({ ...request, questions: {} }), /requires questions/);
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(engine.decide(request, { signal: abort.signal }), { name: "AbortError" });
    const active = new AbortController();
    const cancelled = engine.decide(request, { signal: active.signal });
    active.abort();
    await assert.rejects(cancelled, { name: "AbortError" });
    const results = await Promise.all([engine.decide(request), engine.decide(request)]);
    for (const result of results) {
      const answer = result.answers.route.value.data;
      assert.equal(answer.selected, "efficient");
      assert.ok(answer.probabilities.efficient > answer.probabilities.capable);
      assert.ok(Math.abs(Object.values(answer.probabilities).reduce((a, b) => a + b) - 1) < 1e-9);
    }
  } finally { await engine.close(); }
  await assert.rejects(engine.decide(request), /closed/);
  await engine.close();
  const reloaded = await DllmEngine.load(options);
  await reloaded.close();
});

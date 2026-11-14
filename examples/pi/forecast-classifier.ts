// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DecisionRequest, DecisionResponse } from "@switchyard/libsy";
import type { ClassifierContext, ClassifierResult } from "@earendil-works/pi-ai";
import { tokenUsage } from "./metrics.ts";

export function modelIdentity(reference: unknown) {
  if (typeof reference !== "string") throw new Error("Expected a model in provider/model form");
  // Model IDs may themselves contain slashes, as on OpenRouter.
  const slash = reference.indexOf("/");
  if (slash <= 0 || slash === reference.length - 1) throw new Error("Expected a model in provider/model form");
  return { reference, provider: reference.slice(0, slash), id: reference.slice(slash + 1) };
}

function text(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value ?? null);
}

export function classifierContext(request: DecisionRequest): ClassifierContext {
  const questions: ClassifierContext["questions"] = {};
  for (const [id, question] of Object.entries(request.questions)) {
    if (question.kind.type !== "choice") throw new Error(`Expected a choice question: ${id}`);
    questions[id] = {
      type: "choice",
      instructions: text(question.instructions),
      criteria: Object.fromEntries(question.kind.data.options.map((option) => [option.id, text(option.description)])),
    };
  }
  return { state: request.context as ClassifierContext["state"], questions };
}

export function decisionResponse(request: DecisionRequest, result: ClassifierResult): DecisionResponse {
  if (result.stopReason === "aborted") {
    throw Object.assign(new Error(result.errorMessage ?? "Classification cancelled"), { name: "AbortError", code: "ABORT_ERR" });
  }
  if (result.stopReason === "error") throw new Error(result.errorMessage ?? "Classification failed");
  const answers = Object.fromEntries(Object.entries(request.questions).map(([id, question]) => {
    const answer = result.answers[id];
    if (question.kind.type !== "choice" || answer?.type !== "choice"
      || !question.kind.data.options.some((option) => option.id === answer.choice)) {
      throw new Error(`Expected a declared choice for question: ${id}`);
    }
    return [id, {
      value: { type: "choice" as const, data: { selected: answer.choice, probabilities: answer.probabilities } },
      provider_confidence: answer.confidence,
    }];
  }));
  return { model: `${result.provider}/${result.model}`, answers,
    ...(result.usage ? { usage: tokenUsage(result.usage) } : {}) };
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Message, ModelThinkingLevel } from "@earendil-works/pi-ai";
import { predictionRequest } from "./prediction.ts";
import { observedTokens } from "./forecast-worker.ts";

/** Reuse Greg's history buffer and budget bands with the checkpoint's full-token target. */
export function totalTokenPredictionRequest(judge: string, messages: readonly Message[],
  completionModel: string, thinkingLevel: ModelThinkingLevel) {
  const used = observedTokens(messages);
  if (used === undefined) throw new Error("Full token usage is missing from the observed history");
  const request = predictionRequest(judge, messages);
  const context = request.context as Record<string, unknown>;
  const fullTokens = (text: string) => text
    .replaceAll("uncached input plus output", "all input (including cache reads and writes) plus output")
    .replaceAll("uncached-input-plus-output", "full-input-plus-output");
  request.context = {
    ...context,
    instructions: fullTokens(String(context.instructions))
      + " Count input again on every model call, including repeated and cached context. Cache discounts affect dollars, not token counts."
      + " Include reasoning once as part of output. The predicted total includes all tokens already used."
      + " Forecast through the final response to the latest user request, not future user requests."
      + " Exclude this forecasting model's own usage.",
    completion_model: completionModel,
    thinking_level: thinkingLevel,
    token_usage: {
      definition: "All provider input plus output: input + cacheRead + cacheWrite + output. Reasoning is included in output.",
      tokens_used_so_far: used,
      per_step: messages.flatMap(message => message.role === "assistant" && message.usage ? [{
        step: 0, input_tokens: message.usage.input + message.usage.cacheRead + message.usage.cacheWrite,
        cache_read_tokens: message.usage.cacheRead, cache_write_tokens: message.usage.cacheWrite,
        output_tokens: message.usage.output, reasoning_output_tokens: message.usage.reasoning ?? 0,
        tokens_used: message.usage.input + message.usage.cacheRead + message.usage.cacheWrite + message.usage.output,
      }] : []).map((usage, index) => ({ ...usage, step: index + 1 })),
    },
  };
  for (const question of Object.values(request.questions)) {
    const instructions = question.instructions as Record<string, unknown>;
    question.instructions = { ...instructions, task: fullTokens(String(instructions.task)),
      use_state_fields: ["instructions", "task_history", "token_usage", "completion_model", "thinking_level"] };
    if (question.kind.type === "choice") for (const option of question.kind.data.options) {
      if (typeof option.description === "string") option.description = fullTokens(option.description);
    }
  }
  return request;
}

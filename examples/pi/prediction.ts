// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { taskPredictionRequest, type ResponseUsage } from "@switchyard/libsy";
import type { Message } from "@earendil-works/pi-ai";

/** Keep the initial task and recent content within a bounded classifier prompt. */
export function predictionRequest(model: string, messages: readonly Message[], usageMessages: readonly Message[] = messages) {
  const priorUsage = usageMessages.flatMap<ResponseUsage>((message) => {
    if (message.role !== "assistant" || !message.usage) return [];
    return [{ uncached_input_tokens: message.usage.input + message.usage.cacheWrite,
      output_tokens: message.usage.output, reasoning_tokens: message.usage.reasoning ?? 0 }];
  });
  const steps = messages.map((message, index) => {
    const content = typeof message.content === "string" ? message.content : message.content
      .filter((block) => block.type !== "image")
      .map((block) => block.type === "text" ? block.text : block.type === "thinking" ? block.thinking : JSON.stringify(block)).join("\n");
    return { step_id: index + 1, source: message.role === "assistant" ? "agent" : message.role,
      message: content.length > 4000 ? `${content.slice(0, 4000)}\n[truncated]` : content,
      ...(message.role === "toolResult" ? { tool_name: message.toolName, is_error: message.isError } : {}) };
  });
  const first = steps.find((step) => step.source === "user");
  const recent = [];
  let size = first ? JSON.stringify(first).length : 0;
  for (const step of steps.toReversed()) {
    if (step === first) break;
    size += JSON.stringify(step).length;
    if (size > 60_000) break;
    recent.push(step);
  }
  return taskPredictionRequest(model, [...(first ? [first] : []), ...recent.reverse()], priorUsage);
}

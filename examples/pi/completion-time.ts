// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CallMetrics, MetricContext, TokenUsage } from "@switchyard/libsy";

const turnKey = (context: MetricContext) => JSON.stringify([context.session_id, context.agent_id, context.agent_turn_idx]);

/** Session calibration from paired agent-call timing and uncached token usage. */
export class CompletionTiming {
  #outputShare = 0;
  #cacheReadShare = 0;
  #cacheWriteShare = 0;
  #longWriteShare = 0;
  #promptTokens = 0;
  #output = 0;
  #ttftMs = 0;
  #outputMs = 0;
  #calls = 0;
  #toolMs = 0;
  #timedTurns = new Set<string>();

  record(call: CallMetrics) {
    if (call.call_kind !== "answer" || !call.is_success || call.ttft_ms === undefined
      || call.usage?.input_tokens == null || call.usage.output_tokens == null
      || call.usage.output_tokens <= 0 || call.duration_ms <= call.ttft_ms) return;
    const input = call.usage.input_tokens + (call.usage.cache_creation_input_tokens ?? 0);
    // Each response contributes equally so a large prefill cannot dominate the token mix.
    this.#outputShare += call.usage.output_tokens / (input + call.usage.output_tokens);
    const total = input + call.usage.output_tokens;
    this.#cacheReadShare += (call.usage.cached_input_tokens ?? 0) / total;
    this.#cacheWriteShare += (call.usage.cache_creation_input_tokens ?? 0) / total;
    this.#longWriteShare += (call.usage.cache_creation_1h_input_tokens ?? 0) / total;
    this.#promptTokens += input + (call.usage.cached_input_tokens ?? 0);
    this.#output += call.usage.output_tokens;
    // TTFT includes routing, network delay and input processing; output time starts after it.
    this.#ttftMs += call.ttft_ms;
    this.#outputMs += call.duration_ms - call.ttft_ms;
    this.#calls++;
    this.#timedTurns.add(turnKey(call.context));
  }

  /** Adds one non-overlapping tool execution interval from a calibrated response. */
  recordTools(context: MetricContext, durationMs: number) {
    if (this.#timedTurns.has(turnKey(context))) this.#toolMs += durationMs;
  }

  /** Mean prompt size per call, for selecting the current model's pricing tier. */
  get meanPromptTokens(): number { return this.#calls ? this.#promptTokens / this.#calls : 0; }

  /** Splits the uncached budget and projects additional cache reads for monetary pricing. */
  remainingUsage(remainingTokens: number): TokenUsage | undefined {
    if (this.#calls === 0 && remainingTokens > 0) return undefined;
    const total = Math.round(remainingTokens);
    const share = (value: number) => this.#calls ? Math.round(total * value / this.#calls) : 0;
    const output = share(this.#outputShare);
    const write = Math.min(total - output, share(this.#cacheWriteShare));
    return { input_tokens: total - output - write, output_tokens: output,
      cached_input_tokens: share(this.#cacheReadShare), cache_creation_input_tokens: write,
      cache_creation_1h_input_tokens: Math.min(write, share(this.#longWriteShare)) };
  }

  /** Estimated model and tool time for a remaining uncached-input-plus-output budget. */
  estimateMs(remainingTokens: number): number | undefined {
    if (remainingTokens === 0) return 0;
    if (this.#calls === 0) return undefined;
    const output = remainingTokens * (this.#outputShare / this.#calls);
    const outputTokensPerSecond = this.#output / (this.#outputMs / 1000);
    const remainingCalls = output / (this.#output / this.#calls);
    // TTFT includes per-request delay; it is not an uncached-input token throughput.
    const inputMs = remainingCalls * (this.#ttftMs / this.#calls);
    const toolMs = remainingCalls * (this.#toolMs / this.#calls);
    return inputMs + output / outputTokensPerSecond * 1000 + toolMs;
  }
}

export function completionDuration(milliseconds: number | undefined): string {
  if (milliseconds === undefined) return "—";
  const seconds = Math.ceil(milliseconds / 1000);
  if (seconds < 60) return `~${seconds}s`;
  if (seconds < 3600) return `~${Math.floor(seconds / 60)}m${seconds % 60 ? `${seconds % 60}s` : ""}`;
  const minutes = Math.floor(seconds / 60) % 60;
  return `~${Math.floor(seconds / 3600)}h${minutes ? `${minutes}m` : ""}`;
}

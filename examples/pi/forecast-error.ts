// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const ERROR_ENTRY = "switchyard.lightgbm.error.v1";

export interface ErrorSample {
  sample_id: string;
  session_id: string;
  predicted_tokens: number;
  actual_tokens: number;
  absolute_error: number;
  signed_error?: number; // Predicted minus actual; optional on older saved records.
}

export class NextCallMAE {
  private samples = new Map<string, ErrorSample>();
  private pending?: { sampleId: string; predicted: number };
  private sessionId: string;
  constructor(sessionId: string, saved: ErrorSample[] = []) {
    this.sessionId = sessionId;
    for (const sample of saved) {
      if (sample.session_id === sessionId && Number.isFinite(sample.absolute_error)) {
        this.samples.set(sample.sample_id, sample);
      }
    }
  }
  begin(sampleId: string, predicted: number | undefined) {
    this.pending = predicted !== undefined && Number.isFinite(predicted) ? { sampleId, predicted } : undefined;
  }
  finish(usage: { input: number; output: number; cacheRead: number; cacheWrite: number } | undefined, success: boolean) {
    const pending = this.pending;
    this.pending = undefined;
    if (!pending || !success || !usage || this.samples.has(pending.sampleId)) return;
    const channels = [usage.input, usage.output, usage.cacheRead, usage.cacheWrite];
    if (!channels.every((value) => Number.isFinite(value) && value >= 0)) return;
    // Same target as the checkpoint: all provider input + output, reasoning once.
    const actual = channels.reduce((sum, value) => sum + value, 0);
    const sample: ErrorSample = { sample_id: pending.sampleId, session_id: this.sessionId,
      predicted_tokens: pending.predicted, actual_tokens: actual, signed_error: pending.predicted - actual, absolute_error: Math.abs(pending.predicted - actual) };
    this.samples.set(sample.sample_id, sample);
    return sample;
  }
  summary() {
    const samples = [...this.samples.values()];
    const total = samples.reduce((sum, s) => sum + s.absolute_error, 0);
    // Zero-token responses have no defined percentage error.
    const percentages = samples.filter(s => s.actual_tokens > 0)
      .map(s => 100 * (s.predicted_tokens - s.actual_tokens) / s.actual_tokens);
    const meanPercentageError = percentages.length
      ? percentages.reduce((sum, value) => sum + value, 0) / percentages.length : null;
    return { percentageCount: percentages.length, meanPercentageError, count: samples.length, mae: samples.length ? total / samples.length : null,
      last: samples.at(-1)?.absolute_error ?? null,
      lastSigned: samples.length ? samples.at(-1)!.predicted_tokens - samples.at(-1)!.actual_tokens : null };
  }
}

export function errorStatus(stats: ReturnType<NextCallMAE["summary"]>) {
  return stats.meanPercentageError === null ? " | avg err — (n=0)"
    : ` | avg err ${stats.meanPercentageError.toLocaleString("en-US", { signDisplay: "always", minimumFractionDigits: 1, maximumFractionDigits: 1 })}% (n=${stats.percentageCount})`;
}

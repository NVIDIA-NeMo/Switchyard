// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";

/** Opaque routing state. Store it unchanged and pass it back on the same branch. */
export type CompositeState = { readonly [key: string]: unknown };

export interface CompositeOptions {
  targets: { capable: string; efficient: string; classifier: string };
  classifier: {
    classify_trigger: "user_turn" | "new_session";
    fail_open: boolean;
    decision: {
      cutoff: number;
      candidates: Record<string, "capable" | "efficient">;
      evidence: unknown;
      instructions?: unknown;
    };
  };
  stage: { efficient_first_threshold: number; capable_first_threshold: number };
}

export interface Selection {
  model: string;
  /** Present only when the state changed. */
  state?: CompositeState;
}

/** Text and tool activity used for routing, in Switchyard's message format. */
export type RoutingContent =
  | { type: "text"; text: string }
  | { type: "tool_call"; id: string; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; tool_call_id: string; content: readonly RoutingContent[]; is_error: boolean };

export interface RoutingMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content: readonly RoutingContent[];
}

export interface DecisionRequest {
  model?: string | null;
  context: unknown;
  questions: Record<string, DecisionQuestion>;
}

export interface DecisionQuestion {
  instructions: unknown;
  kind:
    | { type: "boolean"; data: { true_description?: unknown; false_description?: unknown } }
    | { type: "choice"; data: { options: { id: string; description?: unknown }[] } }
    | { type: "score"; data: { levels: unknown[] } };
}

export interface DecisionResponse {
  id?: string | null;
  model?: string | null;
  answers: Record<string, DecisionAnswer>;
  usage?: TokenUsage;
}

/** Normalized usage; input_tokens excludes cache reads and creation. */
export interface TokenUsage {
  provider_cost?: number | null;
  input_tokens?: number | null;
  cached_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_creation_1h_input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  reasoning_tokens?: number | null;
}

/** Host-owned indices, starting at zero within each session and agent. */
export interface MetricContext {
  session_id: string;
  agent_id: string;
  agent_turn_idx: number;
  user_turn_idx: number;
}

export type MetricFilter = Partial<MetricContext>;

/** Usage of a completed agent call, excluding cache reads and counting reasoning once. */
export interface ResponseUsage {
  uncached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
}

/** Total task budget, including usage so far; means are conditional on a finite budget. */
export interface TaskPrediction {
  predicted_tokens: number | null;
  expected_upper_tokens: number | null;
  selected_interval: [number, number] | null;
  selected_choice: string;
  impossible_probability: number;
  finite_budget_probability: number;
}

/** Builds the Rust predictor's 80-range task-completion Choice request. */
export function taskPredictionRequest(model: string, history: readonly unknown[], priorUsage: readonly ResponseUsage[]): DecisionRequest {
  try { return JSON.parse(native.taskPredictionRequest(json({ model, history, prior_usage: priorUsage }))); }
  catch (error) { throw publicError(error); }
}

/** Decodes a complete Choice distribution using the Rust task-budget predictor. */
export function taskPrediction(response: DecisionResponse): TaskPrediction {
  try { return JSON.parse(native.taskPrediction(json(response))); }
  catch (error) { throw publicError(error); }
}

/** A completed host call. Replaying the same ID within a session is idempotent. */
export interface CallMetrics {
  id: string;
  context: MetricContext;
  model: string;
  call_kind: "answer" | "decision" | "request";
  is_success: boolean;
  duration_ms: number;
  /** Time to first generated content; omit when no token was observed. */
  ttft_ms?: number;
  usage?: TokenUsage;
  rates?: TokenRates;
}

/** USD per million tokens, using switchyard-costs pricing fields. */
export interface TokenRates {
  input: number;
  output: number;
  cache_read: number;
  cache_write: number;
  cache_write_1h?: number;
}

export interface CostTotals {
  known_usd: number;
  unknown_calls: number;
  estimated_calls: number;
  provider_reported_calls: number;
  total_usd: number | null;
}

export interface RecordedCost {
  usd: number;
  source: "estimated" | "provider_reported";
}

/** Prices projected usage with switchyard-costs without recording a charge. */
export function estimateCost(usage: TokenUsage, rates: TokenRates): RecordedCost | null {
  try { return JSON.parse(native.estimateCost(json({ usage, rates }))); }
  catch (error) { throw publicError(error); }
}

export interface MetricsSnapshot {
  series: {
    name: string;
    labels: Record<string, string>;
    value:
      | { type: "counter" | "gauge"; value: number }
      | { type: "histogram"; count: number; sum: number; buckets: { upper_bound: number; count: number }[] };
  }[];
}

export interface DecisionAnswer {
  value:
    | { type: "boolean"; data: { type: "value"; data: boolean } | { type: "probability_true"; data: number } }
    | { type: "choice"; data: { selected: string; probabilities?: Record<string, number> | null } }
    | { type: "score"; data: { value: number; probabilities?: number[] | null } };
  provider_confidence?: number | null;
}

type NativeStep =
  | { kind: "decision"; request: string }
  | { kind: "done"; model: string; state: string | null };

interface NativeRun {
  next(response: string | null, error: string | null): Promise<NativeStep>;
  close(): void;
}

interface NativeComposite {
  start(messages: string, state: string | null, metricContext: string | null): NativeRun;
}

interface NativeMetrics {
  record(observation: string): boolean;
  stats(query: string): string;
  prometheus(query: string): string;
  costs(query: string): string;
}

const native: {
  NativeComposite: new (options: string) => NativeComposite;
  NativeMetrics: new () => NativeMetrics;
  taskPredictionRequest(input: string): string;
  taskPrediction(response: string): string;
  estimateCost(input: string): string;
} =
  createRequire(import.meta.url)("./switchyard.node");

function abortError(): Error {
  return Object.assign(new Error("Routing cancelled"), { name: "AbortError", code: "ABORT_ERR" });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function publicError(error: unknown, cause?: unknown): Error {
  const match = /^(ERR_CONFIG|ERR_INVALID_REQUEST|ERR_ROUTING|ERR_METRICS|ABORT_ERR): (.*)$/s.exec(errorMessage(error));
  if (match?.[1] === "ABORT_ERR") return abortError();
  const code = match?.[1] ?? "ERR_ROUTING";
  return Object.assign(new Error(match?.[2] ?? errorMessage(error),code === "ERR_ROUTING" ? { cause } : undefined), { code });
}

function json(value: unknown, code = "ERR_INVALID_REQUEST"): string {
  try {
    const result = JSON.stringify(value);
    if (result === undefined) throw new Error("Expected a JSON value");
    return result;
  } catch (error) {
    throw new Error(`${code}: ${errorMessage(error)}`);
  }
}

/** Reusable composite router. The caller performs decision calls and stores branch state. */
export class CompositeRouter {
  readonly #native: NativeComposite;

  private constructor(options: CompositeOptions) {
    this.#native = new native.NativeComposite(json(options, "ERR_CONFIG"));
  }

  static create(options: CompositeOptions): CompositeRouter {
    try {
      return new CompositeRouter(options);
    } catch (error) {
      throw publicError(error);
    }
  }

  async route(messages: readonly RoutingMessage[], options: {
    state?: CompositeState;
    decide(request: DecisionRequest): Promise<DecisionResponse>;
    signal?: AbortSignal;
    metricContext?: MetricContext;
  }): Promise<Selection> {
    const { signal } = options;
    if (signal?.aborted) throw abortError();
    let run: NativeRun | undefined;
    let decisionError: unknown;
    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => {
        run?.close();
        reject(abortError());
      };
    });
    try {
      run = this.#native.start(
        json(messages), options.state === undefined ? null : json(options.state),
        options.metricContext === undefined ? null : json(options.metricContext),
      );
      signal?.addEventListener("abort", onAbort, { once: true });
      let response: string | null = null;
      let failure: string | null = null;
      for (;;) {
        const step = await Promise.race([run.next(response, failure), aborted]);
        if (signal?.aborted) throw abortError();
        if (step.kind === "done") {
          return step.state === null
            ? { model: step.model }
            : { model: step.model, state: JSON.parse(step.state) };
        }
        response = null;
        failure = null;
        let answer: DecisionResponse;
        try {
          answer = await Promise.race([options.decide(JSON.parse(step.request)), aborted]);
        } catch (error) {
          if (signal?.aborted) throw abortError();
          decisionError = error;
          failure = errorMessage(error);
          continue;
        }
        if (signal?.aborted) throw abortError();
        response = json(answer);
      }
    } catch (error) {
      if (signal?.aborted) throw abortError();
      throw publicError(error, decisionError);
    } finally {
      signal?.removeEventListener("abort", onAbort);
      run?.close();
    }
  }
}

/** Shared in-process Switchyard collector. No server is required.
 * The first instance installs the Rust meter provider. Retains up to 20,000 series per instrument.
 */
export class Metrics {
  readonly #native: NativeMetrics;

  constructor() {
    try { this.#native = new native.NativeMetrics(); }
    catch (error) { throw publicError(error); }
  }

  record(observation: CallMetrics): boolean {
    try { return this.#native.record(json(observation)); }
    catch (error) { throw publicError(error); }
  }

  stats(filter: MetricFilter = {}): MetricsSnapshot {
    try { return JSON.parse(this.#native.stats(json(filter))); }
    catch (error) { throw publicError(error); }
  }

  costs(filter: MetricFilter = {}): CostTotals {
    try { return JSON.parse(this.#native.costs(json(filter))); }
    catch (error) { throw publicError(error); }
  }

  prometheus(filter: MetricFilter = {}): string {
    try { return this.#native.prometheus(json(filter)); }
    catch (error) { throw publicError(error); }
  }
}

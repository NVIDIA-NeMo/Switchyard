// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";

/** Opaque routing state. Store it unchanged and pass it back on the same branch. */
export type CompositeState = { readonly [key: string]: unknown };

/** Opaque algorithm state. Carry it unchanged between requests on the same branch. */
export type RoutingState = CompositeState;

/** A model call in Switchyard's normalized protocol, with ordered candidates. */
export interface ModelCall {
  request: Readonly<Record<string, unknown>>;
  models: readonly string[];
}

/** A buffered response and the configured candidate that served it. */
export interface ModelReply {
  response: Readonly<Record<string, unknown>>;
  servedModel: string;
}

export interface RoutingOptions {
  /** Serve the model call and return a normalized, buffered response. */
  call?(call: ModelCall): Promise<ModelReply>;
  decide?(request: DecisionRequest): Promise<DecisionResponse>;
  state?: RoutingState;
  headers?: Readonly<Record<string, string>>;
  signal?: AbortSignal;
}

/** Complete routing result. A null response leaves the answer call to the host. */
export interface RoutingOutcome {
  selectedModelIds: readonly string[];
  request: Readonly<Record<string, unknown>>;
  metadata: Readonly<Record<string, unknown>> | null;
  response: Readonly<Record<string, unknown>> | null;
  state: RoutingState;
}

/** A configured algorithm whose model calls are handled by the host. */
export interface ConfiguredAlgorithm {
  route(request: Readonly<Record<string, unknown>>, options?: RoutingOptions): Promise<RoutingOutcome>;
}

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
  usage?: {
    input_tokens?: number | null;
    cached_input_tokens?: number | null;
    cache_creation_input_tokens?: number | null;
    output_tokens?: number | null;
    total_tokens?: number | null;
    reasoning_tokens?: number | null;
  };
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
  | { kind: "model"; request: string; models: string[] }
  | { kind: "done"; model: string; state: string | null; outcome: string };

type NativeOutcome = Extract<NativeStep, { kind: "done" }>;

interface NativeRun {
  next(response: string | null, error: string | null): Promise<NativeStep>;
  close(): void;
}

interface NativeComposite {
  start(messages: string, state: string | null): NativeRun;
}

interface NativeConfiguredAlgorithm {
  readonly id: string;
  start(request: string, state: string | null, headers: string | null): NativeRun;
}

interface NativeAlgorithmFactory {
  build(): NativeConfiguredAlgorithm[];
}

const native: {
  NativeComposite: new (options: string) => NativeComposite;
  NativeAlgorithmFactory: new (source: string) => NativeAlgorithmFactory;
} =
  createRequire(import.meta.url)("./switchyard.node");

function abortError(): Error {
  return Object.assign(new Error("Routing cancelled"), { name: "AbortError", code: "ABORT_ERR" });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function publicError(error: unknown, cause?: unknown): Error {
  const match = /^(ERR_CONFIG|ERR_INVALID_REQUEST|ERR_ROUTING|ABORT_ERR): (.*)$/s.exec(errorMessage(error));
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

async function driveRun(start: () => NativeRun, options: RoutingOptions): Promise<NativeOutcome> {
  const { signal } = options;
  if (signal?.aborted) throw abortError();
  let run: NativeRun | undefined;
  let callbackError: unknown;
  let onAbort!: () => void;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => {
      run?.close();
      reject(abortError());
    };
  });
  try {
    run = start();
    signal?.addEventListener("abort", onAbort, { once: true });
    let response: string | null = null;
    let failure: string | null = null;
    for (;;) {
      const step = await Promise.race([run.next(response, failure), aborted]);
      if (signal?.aborted) throw abortError();
      if (step.kind === "done") return step;
      let invoke: () => Promise<unknown>;
      if (step.kind === "decision") {
        const { decide } = options;
        if (decide === undefined) throw new Error("ERR_CONFIG: This algorithm requires a decide callback");
        const request: DecisionRequest = JSON.parse(step.request);
        invoke = () => decide.call(options, request);
      } else {
        const serve = options.call;
        if (serve === undefined) throw new Error("ERR_CONFIG: This algorithm requires a call callback");
        const request: ModelCall = { request: JSON.parse(step.request), models: step.models };
        invoke = () => serve.call(options, request);
      }
      response = null;
      failure = null;
      let answer: unknown;
      try {
        answer = await Promise.race([invoke(), aborted]);
      } catch (error) {
        if (signal?.aborted) throw abortError();
        callbackError = error;
        failure = errorMessage(error);
        continue;
      }
      if (signal?.aborted) throw abortError();
      response = json(answer);
    }
  } catch (error) {
    if (signal?.aborted) throw abortError();
    throw publicError(error, callbackError);
  } finally {
    signal?.removeEventListener("abort", onAbort);
    run?.close();
  }
}

function configuredAlgorithm(algorithm: NativeConfiguredAlgorithm): ConfiguredAlgorithm {
  return {
    async route(request, options = {}) {
      const step = await driveRun(() => algorithm.start(
        json(request),
        options.state === undefined ? null : json(options.state),
        options.headers === undefined ? null : json(options.headers),
      ), options);
      const { selected_model_ids: selectedModelIds, ...outcome } = JSON.parse(step.outcome);
      return { selectedModelIds, ...outcome };
    },
  };
}

/** Builds configured algorithms from deployment TOML without provider clients. */
export class AlgorithmFactory {
  readonly #native: NativeAlgorithmFactory;

  private constructor(source: string) {
    this.#native = new native.NativeAlgorithmFactory(source);
  }

  static fromToml(source: string): AlgorithmFactory {
    try {
      return new AlgorithmFactory(source);
    } catch (error) {
      throw publicError(error);
    }
  }

  /** Builds a reusable algorithm handle for each public route ID. */
  build(): Map<string, ConfiguredAlgorithm> {
    try {
      return new Map(this.#native.build().map((algorithm) => [algorithm.id, configuredAlgorithm(algorithm)]));
    } catch (error) {
      throw publicError(error);
    }
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
  }): Promise<Selection> {
    const step = await driveRun(() => this.#native.start(
      json(messages), options.state === undefined ? null : json(options.state),
    ), options);
    return step.state === null
      ? { model: step.model }
      : { model: step.model, state: JSON.parse(step.state) };
  }
}

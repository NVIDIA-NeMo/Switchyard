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
  | { kind: "done"; model: string; state: string | null };

interface NativeRun {
  next(response: string | null, error: string | null): Promise<NativeStep>;
  close(): void;
}

interface NativeComposite {
  start(messages: string, state: string | null): NativeRun;
}

const native: { NativeComposite: new (options: string) => NativeComposite } =
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
      run = this.#native.start(json(messages), options.state === undefined ? null : json(options.state));
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

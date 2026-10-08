// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";

export interface Decision {
  target: string;
  model: string;
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

interface Cancellation {
  cancel(): void;
}

interface NativeRunner {
  decide(routeId: string, messages: string, cancellation: Cancellation): Promise<Decision>;
}

const native: {
  NativeRunner: new (source: string) => NativeRunner;
  Cancellation: new () => Cancellation;
} = createRequire(import.meta.url)("./switchyard.node");

function abortError(): Error {
  return Object.assign(new Error("Routing cancelled"), { name: "AbortError", code: "ABORT_ERR" });
}

function publicError(cause: unknown): Error {
  const message = cause instanceof Error ? cause.message : String(cause);
  const match = /^(ERR_CONFIG|ERR_UNKNOWN_ROUTE|ERR_INVALID_REQUEST|ERR_ROUTING|ERR_UNSUPPORTED_OUTCOME|ABORT_ERR): (.*)$/s.exec(message);
  if (match?.[1] === "ABORT_ERR") return abortError();
  return Object.assign(new Error(match?.[2] ?? "Routing failed"), {
    code: match?.[1] ?? "ERR_ROUTING",
  });
}

/** A reusable model selector. The caller owns completion dispatch and session state. */
export class Runner {
  readonly #native: NativeRunner;

  private constructor(source: string) {
    this.#native = new native.NativeRunner(source);
  }

  /** Loads a model-selection deployment. Classifier credentials come from its environment settings. */
  static fromToml(source: string): Runner {
    try {
      return new Runner(source);
    } catch (error) {
      throw publicError(error);
    }
  }

  /** Selects a target from a conversation or a single user prompt. */
  async decide(routeId: string, input: string | readonly RoutingMessage[], options: { signal?: AbortSignal } = {}): Promise<Decision> {
    const signal = options.signal;
    if (signal?.aborted) throw abortError();
    const cancellation = new native.Cancellation();
    const onAbort = () => cancellation.cancel();
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
      if (signal?.aborted) throw abortError();
      const messages: readonly RoutingMessage[] = typeof input === "string"
        ? [{ role: "user", content: [{ type: "text", text: input }] }]
        : input;
      const result = await this.#native.decide(routeId, JSON.stringify(messages), cancellation);
      if (signal?.aborted) throw abortError();
      return result;
    } catch (error) {
      if (signal?.aborted) throw abortError();
      throw publicError(error);
    } finally {
      signal?.removeEventListener("abort", onAbort);
    }
  }
}

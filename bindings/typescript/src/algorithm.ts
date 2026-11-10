// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";
import type { DecisionRequest, DecisionResponse } from "./index.js";

type JsonObject = Record<string, unknown>;

export class LibsyError extends Error {
  readonly code = "ERR_ROUTING";
  override name = "LibsyError";
}

export class ContextWindowExceededError extends Error {
  override name = "ContextWindowExceededError";
}

export namespace LlmResponse {
  export class Agg {
    constructor(readonly response: JsonObject) {}
  }

  export class Stream {
    constructor(readonly stream: AsyncIterable<JsonObject>) {}
  }
}

export interface ModelCall {
  readonly algorithm: string;
  readonly request: JsonObject;
  readonly models: readonly string[];
  respond(work: PromiseLike<LlmResponse.Agg | LlmResponse.Stream>): Promise<void>;
  fail(error: Error): Promise<void>;
}

export interface DecisionCall {
  readonly algorithm: string;
  readonly request: DecisionRequest;
  readonly model: string;
  respond(work: PromiseLike<DecisionResponse>): Promise<void>;
  fail(error: Error): Promise<void>;
}

export interface OutcomeMetadata {
  readonly outcome_id: string;
  readonly algorithm: string;
  readonly evidence: unknown | null;
}

export interface RoutingOutcome {
  readonly selected_model_ids: readonly string[];
  readonly request: JsonObject;
  readonly fallback_request: JsonObject | null;
  readonly candidate_requests: Readonly<Record<string, JsonObject>>;
  readonly response: LlmResponse.Agg | LlmResponse.Stream | null;
  readonly metadata: OutcomeMetadata | null;
}

export interface TargetModel {
  readonly client: string;
  readonly id: string;
}

export namespace Step {
  export class CallModel {
    constructor(readonly call: ModelCall) {}
  }

  export class CallDecision {
    constructor(readonly call: DecisionCall) {}
  }

  export class Done {
    constructor(readonly outcome: RoutingOutcome) {}
  }
}

type AlgorithmStep = Step.CallModel | Step.CallDecision | Step.Done;

interface NativeReader<T> {
  next(): Promise<T | null>;
  close(): void;
}

interface NativeInput {
  push(json: string): Promise<void>;
  finish(message: string | null, isContextWindow: boolean): Promise<void>;
  closed(): Promise<void>;
}

interface NativeCall {
  respond(work: Promise<string>): Promise<void>;
  input(): NativeInput;
}

interface NativeStep {
  kind: "decision" | "model" | "done";
  data: string;
  call: NativeCall | null;
  stream: NativeReader<string> | null;
}

interface NativeConfigured {
  start(request: string, headers: string | null, options: string | null): NativeReader<NativeStep>;
  targetModels(): Record<string, TargetModel>;
}

interface NativeFactory {
  build(): Record<string, NativeConfigured>;
  buildConfigured(): Record<string, NativeConfigured>;
}

const native: { NativeAlgorithmFactory: new (source: string) => NativeFactory } =
  createRequire(import.meta.url)("./switchyard.node");

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function publicError(error: unknown): Error {
  const match = /^(ERR_CONFIG|ERR_INVALID_REQUEST|ERR_ROUTING|ABORT_ERR): (.*)$/s.exec(message(error));
  if (match?.[1] === "ABORT_ERR") {
    return Object.assign(new Error(match[2]), { name: "AbortError", code: "ABORT_ERR" });
  }
  if (!match || match[1] === "ERR_ROUTING") return new LibsyError(match?.[2] ?? message(error));
  return Object.assign(new Error(match[2]), { code: match[1] });
}

function json(value: unknown): string {
  const result = JSON.stringify(value);
  if (result === undefined) throw new TypeError("Expected a JSON value");
  return result;
}

function failure(error: unknown): string {
  return json({
    kind: "error", error: message(error),
    context_window_exceeded: error instanceof ContextWindowExceededError,
  });
}

async function pump(source: AsyncIterable<JsonObject>, input: NativeInput): Promise<void> {
  let iterator: AsyncIterator<JsonObject> | undefined;
  let isFinished = false;
  try {
    iterator = source[Symbol.asyncIterator]();
    const closed = input.closed().then(() => null);
    for (;;) {
      const item = await Promise.race([iterator.next(), closed]);
      if (item === null) return;
      if (item.done) break;
      await input.push(json(item.value));
    }
    isFinished = true;
    await input.finish(null, false);
  } catch (error) {
    await input.finish(message(error), error instanceof ContextWindowExceededError).catch(() => {});
  } finally {
    if (!isFinished) void Promise.resolve(iterator?.return?.()).catch(() => {});
  }
}

function reader<T, U>(native: NativeReader<T>, decode: (value: T) => U): AsyncIterableIterator<U> {
  let isClosed = false;
  let tail: Promise<void> = Promise.resolve();
  return {
    [Symbol.asyncIterator]() { return this; },
    next() {
      const result = tail.then(async (): Promise<IteratorResult<U>> => {
        if (isClosed) return { done: true, value: undefined };
        try {
          const value = await native.next();
          if (isClosed) return { done: true, value: undefined };
          if (value === null) {
            isClosed = true;
            native.close();
            return { done: true, value: undefined };
          }
          return { done: false, value: decode(value) };
        } catch (error) {
          if (isClosed) return { done: true, value: undefined };
          isClosed = true;
          native.close();
          throw publicError(error);
        }
      });
      tail = result.then(() => {}, () => {});
      return result;
    },
    async return() {
      isClosed = true;
      native.close();
      return { done: true, value: undefined };
    },
    async throw(error: unknown) {
      isClosed = true;
      native.close();
      throw error;
    },
  };
}

function callMethods(call: NativeCall, isModel: boolean) {
  let isComplete = false;
  async function respond(work: PromiseLike<unknown>): Promise<void> {
    if (isComplete) {
      void Promise.resolve(work).catch(() => {});
      throw new LibsyError("Call has already been completed");
    }
    isComplete = true;
    const reply = Promise.resolve(work).then((value) => {
      if (!isModel) return json({ kind: "decision", response: value });
      if (value instanceof LlmResponse.Agg) return json({ kind: "agg", response: value.response });
      if (value instanceof LlmResponse.Stream) {
        const input = call.input();
        void pump(value.stream, input).catch(() => {});
        return json({ kind: "stream" });
      }
      throw new TypeError("Expected LlmResponse.Agg or LlmResponse.Stream");
    }).catch(failure);
    try {
      await call.respond(reply);
    } catch (error) {
      throw publicError(error);
    }
  }
  return {
    respond,
    fail(error: Error): Promise<void> {
      if (!(error instanceof Error)) throw new TypeError("error must be an Error");
      return respond(Promise.reject(error));
    },
  };
}

function decodeStep(step: NativeStep): AlgorithmStep {
  const data = JSON.parse(step.data);
  if (step.kind === "model") {
    return new Step.CallModel(Object.freeze({ ...data, ...callMethods(step.call!, true) }));
  }
  if (step.kind === "decision") {
    return new Step.CallDecision(Object.freeze({ ...data, ...callMethods(step.call!, false) }));
  }
  if (data.response?.kind === "agg") data.response = new LlmResponse.Agg(data.response.response);
  else if (data.response?.kind === "stream") {
    data.response = new LlmResponse.Stream(reader(step.stream!, (json) => JSON.parse(json)));
  }
  return new Step.Done(Object.freeze(data));
}

let configured: (native: NativeConfigured) => ConfiguredAlgorithm;
let configuredRoute: (native: NativeConfigured) => ConfiguredRoute;

interface RouteRunOptions {
  headers?: Record<string, string>;
  requiredLabels?: readonly string[];
  taskId?: string;
}

export class AlgorithmFactory {
  readonly #native: NativeFactory;

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

  build(): Record<string, ConfiguredAlgorithm> {
    try {
      return Object.fromEntries(Object.entries(this.#native.build()).map(([id, route]) => [id, configured(route)]));
    } catch (error) {
      throw publicError(error);
    }
  }

  /** Build routes with target requirement checks and target lookup. */
  buildConfigured(): Record<string, ConfiguredRoute> {
    try {
      return Object.fromEntries(Object.entries(this.#native.buildConfigured()).map(([id, route]) => [id, configuredRoute(route)]));
    } catch (error) {
      throw publicError(error);
    }
  }
}

export class ConfiguredAlgorithm {
  readonly #native: NativeConfigured;

  protected constructor(native: NativeConfigured) {
    this.#native = native;
  }

  static {
    configured = (native) => new ConfiguredAlgorithm(native);
  }

  protected readTargetModels(): Readonly<Record<string, TargetModel>> {
    try {
      return this.#native.targetModels();
    } catch (error) {
      throw publicError(error);
    }
  }

  runStream(request: JsonObject, options: {
    headers?: Record<string, string>;
  } = {}): AsyncIterableIterator<AlgorithmStep> {
    return this.start(request, options);
  }

  protected start(request: JsonObject, options: RouteRunOptions): AsyncIterableIterator<AlgorithmStep> {
    try {
      if (options.requiredLabels !== undefined && !Array.isArray(options.requiredLabels)) {
        throw new Error("ERR_INVALID_REQUEST: requiredLabels must be an array of strings");
      }
      if (options.taskId !== undefined && typeof options.taskId !== "string") {
        throw new Error("ERR_INVALID_REQUEST: taskId must be a string");
      }
      const requirements = options.requiredLabels === undefined && options.taskId === undefined
        ? null : json({ required_labels: options.requiredLabels, task_id: options.taskId });
      return reader(this.#native.start(
        json(request), options.headers === undefined ? null : json(options.headers), requirements,
      ), decodeStep);
    } catch (error) {
      throw publicError(error);
    }
  }
}

/** A configured route that checks target requirements before routing. */
export class ConfiguredRoute extends ConfiguredAlgorithm {
  private constructor(native: NativeConfigured) {
    super(native);
  }

  static {
    configuredRoute = (native) => new ConfiguredRoute(native);
  }

  /** Resolve call targets to configured clients and provider IDs, not an eligible pool. */
  targetModels(): Readonly<Record<string, TargetModel>> {
    return this.readTargetModels();
  }

  /** Run with host-supplied requirements and a trusted task ID. */
  override runStream(
    request: JsonObject, options: RouteRunOptions = {},
  ): AsyncIterableIterator<AlgorithmStep> {
    return this.start(request, options);
  }
}

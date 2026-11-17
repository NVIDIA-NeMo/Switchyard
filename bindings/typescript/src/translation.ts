// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";

/** Built-in provider JSON formats. */
export enum WireFormat {
  OPENAI_CHAT = "openai_chat",
  OPENAI_RESPONSES = "openai_responses",
  ANTHROPIC_MESSAGES = "anthropic_messages",
}

/** Translation failure with a stable kind and variant-specific details. */
export class TranslationError extends Error {
  constructor(message: string, readonly kind: string, readonly details: Record<string, unknown>) {
    super(message);
    this.name = "TranslationError";
  }
}

interface NativeTranslationEngine {
  decodeRequest(source: string, body: string): string;
  encodeRequest(target: string, request: string): string;
  decodeResponse(source: string, body: string): string;
  encodeResponse(target: string, response: string, requestExtensions: string | null): string;
  translateRequest(source: string, target: string, body: string): string;
  translateResponse(source: string, target: string, body: string, requestExtensions: string | null): string;
}

const native: { NativeTranslationEngine: new () => NativeTranslationEngine } =
  createRequire(import.meta.url)("./switchyard.node");

function json(value: Record<string, unknown>): string {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Expected a JSON object");
  }
  return JSON.stringify(value);
}

/** Buffered provider JSON translation using the default policy. Calls retain no request context. */
export class TranslationEngine {
  readonly #native = new native.NativeTranslationEngine();

  /** Decode a provider request into normalized JSON and diagnostics. */
  decodeRequest(source: WireFormat | string, body: Record<string, unknown>): {
    request: Record<string, unknown>; diagnostics: unknown[];
  } {
    return this.#call(() => this.#native.decodeRequest(source, json(body)));
  }

  /** Encode normalized request JSON into a provider body and diagnostics. */
  encodeRequest(target: WireFormat | string, request: Record<string, unknown>): {
    body: Record<string, unknown>; diagnostics: unknown[];
  } {
    return this.#call(() => this.#native.encodeRequest(target, json(request)));
  }

  /** Decode a complete provider response into normalized JSON and diagnostics. */
  decodeResponse(source: WireFormat | string, body: Record<string, unknown>): {
    response: Record<string, unknown>; diagnostics: unknown[];
  } {
    return this.#call(() => this.#native.decodeResponse(source, json(body)));
  }

  /** Encode a response, optionally restoring tool details from request extensions. */
  encodeResponse(
    target: WireFormat | string,
    response: Record<string, unknown>,
    requestExtensions?: Record<string, unknown> | null,
  ): { body: Record<string, unknown>; diagnostics: unknown[] } {
    return this.#call(() => this.#native.encodeResponse(
      target, json(response), requestExtensions == null ? null : json(requestExtensions),
    ));
  }

  /** Translate a provider request, returning its body and diagnostics. */
  translateRequest(source: WireFormat | string, target: WireFormat | string, body: Record<string, unknown>): {
    body: Record<string, unknown>; diagnostics: unknown[];
  } {
    return this.#call(() => this.#native.translateRequest(source, target, json(body)));
  }

  /** Translate a response, optionally restoring tool details from request extensions. */
  translateResponse(
    source: WireFormat | string,
    target: WireFormat | string,
    body: Record<string, unknown>,
    requestExtensions?: Record<string, unknown> | null,
  ): { body: Record<string, unknown>; diagnostics: unknown[] } {
    return this.#call(() => this.#native.translateResponse(
      source, target, json(body), requestExtensions == null ? null : json(requestExtensions),
    ));
  }

  #call<T>(run: () => string): T {
    try {
      return JSON.parse(run());
    } catch (error) {
      if (error instanceof Error && error.message.startsWith("ERR_TRANSLATION: ")) {
        const { message, kind, details } = JSON.parse(error.message.slice("ERR_TRANSLATION: ".length));
        throw new TranslationError(message, kind, details);
      }
      if (error instanceof Error && error.message.startsWith("ERR_INVALID_REQUEST: ")) {
        throw Object.assign(new Error(error.message.slice("ERR_INVALID_REQUEST: ".length)), { code: "ERR_INVALID_REQUEST" });
      }
      throw error;
    }
  }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";
import type { DecisionRequest, DecisionResponse } from "./index.js";

export interface DllmOptions {
  library: string;
  tokenizer_library: string;
  weights: string;
  tokenizer: string;
}

interface NativeEngine {
  decide(request: string): Promise<string>;
  close(): Promise<void>;
}
const native: { NativeDllmEngine: { load(options: string): Promise<NativeEngine> } } =
  createRequire(import.meta.url)("./switchyard.node");

/** Deciffusion decisions on a dedicated native thread in this process. */
export class DllmEngine {
  private constructor(private readonly engine: NativeEngine) {}

  static async load(options: DllmOptions): Promise<DllmEngine> {
    return new DllmEngine(await native.NativeDllmEngine.load(JSON.stringify(options)));
  }

  async decide(request: DecisionRequest, options: { signal?: AbortSignal } = {}): Promise<DecisionResponse> {
    options.signal?.throwIfAborted();
    // The owner thread drains submitted work even if its caller stops waiting.
    const work = this.engine.decide(JSON.stringify(request)).then((response) => JSON.parse(response) as DecisionResponse);
    const signal = options.signal;
    if (!signal) return work;
    let abort: () => void = () => {};
    const cancelled = new Promise<never>((_, reject) => {
      abort = () => reject(signal.reason ?? new DOMException("Cancelled", "AbortError"));
      signal.addEventListener("abort", abort, { once: true });
      if (signal.aborted) abort();
    });
    try { return await Promise.race([work, cancelled]); }
    finally { signal.removeEventListener("abort", abort); }
  }

  async close(): Promise<void> { await this.engine.close(); }
}

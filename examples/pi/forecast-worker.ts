// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

interface ForecastWorkerOptions {
  python?: string;
  researchRoot?: string;
  tokenCastRoot?: string;
  command?: string[];
}

export interface HorizonForecast {
  next_call: { tokens: number; requirements: unknown | null };
  remaining_task: { tokens: number; requirements: unknown | null };
  checkpoint: string;
  completed_calls: number;
  inference_ms: number;
}

/** One model load per Pi process; each line contains only source-visible state. */
export class ForecastWorker {
  private options: ForecastWorkerOptions;
  private child?: ChildProcessWithoutNullStreams;
  private sequence = 0;
  private pending = new Map<number, { resolve: (v: HorizonForecast) => void; reject: (e: Error) => void }>();

  constructor(options: ForecastWorkerOptions = {}) { this.options = options; }

  close() {
    this.child?.kill();
    this.child = undefined;
    for (const entry of this.pending.values()) entry.reject(new Error("Forecast worker stopped"));
    this.pending.clear();
  }

  private start() {
    if (this.child) return this.child;
    const [command, ...args] = this.options.command ?? [this.options.python ?? process.env.SY_FORECAST_PYTHON ?? "python3",
      "-u", fileURLToPath(new URL("./forecast/lightgbm_worker.py", import.meta.url))];
    if (!command) throw new Error("Forecast worker command cannot be empty");
    const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"], env: {
      ...process.env, OMP_NUM_THREADS: "2", OPENBLAS_NUM_THREADS: "1",
      ...(this.options.researchRoot ? { SY_FORECAST_RESEARCH_ROOT: this.options.researchRoot } : {}),
      ...(this.options.tokenCastRoot ? { SY_FORECAST_TOKENCAST_ROOT: this.options.tokenCastRoot } : {}),
    } });
    this.child = child;
    let stderr = "";
    child.stderr.on("data", (data) => { stderr = (stderr + data).slice(-2000); });
    createInterface({ input: child.stdout }).on("line", (line) => {
      try {
        const value = JSON.parse(line);
        const pending = this.pending.get(value.id);
        if (!pending) return;
        this.pending.delete(value.id);
        if (value.error) pending.reject(new Error(value.error));
        else if (![value.result?.next_call?.tokens, value.result?.remaining_task?.tokens]
          .every(tokens => typeof tokens === "number" && Number.isFinite(tokens) && tokens >= 0)) {
          pending.reject(new Error("Forecast worker returned invalid token predictions"));
        }
        else pending.resolve(value.result);
      } catch { /* Non-protocol output cannot become a forecast. */ }
    });
    const fail = (error: Error) => {
      if (this.child !== child) return;
      this.child = undefined;
      for (const entry of this.pending.values()) entry.reject(error);
      this.pending.clear();
    };
    child.on("error", fail);
    child.on("exit", () => fail(new Error(`Forecast worker exited: ${stderr}`)));
    child.stdin.on("error", fail);
    return child;
  }

  predict(input: object, signal?: AbortSignal): Promise<HorizonForecast> {
    if (signal?.aborted) return Promise.reject(new Error("Forecast cancelled"));
    const child = this.start();
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.close(); }, 15000);
      const cleanup = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); };
      const abort = () => { cleanup(); this.pending.delete(id); reject(new Error("Forecast cancelled")); };
      signal?.addEventListener("abort", abort, { once: true });
      this.pending.set(id, {
        resolve: (value) => { cleanup(); resolve(value); },
        reject: (error) => { cleanup(); reject(error); },
      });
      child.stdin.write(JSON.stringify({ ...input, id }) + "\n");
    });
  }
}

/** Total observed provider input + output; unknown usage stays unavailable. */
export function observedTokens(messages: readonly { role: string; usage?: { input: number; output: number; cacheRead: number; cacheWrite: number } }[]): number | undefined {
  let total = 0;
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const usage = message.usage;
    if (!usage) return undefined;
    const channels = [usage.input, usage.output, usage.cacheRead, usage.cacheWrite];
    if (!channels.every((value) => Number.isFinite(value) && value >= 0)) return undefined;
    total += channels.reduce((sum, value) => sum + value, 0);
  }
  return total;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { Runner } from "@switchyard/runner";
import type { Message } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ModelRoute, ModelRouteRequest } from "@earendil-works/pi-coding-agent";

const TARGETS = {
  complex: { provider: "openai-codex", id: "gpt-5.6-sol" },
  standard: { provider: "openai-codex", id: "gpt-5.6-terra" },
  implementation: { provider: "openai-codex", id: "gpt-5.6-luna" },
} as const;
type Target = keyof typeof TARGETS;
interface State {
  phase: "planning" | "implementation";
  target: Target;
}

function routeTo(request: ModelRouteRequest<State>, ctx: ExtensionContext, target: Target, state?: State): ModelRoute<State> {
  if (!Object.hasOwn(TARGETS, target)) throw new Error(`Unknown Switchyard target: ${target}`);
  const identity = TARGETS[target];
  const model = ctx.modelRegistry.find(identity.provider, identity.id);
  if (!model) throw new Error(`Model ${identity.provider}/${identity.id} is not in the catalog`);
  return { model, thinkingLevel: request.thinkingLevel, state };
}

function lastUserText(messages: readonly Message[]): string {
  const content = messages.filter((message) => message.role === "user").at(-1)?.content ?? "";
  if (typeof content === "string") return content;
  return content.flatMap((block) => block.type === "text" ? [block.text] : []).join("\n");
}

function editedThisTurn(messages: readonly Message[]): boolean {
  const lastUser = messages.findLastIndex((message) => message.role === "user");
  return messages.slice(lastUser + 1).some((message) => message.role === "toolResult"
    && (message.toolName === "edit" || message.toolName === "write") && !message.isError);
}

export default function (pi: ExtensionAPI) {
  let runner: Runner | undefined;
  pi.registerVirtualModel<State>({
    provider: "switchyard",
    id: "auto",
    name: "Auto (Switchyard)",
    thinkingLevels: ["low", "medium", "high", "xhigh"],
    contextWindow: 272_000,
    maxTokens: 128_000,
    async route(request, ctx) {
      if (request.signal?.aborted) throw Object.assign(new Error("Routing cancelled"), { name: "AbortError", code: "ABORT_ERR" });
      if (request.reason === "direct") return routeTo(request, ctx, "implementation");
      if (request.state) {
        if (request.state.phase === "planning" && editedThisTurn(request.messages)) {
          return routeTo(request, ctx, "implementation", { phase: "implementation", target: "implementation" });
        }
        return routeTo(request, ctx, request.state.target);
      }
      const previous = request.previous?.model;
      for (const target of ["complex", "standard"] as const) {
        const identity = TARGETS[target];
        if (previous?.provider === identity.provider && previous.id === identity.id) {
          return routeTo(request, ctx, target, { phase: "planning", target });
        }
      }
      runner ??= Runner.fromToml(readFileSync(process.env.SWITCHYARD_CONFIG ?? new URL("./switchyard.toml", import.meta.url), "utf8"));
      let target: Target = "standard";
      try {
        // Bound classifier input for large pasted prompts; 16,000 follows Pi's Jev example.
        const decision = await runner.decide("switchyard/planner", lastUserText(request.messages).slice(0, 16_000), { signal: request.signal });
        if (decision.target !== "complex" && decision.target !== "standard") {
          throw new Error(`Unknown planning target: ${decision.target}`);
        }
        if (decision.model !== TARGETS[decision.target].id) throw new Error(`Model mapping differs for target: ${decision.target}`);
        target = decision.target;
      } catch (error) {
        if (!(error instanceof Error) || !("code" in error) || error.code !== "ERR_ROUTING") throw error;
      }
      return routeTo(request, ctx, target, { phase: "planning", target });
    },
  });
}

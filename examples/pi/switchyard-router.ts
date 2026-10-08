// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { parse, stringify } from "smol-toml";
import { Runner, type RoutingContent, type RoutingMessage } from "@switchyard/runner";
import type { Message } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

function modelIdentity(reference: unknown) {
  if (typeof reference !== "string") throw new Error("Expected a model in provider/model form");
  // Model IDs may themselves contain slashes, as on OpenRouter.
  const slash = reference.indexOf("/");
  if (slash <= 0 || slash === reference.length - 1) throw new Error("Expected a model in provider/model form");
  return { reference, provider: reference.slice(0, slash), id: reference.slice(slash + 1) };
}

function loadRunner(): Runner {
  const algorithm = parse(readFileSync(process.env.SWITCHYARD_CONFIG ?? new URL("./switchyard.toml", import.meta.url), "utf8"));
  const targets = Object.fromEntries([algorithm.efficient_target, algorithm.capable_target].map((value) => {
    const { reference } = modelIdentity(value);
    return [reference, { id: reference, llm_client: "pi" }];
  }));
  // Keep the provider in the routing identity so equal model IDs on different providers stay distinct.
  return Runner.fromToml(stringify({
    schema_version: 1,
    llm_clients: { pi: { format: "openai_chat", base_url: "http://127.0.0.1:1/v1" } },
    targets,
    routes: { auto: { ...algorithm, id: "switchyard/auto" } },
  }));
}

function routingMessages(messages: readonly Message[]): RoutingMessage[] {
  return messages.map((message) => {
    const blocks = typeof message.content === "string"
      ? [{ type: "text" as const, text: message.content }] : message.content;
    // Route on visible text and tool activity; Pi retains the original media and reasoning.
    const content = blocks.flatMap<RoutingContent>((block) => {
      if (block.type === "text") return [{ type: "text", text: block.text }];
      if (block.type === "toolCall") {
        return [{ type: "tool_call", id: block.id, name: block.name, arguments: block.arguments }];
      }
      return [];
    });
    if (message.role === "toolResult") {
      return { role: "tool", content: [{
        type: "tool_result", tool_call_id: message.toolCallId, content, is_error: message.isError,
      }] };
    }
    if (message.role === "system") {
      for (const text of Object.values(message.sections ?? {})) {
        if (text) content.push({ type: "text", text });
      }
    }
    return { role: message.role, content };
  });
}

export default function (pi: ExtensionAPI) {
  let runner: Runner | undefined;
  pi.registerVirtualModel({
    provider: "switchyard",
    id: "auto",
    name: "Auto (Switchyard)",
    thinkingLevels: ["medium"],
    async route(request, ctx) {
      runner ??= loadRunner();
      const decision = await runner.decide("switchyard/auto", routingMessages(request.messages), { signal: request.signal });
      const identity = modelIdentity(decision.model);
      const model = ctx.modelRegistry.find(identity.provider, identity.id);
      if (!model) throw new Error(`Model ${identity.reference} is not in the catalog`);
      return { model, thinkingLevel: "medium" };
    },
  });
}

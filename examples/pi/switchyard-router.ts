// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  CompositeRouter, DllmEngine, type DllmOptions, type CompositeOptions, type CompositeState, type DecisionRequest,
  type DecisionResponse, type RoutingContent, type RoutingMessage,
} from "@switchyard/libsy";
import type {
  ClassifierApi, ClassifierContext, ClassifierModel, ClassifierResult, JsonValue, Message,
} from "@earendil-works/pi-ai";
import { getAgentDir, type ContextWithSystemEvent, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PLANNING_PROMPT = "You are in the planning phase. Inspect the task and the relevant code, then form a concrete implementation plan before modifying any files. Use read-only tools as needed. Do not make an edit until the plan is complete. After completing the plan, immediately make the first edit in the same task to hand execution off. Do not finish or return a final response with only the plan.";

const POLICY = {
  classifier: {
    classify_trigger: "user_turn",
    fail_open: false,
    cutoff: 0.6,
  },
  stage: {
    efficient_first: { threshold: 0.4, recent_window: 3, capable_hold_turns: 2, efficient_hold_turns: 0 },
    // After the first switch to efficient, stay there until the next user turn.
    capable_first: {
      threshold: 0.0, recent_window: 3, capable_hold_turns: 0, efficient_hold_turns: -1,
      capable_system_prompt: PLANNING_PROMPT,
    },
  },
} satisfies Omit<CompositeOptions, "targets">;

function modelIdentity(reference: unknown) {
  if (typeof reference !== "string") throw new Error("Expected a model in provider/model form");
  // Model IDs may themselves contain slashes, as on OpenRouter.
  const slash = reference.indexOf("/");
  if (slash <= 0 || slash === reference.length - 1) throw new Error("Expected a model in provider/model form");
  return { reference, provider: reference.slice(0, slash), id: reference.slice(slash + 1) };
}

function loadRouter() {
  const path = join(getAgentDir(), "switchyard-router.json");
  let classifier, capable, efficient;
  let embedded: DllmOptions | undefined;
  try {
    const config = JSON.parse(readFileSync(path, "utf8"));
    embedded = config.dllm;
    classifier = modelIdentity(config.decision_model ?? (embedded ? "dllm/deciffusion" : undefined));
    capable = modelIdentity(config.capable_model);
    efficient = modelIdentity(config.efficient_model);
  } catch (cause) {
    throw new Error(`Configure ${path} with capable_model and efficient_model in provider/model form, plus dllm paths or decision_model. See README for full example. ${(cause as Error).message}`, { cause });
  }
  return { classifier, embedded, router: CompositeRouter.create({
    ...POLICY,
    targets: { classifier: classifier.reference, capable: capable.reference, efficient: efficient.reference },
  }) };
}

function classifierContext(request: DecisionRequest): ClassifierContext {
  const questions: ClassifierContext["questions"] = {};
  for (const [id, question] of Object.entries(request.questions)) {
    if (question.kind.type !== "choice") throw new Error(`Expected a choice question: ${id}`);
    questions[id] = {
      type: "choice",
      instructions: question.instructions,
      criteria: Object.fromEntries(question.kind.data.options.map((option) => [option.id, option.description])),
    };
  }
  // RoutingMessage is plain JSON.
  return { state: { input: request.input as unknown as JsonValue }, questions };
}

function decisionResponse(request: DecisionRequest, result: ClassifierResult): DecisionResponse {
  if (result.stopReason === "aborted") {
    throw Object.assign(new Error(result.errorMessage ?? "Classification cancelled"), { name: "AbortError", code: "ABORT_ERR" });
  }
  if (result.stopReason === "error") throw new Error(result.errorMessage ?? "Classification failed");
  const answers = Object.fromEntries(Object.entries(request.questions).map(([id, question]) => {
    const answer = result.answers[id];
    if (question.kind.type !== "choice" || answer?.type !== "choice"
      || !question.kind.data.options.some((option) => option.id === answer.choice)) {
      throw new Error(`Expected a declared choice for question: ${id}`);
    }
    return [id, {
      value: { type: "choice" as const, data: { selected: answer.choice, probabilities: answer.probabilities } },
      provider_confidence: answer.confidence,
    }];
  }));
  return { model: `${result.provider}/${result.model}`, answers };
}

// Routing prompts go first, as libsy places them ahead of the request's own instructions.
function prependSystemPrompt(messages: ContextWithSystemEvent["messages"], prompt: string) {
  const [first, ...rest] = messages;
  if (first?.role !== "system") return [{ role: "system" as const, content: prompt, timestamp: Date.now() }, ...messages];
  // Pi's structured prompt keeps its text in sections and leaves content empty.
  const content = typeof first.content === "string"
    ? (first.content ? [{ type: "text" as const, text: first.content }] : [])
    : first.content;
  return [{ ...first, content: [{ type: "text" as const, text: prompt }, ...content] }, ...rest];
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
  let loaded: ReturnType<typeof loadRouter> | undefined;
  let engine: Promise<DllmEngine> | undefined;
  function initialize() {
    const config = loaded ??= loadRouter();
    if (config.embedded) engine ??= DllmEngine.load(config.embedded);
    return config;
  }
  pi.on("session_start", async () => {
    initialize();
    await engine;
  });
  pi.on("session_shutdown", async () => {
    if (engine) await (await engine).close();
  });
  let classifierModel: ClassifierModel<ClassifierApi> | undefined;
  // Pi builds the request context after routing, so the routed prompt waits for that event.
  let pendingPrompt: string | undefined;
  pi.on("context_with_system", (event) => {
    const prompt = pendingPrompt;
    pendingPrompt = undefined;
    return prompt === undefined ? undefined : { messages: prependSystemPrompt(event.messages, prompt) };
  });
  pi.registerVirtualModel<CompositeState>({
    provider: "switchyard",
    id: "auto",
    name: "Auto (Switchyard)",
    thinkingLevels: ["medium"],
    async route(request, ctx) {
      // Direct requests, such as compaction summaries, skip the agent loop's context event.
      if (request.reason !== "direct") pendingPrompt = undefined;
      const { router, classifier } = initialize();
      const local = engine ? await engine : undefined;
      const judge = local ? undefined : classifierModel ??= ctx.modelRegistry.findOfType("classifier", classifier.provider, classifier.id);
      if (!local && !judge) throw new Error(`Unknown classifier: ${classifier.reference}`);
      const selection = await router.route(routingMessages(request.messages), {
        state: request.state,
        signal: request.signal,
        decide: async (decision) => local
          ? local.decide(decision, { signal: request.signal })
          : decisionResponse(decision, await ctx.modelRegistry.classify(
            judge!, classifierContext(decision), { signal: request.signal },
          )),
      }).catch((error) => {
        // The binding wraps classifier errors as routing failures; preserve cancellation for Pi.
        if (error.cause?.name === "AbortError") throw error.cause;
        throw error;
      });
      const identity = modelIdentity(selection.model);
      const model = ctx.modelRegistry.find(identity.provider, identity.id);
      if (!model) throw new Error(`Model ${identity.reference} is not in the catalog`);
      if (request.reason !== "direct") pendingPrompt = selection.systemPrompt;
      return { model, thinkingLevel: "medium", state: selection.state };
    },
  });
}

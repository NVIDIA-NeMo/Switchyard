// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  CompositeRouter, type CompositeOptions, type CompositeState, type DecisionRequest,
  type DecisionResponse, type RoutingContent, type RoutingMessage, taskPrediction,
} from "@switchyard/libsy";
import type {
  ClassifierApi, ClassifierContext, ClassifierModel, ClassifierResult, Message,
} from "@earendil-works/pi-ai";
import { getAgentDir, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerMetrics, tokenUsage } from "./metrics.ts";
import { predictionRequest } from "./prediction.ts";

const POLICY = {
  classifier: {
    classify_trigger: "user_turn",
    fail_open: false,
    decision: {
      cutoff: 0.4,
      candidates: { a: "capable", b: "efficient" },
      evidence: {
        description: "Candidate a is the larger reasoning-oriented coding and thinking model. It is intended for complex problems. Candidate b is the smaller fast coding model. It is intended for simple questions, simple local reads and edits, routine tests and straightforward fixes. These are qualitative deployment roles, not empirical success rates.",
      },
    },
  },
  stage: { efficient_first_threshold: 0.5, capable_first_threshold: 0.2 },
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
  try {
    const config = JSON.parse(readFileSync(path, "utf8"));
    classifier = modelIdentity(config.decision_model);
    capable = modelIdentity(config.capable_model);
    efficient = modelIdentity(config.efficient_model);
  } catch (cause) {
    throw new Error(`Configure ${path} with decision_model, capable_model and efficient_model in provider/model form. See README for full example. ${(cause as Error).message}`, { cause });
  }
  return { classifier, router: CompositeRouter.create({
    ...POLICY,
    targets: { classifier: classifier.reference, capable: capable.reference, efficient: efficient.reference },
  }) };
}

function text(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value ?? null);
}

function classifierContext(request: DecisionRequest): ClassifierContext {
  const questions: ClassifierContext["questions"] = {};
  for (const [id, question] of Object.entries(request.questions)) {
    if (question.kind.type !== "choice") throw new Error(`Expected a choice question: ${id}`);
    questions[id] = {
      type: "choice",
      instructions: text(question.instructions),
      criteria: Object.fromEntries(question.kind.data.options.map((option) => [option.id, text(option.description)])),
    };
  }
  return { state: request.context as ClassifierContext["state"], questions };
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
  return { model: `${result.provider}/${result.model}`, answers,
    ...(result.usage ? { usage: tokenUsage(result.usage) } : {}) };
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
  const metrics = registerMetrics(pi);
  let loaded: ReturnType<typeof loadRouter> | undefined;
  let classifierModel: ClassifierModel<ClassifierApi> | undefined;
  pi.registerVirtualModel<CompositeState>({
    provider: "switchyard",
    id: "auto",
    name: "Auto (Switchyard)",
    thinkingLevels: ["medium"],
    async route(request, ctx) {
      const measurement = metrics.begin(request.reason, ctx);
      try {
        const { router, classifier } = loaded ??= loadRouter();
        const judge = classifierModel ??= ctx.modelRegistry.findOfType("classifier", classifier.provider, classifier.id);
        if (!judge) throw new Error(`Unknown classifier: ${classifier.reference}`);
        const selection = await router.route(routingMessages(request.messages), {
          state: request.state,
          signal: request.signal,
          metricContext: measurement?.context,
          decide: async (decision) => {
            const classify = () => ctx.modelRegistry.classify(judge, classifierContext(decision), { signal: request.signal });
            return decisionResponse(decision, measurement
              ? await measurement.classify(classifier.reference, classify) : await classify());
          },
        }).catch((error) => {
          // The binding wraps classifier errors as routing failures; preserve cancellation for Pi.
          if (error.cause?.name === "AbortError") throw error.cause;
          throw error;
        });
        const identity = modelIdentity(selection.model);
        const model = ctx.modelRegistry.find(identity.provider, identity.id);
        if (!model) throw new Error(`Model ${identity.reference} is not in the catalog`);
        if (measurement) {
          try {
            const branchMessages = ctx.sessionManager.getBranch().flatMap((entry) => entry.type === "message"
              && (entry.message.role === "assistant" || entry.message.role === "user" || entry.message.role === "toolResult") ? [entry.message] : []);
            const prediction = predictionRequest(classifier.reference, branchMessages.length ? branchMessages : request.messages);
            const result = await measurement.classify(classifier.reference,
              () => ctx.modelRegistry.classify(judge, classifierContext(prediction), { signal: request.signal }));
            const used = (prediction.context as { token_usage: { tokens_used_so_far: number } }).token_usage.tokens_used_so_far;
            measurement.prediction(taskPrediction(decisionResponse(prediction, result)), used);
          } catch {
            measurement.prediction(null);
          }
          if (request.signal?.aborted) throw Object.assign(new Error("Prediction cancelled"), { name: "AbortError", code: "ABORT_ERR" });
        }
        return { model, thinkingLevel: "medium", state: selection.state };
      } catch (error) {
        measurement?.fail();
        throw error;
      }
    },
  });
}

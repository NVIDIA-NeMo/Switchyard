// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Runner, RunnerDecideOptions, RunnerDecision, RoutingRequest,
} from "../src/index.js";

declare const runner: Runner;
declare const request: RoutingRequest;
const options: RunnerDecideOptions = { requiredLabels: [], taskId: "renewal-42" };
const decision: Promise<RunnerDecision> = runner.decide(request, options);
const candidates: Promise<Record<string, RoutingRequest>> = decision.then(
  (outcome) => outcome.candidateRequests,
);

// @ts-expect-error Requirements are a collection, not a single label.
runner.decide(request, { requiredLabels: "private" });
// @ts-expect-error The trusted task identity must be a string.
runner.decide(request, { taskId: 42 });

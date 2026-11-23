// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const routers = new Map();

// The binding is an ES module, so CommonJS code loads it with a dynamic import.
const loadLibsy = () => import('@switchyard/libsy');

// This is the "auto" preset written as a stage_router route, because "auto"
// does not accept a threshold. The host never calls the targets, so their IDs
// are placeholders.
function toml(threshold) {
  return `schema_version = 1

[llm_clients.vscode]
format = "openai_chat"
base_url = "https://unused.invalid/v1"

[targets]
capable = { id = "capable", llm_client = "vscode" }
efficient = { id = "efficient", llm_client = "vscode" }

[routes.switchyard]
id = "switchyard"
type = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = ${threshold}
recent_turn_window = 3
capable_hold_turns = 2
`;
}

// Returns "capable" or "efficient". Routers are reused so that the capable
// hold carries over between requests in the same session.
async function selectTier(threshold, messages, sessionId) {
  const { AlgorithmFactory, Step } = await loadLibsy();
  if (!routers.has(threshold)) routers.set(threshold, AlgorithmFactory.fromToml(toml(threshold)).build().switchyard);
  const headers = sessionId ? { 'x-switchyard-session-id': sessionId } : {};
  for await (const step of routers.get(threshold).runStream({ model: 'switchyard', messages }, { headers })) {
    if (!(step instanceof Step.Done)) throw new Error('Switchyard asked for a model call, which this router does not make.');
    return step.outcome.selected_model_ids[0];
  }
  throw new Error('Switchyard returned no routing decision.');
}

module.exports = { loadLibsy, selectTier };

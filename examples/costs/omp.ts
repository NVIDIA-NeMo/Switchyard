// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { updateCost } from "./display.ts";

// OMP sends its session header on anthropic-messages requests.
export default function(omp: ExtensionAPI): void {
  omp.on("session_start", (_event, ctx) => updateCost(ctx));
  omp.on("session_switch", (_event, ctx) => updateCost(ctx));
  omp.on("model_select", (_event, ctx) => updateCost(ctx));
  omp.on("agent_end", (_event, ctx) => updateCost(ctx));
}

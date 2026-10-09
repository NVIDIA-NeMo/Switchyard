// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { updateCost } from "./display.ts";

export default function(pi: ExtensionAPI): void {
  pi.on("before_provider_headers", (event, ctx) => {
    if (ctx.model?.provider === "switchyard") {
      event.headers["x-switchyard-session-id"] = ctx.sessionManager.getSessionId();
    }
  });
  pi.on("session_start", (_event, ctx) => updateCost(ctx));
  pi.on("session_switch", (_event, ctx) => updateCost(ctx));
  pi.on("model_select", (_event, ctx) => updateCost(ctx));
  pi.on("agent_end", (_event, ctx) => updateCost(ctx));
}

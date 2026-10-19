// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

interface Totals {
  known_usd: number;
  total_usd: number | null;
  unknown_calls: number;
  estimated_calls: number;
}

interface CostContext {
  hasUI: boolean;
  model?: { provider: string };
  sessionManager: { getSessionId(): string };
  ui: { setStatus(key: string, text: string | undefined): void };
}

// Both agents can display a separate footer without rewriting their saved usage.
export async function updateCost(ctx: CostContext): Promise<void> {
  if (!ctx.hasUI) return;
  if (ctx.model?.provider !== "switchyard") {
    ctx.ui.setStatus("switchyard-cost", undefined);
    return;
  }
  const sessionId = ctx.sessionManager.getSessionId();
  const base = (process.env.SWITCHYARD_BASE_URL ?? "http://localhost:4000").replace(/\/$/, "").replace(/\/v1$/, "");
  try {
    const response = await fetch(`${base}/v1/routing/session-stats?${new URLSearchParams({ session_id: sessionId })}`, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json() as { cost: Totals; routing_cost: Totals };
    const cost = snapshot.cost;
    const text = cost.total_usd === null
      ? `$${cost.known_usd.toFixed(6)} known + ${cost.unknown_calls} unpriced calls`
      : `$${cost.total_usd.toFixed(6)} ${cost.estimated_calls ? "estimated" : "reported"}`;
    if (ctx.sessionManager.getSessionId() === sessionId && ctx.model?.provider === "switchyard") {
      ctx.ui.setStatus("switchyard-cost", `Switchyard ${text} (routing $${snapshot.routing_cost.known_usd.toFixed(6)} known)`);
    }
  } catch {
    if (ctx.sessionManager.getSessionId() === sessionId && ctx.model?.provider === "switchyard") ctx.ui.setStatus("switchyard-cost", "Switchyard cost unavailable");
  }
}

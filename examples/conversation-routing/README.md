# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Conversation routing example

A two-tier conversation deployment built entirely from the checked-in
`switchyard-server` config surface — no custom code.

| Route | Mechanism | Behavior |
|---|---|---|
| `switchyard/conversation` | `llm_classifier` custom mode | Pre-hoc scoring card: a judge reads the turn against the conversation capability card (CONV-1..9) and names `efficient` or `capable`. |
| `switchyard/conversation-regret` | `llm_classifier` escalation mode | Regret-driven escalation: the efficient tier answers by default; when the judge sees user regret (correction, re-ask, dissatisfaction), the session latches to capable. |

Run:

```bash
export OPENROUTER_API_KEY=sk-or-...
switchyard-server --config examples/conversation-routing/conversation-routing.toml --port 4000
```

Then point an OpenAI-compatible client at `http://localhost:4000/v1` with
`model = "switchyard/conversation"` or `"switchyard/conversation-regret"`.

Notes:

- Replace the OpenRouter model ids with your own tiers; the judge target is a
  separate small model, not a routing destination.
- The scoring card is uncalibrated. Tune the CONV rules and thresholds against
  your own traffic (user-regret logs are the free calibration label).
- Escalation latches one-way for the session and does not decay back to the
  efficient tier; a conversation "task" spans the whole session.

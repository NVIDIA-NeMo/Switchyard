# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default reasoning_effort for baseline smoke-test (agent, model) combos.

One row per pair, aligned columns. Grep for the agent name (e.g. `codex`) to
see every model it covers.

Pairs NOT listed here: the launcher passes no reasoning_effort and whatever
default the agent ships with fires (harbor-default "high" for codex; nothing
for claude-code+haiku-4-5 since haiku doesn't support effort at all). The
Qwen3.5 family uses a different knob entirely (enable_thinking boolean) and
is handled by a separate launcher branch, not by this table. The qwen-coder
agent (Qwen Code CLI fork of Gemini CLI) similarly has no reasoning_effort
kwarg. Qwen3.x thinking depends on the endpoint's server configuration.

The `pi` agent (earendil-works/pi-coding-agent) uses its own
`--thinking {off,minimal,low,medium,high,xhigh}` CLI flag, exposed by
harbor as the `thinking` agent-kwarg. Driven by the launcher's per-agent
case, not by this table.

See docs/runbooks/baseline-reproducibility.md for rationale + verification recipes.
"""

REASONING_EFFORT: dict[tuple[str, str], str] = {
    # ---------- codex ----------
    # OpenAI recommends `medium` as daily driver for coding; we use `high` since
    # our smoke tasks are harder than average.
    ("codex", "gpt-5.3-codex"): "high",
    ("codex", "gpt-5.4"): "high",
    ("codex", "gpt-5.5"): "high",
    # ---------- claude-code (per Anthropic's effort docs) ----------
    # The Harbor patch adds xhigh to Claude Code's accepted effort values.
    ("claude-code", "claude-opus-4-7"): "xhigh",
    # Opus 4.6: API default is `high`
    ("claude-code", "claude-opus-4-6"): "high",
    # Opus 4.5: effort supported, no specific recommendation — use API default
    ("claude-code", "claude-opus-4-5"): "high",
    # Sonnet 4.6: "Medium effort (recommended default)" for agentic coding
    ("claude-code", "claude-sonnet-4-6"): "medium",
    # ---------- opencode ----------
    # OpenAI-compatible endpoints may accept only low, medium, and high.
    # Keep high as the default here; callers can override it for their endpoint.
    #
    # These keys use the exact model IDs that the endpoint receives.
    # The launcher adds gateway/ only to select OpenCode's configured provider.
    ("opencode", "claude-opus-4-7"): "high",
    ("opencode", "claude-opus-4-6"): "high",
    ("opencode", "claude-opus-4-5"): "high",
    ("opencode", "claude-sonnet-4-6"): "medium",
    ("opencode", "claude-haiku-4-5"): "medium",
    # ---------- pi (renders as --thinking via launcher's pi branch) ----------
    # Pi's `--thinking` flag levels: off, minimal, low, medium, high, xhigh.
    # For reasoning-capable open models we use `high`; haiku-4-5 is not
    # reasoning-capable so we leave it OFF (no row → no --thinking flag).
    ("pi", "Qwen/Qwen3.6-35B-A3B"): "high",
    ("pi", "zai-org/glm-5.1"): "high",
}


def effort_for(agent: str, model: str) -> str | None:
    """Return the default reasoning_effort for (agent, model), or None if not defined.

    Exact-match lookup by (agent, full_model_id). Add a row to REASONING_EFFORT
    when you start smoke-testing a new combo.
    """
    return REASONING_EFFORT.get((agent, model))

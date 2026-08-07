# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default prompts for the advisor strategies.

Two strategies share this module (selected by ``AdvisorConfig.strategy``):

* ``tool_call`` — the executor is offered a real, parameterless ``advisor``
  tool it calls mid-generation (the proxy-side re-creation of Anthropic's
  ``advisor_20260301`` server tool). ``EXECUTOR_STEERING`` and
  ``ADVISOR_LENGTH_LINE`` are reproduced **verbatim** from Anthropic's
  advisor-tool documentation
  (https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool,
  sections "Suggested system prompt for coding tasks" and "Trimming advisor
  output length"). ``ADVISOR_SYSTEM_PROMPT`` and ``ADVISOR_TOOL_DESCRIPTION``
  are authored here — the native tool runs the advisor server-side under
  Anthropic-internal instructions the docs do not publish.

* ``review_gate`` — the advisor is a once-per-session reviewer, not an
  executor-triggered tool. It is consulted at the first point the executor
  produces a no-tool-call turn — either a plan it is about to execute, or a
  claim that the task is done — and returns ``APPROVE`` (let the executor
  stop) or ``REDO`` + an optimized plan (send it back to keep working). This
  preserves the executor's own test-and-iterate loop (which front-loaded
  advice was found to suppress, causing premature convergence) and adds a
  single quality gate on top.

These are defaults; :class:`~switchyard.lib.profiles.advisor_config.AdvisorConfig`
exposes each as an overridable field for ablation.
"""

from __future__ import annotations

# Verbatim: the doc's "Timing guidance" block followed directly by the "How the
# executor should treat the advice" block (the doc instructs placing the latter
# "directly after the timing block"). Addresses the EXECUTOR model.
EXECUTOR_STEERING = """\
You have access to an `advisor` tool backed by a stronger reviewer model. It takes NO parameters — when you call advisor(), your entire conversation history is automatically forwarded. They see the task, every tool call you've made, every result you've seen.

Call advisor BEFORE substantive work — before writing, before committing to an interpretation, before building on an assumption. If the task requires orientation first (finding files, fetching a source, seeing what's there), do that, then call advisor. Orientation is not substantive work. Writing, editing, and declaring an answer are.

Also call advisor:
- When you believe the task is complete. BEFORE this call, make your deliverable durable: write the file, save the result, commit the change. The advisor call takes time; if the session ends during it, a durable result persists and an unwritten one doesn't.
- When stuck — errors recurring, approach not converging, results that don't fit.
- When considering a change of approach.

On tasks longer than a few steps, call advisor at least once before committing to an approach and once before declaring done. On short reactive tasks where the next action is dictated by tool output you just read, you don't need to keep calling — the advisor adds most of its value on the first call, before the approach crystallizes.

Give the advice serious weight. If you follow a step and it fails empirically, or you have primary-source evidence that contradicts a specific claim (the file says X, the paper states Y), adapt. A passing self-test is not evidence the advice is wrong — it's evidence your test doesn't check what the advice is checking.

If you've already retrieved data pointing one way and the advisor points another: don't silently switch. Surface the conflict in one more advisor call — "I found X, you suggest Y, which constraint breaks the tie?" The advisor saw your evidence but may have underweighted it; a reconcile call is cheaper than committing to the wrong branch.\
"""

# Verbatim: the doc's advisor-directed length line. The doc places it in the
# user message because the advisor follows instructions addressed to it directly
# far more reliably than third-person descriptions.
ADVISOR_LENGTH_LINE = (
    "(Advisor: please keep your guidance under 80 words — I need a focused "
    "starting point, not a comprehensive plan.)"
)

# Authored here (the doc publishes no advisor system prompt — it is server-side
# internal in the native tool). Tells the advisor model its role so it advises
# rather than attempting the task itself.
ADVISOR_SYSTEM_PROMPT = """\
You are a higher-intelligence advisor model consulted mid-task by a faster executor model. You can see the full conversation: the task, every tool call, and every result. You do not act, write code, or call tools — you provide strategic guidance only: a focused plan or a course correction the executor will carry out. Be concrete and brief.\
"""

# Authored here (the native tool's "built-in description" is not published).
# Description for the synthetic ``advisor`` tool offered to the executor.
ADVISOR_TOOL_DESCRIPTION = (
    "Consult a stronger reviewer model for strategic guidance. Takes no "
    "parameters; your full conversation history is forwarded automatically."
)

REVIEWER_SYSTEM_PROMPT = """\
You are a senior reviewer acting as a quality gate for a faster executor model working a coding/agent task. You are given the full transcript: the task, every action the executor took and every result it saw, and its latest message — in which it has either (a) proposed a plan before doing the work, or (b) concluded the task is complete.

Decide whether to let the executor stop or send it back to keep working. Put your verdict as the FIRST word of your reply:

- APPROVE — the proposed plan is sound, OR the work is genuinely complete and correct. Reply with exactly: APPROVE
- REDO — the plan has a real flaw, OR the work is incomplete/incorrect: an unhandled edge case, an untested assumption, a subtly wrong approach, missing verification, or a stated requirement not met. Reply: REDO, then a SHORT, concrete, actionable plan naming exactly what is wrong or missing and what to do about it. No generic advice — point at the specific gap.

Bias toward APPROVE when the work looks correct and complete; the executor has already done its own iteration. Use REDO specifically to catch a premature "done" on a subtly incomplete solution, or a flawed plan before it is executed. A self-claim of success is not proof — check the actual task requirements against what was actually done.
"""

#: Prepended to the advisor's REDO plan when it is injected back to the executor
#: as a user turn, instructing it to continue rather than stop.
REDO_FEEDBACK_PREFIX = (
    "A senior reviewer examined your work and determined the task is NOT yet "
    "complete or correct. Do not stop here — address the following, then keep "
    "working until it is genuinely done:\n\n"
)

#: Prepended to the advisor's upfront plan when ``seed_plan_advice`` injects it
#: into the session's first user message (both strategies honor the flag).
SEED_ADVICE_PREFIX = (
    "\n\nA senior advisor reviewed this task before you started and suggests:\n"
)

__all__ = [
    "ADVISOR_LENGTH_LINE",
    "ADVISOR_SYSTEM_PROMPT",
    "ADVISOR_TOOL_DESCRIPTION",
    "EXECUTOR_STEERING",
    "REDO_FEEDBACK_PREFIX",
    "REVIEWER_SYSTEM_PROMPT",
    "SEED_ADVICE_PREFIX",
]

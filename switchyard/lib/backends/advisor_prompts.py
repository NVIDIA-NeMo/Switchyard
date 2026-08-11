# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default prompts for the advisor review gate.

The advisor is a once-per-session reviewer: it is consulted at the first point
the executor produces a no-tool-call turn — either a plan it is about to
execute, or a claim that the task is done — and returns ``APPROVE`` (let the
executor stop) or ``REDO`` + an optimized plan (send it back to keep working).
This preserves the executor's own test-and-iterate loop (which front-loaded
advice was found to suppress, causing premature convergence) and adds a
single quality gate on top.

These are defaults; :class:`~switchyard.lib.backends.advisor_config.AdvisorConfig`
exposes each as an overridable field for ablation.
"""

from __future__ import annotations

# Tells the advisor model its role for the optional ``seed_plan_advice``
# consult, so it advises rather than attempting the task itself.
ADVISOR_SYSTEM_PROMPT = """\
You are a higher-intelligence advisor model consulted mid-task by a faster executor model. You can see the full conversation: the task, every tool call, and every result. You do not act, write code, or call tools — you provide strategic guidance only: a focused plan or a course correction the executor will carry out. Be concrete and brief.\
"""

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
#: into the session's first user message.
SEED_ADVICE_PREFIX = (
    "\n\nA senior advisor reviewed this task before you started and suggests:\n"
)

__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "REDO_FEEDBACK_PREFIX",
    "REVIEWER_SYSTEM_PROMPT",
    "SEED_ADVICE_PREFIX",
]

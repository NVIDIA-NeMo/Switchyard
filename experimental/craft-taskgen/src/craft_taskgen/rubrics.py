# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative rubric text for CRAFT pipeline judge prompts.

All rubric content lives here as module-level string constants. Prompts in
prompts.py import and interpolate these strings directly via f-string
substitution — no file-path references in prompt text, since direct-API judges
have no Read tools.

Four categories of rubric:

- **Evaluation-time** (applied to candidate PRs before build):
  `RUBRIC_QUICK_DECISION_FRAMEWORK`, `RUBRIC_REJECT_PATTERNS`,
  `RUBRIC_DESIGN_PRINCIPLES`.
- **Construction-time** (applied at build and re-audited by the alignment
  judge): `RUBRIC_H_RULES`, `RUBRIC_T2_H1_ORCHESTRATION`,
  `RUBRIC_V4_AUDIT`, `RUBRIC_ANTI_LEAKAGE`.
- **Alignment-judge-specific categorization**: `RUBRIC_ALIGNMENT_CATEGORIES`.
- **Triage-time fairness review** (applied to tasks with Opus smoke
  reward<1): `RUBRIC_FAIRNESS_REVIEW`.

Sync discipline: this module is the Python source of truth. The matching
human-readable sections in `references/task-building-guide.md` are the
contributor mirror. `tests/test_rubric_drift.py` fails CI if canonical
paragraphs here and in that markdown diverge. When editing either, land both
changes in the same commit.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared values
# ---------------------------------------------------------------------------

INSTRUCTION_WORD_RANGE = "50-100 words"
INSTRUCTION_WORD_HARD_MAX = 200


# ---------------------------------------------------------------------------
# Evaluation-time rubric (applied to candidate PRs by _evaluate_one)
# ---------------------------------------------------------------------------

RUBRIC_QUICK_DECISION_FRAMEWORK = """\
Apply these five questions to decide whether the candidate PR is worth building
a task from:

1. Will models diverge in approach? Bug fixes where every model follows
   find-line-fix-line are useless. Bug fixes requiring architectural reasoning
   can be discriminating. Feature implementations are generally better because
   they require integration decisions — but the label isn't what matters,
   strategy divergence is.

2. Is the core problem genuinely hard? Repo size, file count, and
   parameter-threading breadth don't create difficulty. If the underlying logic
   is trivial, the task is trivial. Tightening instructions on an easy problem
   doesn't make it hard.

3. Reference tests exist? The commit must include test files that exercise real
   behavior. No tests = no verifier = no task.

4. Pure Python? Rust/C extension work is untaskable in a Python-only agent
   harness.

5. Low contamination risk? Prefer post-Sept-2025 commits. Well-known library
   features with clear GitHub issues are risky."""


RUBRIC_REJECT_PATTERNS = """\
Reject candidates that match any of these worked patterns:

- Bug fix with obvious strategy (example: SA1 — SQLAlchemy M2M join). All
  models use the same find-line-fix-line pipeline; no strategy discrimination.
- Injected bug in an error-absorbing system (example: MY1 — mypy encoding).
  Cannot verify when errors are silently swallowed.
- Constructed task (example: AL1 — Alembic constraints). Too easy, solvable
  in one minute.
- Trivial core logic in a complex repo (example: BT1 — beets genre migration).
  Easy problem stays easy regardless of surrounding repo size.

Hard rejections, no LLM judgment required: no test files in the commit; not
Python; pure docs / CI config / version bumps; pure mechanical reformatting."""


RUBRIC_DESIGN_PRINCIPLES = """\
Design principles that guide both candidate selection and which aspect of a
multi-change PR to emphasize when building the instruction:

- Integration is the discriminating step, not component creation. The gap
  between "I created a class" and "the system uses it" is where weaker models
  fail. Prefer tasks where wiring into an existing system is the hard part.
- Preserving existing behavior while adding new behavior is a genuine trap.
  Tasks requiring additive changes without regression force models to reason
  about the entire call graph.
- Repository-level variance is signal, not noise. Codebase-specific factors
  (docs quality, architecture, test harness) create authentic difficulty.
  Don't normalize across repos.
- Failure mode signatures differ by model tier. Stronger models fail on
  semantic understanding; weaker models fail on context overflow and
  navigation. Multi-trial discrimination captures this.

If a PR bundles multiple separable features, target the integration /
discriminating aspect — not the easiest. A 2-test single-method task carved
out of a multi-file commit means the wrong slice was chosen."""


# ---------------------------------------------------------------------------
# Construction-time rubric (applied by Build, re-audited by alignment judge)
# ---------------------------------------------------------------------------

RUBRIC_H_RULES = f"""\
Construction-time rules for instruction.md. Fail any of these and the
instruction is not ready.

H1. Outcome-oriented. Tell the agent what "done" looks like; don't enumerate
    steps. Numbered acceptance criteria ("1. All tests pass") are fine.
    Numbered procedures ("1. Read the file, 2. Fix the bug") are not. Test:
    if you could execute the instruction like a recipe without thinking,
    it's too procedural.

H2. Don't give away the diagnosis. State the symptom, not the cause. Don't
    reveal what's wrong, how many bugs, or which files. Test: could someone
    jump straight to the fix after reading only the instruction?

H3. Essential difficulty, not clerical. The core challenge must be reasoning,
    diagnosis, or domain knowledge — not format compliance or spelling.
    Signal: if the instruction spends more words on output format than on
    the problem, or if the most likely failure is a format mismatch rather
    than a reasoning error, it's clerical difficulty.

H4. The obvious first approach should fail. Design tasks where a naive
    one-step attempt doesn't work. Error recovery and replanning are core
    agent skills. Test: write down the most obvious approach; does it work?
    If yes, add a realistic complication.

H5. Not solvable in a trivial sequence. The task must require non-obvious
    decisions about tools and ordering. A single command, short pipeline,
    or mechanical sequence is too easy. Test: can you solve it in under 3
    tool calls? Can a junior engineer describe the approach immediately?
    If either: needs more depth.

H6. Brevity. Target {INSTRUCTION_WORD_RANGE}. Write for a senior engineer.
    Say things once. Every token is a chance for ambiguity or a spec detail
    the verifier might miss. Don't explain how things might fail. Don't
    list available tools. Hard maximum: {INSTRUCTION_WORD_HARD_MAX} words.

(H7 — "Must discriminate between model tiers" — is post-hoc, evaluated from
smoke-test results across model tiers. Not enforceable at construction time.)"""


RUBRIC_T2_H1_ORCHESTRATION = """\
T2-H1. Post-exploration work is genuinely hard (Orchestration dimension).

Exploration can contribute to difficulty but can't be the only difficulty.
The action after finding the right files must require real reasoning.

- Pass: agent must find relevant files AND reason about a subtle multi-file
  bug.
- Fail: agent searches a large repo, but the fix is an obvious one-line
  change.
- Common trap: removing hints from a trivially easy problem doesn't make it
  hard. If the underlying fix is a single arithmetic expression in one
  function, no amount of instruction-tightening will create genuine
  difficulty. The problem itself must require multi-step reasoning.

Test: assume the agent already knows which files matter. Is the remaining
task still hard?"""


RUBRIC_V4_AUDIT = """\
V4 three-layer audit of instruction ↔ reference-test alignment. Apply each
layer to every reference test file:

1. Fixtures layer — do test fixtures (config construction, constructor
   signatures, setup scaffolding) encode design choices that the instruction
   doesn't specify? If yes, any valid alternative implementation will fail the
   fixtures, and the task is unfair. The instruction must name design choices
   that appear in fixtures.

2. Helpers layer — do test helpers access private API (underscore-prefixed
   methods, internal attributes, module-internal paths) without the
   instruction naming the symbol? If yes, the agent has no way to know that
   API exists and will design a valid alternative that the helpers can't
   reach. Instruction must name any non-public symbol the tests reach for.

3. Assertions layer — do assertions check behavioral outcomes or
   format-only details (exact log strings, specific error message wording,
   punctuation)? Behavioral assertions are durable; format-only assertions
   create test fragility that punishes semantically correct solutions.
   Instruction must either specify format precisely or tests should not
   check format.

Rule of thumb: design choices in test FIXTURES must appear in the
instruction. Design choices in test ASSERTIONS (internal attributes, error
messages) may be left for discovery IF they are behavioral, not format."""


RUBRIC_ANTI_LEAKAGE = """\
Anti-leakage rules for instruction.md content. Each rule is grounded in
external benchmark-curation research (SWE-smith system prompt, SWE-Factory
10-point issue-authoring rubric, SWE-bench+ issue-leakage failure modes):

- Do not reveal the fix or any solution code (SWE-smith: "DO NOT GIVE AWAY
  THE FIX").
- Do not name specific failing reference tests, test functions, or test
  files by name (SWE-Factory rule 2: "Do NOT mention test functions or
  files directly").
- Do not mention pytest, hypothesis, or any testing framework (SWE-smith:
  "DO NOT SUGGEST RUNNING ANY TESTING COMMANDS"; SWE-Factory rule 2).
- Do not use `assert` statements in the instruction — describe expected
  behavior in prose (SWE-Factory rule 5: "Do NOT use assert statement in
  issue text").
- Do describe the difference between actual and expected output when the
  expected output is large, rather than quoting it verbatim (SWE-Factory
  rule 5 nuance).

Note on symbol naming: H1 in RUBRIC_H_RULES prohibits numbered-procedure /
recipe framing. H2 in RUBRIC_H_RULES prohibits revealing the diagnosis
(cause, not symptom). V4 in RUBRIC_V4_AUDIT requires naming symbols that
reference-test fixtures import (the agent cannot see the tests). Together
those three cover the "don't reveal the implementation path" concern
without banning every private symbol name — which is necessary to satisfy
V4's instruction-test alignment requirement."""


# ---------------------------------------------------------------------------
# Alignment-judge-specific categorization rubric
# ---------------------------------------------------------------------------

RUBRIC_ALIGNMENT_CATEGORIES = """\
Classify the instruction ↔ reference-tests ↔ diff relationship into exactly
one of these categories. The primary concern is misalignment (tests requiring
things the instruction doesn't specify). Leakage is secondary.

- ok: instruction and tests are well aligned. A senior engineer reading only
  the instruction could predict what the tests check, and a senior engineer
  reading only the tests could infer roughly what the instruction describes.
  No fix is revealed.

- narrow_tests: tests require things the instruction doesn't cover. The
  clearest form of the unfair-hard failure mode — failure would be due to
  missing information, not missing capability. Name the specific tests that
  are narrow and what they require that the instruction omits. (This is the
  V4 fixtures/helpers/assertions layers detecting unstated requirements.)

- vague: instruction is too under-specified to predict what the tests check.
  A reader cannot form any concrete expectation of the target API contract
  or behavior from the instruction alone.

- leaked: instruction reveals the implementation, the fix, or enough
  diagnostic detail that a reader can bypass the reasoning work. Specific
  forms to watch for: naming the exact file and line, naming private
  methods the diff touches, quoting a fix-like code fragment, or including
  a step-by-step procedure that doubles as a recipe. This is the
  Aleithan 2410.06992 / SWE-bench+ issue-body-leakage failure mode.

- misaligned: instruction and tests relate to different problems, or
  collectively don't match the PR diff. Use when a category doesn't cleanly
  fit but the task clearly isn't shippable.

Retention-biased posture: rejection here wastes a completed Build. When
genuinely uncertain between `ok` and another category, lean toward `ok` —
downstream Opus smoke and post-execution reviewer provide a second chance
to catch problems with execution evidence."""


# ---------------------------------------------------------------------------
# Triage-time fairness review rubric (applied to failed Opus smoke trials)
# ---------------------------------------------------------------------------

RUBRIC_FAIRNESS_REVIEW = """\
Return exactly one severity verdict from the three below. `major` is gated
on producing two specific pieces of evidence; if either is missing,
downgrade to `minor` or `none`.

- none: the instruction covers the behavior the tests require. The agent's
  trial failure is a genuine capability gap (hard reasoning, multi-step
  integration, subtle edge case), not an unfairness in the task design.
  Most failed trials should land here.

- minor: the instruction slightly under-specifies something a test depends
  on, but a competent senior engineer could reasonably infer the intent
  from context (repo conventions, PR subject, adjacent code). The task is
  still shippable; flag it for human review rather than rewriting.

- major: a reference test requires behavior the instruction does not
  state, and a reasonable agent reading the instruction alone would
  likely miss it. Commit to `major` when you can (a) name the specific
  unstated behavior and (b) point to one or more failing tests that
  exercise it — even if a maximally-clever agent might have inferred
  the requirement, the fact that reasonable effort misses it is
  enough. The classic unfair-hard failure mode.

Evidence requirements for `major` (BOTH required; if either is missing
downgrade to `minor`):

- `evidence_quote`: an exact sentence or phrase copied verbatim from the
  instruction, showing what the instruction DOES state. This anchors the
  under-specification claim to the instruction text itself, not to your
  summary of it.

- `evidence_test`: the full pytest id of one failing reference test that
  requires behavior NOT covered by the quoted instruction sentence. Name
  the specific behavior the test checks that the instruction omits.

Posture: a rewrite based on a weak signal is expensive (one extra
Build roundtrip plus re-smoke). But an overly cautious review misses
real unfairness. When you have identified both a specific instruction
sentence AND a named failing test that depends on unstated behavior,
commit to `major` — that IS the specific-evidence bar, and hedging to
`minor` when both fields are well-populated defeats the purpose. Only
default to `minor` when your evidence would be speculative or you
cannot name the specific test-to-unstated-behavior link. Do not
invent evidence to justify `major`.

Do not treat these as `major`:
- Tests that fail because the agent chose a different valid
  implementation path than the PR took (that is `none` — a genuine gap
  only if the agent's path would pass a reasonable rewrite of the test;
  otherwise the test itself is over-constrained, which is a separate
  skip-worthy signal handled by the per-test DD, not by this review).
- General instruction quality concerns (wordiness, tone, ordering) that
  don't tie to a specific test failure.
- Tests that check private APIs the instruction names — if the
  instruction says "the `_foo_bar` helper must ...", a test referencing
  `_foo_bar` is aligned, not unfair."""

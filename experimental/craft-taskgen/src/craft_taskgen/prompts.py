# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for pipeline LLM steps.

Each function returns a prompt string. Schemas define structured output format.

Direct-API prompts (evaluate, build, alignment judge, deep-dive, skeptical
reviewer, task summary) inline all rubric content from `rubrics.py` — no
file paths, no skill references, no Read/Grep tools. The only remaining
prompt that reaches the shell is `build_dockerfile_prompt`, which runs
under `claude -p` with filesystem tools and references
`references/task-building-guide.md`. Fix-handler prompts (fix_docker,
fix_f2p_p2p_classify, fix_triage) also shell out for scoped file edits.
"""

from __future__ import annotations

from craft_taskgen import rubrics
from craft_taskgen.config import PipelineContext

# ---------------------------------------------------------------------------
# Challenge questions — canonical list applied by both deep-dive and skeptical
# reviewer agents. Keep this as the single source of truth.
# ---------------------------------------------------------------------------

CHALLENGE_QUESTIONS = """\
Q1: Is this test testing the FEATURE WE ASKED FOR? Or is it testing existing/regression behavior?
    The upstream commit often bundles unrelated changes alongside the feature: locale data
    fixes, internal API refactors, test harness modifications, config changes. The F2P bootstrap
    picks up ALL of these. If a test exercises code unrelated to the feature described in the
    instruction, it doesn't belong in the F2P set.
    Examples: a streaming-fix task had tests about non-streaming post-yield exceptions. A
    dehumanize task had a Bengali ordinal formatting test bundled in the same commit. A MySQL
    DDL task had 155 tests for generic ORM CRUD bundled in the same commit.

Q2: Does the instruction SPECIFY what the test checks? Find the EXACT sentence.
    "Show me the sentence in the instruction that tells the agent to do X."
    If there's no sentence → instruction_scope, not genuine_gap.
    Example: a test expected SystemExit but the instruction never mentioned error handling.

Q3: Does the instruction WORDING match the test SCOPE?
    If instruction says "X" but test requires "X and Y" → instruction_scope.
    Example: an instruction said "generators" but tests used Iterable/AsyncIterable.

Q4: Is this a systematic failure pattern or an Opus-specific quirk?
    Look for signals that the failure would affect any reasonable solver —
    e.g. the test imports a private symbol, asserts an exact string the
    instruction never specifies, or requires a specific library version not
    named in the instruction. Systematic patterns point to
    instruction/verifier issues rather than capability gaps.

Q5: Would a CORRECT ALTERNATIVE implementation fail this test?
    Imagine a senior engineer reading only the instruction. Would their reasonable
    implementation pass? If not → test is over-constrained. Watch for these patterns:
    - Test @patches a specific module attribute that only exists if the agent chose one
      particular implementation approach
    - Test imports a private function by exact name and path — the agent may solve the
      same problem differently
    - Test asserts an exact log message string the instruction never specifies
    Examples: tests patched `threading` but the instruction just said "fire timers during
    shutdown" — an inline polling loop is equally valid. A test imported a private utility
    function; the agent handled unhashable callables a different way. Tests checked exact
    warning strings for meta keys the instruction never mentioned.

Q6: Did PASSING TESTS pass for the RIGHT REASON?
    For each passing test: is there a direct link from instruction to the agent's design choice?
    Or did the agent guess right? Coincidental passes are fragile.
    Example: 5/8 passes were coincidental because the instruction referenced the wrong import path.

Q7: Is this actually an ENVIRONMENT ISSUE, not an agent bug?
    Before classifying a failure, verify the error comes from the agent's code. Missing
    dependencies, broken test infrastructure, or container setup problems can crash tests
    before they reach the assertion. If the traceback points to a package import or infra
    setup rather than the agent's implementation, it's an environment issue.
    Example: a test failed on a missing msgpack serializer, not on the agent's retry logic."""

CHALLENGE_QUESTIONS_SHORT = """\
Q1: Is this test testing the FEATURE WE ASKED FOR, or unrelated bundled changes?
Q2: Does the instruction SPECIFY what the test checks? Find the EXACT sentence.
Q3: Does the instruction WORDING match the test SCOPE?
Q4: Is this a systematic failure pattern (private-symbol import, exact-string assertion) or an Opus-specific quirk?
Q5: Would a CORRECT ALTERNATIVE implementation fail this test?
Q6: Did PASSING TESTS pass for the RIGHT REASON or coincidentally?
Q7: Is this actually an ENVIRONMENT ISSUE, not an agent bug?"""

# Word count TARGET for instruction.md — what we ask the builder to aim for.
# task_format.py validator enforces a hard max of 200.
_INSTRUCTION_WORD_RANGE = "50-100 words"

EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "reject"]},
        "reason": {"type": "string"},
        "instruction_sketch": {"type": "string"},
        "reject_pattern": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

BUILD_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_md": {"type": "string"},
        "task_slug": {"type": "string"},
    },
    "required": ["instruction_md", "task_slug"],
}

ALIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["ok", "vague", "narrow_tests", "leaked", "misaligned"],
        },
        "reason": {"type": "string"},
        "leakage_evidence": {"type": "array", "items": {"type": "string"}},
        "v4_audit": {
            "type": "object",
            "properties": {
                "fixtures_encode_design_choices": {"type": "boolean"},
                "helpers_access_private_api": {"type": "boolean"},
                "assertions_format_only": {"type": "boolean"},
            },
        },
    },
    "required": ["verdict", "reason"],
}

DEEP_DIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "reward": {"type": "number"},
        "ref_tests_passed": {"type": "integer"},
        "ref_tests_total": {"type": "integer"},
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["skip", "keep"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["test_name", "classification"],
            },
        },
        "overall_assessment": {"type": "string"},
    },
    "required": ["reward", "failures", "overall_assessment"],
}

FAIRNESS_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "reason": {"type": "string"},
        "evidence_quote": {"type": "string"},
        "evidence_test": {"type": "string"},
    },
    "required": ["severity", "reason"],
}


SOLVE_SH_TEMPLATE = """\
#!/bin/bash
# Oracle solution: apply PR diff to /code
set -e
cd /code
git apply /solution/changes.patch --ignore-whitespace --allow-empty 2>/dev/null || \\
    patch -p1 < /solution/changes.patch --forward --ignore-whitespace || \\
    { echo "FATAL: patch application failed" >&2; exit 1; }
"""

SCORE_PY_TEMPLATE = '''\
#!/usr/bin/env python3
"""Binary resolved scorer (SWE-bench C.5 aligned) with F2P/P2P diagnostic ratios.

Generated by craft-taskgen at classification time. Do not edit by hand.
Reads test output written by test.sh — do not run this script directly.
"""
import json
import re
from pathlib import Path

reward_dir = Path("/logs/verifier")
reward_dir.mkdir(parents=True, exist_ok=True)


def _load_list(path):
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _load_skip_list(path):
    """Load skip file (test_id | reason). Returns set of test IDs.

    Lines starting with # are comments. Missing file returns empty set.
    """
    p = Path(path)
    if not p.exists():
        return set()
    skipped = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        test_id = line.split("|")[0].strip()
        if test_id:
            skipped.add(test_id)
    return skipped


FAIL_TO_PASS = _load_list("/tests/fail_to_pass.txt")
PASS_TO_PASS = _load_list("/tests/pass_to_pass.txt")

F2P_SKIPPED = _load_skip_list("/tests/f2p_skip.txt")
FAIL_TO_PASS_EFFECTIVE = [t for t in FAIL_TO_PASS if t not in F2P_SKIPPED]

P2P_SKIPPED = _load_skip_list("/tests/p2p_skip.txt")
PASS_TO_PASS_EFFECTIVE = [t for t in PASS_TO_PASS if t not in P2P_SKIPPED]

output_path = reward_dir / "verify_full_output.txt"
if not output_path.exists():
    raise FileNotFoundError(
        f"{output_path} not found — test.sh likely failed before tee could write it. "
        "Check that /logs is mounted and pytest ran successfully."
    )
full_output = output_path.read_text()

passed = set()
for line in full_output.splitlines():
    m = re.match(r"^(\\S+::\\S+)\\s+PASSED", line)
    if m:
        passed.add(m.group(1))

f2p_passed = len([t for t in FAIL_TO_PASS_EFFECTIVE if t in passed])
f2p_failed = sorted([t for t in FAIL_TO_PASS_EFFECTIVE if t not in passed])
f2p_total = len(FAIL_TO_PASS_EFFECTIVE)
p2p_passed = len([t for t in PASS_TO_PASS_EFFECTIVE if t in passed])
p2p_failed = sorted([t for t in PASS_TO_PASS_EFFECTIVE if t not in passed])
p2p_total = len(PASS_TO_PASS_EFFECTIVE)

if f2p_total == 0:
    if len(FAIL_TO_PASS) > 0 and len(F2P_SKIPPED) > 0:
        raise ValueError(
            "All F2P tests skipped — task has no valid F2P tests remaining. "
            f"Original F2P count: {len(FAIL_TO_PASS)}, skipped: {len(F2P_SKIPPED)}. "
            "Review f2p_skip.txt — at least one F2P test must remain."
        )
    raise ValueError(
        "fail_to_pass.txt is empty — no F2P tests defined. "
        "This task has no way to verify the feature was implemented. "
        "Check F2P/P2P classification output."
    )

f2p_score = f2p_passed / f2p_total
p2p_score = p2p_passed / p2p_total if p2p_total > 0 else 1.0

resolved = f2p_score == 1.0 and p2p_score == 1.0
reward = 1.0 if resolved else 0.0

# reward.txt holds the scalar reward (default Mean metric reads this).
# reward.json holds numeric-only diagnostics — harbor>=0.13.1 parses
# reward.json into a dict[str, float|int]; list fields break pydantic
# validation. Failure-test lists move to reward-details.json.
with open(reward_dir / "reward.txt", "w") as f:
    f.write(str(reward))

with open(reward_dir / "reward.json", "w") as f:
    json.dump(
        {
            "reward": reward,
            "resolved": resolved,
            "f2p_passed": f2p_passed,
            "f2p_total": f2p_total,
            "f2p_score": f2p_score,
            "p2p_passed": p2p_passed,
            "p2p_total": p2p_total,
            "p2p_score": p2p_score,
            "f2p_skipped": len(F2P_SKIPPED & set(FAIL_TO_PASS)),
            "f2p_total_before_skips": len(FAIL_TO_PASS),
            "p2p_skipped": len(P2P_SKIPPED & set(PASS_TO_PASS)),
            "p2p_total_before_skips": len(PASS_TO_PASS),
        },
        f,
        indent=2,
    )

with open(reward_dir / "reward-details.json", "w") as f:
    json.dump(
        {
            "f2p_failed": f2p_failed,
            "p2p_failed": p2p_failed,
        },
        f,
        indent=2,
    )
'''


def evaluate_candidate_prompt(
    repo: str,
    sha: str,
    subject: str,
    diff_stat: str,
    diff: str,
    readme_excerpt: str = "",
) -> str:
    """Build the evaluate-step prompt for direct-API dispatch.

    Unlike the prior claude -p version, the LLM has no tools — every piece of
    context (git diff, rubric) must be inlined in the prompt. Rubric constants
    come from rubrics.py.
    """
    readme_section = f"\n<readme_excerpt>\n{readme_excerpt}\n</readme_excerpt>\n" if readme_excerpt else ""
    return f"""You are evaluating a git commit as a candidate for a CRAFT benchmark task.

CRAFT measures HOW agents work — exploration strategy, tool orchestration,
verification quality, iteration behavior — not just whether they produce
correct code. Good tasks are feature implementations where the agent's
APPROACH matters and models diverge in strategy, not just speed.

<candidate>
Repo: {repo}
Commit: {sha}
Subject: {subject}
</candidate>
{readme_section}
<diff_stat>
{diff_stat}
</diff_stat>

<pr_diff>
{diff}
</pr_diff>

<quick_decision_framework>
{rubrics.RUBRIC_QUICK_DECISION_FRAMEWORK}
</quick_decision_framework>

<reject_patterns>
{rubrics.RUBRIC_REJECT_PATTERNS}
</reject_patterns>

<design_principles>
{rubrics.RUBRIC_DESIGN_PRINCIPLES}
</design_principles>

<policy>
Apply the rubric criteria from the three blocks above. Produce a binary
verdict, `accept` or `reject`, grounded in specific named criteria — do not
rely on intuition or gut feel.

Decision procedure (follow in order):

1. Hard filters (see the "Hard rejections" line in reject_patterns): check if
   the commit has test files, is Python, is not pure docs/CI/version-bumps,
   and is not pure mechanical reformatting. A failure on any of these is an
   automatic reject; the matching hard-filter name goes in reject_pattern.

2. Worked reject patterns: compare the candidate to the four worked examples
   in reject_patterns (SA1 / MY1 / AL1 / BT1). If the candidate matches a
   pattern, reject with that pattern name.

3. Rubric application: work through the five questions in
   quick_decision_framework. If question 3 (reference tests exist), question 4
   (pure Python), or question 5 (contamination risk) fails decisively, that's
   a reject. Questions 1 (model divergence) and 2 (genuinely hard problem)
   favor accept when the answer points to integration complexity,
   cross-module reasoning, or preservation-while-adding per the
   design_principles.

4. Tiebreaker: when rubric signals are mixed and no hard filter / reject
   pattern triggered, lean toward accept. Downstream stages (build-time
   alignment judge, Opus smoke trial, post-execution reviewer) provide
   additional gates — rejection here wastes nothing downstream, but a false
   reject loses a real candidate we can't recover.

Citation requirement: your `reason` field MUST name the specific rubric
element you applied — either the question number from quick_decision_framework
("Q2: core problem is trivial — parameter passthrough"), the reject_patterns
entry ("matches BT1 pattern — trivial core logic"), or the design_principles
line ("violates integration-is-the-discriminator"). Vague reasons like "looks
easy" or "seems promising" are not acceptable.

On accept: include instruction_sketch — 2-3 sentences identifying the
HARDEST aspect of this commit (where models will diverge). Describe the
challenge as a behavioral outcome, not implementation steps.
Good: "Both sync and async receive paths must silently consume control frames."
Bad: "(1) add CurlWsFlag.PONG = 1 << 6, (2) fix _read_loop filtering."
Do not name private methods, internal functions, or magic numbers. Omit
reject_pattern (leave empty).

On reject: set reject_pattern to the matching pattern name (hard_filter:<name>,
SA1 / MY1 / AL1 / BT1, or a short phrase naming the triggered criterion).
Omit instruction_sketch.

Do not output `maybe`, `PROMISING`, difficulty bands, or time estimates.
Pre-execution difficulty prediction is unreliable; structural rubric criteria
drive the decision.
</policy>

Return a single JSON object matching the required schema. No markdown fences,
no extra text."""


def build_system_prompt(run_dir: str) -> str:
    """System prompt for build step — contains the run_dir (non-deterministic timestamp path)."""
    return (
        f"The run directory for task output is: {run_dir}\nReplace <run_dir> in instructions with this path."
    )


def build_task_prompt(
    repo: str,
    sha: str,
    merge_base_sha: str,
    subject: str,
    eval_reason: str,
    instruction_sketch: str,
    repo_map: str,
    diff: str,
    reference_test_bodies: list[tuple[str, str]],
    instruction_template: str,
    instruction_example: str,
    pre_fix_pytest_output: str = "",
    alignment_feedback: str = "",
    triage_feedback: str = "",
) -> str:
    """Build-step prompt for direct-API dispatch.

    All context is pre-assembled by the caller (repo map, diff, reference
    test bodies, template content). The LLM has no tools; every file it might
    want to read must appear inline here.

    Two optional feedback blocks may be present:
    - `alignment_feedback` — alignment judge rejected a prior draft; these are
      leakage phrases to avoid.
    - `triage_feedback` — deep-dive classified post-smoke failures as
      instruction-scope issues; these are per-test explanations of what the
      reference tests expected that the prior instruction didn't cover.
    """
    tests_section = "\n\n".join(
        f'<test path="{path}">\n{body}\n</test>' for path, body in reference_test_bodies
    )
    pre_fix_section = (
        f"\n<pre_fix_pytest_output>\n{pre_fix_pytest_output}\n</pre_fix_pytest_output>\n"
        if pre_fix_pytest_output
        else ""
    )
    feedback_section = (
        f"\n<regen_feedback>\n{alignment_feedback}\n</regen_feedback>\n" if alignment_feedback else ""
    )
    triage_section = (
        f"\n<triage_feedback>\n{triage_feedback}\n</triage_feedback>\n" if triage_feedback else ""
    )
    return f"""You are authoring the instruction.md for a CRAFT benchmark task.

CRAFT measures HOW agents work. The instruction you write describes the task
to an agent that will attempt it in isolation, without reading the reference
tests. Your job is to name the API contract the tests will check, describe
the desired behavior, and make the task genuinely hard — without giving away
the fix.

<candidate>
Repo: {repo}
Commit: {sha}
Merge base: {merge_base_sha}
Subject: {subject}
</candidate>

<evaluation_context>
Why this candidate was accepted: {eval_reason}
Task focus (from evaluator — HARDEST aspect, where models will diverge):
{instruction_sketch}
</evaluation_context>

<repo_map>
{repo_map}
</repo_map>

<pr_diff>
{diff}
</pr_diff>
{pre_fix_section}{feedback_section}{triage_section}
<reference_tests>
{tests_section}
</reference_tests>

<instruction_template>
{instruction_template}
</instruction_template>

<instruction_example>
{instruction_example}
</instruction_example>

<h_rules>
{rubrics.RUBRIC_H_RULES}
</h_rules>

<v4_audit>
{rubrics.RUBRIC_V4_AUDIT}
</v4_audit>

<design_principles>
{rubrics.RUBRIC_DESIGN_PRINCIPLES}
</design_principles>

<anti_leakage>
{rubrics.RUBRIC_ANTI_LEAKAGE}
</anti_leakage>

<policy>
Produce two fields in the response JSON:

1. instruction_md — the full content of the task's instruction.md file.
   - Start with the first line of instruction_template verbatim. Replace the
     template's "(the task description)" placeholder with your description.
   - Do NOT add ## Environment, Run tests:, or any other sections from the
     example — the pipeline handles environment/Dockerfile separately.
   - Target length: {rubrics.INSTRUCTION_WORD_RANGE}. Hard maximum
     {rubrics.INSTRUCTION_WORD_HARD_MAX} words.
   - Apply every H-rule above and every anti-leakage rule. Do not reveal the
     fix; do not name private methods, internal helpers, or magic numbers that
     the reference tests don't need to import.
   - If the PR bundles multiple separable features, target the integration /
     discriminating aspect per design_principles. A 2-test single-method task
     carved from a multi-file commit means the wrong slice was picked.
   - V4 fixtures-layer rule: every design choice that appears in
     reference_tests fixtures MUST appear in instruction_md (the agent cannot
     see the tests). Design choices in assertions (internal attributes, error
     message wording) may be left for discovery.

2. task_slug — a short descriptive slug for the task directory name,
   kebab-case, 2-5 words. Examples: "sslmode-dsn", "bm25-kwargs-parallel-fix",
   "pydantic-alias-choices".

Do not fabricate names for files, tests, or symbols that don't appear in the
provided repo_map or reference_tests. The only symbols you may name in
instruction_md are those the reference tests import (public API contract).

If a <regen_feedback> block is present above, the alignment judge rejected a
prior draft of this instruction. Treat its guidance as authoritative: do not
restate the flagged phrases or their equivalents in the regenerated draft.

If a <triage_feedback> block is present above, a prior draft of this
instruction passed alignment and smoke but the post-execution fairness
reviewer found an instruction-test misalignment. Treat the block's own
guidance as authoritative — it frames the cited test as a representative
example, not the whole scope. Preserve the original task's scope; broaden
coverage so the unstated behavior is specified. Re-describe the missing
behavior in your own words (name the behavior, not the test). Continue
to respect anti-leakage rules.
</policy>

Return a single JSON object matching the required schema. No markdown fences,
no extra text."""


def alignment_judge_prompt(
    instruction_md: str,
    reference_test_bodies: list[tuple[str, str]],
    diff: str,
) -> str:
    """Build-time alignment-judge prompt for direct-API dispatch.

    Runs after Build, before assemble-artifacts. Primary concern: instruction ↔
    reference-test alignment (does the instruction specify enough for the agent
    to pass tests a senior engineer would write the same way?). Secondary:
    leakage (does the instruction reveal the implementation path the PR took?).

    Model is cross-family from Build (Opus generates → GPT judges) to mitigate
    self-preference bias (arXiv 2410.21819).
    """
    tests_section = "\n\n".join(
        f'<test path="{path}">\n{body}\n</test>' for path, body in reference_test_bodies
    )
    return f"""You are auditing a CRAFT benchmark task's instruction ↔ reference-test alignment.

<instruction>
{instruction_md}
</instruction>

<reference_tests>
{tests_section}
</reference_tests>

<pr_diff>
{diff}
</pr_diff>

<v4_audit>
{rubrics.RUBRIC_V4_AUDIT}
</v4_audit>

<categories>
{rubrics.RUBRIC_ALIGNMENT_CATEGORIES}
</categories>

<policy>
Assess the task along two axes, primary first:

Primary — **misalignment / narrow tests**: do the reference tests require
something the instruction doesn't specify? Apply the V4 three-layer audit
(fixtures, helpers, assertions). This is the unfair-hard failure mode where
a senior engineer reading only the instruction would write a valid
implementation that nonetheless fails tests — because tests encode design
choices, private symbol names, or exact format details the instruction
omits.

Secondary — **leakage**: does the instruction reveal the implementation
path, the fix itself, or diagnostic detail that would let a reader bypass
the reasoning work? (This is the Aleithan 2410.06992 / SWE-bench+
issue-body-leakage failure mode.)

Choose exactly one verdict from the category enum. Order of preference
when multiple apply:
  1. `narrow_tests` — instruction omits something tests require (primary)
  2. `misaligned` — instruction and tests relate to different problems
  3. `leaked` — instruction over-specifies the fix
  4. `vague` — instruction too under-specified to predict any test
  5. `ok` — neither under- nor over-specified

Citation requirement: your `reason` field MUST quote specific evidence from
one of the three context blocks (instruction snippet, test snippet, diff
snippet) that supports the verdict. No gut-feel "looks ok" / "seems off".

On `leaked`: populate `leakage_evidence` with 1-3 exact quotes from the
instruction that reveal implementation path.

On any non-`ok` verdict involving tests: populate `v4_audit` with booleans
for the three layers (fixtures_encode_design_choices, helpers_access_private_api,
assertions_format_only). `True` = V4 violation detected. Omit `v4_audit`
for `ok` or purely leakage-based verdicts.

Retention-biased posture: rejection wastes a completed Build. When
genuinely uncertain between `ok` and another category — and no clear
evidence supports the non-`ok` claim — lean toward `ok`. Downstream
(Opus smoke, post-execution reviewer) provides another chance to catch
problems with execution evidence.
</policy>

Return a single JSON object matching the required schema. No markdown fences,
no extra text."""


def easiness_triage_feedback_block(grep_read_count: int, easiness_reason: str) -> str:
    """Assemble the <triage_feedback> block body for an easiness-triggered regen.

    Fires when Opus solved the task with reward=1.0 but very low
    trajectory-exploration (grep_read count at or below the
    `_EASINESS_GREP_READ_MAX` threshold). Strong signal that the
    instruction is prescriptive — naming the file, class, data
    structure, or diagnosis so literally that the agent didn't need to
    explore. Asks Build to rewrite at a higher level of abstraction,
    forcing an agent to do real search work to locate the fix.
    """
    return (
        "The previous instruction passed Opus smoke with full reward (1.0), "
        "but the agent reached the solution with only "
        f"{grep_read_count} Grep/Read exploration call(s) "
        f"({easiness_reason}). That is well below the threshold for a "
        "genuine CRAFT-scale task and indicates the instruction is too "
        "prescriptive — it probably names the file, class, data structure, "
        "or exact fix strategy directly, letting the agent bypass the "
        "search/diagnosis work the task is meant to measure.\n\n"
        "Rewrite the instruction at a higher level of abstraction so a "
        "competent senior engineer would have to explore the repo to find "
        "the code that needs fixing. Strip out any of the following that "
        "appear in the current draft:\n"
        "  - Named file paths or filenames (unless the test imports "
        "require the symbol and there is no equivalent name-free framing).\n"
        "  - Named classes, methods, or private helpers whose location "
        "the agent should have to discover.\n"
        "  - The specific data structure or algorithm that the fix uses "
        "(e.g. 'track visited IDs in a set' → reduce to 'prevent "
        "infinite recursion').\n"
        "  - Step-by-step procedural language that reads like a recipe.\n\n"
        "Keep: the outcome / behavior the fix must deliver, any API "
        "contract the reference tests rely on (names/signatures the "
        "tests import), and the anti-leakage rules from the construction "
        "rubric. Do not change the task's scope — only its phrasing and "
        "specificity."
    )


def reviewer_triage_feedback_block(reason: str, quote: str, test: str) -> str:
    """Assemble the <triage_feedback> block body from a fairness-review verdict.

    The reviewer emits one representative example (quote + test). This
    narrative framing tells Build explicitly that the example is a
    representative sample, not the whole scope. A bulleted per-test list
    would bias Build toward narrowing the task to the single cited test.
    """
    reason = reason.strip() or "(no reason provided)"
    quote = quote.strip() or "(no quote provided)"
    test = test.strip() or "(no test cited)"
    return (
        "The fairness reviewer flagged an instruction-test misalignment. "
        "The item below is ONE representative example — similar unstated "
        "behavior may affect other failing tests in the same area. Preserve "
        "all existing requirements from the current instruction; broaden it "
        "so the unstated behavior is specified.\n\n"
        f"Reviewer's reason:\n  {reason}\n\n"
        f'Instruction sentence that is under-specified:\n  "{quote}"\n\n'
        f"Representative failing test that depends on unstated behavior:\n  {test}\n\n"
        "Action: describe the missing behavior in your own words at the "
        "right level of abstraction (name the behavior, not the test). Do "
        "NOT narrow the instruction's scope to just this example — the "
        "original task is still the task; you are adding coverage, not "
        "replacing the focus. Continue to respect anti-leakage rules."
    )


def build_dockerfile_prompt(
    repo: str,
    sha: str,
    merge_base_sha: str,
    task_dir: str,
    ctx: PipelineContext | None = None,
) -> str:
    """Focused prompt for Dockerfile creation only (separated from instruction build)."""
    ctx = ctx or PipelineContext()
    return f"""Create the environment/Dockerfile for an existing CRAFT benchmark task.

Read these BEFORE doing anything:
1. {ctx.task_building_guide} — Dockerfile conventions section
2. {ctx.template_task_dir}/environment/Dockerfile — structural guide; inline comments mark what to adapt

Task directory: {task_dir}
Repo: repos/{repo}
Merge base SHA (clone at this exact state): {merge_base_sha}
Commit SHA (the fix): {sha}

Read {task_dir}/instruction.md to understand what the task requires.

To build the correct Dockerfile, read at merge base SHA:
- setup.py / pyproject.toml / requirements*.txt — install command, extras, Python version (python_requires)
- .github/workflows/, tox.ini, Makefile — often reveal required system packages (apt-get installs)

Run `git -C repos/{repo} remote get-url origin` to get the GitHub clone URL for the Dockerfile.

CREATE: {task_dir}/environment/Dockerfile

Dockerfile requirements:
- Clone repos/{repo} at merge base SHA {merge_base_sha} (exact pre-change state)
- Install ALL deps + test deps (pytest, pytest-timeout, any repo-specific extras)
- Reset git: rm -rf .git && git init && git config user.email test@test.com &&
  git config user.name Test && git add -A && git commit -m init
- Create /code/output/

The Dockerfile is the ONLY file to create. Do not modify any other files."""


def deep_dive_prompt(
    instruction_md: str,
    reward_json: str,
    verify_output_tail: str,
    postmerge_test_bodies: list[tuple[str, str]],
    harbor_lab_errors: str,
    harbor_lab_edits: str,
    harbor_lab_tool_sequence: str,
    harbor_lab_metrics: str,
    f2p_tests: str = "",
    p2p_tests: str = "",
    f2p_skip: str = "",
    p2p_skip: str = "",
    triage_history: str = "",
) -> str:
    """Direct-API deep-dive prompt — per-test `skip` or `keep` verdict only.

    Scope: classify each failing test as `skip` (test should be excluded
    from scoring) or `keep` (counted). A separate fairness-reviewer step
    decides whether the instruction itself needs regeneration; that
    concern is explicitly outside this prompt. Runs against the Opus
    step model.

    All context pre-assembled by the caller; the LLM has no file tools.
    """
    tests_section = "\n\n".join(
        f'<test path="{path}">\n{body}\n</test>' for path, body in postmerge_test_bodies
    )

    history_section = ""
    if triage_history:
        history_section = f"\n<previous_triage_history>\n{triage_history}\n</previous_triage_history>\n"

    skip_section = ""
    if f2p_skip.strip() or p2p_skip.strip():
        skip_section = f"\n<f2p_skip>\n{f2p_skip}\n</f2p_skip>\n<p2p_skip>\n{p2p_skip}\n</p2p_skip>\n"

    return f"""You are auditing a failed Harbor trial for a CRAFT benchmark task.
For each failing reference test, return exactly one verdict: `skip` or
`keep`. A separate fairness-review step handles concerns about whether
the instruction itself was fair — do not comment on instruction quality
here.
{history_section}
<instruction>
{instruction_md}
</instruction>

<reference_tests>
{tests_section}
</reference_tests>

<f2p_tests>
{f2p_tests}
</f2p_tests>

<p2p_tests>
{p2p_tests}
</p2p_tests>
{skip_section}
<reward_json>
{reward_json}
</reward_json>

<verify_output_tail>
{verify_output_tail}
</verify_output_tail>

<harbor_lab_errors>
{harbor_lab_errors}
</harbor_lab_errors>

<harbor_lab_edits>
{harbor_lab_edits}
</harbor_lab_edits>

<harbor_lab_tool_sequence>
{harbor_lab_tool_sequence}
</harbor_lab_tool_sequence>

<harbor_lab_metrics>
{harbor_lab_metrics}
</harbor_lab_metrics>

<infra_check>
If harbor_lab_metrics shows output_tokens = 0 and harbor_lab_edits shows
no source file changes, the agent never meaningfully engaged. Return an
empty `failures[]` and put the infra note in `overall_assessment`.
</infra_check>

<scoping>
Classify only tests that actually FAILED in this trial. A test is a
failure if it appears as FAILED or ERROR in verify_output_tail and its
name is in f2p_tests or p2p_tests. Skip tests that passed, tests outside
those two lists, and tests already named in f2p_skip / p2p_skip.
</scoping>

<verdict_definitions>
`skip` — the failing test does not belong in the scored set. Applies when
any of the following hold:

- The test exercises bundled regression behavior unrelated to the feature
  described in the instruction. (Example: a streaming-fix task with a
  non-streaming post-yield exception test bundled in the same commit.)
- The test asserts an exact format or magic string (log text, error
  message wording, formatting) that the instruction does not specify,
  such that a semantically correct solution would still fail.
- The test imports a private symbol or calls a private helper that the
  instruction does not name, in a way that forces one specific
  implementation path. A correct alternative would not satisfy the
  import.

`keep` — the failure is a legitimate reasoning gap or preservation
failure that the scored set should retain:

- The test exercises behavior the instruction explicitly asks for, and
  the agent's implementation does not deliver it.
- A P2P test broke because the agent's change regressed existing
  behavior. (P2P failures are almost always `keep` — the agent's job
  includes preservation.)
- The test expectation is behavioral (not format-only) and a senior
  engineer reading only the instruction would have produced a solution
  that passes it.

Default bias: prefer `keep` when neither bucket cleanly fits. A
false-`skip` silently loses test coverage; a false-`keep` surfaces later
through the fairness-review step or human audit.
</verdict_definitions>

<workflow>
For each failing test, cross-reference: test expectation (from
reference_tests) × agent's source edits (from harbor_lab_edits) ×
instruction text. Decide `skip` or `keep` and populate `evidence` with a
one-sentence justification. Verdict comes first, reasoning after.

Test-file edits are irrelevant: the verifier overlays its own
`postmerge_tests/` at verification time, so any agent edits to test
files have zero effect on the score. Ignore them when classifying.

If reward_json f2p/p2p counts disagree with verify_output_tail, note
the mismatch in `overall_assessment`.
</workflow>

Return one JSON object matching the schema. `failures[]` has one entry
per failing test with `test_name`, `classification`, and `evidence`.
`overall_assessment` is a one-line summary. No markdown fences, no extra
prose."""


def fairness_review_prompt(
    instruction_md: str,
    reward_json: str,
    verify_output_tail: str,
    postmerge_test_bodies: list[tuple[str, str]],
    harbor_lab_errors: str,
    harbor_lab_edits: str,
    harbor_lab_tool_sequence: str,
    f2p_tests: str = "",
    p2p_tests: str = "",
) -> str:
    """Triage-time fairness-review prompt (cross-family, GPT-5.x default).

    Single task-level verdict: is the agent's trial failure caused by an
    unfair instruction under-specification? Severity-gated; `major`
    requires both a verbatim instruction quote AND a named failing test
    that depends on unstated behavior. If either piece of evidence is
    missing, the judge is instructed to downgrade to `minor` or `none`.

    Replaces the per-test dual-DD secondary classifier (which had a
    systematic bias toward `instruction_scope` verdicts that the old
    merge logic silently dropped). The fairness review is advisory by
    default — only `severity=major` with both evidence fields triggers a
    one-shot Build regen. Everything else sets a `reviewer_concern_flag`
    (soft signal mirroring `easiness_flag`) and ships the task as-is.

    Model default: GPT-5.4 via `LLM_ALIGNMENT_MODEL` (cross-family from
    Opus-generated instruction). Prompt structure follows OpenAI 2026
    conventions — markdown section headers, literal definition of terms,
    affirmative evidence requirements. No XML nesting, no all-caps
    emphasis, no chain-of-thought scaffolding before the verdict.
    """
    tests_section = "\n\n".join(
        f"### {path}\n\n```python\n{body}\n```" for path, body in postmerge_test_bodies
    )
    return f"""You are a fairness reviewer for a CRAFT benchmark task whose agent trial failed.

# Your task

Return one severity verdict about the instruction. The verdict describes
whether the trial failure is attributable to an unfair instruction
under-specification, not to the agent's capability.

Output the severity verdict first; any justification prose belongs in
the `reason` field.

# Severity definitions

{rubrics.RUBRIC_FAIRNESS_REVIEW}

# Instruction

{instruction_md}

# Reference tests

{tests_section}

# F2P tests (fail→pass — must be resolved by the agent)

{f2p_tests or "(none listed)"}

# P2P tests (pass→pass — must stay green)

{p2p_tests or "(none listed)"}

# Reward summary

{reward_json}

# Pytest output tail

{verify_output_tail}

# Agent errors (harbor-lab)

{harbor_lab_errors}

# Agent source edits (harbor-lab)

{harbor_lab_edits}

# Agent tool sequence (harbor-lab, tail)

{harbor_lab_tool_sequence}

# Output format

Return one JSON object with these fields:

- `severity` — one of `none`, `minor`, `major`.
- `reason` — one sentence explaining the verdict. Do not quote the
  entire instruction; reference the parts that matter.
- `evidence_quote` — (required for `major`) a verbatim sentence or
  phrase copied from the instruction. Use the empty string for `none`
  or `minor`.
- `evidence_test` — (required for `major`) the full pytest id of a
  failing reference test that depends on behavior not covered by the
  quoted instruction sentence. Use the empty string for `none` or
  `minor`.

If `severity == "major"` but you cannot produce both `evidence_quote`
and `evidence_test` with specific content, downgrade the verdict to
`minor` or `none`. Do not fabricate evidence to justify `major`.

Return raw JSON only. No markdown fences, no extra prose."""


# Legacy schema + prompt for the skeptical reviewer stage, retained
# only so `scripts/calibrate-deep-dive.py` can replay historical
# cohorts against the old DD+reviewer flow. No production code path
# consumes these.
REVIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "challenges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "original_classification": {"type": "string"},
                    "revised_classification": {"type": "string"},
                    "challenge_question": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["test_name", "original_classification", "revised_classification"],
            },
        },
        "pass_audit": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "passed_for_right_reason": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["test_name", "passed_for_right_reason"],
            },
        },
        "verdict": {
            "type": "string",
            "enum": ["accept", "skip-unfair-tests", "reject"],
        },
        "overall_verdict": {"type": "string"},
        "recommended_fixes": {"type": "array", "items": {"type": "string"}},
        "easiness_concern": {"type": "boolean"},
        "easiness_reason": {"type": "string"},
    },
    "required": ["challenges", "pass_audit", "verdict", "overall_verdict"],
}


def skeptical_reviewer_prompt(
    instruction_md: str,
    reference_test_bodies: list[tuple[str, str]],
    deep_dive_output: dict,
    score: str,
    verify_output_tail: str = "",
    trajectory_tail: str = "",
    other_model_score: str = "",
    other_model_name: str = "",
) -> str:
    """Direct-API skeptical reviewer prompt (legacy — calibration use only).

    No longer called from the live triage path; retained so
    scripts/calibrate-deep-dive.py can still replay the historical
    deep-dive + reviewer flow against the Apr 16-19 2026 cohort.

    All context pre-assembled by the caller — LLM has no file tools and
    cannot Read/Grep. Called against the model the caller passes in
    (calibration scripts override via `--reviewer-model`).

    Reviewer inputs are raw evidence only — instruction, reference tests,
    deep-dive findings, verifier output, agent trajectory. Outputs from prior
    pipeline steps (diagnostics/*.md) are deliberately excluded so the
    reviewer re-judges from the primary artifacts instead of inheriting
    earlier verdicts.
    """
    failures_text = ""
    for f in deep_dive_output.get("failures", []):
        failures_text += (
            f"\n- {f.get('test_name', '?')}: {f.get('classification', '?')}"
            f"\n  Evidence: {f.get('evidence', 'none')[:200]}"
        )

    tests_section = "\n\n".join(
        f'<test path="{path}">\n{body}\n</test>' for path, body in reference_test_bodies
    )

    rank_inversion_warning = ""
    if other_model_score and other_model_name:
        rank_inversion_warning = f"""
<rank_inversion_warning>
{other_model_name} scored {other_model_score} on this same task. If a weaker
model outscores a stronger model, it is VERY LIKELY an instruction/verifier
issue — the test is probably checking an unstated design choice that one
model happened to match. Be EXTRA skeptical of "genuine_gap" classifications
when there's a rank inversion. Most common cause: both models implemented
the feature correctly but made different integration choices, and the
verifier only accepts one of them.
</rank_inversion_warning>
"""

    trajectory_section = (
        f"\n<agent_trajectory_tail>\n{trajectory_tail}\n</agent_trajectory_tail>\n" if trajectory_tail else ""
    )
    verify_section = (
        f"\n<verify_output_tail>\n{verify_output_tail}\n</verify_output_tail>\n" if verify_output_tail else ""
    )

    return f"""You are the SKEPTICAL REVIEWER for a CRAFT benchmark task.

Your job is NOT to agree with the deep dive analysis. Challenge every
"genuine_gap" classification and audit every passing test. Imagine a human
benchmark designer who learned from painful experience that initial deep
dives consistently over-classify failures as genuine when they are actually
instruction/verifier issues.

When surprising results appear (Opus fails, rank inversions, unexpected 0/N
scores), the default is: "the instruction or verifier is probably wrong" —
not "this case is just different." Every time humans pushed back on a deep
dive's "genuine gap" call, it turned out to be an instruction/verifier
issue. Keep digging. Read the agent's implementation, compare to test
expectations, find the gap.
{rank_inversion_warning}
<instruction>
{instruction_md}
</instruction>

<reference_tests>
{tests_section}
</reference_tests>

<deep_dive_findings score="{score}">
Overall assessment: {deep_dive_output.get("overall_assessment", "")}

Per-failure classifications:{failures_text}
</deep_dive_findings>
{verify_section}{trajectory_section}
<challenge_questions>
{CHALLENGE_QUESTIONS}
</challenge_questions>

<classification_patterns>
Concrete patterns, grounded in prior adjudication. When a failure's error
signature matches one of the patterns below, reclassify even if the deep
dive called it `genuine_gap`.

Decision procedure — apply to every failure, in this order:
  1. If 3+ tests in the same class/file share an identical signature-
     level error AND each one exercises a different pre-existing
     behavior unrelated to the instructed feature → Pattern D
     (test_not_relevant).
  2. Otherwise, if the error is AttributeError / NameError / ImportError
     on a specific symbol, AND that symbol name does not appear in the
     instruction (search literally) → Pattern A
     (instruction_missing_symbol).
  3. Otherwise, if the error is TypeError on argument binding, or an
     assertion on an exact string / format the instruction does not
     prescribe → Pattern B (instruction_scope). Optionally narrow to
     test_format_only or ambiguous if a more specific category fits.
  4. Otherwise → Pattern C (genuine_gap, leave as-is).

Pattern A — instruction_missing_symbol (skip-worthy)
  Error signature: AttributeError / NameError / ImportError on a specific
  symbol (function name, class name, module attribute, import path).
  Required conditions (all three):
    1. The exact failing symbol name does not appear anywhere in the
       <instruction>. Search the instruction literally for the symbol.
    2. The agent never created that exact name — they chose a different
       name or architecture to satisfy the behavioral requirement.
    3. The behavior the test exercises IS described in the instruction;
       only the API shape (the specific symbol) is not.
  Example: test calls `tree_builder.refresh_tree_incremental(host)` and
  fails with "module has no attribute 'refresh_tree_incremental'". The
  instruction says "fix the refresh logic" without naming that function.
  Agent modified existing `refresh_tree()` instead. → instruction_missing_symbol.
  Counter-example: if the test fails because the agent's `refresh_tree()`
  produces wrong output (assertion on state, not AttributeError), that is
  genuine_gap — the symbol exists, the logic is wrong.

Pattern B — instruction_scope (skip-worthy when the test is the only offender)
  Error signature: TypeError on argument binding, assertion on exact
  string format, or test requires a specific integration point / calling
  convention / parameter ordering.
  Required conditions (all three):
    1. The instruction names the feature / function / parameter but
       does NOT specify the exact calling convention the test assumes
       (positional order, keyword vs positional, exact message wording,
       exact exception class hierarchy).
    2. The agent's implementation is a defensible alternative — a senior
       engineer reading only the instruction could plausibly produce it.
    3. Other tests for the same feature pass (or would pass) with the
       agent's alternative, demonstrating the behavior itself works.
  Example: instruction says "_chunk_actions must accept flush_after_seconds";
  test calls `_chunk_actions(actions, chunk_size, max_bytes, flush_seconds,
  serializer)` (flush as 4th positional); agent wrote `_chunk_actions(...,
  serializer, flush_after_seconds=None)` (new param at end, classic
  backward-compat). Instruction never specified ordering → instruction_scope.
  Example: instruction says "invalid mode raises ValueError"; test uses
  `pytest.raises(ValueError, match='mode must be')`; agent raises
  ValueError with "Invalid mode..." message. → instruction_scope (or
  test_format_only — both are SKIP-WORTHY).

Pattern C — genuine_gap (NOT skip-worthy, real capability miss)
  Error signature: assertion failure on state or output the instruction
  explicitly describes; agent's code runs but produces wrong behavior.
  Required condition: the failing behavior is named in the instruction
  AND the symbol/API the test uses exists in the agent's code (no
  AttributeError, no NameError). The agent wrote code that satisfies the
  structural contract but gets the logic wrong.
  Example: instruction says "cursor must be preserved when a connection
  moves between folders"; test moves connection, asserts cursor is not
  None, fails with `assert None is not None`. The agent's code runs (no
  AttributeError) but the cursor-restore logic doesn't handle the
  cross-folder case. → genuine_gap.

Pattern D — test_not_relevant (skip-worthy bundled regression)
  Error signature: several tests in the same test class/file fail with
  the same signature-level root cause, but each test exercises a
  DIFFERENT behavior that predates the feature in the instruction.
  Required conditions:
    1. Multiple tests fail with an identical surface error (all hit a
       TypeError on the same new parameter, or all fail at the same
       constructor call).
    2. The tests themselves exercise behaviors unrelated to the feature
       described in the instruction (e.g., locking, error paths, config
       loading, filesystem edge cases).
    3. Tests pass a DEFAULT / EMPTY / SENTINEL value for the new
       parameter — they don't actually exercise the new feature, they
       just got caught by the signature change.
    4. The instruction mentions none of the behaviors the tests cover.
  Example: instruction says "fix installer to mark packages explicit";
  9 tests in `TestScheduleBuilds` all call `schedule_builds(explicit=set())`
  (empty set — not exercising the feature). Tests are really about locks,
  overwrites, and jobserver tokens. → test_not_relevant for all 9.
  Counter-example: one test in the same class calls `schedule_builds(
  explicit={{spec.dag_hash()}})` (non-empty) and asserts on mark-explicit
  behavior. That one exercises the feature → instruction_scope (valid
  alternative API) or genuine_gap, but not test_not_relevant.

Tie-breaker for Pattern A vs Pattern B when both could apply: if the
agent's alternative architecture moves equivalent functionality to a
different layer and the test asserts on both existence AND behavior of a
specific named symbol, prefer Pattern A (instruction_missing_symbol) —
it is more precise about the failure and feeds cleaner skip-list entries.
</classification_patterns>

<workflow>
STEP 1 — CHALLENGE genuine_gap classifications: for each failure classified
as genuine_gap, run the decision procedure from <classification_patterns>
above (Pattern D → A → B → C, in that order) AND apply the 7 challenge
questions. Record every reclassification in `challenges[]` with
`revised_classification` set to the matched pattern's category.

STEP 2 — AUDIT passing tests (Q6 specifically): for each passing test, did
it pass for the right reason — a direct link from instruction to the
agent's design choice? Or did the agent guess right? Record in pass_audit[].

STEP 3 — EASINESS CHECK (only if score indicates reward=1.0): if the agent
trajectory (see trajectory_tail above when present) shows recipe-following
patterns — wrote code purely from instruction without exploring existing
tests, no test-observe-adapt loops, first-try perfect solution on
non-trivial work — set easiness_concern=true. Skip entirely if score < 1.0;
a task that stumped the agent is not trivially easy.

STEP 4 — VERDICT: produce `verdict` from one of:
  - `accept` — deep dive's classifications hold up, instruction + tests are
    aligned, no concerning pass patterns. Task advances.
  - `skip-unfair-tests` — one or more reference tests require things the
    instruction does not specify, but other tests are legit. Pipeline will
    auto-write these into f2p_skip.txt / p2p_skip.txt and re-score. Name
    the specific tests in `challenges` with revised_classification ∈
    {{test_not_relevant, test_format_only, ambiguous, instruction_scope}}.
    If rescoring would leave ≤ 1 F2P test, this becomes a reject.
  - `reject` — the task has fundamental problems (wrong slice of PR,
    instruction describes one feature but tests check another, too many
    unfair tests to salvage). Shelve.
</workflow>

Return a single JSON object matching the required schema. No markdown
fences, no extra text."""


_SPECIFICITY_TRADEOFF = """\
THE SPECIFICITY TRADEOFF (critical — read this):
If instruction is too vague, agents create valid solutions the verifier can't recognize
(false negatives). If instruction is too specific, the task becomes trivially easy.
The sweet spot: name the API CONTRACT (class names, module paths, config keys that tests
import — ~20% given free) but leave the IMPLEMENTATION open (integration, wiring, logic
— the remaining ~80% that's genuinely hard). For example, naming a required class and
its module path is free; figuring out how to integrate it into the existing code is the
real challenge."""


def fix_docker_prompt(
    task_dir: str, issue: str, attempt: int, history: str = "", ctx: PipelineContext | None = None
) -> str:
    """Fix prompt for Docker build failures.

    Only environment/Dockerfile is in scope. Do not create or modify any other files.
    """
    ctx = ctx or PipelineContext()
    history_section = ""
    if history:
        history_section = f"""
PREVIOUS FIX ATTEMPTS (don't repeat what didn't work):
{history}
"""

    return f"""Fix the Dockerfile for this CRAFT benchmark task. This is attempt {attempt}.

Task directory: {task_dir}

Read: {task_dir}/environment/Dockerfile

ONLY environment/Dockerfile is in scope. Do not modify any other files.

PROBLEM TO FIX:
{issue}
{history_section}
FULL DIAGNOSTICS: If {task_dir}/diagnostics/ exists, read ALL .md files in
{task_dir}/diagnostics/ BEFORE making changes. They contain recommended fixes, and full output
from prior fix attempts. May contain the deep dive agent's per-test evidence, the skeptical
reviewer's challenges and reclassifications. Files are numbered chronologically — read them in
order.

COMMON CAUSES:
- Missing dependencies — read the repo's setup.py / pyproject.toml / requirements*.txt at the
  merge base SHA (find the repo from the Dockerfile's git clone URL) and ensure ALL deps
  including test extras (dev, test, ci) are installed
- Missing system packages — if the error is a compile/link failure or missing header, check
  .github/workflows/, tox.ini, Makefile at merge base SHA for apt-get installs the CI uses
- Wrong base SHA — must be the exact pre-change commit, not HEAD or the fix commit SHA
- Wrong repo URL — must match the actual path under repos/

Dockerfile requirements:
- Clone repos/<repo> at the correct merge base SHA (exact pre-change state)
- Install ALL deps + test deps (pytest, pytest-timeout, any repo-specific extras)
- Reset git: rm -rf .git && git init && git config user.email test@test.com &&
  git config user.name Test && git add -A && git commit -m init
- Create /code/output/

After fixing: confirm the base SHA and repo URL match the task."""


def fix_f2p_p2p_classify_prompt(
    task_dir: str, issue: str, attempt: int, history: str = "", ctx: PipelineContext | None = None
) -> str:
    """Fix prompt for F2P/P2P classification failures.

    Classification fails when the Docker container can't run tests correctly — almost always
    a Dockerfile issue (wrong base SHA, missing deps) or a test discovery problem.
    instruction.md and skip lists are not in scope here.
    """
    ctx = ctx or PipelineContext()
    history_section = ""
    if history:
        history_section = f"""
PREVIOUS FIX ATTEMPTS (don't repeat what didn't work):
{history}
"""

    opening = (
        f"Fix the F2P/P2P classification failure for this CRAFT benchmark task. This is attempt {attempt}."
    )
    return f"""{opening}

Task directory: {task_dir}

Read: {task_dir}/environment/Dockerfile
Read: {task_dir}/tests/ directory listing (to check which test files exist and are being targeted)

Only environment/Dockerfile and test discovery configuration are in scope.
Do not modify instruction.md or skip lists.

PROBLEM TO FIX:
{issue}
{history_section}
FULL DIAGNOSTICS: If {task_dir}/diagnostics/ exists, read ALL .md files in
{task_dir}/diagnostics/ BEFORE making changes. They contain recommended fixes, and full output
from prior fix attempts. May contain the deep dive agent's per-test evidence, the skeptical
reviewer's challenges and reclassifications. Files are numbered chronologically — read them in
order.

COMMON CAUSES:
- Patch failed to apply (SOLVE_FAIL) — Dockerfile is cloning at the wrong base SHA.
  The repo state in the container must match the SHA the patch was generated from.
- Oracle passed 0 tests (ORACLE_ZERO) — either missing deps prevent pytest from collecting,
  or the test discovery paths are targeting the wrong files.
- Timeout — Docker container exceeded the time limit, often due to slow dep installation.
  Consider pinning faster dep versions or reducing install scope.

After fixing: confirm base SHA in Dockerfile matches the merge base SHA in the task."""


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def task_summary_prompt(diagnostics: list[tuple[str, str]], stage: str) -> str:
    """Direct-API task-summary prompt.

    `diagnostics` is a pre-assembled list of `(filename, body)` tuples — the
    direct-API judge has no Read tool, so the caller inlines every
    diagnostic file's content.
    """
    diag_section = "\n\n".join(
        f'<diagnostic file="{name}">\n{body}\n</diagnostic>' for name, body in diagnostics
    )

    return f"""You are summarizing the full run of an automated CRAFT benchmark task.

Pipeline flow: build task → alignment check → assemble artifacts → build
Dockerfile → F2P/P2P classify → oracle check → Opus smoke → triage (deep
dive + reviewer + auto-fix loop) → accept/reject.

Diagnostic files below are the per-step outputs in chronological order:
fix files show what changed, triage files show per-failure classification,
review files show reviewer challenges.

<diagnostics>
{diag_section}
</diagnostics>

<final_stage>{stage}</final_stage>

Write a ONE LINE summary (max 20 words) of what happened to this task,
focused on the outcome story — like a human describing the pipeline run.
Tone examples:
- Accepted after 2 verifier fixes, Opus 8/8 F2P, clean genuine gaps
- Stuck in alignment fix loop, instruction leakage persists after rewrites, rejected
- Opus F2P 0/5, all failures genuine_gap — rejected as too hard to specify
- Sailed through clean but efficiency flag tripped — may not discriminate

Include scores if available. No code details. Return JSON:
{{"summary": "<one line>"}}"""

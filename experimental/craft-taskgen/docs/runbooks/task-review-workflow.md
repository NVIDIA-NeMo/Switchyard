# Task review workflow

End-to-end recipe for auditing CRAFT benchmark tasks across multi-trial
multi-model results, using the per-task markdowns produced by
`scripts/task_review.py` plus the `harbor-f2p-p2p-deep-dive` skill bundled
in `.claude/skills/`.

The workflow targets the case "we have N trials × M models on the v2b
cohort; for the tasks that no model resolved, decide whether they're fair
capability tests, broken verifiers, or something in between."

## Setup

### 1. Download trial trajectories

Trial tarballs are shared out of band — ask the maintainers for the
current set.

Drop them into `data/` at the top of this repo. Untar in place:

```bash
mkdir -p data/extracted
cd data && for t in *.tar.gz; do tar xzf "$t" -C extracted/; done && cd ..
```

Each tarball expands into `data/extracted/jobs/<run_dir>/` (and possibly
`<run_dir>/iter<N>/` for multi-iter rescue runs).

### 2. Check out craft-bench

The audit needs the harbor task dirs (instruction.md, solution patch,
verifier code) so the per-task markdown can inline the instruction and so
the deep-dive skill can find the reference solution. Clone craft-bench
as a sibling of craft-taskgen:

```bash
# from craft-taskgen's parent dir
git clone <craft-bench-repo-url>
```

### 3. Check out and install harbor-lab

The deep-dive skill uses `harbor-lab` for trajectory analysis (`$HLAB
errors`, `tool-sequence`, `edits`, etc.). Clone and install it once:

```bash
# from craft-taskgen's parent dir
git clone <harbor-lab-repo-url>
cd harbor-lab
pip install uv
uv pip install -e ".[dev]"
```

This produces `harbor-lab/.venv/bin/harbor-lab`. The skill resolves it
automatically when the session is started from a craft-taskgen checkout
that has `harbor-lab` as a sibling. If you want it on `PATH` directly,
either add `harbor-lab/.venv/bin` to `PATH` or run the skill from inside
the harbor-lab repo (where `uv run harbor-lab` works).

### Expected layout after setup

```
parent/
├── craft-taskgen/              ← you are here
│   └── data/extracted/jobs/    ← untarred trial roots
├── craft-bench/
│   └── harbor-tasks/
│       └── craft-taskgen-v2b/  ← the 92-task cohort
└── harbor-lab/
    └── .venv/bin/harbor-lab    ← used by the deep-dive skill
```

## Generate the review artifacts

Run `scripts/task_review.py` with the trial roots, aliases for vllm-served
generic-name models, the cohort task list, and the harbor task root:

```bash
uv run python scripts/task_review.py \
  data/extracted/jobs/v2-opus47-claude/baseline-claude-code-craft-taskgen-v2-20260427-151228 \
  data/extracted/jobs/v2-codex-gpt55/baseline-codex-craft-taskgen-v2-20260427-151235 \
  data/extracted/jobs/v2-haiku-opencode/baseline-opencode-craft-taskgen-v2-20260427-151243 \
  data/extracted/oc-minimax-m2.7-v2-20260427T234745Z/baseline-opencode-craft-taskgen-v2-20260427-184745 \
  data/extracted/jobs/v2-qwen36-opencode/baseline-opencode-craft-taskgen-v2-20260428-064419 \
  data/extracted/jobs/v2b-opus47-claude-high \
  data/extracted/jobs/v2b-codex-gpt55-xhigh \
  data/extracted/jobs/v2b-sonnet46-claude-high-rerun \
  data/extracted/jobs/v2b-qwen35-397b-opencode-rerun \
  data/extracted/jobs/v2b-nemotron3-super-120b-opencode \
  --alias 'v2-opus47-claude=opus-4.7-xhigh' \
  --alias 'v2b-opus47-claude-high=opus-4.7-high' \
  --alias 'v2-codex-gpt55=gpt-5.5-high' \
  --alias 'v2b-codex-gpt55-xhigh=gpt-5.5-xhigh' \
  --alias 'v2-qwen36-opencode=qwen3.6-35b-a3b' \
  --alias 'v2b-qwen35-397b-opencode-rerun=qwen3.5-397b' \
  --alias 'v2b-nemotron3-super-120b-opencode=nemotron-3-super-120b' \
  --tasks-file audit/v2b_92.txt \
  --harbor-tasks-root ../craft-bench/harbor-tasks/craft-taskgen-v2b \
  --output-dir audit/review_artifacts
```

Add multi-trial roots (e.g. `v2b30-*-4iters/`, `v2b24-*-iter1/`) as
additional positional arguments to tighten the universal-fail intersection.
Also see the `## Example` section in `scripts/task_review.py` for an
always-current canonical invocation.

Output:

```
audit/review_artifacts/
├── review_index.csv     ← all 92 tasks, triage spreadsheet
└── review_md/
    └── <task_id>.md     ← per-task self-contained drill-down (only for
                           tasks where no model+trial resolved — typically
                           ~20-25 of the 92)
```

The script prints a sample audit prompt to stderr at the end. Copy it.

## Audit a task

### 1. Open Claude Code from this directory

```bash
cd /path/to/craft-taskgen
claude
```

Starting from the craft-taskgen root makes the bundled
`.claude/skills/harbor-f2p-p2p-deep-dive` skill auto-loadable when the
session asks about CRAFT task auditing.

### 2. Pick a task to audit

Open `audit/review_artifacts/review_index.csv` in Sheets. Sort by
`flagged_for_review` (T at the top), then by `n_universal_fails` or
`best_f2p` to prioritize. The `review_md` column gives the relative path
to the per-task markdown.

### 3. Paste the audit prompt

The script printed a prompt block to stderr. It looks like:

```
You're helping me audit a CRAFT benchmark task. Goal: decide whether the task is
a genuine capability test or whether the instruction/tests are unfair (test scope
mismatch, ambiguous instruction, brittle assertions, P2P regression coupled to
the F2P fix, etc.).

Local paths on this machine:
  - Review markdown for one task: audit/review_artifacts/review_md/<task_id>.md
  - Harbor task dir (instruction, reference patch, verifier):
        ../craft-bench/harbor-tasks/craft-taskgen-v2b/<task_id>/
  - Trial dirs (agent transcripts, per-trial outputs):
        Each trial's full path is listed in the markdown's "Trial outcomes" table —
        use those paths directly; trials are spread across multiple parent dirs.

Steps:
1. Read the markdown end-to-end. Note the full trial_dir paths in the
   "Trial outcomes" table.
2. Open the harbor task dir and inspect:
     - instruction.md (also inlined in the markdown — verify completeness)
     - solution/changes.patch (the reference fix the verifier expects)
     - tests/test.sh and tests/score.py (verifier mechanics)
3. Pick the highest-F2P trial from the markdown's trial table (the row's
   trial_dir column has the full path); open `<trial_dir>/agent/` to see the
   agent's transcript and `<trial_dir>/verifier/` for per-test results.
4. For each universally-failing F2P test: open the test source, decide whether
   it tests behavior the instruction asks for or unrelated functionality.
5. If P2P regressions are listed: open the regressed test, decide whether the
   assertion is overly tight given the instruction's scope.

Final output: a short verdict (keep / revise instruction / revise tests / drop)
with concrete evidence from each step.
```

Edit the markdown filename to point at the specific task you want to audit
(e.g. replace `<task_id>` with `t2v3-AUb452-regex-character-class-support`),
paste into the Claude Code session, and let it run.

### 4. Read the verdict critically

The session produces a per-failing-test root-cause analysis plus a top-level
verdict (keep / revise instruction / revise tests / drop). Treat verdicts as
**one input** to your decision. Multi-trial multi-model evidence and
per-test triangulation are signals; they don't replace human judgment.
Re-run on a few tasks if the first-pass verdict feels off — the same prompt
on the same artifact has shown different verdicts on different runs as
attention shifts to different evidence.

## Example session

Abridged transcript from one audit run on
`t2v3-AUb452-regex-character-class-support` (prompt as printed by the
script above, response below):

> Verdict: KEEP — task is fair, hard, at the capability frontier
>
> **Evidence summary**
>
> *Task scope.* The instruction asks for a regex character-class /
> escape-sequence subsystem (4 explicit subfeatures + composition with
> existing operators). The reference patch is 599 lines across 3 files —
> a substantial single-PR feature, not an isolated bug fix.
>
> *Cohort signal.* pass@k = 0/27 across 10 models, but the task isn't
> unsolvable — it's at the frontier:
> - gpt-5.5-xhigh reached 5/10 F2P with 24/24 P2P clean. All 6
>   "sometimes-passing" F2P tests pass on at least one trial.
> - The 3 P2P regressions are concentrated in the weakest models (haiku,
>   minimax, qwen) and are genuine — `test_validate_invalid` regressed
>   because the agent skipped the `{n}` exact-form quantifier requirement,
>   which the instruction explicitly names.
>
> *Per-test triangulation of the 4 universally-failing F2P tests.* All map
> to instruction text verbatim:
>
> | Test | Instruction line |
> |---|---|
> | `test_negated_class_with_period` | "Negated classes compute their match set against `input_symbols`" |
> | `test_space_in_patterns` | "Space characters must be valid literal tokens in patterns (not ignored by the lexer)" |
> | `test_email_like_regexes` | composition: classes + escapes + negation + spaces — all required |
> | `test_unicode_character_classes` | `[a-z]` ranges; Unicode chars are just characters to the lexer |
>
> No fixture mismatch, no hidden dependency, no over-specific assertion.
> Tests assert public-contract behavior.
>
> *Three-layer audit.* Fixtures = `set(string.printable)` or explicit small
> sets (no hidden state). Helpers = none beyond `create_range` for unicode.
> Assertions = membership checks against the regex's stated language — not
> implementation details.
>
> *P2P scope.* All 24 P2P tests pass for the strongest trials. The
> regressions in weaker models are caused by the agent's own
> under-implementation, not by the instruction forcing a regression.
> Default verdict for P2P (regression = capability gap) holds.
>
> *Prior diagnostics agree.* Opus deep-dive verdict on all 9 failing tests:
> keep. GPT-5.4 fairness review: severity = none.
>
> **One soft caveat (not blocking):** `test_email_like_regexes` is a
> ~100-line bundled test exercising six different patterns; a single bug
> fails the whole test. This is the kind of single-test brittleness the
> deep-dive guidance says to flag-and-move-on (absorbed by F2P_micro
> averaging). Not a reason to drop.
>
> **Recommendation:** Keep as-is. Legitimately hard frontier task on a real
> OSS PR (`caleb531/automata@b452f99`), with an instruction that names every
> required behavior and a verifier whose tests align with those
> requirements. pass@k = 0 here reflects model capability on a 599-line
> feature implementation, not task unfairness.

The session takes ~2 minutes wall clock and reads ~10 files (markdown,
instruction, tests, patch, prior diagnostics, best trial's verifier
output).

## Notes

- Verdicts vary across runs. The same prompt on the same artifact has
  produced different conclusions when attention shifts to different
  evidence (one run flagged a P2P test as "F2P/P2P bleed-through"; a
  later run inspected the instruction more carefully and found the
  feature *was* explicitly named, recasting the regression as a genuine
  capability gap). Re-run when the first verdict feels off.
- The deep-dive skill's `references/verifier-validation.md` has five
  questions that are the load-bearing checklist for any "is this F2P
  test fair" decision. The skill auto-loads when the prompt mentions
  CRAFT auditing; it's also worth reading directly before manually
  reviewing a task.

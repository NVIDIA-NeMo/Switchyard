---
name: harbor-f2p-p2p-deep-dive
description: Deep dive on Harbor trial results for tasks that use SWE-Bench-style F2P (FAIL_TO_PASS) and P2P (PASS_TO_PASS) reference tests. Diagnoses why an agent failed and audits whether a failing task is genuinely hard or unfair (instruction-vs-verifier mismatch). Use when the user asks to analyze, debug, investigate, or deep dive on a Harbor run, trial, or job directory; when the user says "harbor deep dive", "f2p deep dive", or "p2p deep dive"; when a trial scores lower than expected; or when the user wants to assess task fairness, instruction quality, verifier correctness, or whether an F2P or P2P test is reasonable for the instruction given. Works with any Harbor task whose verifier reports F2P/P2P-style results, and with any supported agent (claude-code, opencode, codex). Produces a per-failure root-cause verdict (capability gap / verifier issue / instruction issue) by triangulating verifier output, the agent's implementation, the reference tests, and the instruction.
---

# Harbor F2P/P2P Deep Dive

Diagnose why an agent failed a Harbor task whose verifier uses SWE-Bench-style F2P/P2P reference tests. Don't assume a failure mode from the score — split failures by class (F2P vs P2P) and trace each through the actual implementation. Produce a root-cause summary per failing test.

## Prerequisites

`harbor-lab` must be available. From this repo: `uv run harbor-lab`. From outside the repo: `harbor-lab` if installed on PATH, otherwise `~/path/to/harbor-lab/.venv/bin/harbor-lab`. Set a `HLAB` shell variable to whichever form works (e.g. `HLAB="uv run harbor-lab"`) and use `$HLAB` throughout.

## Test classes: F2P vs P2P

Harbor verifiers (following [SWE-Bench](https://www.swebench.com/) convention) typically split reference tests into two classes:

| Class | What it is | What a failure means |
|---|---|---|
| **F2P** (FAIL_TO_PASS) | Tests that fail on the unpatched repo and must pass after the agent's edit | Agent didn't implement the requested feature/fix (or implemented it incorrectly) |
| **P2P** (PASS_TO_PASS) | Tests that pass before and must still pass after | Regression — agent broke unrelated existing behavior |

A trial is "resolved" only when **all F2P pass AND all P2P still pass**. The two failure classes have very different diagnostic paths, so split failures by class before triaging:

- **P2P failure** → start with regression analysis (the default verdict): the agent touched too much, didn't bound its edits, or its change cascaded. Apply F2P-style validity questions only in narrow cases, all of them rare:
  - Hidden fixture or path dependency the test relies on, never mentioned in the instruction (a reasonable refactor breaks the path).
  - Pre-existing flaky test (timing, network, unseeded randomness) where the agent's change merely shifted execution.
  - Internal-implementation assertion (`assert _private_helper() == "old"`) rather than a public-contract assertion — agent legitimately refactored the private path.
  - Pre-broken test that was passing for unrelated reasons and the agent's edit removed the masking.
  - Instruction wording that *forces* a change the P2P test catches — agent had no choice (borderline; depends how forcing the wording is).
  
  Default to capability gap unless one of these applies.
- **F2P failure** → could be capability (agent didn't solve the problem) or verifier validity (test over-specifies, instruction undersells, fixture mismatch, etc.). Apply `references/verifier-validation.md` before declaring a verdict.

`reward.json`'s `reference_tests` block typically reports F2P and P2P pass counts separately. If your task framework doesn't surface the split, infer it: any test that exercises new behavior described in the instruction is F2P; any test that touches pre-existing behavior is P2P.

## Locating trial files

Harbor trials live under a job directory. Two layouts are common:

- Single-trial baseline: `<job_dir>/<run_dir>/<task_id>-<7char_suffix>/` — e.g. `jobs/v2b-baseline-codex-gpt55-high-rerun/baseline-codex-craft-taskgen-v2b-...-/t2v3-AUb452-regex-character-clas-V5Y4D7n/`.
- Multi-iter rescue runs add an extra `iter<N>/` level: `<job_dir>/<run_dir>/iter1/<task_id>-<suffix>/`.
- Older job dirs are timestamped (`jobs/2026-04-01__17-20-22/<task_id>-<suffix>/`); newer ones are descriptively named.

Per-trial files (under the trial dir, regardless of which layout produced it):

| File | What it contains |
|------|-----------------|
| `verifier/reward.json` | Score, reference test counts, truncated output, diff stat |
| `verifier/verify_full_output.txt` | Complete pytest/verifier output (not truncated) |
| `agent/claude-code.txt` | Raw JSONL agent log (claude-code agent) |
| `agent/opencode.jsonl` | Raw JSONL agent log (opencode agent) |
| `agent/trajectory.json` | Structured trajectory (may need rebuild) |
| `result.json` | Trial metadata: model, timing, token usage |

## Starting from a `review_md` artifact

When auditing a multi-trial cohort, the recommended first step is the pre-computed review markdown produced by `scripts/task_review.py` (in `craft-taskgen`). It aggregates pass@k, the universal-fail intersection, per-test passing-models, and full trial-dir paths into one file per task (`review_md/<task_id>.md`). Reading that markdown before invoking harbor-lab saves the trial walk that the 9-step workflow otherwise has to do by hand.

The review markdown's "Trial outcomes" table lists the full trial path per row — use those paths directly with `$HLAB` commands rather than guessing which root dir each trial lives under.

If the harbor task dir has a `diagnostics/` subdir, the files there are prior audit reports from earlier pipeline steps. Treat them as additional context; don't take their verdicts as authoritative.

## Multi-trial / multi-model analysis

The 9-step workflow below is single-trial-deep-dive. When you have multiple trials per `(model, task)`, aggregate first to focus the deep-dive — but don't let the aggregated signals substitute for per-test triangulation in step 7. The aggregates are triage; they tell you where to look, not what's broken.

- **Pass@k**: count trials where `resolved=True`. If `pass@k > 0`, the task is solvable; the deep-dive scope is "why did failing trials fail," not "is this task valid." If `pass@k == 0`, the task is either capability-frontier or has a validity issue — both possibilities stay live until per-test triangulation rules one out.
- **Universal-fail F2P tests**: the intersection of failed-test name sets across all complete-listing trials. These are the primary entry points for verifier-validation questions in step 7. The strength of "universal-fail" as evidence scales with trial count — a test failing in 5 trials is suggestive; failing in 30 trials across 8 models is much stronger.
- **Per-test passing-model breakdown**: for tests that aren't universally failing, count which models passed them on which trials. If a single model passes a test on rare trials and others never pass it, that's a sometimes-passing test — useful signal that the test is solvable (so capability-gap is a live hypothesis) and that the discrepancy across models may be a model-style mismatch. See `references/verifier-validation.md` Q3 multi-trial generalization.

Aggregates do NOT replace the per-test triangulation in step 7. A universally-failing test still needs its instruction-vs-test scope checked individually; pass@k = 0 doesn't categorize the failure mode. Don't let triage labels substitute for evidence.

## Workflow

**CRITICAL: job dir vs trial dir.** All `$HLAB` commands (`errors`, `edits`, `tool-sequence`, `metrics`, `compare`, `subagents`) take the **job directory** — the parent containing many trial subdirs. NEVER pass a trial subdirectory like `jobs/2026-04-01__17-20-22/<task>-abc1234/` to harbor-lab — it will return "No trials found." Only use the full trial path when reading individual files directly (`reward.json`, `verify_full_output.txt`, raw agent logs).

### 1. Triage: score + diff stat

```bash
python3 -c "
import json; d=json.load(open('<trial>/verifier/reward.json'))
print('reward:', d.get('reward'))
print('ref_tests:', d.get('reference_tests',{}))
print('diff:', d.get('git_info',{}).get('diff_stat',''))
"
```

If diff stat is empty or tiny, the agent may have crashed early — skip to step 4.

### 2. Failure details

```bash
$HLAB errors <job_dir>/
```

First split failures into F2P-failed vs P2P-failed (see "Test classes" above) — they need different analyses. Then group within each class by root cause:
- Same error across multiple tests → single integration bug
- All tests crash in the same helper or fixture → test setup / verifier issue (do a three-layer audit: fixtures, helpers, assertions)
- Mix of different errors → multiple independent gaps

If only P2P tests are failing, the agent solved the requested problem but broke something else — jump to step 5 (the agent's implementation) and look for over-broad edits, deleted code paths, or changes that touched files outside the instruction's scope.

### 3. Trajectory overview

```bash
$HLAB tool-sequence <job_dir>/       # Full tool sequence
$HLAB metrics <job_dir>/             # Tokens, cost, cache rate, latency
$HLAB rebuild-trajectories <job_dir>/  # If trajectory.json is missing
$HLAB subagents <job_dir>/           # Parallel subagent usage (claude-code only)
```

Key signals: low tool count / output tokens → early exit. High cache rate → caching worked. Subagent usage → exploration was farmed out.

### 4. Last turns: finish or infra issue?

```bash
$HLAB tool-sequence <job_dir>/ --tail 10 --text
```

The `--text` flag includes agent TEXT blocks (declarations, API errors). Check for:
- **API errors** (400, 429, 500) → infra issue, not capability gap
- **Agent declared done** ("All tests pass") → circular testing (its own tests pass, reference tests fail)
- **Agent stopped mid-implementation** → timeout, OOM, or token limit
- **Agent never wrote code** → instruction misunderstood or exploration loop

### 5. Agent's implementation

```bash
$HLAB edits <job_dir>/             # Which files were written/edited
$HLAB edits <job_dir>/ --verbose   # Full content of each Write/Edit
```

For key methods, read the actual code the agent wrote. If the task targets a specific upstream commit, compare the agent's edits against the reference implementation by checking out that commit in the source repo.

### 6. Agent's tests vs reference tests

The agent often writes its own tests — they always pass (circular). Reference tests are hidden until verification. Check:
- What did the agent's tests validate? (Its own design assumptions)
- What do the reference tests check? (The intended behavior)
- Where's the gap?

In `verify_full_output.txt`, the verifier typically runs both: one section for reference tests, one for the agent's own tests. Read carefully which is which.

### 7. Triangulate: error + agent code + test + instruction

For each failing test, answer four questions:
1. **What class of test is it?** F2P or P2P? (Determines which diagnostic path to take.)
2. **What does the test expect?** Read the reference test and its fixtures.
3. **What does the instruction say?** Does it specify the behavior being tested? (For F2P only — P2P tests pre-exist and don't need to be in the instruction.)
4. **What did the agent build?** Trace the code path the test exercises using the implementation from step 5.
5. **Where's the mismatch?**
   - **P2P fail** → default verdict is regression. Look at the agent's edits (step 5) and identify what it changed that broke the test. Only consider verifier-validity if the test exhibits one of the rare exceptions listed in "Test classes" above (hidden fixture dependency, flake, internal-implementation assertion, pre-broken-but-masked, or instruction-forced regression).
   - **F2P fail** → distinguish capability gap from verifier issue. Apply `references/verifier-validation.md`. Key heuristic: if both a strong model and a weak model fail the same F2P test the same way, it's almost certainly an instruction or verifier issue.

   **Note on the multi-model heuristic**: It applies cleanly to F2P only. For P2P, multiple models failing the same way is ambiguous — it could mean (a) the instruction's wording nudges everyone toward the same regression (verifier/instruction issue), or (b) the regression is just easy to introduce and everyone bounds their edits the same wrong way (capability gap, common to all tiers). To disambiguate, read each agent's edits (step 5) and check whether they made the *same* over-broad change (b, capability) or *different* changes that all happen to break the same P2P test (a, more likely instruction).

#### Three-layer audit

Always invoke this audit when verifier-validation Q1 ("does the instruction specify what the test checks?") or Q2 ("does instruction wording match test scope?") is triggered, or when a P2P failure looks like one of the rare-exception cases above. Skip only when the test is unambiguously asserting a public contract that the instruction directly describes.

Audit in three layers:
- **Fixtures** — Does the test setup match what the instruction describes? Are there hidden assumptions (paths, env vars, fixture data shape) the agent couldn't have known about?
- **Helpers** — Are helper functions in the test file doing work the agent isn't expected to replicate? A helper that silently normalizes input can mask whether the agent's output is actually wrong.
- **Assertions** — Are the assertions checking the spec, or an arbitrary implementation detail of the reference solution? `assert x == reference_x` is brittle; `assert satisfies_spec(x)` is robust.

A failure rooted in fixture or helper mismatch is a verifier bug. A failure rooted in an over-specific assertion is an instruction-vs-test scope mismatch (see `references/verifier-validation.md` Q2).

### 8. Cross-run comparison

**Metric-level**: `$HLAB compare <job_a>/ <job_b>/` for pass rates, reward deltas, statistical tests.

**Implementation-level**: Run steps 2 and 5 for each trial separately. Then compare: do different runs fail the same way? Do they make the same design choices? Consistent failures = genuine capability gap. Different failures each time = random variation.

### 9. Docker replay (optional)

Re-score an existing trajectory against a modified verifier without re-running the agent. See "Replaying the verifier" below.

## Analyzing agent strategy

Beyond pass/fail, harbor-lab output reveals *how* the agent worked. Useful when comparing tiers, diagnosing weak agents, or assessing task design.

### Strategy analysis (from `tool-sequence`)

- **Tool choice patterns**: Does the agent use dedicated tools (Read, Edit, Grep, Glob) or fall back to bash equivalents (cat, sed, grep, find)? Stronger models typically use dedicated tools; weaker models bash-for-everything. Check tool frequency in `$HLAB metrics`.
- **Exploration vs implementation ratio**: How many calls were spent reading/searching vs writing/editing? Over-exploration (50 reads, 2 edits) = couldn't locate code. Under-exploration (2 reads, 5 edits) = premature implementation.
- **Thrashing**: Does the agent repeat similar calls without progress? Patterns like bash arithmetic → fail → retry with awk → fail → retry with different syntax → ... 20 more attempts. Thrashing inflates call count without advancing toward the solution.
- **Iteration vs write-and-declare-done**: Did the agent run tests, observe failure, and adjust? Or write everything in one pass? Look for Bash (pytest) → Read/Edit → Bash (pytest) cycles. Iteration signals orchestration sophistication.
- **Verification strategy**: Targeted re-reads after edits and focused checks (`bash -n`, single pytest), or redundant re-reads of the same file 3-4 times? Stronger models tend toward targeted verification.
- **Subagent use**: Did it dispatch a subagent for exploration before implementing? See `$HLAB subagents`.
- **Parallel tool calls**: Did it parallelize independent reads/searches? Visible in `parallel_groups`.
- **Efficiency inversions**: Sometimes a weaker model is more efficient on specific subtasks (e.g., scripting over manual file-by-file reads). Note when this happens; it's signal about task design.

### Strategy divergence across models/tiers

Compare tool-sequence output for the same task across different agent runs:

```bash
$HLAB tool-sequence <strong_model_job>/
$HLAB tool-sequence <weak_model_job>/
```

Key questions:
- Do stronger models explore more selectively (fewer reads, better targeted)?
- Do weaker models get stuck in exploration loops or thrash on subproblems?
- Does the implementation phase look fundamentally different (targeted edits vs full file rewrites vs bash-for-everything)?
- Does one model iterate on test failures while another writes-and-declares-done?
- Are differences coming from the **model** or the **agent harness**? If the same model was run in different harnesses, compare to separate model behavior from harness behavior.

### Metric-level signals (from `metrics`)

- **Tool call count**: More isn't better — compare against what the task requires. Very low = early exit. Very high = thrashing.
- **Tool frequency breakdown**: High Bash count relative to dedicated tools = bash-for-everything pattern. High Read count with low Edit count = over-exploration.
- **Output tokens**: High output with low test scores = wrote a lot of non-working code. Low output with high scores = efficient.
- **Cache rate**: High = re-read context efficiently. Low = context churn.
- **Peak context**: Did the agent hit context limits? Check whether it used compaction or subagents to manage.

## Producing a root-cause summary

After working through the steps above, produce a summary per failing test:

```
=== <test_name> ===
Class: <F2P | P2P>
Error: <one-line error from step 2>
Root cause: <what the agent did wrong, traced in step 7>
Verdict: <capability gap (regression) | capability gap (didn't solve) | verifier issue | instruction issue> — with evidence
```

Then a cross-cutting summary covering:

- **F2P-failed vs P2P-failed tally** — and which dominate. If P2P failures dominate, headline is "agent solved the problem but caused regressions"; if F2P failures dominate, "agent didn't solve the problem" (modulo verifier-validity caveats).
- **Shared root causes** — failures that trace back to a single bug or misunderstanding.
- **Circular self-testing** — call this out explicitly when present: agent declared "all tests pass" while reference tests failed. This is one of the highest-signal patterns and means the agent's confidence didn't match reality. It usually indicates the agent wrote tests against its own design rather than the instruction's spec.
- **What the agent got right** vs **what it missed**.

**Don't over-react to single-test brittleness.** If you find 1–2 F2P tests with weak validity (over-specific assertions, unstated assumptions), don't reflexively drop them. At benchmark scale, single-test brittleness is absorbed by F2P_micro (per-test averaging within a task) and shows up as proportional noise, not catastrophic distortion. Drop a test only when it's *systematically* broken — universal fixture issue, dependency mismatch every agent hits, etc. For one-off brittleness, flag it in the report and move on.

## Replaying the verifier

When debugging verifier behavior or testing a fix, you can reconstruct the agent's container state and rerun the verifier without waiting for a full agent session. This saves hours of agent runtime and API cost.

### When to use

- Investigating whether a score reflects a genuine agent failure or a verifier bug
- Iterating on gold reference tests (add/remove/rewrite assertions, three-layer audit changes)
- Testing fixes to `test_runner.py` or the verify script (parser scoping, `-k` exclusions, test paths)
- Checking partial credit after adjusting what tests are included
- Adding debug output (print statements in gold tests) to understand a specific failure
- Validating that a verifier change doesn't flip a genuine failure to a false pass

The key insight: the agent's implementation is frozen in the trajectory. You only change the verifier side. This turns verifier iteration from a multi-minute agent re-run into a ~30-second Docker replay.

### Steps

1. **Extract the agent's edits** from the trial:
   ```bash
   $HLAB edits <trial_dir>/ --verbose
   ```
   Or parse Write/Edit tool calls directly from the raw log under `<trial_dir>/agent/`.

2. **Replay in a fresh container** with the current (possibly updated) verifier:
   ```bash
   TASK_DIR="path/to/harbor-task-dir"
   docker run --rm \
     -v "$TASK_DIR/tests:/tests:ro" \
     <task-docker-image> bash -c "
       # Reconstruct agent state from step 1
       cat > /repo/path/to/file.py << 'EOF'
       <content from step 1>
       EOF

       # Run verifier
       mkdir -p /logs/verifier
       python3 /tests/test_runner.py
       cat /logs/verifier/reward.json
     "
   ```

3. **Compare** — `reference_tests` counts should now reflect reality. Check `verify_full_output.txt` for full output.

### Why this works

The Docker image has the pre-change repo (from the task's Dockerfile). Applying the agent's edits reconstructs the post-agent state. Mounting `/tests/` from the host uses the current verifier code, so any fixes you've made are picked up immediately. No API calls or agent runtime needed.

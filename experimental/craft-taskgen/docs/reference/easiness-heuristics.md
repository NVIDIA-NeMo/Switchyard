# Easiness Heuristics

Two deterministic signals flag tasks as possibly "too easy." The inline
signal is now action-gated (triggers instruction regen, then escalates
to `NEEDS_FIX` on repeat); the post-run signal is still advisory.

| Signal | Where | When it runs | Action |
|---|---|---|---|
| Inline easiness check | `src/craft_taskgen/steps.py::_deterministic_easiness` | On every `reward == 1.0` task during triage. | First pass → triggers Build regen with prescriptive-instruction feedback (shared budget with reviewer regen, `MAX_TRIAGE_REGENS=1`). Second pass → `NEEDS_FIX` with `needs_human_review=True`. |
| Efficiency flag (post-run) | `scripts/analyze-easiness.py` | Offline tool, invoked manually after a bulk run. Adds cohort-relative p10 rules (fast_wall_time, low_turns) that the inline check can't compute without seeing the whole cohort. | Writes `auto_reject` / `soft_flag` annotations to a CSV; no pipeline action. |

The inline check counts Grep + Read tool calls from `harbor-lab tool-sequence`
(the full output, not the prompt-truncated tail). `pytest_runs` is recorded
on the task for observability but does not by itself trigger the flag.

## Inline easiness → Build regen flow

When a task trial finishes with `reward == 1.0`:

1. `_deterministic_easiness` counts Grep+Read calls on the full trajectory.
2. If `grep_read <= _EASINESS_GREP_READ_MAX` (5 today), sets `task.easiness_flag = True`.
3. If `easiness_flag` AND `task.triage_regen_count < MAX_TRIAGE_REGENS`:
   - Packs a `fixable_issue` with `classification="easiness_too_prescriptive"` and the grep_read count.
   - Calls `_run_triage_build_regen`, which dispatches to `easiness_triage_feedback_block` in `prompts.py`.
   - The feedback asks Build to rewrite more abstractly: strip named files/classes/data-structures/procedural language while preserving outcome + API contracts.
   - Task goes to `Stage.BUILT` with `pending_fix_type="instruction"`; pipeline re-runs alignment → smoke → triage on the new instruction.
4. If `easiness_flag` AND budget exhausted (`triage_regen_count >= MAX_TRIAGE_REGENS`):
   - `Stage.NEEDS_FIX`, `needs_human_review=True`, reason cites persistent easiness.
   - Regen didn't produce a more exploration-forcing instruction → task is structurally too easy.
5. If no easiness flag: normal `reward=1` accept.

Budget is shared with the reviewer-regen path, so a task that hit a reviewer-major regen can't also burn another regen on easiness in a later pass (and vice versa).

## Running the post-run analyzer

```
scripts/analyze-easiness.py --mode run_dir --run-dir <harbor-tasks/.../runs/<ts>/>
```

Shells out to `harbor-lab errors / edits / tool-sequence / metrics` per task,
extracts counts and wall time, writes a CSV. Post-run rules:

| Rule | Threshold | Action in CSV |
|---|---|---|
| `no_exploration` | Grep+Read < 10 | `auto_reject` |
| `zero_iteration` | 0 pytest runs | `auto_reject` |
| `fast_wall_time AND low_turns` | both ≤ cohort p10 | `auto_reject` |
| `fast_wall_time` or `low_turns` alone | one ≤ p10 | `soft_flag` |

`auto_reject` in the CSV means "flagged in the post-run output," not a
pipeline action. A human decides whether to remove the task.

## Re-tuning thresholds

- Re-run `scripts/analyze-easiness.py` on the new cohort.
- Hand-review the flagged tasks.
- Tighten `_EASINESS_GREP_READ_MAX` in `steps.py` if the inline check
  over-flags, or relax if obvious easy tasks slip through.
- The post-run CSV columns are the canonical place to spot-check:
  `num_greps`, `num_reads`, `num_pytest_runs`, `wall_time_s`,
  `exploration_before_first_edit`.

## Known limitations

- **Cause-agnostic.** The inline check flags low-exploration trajectories
  regardless of cause: agent ran from a leaky instruction vs. the task was
  genuinely small and well-scoped. Primary defense against instruction
  leakage lives in the build-time alignment judge; easiness-flag is a
  post-execution safety net.
- **No per-repo normalization.** Thresholds are global. A Harbor-heavy
  repo with long startup overhead shifts cohort p10 down and can mask
  fast tasks elsewhere. Switch to per-repo percentiles once we have ≥3
  accepted tasks per repo.
- **No per-task gold labels.** We don't have human "this was actually
  too easy" labels. The only falsifier is whether the auto-rejected
  tasks look defensible on manual inspection.
- **`pytest_runs` is narrow.** Matches only Bash calls with `"pytest"` in
  args; misses `python -m unittest`, `./test.sh`, `tox -e test`, etc.
  Currently observability-only, so this is not a pipeline bug — but any
  future rule that relies on `pytest_runs` needs a broader test-runner
  detector.

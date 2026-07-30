# N-parallel build+alignment calibration (Apr 25 2026)

Calibration data behind the N-parallel build+alignment change. Tracks
four configs across two cohorts: the standard `rerun-accepts-v2` success
cohort and an eval-rejected cohort drawn from a recent batch.

Configs:

| Label | N (candidates) | α (assessor retries) | r (build rebuilds per candidate) |
|---|---|---|---|
| Main today | 1 | 3 | 1 |
| Phase 1 | 2 | 3 | 1 |
| N=3 default | 3 | 3 | 1 |
| **Phase 2 (shipped)** | **3** | **1** | **2** |

## Cohort 1: `rerun-accepts-v2` success cohort (80 rows)

PRs that the production pipeline has previously eval-accepted at least
once — the right cohort for measuring throughput on the success path.
All 80 rows reached build+align on every config.

| Config | Pass / 80 | Pass% | Align tokens | vs main | Failures |
|---|---|---|---|---|---|
| Main today | 73 | 91.2% | 2.4M | 1.0× | 7 |
| Phase 1 | 77 | 96.2% | 2.4M | 1.0× | 3 |
| N=3 default | 78 | 97.5% | 3.1M | 1.27× | 2 |
| **Phase 2** | **80** | **100%** | **2.0M** | **0.85×** | **0** |

Phase 2 wins on both pass rate and cost. Three orthogonal mechanisms
compose:

- **N=3 parallel candidates** — addresses unlucky-build variance by
  re-rolling the build dice across independent instructions.
- **α=1 (no retention bias)** — stricter per-instruction acceptance;
  removes the prior "false-alarm hack" that allowed one `ok` to override
  two `no`s on the same draft.
- **r=2 rebuilds per candidate** — each candidate gets two chances to
  fix a genuinely-flagged instruction with cumulative feedback.

## Cohort 2: eval-rejected cohort (79 rows from `2026-04-24-062212` state)

PRs the production pipeline rejected at the eval step (`eval_verdict ==
"reject"`). Calibrated with `--skip-eval` so we measure build+align
behavior directly on this material. 5 rows skipped across all configs
for `sha_not_reachable` (transient git context fetch failures), so the
effective denominator is 74.

| Config | Pass / 74 | Pass% |
|---|---|---|
| Main today | 70 | 94.6% |
| Phase 1 | 71 | 95.9% |
| N=3 default | 74 | 100% |
| **Phase 2** | **74** | **100%** |

Caveats:

- No ground truth — these are PRs the eval step rejected, but we can't
  tell which rejections were correct vs false-alarms. Recovery here does
  not equal correctness.
- The narrow spread between configs (4-5 PRs) reflects that this cohort
  was filtered by eval, not by alignment, so it doesn't isolate the
  failure mode N-parallel was designed to fix.
- Useful as a quality-risk bound: Phase 2 isn't recovering substantially
  more eval-rejected material than Phase 1 / N=3-default, which argues
  against "Phase 2 is just a more permissive assessor". The pass-rate
  improvement on cohort 1 is real signal, not threshold drift.

## Real-world validation: 42-task overlap on craftbench04

Side-by-side production run with `BUILD_N_CANDIDATES=3` on a hard-cases
batch (main baseline `2026-04-24-073623` vs new branch
`2026-04-25-144901`):

| Outcome (eval-accepted in both) | Main | New branch |
|---|---:|---:|
| align_ok | 2 | 18 |
| align_leaked | 5 | 0 |
| align_narrow_tests | 15 | 0 |
| align_orch_reject (failed at N=3) | 0 | 3 |

16 of 19 prior alignment-failures rescued (84%). 3 tasks remained
rejected after all 9 build attempts (3 candidates × 3 builds each).

Rescue-mechanism breakdown of the 16 rescues:

- 6 cases: all 3 candidates passed (borderline task; parallel re-rolls
  helped).
- 6 cases: 2 of 3 passed, 1 failed (N≥2 strictly required).
- 4 cases: only 1 of 3 passed (N=3 was the load-bearing safety margin).

10 of 16 rescues required N=3 fanout to find a passer. N=2 would have
recovered 12 of 19; N=3 recovers 16 of 19.

## ToolUniverse exemplar — pattern-stuck rebuild rescued by fresh draft

`ToolUniverse-3ef2ec9e` was alignment-rejected on main for
`narrow_tests` after 3 retention retries + 1 rebuild — Build kept
missing the EUHealth domain layer that `test_euhealth_tool.py`
exercised. On the new branch:

| Cand | Outcome | Rebuilds used | Pattern |
|---|---|---|---|
| cand0 (winner) | pass on first draft | 0 | Restructured behaviorally; explicit EUHealth layer coverage |
| cand1 (loser) | reject narrow_tests | 2 (exhausted) | Followed main's pattern; kept missing fixture details |
| cand2 (loser) | reject narrow_tests | 2 (exhausted) | Followed main's pattern; rejected for path conventions |

Both losers each tried 3 builds (initial + 2 rebuilds with cumulative
feedback) and stayed stuck in the same pattern main was rejected for.
cand0 drew a different framing on its first roll and cleared alignment
immediately, explicitly adding "Build an EUHealth domain layer
(`euhealth/`) with `tools_runtime` exposing `TOPICS`, dynamic per-topic
search functions…" — exactly the coverage the reference tests required.

This is the strongest validation of the design: when Build is stuck in a
structurally similar pattern, feedback-driven rebuilds within the same
candidate cannot escape it. A fresh independent draft is what finds the
solution.

## Throughput implications for downstream stages

Higher alignment pass rate means more tasks reach oracle / docker
classify / smoke / triage per run. For the rerun-accepts-v2 cohort the
downstream load grows from 73 to 80 tasks (+9.6%). On the production
hard-cases cohort the effect is larger because the cohort was filtered
by alignment-failure on main: 18 alignment-passes on the new branch vs 2
on main means downstream stages see ~9× the volume on that subset.

Build+align itself runs ~3× the LLM calls per task (N=3 candidates
in parallel; alignment is α=1 so net assessor calls per candidate are
lower). Wall clock per task is roughly unchanged because the candidate
loops run concurrently within `LLM_CONCURRENCY`. End-to-end batch wall
clock is bounded by downstream stages that aren't fanned out — Harbor
smoke trials in particular dominate the tail.

Operational implication: when running the pipeline at scale, expect
higher throughput at the alignment gate and plan downstream capacity
accordingly. The `build_n_candidates`, `alignment_max_retries`, and
`max_build_regens_per_candidate` knobs in `profiles/craft-tools-v4.toml`
let operators dial back if downstream becomes the bottleneck.

## Reproduction

```bash
# Cohort 1 (rerun-accepts-v2)
uv run python scripts/calibrate-alignment.py \
    --input candidates/rerun-accepts-v2/calibration_input.csv \
    --output /tmp/calib_phase2.csv \
    --mode full --n 3 --sample 85 --concurrency 4 --seed 42 --skip-eval \
    --max-alignment-retries 1 --max-rebuilds 2

# Cohort 2 (eval-rejected from a state.json)
uv run python scripts/state_to_rejected_csv.py \
    harbor-tasks/craft-tools-v4/runs/2026-04-24-062212/state.json \
    --output /tmp/eval_rejected.csv --filter eval

uv run python scripts/calibrate-alignment.py \
    --input /tmp/eval_rejected.csv \
    --output /tmp/calib_eval_rejected_phase2.csv \
    --mode full --n 3 --concurrency 4 --seed 42 --skip-eval \
    --max-alignment-retries 1 --max-rebuilds 2
```

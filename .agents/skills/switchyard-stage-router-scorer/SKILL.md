# Skill: switchyard-stage-router-scorer

**description**: Score benchmark run trajectories through the stage-router Rust scorer and picker, then visualise score distributions. Use when you want to replay trajectories through the picker, analyse routing splits, or compare score distributions across configs.

## Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `benchmark/score_run.py` | Score a live run dir via real picker | run dir path | per-turn JSONL + per-task CSV |

## Quick Reference

```bash
# Score a run
uv run python benchmark/score_run.py --run benchmark/tb_runs/<run-name>
# → /tmp/<run-name>-scores.jsonl   (per turn)
# → /tmp/<run-name>-per-task.csv   (per task)

# Custom threshold or window
uv run python benchmark/score_run.py \
    --run benchmark/tb_runs/<run-name> \
    --threshold 0.15 --window 3
```

## Scoring pipeline (what score_run.py does per turn)

```
trajectory step (tool_use + tool_result)
  → append to cumulative Anthropic messages list
  → ChatRequest.anthropic({"model": ..., "messages": messages})   # Rust binding
  → dc.process(ctx, request)          # DimensionCollector — one per task, accumulates state
  → get_tool_result_signal(ctx)       # read signal from ctx
  → from_signal(signal) + scorer_score(dims)   # raw score + confidence (for analysis)
  → pick_capable_first(ctx, threshold)         # actual cf decision (same ctx, no re-process)
  → pick_efficient_first(ctx, threshold)       # actual ef decision (same ctx, no re-process)
```

**Key:** `dc.process()` is called **once per turn** on a single ctx. Both pickers read the signal
already stored in that ctx — no duplicate processing.

**What the picker does beyond raw score:**
- `_apply_overrides`: `severity >= 1.0` → force CAPABLE; `tests_passed AND depth >= 10 AND writes <= 1` → force EFFICIENT
- `confidence < threshold` → fall_open to default tier (CAPABLE for cf, EFFICIENT for ef)
- Only when `confidence >= threshold`: route by score direction

## Per-turn JSONL schema

```json
{
  "task_name": "terminal-bench/...", "trial_name": "...", "run_id": "...",
  "reward": 1.0, "turn_depth": 5,
  "score": -0.83, "confidence": 0.95,
  "tool_name": "Bash", "is_error": false,
  "write_count": 2, "edit_count": 1, "read_count": 3,
  "no_error_streak": 4, "pure_bash_streak": 2, "tests_passed": false,
  "pick_cf": 1,
  "pick_ef": 0
}
```

`pick_cf` / `pick_ef`: `1` = CAPABLE (Opus), `0` = EFFICIENT (Nemotron)

## Per-task CSV columns

`run_id, task_name, reward, n_turns, mean_score, mean_confidence,
pct_strong_clear, pct_strong_uncertain, pct_weak_uncertain, pct_weak_clear,
opus_pct_cf, nemotron_pct_cf, opus_pct_ef, nemotron_pct_ef`

`opus_pct_cf` / `opus_pct_ef` are derived from actual `pick_cf` / `pick_ef` decisions, not score bands.

## Band definitions (for histogram colouring, threshold T)

| Band | Score range | cf default | ef default |
|------|-------------|------------|------------|
| strong_clear | ≥ T | Opus | Opus |
| strong_uncertain | (0, T) | Opus (fall_open) | Nemotron (fall_open) |
| weak_uncertain | (-T, 0) | Nemotron (fall_open) | Nemotron (fall_open) |
| weak_clear | ≤ -T | Nemotron | Nemotron |

Overrides can change any band. Use `pick_cf`/`pick_ef` for the true decision.

## Key findings (v0.2.0 baseline tarballs, T=0.20)

- Distribution is **highly bimodal** — most turns land in strong_clear or weak_clear
- `cf, t=0.20`: ~34% Opus on partial run; ~42% on full baseline
- `ef, t=0.20`: ~20% Opus on partial run; ~41% on full baseline
- Live routing split differs from shadow score — Nemotron trajectories are shorter and reshape the distribution

## Anti-patterns

- Don't call `dc.process()` multiple times per turn for different pickers — both pickers read from the same ctx. One `dc.process()` call per turn is correct.
- Don't infer routing split from score bands alone — overrides and fall_open change the actual decision. Always use `pick_cf` / `pick_ef`.
- Don't compare cost directly across configs: Nemotron has ~39% cache hit rate vs ~92% for Opus.

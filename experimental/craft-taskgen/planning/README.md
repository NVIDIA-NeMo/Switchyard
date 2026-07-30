# Iterative Planning Evaluation Harness

Single-command runner for evaluating planner + implementer models against a
directory of planning harbor tasks.

## Prereqs

1. A harbor venv (activate + source `.env`).
2. A directory of planning harbor task dirs. Generate one with
   `craft-taskgen-convert --adapter planning`:

   ```bash
   craft-taskgen-convert --adapter planning \
       --candidates-dir /path/to/planning/candidates \
       --output-dir harbor-tasks/craft-planning/
   ```

   Each candidate JSON must include `spec`, `parent_sha`, `merge_sha`,
   `fail_to_pass`, `pass_to_pass`, `docker`, `src_files`, `test_files`,
   and `test_command` fields (see
   `src/craft_taskgen/adapters/planning/converter.py` for the schema).

## Setup

```bash
source /path/to/harbor/.venv/bin/activate && source /path/to/harbor/.env
export PLANNER_AGENT_KWARGS=api_base=$OPENAI_BASE_URL
export IMPL_AGENT_KWARGS=api_base=$OPENAI_BASE_URL
export PLANNER_MODEL=aws/anthropic/bedrock-claude-opus-4-6
```

## Run a single task

```bash
TASKS=hugapi__hug-651 N_PARALLEL=1 ./planning/run_plan_impl.sh
```

## Run a directory of tasks

Default dataset is `harbor-tasks/craft-planning`:

```bash
./planning/run_plan_impl.sh
```

Point at any other harbor task directory (absolute or relative to repo root):

```bash
SOURCE_DATASET=harbor-tasks/my-planning-dataset ./planning/run_plan_impl.sh
SOURCE_DATASET=/absolute/path/to/harbor-tasks-dir ./planning/run_plan_impl.sh
```

Narrow within the directory with a glob filter:

```bash
SOURCE_DATASET=harbor-tasks/my-dataset TASKS='scrapy__*' ./planning/run_plan_impl.sh
```

Implementer defaults to Haiku 4.5 — the fixed implementer in the methodology.
Override either side for ablations:

```bash
# Non-Anthropic implementer (opencode + Nemotron)
IMPL_AGENT=opencode IMPL_MODEL=nvdev/nvidia/nemotron-3-super \
PLANNER_MODEL=aws/anthropic/bedrock-claude-opus-4-6 \
./planning/run_plan_impl.sh
```

## See per-task rewards

```bash
cat jobs/plan-impl-<ts>/results/report.md
```

Markdown table: task · binary reward · F2P passed/total · P2P passed/total ·
plan recall · regressions. Structured version in `results/results.json`;
headline float in `results/reward.txt` (pass rate across the set).

Per-task reward is binary — `1.0` iff every F2P passes AND every P2P passes.
Fractional F2P/P2P counts preserved in `results.json` for analysis.

## How it works

Two stock `harbor run` invocations with plan injection between them:

1. **Planner phase**: `wrap_pipeline.py` generates a planner-mode dataset.
   `harbor run` executes the planner (`PLANNER_AGENT` / `PLANNER_MODEL`)
   which writes `plan.md` + `plan.json` per task.
2. **Implementer phase**: `wrap_pipeline.py` regenerates an implementer-mode
   dataset with plans injected into each task's instruction. `harbor run`
   executes the implementer (`IMPL_AGENT` / `IMPL_MODEL`) which modifies
   the code.
3. **Scoring**: `join_scores.py` reads both job dirs and emits
   `reward.txt`, `results.json`, `report.md`.

No custom harbor agent, no harbor fork. Any (agent, model) combination works
on either phase because harbor natively takes `-a` and `-m` per run.

## Naming note

The script is `run_plan_impl.sh`, not `run_e2e.sh`. In CRAFT paper terminology
"e2e" refers to the single-phase no-plan baseline — the opposite of what this
runner does.

## Full env var reference

See the header comment in `planning/run_plan_impl.sh`.

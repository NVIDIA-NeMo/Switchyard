# craft-taskgen docs

Three buckets:

- **[runbooks/](runbooks/)** — operational walkthroughs. Read these to *do* something.
- **[reference/](reference/)** — definitions, schemas, calibration data. Read these to *understand* what something means.
- **[analyses/](analyses/)** — frozen-in-time research writeups. Read these for *evidence*, not procedures.

## Runbooks

| File | When to read |
|---|---|
| [`runbooks/running-a-batch.md`](runbooks/running-a-batch.md) | Launching a deterministic repo batch through the Tools pipeline |
| [`runbooks/baseline-reproducibility.md`](runbooks/baseline-reproducibility.md) | Running `scripts/run-baselines.sh`; reasoning-effort matrix; version pins |
| [`runbooks/task-review-workflow.md`](runbooks/task-review-workflow.md) | Auditing accepted tasks across multi-trial baselines |

## Reference

| File | What it documents |
|---|---|
| [`reference/task-format.md`](reference/task-format.md) | CRAFT task directory format spec (`task.toml`, `solve.sh`, tests, etc.) |
| [`reference/pipeline-state-machine.md`](reference/pipeline-state-machine.md) | Stage transitions and `state.json` schema |
| [`reference/easiness-heuristics.md`](reference/easiness-heuristics.md) | Objective rules that flag accepted tasks as "too easy" |
| [`reference/n-parallel-calibration-apr25.md`](reference/n-parallel-calibration-apr25.md) | Calibration that drove `build_n_candidates=3` |
| [`reference/adapter-reproducibility.md`](reference/adapter-reproducibility.md) | Adapter-author migration guide for version pinning |
| [`reference/planning-scorer-phase.{mmd,png}`](reference/) | Mermaid + render of the planning-scorer phase |

## Analyses

Date-prefixed (newest first). See [`analyses/README.md`](analyses/README.md) for context on each.

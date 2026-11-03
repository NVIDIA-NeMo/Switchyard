# Tuning Foundations

This guide helps you find a **golden configuration for your objective and
workload**: define the outcome you want, compare candidate settings on
calibration tasks, and validate the selected configuration on held-out tasks
before adopting it as your baseline.

See [Compare and Tune](examples.md) for configuration and model-pair comparisons
with concrete data.

## 1. Define the objective

Define what the user wants to improve, on which workload, and what makes a result
acceptable. Keep this objective fixed throughout the comparison.

- **Reduce cost while preserving accuracy.** Minimize total cost per task while
  meeting a defined accuracy requirement. Include decision calls, retries, and
  cache charges in the cost.
- **Maximize task success.** Maximize the fraction of tasks completed correctly,
  subject to any cost or latency limits.
- **Reach a desired routing split.** For example, target 70% of calls on the
  efficient tier and 30% on the capable tier. Specify the allowed deviation and
  quality requirement.
- **Reach a capable-tier cache-usage target.** For example, require at least 80%
  of capable-tier input tokens to be served from cache. Measure cache-read input
  tokens divided by all input tokens on that tier, with a defined quality floor.

Once the objective is measurable, the next step is to choose configurations that
might achieve it. This separates the outcome we want from the settings we will
test.

## 2. Design configuration candidates

A candidate is a complete configuration to test under the objective. Start with
a baseline and a small set of changes you expect to help:

1. **Record baselines.** Measure efficient-only, capable-only, and the current
   routed configuration on the same tasks. Save their settings and outcomes.
2. **Choose what varies.** For example, change model targets, thresholds, or hold
   durations. Keep other settings fixed and state why each change might help.
3. **Test interactions.** A threshold and hold duration can affect the same
   routing decisions, so include combinations when needed.
4. **Keep runs comparable.** Use the same tasks, evaluation procedure, and initial
   cache setup. Pin model and provider versions, and save each configuration.

Run the same tasks for every candidate. The following examples show how to use
the results to choose between them.

## 3. Choose an example

Start with the comparison that matches what you want to change:

- **[Tune configuration options](examples.md#tune-configuration-options-for-cost):**
  keep the algorithm and model pair fixed, vary routing settings, and find the
  lowest-cost option that meets the task-success target. A Pareto chart shows
  the cost and success tradeoffs.
- **[Select a model pair](examples.md#model-pair-selection-with-fixed-algorithm-settings):**
  keep algorithm settings fixed, swap the efficient/capable models, and compare
  execution cost plus a penalty for failed tasks.
- **[Tune the unified router with GEPA](examples.md#upcoming-gepa-tuning-for-the-unified-router):**
  planned for a future release. Use benchmark feedback to revise routing
  instructions and select a prompt against a fixed objective.

## 4. Check the final choice

Split **100% of the evaluation benchmark before tuning**. A practical starting
allocation is:

| Split | Share | Use |
|---|---|---|
| Calibration | 20% | Collect baselines, try settings, and select the candidate. |
| Held-out validation | 80% | Evaluate the frozen candidate against the baselines. Do not use these results to tune it. |

For 500 tasks, this means 100 calibration tasks and 400 validation tasks. Adjust
the percentages before running if the calibration set is too small to cover the
workload. Here, validation is the final held-out check.

- Keep related examples in the same split. If predicting future data, respect
  time when splitting. Use representative workloads and repeated trials when
  execution is stochastic.
- Record the objective, scoring rule, configurations, and evaluation conditions.
  Changing the models or settings requires new measurements.
- If validation results guide another change, those tasks have become
  calibration data. Use fresh held-out tasks for the next independent check.

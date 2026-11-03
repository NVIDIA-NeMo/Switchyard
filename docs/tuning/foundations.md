# Tuning Foundations

Configuration settings describe how the system behaves. The user's objective
describes what that behavior should achieve. This guide bridges the gap between
the two: define a measurable objective, design candidate configurations, and use
a loss function to compare them against that fixed objective.

A loss is the score used to compare results; lower is better. For the steps to
run a routing experiment, start with [Compare and Tune](index.md).

Use a representative subset of benchmark tasks for development: training tasks
guide changes, while separate validation tasks select configurations and
thresholds. Make these splits before tuning, keep related tasks together, and
reserve an untouched test set for the final evaluation. See
[Check the final choice](#4-check-the-final-choice) for split and evaluation guidance.

For a quick walkthrough with concrete data, jump to
[Section 5: Examples of tuning](examples.md).

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

1. **Record the baseline.** Save its configuration and measured outcomes.
2. **Choose what varies.** For example, change model targets, thresholds, or hold
   durations. Keep other settings fixed and state why each change might help.
3. **Test interactions.** A threshold and hold duration can affect the same
   routing decisions, so include combinations when needed.
4. **Keep runs comparable.** Use the same tasks, evaluation procedure, and initial
   cache setup. Pin model and provider versions, and save each configuration.

Run the same tasks for every candidate. The following examples show how to use
the results to choose between them.

## 3. Choose an example

Each example connects a specific change to a measurable objective:

- **[Tune configuration options](examples.md#5a-tune-configuration-options-for-cost):**
  keep the algorithm and model pair fixed, vary routing settings, and find the
  lowest-cost option that meets the task-success target. A Pareto chart shows
  the cost and success tradeoffs.
- **[Select a model pair](examples.md#5b-model-pair-selection-with-fixed-algorithm-settings):**
  keep algorithm settings fixed, swap the efficient/capable models, and compare
  execution cost plus a penalty for failed tasks.
- **[Tune the unified router with GEPA](examples.md#5c-upcoming-gepa-tuning-for-the-unified-router):**
  planned for a future release. Use benchmark feedback to revise routing
  instructions and select a prompt against a fixed objective.

## 4. Check the final choice

- Use training examples to guide changes, validation examples to compare
  candidates and tune thresholds, and a separate test set for the frozen choice.
- Keep related examples in the same split. If predicting future data, respect
  time when splitting. Use representative workloads and repeated trials when
  execution is stochastic.
- Record the objective, scoring rule, configurations, and evaluation conditions.
  Changing the models or settings requires new measurements.
- If a test result guides another rewrite, it has become development evidence.
  Use fresh data for an independent check.

Continue to [Section 5: Examples of tuning](examples.md) for concrete comparisons.

# Tuning Foundations

Configuration settings describe how the system behaves. The user's objective
describes what that behavior should achieve. This guide bridges the gap between
the two: define a measurable objective, design candidate configurations, and use
a loss function to compare them against that fixed objective.

The objective determines what counts as better. The configurations define the
alternatives we can test. A loss is the score used to compare their results;
lower is better. For the steps to run a routing experiment, start with
[Compare and Tune](index.md).

Use a representative subset of benchmark tasks for development: training tasks
guide changes, while separate validation tasks select configurations and
thresholds. Make these splits before tuning, keep related tasks together, and
reserve an untouched test set for the final evaluation. See
[Check the final choice](#4-check-the-final-choice) for split and evaluation guidance.

For a quick walkthrough with concrete data, jump to
[Section 5: Examples of tuning](examples.md).

## 1. Define the objective

Start with the outcome the user wants. Define what to measure, on which workload,
and what makes a result acceptable. Keep this definition fixed while comparing
candidates, whether the experiment contains two configurations or many.

- **Reduce cost while preserving accuracy.** Minimize total cost per task while
  meeting a defined accuracy requirement. Include decision calls, retries, and
  cache charges in the cost.
- **Maximize task success.** Maximize the fraction of tasks completed correctly,
  subject to any cost or latency limits.
- **Reach a desired routing split.** Target a defined allocation across the
  configured targets. This could be 70% of calls on the efficient tier and 30% on
  the capable tier, or a distribution across three or more targets. Specify the
  counting unit, allowed deviation, and quality requirement.
- **Reach a capable-tier cache-usage target.** Target a specified percentage of
  capable-tier input tokens served from cache, such as at least 80%. Measure
  cache-read input tokens divided by all input tokens on capable-tier calls over
  the evaluation workload. Specify the target percentage and any quality
  requirement before comparing candidates.

Once the objective is measurable, the next step is to choose configurations that
might achieve it. This separates the outcome we want from the settings we will
test.

## 2. Design configuration candidates

A candidate is a complete configuration to evaluate under the fixed objective.
The candidate set can contain any number of alternatives. Give each a distinct
ID, such as A, B, and C, and compare all of them using the same objective and
evaluation procedure.

The objective describes what success means; candidate design determines which
possible solutions the experiment will consider. A loss can rank the candidates
we test, but it cannot find an improvement that none of them can produce.

Build the candidate set in these steps:

1. **Choose a baseline.** Record the current configuration and its measured
   outcomes. Include it in the comparison so improvements have a reference point.
2. **State a reason for each change.** Connect the proposed behavior to the
   objective. For example, staying on one model longer might preserve cached
   context and reduce cost. Test both the cost effect and the quality effect.
3. **Choose what can vary.** This could be model targets, routing policy,
   thresholds, hold durations, or decision instructions. Specify the allowed
   values and hold other settings fixed. Start with a small set that can answer
   the question behind the experiment.
4. **Include combinations when settings interact.** A threshold and a hold
   duration can affect the same routing decisions. After a small initial
   comparison, test selected combinations rather than assuming their individual
   effects will add up.
5. **Make candidates reproducible and comparable.** Give each one an ID and save
   its full configuration. Use the same workload and evaluation procedure, pin
   model and provider versions where possible, and apply the same cache setup
   procedure. Let each candidate's behavior determine subsequent cache usage.

With the objective and candidates fixed, run the same tasks for each candidate.
Use the results to compare cost and task success. The examples below show how to
make that comparison and choose a configuration.

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

Continue to [Section 5: Examples of tuning](examples.md) for concrete data and
reproducible calculations.

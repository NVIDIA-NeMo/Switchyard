# Tuning Foundations

Configuration settings describe how the system behaves. The user's objective
describes what that behavior should achieve. This guide bridges the gap between
the two: define a measurable objective, design candidate configurations, and use
a loss function to compare them against that fixed objective.

The objective determines what counts as better. The configurations define the
alternatives we can test. The loss connects the two by assigning a score to each
candidate's results. For the steps to run a routing experiment, start with
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

For example, suppose the objective is lower cost while maintaining the specified
task-success floor. One small stage-router experiment could be:

- **Baseline:** `confidence_threshold = 0.5` and `capable_hold_turns = 2`.
- **Initial candidates:** thresholds `0.3`, `0.5`, and `0.7`, all with a capable
  hold of `2` turns. This gives three candidates, including the baseline.
- **Interaction check:** add a hold duration of `0` for each threshold. The full
  set now has six candidates, not six plus the baseline.
- **Fixed conditions:** the model pair, picker, classifier configuration, other
  routing settings, and evaluation workload. Compare task success and total cost
  for every candidate, then rank those that meet the success floor.

These values illustrate candidate design, not recommended settings. The effect
of a threshold depends on the picker, classifier, and other routing rules. Follow
the [stage-router threshold guide](../routing_algorithms/stage_router_routing.md#tuning-confidence_threshold)
when choosing values. See
[model selection](../routing_algorithms/model_selection.md#counter-examples) for
cases where switching lost cache savings.

Keep the objective fixed throughout the comparison. A candidate that improves
cost but misses the quality requirement is ineligible under that objective.
Changing the requirement creates a different comparison.

With the objective and candidates fixed, we can measure their outcomes and
assign a loss to each. If the task is to learn which configuration to choose,
those outcomes also provide reference choices. The next section explains how to
represent these choices and score the probabilities assigned to them.

## 3. Outcome-based labels and cross-entropy

Use the fixed objective to identify the preferred configuration for each
evaluated request. This could mean a correct answer at lower cost or a defined
tradeoff between execution cost and failure. The reference label **y** is the ID
of that configuration. With candidates A, B, and C, the label can be A, B, or C;
it is not limited to 0 or 1. Define a consistent rule for ties.

Some objectives apply to a whole workload, such as an accuracy floor or routing
allocation. Compare complete runs against those constraints. Turning them into
per-request reference choices requires an explicit labeling or allocation rule;
the constraint alone does not supply a label.

The objective determines the labeling rule; measured outcomes determine each
label. Labels belong to the configurations and evaluation conditions that
produced them. If the objective or candidate set changes, the labels may need to
change too. The same request can favor one configuration for accuracy and another
for cost.

A model that selects one configuration predicts a probability for every available
choice. These probabilities sum to 1. **p(y)** is the probability assigned to the
reference choice. It represents the chance that this choice is preferred under
the labeling rule, rather than simply the chance that it succeeds. A configuration
can succeed and still cost more than an equally successful alternative.

For a choice question, use the probability distribution over the declared
configuration IDs. Keep answer probabilities separate from provider confidence,
and keep missing probabilities unknown rather than turning a hard choice into
certainty. See [Decision Answers](../reference/protocols/decision_answers.md#answer-values).
The labeling process can use completed outcomes; the model must use only
information available when it chooses.

### 3.1. Cross-entropy for multiple choices

When each request has one reference choice, categorical cross-entropy is:

**Cross-entropy(y, p) = −log p(y)**

In words, take the negative logarithm of the probability assigned to the
reference configuration. Use natural logarithms and average the per-request
losses to compare candidates.

For example, suppose the available configurations are A, B, and C, and the
objective makes **C** the reference choice:

- Predicted probabilities **A: 0.20, B: 0.50, C: 0.30** give a loss of
  **−log(0.30) = 1.204**.
- Predicted probabilities **A: 0.10, B: 0.20, C: 0.70** give a loss of
  **−log(0.70) = 0.357**.

The second prediction has lower loss because it assigns more probability to the
reference configuration. The same calculation applies to any number of mutually
exclusive choices. It scores probability predictions; configurations evaluated
directly by cost, task success, or another operating outcome use the loss defined
for that outcome.

### 3.2. Binary cross-entropy as the two-choice case

With only configurations A and B, encode the reference choice as:

- **y = 1:** B is preferred under the objective's labeling rule.
- **y = 0:** A is preferred. In the worked examples below, ties go to A.

Let **p** be the predicted probability of B. The probability of A is **1 − p**.
Categorical cross-entropy then becomes binary cross-entropy (BCE):

**BCE(y, p) = −y log(p) − (1 − y) log(1 − p)**

A Boolean decision question can express this as "Is B preferable to A under this
objective?" Its probability of true supplies **p**. For a two-option choice
question, use the probability assigned to B.

![BCE decreases as the predicted probability of B increases when B is preferred, and increases when A is preferred.](../assets/tuning/binary-cross-entropy.svg)

*Figure 1. The two-choice case of cross-entropy. The reference label determines
which curve applies.*

At **p = 0.99**, loss is **0.010** when B is preferred and **4.605** when A is
preferred. BCE rewards justified confidence and penalizes confident mistakes.
Two probabilities can produce the same decision at a chosen threshold while
receiving different losses.

**Probability loss is not operating cost.** A reference label records which
choice wins, but not the size of the advantage. Check actual cost, success rate,
or the other application objective alongside cross-entropy. When the size of a
mistake dominates the objective, consider predicting each configuration's
expected operating loss directly.

### 3.3. Example configuration comparison

This comparison uses two execution configurations at one pipeline stage to
illustrate the binary case. We measure both under a cost-and-failure objective,
then reuse their reference labels in the
[prompt-tuning example](examples.md#5a-prompt-tuning-with-gepa).

- **Configuration A (Fast):** a fast executor with a 512-token generation budget;
  usual execution cost of **$0.010** per request.
- **Configuration B (Thorough):** a more capable executor with a 2,048-token
  budget and reusable context; usual cost of **$0.040**, which a verified cache
  hit can reduce.

For this example, define:

**Operating loss = execution cost + $0.10 for a failed task.**

Prefer the configuration with lower operating loss. The failure penalty is a
chosen tradeoff, not a property of BCE. All execution outcomes and prices below
are invented to show the calculation; they are not Switchyard benchmark results.

![Six paired outcomes produce reference labels 0, 1, 0, 1, 0, 1. A wins direct lookup, formatting, and stale B context; B wins evidence combination, conflict resolution, and a current cache hit.](../assets/tuning/outcomes-and-labels.svg)

*Figure 2. Operating loss determines the reference choice for each request.
Hatched segments show the $0.10 failure penalty. Pass/fail outcomes appear beside
each bar.*

- **Request 2:** A fails, so its loss is $0.010 + $0.100 = **$0.110**. B passes
  for **$0.040**. B is preferred, giving **y = 1**.
- **Request 6:** both configurations pass. B has a verified current cache hit and
  costs **$0.004** instead of A's **$0.010**. B is preferred because of the cost
  objective, giving **y = 1**.
- **Request 5:** A has current source data while B's reusable context is stale.
  A passes and B fails, so the reference choice is A, giving **y = 0**.

The target reflects the desired outcome. The more capable executor is not always
the preferred choice. These losses exclude router inference cost. Measure that
cost for each prompt and include it when evaluating deployment cost, because
prompt length and token usage can change it.

## 4. Check the final choice

- Use training examples to guide changes, validation examples to compare
  candidates and tune thresholds, and a separate test set for the frozen choice.
- Keep related examples in the same split. If predicting future data, respect
  time when splitting. Use representative workloads and repeated trials when
  execution is stochastic.
- Record the objective, labeling rule, configurations, and evaluation conditions.
  Replacing an executor requires new measurements and potentially new labels.
- If a test result guides another rewrite, it has become development evidence.
  Use fresh data for an independent check.

Continue to [Section 5: Examples of tuning](examples.md) for prompt tuning and
model-pair comparisons with concrete data and reproducible calculations.

# Tuning

Routing has two outcomes you can measure: how many tasks the agent completes,
and what a run costs. Tuning is the loop below. Every step links the page that
explains it, names what it produces, and says what it costs to run.

Start where your goal is:

| Your goal | Start at |
|---|---|
| Lower cost at the same quality | Step 1, then step 4 for the baseline numbers |
| Add a cheaper model safely | Step 2 |
| Routing is on, and results got worse | Step 5 |

## The loop

1. **Verify routing is active.** Follow
   [Verify routing is active](../getting_started.md#verify-routing-is-active).
   Produces: the selected-model header from two requests and a `/v1/stats`
   capture. Costs: two requests, no agent run.

2. **Choose and check the models.** Run the
   [five checks](../routing_algorithms/model_selection.md#check-the-efficient-model-before-you-route)
   on the efficient model, then the
   [inverted-pair check](../routing_algorithms/model_selection.md#check-for-an-inverted-pair).
   Produces: a pass or fail per check, and solved tasks plus cost for each
   fixed arm. Costs: one short agent run per check, and one agent run per arm
   on at least 20 tasks.

3. **Start from a measured profile.** Pick the closest row in
   [Measured pairs](../routing_algorithms/model_selection.md#measured-pairs)
   and copy its profile from `benchmark/routing-profiles/`. The
   [strategy comparison](../routing_algorithms/overview.md#compare-the-strategies)
   says which route type fits your traffic. Produces: a route file that passes
   `switchyard-server --config routes.toml --dry-run`. Costs: none.

4. **Run the routed arm against the fixed arms.** Use the same tasks, agent,
   and provider for all three arms. The [benchmark guide](../../benchmark/README.md)
   shows the harness the checked-in results used. Produces: solved tasks and
   cost per run for each arm. Costs: `k` runs per arm. The checked-in profiles
   use three runs and report the mean with the sample standard deviation, as in
   the
   [plan/execute profile header](../../benchmark/routing-profiles/deepswe-v11-plan-execute-luna-sol.toml).
   A single run cannot separate a small change from noise.

5. **Read the decision mix.** For stage routes, read
   [decision sources](../routing_algorithms/stage_router_routing.md#decision-sources)
   from `/v1/stats`. For escalation routes, read the
   [observability](../routing_algorithms/escalation_router_routing.md#observability)
   section. For advisor routes, `/v1/stats` reports verdicts and consult
   failures under `advisor_gate`. Start the server with `--routing-log-file`
   to keep one record per request. Produces: the `routing_decisions` capture
   and the routing log. Costs: none beyond the runs in step 4.

6. **Move one dial.** Change one setting, then repeat step 4.

    | Strategy | First dial | Guidance |
    |---|---|---|
    | Execution, Auto | `confidence_threshold` | [Tuning `confidence_threshold`](../routing_algorithms/stage_router_routing.md#tuning-confidence_threshold) and [Calibrating the threshold from run data](../routing_algorithms/stage_router_routing.md#calibrating-the-threshold-from-run-data) |
    | Task | `base_threshold`, then `classify_trigger` | [Tuning options](../routing_algorithms/llm_classifier_routing.md#tuning-options) and [When the judge runs](../routing_algorithms/llm_classifier_routing.md#when-the-judge-runs) |
    | Escalation | `confirmations` | [Tuning options](../routing_algorithms/escalation_router_routing.md#tuning-options) |
    | Advisor Gate | `max_reviews` with `gate_min_tool_results` | [Tuning](../routing_algorithms/advisor_gate_routing.md#tuning) |
    | Composite | `efficient_first_threshold` and `capable_first_threshold` | [What exists today](../routing_algorithms/composite_routing.md#what-exists-today) |

    Produces: a before-and-after table with the arm results from step 4.
    Costs: `k` runs per setting you try.

7. **Record the result.** Write down what you ran, in the form the profile
   headers use: the Switchyard commit, each model with its provider, wire
   format and reasoning effort, the agent and its version, the benchmark and
   task count, `k`, the result band next to the fixed arms, and the cost per
   run. The
   [stage profile header](../../benchmark/routing-profiles/deepswe-v11-stage-router-luna-sol.toml)
   is a complete example. Produces: the header. Costs: none.

8. **Recalibrate on change.** A new model version, provider, wire format,
   reasoning effort, agent version, or workload moves the operating point, and
   the thresholds do not carry over. See
   [Pin the provider and endpoint](../routing_algorithms/model_selection.md#pin-the-provider-and-endpoint).
   Return to step 4 with the recorded header as your baseline.

## Related

- [Routing Overview](../routing_algorithms/overview.md): choose a strategy.
- [Model Selection](../routing_algorithms/model_selection.md): choose and check the models.
- [Troubleshooting](../operations/troubleshooting.md): symptoms and silent misconfigurations.

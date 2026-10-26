# Tuning

Evaluate routing on the complete workload: tasks completed, total model cost,
and latency. Use the same comparison whether routing runs in a hosted provider,
a gateway, or an application embedding libsy.

Start where your goal is:

| Goal | Start at |
|---|---|
| Lower cost while preserving quality | Verify routing, then compare the routed and fixed-model arms. |
| Add an efficient model | Check that model on the actual workload before routing to it. |
| Explain a regression | Inspect which requests changed models, failed, or added judge calls. |

## The loop

### 1. Verify the executed path

Send a small request and confirm which algorithm ran, which model it selected,
and which model actually answered. Selection and execution can differ when
fallback runs. Include routing-time judge calls in your observations.

| Integration | Evidence to collect |
|---|---|
| Direct libsy | The routing outcome, optional decision evidence, and your own dispatch records. The outcome supplies a selection, not proof of a provider call. |
| Supplied HTTP client | Host-exported [client spans and usage](../reference/opentelemetry.md), including candidate attempts and stream completion. |
| Gateway | The gateway's request observations plus the integration's internal-call measurements. See [Relay cost accounting](../integrations/nemo_relay.md#measure-the-cost-of-routing). |
| Hosted provider | Provider-reported served model, usage, and available routing metadata. Follow that provider's documented controls. |

For direct embedding, start with
<a href="../rust/switchyard_libsy/struct.RoutingOutcome.html"><code>RoutingOutcome</code></a>
and the [final-outcome contract](../mechanics_of_libsy.md#stepdone).
Your host must instrument its own model I/O when it uses its own clients.

If using the **local reference server**, the
[two-request verification](../operations/local_evaluation.md#verify-routing-is-active)
shows selected-model headers and HTTP counters. Those endpoints are a demo-server
measurement surface, rather than a requirement for libsy integration.

### 2. Check the models and choose the algorithm

Check tool calling, output format, context length, reasoning controls, and the
endpoint's wire format. Use the actual agent's tool list and realistic request
sizes. The [model checks](../routing_algorithms/model_selection.md#check-the-efficient-model-before-you-route)
show examples through the demo server; you can send equivalent checks through
your provider client or gateway.

Run the candidate models separately before assuming the capable/efficient roles
are correct. Then choose an algorithm using the
[strategy comparison](../routing_algorithms/overview.md#compare-the-strategies).
Record its settings and the model pool. A
[measured configuration](../routing_algorithms/model_selection.md#measured-pairs)
is a starting point for that workload, not a transferable quality guarantee.

### 3. Compare fixed-model and routed arms

Use the same tasks, agent, tool environment, provider settings, and scoring rule.
Run each fixed-model baseline and the routed arm. Repeat runs when sampling or
the environment can change outcomes. Keep a final holdout separate from the
tasks used to choose settings.

| Measurement | Record for every arm |
|---|---|
| Task quality | Completed tasks under the same success criterion; include failures. |
| Cost per task | Answer calls, judge calls, discarded candidates, retries, and fallback attempts where usage or billed cost is available. |
| Latency | End-to-end task duration, plus per-request and routing overhead when relevant. For streaming, distinguish first token from completion. |
| Reliability | Client failures, invalid verdicts, fail-open behavior, and incomplete streams. |
| Routing mix | Selected and served models, switch counts, and judge-call frequency. |

The [benchmark guide](../../benchmark/README.md) explains existing evaluation
setups. [Published results](../benchmarks.md) provide workload-specific examples.

### 4. Account for all model work

Count internal routing work as well as the answer seen by the agent:

```text
observed model cost = sum(cost of each physical attempt with known cost)
cost per successful task = observed total cost / number of successful tasks
```

For task cost, include failed tasks in the numerator. If no tasks succeed, report
that explicitly instead of a cost-per-success figure. Missing usage means unknown
cost, not zero. Prefer provider-billed cost when available. When estimating from
tokens, use that provider's input, output, cache, and reasoning billing rules.
Do not add overlapping token rollups twice.

A response reused from routing is the same answer call, not a second final call.
If a gateway already prices the returned answer, add only the routing work it
has not counted. Some failed or cancelled attempts can be billed without
reporting usage; state that limitation with the result.

### 5. Inspect the decision mix and change one setting

Use outcome evidence, host records, and the
[telemetry reference](../reference/opentelemetry.md) to explain the result.
Libsy emits observations; your application configures collection and export.
The optional client adds I/O measurements. Your custom client owns its own I/O
measurements. Metrics alone do not establish task success.

| Algorithm | First setting to investigate | Guidance |
|---|---|---|
| Stage / Auto | Signal-confidence threshold; use an explicit Stage configuration to tune Auto. | [Threshold behavior](../routing_algorithms/stage_router_routing.md#tuning-confidence_threshold) |
| Task | Solve-probability threshold and classification frequency. | [Classifier tuning](../routing_algorithms/llm_classifier_routing.md#tuning-options) |
| Escalation | Required failure confirmations. | [Escalation tuning](../routing_algorithms/escalation_router_routing.md#tuning-options) |
| Advisor | Review budget and review triggers. | [Advisor tuning](../routing_algorithms/advisor_gate_routing.md#tuning) |
| Composite | Starting-tier policy and Stage thresholds. | [Composite behavior](../routing_algorithms/composite_routing.md#what-exists-today) |

Change one setting and repeat the comparison. Report task quality alongside
cost and latency; a cheaper call is not necessarily a cheaper completed task.

### 6. Record the comparison and recheck on change

Keep the Switchyard revision or hosted integration version, algorithm settings,
candidate models, providers, API formats, reasoning settings, agent version,
task split, scoring rule, and number of repeated runs with the result.

Report the variation across repeated runs. A single run cannot separate a small
gain from noise. Re-evaluate when the model, provider, serving settings, agent,
or workload changes. See [provider and endpoint differences](../routing_algorithms/model_selection.md#pin-the-provider-and-endpoint).

## Related

- [Model selection](../routing_algorithms/model_selection.md): choose and check candidates.
- [Host integration](../reference/host_integration.md): decide where your application observes calls.
- [OpenTelemetry](../reference/opentelemetry.md): exact signals, ownership, and limitations.
- [Local evaluation](../operations/local_evaluation.md): a runnable reference-server example.

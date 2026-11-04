# Diagnose Before You Tune

This page is for a route that runs but does not perform the way you want: cost
is too high, tasks fail, or every request lands on one model. Work through the
three steps below before you change a setting. Then use
[Tuning Foundations](foundations.md) to compare candidates.

## 1. Find the settings you are running

Most deployments start with `type = "auto"`. Auto is a stage-router preset:
`efficient_first`, `confidence_threshold = 0.5`, `recent_turn_window = 3`,
`capable_hold_turns = 2`, and no classifier. See the
[Auto preset](../routing_algorithms/overview.md#auto).

Auto accepts no tuning keys. To change a setting, rewrite the route as the
equivalent `stage_router` route from the
[TOML schema](../reference/toml_schema.md#auto), then edit that route. Record
the starting values; they are your current-routing baseline.

If you copied a profile from the [Start here](../routing_algorithms/overview.md#start-here)
list, that list names the first dial to move for each strategy.

## 2. Check that tuning can help

Read the decision counts first. On the demo server:

```bash
curl -s localhost:4000/v1/stats | jq '.algorithm_stats.stage_router.routing_decisions, (.models | map_values(.calls))'
```

The counts record why each turn was routed; see
[decision sources](../routing_algorithms/stage_router_routing.md#decision-sources).
Three results mean the settings are not the problem:

- **`fall_open` is most of the total.** The tool-signal scorer abstained. This
  happens on chat-only traffic and on the first request of a session. The
  threshold and holds have no effect on these turns. Add the
  [classifier fallback](../routing_algorithms/stage_router_routing.md#optional-llm-classifier-fallback),
  or use Task or Composite routing, which judge the request itself.
- **`.models` lists one entry, or both tiers name the same model.** Every
  request goes to that model. Fix the targets. Then run the
  [inverted-pair check](../routing_algorithms/model_selection.md#check-for-an-inverted-pair):
  if the efficient model is the stronger one, routing only adds cost.
- **Requests carry no session ID.** Holds and escalation state live in the
  session. Without `x-switchyard-session-id`, a hold set by one request is gone
  on the next, and an escalation route never latches. Send the header before
  tuning `capable_hold_turns` or `confirmations`.

[When routing will not help](../routing_algorithms/overview.md#when-routing-will-not-help)
lists the remaining cases.

## 3. Match the symptom to a control

Change one control at a time. The directions below are for the `efficient_first`
picker; `capable_first` reverses the threshold direction. The
[controls table](examples.md#choose-the-stage-router-controls-to-tune) explains
what each value does.

| Symptom | What to read | Move first |
|---|---|---|
| Cost is close to capable-only. | `dimensions` and `capable_hold` are most of the decision counts, and the capable model has most calls in `.models`. | Raise `confidence_threshold` so fewer turns escalate on signals. Then shorten `capable_hold_turns`. |
| `override` is a large share of the decisions. | `override` in the decision counts; `override_reason` in the structured logs. | Repeated failures and critical errors force the capable tier, and the threshold does not change them. [Check the efficient model](../routing_algorithms/model_selection.md#check-the-efficient-model-before-you-route) against the tasks. |
| Tasks fail on the efficient tier and the route rarely escalates. | The efficient model has most calls in `.models`; `override` is low. | Lower `confidence_threshold`, raise `capable_hold_turns`, or add the [classifier fallback](../routing_algorithms/stage_router_routing.md#optional-llm-classifier-fallback). |
| The tiers switch often and the prompt cache is lost. | `cached_tokens` in the routing log drops on the turn after each model change. | Raise `capable_hold_turns` and `efficient_hold_turns`. Measure total cost, not the number of switches. |
| An escalation route never latches. | The log line `stateful escalation has no session ID`. | Send a session ID. Then adjust [`confirmations`](../routing_algorithms/escalation_router_routing.md#tuning-options). |
| The judge never changes the outcome. | `switchyard_classifier_fail_open_total` on `/metrics`; the log line `judge verdict unavailable`. | This is a compatibility problem, not a threshold. See [judge model compatibility](../routing_algorithms/llm_classifier_routing.md#judge-model-compatibility). |

Other algorithms have their own controls. The
[tuning overview](index.md#controls-by-algorithm) lists them.

## Collect the data

Tuning compares candidates on the same tasks, so you need cost, routing split,
and task success for each run.

| Measure | Source |
|---|---|
| Routing split and decision reasons | `GET /v1/stats`: `.models` call counts and `routing_decisions`. `POST /v1/stats/reset` clears them between runs. |
| Cost | The routing log (`--routing-log-file routing.jsonl`): one record per request with the selected model and token usage, including `cached_tokens`. |
| Served model per response | The `x-model-router-selected-model` response header. |
| Metrics in your own host | [OpenTelemetry](../reference/opentelemetry.md#metrics) routing and client metrics. |
| Task success | Your agent harness. For Terminal-Bench and DeepSWE, the [benchmark setup](../../benchmark/README.md#run-with-switchyard-routing) runs a profile and [inspects the result](../../benchmark/README.md#inspect-a-run). |

[Verify routing is active](../operations/local_evaluation.md#verify-routing-is-active)
shows a short session with these outputs.

## Next

[Tuning Foundations](foundations.md) explains how to define the objective,
design candidates, and validate the selected configuration on held-out tasks.

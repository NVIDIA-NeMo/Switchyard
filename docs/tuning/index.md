# Tuning

Tuning helps you find a configuration that meets your goals on your workload.
Start with Foundations to check your current route and plan the comparison, then
use Examples to see how to evaluate candidates and choose a configuration.

- [Tuning Foundations](foundations.md): check your current route, then define your objective, candidates, and evaluation splits.
- [Examples](examples.md): compare routing settings and model pairs with worked examples, and check the efficient model when overrides dominate.

## Controls by algorithm

Examples walks through the stage router. Each algorithm has its own controls:

- [Stage router](../routing_algorithms/stage_router_routing.md#knob-reference): every route key, [how `confidence_threshold` works](../routing_algorithms/stage_router_routing.md#tuning-confidence_threshold), and [how to calibrate it from run data](../routing_algorithms/stage_router_routing.md#calibrating-the-threshold-from-run-data).
- [LLM classifier](../routing_algorithms/llm_classifier_routing.md#tuning-options): thresholds, judge context, and when the judge runs.
- [Escalation router](../routing_algorithms/escalation_router_routing.md#tuning-options): confirmations and judge window.
- [Advisor gate](../routing_algorithms/advisor_gate_routing.md#tuning): review count and stall limits, with the benchmarked settings.
- [Composite](../routing_algorithms/composite_routing.md#independent-stage-profiles): separate stage-router settings per profile.
- [Plan/Execute](../routing_algorithms/plan_execute_routing.md#optional-settings): handoff triggers and prompts.
- [Verification-gated](../routing_algorithms/vgr_routing.md#serving-modes): serving mode and [execution limits](../routing_algorithms/vgr_routing.md#execution-limits).
- [Random](../routing_algorithms/random_routing.md#control-the-traffic-split): target weights.

## Related

- [Start here](../routing_algorithms/overview.md#start-here): a strategy, a profile to copy, and the first dial to move.
- [Model Selection](../routing_algorithms/model_selection.md): choose and check the models.
- [Troubleshooting](../operations/troubleshooting.md): symptoms and silent misconfigurations.

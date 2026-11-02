# Routing profiles

Schema 2 separates reusable strategy settings from the routes clients call.
A route uses a profile directly or asks a Decision model to select one at the
start of a session. The route's public model ID stays the same in both cases.

## Start with one strategy

For a single route, keep the strategy inline. Existing strategy parameters keep
their names and behavior. Client and target definitions are unchanged.

```toml
schema_version = 2

[routes.default]
id = "switchyard"
strategy = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.4
```

This replaces schema 1's `type` key with `strategy`. Both versions are accepted.
Use either key in schema 2, with no mixing in the same strategy table.

## Reuse a profile

Move the strategy settings into a named profile. Each route builds its own
strategy instance, so sharing a definition does not share session state.

```toml
[profiles.adaptive]
strategy = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.4

[routes.default]
id = "switchyard"
profile = "adaptive"
```

All existing algorithm types can be fixed profiles, including `auto`, Stage,
Plan/Execute, Advisor, VGR, and the existing Composite configuration. Their
strategy-specific guides remain the reference for parameters.

A profile contains the full strategy configuration. Routes cannot override its
parameters. Route capabilities such as `context_window` stay on the route.
Profiles cannot inherit from or reference other profiles.

## Select between strategies

Define a second profile, then add a selector to the route. This example chooses
between [Stage](stage_router_routing.md) and [Plan/Execute](plan_execute_routing.md).

```toml
[profiles.planning]
strategy = "plan_execute"
capable_target = "capable"
efficient_target = "efficient"

[routes.default.selector]
kind = "decision"
target = "decider"
choices = { adaptive = "adaptive", planning = "planning" }
positive_choice = "planning"
classify_trigger = "new_session"
on_error = "retain_or_default"

[routes.default.selector.decision]
cutoff = 0.92
candidates = { adaptive = "efficient", planning = "capable" }
criteria = { adaptive = "Adapt during execution", planning = "Plan before execution" }
instructions = { task = "Choose the execution strategy that best fits this task." }
evidence = {}
```

`choices` maps two judge labels to profile names. `candidates` maps those same
labels to model targets for the Decision request. Candidate models describe the
comparison. The selected profile determines which models actually run.
`positive_choice` selects its profile only when its probability is strictly
greater than `cutoff`. Equality selects the other profile.

The selector runs once per session identity. Concurrent first requests share one
selection. Child agents use separate session and agent identities. Without a
session identity, hosts can carry the returned routing state between requests,
or enable `message_hash_fallback` to use the opening user message as identity.
Otherwise each request makes a fresh choice. Retention is process-local and
bounded, like existing routing affinity.

`retain_or_default` uses the route's `profile` when the selector fails or returns
no valid choice, and retains that choice for the session. Set `on_error = "error"`
to fail the request instead. Omit the selector table to run the fixed profile
without a Decision call.

Each selected profile keeps its own model pools, fallbacks, judge configuration,
and routing state. Routing outcomes report `profile` and `strategy`, alongside
the existing judge evidence and usage. An answer produced during routing is
returned directly. It does not trigger a second answer request.

The current selector supports binary Decisions at `new_session`. Existing
Composite still supports its other selection modes. VGR profiles selected by a
Decision model require an explicit route `context_window` when their client uses
context discovery. Target aliases must still agree on model settings and prompts.
Provider-owned Responses continuations retain the existing model pinning behavior.

See the complete [profile configuration](../../examples/profile-routing.toml),
including clients, targets, and a fixed and selected route.

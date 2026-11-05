# Capability Router

`capability_router` composes named signals into one decision. Each signal scores a turn
against a capability tag — `efficient`, `balanced`, `capable`, or a custom tag
your deployment defines — instead of resolving a model itself. The
[pool](pool.md) is the one place that turns a tag into a model.

This replaces the older pattern where one algorithm (such as `composite`) held
another's configuration and wrote a decision into its session state. Signals
sit side by side here; nothing writes into another signal's state.

## Configuration

```toml
[pool]
models = [
  { target = "sol",  provides = ["capable"] },
  { target = "terra", provides = ["balanced"] },
  { target = "luna", provides = ["efficient"] },
  { target = "vega", provides = ["compatibility"] },
]
default = "efficient"

[routes.switchyard]
id = "switchyard"
type = "capability_router"

[routes.switchyard.signals]
affinity = true

[routes.switchyard.signals.task]
judge = "terra"
base_threshold = 0.5

[routes.switchyard.signals.rules]
capable = ["grace-window"]

[routes.switchyard.signals.rules.named]
compatibility = ["historical tie-break"]

[routes.switchyard.signals.tool_loop]
picker = "efficient_first"
confidence_threshold = 0.5
```

A route needs at least one of `task`, `rules`, `tool_loop`, or `review`.

### `task`

A judge-backed capability signal: the same judge behind the `llm_classifier`
route's `capability` mode, scoring `capable` or `efficient` instead of
resolving a target directly.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `judge` | Yes | — | Target the judge is called through. Not a routing destination. |
| `base_threshold` | Yes | — | `p_solve` floor that still names `efficient`. In `[0, 1]`. |
| `threshold_step` | No | `0` | Added per capability-boundary step the judge is uncertain about. |
| `recent_turn_window` | No | unset | Trailing conversation turns the judge sees. Unset shows it the opening task and the latest user follow-up only. |
| `prompt` | No | packaged prompt | Replaces the packaged judge prompt. |
| `response_format_type` | No | `json_schema` | Use `json_object` when the judge provider cannot do JSON Schema. |
| `max_output_tokens` | No | `4096` | Completion cap for the judge's verdict. |

`task` always re-judges every request. Session stickiness is `signals.affinity`'s
job, not a per-signal setting — a route has one place deciding how long a
decision lives.

### `rules`

A deterministic, judge-free signal that matches case-insensitive substrings in
the opening user task. Matches score at full confidence, so they can select a
tier or a specialist from the first request while unmatched tasks leave the
other signals and pool default unchanged.

```toml
[routes.switchyard.signals.rules]
capable = ["grace-window"]
balanced = ["multi-step migration"]

[routes.switchyard.signals.rules.named]
compatibility = ["historical tie-break"]
```

Named tags are checked before `capable`, then `balanced`. Blank terms are
ignored. Every selected tag must be provided by a model in the top-level pool.

### `tool_loop`

The stage-router scorer: reads recent tool-result signals and scores `capable`
or `efficient` from them.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `picker` | Yes | — | `efficient_first` keeps `pool.default` when all signals abstain. `capable_first` (experimental, unbenchmarked) uses `capable` instead. A decisive signal takes precedence over either default. |
| `confidence_threshold` | Yes | — | Corroboration a decisive pick needs. In `[0, 1]`. |
| `capable_score_threshold` | No | `0.55` | With a balanced provider, capable scores below this cutoff use balanced. In `[0, 1]`. |
| `recent_turn_window` | No | `3` | Trailing tool results the signal is computed over. |
| `escalation_hold_turns` | No | `0` | Following requests that keep an elevated tier as a minimum. Requires an explicit session identity. |
| `tool_semantics.observe` / `.mutate` / `.plan` / `.new` | No | `[]` | Additional exact ASCII case-insensitive tool names, same shape as `stage_router`. |
| `handoff_notes` | No | unset | Rejected at load in this release: the note is written while the signal scores, before the pool has chosen a model, so it could describe a handoff that did not happen. |

Set `escalation_hold_turns = 3` to keep a selected `balanced` or `capable`
tier for three following requests. Stronger evidence takes effect immediately.
Fresh elevated selections restart the hold. The efficient tier is never held.
A new human user turn clears the previous hold. Named-only specialist selections
take precedence and clear it. Sessions and child agents have separate holds;
requests without a full explicit identity do not retain decisions. Storage is
process-local and bounded to 4096 identities. A restart loses the holds.
This differs from `stage_router`'s test-pass-sensitive `capable_hold_turns`.
Do not combine this with `affinity` to obtain temporary retention: affinity
bypasses scoring and keeps its first model instead.

When the pool provides `balanced`, `tool_loop` uses three confidence bands:

1. When all signals abstain, `efficient_first` uses `pool.default`, normally `efficient`. `capable_first` uses `capable` instead, including on the first request before any tool activity.
2. A `capable` score below `capable_score_threshold` is changed to `balanced`.
3. A `capable` score at or above that threshold stays `capable`.

The default is `0.55`. Lowering this cutoff sends more moderate tool evidence
to capable. This trades cost for stronger execution; the score is a tool-signal
heuristic, not a calibrated probability that a model will solve the task.

Hard overrides, such as critical severity or repeated failure, have confidence
`1.0` and stay `capable`. A pool without a `balanced` provider keeps the older
two-level behavior. The task judge is optional; a deployment can use only
`rules` and `tool_loop` and make no routing-time LLM call.

### `review`

Wraps the whole turn in an advisor gate: the executor answers every
client-visible turn, and a stronger advisor reviews terminal turns before the
caller sees them. This is the same behavior as the standalone `advisor` route;
see [Advisor-Gate Routing](advisor_gate_routing.md) for the full field
reference (`executor_target`, `advisor_target`, `gate_trigger`, `max_reviews`,
and the rest).

The advisor gate picks its own executor and advisor targets before a pool
decision would even exist, so **`review` cannot be combined with `task`,
`rules`, or `tool_loop`** — a route config with `review` and any other signal is
rejected at startup.

### `affinity`

```toml
[routes.switchyard.signals]
affinity = true
```

When `true`, the first model chosen for a session is retained and replayed on
every later turn in that session, bypassing signal scoring and the pool
entirely. This is `capability_router`'s own stickiness; it is independent of the `pool`
and of any per-signal setting.

## How it decides

1. If `affinity` is on and this session already has an assignment, that model
   is used and nothing else runs.
2. Otherwise, every configured signal (`task`, `rules`, `tool_loop`) scores the turn.
   Unlike a cascade, every signal runs — a later signal is not skipped because
   an earlier one decided.
3. The collected scores are handed to the [pool](pool.md), which picks the
   model.
4. If escalation retention is enabled, a live elevated floor can replace a
   lower tier selection. Named-only specialist selections are left unchanged.
5. If `affinity` is on, this turn's choice is retained for later turns in the
   session.

`review`, when configured, skips all of this: it owns the whole turn itself.

### Target eligibility

When the route enables target eligibility, request requirements filter the pool
before routing. Answer fallbacks use that same filtered pool. The remaining
models must still provide `pool.default` and every capability the configured
signals can require. Required judges must also qualify. A missing required
capability or judge rejects the request before any model call.

### Decision metrics

`switchyard.stage_router.routing_decisions` records one decision after the pool
and escalation retention choose the model. `target_name` is the selected target's
provider model ID. Rules, picker defaults, escalation holds, and affinity replays
carry their own `decision_source`. Tool-score histograms still describe the
scorer's inputs and output, even when another signal wins. The decision counter
describes the first routing candidate; provider execution may use an answer
fallback if that candidate fails.

## Migrating from `composite`

`composite` stacked a judge over a stage router and wrote the judge's tier
into the stage router's session state. The equivalent `capability_router` route declares
the judge as `task` and the stage scorer as `tool_loop`, so tool results keep
driving the turns between judge calls, as the nested stage router did:

```toml
# composite
[routes.switchyard]
type = "composite"
[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"
[routes.switchyard.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5

# capability_router
[pool]
models = [
  { target = "strong", provides = ["capable"] },
  { target = "weak", provides = ["efficient"] },
]
default = "efficient"

[routes.switchyard]
type = "capability_router"
[routes.switchyard.signals.task]
judge = "judge"
base_threshold = 0.5
[routes.switchyard.signals.tool_loop]
picker = "efficient_first"
confidence_threshold = 0.5
```

One behaviour does not carry over. `composite` judged once per user turn and
retained that verdict for the turns in between; `task` judges every request,
because `capability_router` forces its `classify_trigger` to `every_request` so
that one route has one place deciding how long a decision lives. Budget for the
extra judge calls, or drop `task` and let `tool_loop` carry the route.

`signals.affinity` is not the replacement for that retention: it replays the
first model chosen for a session on every later turn, bypassing the pool and
`tool_loop` scoring alike, so the tool-result escalation `composite` provided
would stop after the first turn.

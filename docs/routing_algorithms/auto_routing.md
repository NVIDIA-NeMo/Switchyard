# Unified Auto Routing

Auto collects request signals and selects an answer model from one shared pool.
Signals describe a capability need and confidence. The pool resolves that need
to a configured target. A route can use task classification, tool results, and
answer review together.

## Configure a route

```toml
[pool]
models = [
  { target = "sol", provides = ["capable"] },
  { target = "luna", provides = ["efficient"] },
]
default = "capable"

[routes.switchyard]
id = "switchyard"
type = "auto"

[routes.switchyard.signals]
task = { judge = "terra", trigger = "user_turn", threshold = 0.9 }
tool_loop = { enabled = true, confidence = 0.5 }
review = { judge = "sol", max_reviews = 3, confirmations = 2 }
```

Declare `sol`, `luna`, and `terra` under `[targets.*]` and their clients under
`[llm_clients.*]`. The pool is shared by every Auto route that uses signals.
With no signals configured, Auto resolves `pool.default` without a judge call.

## Task signal

The judge emits its estimate that an efficient model can solve the task. This
raw `p_solve` reaches the selection boundary unchanged. There, an estimate at
or above `threshold` asks for `efficient`, with confidence `p_solve`. A lower
estimate asks for `capable`, with confidence `1 - p_solve`.

| Key | Default | Meaning |
|---|---|---|
| `judge` | Required | Target used for the classification call. |
| `threshold` | Required | Efficient-model solve threshold, from 0 to 1. |
| `trigger` | `every_request` | When to consult the judge. |
| `recent_turn_window` | Unset | Recent conversation shown to the judge. |
| `prompt` | Packaged prompt | Optional replacement judge prompt. |
| `response_format_type` | `json_schema` | Also accepts `json_object`. |
| `max_output_tokens` | `4096` | Judge completion limit. |

`every_request` refreshes the estimate on each request. `user_turn` refreshes
on a new human turn and keeps the estimate for tool continuations. `new_session`
consults once per explicit identity. `on_demand` consults only when the tool
signal abstains. Invalid judge output or a failed consult abstains; the other
signals and pool default remain available.

Retained task evidence competes with fresh tool evidence. It never pins a model
or writes a default into the tool scorer. An explicit session ID is needed for
retention; child agents also need an agent ID. Without identity, the judge runs
when its trigger is considered. Compaction does not start a new user turn.

## Tool-loop signal

`tool_loop` uses the existing stage scorer on recent tool results. It emits
`efficient` or `capable`, with the scorer's confidence. The threshold determines
whether the scorer decides or abstains.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Enables this configured signal. |
| `confidence` | `0.5` | Minimum corroboration for a decisive score. |
| `recent_turn_window` | `3` | Positive number of recent tool results. |
| `tool_semantics` | Built-in vocabulary | Additional tool names, as in Stage Router. |

Auto does not infer a tier ordering from tag names. The tool signal uses the
existing two-capability scorer. Other pool tags can be defaults or be requested
by library signals. Model categories, stage band thresholds, and escalation
holds are configured separately on standalone routes where supported.

## Selection and disagreements

1. Collect all enabled tool and due task scores.
2. Interpret the task probability at its configured threshold.
3. Select the highest-confidence score. On a tie, tool evidence takes precedence.
4. Match any of the winning score's capability tags. With no score, use `pool.default`.
5. Among matching models, prefer one previously selected for the same explicit
   identity. Otherwise use declaration order.

This is a deterministic policy. Tool confidence is a heuristic, so these scores
are not calibrated probabilities across all signal sources. Prices, budget
reservations, and learned weighting do not participate in this policy.

The warm-model preference applies only to candidates satisfying the winning
need. Every request still scores its signals and can change models. History
records routing selections; a provider failure after routing can leave a selected
model without a warm cache. It is a preference, not a provider-cache guarantee.
History and task evidence are process-local, each bounded to 4096 identities,
and cleared for that identity when `session_final = true` succeeds.

## Review

Review runs after the pool selects an executor. That executor produces the
candidate answer. The configured reviewer checks terminal answers using the
existing [Advisor Gate](advisor_gate_routing.md) APPROVE/REDO contract. APPROVE
releases the answer; REDO adds feedback and sends it to the selected executor.
The review budget and fail-open behavior are the existing gate's behavior.

`review.judge` is separate from `task.judge`, and is not an answer fallback
unless it also appears in the pool. `max_reviews` defaults to 1. Optional
`reviewer_system_prompt` replaces the reviewer instruction; `fail_open` defaults
to true. `confirmations` defaults to 1 and must be at least 1 and no greater
than `max_reviews`. It requires that many APPROVE verdicts on the same candidate.
Each consult consumes one review. The group is reserved atomically before
consulting; an insufficient remaining budget uses the existing passthrough
behavior. REDO or failure ends the group and releases unattempted reservations. Review is an answer gate rather than a pre-answer score, so it composes
with task and tool signals without competing on incompatible evidence.

## Migration

Existing standalone routes and Auto's `capable_target`/`efficient_target` preset
continue to load. To migrate a route, remove its model pair, add a shared pool,
and declare its signals. Mixing the old Auto model pair with `signals` is rejected.
HTTP endpoints, target definitions, and client definitions keep their meanings.

For Composite, move the classifier into `signals.task`, with `trigger = "user_turn"`,
and move stage confidence into `signals.tool_loop.confidence`. The new route
retains task evidence and combines it with tool evidence through the pool. The
legacy Composite route keeps its original tier-setter behavior for compatibility.

Target eligibility filters the model pool before the route runs. Both answer
fallbacks and the warm-model preference use the filtered pool. Required signal
capabilities and judges must remain eligible; otherwise the request is rejected.

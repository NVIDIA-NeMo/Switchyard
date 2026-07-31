# TOML Schema

The native deployment file defines the LLM clients, targets, and routes a
Switchyard server serves. It is read by `switchyard-server --config` and by
`switchyard launch --config`.

Validate a file without starting the server:

```bash
switchyard-server --config routes.toml --dry-run
```

## Minimal Example

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "anthropic/claude-sonnet-4.5"
llm_client = "openrouter"

[routes.default]
id = "switchyard"
type = "passthrough"
target = "strong"
```

Clients send the route's `id` as the model name.

## Top Level

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `schema_version` | Yes | — | Must be `1`. |
| `[llm_clients.*]` | No | empty | Upstream API clients. Needed in practice, since every target names one. |
| `[targets.*]` | Yes | — | Named upstream models. |
| `[routes.*]` | Yes | — | Routes this deployment serves. |

Table names under `llm_clients`, `targets`, and `routes` are local references
only. They must be non-empty and carry no surrounding whitespace.

## `[llm_clients.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `format` | Yes | — | `openai_chat`, `openai_responses`, or `anthropic_messages`. |
| `base_url` | Yes | — | Upstream base URL. Must not be empty. |
| `api_key_env` | No | unset | Name of the environment variable holding the key. Omit to send no authentication. |
| `extra_headers` | No | `{}` | Extra HTTP headers sent upstream. |
| `max_retries` | No | `2` | Retry budget, `0`–`10`. Applies to transport failures, timeouts, HTTP 408/429, and 5xx. |

`api_key_env` names a variable; the TOML never contains the secret itself. The
variable must exist and be non-empty at load time.

## `[targets.<name>]`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Exact model ID sent upstream. |
| `llm_client` | Yes | — | Key under `[llm_clients]`. |
| `extra_body` | No | `{}` | Values shallow-merged into the upstream request when the request does not already set that key. |

## `[routes.<name>]`

Every route takes these two keys, plus the keys for its `type`:

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `id` | Yes | — | Model name clients send to select this route. |
| `type` | Yes | — | One of the five route types below. |

Route types are exactly `noop`, `random`, `passthrough`, `llm_classifier`, and
`stage_router`. Any other value is a load error.

### `noop`

Takes no keys beyond `id` and `type`.

### `random`

See [Random Routing](../routing_algorithms/random_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `targets` | Yes | — | Target names to choose from. Must be unique. |
| `weights` | No | equal | Relative weights, in `targets` order. Need not sum to one; at least one must be positive. |
| `seed` | No | unset | Reproduces the selection sequence for the same call order. |

### `passthrough`

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `target` | Yes | — | Target every request is sent to. |

### `llm_classifier`

Classifies each task with `classifier_target`, then routes to `weak_target` or
`strong_target`. See
[LLM Classifier Routing](../routing_algorithms/llm_classifier_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `classifier_target` | Yes | — | Target the judge is called through. Not a routing destination. |
| `strong_target` | Yes | — | Capable tier. |
| `weak_target` | Yes | — | Efficient tier. |
| `base_threshold` | Yes | — | Lowest solve probability that routes to `weak_target`. In `[0, 1]`. |
| `min_confidence` | No | `0.0` | Lowest judge confidence that permits weak routing. In `[0, 1]`; `0.0` disables the gate. |
| `capability_elevated_floor` | No | unset | Higher floor for uncertain, unmatched, and unsupported tasks. In `[0, 1]`, and must exceed `base_threshold`. Unset reuses `base_threshold`. |
| `session_affinity` | No | `false` | Reuses a session's first decision on later turns. |
| `message_hash_fallback` | No | `false` | Keys affinity on the first user message when session metadata is absent. Requires `session_affinity = true`. |
| `recent_turn_window` | No | unset | Trailing turns the judge sees. Unset judges the newest user message alone. |
| `escalation` | No | unset | Switches the route to escalation judging. See below. |

#### `escalation`

An optional table on an `llm_classifier` route. Present, the classifier target
becomes a trajectory judge: the weak tier serves every unlatched turn and is
judged afterward, and the session latches to the strong tier once
`confirmations` consecutive escalate verdicts accumulate. See
[Escalation-Router Routing](../routing_algorithms/escalation_router_routing.md).

There is no `escalation_router` route type.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `confirmations` | No | `2` | Consecutive escalate verdicts required to latch. At least `1`. Values above `1` need a session ID. |
| `recent_turn_window` | No | `28` | Trailing messages shown to the judge. At least `1`. |
| `window_message_chars` | No | `500` | Per-message cap inside that window. At least `50`. |

`base_threshold` stays required, but escalation ignores it, along with
`min_confidence`, `capability_elevated_floor`, `session_affinity`,
`message_hash_fallback`, and the route-level `recent_turn_window`. It is still
range-checked at load time.

### `stage_router`

Scores tool signals to pick a tier per turn, with no classifier call on most
turns. See
[Stage-Router Routing](../routing_algorithms/stage_router_routing.md).

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `capable_target` | Yes | — | Capable tier. |
| `efficient_target` | Yes | — | Efficient tier. |
| `picker` | Yes | — | `capable_first` or `efficient_first`. Tier a turn falls back to when the signals are not confident. |
| `confidence_threshold` | Yes | — | Corroboration a decisive pick needs. In `[0, 1]`. |
| `recent_turn_window` | No | `3` | Trailing tool results the signals are computed over. |
| `capable_system_prompt` | No | unset | System prompt handed to the capable tier on every turn it serves. |
| `efficient_system_prompt` | No | unset | Same, for the efficient tier. |
| `handoff_notes` | No | unset | See below. |
| `classifier` | No | unset | See below. |

#### `[routes.<name>.handoff_notes]`

Notes handed to the model a signal-driven switch routes to.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `escalation_note` | Yes | — | Note handed to the capable tier on an escalation. |
| `deescalation_note` | No | unset | Note handed back to the efficient tier. |
| `only_on_wrong_signal_escalation` | No | `true` | Restricts the escalation note to signal-driven escalations. Set `false` to always send it. |

#### `[routes.<name>.classifier]`

A capability judge consulted only on turns the signals leave undecided.

| Key | Required | Default | Meaning |
|---|:---:|---|---|
| `target` | Yes | — | Target the judge is called through. Not a routing destination. |
| `base_threshold` | Yes | — | As in `llm_classifier`. |
| `min_confidence` | No | `0.0` | As in `llm_classifier`. |
| `capability_elevated_floor` | No | unset | As in `llm_classifier`. |
| `recent_turn_window` | No | unset | Trailing turns the judge sees. Worth setting to the route's `recent_turn_window`. |

`session_affinity` and `message_hash_fallback` parse here but have no effect:
the judge runs as a cascade classifier, not a standalone algorithm.

## Validation and Errors

Every check below runs at load time, so `--dry-run` catches all of them.

### Unknown Keys

Unknown keys are rejected at the top level and in `[llm_clients.*]`,
`[targets.*]`, and `[routes.*]`, with ``unknown field `<name>` ``. A typo in
those tables is a hard load error.

Unknown keys are **silently ignored** in the three nested tables:
`escalation`, `[routes.<name>.handoff_notes]`, and
`[routes.<name>.classifier]`. A typo there is not reported, and the setting
takes its default.

### Messages

| Condition | Error |
|---|---|
| `schema_version` is not `1` | `unsupported schema_version 2; expected 1` |
| Name or ID is empty or padded | `target name must be non-empty and have no surrounding whitespace` |
| `base_url` is empty | `llm client c base_url must not be empty` |
| `max_retries` above `10` | `llm client c max_retries must be at most 10` |
| `llm_client` names no entry | `target strong references unknown llm client missing` |
| Route names no target | `route cls references unknown target missing` |
| Unknown `type` or `format` | ``unknown variant `...`, expected ...`` |
| A required key is absent | ``missing field `base_threshold` `` |
| `random` repeats a target | `random targets must be unique` |
| `weights` length differs from `targets` | `expected 2 weights, got 1` |
| All `weights` are zero | `at least one weight must be positive` |
| `base_threshold` out of range | `base_threshold must be between 0 and 1, got 1.5` |
| `min_confidence` out of range | `min_confidence must be between 0 and 1, got 2` |
| `capability_elevated_floor` out of range | `capability_elevated_floor must be between 0 and 1, got 1.5` |
| `capability_elevated_floor` not above `base_threshold` | `capability_elevated_floor must be greater than base_threshold, got 0.4` |
| `message_hash_fallback` without affinity | `message_hash_fallback requires session_affinity` |
| `confirmations` is `0` | `confirmations must be at least 1` |
| Escalation `recent_turn_window` is `0` | `recent_turn_window must be at least 1` |
| `window_message_chars` below `50` | `window_message_chars must be at least 50, got 10` |
| `confidence_threshold` out of range | `confidence_threshold must be between 0 and 1, got 1.5` |

Route-level messages are prefixed with the route type and table name, as in
`llm_classifier route cls: base_threshold must be between 0 and 1, got 1.5`.

## Related Documentation

- [CLI Reference](../cli_reference.md)
- [Routing Overview](../routing_algorithms/overview.md)

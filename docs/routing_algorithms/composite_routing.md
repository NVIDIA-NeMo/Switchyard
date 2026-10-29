# Composite Routing

Routing algorithms composed together. One sets the configuration of another and
hands off, so each algorithm runs where its evidence is strongest. The composition
is not restricted to a tree: sequential, conditional, and turn-boundary shapes all
fit the same route type.

## What exists today

One pairing: a classifier selects the stage router's starting policy.
An efficient classification selects efficient-first with `efficient_first_threshold`
(default `0.5`).
A capable classification selects capable-first with `capable_first_threshold`
(default `0.0`). Both thresholds are configurable from `0.0` to `1.0`.
Tool signals can still change the tier
during execution. Both algorithms and the
relationship between them are fixed, and the configuration below names them
directly rather than composing arbitrary algorithms.

Under capable-first, `stage.efficient_hold_turns` defaults to `-1`: after switching
to efficient, it stays there until session state is cleared. Efficient-first
defaults to `0`. Set this field explicitly to override either default; `0`
disables the hold, and a positive value holds that many subsequent requests.

The classifier can use an LLM judge or a decision model. A decision
model is configured through a decision client and target, separately from the
capable and efficient completion targets.

The general form, where any algorithm can target another and set its
configuration, is the direction rather than the current behaviour. Expect this
route's configuration to change as more pairings are added.

## Configuration

```toml
[routes.switchyard]
id = "switchyard"
type = "composite"

[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "strong"
efficient_target = "weak"
efficient_first_threshold = 0.5
capable_first_threshold = 0.0

[routes.switchyard.stage.tool_semantics]
observe = ["lookup_customer"]
mutate = ["update_inventory"]
plan = ["create_research_plan"]
new = ["send_message"]
```

`[routes.switchyard.classifier]` is required and takes the `stage_router` classifier fields.
`classify_trigger` sets how often the judge runs: `user_turn` re-picks the tier
whenever the user speaks, `new_session` picks once and holds it.
`[routes.switchyard.stage]` takes the stage settings, but accepts neither `picker`
nor a second `classifier`. The outer classifier supplies the default tier.
`confidence_threshold` remains an alias for `efficient_first_threshold`.
Custom `tool_semantics` remain stage-local and have the same additive behavior as
on a standalone stage route.

### Independent stage profiles

Add `efficient_first` and `capable_first` tables beside `stage` to configure each
profile separately. Both profiles use the target pair in `stage`. The classifier
selects a profile, and only that profile processes the request.

```toml
[routes.switchyard.efficient_first]
confidence_threshold = 0.5
capable_hold_turns = 2
efficient_hold_turns = 0
recent_turn_window = 3
capable_system_prompt = "Resolve the difficult step and continue the task."
efficient_system_prompt = "Carry out the next concrete step."

[routes.switchyard.capable_first]
confidence_threshold = 0.2
capable_hold_turns = 0
efficient_hold_turns = -1
recent_turn_window = 5
capable_system_prompt = "Plan the work and resolve uncertain steps."
efficient_system_prompt = "Continue implementing the established plan."

[routes.switchyard.capable_first.handoff_notes]
escalation_note = "Review the latest error before proceeding."
deescalation_note = "Continue with the current plan."
```

Each profile accepts `confidence_threshold`, `capable_hold_turns`,
`efficient_hold_turns`, `recent_turn_window`, `tool_semantics`, `handoff_notes`,
`capable_system_prompt`, and `efficient_system_prompt`. Profile prompts are
prepended only to requests handled by that profile and tier.

Omitted settings inherit from `stage`. The threshold inherits from
`stage.efficient_first_threshold` or `stage.capable_first_threshold` for the
corresponding profile. Without an explicit efficient hold, efficient-first uses
`0` and capable-first uses `-1`. Prompts default to unset. A profile's
`tool_semantics` or `handoff_notes` table replaces the whole shared table.

Either profile table may be omitted. When either is present, both profiles are
built at startup and keep separate hold state. Switching back resumes that
profile's holds, and inactive profiles do not consume hold turns. Configurations
with neither table keep their existing behavior.

## Custom LLM classifier

To select a tier directly with your own prompt, keep the stage block and use:

```toml
[routes.switchyard.classifier]
mode = "custom"
target = "judge"
classify_trigger = "user_turn"
max_output_tokens = 64
prompt = 'Choose efficient unless capable offers a concrete advantage for this task. Return JSON with decision.target set to efficient or capable.'
response_schema = '''
{"type":"object","properties":{"decision":{"type":"object","properties":{"target":{"type":"string","enum":["efficient","capable"]}},"required":["target"],"additionalProperties":false}},"required":["decision"],"additionalProperties":false}
'''
policy = { type = "target_selector", selector = "/decision/target" }
```

The schema is validated locally. The selector names a tier, and the existing
stage targets supply its model. Custom classifiers always fall open: a missing
or invalid verdict or a client failure retains the previous tier, or uses the
configured stage default for a new session. A failed `new_session` judgment retains
that fallback so tool continuations do not call the judge again.
`fail_open = false` is rejected for custom classifiers. Capability thresholds and
decision-model settings cannot be combined with `mode = "custom"`.

## Decision model classifier

Use a decision target as the classifier. Keep the stage block above
and replace its LLM classifier settings with:

```toml
[decision_clients.decisions]
format = "system_one"
endpoint = "https://openrouter.ai/api/alpha/decisions"
api_key_env = "OPENROUTER_API_KEY"
timeout_ms = 5000

[decision_targets.judge]
id = "inception/mercury-decide:free"
decision_client = "decisions"

[routes.switchyard.classifier]
target = "judge"
classify_trigger = "user_turn"

[routes.switchyard.classifier.decision]
cutoff = 0.4
candidates = { a = "strong", b = "weak" }
evidence = { description = "Add task-relevant evidence comparing a and b." }
```

Both candidates must be completion targets. The judge must be a decision target.
See the [decision client reference](../reference/toml_schema.md#decision_clientsname)
for OpenAI Decisions configuration.

The classifier selects capable when `probabilities.advantage` exceeds `cutoff`.
Calibrate the cutoff for your tasks and models. Omit LLM-only settings such as
`base_threshold`, `threshold_step` and `prompt` when using a decision model.
On judge failure, the default is to retain the last classified tier or use
efficient for a new session. Set `classifier.fail_open = false` to return the error.

## Sessions

The tier is retained per session, so send `x-switchyard-session-id` and each
conversation keeps its own. A client that sends no session ID can set
`classifier.message_hash_fallback = true`, which keys on the first user message
instead. Conversations that open with the same text then share a tier.

Rust code that calls `Algorithm::run_stream` directly can store the tier
itself. Pass the stored `State` to `run_stream`, save `RoutingOutcome.state`
after each turn, and send no session ID. With a session ID, the router uses its
own session state and returns the passed `State` unchanged. `switchyard-server`
starts each request with an empty `State`, so server clients still send
`x-switchyard-session-id`.

## Example use

An interactive coding agent in auto-mode. The classifier reads the user turn and
picks the tier. The stage router runs the tool execution loop on its own signals
until the next user turn.

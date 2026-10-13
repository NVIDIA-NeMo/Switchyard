# Composite Routing

Routing algorithms composed together. One sets the configuration of another and
hands off, so each algorithm runs where its evidence is strongest. The composition
is not restricted to a tree: sequential, conditional, and turn-boundary shapes all
fit the same route type.

## What exists today

One pairing: a classifier sets the tier a stage router falls open to when
its own signals are not confident. The stage router's `picker` mode is unchanged,
and so is how it scores signals, escalates, and de-escalates. Both algorithms and the
relationship between them are fixed, and the configuration below names them
directly rather than composing arbitrary algorithms.

The classifier can use an LLM judge or a System One decision model. A decision
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
confidence_threshold = 0.5

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
Custom `tool_semantics` remain stage-local and have the same additive behavior as
on a standalone stage route.

## Decision model classifier

To use a decision model, replace the classifier's LLM settings with a `decision`
table. Keep the stage configuration above. This example uses OpenRouter's
Decisions endpoint with a System One model. The endpoint, model ID and credential
environment variable can be changed for another compatible provider.

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
fail_open = false

[routes.switchyard.classifier.decision]
cutoff = 0.4
candidates = { a = "strong", b = "weak" }

[routes.switchyard.classifier.decision.evidence]
description = "Add task-relevant evidence comparing candidates a and b."
```

The candidates map opaque labels to completion target names. It must include
both stage tiers. The decision target belongs in `decision_targets`, while the
two completion targets belong in `targets`.

The classifier selects capable when `probabilities.advantage` is strictly above
`cutoff`. That event means capable succeeds and efficient fails. The separate
provider confidence field does not select the tier. Choose the cutoff using
evaluations for your tasks and models. The value above is an example.

Do not combine `decision` with `base_threshold`, `threshold_step`, `prompt`, or
non-default LLM output settings. Use `decision.instructions` to replace the
decision question, and `decision.evidence` to supply comparison evidence.

`fail_open` defaults to `true`. On a judge failure, composite retains the last
classified tier when one is available. A new session defaults to efficient.
Set `fail_open = false` to return the judge error. Stage signals can still
override the default tier on tool turns.

## Sessions

The tier is retained per session, so send `x-switchyard-session-id` and each
conversation keeps its own. A client that sends no session ID can set
`classifier.message_hash_fallback = true`, which keys on the first user message
instead. Conversations that open with the same text then share a tier.

## Example use

An interactive coding agent in auto-mode. The classifier reads the user turn and
picks the tier. The stage router runs the tool execution loop on its own signals
until the next user turn.

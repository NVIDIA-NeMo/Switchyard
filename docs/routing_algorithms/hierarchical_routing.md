# Hierarchical Routing

Routing algorithms composed in a hierarchy. An algorithm higher in the hierarchy
sets the configuration of the one below it before handing off, so each algorithm
runs where its evidence is strongest.

## What exists today

One pairing: an LLM classifier sets the tier a stage router falls open to when
its own signals are not confident. The stage router's `picker` mode is unchanged,
and so is how it scores signals, escalates, and de-escalates. Both algorithms and the
relationship between them are fixed, and the configuration below names them
directly rather than composing arbitrary algorithms.

The general form, where any algorithm can target another and set its
configuration, is the direction rather than the current behaviour. Expect this
route's configuration to change as more pairings are added.

## Configuration

```toml
[routes.switchyard]
id = "switchyard"
type = "hierarchical"

[routes.switchyard.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "strong"
efficient_target = "weak"
confidence_threshold = 0.5
```

`[routes.switchyard.classifier]` takes the `stage_router` classifier fields.
`classify_trigger` sets how often the judge runs: `user_turn` re-picks the tier
whenever the user speaks, `new_session` picks once and holds it.
`[routes.switchyard.stage]` takes the `stage_router` fields except `picker`,
whose job the classifier does per turn, and `classifier`, which would overrule it.

## Example use

An interactive coding agent in auto-mode. The classifier reads the user turn and
picks the tier. The stage router runs the tool execution loop on its own signals
until the next user turn.

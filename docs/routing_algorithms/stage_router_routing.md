# Stage Router

**Execution** routing uses the stage router to choose a model from tool results
and agent progress. Configure it with `type = "stage_router"`.

Stage-router routing sends each request to either a **capable** model or a
typically cheaper **efficient** one, depending on where the agent is in its run.
The goal is to spend the capable model on turns that need exploration or error
recovery and let the efficient model carry routine, mechanical work. Choose a
required `picker` for ambiguous turns, tune the split with
`confidence_threshold`, and optionally add an LLM classifier.

Model-call fallback happens after the routing decision. Eligible context-window
or availability failures can try the route's other completion target. See
[Context-Window Handling](../operations/context_window.md).

## How it works

A tool-using agent's run moves through stages that call for different amounts of
model capability. The built-in vocabulary is calibrated for coding agents: it
recognizes file observation, mutation, planning, shell activity, and test results.

For each LLM call, stage-router estimates the agent's state from the
conversation's **tool-result history**, scoring signals in two directions:

- **RECOVERY → capable**: `severity` (windowed error severity), `spinning` (deep
  churn with no reads or writes), and `exploring` (reading or planning without
  producing) push toward the capable tier.
- **PROGRESS → efficient**: `production_intensity` (the share of recent
  recognized operations that are writes or edits) pushes toward the efficient
  tier with 1.5 times the weight of one signal. Production with no error in the
  recent window adds 1.2 more signals toward efficient, so clean production
  resolves a turn to the efficient tier on its own, even when writes are a small
  share of recent operations.

The axes are **corroborative**: the signed score is `tanh`-squashed to a
confidence in `[0, 1]`. One maxed scoring dimension produces about `0.46`;
corroborating evidence pushes confidence decisively past a `0.5` threshold.
Repeated failures, critical-error severity, and context compaction are hard
overrides to the capable tier. An active capable hold also bypasses the scorer.
Severity and test results come from tool output that ran something. The contents
returned by the built-in read and search tools do not count, unless the tool
reports a failure.

`confidence_threshold` sets how sure that estimate must be before the router acts
on the signal alone. Scores inside the ambiguous band go to the optional
classifier and then, if it cannot decide, to the picker's default tier. A turn
with no tool-result history follows the same fallback path.

The routing decision for one turn:

```mermaid
%%{init: {"flowchart": {"curve": "linear", "nodeSpacing": 36, "rankSpacing": 38}}}%%
flowchart TB
    start(["New turn"])
    history[/"Read tool-result history"/]
    recovery{"Hard recovery signal<br/>or capable hold?"}
    scorer["Compute signed tool-signal score"]
    band{"Outside the<br/>ambiguous band?"}
    classifier{"Usable classifier verdict?"}
    capable(["Route to CAPABLE"])
    signalRoute(["Route by score sign<br/>negative → efficient · positive → capable"])
    classifierRoute(["Route by classifier verdict"])
    defaultRoute(["Route to picker default"])

    start --> history
    history --> recovery
    recovery -->|Yes| capable
    recovery -->|No| scorer
    scorer --> band
    band -->|Yes| signalRoute
    band -->|No · ambiguous| classifier
    classifier -->|Yes| classifierRoute
    classifier -->|No · absent or invalid| defaultRoute

    classDef input fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e,stroke-width:2px;
    classDef process fill:#f8fafc,stroke:#64748b,color:#1e293b,stroke-width:1.5px;
    classDef decision fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px;
    classDef capable fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:3px;
    classDef outcome fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;

    class start,history input;
    class scorer process;
    class recovery,band,classifier decision;
    class capable capable;
    class signalRoute,classifierRoute,defaultRoute outcome;
```

After a signal-driven capable decision, `capable_hold_turns` keeps recovery turns
on that tier for up to two requests by default. A clean passing test clears the
hold early. Set the value to `0` to disable it.

## Pickers

The picker name says which tier is the **default**: the tier used when the
signals are ambiguous and no classifier verdict is available.

- **`efficient_first`**: efficient is the default; escalate to capable only when
  the signals (or the classifier) clearly say so. Cost-first.
- **`capable_first`** *(experimental)*: capable is the default; drop to efficient
  only when the signals (or the classifier) clearly say so. Quality-first.

Both pickers read the same signals; only the default tier differs.

!!! warning "`capable_first` is experimental"

    Every published threshold and routing result comes from `efficient_first`
    runs. `capable_first` works and the server accepts it, but it has not been
    benchmarked, so there are no calibrated thresholds for it and no measured
    accuracy or cost figures to set expectations against. The server logs a
    warning at startup when a route selects it. Use `efficient_first` unless you
    are running your own calibration. See
    [Reachable scores per picker](#reachable-scores-per-picker) for what a
    threshold can and cannot do with each picker.

## Tuning `confidence_threshold`

The tool-signal scorer gives each turn a signed score in `(-1, 1)`: negative
scores point to the efficient tier, positive scores point to the capable tier,
and the absolute value is the confidence. `confidence_threshold` creates a
closed **ambiguous band** from `-threshold` to `+threshold`. Scores outside the
band make a signal-based decision; scores inside it go to the optional
classifier or fall back to the picker's default tier.

The TOML schema requires you to choose `picker` explicitly; there is no implicit
default.

**Set `0.5` explicitly.** `confidence_threshold` is required by the TOML schema;
`0.5` is the recommended starting point, derived from many coding benchmarks,
and what the example below uses.

### How the score is built

The scorer reads four dimensions from the recent tool results:

| Dimension | Range | Meaning |
|---|---|---|
| `severity` | `0`, `0.3` (soft), `0.7` (hard) | The worst error in the window. A critical error (`1.0`) never reaches the scorer; it is a hard override. |
| `spinning` | `0` or `1` | A deep turn with no reads, writes, plans, or new activity. |
| `exploring` | `0` or `1` | A deep turn that reads or plans but does not write. Never set together with `spinning`. |
| `production_intensity` | `0` to `1` | The share of recent recognized operations that were writes or edits. |

The score is a signed sum of these dimensions in signal units, squashed by
`tanh`:

```text
units = severity / 0.7 + spinning + exploring - 1.5 * production_intensity - clean_production
score = tanh(5.0 * 0.10 * units)
confidence = |score|
```

`clean_production` is `1.2` when the window has production and no error, and
`0` otherwise. The constants live in
[`crates/libsy/src/algorithms/util/stage.rs`](../../crates/libsy/src/algorithms/util/stage.rs):
`SIGNAL_UNIT` 0.10, `SCORE_GAIN` 5.0, `HARD_SEVERITY` 0.7, `PRODUCTION_WEIGHT`
1.5, `CLEAN_PRODUCTION_WEIGHT` 1.2, and `STALL_MIN_TURN_DEPTH` 8, below which
`spinning` and `exploring` stay `0`.

One signal unit is one maxed dimension. The confidence a given number of units
produces:

| Units | Confidence | Example |
|---|---|---|
| 0.43 | 0.21 | A soft error alone. |
| 1.0 | 0.46 | A hard error alone, or a stall alone. |
| 1.2 | 0.54 | Clean production at the smallest share. |
| 1.5 | 0.64 | Every recent operation is a write, with an error in the window. |
| 2.0 | 0.76 | A hard error plus a stall. The most the signals reach toward capable. |
| 2.7 | 0.87 | Every recent operation is a write and no error. The most the signals reach toward efficient. |

Reproduce the values with
`python3 -c 'import math; print(math.tanh(0.5), math.tanh(0.75), math.tanh(1.0), math.tanh(1.35))'`.

The dimensions limit how far the score can reach in each direction. Toward
capable, the most the signals can produce is a hard error plus a stall, 0.76.
Production turns the stall signals off, so with any production in the window
the capable side stays below 0.46. Toward efficient, the most is full clean
production, 0.87. An error in the window removes the clean-production bonus and
pulls the other way: with a soft error the efficient side tops out at 0.49, and
with a hard error at 0.24.

### Reachable scores per picker

Both pickers use the same score. The picker only sets which direction the
signals must prove.

| Picker | The signals move a turn to | Reachable confidence | What a threshold means |
|---|---|---|---|
| `efficient_first` | Capable (escalation) | Up to 0.76 | At `0.5`, a hard error alone (0.46) stays in the band, so an escalation needs a second signal, a hard override, or the classifier. At `0.76` and above, the signals never escalate; only overrides, capable holds, and the classifier can. |
| `capable_first` | Efficient (offload) | Up to 0.87, and at most 0.49 when an error is in the window | At `0.5`, clean production offloads (0.54 and up) and nothing else does. At `0.87` and above, the signals never offload. |

How to move the dial:

- With `efficient_first`, lower the threshold to escalate on weaker evidence.
  Below 0.46 a hard error alone escalates. Raise it to keep more turns on the
  efficient tier.
- With `capable_first`, lower the threshold to offload more. Below 0.49,
  production with a soft error in the window also offloads. Raise it to keep
  more turns on the capable tier. The server warns at startup that
  `capable_first` has no calibrated threshold.

Hard overrides, capable holds, and the classifier do not use the threshold.

### What `0.5` means with `efficient_first`

![Illustrative Stage score distribution with an ambiguous band from -0.5 to 0.5. Scores below -0.5 route efficient, scores in the band fall back to efficient, and scores above 0.5 route capable. Dotted markers show the signal ceilings: +0.76 toward capable and -0.87 toward efficient.](../assets/stage-router-threshold.svg)

With `picker = "efficient_first"`, no classifier, and a threshold of `0.5`:

- scores below `-0.5` route to efficient from the tool signals;
- scores from `-0.5` through `+0.5` are ambiguous and fall back to efficient;
- scores above `+0.5` route to capable from the tool signals.

Hard overrides and capable-hold state can still select capable independently of
this score. Lowering the threshold narrows the ambiguous band, so more turns are
decided directly by the scorer. Raising it widens the band, so more turns use the
picker default (or the classifier, when configured). That movement changes the
efficient/capable routing split.

| `confidence_threshold` | Behavior | Typical use |
|---|---|---|
| `0.0` | Every non-zero score makes a signal-based decision. A neutral score or missing tool history still falls through. | Maximize decisions by the tool-signal scorer. |
| `0.5` | Scores from `-0.5` through `+0.5` are ambiguous. | Recommended starting point, derived from many coding benchmarks. |
| `1.0` | Every ordinary score is ambiguous; hard overrides and capable holds still apply. | Classifier-driven when a classifier is configured; picker-driven otherwise. |

A route with a `1.0` threshold remains valid without a classifier. In that case,
ordinary sub-threshold turns fall back to the picker's default tier; hard
overrides and capable-hold behavior still apply.

The signal-vs-classifier split is dataset-dependent. Measure it in production:
`/v1/stats` reports decision counts by source and selected target, plus aggregate
scorer metrics. Response headers and structured decision logs explain individual
selections.

### Calibrating the threshold from run data

Use about 10% of your representative tasks. Replay their agent histories through
the Stage tool-signal scorer and collect the unthresholded signed score for each
ordinary turn. Track hard overrides and capable holds separately because changing
the threshold does not affect them. Plot the score distribution, then overlay
candidate ambiguous bands.

For `efficient_first`, count how many turns fall below the band, inside it, and
above it. Without a classifier, those regions map to signal-selected efficient,
default-efficient, and signal-selected capable decisions. Choose the threshold
that gives the routing split you want, then validate it on the same sample before
running the full task set.

## Route configuration

### Knob reference

Every key a `stage_router` route accepts. The `handoff_notes` and `classifier`
tables are optional. The
[TOML schema](../reference/toml_schema.md#stage_router) has the full table with
required columns.

| Key | Default | Meaning |
|---|---|---|
| `capable_target` | required | Capable tier. |
| `efficient_target` | required | Efficient tier. |
| `picker` | required | `efficient_first` or `capable_first` (experimental). Tier used when the signals are not confident. |
| `confidence_threshold` | required | Corroboration a decisive pick needs, `0` to `1`. Alias `efficient_first_threshold`. |
| `capable_first_threshold` | `0.0` | Used only after an outer `composite` classifier selects the capable tier. No effect on a standalone route. |
| `recent_turn_window` | `3` | Trailing tool results the signals are computed over. |
| `capable_hold_turns` | `2` | Requests kept on the capable tier after a signal-driven escalation. `0` disables. |
| `tool_semantics.observe` | `[]` | Domain tool names that count as investigation. |
| `tool_semantics.mutate` | `[]` | Domain tool names that count as production. |
| `tool_semantics.plan` | `[]` | Domain tool names that count as planning. |
| `tool_semantics.new` | `[]` | Domain tool names that count as neutral forward activity. |
| `handoff_notes.escalation_note` | required in the table | Note handed to the capable tier on a signal-driven escalation. |
| `handoff_notes.deescalation_note` | unset | Note handed to the efficient tier on a signal-driven return. |
| `handoff_notes.only_on_wrong_signal_escalation` | `true` | `false` also adds the escalation note to capable-hold turns. |
| `classifier.target` | required in the table | Judge target. Not a routing destination. |
| `classifier.base_threshold` | required for LLM judges | Lowest `p_solve` that still routes efficient. |
| `classifier.threshold_step` | `0.0` | Added once for uncertain or unmatched verdicts, twice for unsupported. |
| `classifier.fail_open` | `true` | On judge failure, fall open to the picker default instead of failing the request. |
| `classifier.decision` | unset | Use a decision model instead of an LLM judge. |
| `classifier.classify_trigger` | `every_request` | No effect here. The judge runs on every ambiguous turn. |
| `classifier.message_hash_fallback` | `false` | No effect here. A load error with the default trigger. |
| `classifier.recent_turn_window` | unset | Trailing turns the judge sees after the opening task. |
| `classifier.prompt` | packaged | Replaces the judge prompt. |
| `classifier.response_format_type` | `json_schema` | `json_object` for providers without JSON Schema. |
| `classifier.max_output_tokens` | `4096` | Token budget for the verdict. |
| `subagents` | unset | Separate policy for delegated sub-agent work. |

`classifier.mode`, `classifier.response_schema` and `classifier.policy` belong to
[composite](composite_routing.md#custom-llm-classifier) routes and are rejected
here.

### Minimal route

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.capable]
id = "openai/gpt-4o"
llm_client = "openrouter"
# system_prompt = "diagnose before you edit"  # optional

[targets.efficient]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"
# system_prompt = "follow the settled plan"  # optional

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.5
recent_turn_window = 3          # optional, defaults to 3
capable_hold_turns = 2          # optional, defaults to 2
```

Save as `routes.toml` and start the server:

```bash
switchyard-server --config routes.toml --port 4000
```

This is the recommended default: routing on tool signals alone, no classifier.

### Optional: custom tool semantics

Extend the built-in coding vocabulary when an agent uses domain-specific tool
names. Mappings are route-scoped, additive, and matched by exact name without
regard to ASCII case:

```toml
[routes.stage.tool_semantics]
observe = ["KB_search", "get_customer_by_phone"]
mutate = ["send_payment_request", "update_inventory"]
plan = ["create_research_plan"]
new = ["start_conversation", "send_message_to_user"]
```

The categories affect existing stage signals:

- `observe` counts as investigation, like the built-in read and search tools.
- `mutate` counts as production, like the built-in write and edit tools.
- `plan` counts as investigation, like the built-in planning tools.
- `new` records forward activity that suppresses false `spinning` and
  `exploring` signals, but does not otherwise favor either tier.
- Unmatched tools remain `unknown`; there is no `unknown` configuration key.

Configuration cannot reclassify a built-in tool. Empty names, duplicate names
across categories, and unknown category keys are rejected when the route is
loaded. Argument-aware wrapper tools, inferred semantics, and learned routing
rules are outside this exact-name configuration.

MCP tools match by their bare tool name or by their full name. Claude Code sends
an MCP tool as `mcp__<server>__<tool>`, and Codex sends it with a separate
`namespace`. For both, `send_payment_request` matches the `send_payment_request`
tool on the `billing` MCP server, and so does
`mcp__billing__send_payment_request`. A server name that contains `__` needs the
full name.

### Optional: handoff notes

Add a `[routes.stage.handoff_notes]` section to append contextual guidance to a
request when tool signals decide its tier. Notes exist only in the forwarded
request; they do not accumulate in the caller's conversation. The escalation
note accompanies a signal-driven capable decision, and the optional de-escalation
note accompanies a signal-driven efficient decision.

```toml
[routes.stage.handoff_notes]
escalation_note = "the previous model was stalling; pick up the diagnosis"
# deescalation_note = "..."          # optional
# only_on_wrong_signal_escalation = true  # default; false also includes capable holds
```

Classifier and picker-default decisions do not add handoff notes.

### Optional: LLM classifier fallback

The block is optional. Without it, an ambiguous turn goes directly to the
`picker` tier.

To judge ambiguous turns with a model call, add the block. The classifier runs
when the scorer has no tool history or produces a score inside the closed
ambiguous band:

```toml
[targets.stage_judge]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.stage.classifier]
target = "stage_judge"     # judge-only; never a completion destination
base_threshold = 0.5       # p_solve floor to route efficient; below this → capable
threshold_step = 0.1       # +0.1 for uncertain/unmatched; +0.2 for unsupported
recent_turn_window = 3     # trailing turns added to the opening task
prompt = "Estimate whether the efficient target can complete this request."
response_format_type = "json_object"  # optional; default is "json_schema"
```

`prompt` replaces the packaged capability-classifier prompt. In the default
`json_schema` mode, Switchyard sends the verdict schema through the
structured-output request. Set `response_format_type = "json_object"` for
providers that only support JSON Object mode; Switchyard then adds the schema to
the judge prompt and validates the returned object locally. The verdict schema
and routing thresholds remain unchanged.

The nested stage table in [Composite Routing](composite_routing.md) does not
accept a `classifier` block. Its outer classifier supplies the tier used when
the stage signals are ambiguous.

Give the classifier its own LLM client or quota bucket where possible. Sharing
one provider bucket with the efficient tier adds a request per classified turn
and can cause sustained 429s at scale.

## Observability

When a model serves the request, the response identifies it with this routing
header:

| Header | Content |
|---|---|
| `x-model-router-selected-model` | The model ID that served the response, including any model-call fallback. |

### Decision sources

The router records an internal `decision_source` for each turn to distinguish the
paths through its cascade:

| Source | When |
|---|---|
| `override` | A repeated failure, critical-error severity, or context-compaction marker forced the capable tier. Structured logs set `override_reason` to `repeated_failure`, `critical_error`, or `compaction`. |
| `capable_hold` | A recent escalation kept this recovery turn on the capable tier. |
| `dimensions` | The signed score fell outside the ambiguous band and selected a tier. |
| `ambiguous` | The signed score stayed inside the band. This is a step, not a final source: the turn continues to the classifier or the picker default, and `/v1/stats` records that later source. |
| `llm-classifier` | The signals were ambiguous and the classifier returned a usable verdict. |
| `fall_open` | The signal scorer abstained and the classifier was absent or could not decide, so the picker default was used. |

## When *not* to use stage-router

- **Single-model deployments.** Use a `passthrough` route instead.
- **Probabilistic A/B splits.** Use
  [Random Routing](random_routing.md) (`type = "random"`).
  The stage-router's signals are wasted on a fixed traffic ratio.
- **No tool-result history.** Stage-router needs meaningful tool-call traffic to
  populate its signals. For pure chat-completion workloads, the signal scorer
  abstains; the optional classifier decides, or the request uses the picker
  default when no classifier is configured.

## Related

- [Model Selection](model_selection.md): choose and check the two models before
  tuning the threshold.
- [Architecture](../architecture.md): the end-to-end request lifecycle and
  system boundaries.
- [TOML Schema](../reference/toml_schema.md#stage_router): the complete
  configuration reference.

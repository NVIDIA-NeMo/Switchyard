# Routing Overview

Switchyard profile configs register profiles and targets as model IDs that
clients can select through OpenAI Chat Completions, Anthropic Messages, or
OpenAI Responses API requests. Start a profile config with:

```bash
switchyard serve --config profiles.yaml --port 4000
```

Use this page to choose a routing strategy first, then open its detailed page
for configuration and tuning.

## Choose a strategy

| Strategy | Use it when | Profile `type` |
|---|---|---|
| [Random Routing](random_routing.md) | You need a fixed strong/weak split for A/B tests, baselines, or cost experiments. | `random-routing` |
| [LLM Classifier Routing](llm_classifier_routing.md) | Request content should decide whether a turn needs the weak or strong tier. | `llm-routing` |
| [Stage-Router Routing](stage_router_routing.md) | Tool-result and agent-progress signals should route most turns without an extra classifier call. | `stage_router` |
| [Escalation-Router Routing](escalation_router_routing.md) | Start every task on the weak tier and let an LLM judge escalate to strong — one-way per task — when the run is in trouble. | `escalation_router` (`routes:` bundle) |

[Session Affinity (Sticky Routing)](sticky_routing.md) is an opt-in feature of
LLM classifier routing, not a standalone routing strategy. The classifier
integrates the pin into its decision path. See
[How session affinity composes](#how-session-affinity-composes) for the exact
behavior.

## Common profile shape

Profile configs separate provider connectivity, upstream targets, and
client-facing profiles:

```yaml
endpoints:
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
    base_url: https://openrouter.ai/api/v1

targets:
  strong:
    endpoint: openrouter
    model: openai/gpt-4o
    format: openai
  weak:
    endpoint: openrouter
    model: openai/gpt-4o-mini
    format: openai

profiles:
  smart:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 0.3
```

The profile ID (`smart`) is the model ID clients send when they want the
routing policy. Target IDs (`strong` and `weak`) are also directly selectable.
When an upstream model ID differs from its target ID, the profile server
registers that model ID as an additional direct alias.

The examples use model IDs from the
[OpenRouter model catalog](https://openrouter.ai/api/v1/models). Select IDs
available to your account before deploying; catalog availability can change.

## Multiple profiles

A single file can declare multiple profiles over the same targets. Each
profile and target appears on `GET /v1/models`:

```yaml
profiles:
  fast:
    type: passthrough
    target: weak

  smart:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 0.3
```

Use the profile ID to select policy behavior (`fast` or `smart`) and a target ID
to bypass routing (`weak` or `strong`).

## Direct targets and passthrough aliases

For new profile configs, use one public model concept:

- Select a target ID directly when its configured name is the client-facing
  name you want.
- Add a `type: passthrough` profile only when you need another stable alias for
  that target, such as `fast` above.

Both choices call the same single target. There is no `type: model` profile in
the current profile schema.

The deprecated `--routing-profiles` route-bundle format has a separate legacy
distinction: `type: model` registers one explicit alias without model discovery,
while `type: passthrough` queries the upstream model catalog and registers the
discovered models too. That distinction applies only to legacy `routes:`
bundles used by the launcher compatibility path.

## Self-hosted targets

Any profile target can point at an OpenAI-compatible model server you operate.
For example, start a local vLLM server:

```bash
vllm serve ./my-rl-qwen --served-model-name my-rl-qwen --port 8000
```

Then declare it as a normal endpoint and target:

```yaml
endpoints:
  local:
    base_url: http://localhost:8000/v1
    api_key: dummy

targets:
  local-weak:
    endpoint: local
    model: my-rl-qwen
    format: openai

profiles:
  fast-local:
    type: passthrough
    target: local-weak
```

Reference `local-weak` from any routing profile field that accepts a target ID,
including `strong`, `weak`, `target`, or `targets`. Switchyard does not start or
manage the model server; it only sends requests to the configured endpoint.

## How session affinity composes

Session affinity is configured directly on the LLM classifier router. It is not
a generic wrapper applied after every routing strategy. After the configured
warmup, the first confident policy, tool-planning, or alignment verdict pins
the tier. Abstain, low-confidence, missing-signal, and fail-open decisions never
pin. The classifier and tier selector share one affinity store, and later turns
check that store before classification, reuse the tier, and skip the classifier
call.

Pins use a bounded in-process LRU keyed from the stable conversation prefix.
They are not shared across workers or restarts. See
[Sticky Routing](sticky_routing.md) for configuration and key derivation.

Random and stage-router routing do not expose session-affinity settings; they
continue to make a routing decision for each request.

The escalation router uses the same affinity store with the opposite policy:
affinity is always on (the pin *is* the escalation latch), only the strong
tier is ever pinned, and the judge keeps running every turn until the latch
fires. See
[Escalation-Router Routing](escalation_router_routing.md).

!!! note "CLI schema availability"
    The CLI currently accepts these settings in a `deterministic` entry in a
    `routes:` bundle loaded with `--routing-profiles`. The Rust `llm-routing`
    profile loaded by `switchyard serve --config` does not yet expose
    session-affinity fields.

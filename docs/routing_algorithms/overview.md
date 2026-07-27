# Routing Overview

The Python CLI loads routing policies from a YAML bundle. Each key under
`routes:` becomes a model ID available through OpenAI Chat Completions,
Anthropic Messages, and OpenAI Responses requests:

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000
```

Use this page to choose a routing strategy, then open its detailed page for
configuration and tuning.

## Choose a strategy

| Strategy | Use it when | Route `type` |
|---|---|---|
| [Random Routing](random_routing.md) | You need a fixed strong/weak split for A/B tests, baselines, or cost experiments. | `random_routing` |
| [LLM Classifier Routing](llm_classifier_routing.md) | Request content should decide whether a turn needs the weak or strong tier. | `deterministic` |
| [Stage-Router Routing](stage_router_routing.md) | Tool-result and agent-progress signals should route most turns without an extra classifier call. | `stage_router` |
| [Escalation-Router Routing](escalation_router_routing.md) | Start every task on the weak tier and escalate to strong when an LLM judge detects trouble. | `escalation_router` |

[Session Affinity (Sticky Routing)](sticky_routing.md) is an opt-in feature of
LLM classifier routing, not a standalone routing strategy.

## Common route shape

Provider defaults can be shared across all routes, while each route owns its
target configuration:

```yaml
defaults:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  format: openai

routes:
  fast:
    type: model
    target: openai/gpt-4o-mini

  smart:
    type: random_routing
    strong:
      model: openai/gpt-4o
    weak:
      model: openai/gpt-4o-mini
    strong_probability: 0.3
    fallback_target_on_evict: strong
```

Use the route name (`fast` or `smart`) as the request's model ID. A single
bundle can serve multiple routes on the same host and port.

The examples use model IDs from the
[OpenRouter model catalog](https://openrouter.ai/api/v1/models). Select IDs
available to your account before deploying; catalog availability can change.

## Model and passthrough routes

- `type: model` registers one explicit model alias without model discovery.
- `type: passthrough` queries the upstream model catalog and registers the
  discovered models.

Both create direct, single-target chains. Use a routing policy when requests
must be split or classified across targets.

## Self-hosted targets

Any route target can point at an OpenAI-compatible model server you operate.
For example, start a local vLLM server:

```bash
vllm serve ./my-rl-qwen --served-model-name my-rl-qwen --port 8000
```

Then configure it as a normal route:

```yaml
routes:
  local:
    type: model
    target: my-rl-qwen
    base_url: http://localhost:8000/v1
    api_key: dummy
    format: openai
```

Switchyard does not start or manage the model server; it only sends requests
to the configured endpoint.

## Sub-agent override (`subagent_target`)

Any route may name an optional `subagent_target` in its common envelope,
alongside `type`. A request carrying a recognized sub-agent signal — Claude Code
agent-lineage headers, Codex delegated-work kinds (`x-openai-subagent:
collab_spawn` or `review`), or an explicit `x-switchyard-is-subagent: true` —
bypasses the route's own chain and runs as a direct passthrough to that target:

```yaml
routes:
  assistant:
    type: model
    target:
      model: strong-model
    subagent_target:
      model: cheaper-worker-model
```

This keeps a sub-agent loop on one intentional, cache-compatible target instead
of re-routing every worker turn. The worker may live on a different provider
entirely — give it its own `base_url` and `api_key` and one route spans two
upstreams:

```yaml
    subagent_target:
      model: my-local-model
      base_url: http://localhost:8000/v1
      api_key: dummy
      format: openai
```

Detection is the protocol crate's canonical policy, shared with the libsy
`SubagentOverride` classifier, so both engines agree on what counts as
delegated work. Harness-maintenance turns (`compact`, `memory_consolidation`)
and unrecognized kinds stay on normal routing, as does everything else when the
field is absent. A worker-target failure surfaces as a normal target error — it
is never silently re-routed through the route's own chain.

To suppress sub-agent routing for a request that carries a recognized signal,
send `x-switchyard-is-subagent: false`. This explicit header overrides Codex and
Claude Code lineage signals in either direction: `false` keeps the request on
normal routing even when delegated-work headers are present, and `true` marks a
request as a sub-agent even when no harness headers appear.

The key applies to `model`, `passthrough`, `deterministic`, `escalation_router`,
and `stage_router` routes. `random_routing` expands into its table entries on a
separate path and does not consume it.

## How session affinity composes

Session affinity is configured directly on the LLM classifier router. After
the configured warmup, the first confident policy, tool-planning, or alignment
verdict pins the tier. Abstain, low-confidence, missing-signal, and fail-open
decisions never pin. Later turns reuse the tier and skip the classifier call.

Pins use a bounded in-process LRU keyed from the stable conversation prefix.
They are not shared across workers or restarts. See
[Sticky Routing](sticky_routing.md) for configuration and key derivation.

Random and stage-router routing make a fresh routing decision for each request.
The escalation router instead uses affinity as a one-way escalation latch: only
the strong tier is pinned, and the judge runs until that latch fires.

## Rust server configuration

The `switchyard-server` binary has a separate TOML schema that explicitly
constructs LLM clients, targets, and libsy algorithms. It does not load Python
route bundles. See the
[Rust server README](../../crates/switchyard-server/README.md) for its supported
configuration.

# Switchyard Documentation

Switchyard is a typed control plane for LLM traffic. It sits between client
applications and model backends, translates OpenAI Chat / Anthropic Messages /
OpenAI Responses formats, and routes each request through profile-backed chains.

Use Switchyard when you want coding agents, SDK clients, or internal services to
keep their native API shape while traffic is served by a different provider,
split across model tiers, or selected by routing policy.

## Project Overview

| Area | What Switchyard provides |
|---|---|
| Client ingress | OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses compatible endpoints. |
| Agent launchers | One-command local proxies for Claude Code, Codex, and OpenClaw. |
| Format translation | Request and response translation between supported wire formats. |
| Routing policies | Random splits, LLM classifier routing with optional session affinity, signal-driven stage-router routing, and YAML route bundles. |
| Operations | Request/token statistics and context-window fallback behavior. |
| Deployment options | Local coding-agent proxy, shared HTTP service, or embedded Python runtime. |

At a high level, Switchyard keeps client integrations separate from model
providers and routing policy:

```text
clients -> compatible API surface -> routing and resilience -> model backends
```

For system context and request lifecycle diagrams, see
[Architecture](architecture.md).

## First Run

```bash
pip install "nemo-switchyard[cli,server]"
switchyard configure
switchyard launch claude
```

For source installs, non-interactive configuration, and a curl sanity check, use
[Getting Started](getting_started.md).

## Main Workflows

<div class="grid cards" markdown>

- **Run coding agents**

    Launch Claude Code, Codex, or OpenClaw through a local Switchyard proxy.

    [Agent Launchers](guides/agent_launchers.md)

- **Configure routing**

    Pick between fixed splits, classifier routing, and stage-router routing, with
    optional session affinity for classifier-driven conversations.

    [Routing Overview](routing_algorithms/overview.md)

- **Prepare skill distillation**

    Save a namespace for the intended automatic skill learning flow without
    changing request routing.

    [Skill Distillation](skill_distillation.md)

- **Understand the system**

    See how clients, routing policy, model backends, and operations fit
    together.

    [Architecture](architecture.md)

- **Operate the proxy**

    Understand context-window overflow handling and fallback behavior.

    [Context-Window Handling](operations/context_window.md)

</div>

## Configuration Model

Python CLI deployments define client-facing routes in a YAML bundle:

```yaml
defaults:
  api_key: ${OPENROUTER_API_KEY}
  base_url: https://openrouter.ai/api/v1
  format: openai

routes:
  smart:
    type: random_routing
    strong:
      model: openai/gpt-4o
    weak:
      model: openai/gpt-4o-mini
    strong_probability: 0.3
    fallback_target_on_evict: strong
```

Run it as a long-lived proxy. Route names appear as models on `GET /v1/models`,
and clients select one with the request's `model` field:

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000
```

The same bundle can drive a launcher or be saved as the default:

```bash
switchyard --routing-profiles routes.yaml -- launch claude
switchyard --routing-profiles routes.yaml -- configure --target provider \
  --provider openrouter --api-key "$OPENROUTER_API_KEY" \
  --base-url https://openrouter.ai/api/v1 --no-tui --no-model-discovery
```

Non-interactive `configure` does not read provider credentials from the routing
bundle; pass `--api-key` explicitly when persisting the bundle for CI.

Route types, launcher use, and persistence are covered in
[Routing Overview](routing_algorithms/overview.md). The Rust server uses its
own explicit TOML configuration for LLM clients, targets, and libsy algorithms.

## Routing Reference

| Need | Read |
|---|---|
| Fixed strong/weak traffic split for baselines or A/B tests | [Random Routing](routing_algorithms/random_routing.md) |
| Per-request strong/weak decisions from a classifier model | [LLM Classifier Routing](routing_algorithms/llm_classifier_routing.md) |
| Signal-driven weak/strong escalation with optional classifier fallback | [Stage-Router Routing](routing_algorithms/stage_router_routing.md) |
| Conversation-level affinity for cache reuse | [Sticky Routing](routing_algorithms/sticky_routing.md) |

## Operations and Reference

| Topic | Read |
|---|---|
| Known limitations and workarounds for 0.1.0 | [Known Issues](known_issues.md) |
| CLI syntax, flags, resolution rules, and environment variables | [CLI Reference](cli_reference.md) |
| Context-window overflow retry and fallback behavior | [Context-Window Handling](operations/context_window.md) |

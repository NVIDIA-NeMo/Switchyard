---
name: switchyard-lib-core
description: Use when adding, modifying, refactoring, renaming, restructuring, deprecating, or reviewing anything under `switchyard/lib/` — profiles, request/response processors, backends, translators, stats collection, intake, telemetry, observability, routing decisions, or CLI wiring that builds a runnable profile. Triggers on phrases like "add a profile", "new processor", "new backend", "wire stats", "add a preset", "track per-tier …", "intake telemetry", "rename random_routing", "refactor construction", or any edit to `switchyard/lib/profiles/`, `processors/`, `backends/`, `translators/`, or route-table code.
---

# Switchyard Lib Core

## Overview

Switchyard construction is profile-backed. A typed profile config owns how its
runtime is built, and serving through the current Python endpoints adapts that
runtime with `ProfileSwitchyard`. Do not reintroduce factories, recipes,
middleware bundles, request/response pipeline wrappers, or resource-cache
construction paths.

The chain executor still runs the same logical stages:

```text
request-side work -> LLMBackend -> response-side work -> TranslationEngine
```

Profiles make that shape local to the behavior being implemented instead of
splitting it across factory hooks and global registries.

Pair this skill with [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md)
before editing to build an impact map, and with
[`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) after to pick the
right validation set. If the change is driven by a launcher need, also read
[`switchyard-coding-agent-launchers`](../switchyard-coding-agent-launchers/SKILL.md).

## Quick Reference

| I want to add or change… | Start here |
|---|---|
| A new profile | Add a typed config + runtime under `switchyard/lib/profiles/`. The config should expose `build()` and return a profile runtime. Wrap with `ProfileSwitchyard` only when the existing Python HTTP endpoint contract needs `.call(...)`. |
| A new request-side behavior | Prefer making it profile-local. If it still needs to be reusable before the backend call, implement a plain component with async `process(ctx, request)` and compose it inside the owning profile config. |
| A new response-side behavior | Prefer making it profile-local. If it still needs to be reusable after the backend call, implement a plain component with async `process(ctx, response)` and compose it inside the owning profile config. |
| A new backend | Subclass the Rust-owned `LLMBackend` from `switchyard/lib/roles.py`; place it under `switchyard/lib/backends/`. Declare `supported_request_types` so translation can normalize. Compose it from the profile config that owns the behavior. |
| An OpenAI-compatible provider target such as NVIDIA Inference Hub or OpenRouter | Use the existing OpenAI-compatible backend/profile with `base_url`, `api_key`, and model id wiring. Add a new backend only when the provider has a real wire-format, auth, retry, or health contract that cannot fit that path. |
| Direct Rust component bindings | Add concrete PyO3 classes under `crates/switchyard-py/src/component_bindings/`, keep config bindings near the component binding that consumes them, and expose them lazily from `switchyard_rust/components.py`. Do not keep growing `core_bindings.rs` or `switchyard_rust/core.py` with concrete component classes. |
| Route YAML / model dispatch | Use `switchyard/cli/route_bundle.py` and `switchyard/lib/route_table_builders.py`. They build `RouteTable` entries from profile-backed runtimes and keep launchers plus `switchyard serve --routing-profiles` on one path. |
| Shared/persistent session-affinity pins across workers or pod churn | Configure the latency route with `session_affinity: true` + `affinity_store: redis` + `affinity_store_url` (optional `affinity_store_ttl_seconds`, `affinity_key_prefix`). `SessionAffinity` keeps the Rust `SessionCache` as L1 and reads/writes through the `AffinityPinStore` L2 (`switchyard/lib/redis_pin_store.py`), fail-open behind a 0.1s socket timeout and a 3-failure/10s-cooldown circuit breaker (`switchyard_affinity_l2_breaker_open` gauge). Requires the `switchyard[affinity-redis]` extra. |
| Stats / telemetry | Reuse `StatsRequestProcessor`, `StatsResponseProcessor`, `StatsLlmBackend`, and `StatsAccumulator`. A profile config should thread one accumulator through all three when stats are enabled. Do not write a parallel collector. |
| A fixed-path endpoint contributed by per-route components | Set `Endpoint.register_once = True`; `build_switchyard_app(...)` mounts the first instance while still running every component's lifecycle. Leave the default `False` for configurable endpoint classes that may mount distinct instances. |
| Per-endpoint attribution on `/metrics` for a Python backend that can't be wrapped by `StatsLlmBackend` | Set `ctx.selected_model = endpoint_id` before returning the response. Also set `ctx.backend_call_latency_ms = upstream_call_ms` so the response processor can compute routing overhead. `LatencyServiceLLMBackend.call` is the reference. |
| State metrics on `/metrics` | Register a `PrometheusEmitter` via `switchyard.lib.endpoints.prometheus_emitter.register(...)` and unregister on `shutdown()`. This is for backend-owned state, not request-flow counters. |
| Error-rate / retry-recovery counters | Use `switchyard.lib.endpoints.outcome_metrics`. FastAPI middleware records client outcomes. A retrying Python backend records each upstream attempt itself (and `record_retry_recovered()`) and sets `CTX_UPSTREAM_ATTEMPTS_RECORDED` so the endpoint skips its fallback — `LatencyServiceLLMBackend.call` is the reference. Single-attempt backends (Rust native / passthrough / multi) record nothing themselves; the endpoint fallback (`record_upstream_attempt_success` / `record_upstream_attempt_failure` in `upstream_error.py`, called from `dispatch_chat_request` and `handle_chain_exception`) counts their one attempt. Don't add a `model` label — these counters are layer-aggregate. Keep labels bounded. For a per-model error breakdown, emit a backend-owned counter via the `PrometheusEmitter` instead (labels bounded to config-derived ids: the route id `config.route_model` + endpoint ids, else the `other` sentinel) — `LatencyServiceLLMBackend._render_prometheus_lines` exposes `switchyard_latency_upstream_attempts_total{requested_model,upstream_model,outcome,code}` next to the aggregate, leaving the shared counter model-free. |
| Per-event error log | Use `switchyard.lib.endpoints.upstream_error_log.log_upstream_attempt_failure(...)` on the failure path. Events belong in logs/traces, not Prometheus sample timestamps. |
| CLI launcher integration | Build one profile-backed `SwitchyardApp` with `build_tier_passthrough_switchyard(...)` for single-target mode, or merge route YAML with `load_route_bundle_table(...)`. Hand the result to `build_switchyard_app`. |
| A new preset | Put preset helpers beside the profile config they produce, under `switchyard/lib/profiles/`. Presets should return typed config objects, not runnable chains. |
| A backend that pairs the executor with a stronger advisor model | `AdvisorProfileConfig` (`switchyard/lib/profiles/advisor.py`) dispatches on `AdvisorConfig.strategy`. **`tool_call`** (default) builds `AdvisorToolCallBackend` (`switchyard/lib/backends/advisor_tool_call_backend.py`): the proxy-side re-creation of Anthropic's `advisor_20260301` server tool — a real, **parameterless** `advisor` tool is appended to the client's tools (plus doc-verbatim executor steering and a length line injected cache-stably: system prepend + **first** user message); each advisor `tool_use` is intercepted before it reaches the client, the advisor is consulted on the transcript (tools summary + tail-kept conversation + the executor's current-turn text), and the advice loops back as a `tool_result` until the turn is advisor-free (`max_uses` consults per request, then `max_uses exceeded` error results; hard cap 8 turns; mixed advisor+client-tool turns are regenerated with siblings dropped). **`review_gate`** builds `AdvisorLoopBackend` (`switchyard/lib/backends/advisor_loop_backend.py`): no tool injected; at the executor's first **no-tool-call** turn (a plan, or "done") the advisor reviews **once per session** (hash of the conversation prefix, in-process) → `APPROVE` (return as-is) or `REDO` (feed the plan back and re-invoke) — near-superset of solo; front-loaded advice was found to cause premature convergence on Opus executors (see its docstring). Compose via a `type: advisor` route (`cli/route_bundle.py`); preset `AdvisorPresets.opus47_exec_opus48_advisor(strategy=...)`. Both tiers are **format-dispatched on `LlmTarget.format` and mix freely**: `anthropic` executors delegate **verbatim** to `AnthropicNativeBackend` (`/v1/messages`, Bearer — the client's `cache_control` prompt caching survives), `openai` executors (Qwen/DeepSeek/vLLM/NIM/OpenAI) delegate verbatim to `OpenAiNativeBackend` (`/chat/completions`) with the loop run on OpenAI-Chat wire via the private `_AnthropicDialect`/`_OpenAiDialect` objects in `advisor_tool_call_backend.py`; the advisor caller likewise dispatches on `advisor.format` (`_AnthropicAdvisorCaller` httpx `/v1/messages`, or `_OpenAiAdvisorCaller` via `OpenAILLMClient` `/chat/completions`). `responses` targets are rejected at `AdvisorConfig` validation (the loop is Chat-shaped). Both strategies are wire-generic: `review_gate` also dispatches on `executor.format` (its REDO feedback is plain-string assistant/user turns, valid on both wires, with a config-tunable `redo_feedback_prefix`), and because its trigger is proxy-side it fires regardless of the executor's tool-use discipline — prefer it over `tool_call` for weak executors that rarely call tools. The gate's trigger is selectable: `gate_trigger: no_tool_call` (default; function-calling harnesses) or `gate_trigger: pattern` + `gate_trigger_pattern` regex (text-protocol harnesses, e.g. terminus's `task_complete: true` completion marker — reviews the done-claim before the client sees it). Like `LatencyServiceLLMBackend`, both are Python multi-call backends doing their **own** stats accounting (`ctx.selected_model`, plus advisor consults — and, for tool_call, the intercepted executor turns — into the planner bucket) — do **not** wrap in `StatsLlmBackend`; `with_runtime_components` attaches the accumulator through the `_stats` hook. Prompts in `switchyard/lib/profiles/advisor_prompts.py`. |

## Profile Pattern

Profile-owned construction keeps the full behavior in one reviewable module:

1. Define a typed config with validation close to the profile.
2. Implement `build()` on the profile config.
3. Construct request-side helpers, backend, response-side helpers, and shared
   stats accumulator inside `build()`.
4. Return the profile runtime.
5. Use `ProfileSwitchyard(config.build())` only when serving through the
   existing Python endpoint contract.

Random routing, deterministic routing, plan-execute, passthrough, no-op,
stage-router, latency-service, RouteLLM, and OSS-router profiles are the reference
set under `switchyard/lib/profiles/`.

## Anti-Patterns

| Anti-pattern | What to do instead |
|---|---|
| Reintroducing factories, recipes, middleware bundles, request/response pipeline wrappers, or a resource cache to build a chain. | Put construction in the owning profile config. Use `RouteTable` only for model-id dispatch across already-built profile-backed runtimes. |
| Writing a new stats collector beside the existing stats stack. | Thread one `StatsAccumulator` through `StatsRequestProcessor`, `StatsLlmBackend`, and `StatsResponseProcessor`. |
| A CLI launcher assembling divergent chains inline. | Use `build_tier_passthrough_switchyard(...)` for one target or `load_route_bundle_table(...)` for route YAML. |
| A new role-shaped abstraction invented outside the backend role. | Keep pre-call logic in request-side profile code or plain request components, call logic in `LLMBackend`, post-call logic in response-side profile code or plain response components, and final wire conversion in `TranslationEngine`. |
| Adding an OpenRouter-specific backend or translator just to point at `https://openrouter.ai/api/v1`. | Route it through the OpenAI-compatible backend/profile and provider configuration. Keep provider-specific code for actual protocol differences. |
| Making route YAML declare arbitrary Python processors. | Route YAML is deployment config for supported profile route types. Runtime-only hooks such as intake are injected by the caller through the route-table builder kwargs. |

## When Something Genuinely Doesn't Fit

Ask before changing Rust-owned backend role classes, public API exports, HTTP endpoints,
or dependencies. Bring the smallest concrete delta you can describe. If the
change needs reusable request-side or response-side behavior, first prove why a
profile-local helper is insufficient.

## Related Skills

- [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) — run before any change here to map importers, profiles, and tests.
- [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — run after to pick the smallest local validation set that mirrors CI for the surface you touched.

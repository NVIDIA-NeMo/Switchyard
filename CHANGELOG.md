# Changelog

All notable changes to Switchyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Deprecated Python server stack** — `switchyard serve`, YAML route bundles,
  the FastAPI endpoints and legacy chain, the `switchyard-components` crate,
  and their compatibility PyO3 bindings are removed. Use `switchyard-server`
  with native TOML deployments, or `switchyard launch` for coding agents.

## [0.2.0]

Switchyard 0.2.0 introduces the native Rust server and libsy library path,
with explicit TOML deployments, provider-neutral routing algorithms, and
production-facing observability.

### Added

- **Standalone Rust server** — `switchyard-server` serves OpenAI Chat
  Completions, OpenAI Responses, and Anthropic Messages from one explicit TOML
  deployment. It includes TLS, graceful shutdown, upstream retries, token
  counting, health and model discovery, and optional durable session routing
  logs.
- **Rust library and protocol crates** — `switchyard-libsy` provides composable
  multi-LLM algorithms, `switchyard-protocol` owns the provider-neutral request
  and response contracts, `switchyard-translation` handles wire-format
  conversion, and `switchyard-llm-client` provides translated HTTP model calls.
- **Native routing algorithms** — weighted and reproducible random routing,
  capability, escalation, and custom-schema modes for LLM-classifier routing,
  multi-target policy selection, session affinity, context-window fallback, and
  signal-driven stage routing with handoff notes, per-target prompts, and an
  optional classifier fallback.
- **Python bindings for the native path** — `switchyard.libsy` runs Rust-owned
  algorithms with Python LLM clients, while `switchyard_rust.server.Server`
  hosts the Rust server in-process for the coding-agent launchers.
- **Native observability** — Prometheus metrics, GenAI OpenTelemetry spans,
  structured request logs, `/v1/stats`, `/v1/stats/reset`, and optional
  `/v1/routing/session-stats` expose request, routing, latency, token, cache,
  retry, and error data.
- **Evaluation and integration support** — native-server benchmark wiring,
  Terminal-Bench 2.1 dataset support, retry-adjusted task routing statistics,
  and an experimental LiteLLM stage-router integration.

### Changed

- **Native TOML is the primary deployment format** — LLM clients, targets, and
  routes are declared explicitly and validated by `switchyard-server`. The
  launcher path accepts the same TOML schema and includes a packaged OpenRouter
  deployment for zero-config startup.
- **Serving is built around libsy algorithms** — the native server and Python
  native-server binding construct algorithms directly instead of using the
  legacy profile and components-v2 serving stack. The Python YAML server keeps
  its existing profile APIs in this release.
- **Coding-agent launchers host the native Rust server** and use its routes,
  statistics, translation, and OpenTelemetry paths instead of constructing the
  legacy Python routing stack.
- **Cascade routing is now stage routing** — the `cascade` route and Python API
  names are replaced by `stage_router` and the native `StageRouter` algorithm.
- **The CLI is focused on serving and launching** — `switchyard serve` remains
  for Python routing-profile YAML bundles, while `switchyard launch` starts
  Claude Code, Codex CLI, or OpenClaw against a selected native route.
- **Python dependency compatibility is broader** — the supported OpenAI SDK
  floor moves from 2.34 to 2.7 while retaining the `<3.0` upper bound.
- **The Rust workspace uses Rust 1.96.1 and edition 2024.**

### Deprecated

- **The Python `switchyard serve` path** — the Python server, YAML route
  bundles, and profile APIs remain available in 0.2.0 for transition purposes
  but are deprecated. New deployments should use `switchyard-server`, native
  TOML configuration, and libsy algorithms.

### Fixed

- **Response `model` now names the model that actually served the request**, on
  every serving path and wire format. Streamed Anthropic and Responses replies,
  and every libsy-served reply, previously echoed the model id the client
  requested — for a route bundle whose key is an alias, that meant the alias
  rather than the routed target, so trajectories, dashboards, and client UIs
  labelled routed turns with the route name. The routed model was already
  reported by `x-model-router-selected-model`, `x-switchyard-selected-model`,
  `/v1/routing/stats`, and Intake's `served_model`; the response body now agrees
  with them. Streamed OpenAI Chat replies report the routed target instead of
  the provider's own id, and no longer fall back to `"unknown"` when a provider
  omits `model` on delta chunks.
- **Buffered Responses output is preserved** rather than dropping final answer
  items when translating a non-streaming response.
- **Cross-format response fidelity is improved** — Responses tool turns and
  reasoning items survive translation, raw stream events remain available, and
  Responses usage details and max-token truncation are represented correctly.
- **Known request fields are validated before translation**, so malformed
  OpenAI and Anthropic inputs return client errors instead of being silently
  coerced or omitted.
- **Anthropic interoperability is hardened** — Messages endpoints return
  Anthropic error envelopes, accept the `done` stream terminator, filter
  incompatible beta headers and OpenAI-only fields, and omit unsigned thinking
  blocks that Anthropic-compatible upstreams reject.
- **Prompt-cache usage survives format translation**, including cached and
  cache-creation token counts from OpenRouter and Anthropic-compatible
  providers; Anthropic prompt caching is enabled by default for translated
  calls.
- **Streaming stops after in-band upstream errors** instead of forwarding
  trailing events after the error.
- **Routing state and prompts remain coherent across turns** — target prompts
  and handoff notes survive same-format calls, classifier history keeps tool
  calls paired with their results, inactive session state is evicted, and
  context-overflow history is isolated by session and agent.
- **Native server model metadata is more reliable** — duplicate upstream model
  IDs produce a warning, `/v1/models` reports declared capabilities and Codex
  metadata, and streamed replies no longer fall back to an unknown model ID.

### Removed

- **Legacy Rust compatibility stacks** — the `switchyard-components-v2` and
  `switchyard-core` crates, the components-v2 profile macros, and the old PyO3
  profile and core bindings are removed. Native serving uses libsy; the Python
  profile APIs remain available in 0.2.0.
- **Legacy routing integrations** — plan-and-execute routing, RouteLLM, and the
  external OSS-router plugin path are removed. The `gpu` optional dependency
  extra is also gone with RouteLLM.
- **Latency-aware router** — the `latency_service` route type and its
  `LatencyServiceLLMBackend`, `LatencyServiceBackendConfig`,
  `LatencyServiceEndpoint`, and `LatencyServiceProfileConfig` public API are
  removed. It depended on NVIDIA Inference Hub's latency endpoint and schema.
  Deployments that need multi-endpoint, load- or latency-aware routing should
  move endpoint selection to a dedicated upstream load balancer.
- **Public `type: noop` and `type: passthrough` YAML routes** — removed from
  Python routing-profile bundles. Use an explicit `type: model` route for a
  direct target. Automatic catalog discovery from a bare `type: passthrough`
  route is also removed; list each model ID as its own `type: model` route.
- **Legacy Intake sink** — direct Intake request and response processors,
  launcher flags, and the `intake` optional dependency extra are removed. The
  native server exports telemetry through OpenTelemetry and OTLP instead.
- **Legacy CLI setup and diagnostics** — `switchyard configure`, `verify`, and
  `status`, the interactive setup and model-picker TUI, saved provider settings,
  and launcher smoke mode are removed when the CLI is narrowed to `serve` and
  `launch`. Name the credential environment variable with `api_key_env` in a
  native TOML deployment, export it, and pass the deployment to each
  `switchyard launch`. Validate a deployment with
  `switchyard-server --config <deployment.toml> --dry-run`.

### Known Issues

1. Buffered upstream work continues after the client disconnects, so a
   cancelled request can still incur provider cost.
2. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for
   LLM-classifier judge failures that route to the default target, escalation
   decisions, and `stage_router` fallback decisions.
3. The retry recovery counter stays at zero after a successful upstream retry.
4. `x-switchyard-session-id` is not recorded in native session stats.
5. The native server does not send the documented `X-Switchyard-Version` header
   upstream.

## [0.1.0] — Initial release

First public release of Switchyard — a typed, composable control plane for LLM
traffic that sits between client applications and LLM backends.

### Added

- **Four-role chain** — `RequestProcessor → LLMBackend → ResponseProcessor →
  TranslationEngine`, executed by the Rust-backed core. See
  the [0.1.0 architecture](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.1.0/docs/architecture.md).
- **Protocol translation** — convert between OpenAI Chat Completions, Anthropic
  Messages, and OpenAI Responses wire formats, so each client keeps speaking its
  native API regardless of the upstream backend.
- **YAML route bundles** (`switchyard serve --routing-profiles`) — one bundle,
  many named routes, each its own chain. Supported route `type`s: `model`,
  `passthrough`, `random_routing`, `cascade`, `deterministic`
  (LLM-as-classifier), `latency_service`, and `noop`.
- **Routing strategies** — weighted random split, signal-driven **cascade**
  escalation (see the [0.1.0 cascade documentation](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.1.0/docs/routing_algorithms/cascade_routing.md)),
  LLM-as-classifier strong/weak routing, and latency-aware multi-endpoint
  failover.
- **One-command launchers** — `switchyard launch claude`, `launch codex`, and
  `launch openclaw` spin up a local proxy and drop you into the target CLI.
  All three **default to LLM-as-classifier routing** (validated coding-agent
  trio) with `--model` / `--routing-profiles` to opt out.
- **CLI** — `serve`, `launch`, `configure` (saved defaults, `--show`,
  `--list-models`), and `verify` / `launch --smoke` round-trip checks.
- **Observability** — Prometheus `/metrics`, a JSON `/v1/stats`
  (`/v1/routing/stats` alias), and per-request cost/token/latency stats. See
  [Metrics Reference](docs/METRICS_REFERENCE.md).
- **Python library** — `SwitchyardRecipes` (`passthrough_recipe`,
  `random_routing_recipe`, `cascade_recipe`, `deterministic_routing_recipe`,
  …) and typed `ChatRequest` / `ChatResponse` containers for in-process use.
- **Rust core** (PyO3) — chain execution, the latency-aware router, and the
  tool-result signal collector are implemented in Rust and re-exported to
  Python.
- **Packaging** — `pip install nemo-switchyard` with optional extras `[server]`,
  `[cli]`, `[gpu]`, `[all]`. See [Installation](INSTALLATION.md).

### Deprecated

- **`--plan-execute` launcher flag** — slated for removal; plan-execute will be
  configured through a `--routing-profiles` YAML bundle instead.

### Notes

- The `--deterministic` launcher flag was removed during pre-release
  development — LLM-as-classifier routing is now the implicit default for the
  `claude` / `codex` / `openclaw` launchers.
- Inference Hub integration docs are out of scope for this release.

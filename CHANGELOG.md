# Changelog

All notable changes to Switchyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-08-10

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
  capability and escalation modes for LLM-classifier routing, session affinity,
  context-window fallback, and signal-driven stage routing with handoff notes,
  per-tier prompts, and an optional classifier fallback.
- **Native observability** — Prometheus metrics, GenAI OpenTelemetry spans,
  structured request logs, `/v1/stats`, `/v1/stats/reset`, and optional
  `/v1/routing/session-stats` expose request, routing, latency, token, cache,
  retry, and error data.

### Changed

- **Native TOML is the primary deployment format** — LLM clients, targets, and
  routes are declared explicitly and validated by `switchyard-server`. The
  launcher path accepts the same TOML schema and includes a packaged OpenRouter
  deployment for zero-config startup.
- **Serving is built around libsy algorithms** — the native server and Python
  server binding construct algorithms directly instead of using the legacy
  profile and components-v2 serving stack.
- **The CLI is focused on serving and launching** — `switchyard serve` remains
  for the minimal Python route bundle, while `switchyard launch` starts Claude
  Code, Codex CLI, or OpenClaw against a selected route.
- **The Rust workspace uses Rust 1.96.1 and edition 2024.**

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
- **Known request fields are validated before translation**, so malformed
  OpenAI and Anthropic inputs return client errors instead of being silently
  coerced or omitted.
- **Prompt-cache usage survives format translation**, including cached and
  cache-creation token counts from OpenRouter and Anthropic-compatible
  providers.
- **Streaming stops after in-band upstream errors** instead of forwarding
  trailing events after the error.
- **Context-overflow history is isolated by session and agent**, preventing one
  task's overflow recovery from changing another task's routing state.

### Removed

- **Latency-aware router** — the `latency_service` route type and its
  `LatencyServiceLLMBackend`, `LatencyServiceBackendConfig`,
  `LatencyServiceEndpoint`, and `LatencyServiceProfileConfig` public API are
  removed. It depended on NVIDIA Inference Hub's latency endpoint and schema.
  Deployments that need multi-endpoint, load- or latency-aware routing should
  migrate to [Dynamo](https://github.com/ai-dynamo/dynamo) (backend-load /
  KV-cache-aware routing with request failover) or an external load balancer
  such as [Traefik](https://doc.traefik.io/traefik/reference/routing-configuration/http/load-balancing/service/)
  or HAProxy.
- **Public `type: noop` and `type: passthrough` route types** — removed from
  route bundles. Use `type: model` to register a single explicit model target.
  Catalog auto-discovery via a bare `type: passthrough` route is gone; there is
  no `type: model` equivalent, so list the model ids you want as explicit
  `type: model` routes.
- **`switchyard configure` and `switchyard verify` CLI commands** — removed when
  the CLI was narrowed to `serve` and `launch`. Switchyard no longer saves
  provider credentials or deployment paths. Name the credential environment
  variable with `api_key_env` in a native TOML deployment, export it, and pass
  the deployment to each `switchyard launch`. Validate a deployment with
  `./target/release/switchyard-server --config <deployment.toml> --dry-run`.

### Known Issues

1. `stage_router` drops a configured tier system prompt when the inbound
   request and the selected target both use `openai_chat`. The call succeeds and
   no warning is emitted.
2. Errors returned from `/v1/messages` use the OpenAI error envelope rather than
   Anthropic's `{"type": "error", ...}` shape, so Anthropic SDK clients cannot
   dispatch on `error.type`.
3. Session state is retained without a capacity limit or eviction, so memory
   grows with the number of sessions a process has served.
4. Buffered upstream work continues after the client disconnects, so a
   cancelled request can still incur provider cost.
5. Routing-tier attribution is missing from `GET /v1/stats` and `/metrics` for
   fail-open, escalation, and `stage_router` fallback decisions.
6. The retry recovery counter stays at zero after a successful upstream retry.
7. `x-switchyard-session-id` is not recorded in native session stats.
8. The native server does not send the documented `X-Switchyard-Version` header
   upstream.
9. LLM-classifier history trimming can separate a tool result from the tool
   call it belongs to when `recent_turn_window` is configured.

## [0.1.0] — Initial release

First public release of Switchyard — a typed, composable control plane for LLM
traffic that sits between client applications and LLM backends.

### Added

- **Four-role chain** — `RequestProcessor → LLMBackend → ResponseProcessor →
  TranslationEngine`, executed by the Rust-backed core. See
  [Architecture](docs/ARCHITECTURE.md).
- **Protocol translation** — convert between OpenAI Chat Completions, Anthropic
  Messages, and OpenAI Responses wire formats, so each client keeps speaking its
  native API regardless of the upstream backend.
- **YAML route bundles** (`switchyard serve --routing-profiles`) — one bundle,
  many named routes, each its own chain. Supported route `type`s: `model`,
  `passthrough`, `random_routing`, `stage_router`, `deterministic`
  (LLM-as-classifier), `latency_service`, and `noop`.
- **Routing strategies** — weighted random split, signal-driven **stage-router**
  escalation (see [Stage-Router Routing](docs/stage_router_routing.md)),
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
- **Python library** — `ProfileSwitchyard` driven by typed profile configs
  (`PassthroughProfileConfig`, `RandomRoutingProfileConfig`,
  `StageRouterProfileConfig`, …) and typed `ChatRequest` / `ChatResponse`
  containers for in-process use.
- **Rust core** (PyO3) — chain execution, the latency-aware router, and the
  tool-result signal collector are implemented in Rust and re-exported to
  Python.
- **Packaging** — `pip install nemo-switchyard` with optional extras `[server]`,
  `[cli]`, `[tracing]`, `[intake]`, `[affinity-redis]`, `[all]`. See
  [Installation](INSTALLATION.md).

### Notes

- The `--deterministic` launcher flag was removed during pre-release
  development — LLM-as-classifier routing is now the implicit default for the
  `claude` / `codex` / `openclaw` launchers.
- Inference Hub integration docs are out of scope for this release.

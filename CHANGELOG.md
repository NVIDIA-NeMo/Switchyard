# Changelog

All notable changes to Switchyard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Initial release

First public release of Switchyard — a typed, composable control plane for LLM
traffic that sits between client applications and LLM backends.

### Added

- **Four-role chain** — `RequestProcessor → LLMBackend → ResponseProcessor →
  TranslationEngine`, executed by the Rust-backed core. See
  [Architecture](docs/architecture.mdx).
- **Protocol translation** — convert between OpenAI Chat Completions, Anthropic
  Messages, and OpenAI Responses wire formats, so each client keeps speaking its
  native API regardless of the upstream backend.
- **YAML route bundles** (`switchyard serve --routing-profiles`) — one bundle,
  many named routes, each its own chain. Supported route `type`s: `model`,
  `passthrough`, `random_routing`, `stage_router`, `deterministic`
  (LLM-as-classifier), `latency_service`, and `noop`.
- **Routing strategies** — weighted random split, signal-driven **stage-router**
  escalation (see [Stage-Router Routing](docs/routing_algorithms/stage_router_routing.mdx)),
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
  [Metrics Reference](docs/internal/metrics_reference.md).
- **Python library** — `SwitchyardRecipes` (`passthrough_recipe`,
  `random_routing_recipe`, `stage_router_recipe`, `deterministic_routing_recipe`,
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

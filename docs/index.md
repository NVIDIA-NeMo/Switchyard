# Switchyard

Switchyard routes and translates LLM traffic for coding agents and API clients.
It supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.

## Choose a Path

| Goal | Path | Start here |
|---|---|---|
| Run Claude Code, Codex, or OpenClaw through Switchyard | Launcher Path | [Install and launch an agent](getting_started.md#launcher-path) |
| Run Switchyard as a standalone proxy for API clients | Server Path | [Build and run the Rust server](getting_started.md#server-path) |
| Add Switchyard routing to a Rust application | Library Path | [`switchyard-libsy`](../crates/libsy/README.md) |

The Launcher Path installs the `switchyard` CLI and hosts the native Rust server
through its packaged PyO3 binding. The Server Path builds and runs the
standalone `switchyard-server` binary.

## Explore

- [Core Concepts](core_concepts.md): learn the LLM client, target, and route layers
- [Routing Algorithms](routing_algorithms/overview.md): choose how requests select a model
- [Architecture](architecture.md): understand the proxy and library components
- [CLI Reference](cli_reference.md): inspect supported launcher commands and options
- [Context-Window Handling](operations/context_window.md): configure eviction and retry behavior

## Reference

- [`switchyard-server`](../crates/switchyard-server/README.md): server configuration, endpoints, and metrics
- [`switchyard-libsy`](../crates/libsy/README.md): embeddable routing algorithms
- [`switchyard-protocol`](../crates/protocol/README.md): provider-neutral API types
- [`switchyard-translation`](../crates/switchyard-translation/README.md): protocol translation

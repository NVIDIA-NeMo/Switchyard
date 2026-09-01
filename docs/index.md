# Switchyard

Switchyard routes and translates LLM traffic for coding agents and API clients.
It supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.

## Choose a Path

| Goal | Path | Start here |
|---|---|---|
| Run Switchyard as a standalone proxy for API clients | Server Path | [Build and run the Rust server](getting_started.md#server-path) |
| Add Switchyard routing to a Rust application | Library Path | [`switchyard-libsy`](../crates/libsy/README.md) |
| Add Switchyard routing to NeMo Relay | Native Plugin Path | [`switchyard-nemo-relay-plugin`](../crates/switchyard-nemo-relay-plugin/README.md) |

The Server Path builds and runs the standalone `switchyard-server` binary.

## Explore

- [Core Concepts](core_concepts.md): learn the LLM client, target, and route layers
- [Routing Algorithms](routing_algorithms/overview.md): choose how requests select a model
- [Architecture](architecture.md): understand the proxy and library components
- [Server CLI Reference](cli_reference.md): inspect `switchyard-server` commands and options
- [Rust API](reference/rust_api.md): browse libsy and protocol crate documentation
- [Context-Window Handling](operations/context_window.md): configure eviction and retry behavior

## Reference

- [`switchyard-server`](../crates/switchyard-server/README.md): server configuration, endpoints, and metrics
- [`switchyard-libsy`](reference/rust_api.md#switchyard-libsy): embeddable routing algorithms
- [`switchyard-protocol`](reference/rust_api.md#switchyard-protocol): provider-neutral API types
- [`switchyard-translation`](../crates/switchyard-translation/README.md): protocol translation
- [`switchyard-nemo-relay-plugin`](../crates/switchyard-nemo-relay-plugin/README.md): native NeMo Relay integration

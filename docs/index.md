# Switchyard

Switchyard helps an AI agent choose which model handles each request. Embed its Rust core, **libsy**, in your gateway or agent harness. Your application keeps its model clients, credentials, and service.

**[Start with the Rust quickstart](getting_started.md)**: add the crates, run a routing example, and send the selected request through your model client.

## Choose a Path

| Goal | Path | Start here |
|---|---|---|
| Add routing to your Rust gateway or application | Library Path | [Rust quickstart](getting_started.md) |
| Use libsy from Python | Python Bindings | [Python embedding](reference/host_integration.md#python-bindings) |
| Evaluate routing locally with your own models | Local Evaluation | [Run the demo server](operations/local_evaluation.md) |
| Route Codex or Claude Code using your existing login | Server Path | [Single-provider coding agents](recipes/single_provider_coding_agents.md) |
| Add Switchyard routing to NeMo Relay | Native Plugin Path | [Use Switchyard with NeMo Relay](integrations/nemo_relay.md) |
| Point the pi coding agent at the standalone proxy | Server Path | [Use Switchyard with pi](integrations/pi.md) |
| Point Oh My Pi at the standalone proxy | Server Path | [Use Switchyard with Oh My Pi](integrations/oh_my_pi.md) |

The standalone server is for demos, evaluation, and personal use. It is not a supported production deployment path. Its HTTP API supports OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.

## Explore

- [Benchmarks](benchmarks.md): compare Terminal-Bench 2.1 and DeepSWE results and open their routing configs
- [Core Concepts](core_concepts.md): learn the LLM client, target, and route layers
- [Routing Algorithms](routing_algorithms/overview.md): choose how requests select a model
- [Architecture](architecture.md): understand the proxy and library components
- [Server CLI Reference](cli_reference.md): inspect `switchyard-server` commands and options
- [Rust API](reference/rust_api.md): browse libsy, protocol, and translation crate documentation
- [Context-Window Handling](operations/context_window.md): configure eviction and retry behavior
- [Tuning](tuning/index.md): the loop from verifying routing to recording a result
- [Troubleshooting](operations/troubleshooting.md): symptoms, silent misconfigurations, and fixes

## Reference

- [`switchyard-server`](../crates/switchyard-server/README.md): server configuration, endpoints, and metrics
- [`switchyard-libsy`](reference/rust_api.md#switchyard-libsy): embeddable routing algorithms
- [`switchyard-protocol`](reference/rust_api.md#switchyard-protocol): provider-neutral API types
- [`switchyard-translation`](reference/rust_api.md#switchyard-translation): protocol translation
- [`switchyard-nemo-relay-plugin`](../crates/switchyard-nemo-relay-plugin/README.md): native plugin build and configuration
- [Dynamo / agentgateway](../examples/dynamo-preproc/README.md): Switchyard PreProc integration example

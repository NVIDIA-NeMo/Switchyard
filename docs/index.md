# Switchyard

Switchyard helps an AI agent choose which model handles each request. Embed its Rust core, **libsy**, in your gateway or agent harness. Your application keeps its model clients, credentials, and service.

Choose where routing belongs in your application. Use an existing integration,
or embed libsy and keep your current model clients.

## Choose a Path

| Goal | Start here | What you own |
|---|---|---|
| Use routing through an inference provider | [Switchyard on OpenRouter](https://openrouter.ai/docs/guides/routing/routers/switchyard-router) | Your API requests and provider account. Follow the provider's available algorithms and controls. |
| Add routing to your gateway | [NeMo Relay](integrations/nemo_relay.md) or the experimental [LiteLLM integration](../examples/litellm/README.md) | Gateway deployment, credentials, and operation. Check each integration's limits. |
| Embed routing in a gateway, harness, or application | [Rust quickstart](getting_started.md), then [Mechanics of Libsy](mechanics_of_libsy.md) | Model execution and application lifecycle. [Python bindings](reference/host_integration.md#python-bindings) use the same core. |

## Try With a Local Agent

The [reference server](operations/local_evaluation.md) gives you a quick way to
try routing with your own models. It demonstrates libsy, protocol conversion, and
configured execution. It is for local demos and personal use, not a supported
production deployment or a service to scale.

Use it with [Claude Code or Codex](recipes/single_provider_coding_agents.md),
[pi](integrations/pi.md), or [Oh My Pi](integrations/oh_my_pi.md).

## Explore

- [Benchmarks](benchmarks.md): compare Terminal-Bench 2.1 and DeepSWE results and open their routing configs
- [Mechanics of Libsy](mechanics_of_libsy.md): understand steps, calls, and outcomes
- [Switchyard Protocols](protocols/index.md): connect your model data to libsy
- [Routing Algorithms](routing_algorithms/overview.md): choose how requests select a model
- [Host Integration](reference/host_integration.md): connect your clients and application lifecycle
- [Rust API](reference/rust_api.md): browse the core library, protocols, translation, clients, and runner
- [Context-Window Handling](operations/context_window.md): configure eviction and retry behavior
- [Tuning](tuning/index.md): the loop from verifying routing to recording a result
- [Troubleshooting](operations/troubleshooting.md): symptoms, silent misconfigurations, and fixes

## Reference

- <a href="rust/switchyard_libsy/"><code>switchyard-libsy</code> API</a>: embeddable routing algorithms
- <a href="rust/switchyard_protocol/"><code>switchyard-protocol</code> API</a>: provider-neutral model data
- <a href="rust/switchyard_translation/"><code>switchyard-translation</code> API</a>: provider-format conversion
- <a href="rust/switchyard_llm_client/"><code>switchyard-llm-client</code> API</a>: optional HTTP clients and execution helpers
- <a href="rust/switchyard_runner/"><code>switchyard-runner</code> API</a>: configured route construction and execution
- [`switchyard-nemo-relay-plugin`](../crates/switchyard-nemo-relay-plugin/README.md): native plugin build and configuration
- [Dynamo / agentgateway](../examples/dynamo-preproc/README.md): Switchyard PreProc integration example
- [Component responsibilities](reference/rust_api.md#crate-boundary): choose the crates your integration needs
- [Demo server CLI](cli_reference.md): local server commands and options

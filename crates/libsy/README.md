# switchyard-libsy

Provider-neutral orchestration for multi-LLM optimization. A libsy
[`Algorithm`] decides which model targets to call, in what order, and how to
combine their results. It can use target-owned clients or hand each call back to
the host, allowing it to embed in proxies, gateways, and agent runtimes without
owning an HTTP stack.

## Setup

```toml
[dependencies]
async-trait = "0.1"
futures = "0.3"
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
tokio = { version = "1", features = ["macros", "rt"] }
```

## Quick start

The runnable examples cover both ways to integrate libsy:

```bash
cargo run -p switchyard-libsy --example research_agent
cargo run -p switchyard-libsy --example research_agent_core
```

[`research_agent.rs`](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent.rs)
uses target-owned clients with [`Algorithm::run`].
[`research_agent_core.rs`](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent_core.rs)
uses [`Algorithm::run_stream`] when the host owns model transport.

## Built-in algorithms

| Type | Purpose |
|---|---|
| [`Passthrough`] | Always call one configured target. |
| [`Random`] | Select among any number of targets using uniform or weighted routing. |
| [`LlmTaskClassifier`] | Ask a judge model to choose an efficient or capable target. |
| [`StageRouter`] | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |

[`Noop`] is a test helper, not a production routing algorithm.

## How it fits together

[`LlmTarget`] pairs a semantic routing name with an optional
[`RoutedLlmClient`](switchyard_protocol::RoutedLlmClient). An [`Algorithm`]
selects targets and records [`Decision`](switchyard_protocol::Decision)s. Use
[`Algorithm::run`] with target-owned clients, or [`Algorithm::run_stream`] when
the host owns model transport. The provider-neutral [`Request`], [`Response`],
[`Usage`], and [`LlmResponse`] contracts come from `switchyard-protocol`.

[`Request`]: switchyard_protocol::Request
[`Response`]: switchyard_protocol::Response
[`Usage`]: switchyard_protocol::Usage
[`LlmResponse`]: switchyard_protocol::LlmResponse

## Examples

- [Client-backed execution](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent.rs)
- [Host-owned model calls](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent_core.rs)
- [Streaming responses](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/streaming_agent.rs)
- [Custom algorithm](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/ensemble.rs)

## License

Licensed under the Apache License, Version 2.0.

# switchyard-libsy

Provider-neutral orchestration for multi-LLM optimization. A libsy
[`Algorithm`] decides which model targets to call, in what order, and how to
combine their results. It hands every call back to the host rather than making
it, allowing it to embed in proxies, gateways, and agent runtimes without owning
an HTTP stack.

## Setup

```toml
[dependencies]
async-trait = "0.1"
futures = "0.3"
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
tokio = { version = "1", features = ["macros", "rt"] }
```

## Built-in algorithms

| Type | Purpose |
|---|---|
| [`Passthrough`] | Always call one configured target. |
| [`Random`] | Select among any number of targets using uniform or weighted routing. |
| [`LlmTaskClassifier`] | Ask a judge model to choose an efficient or capable target. |
| [`StageRouter`] | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |

[`Noop`] is a test helper, not a production routing algorithm.

## How it fits together

A target is a bare model id naming a routing destination. An [`Algorithm`] selects targets and
records [`Decision`](switchyard_protocol::Decision)s, offloading every model call
to its caller: [`Algorithm::run_stream`] yields a [`Step`] stream whose
[`Step::CallModel`] items the host serves over its own transport. libsy makes no
network calls itself — `switchyard-llm-client`'s `run` is a ready-made consumer
that drives the stream and performs the calls over HTTP.

The provider-neutral [`Request`], [`Response`], [`Usage`], and [`LlmResponse`]
contracts come from `switchyard-protocol`.

[`Request`]: switchyard_protocol::Request
[`Response`]: switchyard_protocol::Response
[`Usage`]: switchyard_protocol::Usage
[`LlmResponse`]: switchyard_protocol::LlmResponse

## License

Licensed under the Apache License, Version 2.0.

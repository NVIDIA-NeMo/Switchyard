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
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git", tag = "v0.3.0" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git", tag = "v0.3.0" }
tokio = { version = "1", features = ["macros", "rt"] }
```

## Built-in algorithms

| Type | Purpose |
|---|---|
| [`Passthrough`] | Always select one configured target. |
| [`Random`] | Select among any number of targets using uniform or weighted routing. |
| [`LlmTaskClassifier`] | Ask a judge model to choose an efficient or capable target. |
| [`StageRouter`] | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |
| [`InfraRouter`] | Wrap a tier router and optionally move an efficient choice to the capable model using host-supplied observations. |

[`Noop`] is a test helper, not a production routing algorithm.

## How it fits together

A target is a bare model id naming a routing destination. An [`Algorithm`] may offload
routing-time classifier or judge calls to its caller: [`Algorithm::run_stream`] yields a [`Step`]
stream whose [`Step::CallModel`] items the host serves over its own transport. The stream ends
with [`Step::Done`] carrying a [`RoutingOutcome`]: the algorithm-selected model, ordered
fallbacks, rewritten request, and an optional response already produced while routing. libsy
makes no network calls itself — `switchyard-llm-client`'s `run` is a ready-made consumer that
drives the stream and performs the terminal answer call, retries, and fallback over HTTP.

The provider-neutral [`Request`], [`Response`], [`Usage`], and [`LlmResponse`]
contracts come from `switchyard-protocol`.

## Infrastructure observations

`InfraRouter` requests fresh signals after the inner algorithm chooses the efficient
model. The host receives `Call::Infra` (or `Step::CallInfra` when consuming the stream
directly) and fulfills it with `call.respond(work).await`. Return an empty map when
signals are unavailable. The policy keeps the task decision on missing or stale data
and on lookup timeout. It never selects a worker or reserves serving capacity.

`switchyard-llm-client::ClientRouter::with_infra_provider` accepts a
`switchyard_protocol::InfraProvider`. Runner hosts can pass a provider per request
with `Route::decide_with_infra`. The provider owns transport and preprocessing;
libsy only sees candidate observations keyed by model ID. These APIs are workspace
additions and are not available in the v0.3.0 release shown above.

Hosts matching all `Call` or `Step` variants must handle the new infrastructure
variant. The standard client returns unavailable observations if no provider is set.
Python and Node do not expose infrastructure-provider bindings yet.

## Concurrent calls and cancellation

An algorithm can poll several `Driver::call_model`, `Driver::call_decision`, or
`Driver::call_infra` futures
with `tokio::join!`, `tokio::select!`, or `FuturesUnordered`. Each call has its own
reply channel. Dropping a waiting future cancels only that call; the algorithm can
continue making other calls.

Hosts pass the work itself to `call.respond(work).await`. Awaiting the provider
before calling `respond` prevents libsy from cancelling that work. For a response
that is already available, use `call.respond(std::future::ready(result)).await`.
Model-call errors stop the host run unless `call.recover_errors` is enabled.
Decision-call errors are returned to the algorithm.

`drive` polls host handlers concurrently and drops outstanding handlers when the
algorithm finishes or the run is dropped. Custom `run_stream` consumers must also
poll handlers concurrently and drop them when the run ends. Separately spawned
tasks need explicit abort and cleanup; dropping a task handle does not stop its work.

Cancellation drops the supplied work future. Once a response stream is delivered,
its consumer controls its lifetime. Whether the provider stops generation depends
on the transport and provider.

[`Request`]: switchyard_protocol::Request
[`Response`]: switchyard_protocol::Response
[`Usage`]: switchyard_protocol::Usage
[`LlmResponse`]: switchyard_protocol::LlmResponse

## License

Licensed under the Apache License, Version 2.0.

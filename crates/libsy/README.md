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

## Queue-based infrastructure policy

`InfraPolicy` applies after a tier router finishes. It can move an efficient-model
choice to the capable model when every efficient worker has a queue and calibrated
service times predict a saving. Missing or stale worker observations keep the task
choice. The policy does not change task affinity or choose a worker.

The host periodically collects `switchyard_protocol::InfraSignals`, then calls
`Route::apply_infra` on the result of `Route::decide`. Hold the host's in-flight
accounting lock through the policy check and increment for the selected model.
Decrement when the response finishes or is cancelled. These are process-local routing
limits; they do not reserve backend capacity. Plain `decide` and `execute` keep their
task-routing behavior. No new `Call` or `Step` variant is required.

See the [Dynamo PreProc example](../../examples/dynamo-preproc/README.md#task-classification-and-queue-based-routing)
for collection, configuration and response-lifetime tracking. This policy is a
workspace addition and is not in the v0.3.0 release shown above.

## Concurrent calls and cancellation

An algorithm can poll several `Driver::call_model` or `Driver::call_decision` futures
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

# switchyard-libsy

`switchyard-libsy` is Switchyard's provider-neutral Rust library for routing and
multi-model orchestration. An algorithm decides which semantic target to call.
The target's `RoutedLlmClient` performs the provider I/O, or a host can drive the
algorithm's step stream and perform each call itself.

The package name is `switchyard-libsy`; Rust imports use `switchyard_libsy`.
Request, response, client, and metadata types come from `switchyard-protocol`.

## Add the crates

```toml
[dependencies]
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
```

Both crates must use compatible versions.

## Quick start

Create `src/main.rs` to construct a uniform random router over two semantic
targets:

```rust
use std::sync::Arc;

use switchyard_libsy::{Algorithm, LlmTarget, LlmTargetSet, Random};

fn main() -> switchyard_libsy::Result<()> {
    let target = |name: &str| LlmTarget {
        semantic_name: name.into(),
        llm_client: None,
    };
    let targets = LlmTargetSet::new(vec![target("fast"), target("strong")]);
    let router: Arc<dyn Algorithm> = Arc::new(Random::new(targets, None, None)?);
    println!("{}", router.name());
    Ok(())
}
```

Attach an `Arc<dyn RoutedLlmClient>` to each target before using `run`, or use
`run_stream` when the host owns model transport.

## Type ownership

`switchyard-protocol` owns the provider-neutral contract:

- `Context`, `Request`, `Response`, and `Metadata`
- `LlmRequest`, `AggLlmResponse`, and `LlmResponse`
- `Message`, `ContentBlock`, tools, usage, and streaming chunks
- `Decision`, `RoutedLlmClient`, and `LlmClientError`

`switchyard-libsy` owns orchestration:

- `Algorithm`, `Driver`, `Step`, and routed-call types
- `LlmTarget` and `LlmTargetSet`
- routing algorithms and their configuration
- routing errors, observations, tracing, and metrics

Import protocol types from `switchyard_protocol`; libsy does not re-export them.

## Targets and clients

An `LlmTarget` contains:

- `semantic_name`: the logical name selected by an algorithm
- `llm_client`: the default client used by `Algorithm::run`, or `None` when the
  host will fulfill calls through `run_stream`

A client maps the selected semantic name to the provider model and transport it
serves. See [`research_agent`](examples/research_agent.rs) for a client-backed
run.

## Running algorithms

Hold algorithms as `Arc<dyn Algorithm>` so one thread-safe instance can serve
concurrent requests.

```rust
let (decisions, response) = algorithm
    .clone()
    .run(Context::default(), request)
    .await?;
```

`run` serves every routed call through the target's default client. It fails
with `LibsyError::MissingClient` when a selected target has no client.

For a host-owned transport, use `run_stream(ctx, request, observer)`. Pass
`None` when model-call observations are not needed. Each `Step::CallLlm` must be
fulfilled exactly once with `CallLlmRequest::respond`; the terminal step is
`Step::ReturnToAgent`.

See [`research_agent_core`](examples/research_agent_core.rs) for a complete
host-driven loop.

Request, response, usage, preservation, and response-stream semantics belong to
[`switchyard-protocol`](../protocol/README.md).

## Included algorithms

| Algorithm | Purpose |
|---|---|
| `Passthrough` | Always call one target. |
| `Random` | Uniform or weighted selection across any number of targets. |
| `LlmTaskClassifier` | Ask a judge to choose an efficient or capable target. |
| `StageRouter` | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |

`Noop` is a test helper and should not be configured as a production route.

### Random routing

`Random::new(targets, weights, seed)` accepts relative nonnegative weights; they
do not need to sum to one. Missing weights default to one per target. A seed
makes the generated sequence reproducible for a given shared algorithm instance.

### Task classifier configuration

| Field | Default | Meaning |
|---|---|---|
| `base_threshold` | `0.0` | Minimum solve probability for the efficient target. Set this deliberately for production routing. |
| `min_confidence` | `0.0` | Minimum judge confidence for efficient routing. |
| `capability_elevated_floor` | `None` | Higher threshold for uncertain, unsupported, or unmatched tasks. |
| `session_affinity` | `false` | Reuse the first decision for later requests in the same session. |
| `message_hash_fallback` | `false` | Derive affinity from the first user message when session metadata is absent. Requires session affinity. |
| `recent_turn_window` | `None` | Judge only the newest user message. `Some(n)` includes instructions, the opening task, and the last `n` turns. |
| `max_output_tokens` | `4096` | Maximum tokens available to the judge verdict. |

Affinity state lasts for the process lifetime. Message-hash fallback is based on
content, so unrelated callers with identical opening messages can share an
assignment.

## Implementing an algorithm

Implement `Algorithm::name` and `Algorithm::create_run_task`. Use the supplied
`Driver` to call targets and publish decisions. `self: Arc<Self>` means one
algorithm instance runs concurrently and must own the synchronization for any
shared state.

The authoritative trait contract is in the generated `Algorithm` documentation.
The [`ensemble`](examples/ensemble.rs) example shows a custom stateful algorithm.

## Observability

libsy emits `tracing` spans and structured events plus OpenTelemetry metrics
through the global provider. Hosts install the subscriber, trace exporter, and
meter provider. Without them, instrumentation is a no-op.

The primary spans are `libsy.run`, `libsy.llm_call`, and `libsy.client_call`.
Metrics include run, call, decision, latency, error, and compatibility request
families. Call `switchyard_libsy::initialize_metrics()` after installing the
meter provider when zero-valued compatibility gauges must exist before the
first request.

## Reference

- [Generated Rust API reference](../../docs/reference/rust_api.md)
- [`research_agent`](examples/research_agent.rs): default-client execution
- [`research_agent_core`](examples/research_agent_core.rs): host-owned calls
- [`streaming_agent`](examples/streaming_agent.rs): response streaming
- [`ensemble`](examples/ensemble.rs): custom stateful algorithm

## License

Licensed under the Apache License, Version 2.0.

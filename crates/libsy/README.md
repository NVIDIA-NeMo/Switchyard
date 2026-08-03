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
switchyard-libsy = "0.2"
switchyard-protocol = "0.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Both crates must use compatible versions.

## Quick start

Create `src/main.rs` with a no-op algorithm that verifies integration without
making an upstream call:

```rust
use std::sync::Arc;

use switchyard_libsy::{Algorithm, Noop};
use switchyard_protocol::{Context, Request, text_request};

#[tokio::main]
async fn main() -> switchyard_libsy::Result<()> {
    let algorithm: Arc<dyn Algorithm> = Arc::new(Noop {});
    let request = Request {
        llm_request: text_request(Some("switchyard/noop".into()), "hello"),
        ..Request::default()
    };

    let (decisions, response) = algorithm.run(Context::default(), request).await?;
    for decision in decisions {
        println!("selected {}", decision.selected_model());
    }
    assert_eq!(response.selected_model(), Some("switchyard/noop"));
    Ok(())
}
```

For real model calls, construct `LlmTarget` values with an
`Arc<dyn RoutedLlmClient>` and pass them to an algorithm. The client maps each
target's semantic name to its provider model and transport configuration.

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

`request.llm_request.model` remains the model requested by the inbound caller.
The actual call target is `decision.selected_model()`. A client must use the
decision and map its semantic name to the provider model it serves.

`RoutedLlmClient::call` may return a buffered `LlmResponse::Agg` or a live
`LlmResponse::Stream`. See the
[`research_agent`](examples/research_agent.rs) example for a client-backed run.

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

See [`research_agent_core`](examples/research_agent_core.rs) for a complete host-
driven loop. The step stream is separate from the model response stream: one
algorithm step can return either a buffered or streaming `LlmResponse`.

## Streaming responses

`LlmResponse::into_agg` consumes a live stream and folds it into an
`AggLlmResponse`, surfacing stream and decoding failures. Algorithms that must
inspect a complete answer, such as judges, may buffer a response internally.

`AggLlmResponse::into_stream` is a synthetic, lossy conversion. It represents
text, reasoning, and tool calls, but omits content that has no neutral stream
event, including refusals, tool results, media, files, unknown blocks,
extensions, and preservation metadata. `ResponseAccumulator` also folds text
and reasoning into one assistant output, regardless of source output indices.

## Request preservation

`LlmRequest::preservation` and `AggLlmResponse::preservation` retain exact source
bodies for lossless same-format replay. With translation's default preservation
policy, a stored same-format body takes precedence over reconstruction from the
normalized fields.

If code mutates normalized messages, instructions, tools, or sampling controls
and those edits must reach a same-format upstream, clear the corresponding
preserved body or encode with preservation disabled. `Request::raw_request` is a
separate host envelope field; libsy does not read it.

The `prompt_text` and `completion_text` protocol helpers are also intentionally
lossy. They are convenient text views, not complete serialization APIs.

## Included algorithms

| Algorithm | Purpose |
|---|---|
| `Passthrough` | Always call one target. |
| `Random` | Uniform or weighted selection across any number of targets. |
| `LlmTaskClassifier` | Ask a judge to choose an efficient or capable target. |
| `StageRouter` | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |
| `Noop` | Return a fixed local response for host integration and testing. |

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

The authoritative trait contract is in the generated
[`Algorithm` documentation](https://docs.rs/switchyard-libsy/latest/switchyard_libsy/trait.Algorithm.html).
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

- [libsy API documentation](https://docs.rs/switchyard-libsy)
- [protocol API documentation](https://docs.rs/switchyard-protocol)
- [`research_agent`](examples/research_agent.rs): default-client execution
- [`research_agent_core`](examples/research_agent_core.rs): host-owned calls
- [`streaming_agent`](examples/streaming_agent.rs): response streaming
- [`ensemble`](examples/ensemble.rs): custom stateful algorithm

## License

Licensed under the Apache License, Version 2.0.

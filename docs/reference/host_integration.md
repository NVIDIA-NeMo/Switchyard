# Host Integration

Use this guide when embedding libsy algorithms that ask the host to serve model or decision calls. For a first routing result and model-client handoff, start with the [libsy quickstart](../getting_started.md).

## Development API

This page uses the development API, including its asynchronous `respond` method. The Rust examples pin the same revision as the [quickstart](../getting_started.md). The released `v0.3.0` callback API is synchronous. Use matching source-built [Python bindings](../../INSTALLATION.md#development) for this page's Python callback example.

## Route in Rust

With Rust 1.96.1+ and a native build toolchain, create a project:

```bash
cargo new switchyard-rust-demo
cd switchyard-rust-demo
```

Add these dependencies to `Cargo.toml`. Both Switchyard crates use the same pinned revision:

```toml
[dependencies]
async-trait = "0.1"
futures = "0.3"
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git", rev = "5415db205c1f4db7e079731e7bf33a6609c67c8e" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git", rev = "5415db205c1f4db7e079731e7bf33a6609c67c8e" }
tokio = { version = "1", features = ["macros", "rt"] }
```

Replace `src/main.rs` with this complete routing-only example:

```rust
use std::{collections::HashMap, sync::Arc};

use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_protocol::{Category, LlmRequest, Message, Request, Role};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec!["Efficient".into()]),
        (Category::Capable, vec!["Capable".into()]),
        (Category::Any, vec!["Efficient".into(), "Capable".into()]),
    ])));
    let request = Request {
        llm_request: LlmRequest {
            messages: vec![Message::text(Role::User, "Hello")],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::Done(outcome) => println!("Selected: {}", outcome.selected_model_ids[0]),
            _ => unreachable!("This Stage configuration makes no routing-time calls"),
        }
    }
    Ok(())
}
```

Run `cargo run`. It prints `Selected: Efficient` without making a network call. This example uses the pinned development API, including its `State` argument. For a complete routing and client example, follow the [Rust quickstart](../getting_started.md#use-libsy-with-your-client). Use [switchyard-translation](../../crates/switchyard-translation/README.md) to convert provider requests and libsy's protocol types, or [switchyard-llm-client](../../crates/libsy-llm-client/README.md) to drive an algorithm over HTTP.

## Serve routing-time calls

An algorithm yields a stream of steps. Each `Step::CallModel` is a routing-time classifier or judge call your host performs over its own transport. The run ends with `Step::Done` carrying a `RoutingOutcome`: the selected model, ordered fallbacks, rewritten request, and an optional response when routing already produced the answer. Otherwise the host makes the terminal answer call from that outcome.

Pass each call's work to `respond` before awaiting it. In Rust:

```rust
use switchyard_libsy::{CallDecision, CallModel, LibsyError, Result};
use switchyard_protocol::{RoutedDecisionClient, RoutedLlmClient};

async fn serve(call: CallModel, client: &dyn RoutedLlmClient) -> Result<()> {
    let request = call.request.clone();
    let model = call.models[0].clone();
    call.respond(async move {
        client.call(request).await.map_err(|error| LibsyError::client_call(model, error))
    }).await
}

async fn serve_decision(call: CallDecision, client: &dyn RoutedDecisionClient) -> Result<()> {
    let request = call.request.clone();
    let model = call.model.clone();
    call.respond(async move {
        client.call(request).await.map_err(|error| LibsyError::client_call(model, error))
    }).await
}
```

These traits belong to `switchyard-protocol`. Implement them over your own transport or SDK. The quickstart's custom [OpenRouter client](../getting_started.md#use-libsy-with-your-client) calls HTTP directly; `run_stream` also works without implementing these traits. Switchyard's HTTP client is optional.

In Python, the client coroutine returns `LlmResponse.Agg` or `LlmResponse.Stream`:

```python
await call.respond(client.call(call.request, call.models[0]))
```

`respond` accepts a future or awaitable instead of a completed response. Move the provider await into this method. Python `call.fail(error)` is also awaitable. Decision calls use the same pattern with their existing decision response type.

## Concurrent calls and cancellation

In Rust, `respond` drops the supplied work and returns `Ok(())` when the algorithm stops waiting for that call. Dropping a pending `respond` or `libsy::drive` future also drops its host work. Use `libsy::drive` to manage concurrent handlers and their lifetime.

Python `respond` cancels and awaits the supplied provider work before returning. For concurrent algorithms, the host creates handler tasks with `asyncio.create_task` and must cancel and await remaining handlers when the run ends; see the complete [Python example](../../examples/libsy.py). Awaiting handlers one at a time inside the step loop serializes their execution.

After a response stream is delivered, its consumer controls its lifetime. Cancellation cannot guarantee that an upstream provider stops generating or billing.

## Metadata and telemetry

Successful runs also include `OutcomeMetadata`: a unique `outcome_id`, the algorithm name, and optional JSON evidence. In Python, read `outcome.metadata.outcome_id`, `outcome.metadata.algorithm`, and `outcome.metadata.evidence` after checking that `outcome.metadata` is present. Evidence is a normal Python value, usually a dictionary; algorithms without evidence return `None`.

With a host-installed OpenTelemetry subscriber, the existing `libsy.run` span records the same identity, selected models, and supported evidence fields. See the [OpenTelemetry reference](opentelemetry.md) for field names, metrics, and export setup. libsy does not install an exporter or send telemetry itself.

For the request, response, and streaming types the steps carry, see [`switchyard-protocol`](../../crates/protocol/README.md).

For plan/execute, supply conversation history, including tool calls and results, before routing. Responses API continuation IDs alone are not enough. See the [Responses API history requirement](../routing_algorithms/plan_execute_routing.md#responses-api-history-requirement).

## Python bindings

Python bindings run the same Rust algorithms. With Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/), create a project and install the released package:

```bash
uv init --bare switchyard-python-demo
cd switchyard-python-demo
uv add "nemo-switchyard==0.3.0"
```

Save as `route.py`:

```python
import asyncio
from switchyard.libsy import Step, algorithms

async def main():
    router = algorithms.stage_router(
        picker="efficient_first", confidence_threshold=0.5)
    models = {"efficient": ["Efficient"], "capable": ["Capable"]}
    models["any"] = ["Efficient", "Capable"]
    text = {"type": "text", "text": "Hello"}
    request = {"messages": [{"role": "user", "content": [text]}]}
    async for step in router.run_stream(request, models):
        if isinstance(step, Step.Done):
            print("Selected:", step.outcome.selected_model_ids[0])

asyncio.run(main())
```

`uv run route.py` prints `Selected: Efficient` without making a network call. This routing-only example works with the released package. For the development API's host callbacks and cancellation, build the matching bindings from source as described at the top of this page, then see [the complete host example](../../examples/libsy.py).

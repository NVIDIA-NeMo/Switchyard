# Host Integration

Use this guide for shared client-trait adapters and Python bindings. For Rust
project setup and a working model-client handoff, start with the
[libsy quickstart](../getting_started.md).

For an explanation of steps, call objects, and the request and response contracts,
read [Mechanics of Libsy](../mechanics_of_libsy.md).

## Development API

The callback examples use the development API, including its asynchronous
`respond` method. Use the [quickstart's dependencies](../getting_started.md#install)
for the Rust snippets. The released `v0.3.0` callback API is synchronous. Use
matching source-built [Python bindings](../../INSTALLATION.md#development) for
the Python callback example below.

## Serve routing-time calls

The [Step loop](../mechanics_of_libsy.md#read-the-next-step) uses adapters that
return `switchyard_libsy::Result`. When using `RoutedLlmClient` or
`RoutedDecisionClient`, map their errors to `LibsyError`:

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

In Python, the client coroutine returns `LlmResponse.Agg` or `LlmResponse.Stream`:

```python
await call.respond(client.call(call.request, call.models[0]))
```

Pass the coroutine itself to `respond`. Python `call.fail(error)` is also
awaitable. Decision calls use the same pattern with their decision response type.

## Concurrent calls and cancellation

See [Control how work runs](../mechanics_of_libsy.md#control-how-work-runs) for
Rust execution, cancellation behavior, and upstream limits.

Python `respond` cancels and awaits the supplied provider work before returning. For concurrent algorithms, the host creates handler tasks with `asyncio.create_task` and must cancel and await remaining handlers when the run ends; see the complete [Python example](../../examples/libsy.py). Awaiting handlers one at a time inside the step loop serializes their execution.

## Metadata and telemetry

The [final outcome](../mechanics_of_libsy.md#stepdone) carries
routing metadata. In Python, read `outcome.metadata.outcome_id`,
`outcome.metadata.algorithm`, and `outcome.metadata.evidence` after checking that
`outcome.metadata` is present. Evidence is a normal Python value, usually a
dictionary; algorithms without evidence return `None`.

See the [OpenTelemetry reference](opentelemetry.md) for field names, metrics,
and host export setup.

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

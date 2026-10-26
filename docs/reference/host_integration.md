# Host Integration

Add libsy where your gateway or agent harness chooses and calls a model. Your
application keeps its transport, credentials, and lifecycle.

Start with the **[Rust quickstart](../getting_started.md)** for a working example.
Use **[Mechanics of Libsy](../mechanics_of_libsy.md)** for execution and
**[Switchyard Protocols](../protocols/index.md)** for data conversion. This page
covers the responsibilities specific to your host.

## Choose the integration boundary

| Host | Insert routing here | Preserve from the existing application |
|---|---|---|
| Gateway | After request admission and input decoding, before model dispatch. | Authentication, tenant isolation, budgets, model inventory, provider credentials, and response delivery. |
| Agent harness | Before each model turn, after assembling conversation and tool history. | Tool execution, task state, model clients, conversation storage, and the agent loop. |

## Host responsibilities

**Dispatch every requested call.** Apply your credentials, budgets, and access
policy to routing-time judge calls as well as answer calls. Send internal calls
through model dispatch without entering the same router again. Supply only
models your host can reach. See [call handling](../mechanics_of_libsy.md#read-the-next-step)
and [final-outcome handling](../mechanics_of_libsy.md#stepdone).

**Keep history and state.** Supply the tool history required by the algorithm;
provider continuation IDs alone can hide that evidence. Scope session IDs to
the tenant and conversation. Reuse algorithm instances when they retain session
decisions, and decide how your host handles concurrent turns and multiple
processes. State behavior is algorithm-specific; see [Composite sessions](../routing_algorithms/composite_routing.md#sessions)
and the [Plan/Execute history requirement](../routing_algorithms/plan_execute_routing.md#responses-api-history-requirement).

**Observe the complete run.** Include judge calls and final model work in your
host's observations. Configure telemetry collection in your application and
instrument model I/O performed by your own clients. See the
[OpenTelemetry reference](opentelemetry.md#host-setup).

For existing integrations, check [NeMo Relay](../integrations/nemo_relay.md)
or the [LiteLLM compatibility notes](../../examples/litellm/README.md#supported-algorithm-semantics).
The experimental LiteLLM plugin supports decision-only routing and cannot serve
intermediate model calls.

## Serve routing-time calls

Use your existing client as shown in the [Step loop](../mechanics_of_libsy.md#read-the-next-step).
The shared client traits below are optional.

<details markdown="1">
<summary>Optional: adapt the shared client traits</summary>

### Development API

These adapters use the [quickstart's development dependencies](../getting_started.md#install).
Their asynchronous reply method differs from the synchronous callback API in
the released `v0.3.0` crates.

When using
<a href="../../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>
or <a href="../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>,
adapt errors to <a href="../../rust/switchyard_libsy/enum.LibsyError.html"><code>LibsyError</code></a>:

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

</details>

## Python bindings

The bindings run the same Rust algorithms. The local routing example below works
with the released package. For the asynchronous callbacks and cleanup described
after it, build matching [development bindings](../../INSTALLATION.md#development)
and follow the complete [Python host example](../../examples/libsy.py).

<details markdown="1">
<summary>Python setup, callbacks, and metadata</summary>

With Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/),
create a project and install the released package:

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

`uv run route.py` prints `Selected: Efficient` without making a network call.

### Concurrent calls and cancellation

The client coroutine returns `LlmResponse.Agg` or `LlmResponse.Stream`.
Pass the coroutine itself to `respond`:

```python
await call.respond(client.call(call.request, call.models[0]))
```

`respond` cancels and awaits the supplied provider work before returning.
For concurrent algorithms, create handler tasks and cancel and await the remaining
handlers when the run ends, as in the [Python host example](../../examples/libsy.py).
Awaiting one handler at a time serializes execution. `call.fail(error)` is also
awaitable; decision calls use the same reply pattern.

For the shared cancellation behavior and upstream limits, see
[Mechanics](../mechanics_of_libsy.md#control-how-work-runs).

### Metadata and telemetry

After checking that `outcome.metadata` is present, read its `outcome_id`,
`algorithm`, and `evidence` attributes. Evidence is a normal Python value, usually
a dictionary; algorithms without evidence return `None`. See the
[OpenTelemetry reference](opentelemetry.md) for signals and export setup.

</details>

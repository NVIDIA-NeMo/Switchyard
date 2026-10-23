# Mechanics of Libsy

Libsy expresses work as a stream of **Steps**. Your application executes that work
with its own clients. `Done` is the exception: it carries the completed routing
outcome.

Follow one request through that exchange below. The snippets show parts of an
integration; the [Rust quickstart](getting_started.md) has a complete program,
imports, and matching development-API dependencies.

## Why execution stays with your application

Libsy separates routing policy from model execution. The algorithm decides which
models to use and how their results affect the next step. Your application decides
how those models are called.

```text
Libsy algorithm              Your gateway or harness
    CallModel  ------------> Execute with your model client
               <------------ Return a Response
    Read the result
    Request more work, or
    Done       ------------> Use the final routing outcome
```

A gateway can fulfill these calls through its existing dispatch, credentials,
retries, and observability. That includes calls to judges used during routing.
An agent harness can supply its conversation and tool history, then use its model
clients to execute the requested work. Both adapt their inputs and outputs to the
shared protocol types.

Routing can require more than choosing a model ID. An algorithm may change a
request, ask a judge, inspect an answer, or compare results from several models.
Returning responses through `call.respond` lets it continue that work. When a
selection alone is enough, `Done` leaves the final model call to your application.

The shared protocol crate lets clients and algorithms use the same request and
response contracts. A client can implement those contracts without depending on
the routing algorithm implementation.

## Start with a request

Your application supplies a
<a href="../rust/switchyard_protocol/envelope/struct.Request.html"><code>Request</code></a>.
Its `llm_request` field holds the
<a href="../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>:
the conversation, instructions, tools, and generation settings. These
provider-neutral types come from `switchyard-protocol`.

```rust
let request = Request {
    llm_request: LlmRequest {
        messages: vec![Message::text(Role::User, "Fix the failing tests.")],
        ..Default::default()
    },
    metadata: Some(Metadata {
        session_id: Some("session-42".into()),
        ..Default::default()
    }),
    ..Default::default()
};
```

<a href="../rust/switchyard_protocol/metadata/struct.Metadata.html"><code>Metadata</code></a>
holds the session ID above, which algorithms can use to retain session decisions.
It can also include a correlation ID for your logs. The optional `raw_request`
field holds data your application retains; libsy does not read it.

Next, supply the models the algorithm can use.
<a href="../rust/switchyard_libsy/struct.RuntimeModels.html"><code>RuntimeModels</code></a>
groups model IDs by
<a href="../rust/switchyard_protocol/category/enum.Category.html"><code>Category</code></a>,
such as `Efficient`, `Capable`, or `Judge`:

```rust
let models = Arc::new(RuntimeModels::new(HashMap::from([
    (Category::Efficient, vec!["Efficient".into()]),
    (Category::Capable, vec!["Capable".into()]),
    (Category::Judge, vec!["Judge".into()]),
    (Category::Any, vec!["Efficient".into(), "Capable".into()]),
])));
```

These are local example IDs. Within a category, models are ordered best-first. A
<a href="../rust/switchyard_protocol/model_id/struct.ModelId.html"><code>ModelId</code></a>
names a destination; your application decides which client serves it.

With an algorithm, request, model pool, and initial
<a href="../rust/switchyard_libsy/struct.State.html"><code>State</code></a>, start a run:

```rust
let state = State::default();
let mut steps = algorithm.run_stream(request, models, state);
```

<a href="../rust/switchyard_libsy/trait.Algorithm.html#method.run_stream"><code>run_stream</code></a>
returns a
<a href="../rust/switchyard_libsy/type.StepStream.html"><code>StepStream</code></a>.
Each item is a `Result<Step>`: either the next step or an error that ends the run.

## Read the next Step

A <a href="../rust/switchyard_libsy/enum.Step.html"><code>Step</code></a> describes
one unit of work for your application, or reports that routing has finished.
Its three variants are:

```rust
pub enum Step {
    CallModel(Box<CallModel>),
    CallDecision(Box<CallDecision>),
    Done(Box<RoutingOutcome>),
}
```

When you write `Step::CallModel(call)` in a match, `call` is the `Box<CallModel>`
stored in that variant. It is an object describing a requested operation, with a
way to return its result to the waiting algorithm.

For example, a signal-only router can finish locally. A classifier might first
ask your client to call a judge:

```text
Signal-only routing:  Done(outcome)
Judge-based routing:  CallModel(call) -> your response -> Done(outcome)
Decision routing:     CallDecision(call) -> your typed response -> Done(outcome)
```

## Fulfill a model call

Suppose the next step is `CallModel`. The
<a href="../rust/switchyard_libsy/struct.CallModel.html"><code>CallModel</code></a>
object describes the operation:

```rust
// The model has already been set to the first candidate.
let request = call.request.clone();
let first_candidate = &call.models[0];
assert_eq!(request.model_id().as_ref(), Some(first_candidate));
```

`models` is a nonempty candidate list in fallback order. Your application executes
the attempts and must update the request's model before trying a later candidate.
`call.algorithm` names the requesting algorithm for your own telemetry.

Your application returns the result through
<a href="../rust/switchyard_libsy/struct.CallModel.html#method.respond"><code>call.respond</code></a>.
Inside the `Step::CallModel(call)` arm, the exchange looks like this:

```rust
let request = call.request.clone();
// Pass the work future to respond; it returns the result to the algorithm.
call.respond(client.call(request)).await?;
```

Here, `client.call` is your async adapter, returning
`switchyard_libsy::Result<Response>`. It owns the provider call and any format
conversion, as in the [OpenRouter adapter](getting_started.md#use-libsy-with-your-client).
`respond` consumes the call, allowing the algorithm to continue with its result.

The returned
<a href="../rust/switchyard_protocol/envelope/struct.Response.html"><code>Response</code></a>
contains an
<a href="../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>
in its `llm_response` field:

```rust
pub enum LlmResponse {
    Stream(LlmResponseStream), // Live response events.
    Agg(AggLlmResponse),       // A complete, buffered answer.
}
```

The
<a href="../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>event type</code></a>
provides access to provider-neutral text, tool-call updates, and usage information.

`StepStream` carries work and the routing outcome. `LlmResponse::Stream` carries
model output. Your client can return that live stream without reading it to
completion first.

If you want a shared client interface, `switchyard-protocol` defines
<a href="../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>.
Its async `call(Request)` returns a `Response` or `LlmClientError`; the
[host guide](reference/host_integration.md#serve-routing-time-calls) shows the error
adapter. Using your own adapter method, as above, also works.

## Return a typed decision

An algorithm can ask a decision model to evaluate a question through
`Step::CallDecision(call)`. This time, `call` holds a
<a href="../rust/switchyard_libsy/struct.CallDecision.html"><code>CallDecision</code></a>.
It has one `model` and a
<a href="../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>
in `request`.

A decision request contains shared `context` and named questions. For example:

```rust
let request = DecisionRequest {
    model: Some("Judge".into()),
    context: serde_json::json!({"task": "Fix the failing tests."}),
    questions: BTreeMap::from([(
        "needs_capable".into(),
        DecisionQuestion {
            instructions: serde_json::json!("Does this task need Capable?"),
            kind: DecisionKind::Boolean {
                true_description: None,
                false_description: None,
            },
        },
    )]),
};
```

Each
<a href="../rust/switchyard_protocol/decision/struct.DecisionQuestion.html"><code>question</code></a>
declares its expected answer shape: Boolean, choice among options, or score on a
rubric. Your client returns a
<a href="../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>,
with matching question IDs. One possible answer to the question above is:

```rust
let response = DecisionResponse {
    id: None,
    model: Some("Judge".into()),
    answers: BTreeMap::from([(
        "needs_capable".into(),
        DecisionAnswer {
            value: DecisionValue::Boolean(BooleanEstimate::Value(true)),
            provider_confidence: None,
        },
    )]),
    usage: Default::default(),
};
```

The reply follows the same pattern as a model call. Inside the decision-call arm:

```rust
let request = call.request.clone();
call.respond(decision_client.call(request)).await?;
```

The adapter returns `switchyard_libsy::Result<DecisionResponse>`. The optional
shared interface is
<a href="../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>.
The algorithm applies its routing policy to the answer. The decision response is
distinct from the final conversational answer.

## Receive the final outcome

Once the algorithm has enough information, it emits `Step::Done(outcome)` with a
<a href="../rust/switchyard_libsy/struct.RoutingOutcome.html"><code>RoutingOutcome</code></a>.
Your application checks whether routing has already produced an answer:

```rust
let response = match outcome.response {
    Some(response) => response, // Reuse the answer obtained during routing.
    None => client.call(outcome.request).await?, // Make the final call.
};
```

With `Some(response)`, another final call would repeat work already done. With
`None`, your application can send `outcome.request` now, as above, or return it to
a gateway to send later. Signal-only Stage takes this second path.

Use the returned request: it includes algorithm changes and already names the
selected model. `outcome.selected_model_ids` lists that model first, followed by
ordered fallbacks.

To read the final answer as text, collect a live stream or use the existing
buffered answer:

```rust
let answer = response.llm_response.into_agg().await?;
println!("{}", completion_text(&answer));
```

For live text instead, poll response events as they arrive:

```rust
if let LlmResponse::Stream(mut events) = response.llm_response {
    while let Some(event) = events.next().await {
        for chunk in event?.normalized() {
            if let LlmResponseChunk::TextDelta { text, .. } = chunk {
                print!("{text}");
            }
        }
    }
}
```

These are alternative ways to consume a response. The text-only loop omits tool
and usage events; handle those chunks too when your application needs them.

`outcome.state` lets supporting algorithms retain caller-managed state between
turns. See [Composite sessions](routing_algorithms/composite_routing.md#sessions)
for how this differs from a session ID. Outcome
<a href="../rust/switchyard_libsy/struct.OutcomeMetadata.html"><code>metadata</code></a>
identifies the run and algorithm and may include evidence for the decision.

## Control how work runs

Each call has its own reply channel. Awaiting calls one at a time in the step loop
serves them sequentially. To run calls concurrently, your application must keep
reading steps while polling their work.

For concurrent execution, use
<a href="../rust/switchyard_libsy/fn.drive.html"><code>libsy::drive</code></a>.
It polls your call handlers concurrently and returns the `RoutingOutcome`. Your
handler receives
<a href="../rust/switchyard_libsy/enum.Call.html"><code>Call::Model</code> or <code>Call::Decision</code></a>;
`drive` handles `Done` itself:

```rust
// Borrow the adapters so every call can use them.
let client = &client;
let decision_client = &decision_client;
let outcome = switchyard_libsy::drive(algorithm, request, models, |call| async move {
    match call {
        Call::Model(call) => {
            let request = call.request.clone();
            call.respond(client.call(request)).await
        }
        Call::Decision(call) => {
            let request = call.request.clone();
            call.respond(decision_client.call(request)).await
        }
    }
}).await?;
```

It starts with default state. Use `run_stream` directly to supply your own state
or execution loop.

If the algorithm stops waiting, `respond` drops that call's work future. Dropping
the step stream aborts the algorithm task; your application cleans up separately
started work. `drive` drops its outstanding handlers when the run ends or is dropped.

A model-call failure normally stops the host run when propagated with `?`.
`call.recover_errors` lets the algorithm receive and handle it instead. Decision
failures passed through `respond` also return to the algorithm. Dropping an
unanswered call gives the waiting algorithm `DriverError::ResponseDropped`.

Once a response stream has been delivered, its consumer owns its lifetime.
Cancelling local work cannot guarantee that a provider stops generation or billing,
or undo side effects that have already occurred. The [host integration
guide](reference/host_integration.md#concurrent-calls-and-cancellation) covers host
cleanup in Rust and Python.

The [Rust API reference](reference/rust_api.md) links the complete crate
documentation for these types.

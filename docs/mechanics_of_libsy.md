# Mechanics of Libsy

Libsy expresses work as a stream of **Steps**, leaving your application to execute
that work with its own clients. `Done` is the exception: it carries the completed
routing outcome. Understanding what each step contains tells you what your
application needs to do next.

This page follows one request through that exchange. It uses the development API
from the [Rust quickstart](getting_started.md), including asynchronous `respond`.
The quickstart has a complete working example and matching dependency versions.

## Start with a request

Your application supplies a
<a href="../rust/switchyard_protocol/envelope/struct.Request.html"><code>Request</code></a>.
Its `llm_request` field contains the
<a href="../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>:
the conversation, instructions, tools, and generation settings. These are
provider-neutral types defined by `switchyard-protocol`. Your client translates
them to the format its provider accepts.

The request can also carry
<a href="../rust/switchyard_protocol/metadata/struct.Metadata.html"><code>Metadata</code></a>,
such as a session ID or correlation ID. This information helps algorithms retain
session decisions and lets your application connect a request to its logs. The
optional `raw_request` field holds data your application retains; libsy does not
read it.

Next, supply the models the algorithm can use.
<a href="../rust/switchyard_libsy/struct.RuntimeModels.html"><code>RuntimeModels</code></a>
groups model IDs by
<a href="../rust/switchyard_protocol/category/enum.Category.html"><code>Category</code></a>,
such as `Efficient`, `Capable`, or `Judge`. Within each category, models are ordered
best-first. A
<a href="../rust/switchyard_protocol/model_id/struct.ModelId.html"><code>ModelId</code></a>
names a destination; your application decides which client serves it.

With an algorithm, request, model pool, and initial
<a href="../rust/switchyard_libsy/struct.State.html"><code>State</code></a>, start a run:

```rust
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

An algorithm may select a model using local rules and go straight to `Done`.
Another may ask for a judge call, read its result, and then finish. An algorithm
can also request several calls and use their results to decide what happens next.
The variants stay the same across these workflows.

## Fulfill a model call

Suppose the next step is `CallModel`. The
<a href="../rust/switchyard_libsy/struct.CallModel.html"><code>CallModel</code></a>
object contains `request`, the request to serve, and `models`, a nonempty list of
candidate model IDs in fallback order. Its `algorithm` field names the algorithm
that requested the call, which is useful when recording your own telemetry.

The request already names the first candidate in `request.llm_request.model`.
Your client can send it through your existing SDK, HTTP transport, or local model.
If your application tries a later candidate, it must update the request's model
before making that attempt. Libsy supplies the candidates; your client executes
the attempts.

Your application returns the result through
<a href="../rust/switchyard_libsy/struct.CallModel.html#method.respond"><code>call.respond</code></a>.
Inside the `Step::CallModel(call)` arm, the exchange looks like this:

```rust
let request = call.request.clone();
call.respond(client.call(request)).await?;
```

Here, `client.call` is your adapter's async method, returning
`switchyard_libsy::Result<Response>`. The [quickstart's OpenRouter
adapter](getting_started.md#use-libsy-with-your-client) is one example.
`respond` consumes the call and passes the result back to the algorithm. The
algorithm can then read that result and continue.

The returned
<a href="../rust/switchyard_protocol/envelope/struct.Response.html"><code>Response</code></a>
contains an
<a href="../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>
in its `llm_response` field. `LlmResponse::Agg` holds a complete answer.
`LlmResponse::Stream` holds a live stream of response events, including text,
tool-call updates, and usage information. The
<a href="../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>event type</code></a>
provides access to those provider-neutral chunks.

These are two separate streams. `StepStream` carries work requests and the routing
outcome. An `LlmResponse::Stream` carries the model's output. Returning a live
response stream to the algorithm does not require reading it to completion first.

If you want a shared client interface, `switchyard-protocol` defines
<a href="../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>.
Its `call` method accepts a `Request` and returns a `Response` or `LlmClientError`.
The [host integration guide](reference/host_integration.md#serve-routing-time-calls)
shows how to adapt those errors for `respond`. Implementing this trait is optional
when you consume `run_stream` yourself; an adapter method with the expected return
type is enough.

## Return a typed decision

An algorithm can ask a decision model to evaluate a question through
`Step::CallDecision(call)`. This time, `call` holds a
<a href="../rust/switchyard_libsy/struct.CallDecision.html"><code>CallDecision</code></a>.
It has one `model` and a
<a href="../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>
in `request`.

A decision request contains shared `context` and named questions. Each
<a href="../rust/switchyard_protocol/decision/struct.DecisionQuestion.html"><code>question</code></a>
has instructions and an expected answer shape: a Boolean judgment, a choice among
declared options, or a score on an ordered rubric. Your decision client sends this
request and returns a
<a href="../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>,
whose answers use the matching question IDs.

The reply follows the same pattern as a model call. Inside the decision-call arm:

```rust
let request = call.request.clone();
call.respond(decision_client.call(request)).await?;
```

Here, the adapter returns `switchyard_libsy::Result<DecisionResponse>`. For a shared
client interface, the protocol crate also defines
<a href="../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>.
The algorithm interprets the answer and applies its routing policy. A decision
response is distinct from the final answer to the user's conversation.

## Receive the final outcome

Once the algorithm has enough information, it emits `Step::Done(outcome)` with a
<a href="../rust/switchyard_libsy/struct.RoutingOutcome.html"><code>RoutingOutcome</code></a>.
The step stream has finished its work. What your application does next depends on
whether routing has already produced an answer.

When `outcome.response` is `Some(response)`, use that response. An algorithm may
already have requested and evaluated the answer through a model-call step.
Sending another final request would repeat that work.

When `outcome.response` is `None`, the algorithm has selected a destination and
prepared `outcome.request` for it. Your application can send that request now or
return it to a gateway that makes the final call later. This is the path taken by
the signal-only Stage configuration in the quickstart.

The returned request includes any changes the algorithm made and already names
the selected model. `outcome.selected_model_ids` lists that model first, followed
by ordered fallbacks. Use the returned request when making the final call so those
changes reach the selected model.

The outcome also carries `state` for applications that retain routing state between
turns. Algorithms that support caller-managed state can consume it on the next run;
the [Composite sessions guide](routing_algorithms/composite_routing.md#sessions)
explains how that differs from supplying a session ID. Outcome
<a href="../rust/switchyard_libsy/struct.OutcomeMetadata.html"><code>metadata</code></a>
identifies the run and algorithm and may include evidence for the decision.

## Control how work runs

Each call has its own reply channel. Several calls can be in progress at once,
provided your application keeps reading steps and polls the work concurrently.
Awaiting each call inside the step loop serves them sequentially.

For concurrent execution, use
<a href="../rust/switchyard_libsy/fn.drive.html"><code>libsy::drive</code></a>.
It reads the step stream, polls your call handlers concurrently, and returns the
`RoutingOutcome`. Your handler receives
<a href="../rust/switchyard_libsy/enum.Call.html"><code>Call::Model</code> or <code>Call::Decision</code></a>;
`drive` handles `Done` itself. It starts with default routing state. Use
`run_stream` directly when you need to supply your own state or execution loop.

Passing the work future to `respond` also gives cancellation a defined lifetime.
If the algorithm stops waiting for a call, `respond` drops that call's work.
Dropping the step stream aborts the algorithm task; your application must also
clean up any work it started separately. `drive` drops its outstanding handlers
when the run ends or is dropped.

A model-call failure normally stops the host run when propagated with `?`. When
`call.recover_errors` is enabled, `respond` returns the failure to the algorithm so
it can apply its recovery policy. A decision-call failure passed through `respond`
also returns to the algorithm. Dropping a call without responding closes its reply
channel, and the waiting algorithm receives `DriverError::ResponseDropped`.

Once a response stream has been delivered, its consumer owns its lifetime.
Cancelling local work cannot guarantee that a provider stops generation or billing,
or undo side effects that have already occurred. The [host integration
guide](reference/host_integration.md#concurrent-calls-and-cancellation) covers host
cleanup in Rust and Python.

The exchange is the same whether your application makes one call or manages a
larger workflow: read the requested work, return its typed result, and use the
outcome when routing finishes. The [Rust API reference](reference/rust_api.md)
links the complete crate documentation for the types used here.

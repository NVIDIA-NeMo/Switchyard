# libsy — Switchyard-Lib

A lightweight, provider-agnostic library for multi-LLM agent optimization, with
**routing** as the first case. libsy *decides* how to serve each request — which
model(s) to call, in what order, how to combine them — and either makes the calls
itself or hands them back for you to make. It owns no HTTP client and no provider SDK,
so it drops into a proxy, gateway, or agent runtime.

## Example

Build a target set, pick an algorithm, run a request:

```rust
use switchyard_libsy::{Algorithm, Context, RoutedLlmClient, LlmTarget, LlmTargetSet, Request};
use switchyard_libsy::algorithms::LlmClassifier;
use switchyard_protocol::{completion_text, text_request};
use std::sync::Arc;

// Targets the algorithm routes among, each backed by your RoutedLlmClient (see below).
let client = Arc::new(MyClient { /* .. */ }) as Arc<dyn RoutedLlmClient>;
let target = |name: &str| LlmTarget { semantic_name: name.into(), llm_client: Some(client.clone()) };

let algo: Arc<dyn Algorithm> = Arc::new(LlmClassifier::new(
    "classifier", "strong", "weak", 0.5,
    LlmTargetSet::new(vec![target("classifier"), target("strong"), target("weak")]),
));

let req = Request {
    // `text_request` is the single-turn shortcut; build an `LlmRequest` directly for
    // multi-turn conversations, tools, or sampling parameters.
    llm_request: text_request(Some("auto".into()), "explain tail latency"),
    raw_request: None,
    metadata: None,
};
let (trace, response) = algo.clone().run(Context::default(), req).await?;  // calls in, trace + response out
println!("routed to {}", trace.last().unwrap().selected_model());
// `response.llm_response` is buffered or streamed; fold it to the aggregate to read text.
println!("answer: {}", completion_text(&response.llm_response.into_agg().await?));
```

Runnable: [`research_agent`](examples/research_agent.rs)

## Sub-agent affinity

Compose `AffinityRouter` as both the first classifier and a processor. On a child
agent's first turn it abstains, the random fallback chooses a target, and the
processor retains that decision. Later turns from the same `(session_id, agent_id)`
short-circuit to the retained target. Root-agent traffic continues through the
fallback on every turn.

```rust
use switchyard_libsy::{
    algorithms::{AffinityRouter, FallThrough, RandomClassifier},
    Algorithm, Classifier, Processor,
};
use std::sync::Arc;

let affinity = Arc::new(AffinityRouter::for_subagents());
let algo: Arc<dyn Algorithm> = Arc::new(
    FallThrough::new(targets)
        .with_processor(affinity.clone() as Arc<dyn Processor>)
        .with_classifier(affinity as Arc<dyn Classifier>)
        .with_classifier(Arc::new(RandomClassifier::new(["frontier", "fast"]))),
);
```

Request identity comes from `switchyard_protocol::Metadata`; HTTP integrations can
normalize supported harness headers with `Metadata::from_headers` before routing.

## Requests & responses

```rust
pub struct Request {
    pub llm_request: LlmRequest,
    pub raw_request: Option<serde_json::Value>, // optional original provider body for exact-fidelity hosts
    pub metadata: Option<Metadata>,             // correlation: session / agent / task / correlation_id / extra
}

// `Response` is a plain struct. The buffered-or-streamed choice lives one level down,
// in its `llm_response` — which *is* an enum.
pub struct Response {
    pub llm_response: LlmResponse,
    pub metadata: Option<Metadata>,
}

pub enum LlmResponse {
    Stream(LlmResponseStream), // a live stream of `LlmResponseChunk`s (token-by-token)
    Agg(AggLlmResponse),       // the buffered aggregate — outputs, usage, stop reason
}
```

`switchyard-protocol` owns the conversation IR: `LlmRequest`, the buffered
`AggLlmResponse`, the streaming `LlmResponseChunk`, and the building blocks `Message`,
`ContentBlock`, `ResponseOutput`, and `Role`. `libsy` re-exports the envelope
(`Request`/`Response`/`Metadata`) plus `LlmRequest`, `LlmResponse`, `AggLlmResponse`,
`LlmResponseChunk`, and `LlmResponseStream`; import the rest (`Message`, `Role`, …) and the
`text_request` / `text_response` / `completion_text` helpers from `switchyard_protocol`.
Construct and inspect the conversation model directly so tools, sampling parameters,
reasoning, and provider extensions stay visible instead of being hidden behind a second
convenience API. `raw_request` remains available when a host needs exact source-body
fidelity.

## Targets and clients

An `LlmTarget` pairs a routing `semantic_name` with an optional `RoutedLlmClient`. Mapping that
name to a provider model id is the client's job, not the algorithm's — `RoutedLlmClient` is
meant to be implemented by you.

```rust
struct MyClient { /* http client, base url, key */ }

#[async_trait::async_trait]
impl RoutedLlmClient for MyClient {
    async fn call(&self, ctx: Context, request: Request, decision: Arc<dyn Decision>)
        -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
        let model = decision.selected_model();   // the routed target — map it to a provider id
        // request.llm_request.model is the agent's original name (not a call target)
        // ... POST to your endpoint, read the completion ...
        let completion = String::from("provider response text");
        Ok(Response {
            // Buffered: wrap the aggregate in `LlmResponse::Agg`. `text_response` builds a
            // single-turn `AggLlmResponse`; construct one directly for tool calls, usage, etc.
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        })
    }
}
```

To stream instead, return `LlmResponse::Stream(..)` — a boxed
`Stream<Item = Result<LlmResponseChunk, _>>` — and emit chunks as they arrive from your
upstream. See [Streaming responses](#streaming-responses) below.

A `RoutedRequest` bundles the `request` with the routing `decision`, the target's
`default_client`, and the request's `ctx`; the model to call is
`decision.selected_model()`, never a mutated request field. `semantic_name` is the label an algorithm routes by; the client maps it to
the id it calls — they can differ (`"strong"` → `"openai/gpt-4o"`) or coincide.

## Running a request

Hold the algorithm as `Arc<dyn Algorithm>` and choose one of two entry points:

```rust
// run: libsy drives the request to completion, serving each call with the target's
// client, and returns (trace, response). Errors if a routed target has no client.
let (trace, response) = algo.clone().run(Context::default(), req).await?;

// run_stream: "ask, don't call" — you drive the stream and make the calls.
let stream = algo.clone().run_stream(Context::default(), req);
```

Under the hood every model call is *offloaded* to the request's `Step` stream; `run` is
the convenience that serves each one via the target's client. The step stream is bounded,
so pulling it paces the algorithm; each run is independent, so many run concurrently.

## Streaming responses

A `Response` carries an `LlmResponse` that is either buffered (`Agg`) or a live token
stream (`Stream`). An `RoutedLlmClient` — or an algorithm — chooses: return
`LlmResponse::Agg(..)` for a whole answer, or `LlmResponse::Stream(..)` to forward tokens
as they arrive. libsy never buffers a stream on your behalf; it flows through the algorithm
untouched, so `run` / `run_stream` hand back whatever was produced and **the caller
decides**:

```rust
use futures::StreamExt;

let (_trace, response) = algo.clone().run(Context::default(), req).await?;
match response.llm_response {
    // Forward tokens as they arrive — e.g. re-encode each chunk to your client's SSE.
    LlmResponse::Stream(mut chunks) => {
        while let Some(chunk) = chunks.next().await {
            let chunk: LlmResponseChunk = chunk?;   // TextDelta / ToolCallDelta / Usage / MessageStop / ..
            /* emit chunk */
        }
    }
    // Already buffered — read it directly.
    LlmResponse::Agg(agg) => println!("{}", completion_text(&agg)),
}
```

Need the whole answer regardless of how it arrived? `into_agg()` folds a `Stream` (or
returns an `Agg` unchanged) into a single `AggLlmResponse`, surfacing any mid-stream error:

```rust
let agg = response.llm_response.into_agg().await?;   // drives the stream to completion
println!("{}", completion_text(&agg));
```

An `LlmResponseChunk` is the provider-neutral streaming event (`MessageStart`, `TextDelta`,
`ReasoningDelta`, `ToolCallDelta`, `Usage`, `MessageStop`, `Error`); `ResponseAccumulator`
is the fold behind `into_agg()` if you need to assemble one yourself. Runnable end-to-end
demo: [`streaming_agent`](examples/streaming_agent.rs).

## Driving the calls yourself (`run_stream`)

`run_stream` yields `CallLlm` promises you fulfill with your own transport (or the call's
`default_client`), streams each `Decision` as it happens, and ends with `ReturnToAgent`.
This is the *step* stream (one `Step` at a time); it is orthogonal to whether any single
response is itself streamed — see [Streaming responses](#streaming-responses).
Runnable: [`research_agent_core`](examples/research_agent_core.rs).

```rust
let stream = algo.clone().run_stream(Context::default(), req);
tokio::pin!(stream);
while let Some(step) = stream.next().await {
    match step? {
        Step::CallLlm(call) => {
            let routed = call.get_routed().clone();                  // which target, and its default client
            let response = call_model(routed.decision.selected_model(), &routed.request).await;  // your real call
            call.respond(Ok(response))?;                             // or Err(..) to propagate a failure
        }
        Step::Decision(decision) => { /* decision.selected_model(), decision.reasoning() */ }
        Step::ReturnToAgent(response) => { /* done */ }
    }
}
```

## Building an algorithm (`Algorithm`)

Implement `Algorithm` to add a strategy. You write `create_run_task` — one call per
request; `run` / `run_stream` are provided and drive it. Make model calls on the `Driver`
you're handed, and publish a `Decision` for each so consumers (and clients) see *which*
model and *why*.

```rust
#[async_trait]
pub trait Algorithm: Send + Sync + 'static {
    // `self: Arc<Self>` (not `&mut`): one algorithm serves requests concurrently — use
    // interior mutability for state. Offload calls/decisions on `driver`.
    async fn create_run_task(self: Arc<Self>, ctx: Context, driver: Driver, request: Request)
        -> Result<Response, Box<dyn Error + Send + Sync>>;
    async fn process_signals(self: Arc<Self>, signals: Signals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
    // provided: run(ctx, request) -> (trace, response), run_stream(ctx, request) -> Stream<Step>
}

pub trait Decision: Send + Sync {
    fn selected_model(&self) -> &str;          // the model chosen — the client's call target
    fn reasoning(&self) -> Option<&str>;       // human-readable "why"
    fn as_any(&self) -> &dyn std::any::Any;    // downcast to the concrete decision
}
```

Give it a `new(config.., target_set)` constructor and `Arc`-wrap it — there is no builder.
Example — the LLM classifier (classify, then route; full version in
[`src/algorithms/llm_class.rs`](src/algorithms/llm_class.rs)):

```rust
#[async_trait]
impl Algorithm for LlmClassifier {
    async fn create_run_task(self: Arc<Self>, ctx: Context, driver: Driver, request: Request)
        -> Result<Response, Box<dyn Error + Send + Sync>> {
        // Thread `ctx` into every offloaded call and decision — it carries the request's
        // cross-cutting state (correlation ids, budgets) for observers downstream.

        // 1. Classify: ask the classifier target for a score.
        let classifier = self.target_set.get_target(&self.classifier_model)?;
        driver.info(ctx.clone(), classify_decision.clone()).await?;
        let classify_response =
            driver.call_llm_target(ctx.clone(), &classifier, classify_req, classify_decision).await?;
        // Abbreviated: fold the (buffered or streamed) response to its aggregate and read
        // the completion text as a score. `.as_agg()` alone is `None` for an unfolded stream.
        let score = classify_response.llm_response.into_agg().await
            .ok()
            .and_then(|agg| completion_text(&agg).trim().parse::<f64>().ok());

        // 2. Route: strong if score >= threshold, else weak (fail open on None).
        let model = if score.map_or(true, |s| s >= self.threshold) { &self.strong_model } else { &self.weak_model };
        let routed = self.target_set.get_target(model)?;
        driver.info(ctx.clone(), route_decision.clone()).await?;
        driver.call_llm_target(ctx, &routed, routed_req, route_decision).await
    }

    async fn process_signals(self: Arc<Self>, _s: Signals) -> Result<(), Box<dyn Error + Send + Sync>> { Ok(()) }
}
```

## Explore

The core crate includes uniform random routing and naive LLM classifier. Runnable
agents live in [`examples`](examples/) folder.

**Reference algorithms** — implementations to read and route with:

- [`Random`](src/algorithms/rand.rs) — uniform random over the set
  (one call).
- [`LlmClassifier`](src/algorithms/llm_class.rs) — classify, then route
  strong/weak; fail open to strong.
- [`EnsembleOrchAlgo`](examples/ensemble.rs) — stateful: fan out to
  candidates, judge the best, commit to the winner after N exploration turns.

**Runnable agents** (`cargo run -p switchyard-libsy --example <name>`):

- [`ensemble`](examples/ensemble.rs) — query three NVIDIA-hosted candidates, then judge and commit.
- [`research_agent`](examples/research_agent.rs) — client-backed
  targets, `run` (libsy makes the calls).
- [`research_agent_core`](examples/research_agent_core.rs) — client-less
  targets, `run_stream` (the agent makes the calls).
- [`streaming_agent`](examples/streaming_agent.rs) — a target that streams
  its response; the agent forwards each `LlmResponseChunk` to the caller token-by-token.

## Not yet built

- **`Signals` events** — `process_signals` / `Signals` exist but carry nothing yet.
- **`Context` fields** — the per-request state carrier is an empty placeholder today.
- **Observability** — spans + a metrics sink (`Decision` is the hook).
- **Config-driven construction**, **typed errors** (vs `Box<dyn Error>`), **weighted random**.

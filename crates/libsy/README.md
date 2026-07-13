# libsy — Switchyard-Lib

A lightweight, provider-agnostic library for multi-LLM agent optimization, with
**routing** as the first case. libsy *decides* how to serve each request — which
model(s) to call, in what order, how to combine them — and either makes the calls
itself or hands them back for you to make. It owns no HTTP client and no provider SDK,
so it drops into a proxy, gateway, or agent runtime.

## Example

Build a target set, pick an algorithm, run a request:

```rust
use libsy::llm_class::LlmClassifierOrchAlgo;
use libsy::{text_request, Algorithm, Context, LlmClient, LlmTarget, LlmTargetSet, Request};
use std::sync::Arc;

// Targets the algorithm routes among, each backed by your LlmClient (see below).
let client = Arc::new(MyClient { /* .. */ }) as Arc<dyn LlmClient>;
let target = |name: &str| LlmTarget { semantic_name: name.into(), llm_client: Some(client.clone()) };

let algo: Arc<dyn Algorithm> = Arc::new(LlmClassifierOrchAlgo::new(
    "classifier", "strong", "weak", 0.5,
    LlmTargetSet::new(vec![target("classifier"), target("strong"), target("weak")]),
));

let req = Request {
    llm_request: text_request("auto", "explain tail latency"),
    raw_request: None,
    metadata: None,
};
let (trace, response) = algo.clone().run(Context::default(), req).await?;  // calls in, trace + response out
println!("routed to {}", trace.last().unwrap().selected_model());
```

Runnable: [`research_agent`](../libsy-examples/examples/research_agent.rs) (in the `libsy-examples` crate).

## Requests & responses

```rust
pub struct Request {
    pub llm_request: LlmRequest,                // type alias for ConversationRequest
    pub raw_request: Option<serde_json::Value>, // optional original provider body for exact-fidelity hosts
    pub metadata: Option<Metadata>,             // correlation: session / agent / task / correlation_id / extra
}

pub struct Response {
    pub llm_response: LlmResponse,              // type alias for ConversationResponse
    pub metadata: Option<Metadata>,
}
```

`LlmRequest` and `LlmResponse` are semantic aliases over Switchyard's shared
conversation IR. Text-only algorithms and examples can use `text_request`,
`text_response`, `request_text`, and `response_text`; richer provider details can ride
in the IR itself or in `raw_request` when a host needs exact source-body fidelity.

## Targets and clients

An `LlmTarget` pairs a routing `semantic_name` with an optional `LlmClient`. Mapping that
name to a provider model id is the client's job, not the algorithm's — `LlmClient` is
meant to be implemented by you.

```rust
struct MyClient { /* http client, base url, key */ }

#[async_trait::async_trait]
impl LlmClient for MyClient {
    async fn call(&self, routed: RoutedRequest)
        -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
        let model = routed.decision.selected_model();   // the routed target — map it to a provider id
        // routed.request.llm_request.model is the agent's original name (not a call target)
        // ... POST to your endpoint, read the completion ...
        Ok(Response { llm_response: text_response(completion), metadata: None })
    }
}
```

A `RoutedRequest` bundles the `request` with the routing `decision` and the target's
`default_client`; the model to call is `decision.selected_model()`, never a mutated
request field. `semantic_name` is the label an algorithm routes by; the client maps it to
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

## Streaming — you own the model calls (`run_stream`)

`run_stream` yields `CallLlm` promises you fulfill with your own transport (or the call's
`default_client`), streams each `Decision` as it happens, and ends with `ReturnToAgent`.
Runnable: [`research_agent_core`](../libsy-examples/examples/research_agent_core.rs) (in the `libsy-examples` crate).

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
[`libsy-examples/src/llm_class.rs`](../libsy-examples/src/llm_class.rs)):

```rust
#[async_trait]
impl Algorithm for LlmClassifierOrchAlgo {
    async fn create_run_task(self: Arc<Self>, _ctx: Context, driver: Driver, request: Request)
        -> Result<Response, Box<dyn Error + Send + Sync>> {
        // 1. Classify: ask the classifier target for a score.
        let classifier = self.target_set.get_target(&self.classifier_model)?;
        driver.info(classify_decision.clone()).await?;
        let classify_response =
            driver.call_llm_target(&classifier, classify_req, classify_decision).await?;
        let score = response_text(&classify_response.llm_response)
            .trim()
            .parse::<f64>()
            .ok();

        // 2. Route: strong if score >= threshold, else weak (fail open on None).
        let model = if score.map_or(true, |s| s >= self.threshold) { &self.strong_model } else { &self.weak_model };
        let routed = self.target_set.get_target(model)?;
        driver.info(route_decision.clone()).await?;
        driver.call_llm_target(&routed, routed_req, route_decision).await
    }

    async fn process_signals(self: Arc<Self>, _s: Signals) -> Result<(), Box<dyn Error + Send + Sync>> { Ok(()) }
}
```

## Explore

Reference algorithms and runnable agents live in the sibling
[`libsy-examples`](../libsy-examples) crate (kept out of `libsy` itself, but compiled and
tested — `cargo test -p libsy-examples`).

**Reference algorithms** — implementations to read and route with:

- [`RandomOrchAlgo`](../libsy-examples/src/rand.rs) — uniform random over the set (one call).
- [`LlmClassifierOrchAlgo`](../libsy-examples/src/llm_class.rs) — classify, then route
  strong/weak; fail open to strong.
- [`EnsembleOrchAlgo`](../libsy-examples/src/ensemble.rs) — stateful: fan out to
  candidates, judge the best, commit to the winner after N exploration turns.

**Runnable agents** (`cargo run -p libsy-examples --example <name>`):

- [`research_agent`](../libsy-examples/examples/research_agent.rs) — client-backed
  targets, `run` (libsy makes the calls).
- [`research_agent_core`](../libsy-examples/examples/research_agent_core.rs) — client-less
  targets, `run_stream` (the agent makes the calls).

## Not yet built

- **`Signals` events** — `process_signals` / `Signals` exist but carry nothing yet.
- **`Context` fields** — the per-request state carrier is an empty placeholder today.
- **Observability** — spans + a metrics sink (`Decision` is the hook).
- **Config-driven construction**, **typed errors** (vs `Box<dyn Error>`), **weighted random**.

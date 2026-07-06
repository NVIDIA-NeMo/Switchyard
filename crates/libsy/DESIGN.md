# Switchyard-Lib Design

Audience: contributors building or embedding `libsy`, and integrators wiring it into an inference
platform, an agentic system, or a routing proxy.

Prerequisites: familiarity with async Rust (`async_trait`, `Send + Sync`), LLM request/response
shapes, and the idea of "routing" a request to one of several models. You do **not** need to know
the wider Switchyard proxy, its Python chain, or the profile v2 component model to read this.

Status: design draft. The traits in `src/lib.rs` and the two reference algorithms in `src/rand.rs`
and `src/llm_class.rs` are a proof of concept. This document describes the intended shape of the
library and marks where the current POC diverges from it (see *Current POC State and Gaps*).

## Summary

`libsy` (Switchyard-Lib) is a lightweight Rust crate for **multi-LLM agent optimization**, of which
**routing** is the first and most important case. It sits at the point in a system where a request
is *about to* be sent to a model and decides — statefully, and using more than just the request
itself — *which* model(s) to call, *how* to rewrite the call, or whether to skip the call entirely.

The whole library is built around one small pair of traits:

```text
OptAlgorithm  --optimizer()-->  Optimizer  --feed()/optimize()-->  Decision
   (factory,                     (per-session
    from config or code)          stateful instance)
```

An `Optimizer` is a stateful, per-session state machine. You **feed** it inputs — the inbound
request, model responses, and signals from the agentic/inference stack — and you ask it to
**optimize**, which returns a `Decision`: either "make these model calls and feed me the results"
(`ModelInference`) or "you're done, hand control back to the agent" (`Return`). A one-shot router
and a multi-round LLM classifier are the *same* control loop with different internal state.

Everything else in the library exists to make that loop **production-grade**: provider/transport
neutrality so it embeds anywhere, correlation so decisions can be attributed to a session/agent/
task/tool, and observability so every decision, token count, timing, and failure is a tagged span
you can drive dashboards, evaluations, and benchmarks from.

## Scope and Positioning

`libsy` is a **library, not a service**. It carries no HTTP server, no provider SDK, and no
transport. It is designed to be embedded in three kinds of host:

- **Inference platforms** — the serving layer decides, per request, which backend/model/replica to
  use. `libsy` is the decision core; the platform owns the sockets and the model calls.
- **Agentic systems** — an agent framework wants tier selection, cascade/escalation, or
  cost/latency optimization *inside* its reasoning loop, informed by tool calls and task state, not
  just the prompt.
- **Routing proxies** — a proxy (such as Switchyard itself) terminates a client protocol and needs
  to route or rewrite before forwarding. `libsy` is the routing brain behind the proxy's I/O.

Two properties make this possible:

- **Provider-API agnostic.** `libsy` reasons over neutral request/response types, never over
  `openai::ChatCompletion` or `anthropic::Message`. The host converts at the edge.
- **Transport agnostic.** `libsy` never performs a model call. A `ModelInference` decision *asks*
  the host to make calls; the host uses whatever HTTP/gRPC/in-process transport it already owns and
  feeds results back. This keeps `libsy` free of runtime and dependency lock-in.

## Goals and Non-Goals

Goals:

- One small, stable trait surface for optimization algorithms, with routing as the primary case.
- Statefulness as a first-class property: algorithms accumulate context across a session.
- Inputs richer than the request: responses **and** events/signals from the agentic stack.
- First-class correlation: session, agent, task, tool.
- Full observability suitable for both production and research/evaluation/benchmarking.
- Two ways to construct: instantiate a concrete algorithm directly, or build a
  `SwitchyardRouter` / `SwitchyardOptimizer` from config.

Non-Goals:

- Not a proxy, gateway, or server; no listeners, no protocol termination.
- No provider SDKs and no HTTP client. Model calls are the host's job.
- No model *hosting*, batching, or KV-cache management.
- No global scheduler across sessions. An `Optimizer` optimizes one session; cross-session policy
  (fleet load, global budgets) enters as *fed signals*, not as hidden shared state.

## Core Mental Model

The library is one loop. The host drives it:

```text
feed(request)                      // inbound request enters the session
loop:
    decision = optimize()
    match decision:
        ModelInference(reqs) ->     // host performs the model call(s) it names,
            for r in reqs:          // using its own transport,
                resp = host_call(r) // then feeds each result back
                feed(response=resp)
        Return                 ->   // optimizer is done; host returns to the agent
            break
```

A **single-round** algorithm (weighted random routing) rewrites the model once and returns. A
**multi-round** algorithm (LLM classifier) emits a classifier call, consumes its score, emits the
routed call, then returns — all through the same loop, with no special-casing in the host. This is
the central design property: *the host's integration code does not change when the algorithm's round
count does.*

### The two traits

Grounded in `src/lib.rs` (names shown in their intended, corrected form — see *Current POC State*):

```rust
#[async_trait]
pub trait Optimizer<D>: Send + Sync {
    /// Feed one input plus its correlation/enrichment context. Accumulates state.
    async fn feed(&mut self, input: OptInput, enrichment: EnrichmentData)
        -> Result<(), OptError>;

    /// Decide what to do next given accumulated state.
    async fn optimize(&mut self) -> Result<Decision<D>, OptError>;
}

pub trait OptAlgorithm<D>: Send + Sync {
    /// Mint a fresh optimizer for one session.
    fn optimizer(&self) -> Box<dyn Optimizer<D>>;
}
```

`OptAlgorithm` is a **factory** — cheap, shareable config that mints one `Optimizer` per session.
The `Optimizer` is the **stateful instance** for that session. `feed` takes `&mut self` on purpose:
state mutation is explicit and there is no hidden interior locking (see *Concurrency*).

`D` is the algorithm-specific **decision info** type (e.g. which tier a classifier picked and why).
It is how a caller that statically knows its algorithm gets typed, first-class decision metadata.
The config-driven path erases `D`; see *Construction*.

## The Data Model

### Neutral request and response

```rust
/// Provider- and transport-neutral request. The host converts to/from provider wire types.
pub struct AgentRequest {
    pub prompt: String,          // POC is single-prompt; see Open Questions for message lists
    pub model: String,
    // future: params (temperature, max_tokens…), tool schema, structured content
}

/// Provider-neutral response.
pub struct AgentResponse {
    pub completion: String,
    // future: usage (token counts), finish reason, tool calls, latency
}
```

These are deliberately minimal in the POC. The design intent is that they carry *enough* to route
and to account for cost — notably token usage on the response — without importing any provider's
schema. Losslessness against a specific provider is the host adapter's concern, not the library's.

### Input taxonomy — more than requests

Optimization is fed a **tagged union of inputs**, not just requests. This is what lets an algorithm
route on agentic-stack information rather than the prompt alone:

```rust
pub enum OptInput {
    /// The inbound request to optimize.
    Request(AgentRequest),
    /// A model response the host produced for a prior ModelInference decision.
    Response(AgentResponse),
    /// A discrete signal/event from the agent or inference stack.
    Signal(Signal),
    /// Free-form correlation/metadata attached out of band.
    Metadata(BTreeMap<String, String>),
}
```

`Signal` is the extension point that realizes "route from events and information from the agentic
inference stack." It is an open, versioned enum of things a host can observe:

```rust
pub enum Signal {
    ToolCallStarted { tool: String, arguments: Option<String> },
    ToolCallCompleted { tool: String, ok: bool, latency_ms: u64 },
    TaskStarted { task: String },
    TaskCompleted { task: String, ok: bool },
    PlanStep { index: u32, description: String },
    RetrievalResult { hits: u32 },
    Budget { spent_tokens: u64, remaining_tokens: Option<u64> },
    Telemetry { key: String, value: f64 }, // e.g. upstream queue depth, replica health
    // …host-defined additions are additive and never break the loop
}
```

An algorithm ignores signals it does not understand (the default `feed` is a no-op), so hosts can
emit rich telemetry without every algorithm having to handle it. A cost-aware router consumes
`Budget`; a cascade escalates on `ToolCallCompleted { ok: false }`; a plain random router consumes
none of them.

### Decision

```rust
pub enum Decision<D> {
    /// Perform these model calls, then feed the responses back and optimize again.
    ModelInference(OptimizerResponse<D>),
    /// Done. Hand control back to the calling agent.
    Return,
}

pub struct OptimizerResponse<D> {
    pub requests: Vec<AgentRequest>,          // one or more calls the host should make
    pub enrichment_data: Vec<EnrichmentData>, // per-request correlation to propagate
    pub decision_reasoning: Option<String>,   // human-readable "why" (logs, traces)
    pub decision_info: Option<D>,             // typed, machine-readable "why" (metrics, eval)
}
```

`decision_reasoning` and `decision_info` are not afterthoughts — they are the **explainability
contract**. Every decision can say, in prose and in typed form, why it was made. Observability and
offline evaluation are built directly on this (see *Observability*).

### Correlation — session, agent, tool, task

```rust
pub struct EnrichmentData {
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub task_id: Option<String>,
    pub tool_id: Option<String>,        // add: complete the session/agent/tool/task set
    pub correlation_id: Option<String>, // trace/request id that ties everything together
    pub extra_metadata: Option<BTreeMap<String, String>>,
}
```

Every `feed` carries `EnrichmentData`, and every emitted request carries its own. This is the spine
of both routing (an algorithm can key state or policy on agent/task/tool) and observability (every
span and metric is tagged with these ids). The correlation set is deliberately fixed at four
first-class dimensions — session, agent, task, tool — plus an opaque `correlation_id` for external
trace joins and an `extra_metadata` escape hatch.

## Optimization Algorithms As Traits

An algorithm is "simple" by construction: implement `OptAlgorithm` (the factory) and `Optimizer`
(the state machine). No registration ceremony, no base class, no framework. The reference
implementations set the pattern:

- **`RandomRouterAlgorithm`** (`src/rand.rs`) — weighted random selection over N targets. One round:
  buffer request → draw a weighted target → rewrite `model` → `ModelInference` → after the response
  is fed, `Return`. Decision info records the draw, total weight, and selected model. This
  generalizes the strong/weak `RandomRoutingProfile` to N targets.

- **`LlmClassifierAlgorithm`** (`src/llm_class.rs`) — an LLM-driven classifier. Multiple rounds:
  emit a classifier model call → parse the score from its response → route to a strong or weak model
  by threshold → `ModelInference` → after that response, `Return`. Its phase machine
  (`AwaitingRequest → Classify → AwaitingScore → Route → AwaitingResponse → Done`) is the canonical
  example of why the loop must support the host calling back in for more than one model inference.

New algorithms to expect on this surface: latency/health-aware routing (consumes `Telemetry`),
cost-budget routing (consumes `Budget`), cascade/escalation (consumes `Response` quality and
`ToolCallCompleted`), speculative/draft-then-verify, and semantic caching.

## Statefulness, Lifecycle, and Concurrency

- **One optimizer per session.** `OptAlgorithm::optimizer()` mints a fresh instance; that instance
  lives for the session's optimization lifecycle and is discarded at the end. Algorithm-level config
  is shared and immutable; per-session mutable state lives only in the instance.
- **Asynchronous, serialized feeds.** Inputs arrive asynchronously (a response completes, a tool
  fires, telemetry ticks), but `feed`/`optimize` take `&mut self` and are therefore **not**
  internally synchronized. The intended pattern is one **per-session task/mailbox** owned by the
  host: async inputs are enqueued, and the owning task applies them to the optimizer serially. This
  keeps algorithm code single-threaded and simple, and pushes fan-in concurrency to the host, where
  it belongs.
- **Cross-session concurrency** is trivially safe: separate sessions are separate instances with no
  shared mutable state. `Send + Sync` bounds allow moving an instance between threads/tasks, but not
  concurrent aliased mutation.
- **Cross-session policy** (global load, fleet health, org-wide budgets) is *not* hidden shared
  state. It enters a session as fed `Signal`s (e.g. `Telemetry`, `Budget`), keeping every decision a
  pure function of that instance's fed history.

## Observability

`libsy` is meant to run in production **and** to be the substrate for research, evaluation, and
benchmarking. Those need the same data, so observability is a core feature, not a bolt-on.

- **Spans.** Each `optimize()` opens a span; each model call the host makes for a `ModelInference`
  decision is a child span. Spans are tagged with the full correlation set (session/agent/task/tool
  /correlation id), the algorithm name, and the round index. `libsy` emits via a `tracing`-style
  facade so the host's collector (OpenTelemetry, etc.) is the sink.
- **Metrics, tagged with those spans.** The metrics of interest — **token counts** (prompt/
  completion), **timings** (per model call, per optimize round, end-to-end session), decision counts
  by selected model/tier, and **failures** — are emitted with the same correlation tags. Because
  token usage rides back on `AgentResponse`/`Signal::Budget`, cost accounting is a fed input, not a
  guess.
- **Failures are tagged, not swallowed.** An error from `feed`/`optimize`, or a failed model call
  reported back via `Signal::ToolCallCompleted`/response, is recorded on the span with cause and
  correlation, so failure rates are queryable per algorithm/agent/task.
- **Explainability as data.** `decision_reasoning` (prose) and `decision_info` (typed) are logged on
  every decision. In production they explain a route; in research they *are* the dataset — win-rate,
  cost delta, and latency delta per decision are computable from the recorded `decision_info` plus
  the fed outcomes.
- **Evaluation modes.** The same loop supports (a) **shadow/dry-run**, where `ModelInference`
  decisions are logged but the host executes only a baseline, and (b) **A/B**, expressed directly as
  a `RandomRouterAlgorithm` over candidates. No separate benchmarking harness is required — a
  benchmark is a batch of sessions with recording turned on.

The library keeps the *facade* thin (a metrics/trace sink trait) so it adds no heavy telemetry
dependency to hosts that do not want one; a no-op sink compiles the instrumentation away.

## Construction: Specific Implementations vs Config-Driven

Two supported entry points, matching the two audiences:

**1. Instantiate a concrete algorithm (code path).** The caller statically knows the algorithm and
gets the typed `D`:

```rust
let algo = RandomRouterAlgorithm { models, rng_seed: None };
let mut opt = algo.optimizer(); // Optimizer<RandomRoutingDecision>
```

**2. Build a `SwitchyardRouter` / `SwitchyardOptimizer` from config (integration path).** A host that
selects the algorithm from a YAML/JSON/env config builds a dynamically-dispatched optimizer:

```rust
let config: SwitchyardConfig = load()?;       // { "algorithm": "llm-classifier", …params }
let router = SwitchyardOptimizer::from_config(&config, &registry)?;
let mut opt = router.optimizer();              // erased decision info (see below)
```

`SwitchyardOptimizer` (a routing-focused alias, `SwitchyardRouter`) is a thin façade over a
**registry** of named algorithm builders:

```rust
pub trait AlgorithmBuilder: Send + Sync {
    fn name(&self) -> &str;                              // e.g. "random", "llm-classifier"
    fn build(&self, params: &Value) -> Result<Box<dyn OptAlgorithm<DynDecision>>, ConfigError>;
}
```

Each reference algorithm registers a builder; hosts can register their own. Config selects a builder
by `name` and hands it its params.

**The generic-`D` erasure decision.** The typed code path keeps `Optimizer<D>`. The config path
cannot — a registry holds heterogeneous algorithms with different `D`, so it standardizes on an
**erased** decision info: `DynDecision = Option<Box<dyn DecisionInfo>>`, where

```rust
pub trait DecisionInfo: Debug + erased_serde::Serialize + Send + Sync {}
```

Erased-but-serializable preserves the observability contract (reasoning and typed info still land in
logs/metrics as structured data) while allowing one uniform `Box<dyn Optimizer<DynDecision>>`. A
typed embedder pays nothing for this; a config-driven embedder trades the concrete `D` for a
serializable trait object. This tension is the single most important construction decision and is
called out here so it is not rediscovered later.

## Integration Patterns

**Routing proxy (e.g. Switchyard).** On request in: convert wire → `AgentRequest`, `feed(Request)`,
drive the loop, perform each `ModelInference` call over the proxy's existing backend transport, feed
responses, on `Return` translate the final response back to the client protocol. Routing decisions
and token/latency metrics flow into the proxy's existing telemetry via the sink.

**Inference platform.** The scheduler owns replicas and health. It feeds `Telemetry` signals (queue
depth, replica health) alongside the request; a latency/health-aware algorithm rewrites `model`/
target accordingly. The platform's dispatcher executes `ModelInference` on the chosen replica.

**Agentic system.** The agent runtime feeds `ToolCallCompleted`, `TaskStarted/Completed`, and
`PlanStep` signals as it runs. A cascade algorithm escalates to a stronger model after a failed tool
call or a stalled task; a budget algorithm downshifts as `Budget.remaining_tokens` falls. The agent
performs the model calls it is told to and continues its loop on `Return`.

## Error Handling

- `feed`/`optimize` return `Result<_, OptError>` (the POC uses `Box<dyn Error>`; the design
  standardizes on a typed `OptError` enum for matchability across the FFI/observability boundary).
- Illegal call sequences (optimize-before-feed, response-out-of-phase) are typed errors, not panics.
  Algorithms never `.expect()` (repo Rust rule).
- **Fail-open by default for routing.** Where a decision cannot be made safely (e.g. an unparseable
  classifier score), the reference algorithms keep traffic flowing on a safe tier rather than
  erroring — the LLM classifier defaults to the strong model. This is a per-algorithm policy, and it
  is recorded in `decision_reasoning` so fail-open events are observable, not silent.
- Context-window and upstream 4xx handling stays with the host (it owns the transport); the host may
  surface such conditions back as `OptInput` so a future algorithm can react (e.g. evict-and-retry).

## Key Design Decisions and Tradeoffs

- **Ask-don't-call.** `ModelInference` describes calls for the host to make instead of `libsy`
  making them. Cost: an extra feed/optimize round trip per model call. Benefit: total transport and
  provider neutrality, testability without network, and a uniform loop for 1..N rounds.
- **`&mut self` over interior mutability.** Simpler, allocation-free algorithm code and clear state
  ownership, at the cost of pushing feed serialization to the host mailbox. Chosen because
  algorithms should be trivial to write and reason about.
- **Typed `D` + erased `DynDecision`.** Keep typed decision info for code users; erase (but keep
  serializable) for config users. Avoids forcing every embedder into `dyn Any` while still allowing
  one dynamic optimizer type.
- **Open `Signal` enum with no-op default `feed`.** Hosts emit rich telemetry; algorithms opt in.
  Additive signals never break existing algorithms.
- **Fixed four-dimension correlation.** Session/agent/task/tool as first-class fields (plus opaque
  trace id) rather than a bag of strings, because these four are what production routing and
  evaluation actually slice by.

## Reference Walkthroughs

**Weighted random (single round).** feed `Request` → `optimize` draws a weighted target, rewrites
`model`, returns `ModelInference([req'])` with `RandomRoutingDecision { selected_model, draw,
total_weight }` → host calls the model, feeds `Response` → `optimize` returns `Return`.

**LLM classifier (multi-round).** feed `Request` → `optimize` returns `ModelInference([classifier
call])` → host runs classifier, feeds its `Response` (a score) → `optimize` parses the score,
routes strong/weak, returns `ModelInference([routed call])` with `ClassifierRoutingDecision { score,
threshold, tier, selected_model }` → host runs routed call, feeds `Response` → `optimize` returns
`Return`. The host code is identical to the single-round case; only the number of loop iterations
differs.

## Current POC State and Gaps

The code in `src/` is a proof of concept and diverges from this design in known ways:

- **Naming is mid-rename and inconsistent.** `src/lib.rs` renamed `ChatRequest`/`ChatResponse` to
  `AgentApitRequest` (a typo for `AgentApiRequest`) / `AgentApiResponse`, but `rand.rs` and
  `llm_class.rs` still reference `ChatRequest`, and `EnrichementData` is misspelled. The crate does
  not currently compile as a result. This document uses the corrected, canonical names
  (`AgentRequest`/`AgentResponse`, `EnrichmentData`, `Optimizer`/`OptAlgorithm`); reconciling the
  source to one coherent naming is the first cleanup.
- **`OptInput` has no `Signal` variant yet.** Today it is `Request`/`Response`/`Metadata` only, and
  `Response` currently wraps a request-shaped struct. The event/signal taxonomy above is designed,
  not built.
- **`EnrichmentData` lacks `tool_id`.** The tool correlation dimension is proposed here.
- **No observability layer.** Spans, the metrics/trace sink facade, token/timing/failure tagging,
  and the shadow/A-B modes are designed but not implemented. `decision_reasoning`/`decision_info`
  already exist and are the hook they build on.
- **No config-driven construction.** `SwitchyardOptimizer`/`SwitchyardRouter`, the
  `AlgorithmBuilder` registry, `SwitchyardConfig`, and `DynDecision`/`DecisionInfo` erasure do not
  exist yet; only direct instantiation of the two reference algorithms does.
- **Error type is `Box<dyn Error>`.** The design calls for a typed `OptError`.
- **Request is single-prompt.** No message list, params, or tool schema yet.

## Open Questions

- **Message list vs single prompt.** Real routing often needs system/history separation and tool
  schemas. How much of a message model does `libsy` adopt without drifting toward a provider schema?
- **Where does token usage come from before the call?** Cost-aware pre-routing needs an estimate;
  is tokenization a host-provided `Signal`, or does `libsy` grow a pluggable estimator trait?
- **Batching / multi-request rounds.** `OptimizerResponse.requests` is a `Vec`, but the reference
  algorithms and the host loop assume one request per round. Do we commit to parallel multi-call
  rounds (fan-out/speculative), and if so, how are their responses fed back in order?
- **Cancellation and timeouts.** How does a host abandon an in-flight session, and how is that
  surfaced to the optimizer for cleanup and accounting?
- **Cross-language surface.** If Python/PyO3 bindings follow (as elsewhere in Switchyard), the
  `DynDecision` erasure and `OptError` typing become the binding boundary — worth fixing before
  bindings, not after.

## Staleness Risks

This document mirrors `src/lib.rs`, `src/rand.rs`, and `src/llm_class.rs`. If the traits, the input
taxonomy, the correlation fields, or the two reference algorithms change, update the corresponding
section here — especially *Current POC State and Gaps*, which is only useful while it is accurate.
When the naming reconciliation lands, delete the naming-divergence note rather than letting it rot.

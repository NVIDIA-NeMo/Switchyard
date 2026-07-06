# Switchyard-Lib Design

Audience: contributors building `libsy`, and integrators embedding it in an inference platform, an
agentic system, or a routing proxy.

Prerequisites: familiarity with asynchronous programming, LLM request/response shapes, and the idea
of "routing" a request to one of several models.

Status: design draft. The interfaces in `src/lib.rs` and the two reference algorithms in
`src/rand.rs` and `src/llm_class.rs` are a proof of concept. This document describes the intended
design and marks where the current POC diverges from it (see *Current POC State and Gaps*).

Language note: this document centers on the **Rust crate** in `src/` — that is the design of record,
and the sketches below are the actual (intended) Rust traits. The underlying model is
language-independent — "trait" reads as *interface/contract* — but the requirements and the design
are worked out against the Rust POC, not any binding or port.

## Summary

`libsy` (Switchyard-Lib) is a lightweight library for **multi-LLM agent optimization**, of which
**routing** is the first and most important case. It sits at the point in a system where a request
is *about to* be sent to a model and decides — statefully, using more than the request alone —
*which* model(s) to call, *how* to rewrite the call, or whether to skip it.

The library is built around one small pair of interfaces:

```text
OptAlgorithm  --optimizer()-->  Optimizer  --feed()/optimize()-->  Decision
   (factory,                     (per-session
    from config or code)          stateful instance)
```

An `Optimizer` is a **stateful, per-session** state machine. A host **feeds** it inputs — the
inbound request, model responses, and signals from the agentic/inference stack — and asks it to
**optimize**, which returns a `Decision`: either "make these model calls and feed me the results"
(`ModelInference`) or "you're done, hand control back to the agent" (`Return`). A one-shot router
and a multi-round LLM classifier are the *same* control loop with different internal state.

Everything else in the library exists to make that loop **embeddable and production-grade**:
provider/transport neutrality so it drops into any stack, correlation so decisions attribute to a
session/agent/task/tool, and observability so every decision, token count, timing, and failure is a
tagged span you can drive dashboards, evaluations, and benchmarks from.

## Requirements

This design targets a specific set of requirements. Each is addressed by the section named:

1. A lightweight library for multi-LLM agent optimization, including routing → *Summary*, *Core
   Mental Model*.
2. Embeddable in inference platforms, agentic systems, or routing proxies → *Scope and Positioning*,
   *Integration Patterns*.
3. Agnostic of provider API and transport → *Scope and Positioning*, *The Data Model*.
4. Optimization algorithms (e.g. routers) implemented as simple traits → *Optimization Algorithms As
   Traits*.
5. Integrators can instantiate a specific implementation, or build a `SwitchyardRouter` /
   `SwitchyardOptimizer` from config → *Construction*.
6. Fully observable for production **and** research/evaluation/benchmarking; token counts, timings,
   etc. tagged with spans/failures → *Observability*.
7. Optimization driven not only by request objects but by events and information from the agentic
   inference stack → *Input Taxonomy*.
8. Session, agent, tool, and task correlation → *Correlation*.
9. Each algorithm is stateful and asynchronously fed inputs (request, response, signals) →
   *Statefulness, Lifecycle, and Concurrency*.

## Scope and Positioning

`libsy` is a **library, not a service**. It carries no server, no provider SDK, and no transport. It
is designed to be embedded in three kinds of host:

- **Inference platforms** — the serving layer decides, per request, which backend/model/replica to
  use. `libsy` is the decision core; the platform owns the sockets and the model calls.
- **Agentic systems** — an agent framework wants tier selection, cascade/escalation, or
  cost/latency optimization *inside* its reasoning loop, informed by tool calls and task state, not
  just the prompt.
- **Routing proxies** — a proxy terminates a client protocol and needs to route or rewrite before
  forwarding. `libsy` is the routing brain behind the proxy's I/O.

Two properties make this possible, and both are hard requirements:

- **Provider-API agnostic.** `libsy` reasons over neutral request/response types, never over a
  specific provider's schema. The host converts at the edge.
- **Transport agnostic.** `libsy` never performs a model call. A `ModelInference` decision *asks*
  the host to make calls; the host uses whatever transport it already owns and feeds results back.
  This keeps `libsy` free of runtime, I/O, and dependency lock-in — the central enabler of "drops
  into any stack."

## Goals and Non-Goals

Goals:

- One small, stable interface for optimization algorithms, with routing as the primary case.
- Statefulness as a first-class property: algorithms accumulate context across a session.
- Inputs richer than the request: responses **and** events/signals from the agentic stack.
- First-class correlation: session, agent, task, tool.
- Full observability suitable for both production and research/evaluation/benchmarking.
- Two ways to construct: instantiate a concrete algorithm, or build a `SwitchyardRouter` /
  `SwitchyardOptimizer` from config.

Non-Goals:

- Not a proxy, gateway, or server; no listeners, no protocol termination.
- No provider SDKs and no network client. Model calls are the host's job.
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

### The two interfaces

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
The `Optimizer` is the **stateful instance** for that session. This two-part split (immutable
algorithm config + per-session mutable instance) is what makes "each algorithm is stateful" safe:
sessions never share mutable state.

`D` is the algorithm-specific **decision metadata** type (e.g. which tier a classifier picked and
why). A caller that statically knows its algorithm gets typed, first-class decision metadata; the
config-driven path standardizes it to a serializable structured value (see *Construction*).

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
This neutrality is what satisfies requirement 3.

### Input Taxonomy — more than requests

Optimization is fed a **tagged union of inputs**, not just requests. This is what lets an algorithm
optimize on agentic-stack information rather than the prompt alone (requirement 7):

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

`Signal` is the extension point. It is an open, versioned enum of things a host can observe:

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
contract**. Every decision can say, in prose and in structured form, *why* it was made.
Observability and offline evaluation are built directly on this.

### Correlation — session, agent, tool, task

```rust
pub struct EnrichmentData {
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub task_id: Option<String>,
    pub tool_id: Option<String>,        // add: complete the session/agent/tool/task set
    pub correlation_id: Option<String>, // external trace/request id that joins everything
    pub extra_metadata: Option<BTreeMap<String, String>>,
}
```

Every `feed` carries `EnrichmentData`, and every emitted request carries its own. This is the spine
of both optimization (an algorithm can key state or policy on agent/task/tool) and observability
(every span and metric is tagged with these ids). The correlation set is deliberately fixed at four
first-class dimensions — session, agent, task, tool (requirement 8) — plus an opaque
`correlation_id` for external trace joins and an `extra_metadata` escape hatch.

## Optimization Algorithms As Traits

An algorithm is "simple" by construction (requirement 4): implement `OptAlgorithm` (the factory) and
`Optimizer` (the state machine). No registration ceremony, no base class, no framework. The
reference implementations set the pattern:

- **`RandomRouterAlgorithm`** (`src/rand.rs`) — weighted random selection over N targets. One round:
  buffer request → draw a weighted target → rewrite `model` → `ModelInference` → after the response
  is fed, `Return`. Decision metadata records the draw, total weight, and selected model.

- **`LlmClassifierAlgorithm`** (`src/llm_class.rs`) — an LLM-driven classifier. Multiple rounds:
  emit a classifier model call → parse the score from its response → route to a strong or weak model
  by threshold → `ModelInference` → after that response, `Return`. Its phase machine
  (`AwaitingRequest → Classify → AwaitingScore → Route → AwaitingResponse → Done`) is the canonical
  example of why the loop must support the host calling back in for more than one model inference.

New algorithms to expect on this surface: latency/health-aware routing (consumes `Telemetry`),
cost-budget routing (consumes `Budget`), cascade/escalation (consumes `Response` quality and
`ToolCallCompleted`), speculative/draft-then-verify, and semantic caching. Each is "just another
trait implementation" against the same loop.

## Statefulness, Lifecycle, and Concurrency

- **One optimizer per session.** `OptAlgorithm::optimizer()` mints a fresh instance; that instance
  lives for the session's optimization lifecycle and is discarded at the end. Algorithm-level config
  is shared and immutable; per-session mutable state lives only in the instance.
- **Asynchronous, serialized feeds.** Inputs arrive asynchronously (a response completes, a tool
  fires, telemetry ticks). The instance is not internally synchronized; the intended pattern is one
  **per-session queue/task** owned by the host that applies fed inputs to the optimizer serially.
  This keeps algorithm code single-threaded and simple, and pushes fan-in concurrency to the host,
  where it belongs. (This satisfies requirement 9: stateful, asynchronously fed.)
- **Cross-session concurrency** is trivially safe: separate sessions are separate instances with no
  shared mutable state.
- **Cross-session policy** (global load, fleet health, org-wide budgets) is *not* hidden shared
  state. It enters a session as fed `Signal`s (e.g. `Telemetry`, `Budget`), keeping every decision a
  function of that instance's fed history — which is also what makes decisions reproducible for
  evaluation.

## Observability

`libsy` must run in production **and** be the substrate for research, evaluation, and benchmarking
(requirement 6). Those need the same data, so observability is a core feature, not a bolt-on.

- **Spans.** Each `optimize()` opens a span; each model call the host makes for a `ModelInference`
  decision is a child span. Spans are tagged with the full correlation set (session/agent/task/tool
  + external id), the algorithm name, and the round index.
- **Metrics, tagged with those spans.** The metrics of interest — **token counts** (prompt/
  completion), **timings** (per model call, per optimize round, end-to-end session), decision counts
  by selected model/tier, and **failures** — are emitted with the same correlation tags. Because
  token usage rides back on `AgentResponse`/`Signal::Budget`, cost accounting is a fed input, not a
  guess.
- **Failures are tagged, not swallowed.** An error from `feed`/`optimize`, or a failed model call
  reported back via a signal/response, is recorded on the span with cause and correlation, so
  failure rates are queryable per algorithm/agent/task.
- **Explainability as data.** `decision_reasoning` (prose) and `decision_info` (structured) are
  recorded on every decision. In production they explain a route; in research they *are* the
  dataset — win-rate, cost delta, and latency delta per decision are computable from the recorded
  decision metadata plus the fed outcomes.
- **Evaluation modes.** The same loop supports (a) **shadow/dry-run**, where `ModelInference`
  decisions are logged but the host executes only a baseline, and (b) **A/B**, expressed directly as
  a weighted random algorithm over candidates. A benchmark is just a batch of sessions with
  recording on — no separate harness required.

Observability is exposed through a **thin sink abstraction** (a metrics/trace sink the host
implements), so `libsy` imposes no particular telemetry backend and a no-op sink compiles the
instrumentation away for hosts that do not want it.

## Construction

Two supported entry points, matching the two integrator styles (requirement 5):

**1. Instantiate a concrete algorithm (code path).** The integrator statically knows the algorithm
and gets typed decision metadata:

```rust
let algo = RandomRouterAlgorithm { models, rng_seed: None };
let mut opt = algo.optimizer(); // Optimizer<RandomRoutingDecision>
```

**2. Build a `SwitchyardRouter` / `SwitchyardOptimizer` from config (integration path).** A host that
selects the algorithm from a config file/env builds a dynamically-dispatched optimizer:

```rust
let config: SwitchyardConfig = load()?;      // { "algorithm": "llm-classifier", …params }
let router = SwitchyardOptimizer::from_config(&config, &registry)?;
let mut opt = router.optimizer();
```

`SwitchyardOptimizer` (aliased `SwitchyardRouter` for the routing-focused case) is a thin façade over
a **registry** of named algorithm builders. Each reference algorithm registers a builder under a
name; hosts register their own. Config selects a builder by name and hands it its parameters.

**Typed vs config-erased decision metadata.** The typed code path preserves the algorithm's concrete
decision-metadata type. The config path cannot — a registry holds heterogeneous algorithms — so it
standardizes on a **serializable structured value** for decision metadata (a self-describing,
loggable form) rather than a concrete per-algorithm type. This is the one real cost of the
config-driven path: you trade the concrete metadata type for a serializable one, while the
observability contract (reasoning + structured info in logs/metrics) is fully preserved either way.
This tradeoff is called out here so it is a conscious design choice, not a later surprise.

## Integration Patterns

**Routing proxy.** On request in: convert wire → `AgentRequest`, `feed(Request)`, drive the loop,
perform each `ModelInference` call over the proxy's existing backend transport, feed responses, and
on `Return` translate the final response back to the client protocol. Decisions and token/latency
metrics flow into the proxy's existing telemetry via the sink.

**Inference platform.** The scheduler owns replicas and health. It feeds `Telemetry` signals (queue
depth, replica health) alongside the request; a latency/health-aware algorithm rewrites the target
accordingly. The platform's dispatcher executes `ModelInference` on the chosen replica.

**Agentic system.** The agent runtime feeds `ToolCallCompleted`, `TaskStarted/Completed`, and
`PlanStep` signals as it runs. A cascade algorithm escalates to a stronger model after a failed tool
call or a stalled task; a budget algorithm downshifts as `Budget.remaining_tokens` falls. The agent
performs the model calls it is told to and continues its loop on `Return`.

## Error Handling

- `feed`/`optimize` return a result with a typed error, so illegal call sequences
  (optimize-before-feed, response-out-of-phase) are matchable errors, not panics.
- **Fail-open by default for routing.** Where a decision cannot be made safely (e.g. an unparseable
  classifier score), the reference algorithms keep traffic flowing on a safe tier rather than
  erroring — the LLM classifier defaults to the strong model. This is a per-algorithm policy, and it
  is recorded in `decision_reasoning` so fail-open events are observable, not silent.
- Transport-level conditions (context-window overflow, upstream 4xx) stay with the host, which owns
  the transport; the host may surface them back as `OptInput` so a future algorithm can react (e.g.
  evict-and-retry).

## Key Design Decisions and Tradeoffs

- **Ask-don't-call.** `ModelInference` describes calls for the host to make instead of `libsy`
  making them. Cost: an extra feed/optimize round trip per model call. Benefit: total transport and
  provider neutrality, testability without I/O, and a uniform loop for 1..N rounds. This is the
  decision that makes the library embeddable everywhere.
- **Factory + per-session instance.** Immutable algorithm config mints a fresh mutable optimizer per
  session — the simplest safe model for stateful algorithms under concurrency.
- **Open `Signal` union with no-op default feed.** Hosts emit rich telemetry; algorithms opt in.
  Additive signals never break existing algorithms.
- **Fixed four-dimension correlation.** Session/agent/task/tool as first-class fields (plus opaque
  external id) rather than a bag of strings, because these four are what production routing and
  evaluation actually slice by.
- **Typed vs config-erased decision metadata.** Keep the concrete type for code users; standardize on
  a serializable value for config users. Preserves observability for both.

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

- **Naming is mid-rename and inconsistent.** `src/lib.rs` renamed the request/response types to
  `AgentApitRequest` (a typo for `AgentApiRequest`) / `AgentApiResponse`, but `rand.rs` and
  `llm_class.rs` still reference the old `ChatRequest`, and `EnrichementData` is misspelled — so the
  crate does not currently compile. This document uses the corrected, canonical names; reconciling
  the source to one coherent naming is the first cleanup.
- **`OptInput` has no `Signal` variant yet.** Today it is `Request`/`Response`/`Metadata` only, and
  `Response` currently wraps a request-shaped struct. The event/signal taxonomy above is designed,
  not built — and it is what realizes requirement 7.
- **`EnrichmentData` lacks `tool_id`.** The tool correlation dimension (requirement 8) is proposed
  here.
- **No observability layer.** Spans, the sink abstraction, token/timing/failure tagging, and the
  shadow/A-B modes are designed but not implemented. `decision_reasoning`/`decision_info` already
  exist and are the hook they build on.
- **No config-driven construction.** `SwitchyardOptimizer`/`SwitchyardRouter`, the builder registry,
  the config schema, and decision-metadata erasure do not exist yet; only direct instantiation of
  the two reference algorithms does.
- **Request is single-prompt.** No message list, params, or tool schema yet.

## Open Questions

- **Message list vs single prompt.** Real routing often needs system/history separation and tool
  schemas. How much of a message model does `libsy` adopt without drifting toward a provider schema?
- **Where does token usage come from before the call?** Cost-aware pre-routing needs an estimate;
  is tokenization a host-provided `Signal`, or does `libsy` grow a pluggable estimator interface?
- **Batching / multi-request rounds.** `OptimizerResponse.requests` is a list, but the reference
  algorithms and the host loop assume one request per round. Do we commit to parallel multi-call
  rounds (fan-out/speculative), and if so, how are their responses fed back in order?
- **Cancellation and timeouts.** How does a host abandon an in-flight session, and how is that
  surfaced to the optimizer for cleanup and accounting?

## Staleness Risks

This document mirrors `src/lib.rs`, `src/rand.rs`, and `src/llm_class.rs`. If the interfaces, the
input taxonomy, the correlation fields, or the two reference algorithms change, update the
corresponding section here — especially *Current POC State and Gaps*, which is only useful while it
is accurate. When the naming reconciliation lands, delete the naming-divergence note rather than
letting it rot.

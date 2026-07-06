# Switchyard-Lib Design

Status: design draft. The Rust crate in `src/` (`lib.rs` traits, `rand.rs`, `llm_class.rs`) is the
proof of concept and the design of record; this doc describes the intended shape and marks where the
POC still diverges (see *POC Gaps*). The model is language-independent — "trait" means
interface/contract — but it is worked out against the Rust code, not any binding or port.

## Summary

`libsy` is a lightweight **library** for multi-LLM agent optimization, with **routing** as the first
case. It sits where a request is about to hit a model and decides — statefully, using more than the
request — which model(s) to call, how to rewrite the call, or whether to skip it.

It is not a service: no server, no provider SDK, no transport. It is **provider- and
transport-agnostic** so it embeds in inference platforms, agentic systems, and routing proxies.
Model calls are the host's job (see *The Loop*).

## The Loop

The whole library is one pair of traits and one loop:

```text
OptAlgorithm --optimizer()--> Optimizer --feed()/optimize()--> Decision
  (config/factory)            (per-session, stateful)
```

The host feeds the optimizer inputs and calls `optimize()`, which returns either "make these model
calls and feed me the results" (`ModelInference`) or "done, return to the agent" (`Return`):

```text
feed(request)
loop:
    match optimize():
        ModelInference(reqs) -> for r in reqs { feed(host_call(r)) }   // host owns the transport
        Return               -> break
```

A one-round router and a multi-round LLM classifier are the *same* loop; the host's code does not
change when the algorithm's round count does.

```rust
#[async_trait]
pub trait Optimizer<D>: Send + Sync {
    async fn feed(&mut self, input: OptInput, enrichment: EnrichmentData) -> Result<(), OptError>;
    async fn optimize(&mut self) -> Result<Decision<D>, OptError>;
}

pub trait OptAlgorithm<D>: Send + Sync {
    fn optimizer(&self) -> Box<dyn Optimizer<D>>; // fresh instance per session
}
```

`OptAlgorithm` is immutable, shareable config; the `Optimizer` holds the per-session mutable state.
`D` is the algorithm's typed **decision metadata** (see *Construction* for the config path).

## Data Model

Neutral request/response — the host converts to/from provider wire types at the edge:

```rust
pub struct AgentRequest  { pub prompt: String, pub model: String }  // +params/tools later
pub struct AgentResponse { pub completion: String }                 // +token usage, latency later
```

Inputs are a tagged union, so optimization is driven by more than the request — responses **and**
events from the agentic stack:

```rust
pub enum OptInput {
    Request(AgentRequest),
    Response(AgentResponse),
    Signal(Signal),                     // tool/task/plan/budget/telemetry events
    Metadata(BTreeMap<String, String>),
}

pub enum Signal {                       // open, versioned; algorithms opt in, ignore the rest
    ToolCallCompleted { tool: String, ok: bool, latency_ms: u64 },
    TaskStarted { task: String }, TaskCompleted { task: String, ok: bool },
    Budget { spent_tokens: u64, remaining_tokens: Option<u64> },
    Telemetry { key: String, value: f64 },   // e.g. replica health, queue depth
    // …additive; a new signal never breaks an existing algorithm
}
```

A decision names the calls to make plus its own explanation:

```rust
pub enum Decision<D> { ModelInference(OptimizerResponse<D>), Return }

pub struct OptimizerResponse<D> {
    pub requests: Vec<AgentRequest>,
    pub enrichment_data: Vec<EnrichmentData>,
    pub decision_reasoning: Option<String>,   // human-readable "why" (traces)
    pub decision_info: Option<D>,             // structured "why" (metrics, eval)
}
```

Correlation is carried on every feed and every emitted request — the spine of both policy and
observability:

```rust
pub struct EnrichmentData {
    pub session_id: Option<String>, pub agent_id: Option<String>,
    pub task_id: Option<String>,    pub tool_id: Option<String>,   // session/agent/task/tool
    pub correlation_id: Option<String>,                            // external trace join
    pub extra_metadata: Option<BTreeMap<String, String>>,
}
```

## Algorithms Are Traits

Implement `OptAlgorithm` + `Optimizer`; no base class, no framework. Reference impls:

- **`RandomRouterAlgorithm`** (`rand.rs`) — weighted random over N targets. One round: draw a target,
  rewrite `model`, return; after the response is fed, `Return`.
- **`LlmClassifierAlgorithm`** (`llm_class.rs`) — multi-round: emit a classifier call, parse its
  score, route strong/weak by threshold, return; after that response, `Return`.

Future algorithms on the same surface: latency/health-aware routing (`Telemetry`), cost-budget
routing (`Budget`), cascade/escalation (`ToolCallCompleted` + response quality), semantic caching.

## Statefulness and Concurrency

- One optimizer per session; discarded at session end. Immutable algorithm config, mutable instance.
- Feeds arrive asynchronously but are applied serially — the host owns a per-session queue/task; the
  instance is not internally synchronized. Separate sessions are separate instances, so cross-session
  use is safe with no shared state.
- Cross-session policy (fleet load, global budgets) enters as fed `Signal`s, not hidden state — so
  every decision is a function of that session's fed history, and reproducible for evaluation.

## Observability

Production and research/benchmarking need the same data, so it is a core feature:

- `optimize()` and each host model call are **spans**, tagged with the correlation set + algorithm
  name + round.
- Metrics on those spans: **token counts, timings, failures**, decisions by model/tier. Token usage
  is a fed input (on `AgentResponse`/`Budget`), not a guess.
- `decision_reasoning`/`decision_info` are recorded per decision: an explanation in production, the
  dataset in research (win-rate / cost / latency deltas). Shadow and A/B modes fall out of the loop
  directly; a benchmark is a batch of recorded sessions.
- Emitted through a thin sink abstraction, so `libsy` mandates no telemetry backend and a no-op sink
  compiles it away.

## Construction

Two entry points:

- **Direct** — instantiate a concrete algorithm; you get typed decision metadata.
  `RandomRouterAlgorithm { models, rng_seed }.optimizer()`.
- **Config** — `SwitchyardOptimizer::from_config(cfg, registry)` (aliased `SwitchyardRouter`) selects
  a named algorithm from a builder registry. Because the registry is heterogeneous, the config path
  standardizes decision metadata to a **serializable structured value** instead of the concrete `D`;
  the observability contract is preserved either way. That erasure is the one cost of the config
  path.

Errors are typed (`OptError`), so illegal sequences (optimize-before-feed, out-of-phase response)
are matchable, not panics. Routing fails **open**: when a decision can't be made safely (e.g. an
unparseable classifier score) the algorithm keeps traffic on a safe tier and records why.

## POC Gaps

The code in `src/` diverges from this design in known ways:

- **Naming is mid-rename and does not compile:** `lib.rs` renamed the request type to
  `AgentApitRequest` (typo) / `AgentApiResponse` while `rand.rs`/`llm_class.rs` still use
  `ChatRequest`, and `EnrichementData` is misspelled. Reconciling to the canonical names above is
  the first cleanup.
- **`OptInput` has no `Signal` variant** (only Request/Response/Metadata), and `EnrichmentData` lacks
  `tool_id` — these realize the event-driven and tool-correlation requirements.
- **No observability layer** and **no config construction** (`SwitchyardOptimizer`/`SwitchyardRouter`,
  registry, metadata erasure) yet; `decision_reasoning`/`decision_info` are the hook they build on.
- Request is single-prompt; a message model, params, and tool schema are open questions.

Keep this section accurate; delete the naming note once the rename lands. Update the sketches here if
the traits in `src/` change.

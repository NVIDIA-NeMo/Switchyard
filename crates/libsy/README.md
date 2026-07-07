# libsy — Switchyard-Lib

A lightweight library for multi-LLM agent optimization, with **routing** as the first case. It
decides — statefully, using more than the request — which model(s) to call and how, and never
performs the call itself ("ask, don't call"), which keeps it provider- and transport-agnostic.
This README shows the scope and shape through code; the narrative design is in
[`DESIGN.md`](DESIGN.md).

## Core: two traits

The whole core is a stateful, per-session state machine plus a factory that mints one per session.

```rust
// Fed inputs (request, model responses, metadata); asked to decide what to do next.
#[async_trait]
pub trait AgentApiOptimizer<D>: Send + Sync {
    async fn feed(&mut self, input: AgentApiOptInput, enrichment: EnrichmentData)
        -> Result<(), Box<dyn Error>>;
    async fn optimize(&mut self) -> Result<Decision<D>, Box<dyn Error>>;
}

// Factory: a fresh optimizer per session. `D` is the algorithm's typed decision metadata.
pub trait AgentApiOptAlgorithm<D>: Send + Sync {
    fn optimizer(&self) -> Box<dyn AgentApiOptimizer<D>>;
}
```

Supporting types (in `lib.rs`):

```rust
pub enum Decision<D> {
    ModelInference(AgentApiOptimizerResponse<D>), // host: make these calls, feed results back
    Return(),                                     // done: hand control back to the agent
}

pub enum AgentApiOptInput {
    Request(AgentApiRequest),
    Response(AgentApiRequest),      // MISSING: reuses the request struct (completion in `prompt`);
                                    //          should carry AgentApiResponse + token usage/latency
    Metadata(BTreeMap<String, String>),
    // MISSING: no `Signal(..)` variant yet for agentic-stack events (tool/task/budget/telemetry)
}

pub struct AgentApiRequest  { pub prompt: String, pub model: String } // MISSING: single-prompt only
pub struct AgentApiResponse { pub completion: String }

pub struct EnrichmentData {         // correlation carried on every feed
    pub session_id: Option<String>, pub agent_id: Option<String>,
    pub task_id: Option<String>,    pub correlation_id: Option<String>,
    pub extra_metadata: Option<BTreeMap<String, String>>,
    // MISSING: no `tool_id` yet
}
```

## Wrapper: a blackbox LLM client

`RoutedClient` runs the `feed`/`optimize` loop for you, so routing becomes a drop-in for an ordinary
client: request in, response out. The one bit of I/O the core does not own is the model call, which
you supply as a `ModelCaller` (or use the built-in HTTP one).

```rust
impl<D> RoutedClient<D> {
    pub fn new(algorithm: Box<dyn AgentApiOptAlgorithm<D>>, caller: Box<dyn ModelCaller>) -> Self;
    // Convenience: default OpenAI-compatible HTTP caller.
    pub fn with_http(algorithm: Box<dyn AgentApiOptAlgorithm<D>>, base_url, api_key) -> Self;

    async fn complete(&self, request: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>>;
    async fn complete_with(&self, request, enrichment) -> Result<AgentApiResponse, Box<dyn Error>>;
}

#[async_trait]
pub trait ModelCaller: Send + Sync {
    async fn call(&self, request: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>>;
}
```

## Implementing a router (LLM classifier, minimized)

A multi-round router: run a classifier model, then route to a strong/weak model by its score. This
is why `optimize` returns `ModelInference` more than once — the host calls back in with each
response. (Full, error-checked version in [`src/llm_class.rs`](src/llm_class.rs); a one-round
weighted-random router is in [`src/rand.rs`](src/rand.rs).)

```rust
#[derive(Clone)]
pub struct LlmClassifier { pub classifier: String, pub strong: String, pub weak: String, pub threshold: f64 }

// `D = ()`: this router attaches no typed decision metadata (kept minimal here).
impl AgentApiOptAlgorithm<()> for LlmClassifier {
    fn optimizer(&self) -> Box<dyn AgentApiOptimizer<()>> {
        Box::new(Session { cfg: self.clone(), phase: Phase::Classify, prompt: None, score: None })
    }
}

enum Phase { Classify, Route, Done }
struct Session { cfg: LlmClassifier, phase: Phase, prompt: Option<String>, score: Option<f64> }

#[async_trait]
impl AgentApiOptimizer<()> for Session {
    async fn feed(&mut self, input: AgentApiOptInput, _e: EnrichmentData) -> Result<(), Box<dyn Error>> {
        match input {
            AgentApiOptInput::Request(r) => self.prompt = Some(r.prompt),
            // Responses arrive in phase order: first the classifier score, then the final answer.
            AgentApiOptInput::Response(r) => match self.phase {
                Phase::Classify => { self.score = r.prompt.trim().parse().ok(); self.phase = Phase::Route; }
                _               => self.phase = Phase::Done,
            },
            _ => {}
        }
        Ok(())
    }

    async fn optimize(&mut self) -> Result<Decision<()>, Box<dyn Error>> {
        let prompt = self.prompt.clone().ok_or("optimize before request was fed")?;
        Ok(match self.phase {
            Phase::Classify => call(&self.cfg.classifier, &format!("score 0..1: {prompt}")),
            Phase::Route => {
                // Fail open: an unparseable score (None) routes to the strong model.
                let strong = self.score.map_or(true, |s| s >= self.cfg.threshold);
                call(if strong { &self.cfg.strong } else { &self.cfg.weak }, &prompt)
            }
            Phase::Done => Decision::Return(),
        })
    }
}

// Emit a single model call for the host to perform.
fn call(model: &str, prompt: &str) -> Decision<()> {
    Decision::ModelInference(AgentApiOptimizerResponse {
        requests: vec![AgentApiRequest { prompt: prompt.into(), model: model.into() }],
        enrichment_data: vec![], decision_reasoning: None, decision_info: None,
    })
}
```

## Using the wrapper (research agent)

The agent owns a `RoutedClient` and nothing routing-specific; routing is one `complete` call. Here
it uses the multi-round LLM classifier, so a single `complete` triggers two model calls (classify,
then route) — invisible to the agent. Runnable: [`examples/research_agent.rs`](examples/research_agent.rs).

```rust
// Configure routing once, in one place. The classifier scores each request with a
// model call, then routes strong/weak — two rounds the agent never sees.
let algorithm = LlmClassifierAlgorithm {
    classifier_model: "classifier/model".into(), strong_model: "strong/model".into(),
    weak_model: "weak/model".into(), threshold: 0.5 };
let client = RoutedClient::new(Box::new(algorithm), Box::new(StubCaller)); // or ::with_http(..)

// ...inside the agent, the ONLY integration point — one call, one response
// (the classifier + routed calls happen under the hood). "auto" is a placeholder
// the router replaces; the agent never learns which model served it.
let answer = client.complete(AgentApiRequest { prompt: step, model: "auto".into() }).await?;
```

Swapping in a one-round `RandomRouterAlgorithm` or `with_http` needs **no** change to the agent.

## Using the core API directly (research agent)

Same trivial agent, but driving `feed`/`optimize` by hand — exactly what the wrapper hides. Reach
for this when you need the raw loop (custom transport per round, inspecting `decision_reasoning`,
interleaving your own signals). Runnable: [`examples/research_agent_core.rs`](examples/research_agent_core.rs).

```rust
let mut optimizer = algorithm.optimizer();          // fresh, isolated state per request
optimizer.feed(AgentApiOptInput::Request(req), EnrichmentData::default()).await?;

let mut last = None;
// One round for weighted-random routing, N for the classifier — the loop is identical.
while let Decision::ModelInference(decision) = optimizer.optimize().await? {
    // decision.decision_reasoning / decision_info explain the route (logging/eval); optional.
    for req in decision.requests {
        let model = req.model.clone();
        let answer = call_model(&req).await;        // the host owns the transport ("ask, don't call")
        // MISSING: Response reuses AgentApiRequest (completion goes in `prompt`).
        optimizer.feed(AgentApiOptInput::Response(AgentApiRequest {
            prompt: answer.completion.clone(), model }), EnrichmentData::default()).await?;
        last = Some(answer);                         // the routed answer; the classifier call is consumed internally
    }
}
```

## Not yet built

Annotated inline above; collected here:

- **`Signal` inputs** for agentic-stack events (tool/task/budget/telemetry) — routing on more than the request.
- **`Response` should carry `AgentApiResponse`** (with token usage/latency), not reuse the request struct.
- **`tool_id`** on `EnrichmentData` to complete session/agent/task/tool correlation.
- **Observability** — spans + a metrics sink for token counts/timings/failures (`decision_reasoning`/`decision_info` are the hook).
- **Config-driven construction** — a `SwitchyardOptimizer`/`SwitchyardRouter` built from config over a builder registry.
- **Typed errors** (`OptError`) instead of `Box<dyn Error>`, and a **message model** (system/history, params, tools) beyond a single prompt.

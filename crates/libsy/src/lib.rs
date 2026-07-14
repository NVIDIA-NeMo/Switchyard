// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # libsy — multi-LLM agent optimization (routing first)
//!
//! `libsy` decides, per request, *how* to serve an LLM call: which model(s) to
//! invoke, in what order, and how to combine the results. Routing is the first
//! and simplest case; the same interfaces also express classifier routing,
//! ensembles, cascades, and other optimizations. The library owns no HTTP client
//! and no provider SDK — it decides, and the host makes (or is asked to make) the
//! actual calls — so it embeds cleanly in a proxy, gateway, or agent runtime.
//!
//! ## The model
//!
//! - An [`Algorithm`] is the optimization *algorithm*. Its
//!   [`create_run_task`](Algorithm::create_run_task) runs once per request
//!   and makes as many model calls as it needs — via [`Driver::call_llm_target`], which look
//!   like ordinary calls — publishes its [`Decision`]s with [`Driver::info`], and
//!   returns the final [`Response`]. The provided
//!   [`run_stream`](Algorithm::run_stream) drives that on its own task and hands
//!   back a stream of [`Step`]s; [`run`](Algorithm::run) runs
//!   it to completion with the targets' default clients.
//! - An [`LlmTarget`] names a routing target by its [`semantic_name`](LlmTarget::semantic_name).
//!   Every call is *offloaded* to the request's stream as a [`Step::CallLlm`]; the
//!   target's [`LlmClient`], if any, rides along as
//!   [`RoutedRequest::default_client`] so the host can serve it by default or
//!   override it (see below).
//!
//! ## Running a request
//!
//! Hold the algorithm as `Arc<dyn Algorithm>` and call one of two provided methods:
//!
//! - [`run`](Algorithm::run) — run to completion, serving each
//!   offloaded call via its [`RoutedRequest::default_client`], and return the decision
//!   trace plus the final [`Response`]. The simplest integration; use it when the
//!   algorithm holds the model clients (it errors if a routed target has no client).
//! - [`run_stream`](Algorithm::run_stream) — return a stream of [`Step`]s. Each
//!   model call is offloaded: the stream yields a [`Step::CallLlm`] carrying a promise;
//!   the host performs the real model call (optionally via the promise's
//!   `default_client`) and fulfills it with [`CallLlmRequest::respond`]. Decisions
//!   arrive as [`Step::Decision`] as the algorithm makes them, and the run ends with a
//!   [`Step::ReturnToAgent`] carrying the final response. The step stream is bounded,
//!   so pulling it paces the algorithm one step at a time — an "ask, don't call" mode
//!   that lets a host that owns its transport keep control of every call.
//!
//! ## Concurrency
//!
//! [`Algorithm::create_run_task`] takes `self: Arc<Self>`, so one shared
//! `Arc<dyn Algorithm>` (no lock) serves many requests in parallel. Each
//! [`run_stream`](Algorithm::run_stream) call builds its own [`Driver`], so
//! offloaded calls never cross between concurrent requests. An algorithm is
//! responsible for its own thread-safety — stateless (like the reference routers) or
//! interior mutability over just its own state.
//!
//! ## Reference algorithms
//!
//! Worked implementations — a random router, an LLM classifier, and a stateful
//! ensemble — plus runnable agents live in the `libsy-examples` crate.

mod driver;

use std::{error::Error, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::driver::{DriverRequest, DriverStep, TypeErasedDriver};

pub use switchyard_protocol::{
    ContentBlock, ConversationRequest, ConversationResponse, Message, ResponseOutput, Role,
};

/// Shorthand for the crate's boxed, thread-safe error type.
type BoxErr = Box<dyn Error + Send + Sync>;

/// A boxed, `Send` stream of [`Step`]s — the output of
/// [`Algorithm::run_stream`]. Boxed so the trait method that produces it keeps
/// `Arc<dyn Algorithm>` object-safe.
pub type StepStream = Pin<Box<dyn Stream<Item = Result<Step, BoxErr>> + Send>>;

/// Correlation and routing metadata attached to a request or response.
///
/// All fields are optional; algorithms and observers use whichever are present
/// (e.g. to key per-session state or emit correlated telemetry). `extra_metadata`
/// is a free-form escape hatch for host-specific keys.
#[derive(Clone)]
pub struct Metadata {
    /// Stable id for a multi-request session/conversation.
    pub session_id: Option<String>,
    /// Id of the agent making the request.
    pub agent_id: Option<String>,
    /// Id of the task the request belongs to.
    pub task_id: Option<String>,
    /// External trace/request id for joining with the host's telemetry.
    pub correlation_id: Option<String>,
    /// Arbitrary host-defined key/value metadata.
    pub extra_metadata: Option<std::collections::BTreeMap<String, String>>,
}

/// A request entering the orchestrator: the normalized [`ConversationRequest`] plus the
/// original provider payload and correlation [`Metadata`].
#[derive(Clone)]
pub struct Request {
    /// The normalized request an algorithm routes.
    pub llm_request: ConversationRequest,
    /// Optional source provider body, carried only for hosts that need exact fidelity.
    pub raw_request: Option<serde_json::Value>,
    /// Correlation metadata carried through the request.
    pub metadata: Option<Metadata>,
}

/// Agentic-stack events fed to an algorithm out of band via
/// [`Algorithm::process_signals`] (e.g. tool results, budget updates).
///
/// A placeholder today; a stateful algorithm can begin consuming signals as the
/// enum grows without changing the orchestrator contract.
#[derive(Clone)]
pub struct Signals {}

/// A response leaving the orchestrator: the neutral [`ConversationResponse`] plus optional
/// correlation [`Metadata`].
#[derive(Clone)]
pub struct Response {
    /// The neutral model response.
    pub llm_response: ConversationResponse,
    /// Correlation metadata carried through the response.
    pub metadata: Option<Metadata>,
}

/// A decision/trace object produced by an algorithm.
///
/// Carried as a trait object (not a generic parameter) so a stream consumer can
/// inspect any algorithm's decision through this common interface without
/// knowing the concrete type. `as_any` is the escape hatch for a consumer that
/// *does* know the algo and wants to downcast to the concrete decision.
pub trait Decision: Send + Sync {
    /// The model this decision selected (e.g. the routed target's name).
    fn selected_model(&self) -> &str;
    /// A human-readable explanation of the decision, for logs and traces.
    fn reasoning(&self) -> Option<&str>;
    /// Downcast handle: a consumer that knows the algorithm can recover the
    /// concrete decision type via `as_any().downcast_ref::<ConcreteDecision>()`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A request paired with the routing [`Decision`] that produced it — the unit an
/// [`LlmClient`] (or an offload host) is handed to serve.
///
/// The two model identifiers live in separate, unambiguous places: the model to
/// call is [`decision.selected_model()`](Decision::selected_model), while
/// `request.llm_request.model` is the inbound name the agent asked for. A client
/// maps `selected_model()` to the provider model id it hits.
#[derive(Clone)]
pub struct RoutedRequest {
    /// The request to serve; its `model` is the agent's original inbound model.
    pub request: Request,
    /// The routing decision behind this call; `selected_model()` is the model to hit.
    pub decision: Arc<dyn Decision>,
    /// The client that serves this call by default, or `None` when the routed target
    /// had no client. Rides along on the offloaded call so a host driving the stream
    /// can serve it by default or override it with its own transport.
    pub default_client: Option<Arc<dyn LlmClient>>,
}

/// The host-facing half of an offloaded model call, surfaced inside [`Step::CallLlm`].
///
/// Wraps a [`DriverRequest`] whose payload is a [`RoutedRequest`]. The host reads the
/// routed request ([`get_routed`](Self::get_routed)) and the decision behind it
/// ([`get_decision`](Self::get_decision)), performs (or delegates) the model call, and
/// fulfills it with [`respond`](Self::respond) — unblocking the algorithm's
/// [`Driver::call_llm`] on the other side.
pub struct CallLlmRequest {
    inner: DriverRequest,
}

impl CallLlmRequest {
    /// Wrap a driver request whose payload is a [`RoutedRequest`].
    fn new(inner: DriverRequest) -> Self {
        Self { inner }
    }

    /// The routed request the host should serve. Its
    /// [`default_client`](RoutedRequest::default_client) serves the call by default,
    /// and its `decision.selected_model()` names the model to hit.
    pub fn get_routed(&self) -> &RoutedRequest {
        match self.inner.request::<RoutedRequest>() {
            Ok(routed) => routed,
            Err(_) => unreachable!("CallLlmRequest payload is always a RoutedRequest"),
        }
    }

    /// The model request to perform (the [`Request`] inside the routed request).
    pub fn get_request(&self) -> &Request {
        &self.get_routed().request
    }

    /// The decision that led to this call — its `selected_model()` is the model to hit.
    pub fn get_decision(&self) -> &dyn Decision {
        self.get_routed().decision.as_ref()
    }

    /// Fulfill the promise with the caller's model-call result. Pass `Err(..)` to
    /// propagate a failed model call back to the algorithm. Consumes the promise: it
    /// can only be fulfilled once.
    pub fn respond(self, result: Result<Response, BoxErr>) -> Result<(), BoxErr> {
        self.inner.respond::<Response>(result)
    }
}

/// The offload channel handed to an algorithm's
/// [`create_run_task`](Algorithm::create_run_task). The algorithm makes model calls
/// with [`call_llm_target`](Self::call_llm_target) (or [`call_llm`](Self::call_llm)) and
/// publishes its [`Decision`]s with [`info`](Self::info); each call is offloaded to the
/// request's [`Step`] stream and awaits the consumer's response. The step channel is
/// bounded, so the consumer paces the algorithm one step at a time.
#[derive(Clone)]
pub struct Driver {
    driver: TypeErasedDriver,
}

impl Driver {
    /// Build an empty driver with its step channel ready. Created per call by
    /// [`run_stream`](Algorithm::run_stream).
    pub(crate) fn new() -> Self {
        Self {
            driver: TypeErasedDriver::new(),
        }
    }

    /// Offload a model call: publish `routed` as a [`Step::CallLlm`] and await the
    /// consumer's [`Response`]. Errors if the stream is closed or the call failed.
    pub async fn call_llm(&self, routed: RoutedRequest) -> Result<Response, BoxErr> {
        self.driver
            .fulfill_request::<RoutedRequest, Response>(routed)
            .await
    }

    /// Offload a call to `target`: pair `request` with `decision` and the target's
    /// default client into a [`RoutedRequest`], then publish it (see
    /// [`call_llm`](Self::call_llm)). The convenience most algorithms use;
    /// `decision.selected_model()` names the model to hit, and `request`'s
    /// inbound `model` is left untouched.
    pub async fn call_llm_target(
        &self,
        target: &LlmTarget,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, BoxErr> {
        self.call_llm(RoutedRequest {
            request,
            decision,
            default_client: target.llm_client.clone(),
        })
        .await
    }

    /// Publish a routing [`Decision`] as a [`Step::Decision`] on the stream.
    pub async fn info(&self, decision: Arc<dyn Decision>) -> Result<(), BoxErr> {
        self.driver.info(decision).await
    }

    /// Emit the terminal step: [`Step::ReturnToAgent`] on `Ok`, or an `Err` stream
    /// item on failure. Internal: called once by [`run_stream`](Algorithm::run_stream)
    /// when the algorithm finishes.
    pub(crate) async fn finish(&self, result: Result<Response, BoxErr>) -> Result<(), BoxErr> {
        match result {
            Ok(response) => self.driver.done(response).await,
            Err(err) => self.driver.fail(err).await,
        }
    }

    /// Transform the raw driver stream into a stream of [`Step`]s. Internal: the
    /// consumer stream is taken (once) by [`run_stream`](Algorithm::run_stream). A
    /// payload that does not match the expected type for its step becomes an `Err` item.
    pub(crate) fn stream(&self) -> impl Stream<Item = Result<Step, BoxErr>> {
        self.driver.stream().map(|item| match item? {
            DriverStep::Request(req) => Ok(Step::CallLlm(CallLlmRequest::new(req))),
            DriverStep::Info(payload) => payload
                .downcast::<Arc<dyn Decision>>()
                .map(|decision| Step::Decision(*decision))
                .map_err(|_| "driver: info payload was not a Decision".into()),
            DriverStep::Done(payload) => payload
                .downcast::<Response>()
                .map(Step::ReturnToAgent)
                .map_err(|_| "driver: done payload was not a Response".into()),
        })
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-request state threaded to an algorithm alongside its [`Driver`]. A placeholder
/// for cross-cutting state (correlation ids, budgets, deadlines) an algorithm will
/// read; empty today. It does not carry the offload driver, so it is safe to share.
#[derive(Clone, Default)]
pub struct Context {}

impl Context {
    /// Build an empty context.
    pub fn new() -> Self {
        Self {}
    }
}

/// One item in the stream returned by [`Driver::stream`] / [`Algorithm::run_stream`].
pub enum Step {
    /// The algorithm needs this model call performed. The host serves it (optionally
    /// via [`RoutedRequest::default_client`]) and fulfills it with
    /// [`CallLlmRequest::respond`].
    CallLlm(CallLlmRequest),
    /// A routing decision the algorithm made, published via [`Driver::info`] as it
    /// happens (rather than collected into a trace returned at the end).
    Decision(Arc<dyn Decision>),
    /// The algorithm finished with its final response — the last step of a run.
    ReturnToAgent(Box<Response>),
}

/// Performs the actual model call for a target. This is the one piece of I/O
/// `libsy` does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as [`RoutedRequest::default_client`] (see [`Algorithm::run_stream`]).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Serve `routed`, returning the model's response. Call the model named by
    /// [`routed.decision.selected_model()`](Decision::selected_model) — the target
    /// the algorithm routed to — mapping it to whatever provider model id this
    /// client hits. `routed.request.llm_request.model` is the agent's original
    /// inbound name, carried through for reference, not a call target.
    async fn call(&self, request: RoutedRequest) -> Result<Response, Box<dyn Error + Send + Sync>>;
}

/// A named routing target: a `semantic_name` an algorithm routes by, and an optional
/// [`LlmClient`] to serve its calls. An algorithm hands a target to
/// [`Driver::call_llm_target`]; the client rides along as
/// [`RoutedRequest::default_client`] for the stream consumer to serve or override.
#[derive(Clone)]
pub struct LlmTarget {
    /// The routing label an algorithm selects this target by — a logical tier like
    /// `"strong"`, or the model id when they coincide. Mapping it to a provider model
    /// id is the client's concern, never the algorithm's.
    pub semantic_name: String,
    /// The client that serves this target's calls by default, or `None` (then the
    /// stream consumer must serve them).
    pub llm_client: Option<Arc<dyn LlmClient>>,
}

/// The set of targets an algorithm may route among. An algorithm is constructed
/// with one and picks targets by position ([`targets`](Self::targets)) or by name
/// ([`get_target`](Self::get_target)).
#[derive(Clone)]
pub struct LlmTargetSet {
    targets: Vec<LlmTarget>,
}

impl LlmTargetSet {
    /// Build a target set from a list of targets.
    pub fn new(targets: Vec<LlmTarget>) -> Self {
        Self { targets }
    }

    /// All targets in the set — e.g. for an algorithm to select among.
    pub fn targets(&self) -> &[LlmTarget] {
        &self.targets
    }

    /// Look up a target by name; errors if no target has that name.
    pub fn get_target(&self, name: &str) -> Result<LlmTarget, Box<dyn Error + Send + Sync>> {
        self.targets
            .iter()
            .find(|t| t.semantic_name == name)
            .cloned()
            .ok_or(format!("Target {} not found", name).into())
    }
}

/// An optimization strategy. Implement [`create_run_task`](Self::create_run_task);
/// callers drive it with the provided [`run`](Self::run) (serve calls, get the answer)
/// or [`run_stream`](Self::run_stream) (drive the [`Step`] stream yourself).
///
/// Methods take `self: Arc<Self>`: one algorithm (`Arc<dyn Algorithm>`) is shared across
/// requests and run concurrently, so it owns its thread-safety. Stateless algorithms
/// (the reference routers) get this for free; a stateful one uses interior mutability
/// over just its own state.
#[async_trait]
pub trait Algorithm: Send + Sync + 'static {
    /// Run one request to completion: make model calls with [`Driver::call_llm_target`],
    /// publish [`Decision`]s with [`Driver::info`], and return the final [`Response`].
    /// The method an algorithm implements; [`run`](Self::run) / [`run_stream`](Self::run_stream)
    /// drive it. `ctx` carries cross-cutting request state (empty today).
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>>;

    /// Feed the algorithm agentic-stack events (tool results, budgets, etc.). The
    /// reference algorithms ignore signals; a stateful algorithm updates its own
    /// (interior-mutable) state. Takes `self: Arc<Self>` like the other run methods.
    async fn process_signals(
        self: Arc<Self>,
        signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Run one request as a stream of [`Step`]s (provided). The algorithm runs on its
    /// own task; drive the stream: serve each [`Step::CallLlm`] (via its
    /// [`default_client`](RoutedRequest::default_client) or your own transport) and read
    /// [`Step::Decision`]s until the final [`Step::ReturnToAgent`]. The step channel is
    /// bounded, so pulling paces the algorithm; each call is independent, so many run
    /// concurrently.
    fn run_stream(self: Arc<Self>, ctx: Context, request: Request) -> StepStream {
        // This call's own driver: take its consumer stream, hand a producer-side clone to
        // the algorithm task, and keep one to emit the terminal step. The task blocks
        // publishing a step until the consumer pulls the previous one.
        let driver = Driver::new();
        let stream = driver.stream();
        tokio::spawn(async move {
            let outcome = self.create_run_task(ctx, driver.clone(), request).await;
            let _ = driver.finish(outcome).await;
        });
        Box::pin(stream)
    }

    /// Run one request to completion, serving each offloaded call with its
    /// [`RoutedRequest::default_client`], and return the decision trace plus the final
    /// [`Response`]. Provided: drives [`run_stream`](Self::run_stream) internally,
    /// collecting each [`Step::Decision`]. Use it when the algorithm holds its own model
    /// clients and the host wants the answer (and the decisions behind it); drive
    /// [`run_stream`](Self::run_stream) instead to serve the calls yourself. Errors
    /// if a routed target has no client to serve its call, or the algorithm fails.
    async fn run(
        self: Arc<Self>,
        ctx: Context,
        request: Request,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>> {
        // Serve up to this many offloaded calls concurrently, so an algorithm that
        // fans out (e.g. an ensemble) isn't serialized on the client.
        const MAX_CONCURRENT_CALLS: usize = 10;

        // Serve one offloaded call with its target's default client. A failed *model*
        // call is forwarded to the algorithm via `respond`; this errors only on an
        // infrastructure failure (no default client, or the promise was dropped).
        async fn serve(call: CallLlmRequest) -> Result<(), Box<dyn Error + Send + Sync>> {
            let routed = call.get_routed().clone();
            let client = routed.default_client.clone().ok_or_else(|| {
                format!(
                    "run: target '{}' has no client to serve the call",
                    routed.decision.selected_model()
                )
            })?;
            call.respond(client.call(routed).await)
        }

        let stream = self.run_stream(ctx, request);
        tokio::pin!(stream);

        let mut trace: Vec<Arc<dyn Decision>> = Vec::new();
        let mut in_flight = futures::stream::FuturesUnordered::new();
        let mut final_response: Option<Response> = None;
        let mut stream_open = true;

        while stream_open || !in_flight.is_empty() {
            tokio::select! {
                // Surface a failed serve as soon as it completes.
                Some(result) = in_flight.next(), if !in_flight.is_empty() => result?,
                // Pull the next step, unless the stream ended or we're at the cap.
                step = stream.next(), if stream_open && in_flight.len() < MAX_CONCURRENT_CALLS => {
                    match step {
                        None => stream_open = false,
                        Some(item) => match item? {
                            Step::CallLlm(call) => in_flight.push(serve(call)),
                            Step::Decision(decision) => trace.push(decision),
                            Step::ReturnToAgent(response) => {
                                final_response = Some(*response);
                                stream_open = false;
                            }
                        },
                    }
                }
            }
        }

        final_response
            .map(|response| (trace, response))
            .ok_or_else(|| "run: stream ended without a final response".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    /// Mock client that echoes back the target name it was called with.
    struct EchoClient;

    #[async_trait]
    impl LlmClient for EchoClient {
        async fn call(
            &self,
            routed: RoutedRequest,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            // Echo back the model the algorithm routed to (the decision's selection).
            Ok(Response {
                llm_response: ConversationResponse {
                    model: Some(routed.decision.selected_model().to_string()),
                    ..ConversationResponse::default()
                },
                metadata: None,
            })
        }
    }

    /// Trivial decision + algo used only to exercise the orchestrator: calls the
    /// first target and returns its response with a one-item trace.
    struct TestDecision {
        model: String,
    }

    impl Decision for TestDecision {
        fn selected_model(&self) -> &str {
            &self.model
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct TestAlgo {
        target_set: LlmTargetSet,
    }

    #[async_trait]
    impl Algorithm for TestAlgo {
        async fn create_run_task(
            self: Arc<Self>,
            _ctx: Context,
            driver: Driver,
            request: Request,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            let target = self
                .target_set
                .targets()
                .first()
                .ok_or("no targets")?
                .clone();
            let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                model: target.semantic_name.clone(),
            });
            driver.info(decision.clone()).await?;
            driver.call_llm_target(&target, request, decision).await
        }

        async fn process_signals(
            self: Arc<Self>,
            _signals: Signals,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Ok(())
        }
    }

    /// Build a shared `TestAlgo` over the given target set.
    fn orch(target_set: LlmTargetSet) -> Arc<dyn Algorithm> {
        Arc::new(TestAlgo { target_set })
    }

    fn request() -> Request {
        Request {
            llm_request: ConversationRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                ..ConversationRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    /// `(name, has_client)` — `has_client: false` builds a target with no default client.
    fn target_set(names: &[(&str, bool)]) -> LlmTargetSet {
        let targets = names
            .iter()
            .map(|(name, has_client)| LlmTarget {
                semantic_name: name.to_string(),
                llm_client: has_client.then(|| Arc::new(EchoClient) as Arc<dyn LlmClient>),
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    #[tokio::test]
    async fn run_offloads_via_promise_then_returns_to_agent(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target -> its call is offloaded via a promise the
        // orchestrator surfaces as a `CallLlm` step for us to fulfill.
        let stream =
            orch(target_set(&[("offload/model", false)])).run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_call = false;
        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallLlm(call) => {
                    saw_call = true;
                    // The decision rode along with the promise.
                    assert_eq!(call.get_decision().selected_model(), "offload/model");
                    // Fulfilling the promise is the "real" model call the caller makes.
                    call.respond(Ok(Response {
                        llm_response: ConversationResponse {
                            model: Some("fulfilled".to_string()),
                            ..ConversationResponse::default()
                        },
                        metadata: None,
                    }))?;
                }
                Step::Decision(decision) => {
                    assert_eq!(decision.selected_model(), "offload/model");
                }
                Step::ReturnToAgent(response) => {
                    final_completion = response.llm_response.model.clone();
                }
            }
        }

        assert!(saw_call, "expected a CallLlm step before ReturnToAgent");
        assert_eq!(
            final_completion.ok_or("no ReturnToAgent step")?,
            "fulfilled"
        );
        Ok(())
    }

    #[tokio::test]
    async fn client_backed_target_offloads_with_a_default_client(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Every call now offloads to the stream; a client-backed target rides its
        // client along as `default_client` so the consumer can serve it by default.
        let stream =
            orch(target_set(&[("direct/model", true)])).run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallLlm(call) => {
                    let routed = call.get_routed().clone();
                    let client = routed
                        .default_client
                        .clone()
                        .ok_or("expected a default client")?;
                    let result = client.call(routed).await;
                    call.respond(result)?;
                }
                Step::Decision(_) => {}
                Step::ReturnToAgent(response) => {
                    final_completion = response.llm_response.model.clone();
                }
            }
        }

        // EchoClient echoes the model name back as the completion.
        assert_eq!(final_completion.ok_or("no ReturnToAgent")?, "direct/model");
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_the_response_when_all_targets_have_clients(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Every target has a client, so run serves every call via the
        // default client and returns the trace + final response.
        let (trace, response) = orch(target_set(&[("direct/model", true)]))
            .run(Context::default(), request())
            .await?;
        // TestAlgo calls the first target; EchoClient echoes its name.
        assert_eq!(response.llm_response.model.as_deref(), Some("direct/model"));
        assert_eq!(trace[0].selected_model(), "direct/model");
        Ok(())
    }

    #[tokio::test]
    async fn run_errors_when_a_target_lacks_a_client() -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target has no default client to serve its offloaded call, so
        // driving it to completion errors.
        assert!(orch(target_set(&[("offload/model", false)]))
            .run(Context::default(), request())
            .await
            .is_err());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 12)]
    async fn requests_are_processed_in_parallel() -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::time::Duration;
        use tokio::sync::Barrier;

        const N: usize = 12;

        // A client that blocks until all N concurrent calls have arrived. If
        // requests were serialized (one algorithm behind a `Mutex`), only one
        // call could be in flight, the barrier would never reach N, and the test
        // would time out. It passes only because the shared algorithm is driven
        // concurrently across requests.
        struct BarrierClient {
            barrier: Arc<Barrier>,
        }

        #[async_trait]
        impl LlmClient for BarrierClient {
            async fn call(
                &self,
                routed: RoutedRequest,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                self.barrier.wait().await;
                Ok(Response {
                    llm_response: ConversationResponse {
                        model: Some(routed.decision.selected_model().to_string()),
                        ..ConversationResponse::default()
                    },
                    metadata: None,
                })
            }
        }

        let barrier = Arc::new(Barrier::new(N));
        let targets = LlmTargetSet::new(vec![LlmTarget {
            semantic_name: "m".to_string(),
            llm_client: Some(Arc::new(BarrierClient {
                barrier: barrier.clone(),
            })),
        }]);
        // One shared algorithm driven by many concurrent requests.
        let algo = orch(targets);

        let mut handles = Vec::new();
        for _ in 0..N {
            let algo = algo.clone();
            handles.push(tokio::spawn(async move {
                algo.run(Context::default(), request())
                    .await
                    .map(|(_, response)| response.llm_response.model.unwrap_or_default())
            }));
        }

        for handle in handles {
            // The timeout turns a serialization deadlock into a failure, not a hang.
            let completion = tokio::time::timeout(Duration::from_secs(5), handle).await???;
            assert_eq!(completion, "m");
        }
        Ok(())
    }

    #[tokio::test]
    async fn offload_error_propagates_back_to_the_algorithm(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target offloads its call; we fulfill the promise with an
        // Err, which must flow back through `call_llm_target` into the algorithm and
        // out as an error step — not a response.
        let stream =
            orch(target_set(&[("offload/model", false)])).run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(Step::CallLlm(call)) => {
                    call.respond(Err("upstream model call failed".into()))?;
                }
                Ok(Step::Decision(_)) => {}
                Ok(Step::ReturnToAgent(..)) => {
                    return Err("expected the offload error to propagate, got a response".into());
                }
                Err(err) => {
                    // The algorithm's `call_llm_target` saw the error via the promise.
                    assert!(err.to_string().contains("upstream model call failed"));
                    saw_error = true;
                }
            }
        }

        assert!(saw_error, "expected an error step");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_caps_concurrent_calls() -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        use tokio::sync::{mpsc, Semaphore};

        // Mirrors `run`'s private `MAX_CONCURRENT_CALLS`. The algorithm fans out more
        // calls than the cap; only `CAP` should ever be in flight, the rest wait.
        const CAP: usize = 10;
        const TOTAL_CALLS: usize = CAP + 5;

        // Records peak concurrency and holds each call open until the gate is released,
        // so calls pile up and the cap can be observed.
        struct ProbeClient {
            current: Arc<AtomicUsize>,
            max: Arc<AtomicUsize>,
            entered: mpsc::UnboundedSender<()>,
            gate: Arc<Semaphore>,
        }

        #[async_trait]
        impl LlmClient for ProbeClient {
            async fn call(
                &self,
                routed: RoutedRequest,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
                self.max.fetch_max(now, Ordering::SeqCst);
                let _ = self.entered.send(());
                // Block until the test opens the gate.
                let permit = self.gate.acquire().await.map_err(|e| e.to_string())?;
                drop(permit);
                self.current.fetch_sub(1, Ordering::SeqCst);
                Ok(Response {
                    llm_response: ConversationResponse {
                        outputs: vec![ResponseOutput {
                            role: Role::Assistant,
                            content: vec![ContentBlock::Text {
                                text: routed.decision.selected_model().to_string(),
                            }],
                            stop_reason: None,
                        }],
                        ..ConversationResponse::default()
                    },
                    metadata: None,
                })
            }
        }

        // Fans out `n` concurrent calls to one target.
        struct FanOut {
            n: usize,
            target: LlmTarget,
        }

        #[async_trait]
        impl Algorithm for FanOut {
            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let calls = (0..self.n).map(|i| {
                    let driver = driver.clone();
                    let target = self.target.clone();
                    async move {
                        let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                            model: format!("m{i}"),
                        });
                        driver.call_llm_target(&target, request(), decision).await
                    }
                });
                futures::future::join_all(calls).await;
                Ok(Response {
                    llm_response: ConversationResponse {
                        outputs: vec![ResponseOutput {
                            role: Role::Assistant,
                            content: vec![ContentBlock::Text {
                                text: "done".to_string(),
                            }],
                            stop_reason: None,
                        }],
                        ..ConversationResponse::default()
                    },
                    metadata: None,
                })
            }

            async fn process_signals(
                self: Arc<Self>,
                _signals: Signals,
            ) -> Result<(), Box<dyn Error + Send + Sync>> {
                Ok(())
            }
        }

        let current = Arc::new(AtomicUsize::new(0));
        let max = Arc::new(AtomicUsize::new(0));
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Semaphore::new(0)); // starts closed

        let client = Arc::new(ProbeClient {
            current: current.clone(),
            max: max.clone(),
            entered: entered_tx,
            gate: gate.clone(),
        }) as Arc<dyn LlmClient>;
        let target = LlmTarget {
            semantic_name: "m".to_string(),
            llm_client: Some(client),
        };

        let algo: Arc<dyn Algorithm> = Arc::new(FanOut {
            n: TOTAL_CALLS,
            target,
        });
        let handle = tokio::spawn(async move { algo.run(Context::default(), request()).await });

        // Exactly `CAP` calls should enter; each `recv` times out into an error if the
        // run dispatched fewer than the cap.
        for _ in 0..CAP {
            tokio::time::timeout(Duration::from_secs(5), entered_rx.recv())
                .await
                .map_err(|_| "timed out waiting for a call to enter")?;
        }
        assert_eq!(
            current.load(Ordering::SeqCst),
            CAP,
            "exactly the cap should be in flight"
        );
        // No further call enters while the cap is saturated — extra steps wait for capacity.
        assert!(
            tokio::time::timeout(Duration::from_millis(200), entered_rx.recv())
                .await
                .is_err(),
            "a call beyond the cap entered while {CAP} were already in flight"
        );

        // Release the gate; every call completes and the run finishes.
        gate.add_permits(TOTAL_CALLS);
        let (_trace, response) = tokio::time::timeout(Duration::from_secs(5), handle).await???;
        assert_eq!(
            response
                .llm_response
                .first_output()
                .and_then(|output| output.content.first()),
            Some(&ContentBlock::Text {
                text: "done".to_string()
            })
        );
        assert_eq!(
            max.load(Ordering::SeqCst),
            CAP,
            "peak in-flight should equal the cap"
        );
        Ok(())
    }
}

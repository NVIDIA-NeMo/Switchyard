// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`Algorithm`] trait and its [`Driver`] — the orchestration contract every
//! routing/optimization algorithm implements, and the offload channel it makes model
//! calls and publishes [`Decision`]s over. See the crate root for the narrative model.

use std::{error::Error, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::{Stream, StreamExt};

/// The request/response protocol types, re-exported from [`switchyard_protocol`].
/// [`LlmRequest`] is the normalized request; [`AggLlmResponse`] is the buffered response;
/// [`LlmResponseChunk`] is one streaming event; [`LlmResponse`] is the streamed response
/// (a live [`LlmResponseStream`] or the terminal aggregate).
pub use switchyard_protocol::{
    AggLlmResponse, Context, Decision, LlmRequest, LlmResponse, LlmResponseChunk,
    LlmResponseStream, Metadata, Request, Response, RoutedLlmClient, Signals,
};

use super::driver::{DriverRequest, DriverStep, TypeErasedDriver};

/// Shorthand for the crate's boxed, thread-safe error type.
type BoxErr = Box<dyn Error + Send + Sync>;

/// A boxed, `Send` stream of [`Step`]s — the output of
/// [`Algorithm::run_stream`]. Boxed so the trait method that produces it keeps
/// `Arc<dyn Algorithm>` object-safe.
pub type StepStream = Pin<Box<dyn Stream<Item = Result<Step, BoxErr>> + Send>>;

/// A request paired with the routing [`Decision`] that produced it — the offload
/// payload a host reads (via [`CallLlmRequest::get_routed`]) to serve the call.
///
/// The two model identifiers live in separate, unambiguous places: the model to
/// call is [`decision.selected_model()`](Decision::selected_model), while
/// `request.llm_request.model` is the *inbound* name the agent asked for (libsy
/// never overwrites it). A client maps `selected_model()` to the provider model
/// id it hits.
#[derive(Clone)]
pub struct RoutedRequest {
    /// The request to serve; its `model` is the agent's original name.
    pub request: Request,
    /// The routing decision behind this call; `selected_model()` is the model to hit.
    pub decision: Arc<dyn Decision>,
    /// The client that serves this call by default, or `None` when the routed target
    /// had no client. Rides along on the offloaded call so a host driving the stream
    /// can serve it by default or override it with its own transport.
    pub default_client: Option<Arc<dyn RoutedLlmClient>>,
    /// The request's cross-cutting context, carried through the offload so whoever
    /// serves the call (libsy's own `run`, or a host driving the stream) hands it to
    /// [`RoutedLlmClient::call`].
    pub ctx: Context,
}

/// The host-facing half of an offloaded model call, surfaced inside [`Step::CallLlm`].
///
/// Wraps a `DriverRequest` whose payload is a [`RoutedRequest`]. The host reads the
/// routed request ([`get_routed`](Self::get_routed)) and the decision behind it
/// ([`get_decision`](Self::get_decision)), performs (or delegates) the model call, and
/// fulfills it with [`respond`](Self::respond) — unblocking the algorithm's
/// [`Driver::call_llm`] on the other side.
pub struct CallLlmRequest {
    inner: DriverRequest,
    routed: RoutedRequest,
}

impl CallLlmRequest {
    /// Wrap a driver request whose payload is a [`RoutedRequest`]. Caches an owned copy
    /// so the accessors are plain field reads.
    fn new(inner: DriverRequest) -> Self {
        // The payload is always a `RoutedRequest` (set by `Driver::call_llm`); a
        // mismatch would be a libsy bug, not a runtime condition.
        let routed = match inner.request::<RoutedRequest>() {
            Ok(routed) => routed.clone(),
            Err(_) => unreachable!("CallLlmRequest payload is always a RoutedRequest"),
        };
        Self { inner, routed }
    }

    /// The routed request the host should serve. Its
    /// [`default_client`](RoutedRequest::default_client) serves the call by default,
    /// and its `decision.selected_model()` names the model to hit.
    pub fn get_routed(&self) -> &RoutedRequest {
        &self.routed
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
    /// consumer's [`Response`]. The call's context travels inside
    /// [`routed.ctx`](RoutedRequest::ctx). Errors if the stream is closed or the call failed.
    pub async fn call_llm(&self, routed: RoutedRequest) -> Result<Response, BoxErr> {
        self.driver
            .fulfill_request::<RoutedRequest, Response>(routed.ctx.clone(), routed)
            .await
    }

    /// Offload a call to `target`: pair `request` with `decision` and the target's
    /// default client into a [`RoutedRequest`], then publish it (see
    /// [`call_llm`](Self::call_llm)). The convenience most algorithms use;
    /// `decision.selected_model()` names the model to hit, and `request`'s
    /// `model` is left untouched.
    pub async fn call_llm_target(
        &self,
        ctx: Context,
        target: &LlmTarget,
        request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, BoxErr> {
        self.call_llm(RoutedRequest {
            request,
            decision,
            default_client: target.llm_client.clone(),
            ctx,
        })
        .await
    }

    /// Publish a routing [`Decision`] as a [`Step::Decision`] on the stream.
    pub async fn info(&self, ctx: Context, decision: Arc<dyn Decision>) -> Result<(), BoxErr> {
        self.driver.info(ctx, decision).await
    }

    /// Emit the terminal step: [`Step::ReturnToAgent`] on `Ok`, or an `Err` stream
    /// item on failure. Internal: called once by [`run_stream`](Algorithm::run_stream)
    /// when the algorithm finishes.
    pub(crate) async fn finish(
        &self,
        ctx: Context,
        result: Result<Response, BoxErr>,
    ) -> Result<(), BoxErr> {
        match result {
            Ok(response) => self.driver.done(ctx, response).await,
            Err(err) => self.driver.fail(ctx, err).await,
        }
    }

    /// Transform the raw driver stream into a stream of [`Step`]s. Internal: the
    /// consumer stream is taken (once) by [`run_stream`](Algorithm::run_stream). A
    /// payload that does not match the expected type for its step becomes an `Err` item.
    pub(crate) fn stream(&self) -> impl Stream<Item = Result<Step, BoxErr>> {
        self.driver.stream().map(|item| match item? {
            DriverStep::Request(req) => Ok(Step::CallLlm(Box::new(CallLlmRequest::new(req)))),
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

/// One item in the stream returned by `Driver::stream` / [`Algorithm::run_stream`].
pub enum Step {
    /// The algorithm needs this model call performed. The host serves it (optionally
    /// via [`RoutedRequest::default_client`]) and fulfills it with
    /// [`CallLlmRequest::respond`]. Boxed: it is by far the largest variant.
    CallLlm(Box<CallLlmRequest>),
    /// A routing decision the algorithm made, published via [`Driver::info`] as it
    /// happens (rather than collected into a trace returned at the end).
    Decision(Arc<dyn Decision>),
    /// The algorithm finished with its final response — the last step of a run.
    ReturnToAgent(Box<Response>),
}

/// Abort guard
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A named routing target: a `semantic_name` an algorithm routes by, and an optional
/// [`RoutedLlmClient`] to serve its calls. An algorithm hands a target to
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
    pub llm_client: Option<Arc<dyn RoutedLlmClient>>,
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
    #[allow(unused_variables)]
    async fn process_signals(
        self: Arc<Self>,
        signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    /// Process a request to completion, returning a stream of [`Step`]s.
    /// Each [`Step::CallLlm`] is an offloaded model call the consumer must serve.
    /// The stream ends with a [`Step::ReturnToAgent`] on success, or an `Err` item on failure.
    fn run_stream(self: Arc<Self>, ctx: Context, request: Request) -> StepStream {
        let driver = Driver::new();
        let task_driver = driver.clone();
        let task_ctx = ctx.clone();
        let stream = task_driver.stream();
        let handle =
            tokio::spawn(async move { self.create_run_task(task_ctx, task_driver, request).await });
        // Dropping the stream aborts the algorithm task, so it doesn't keep running after the
        let abort_guard = AbortOnDrop(handle.abort_handle());

        let finish_driver = driver.clone();
        let finish_ctx = ctx.clone();
        let tail: StepStream = Box::pin(
            futures::stream::once(async move {
                let result = match handle.await {
                    Ok(response) => response,
                    Err(e) => Err(format!("Algorithm task panicked: {e}").into()),
                };
                finish_driver.finish(finish_ctx, result).await
            })
            .filter_map(|finish_result| async move { finish_result.err().map(Err) }),
        );

        let stream: StepStream = Box::pin(stream);
        Box::pin(futures::stream::select(stream, tail).map(move |step| {
            // link abort guard to stream
            let _keep_alive = &abort_guard;
            step
        }))
    }

    /// Process a request to completion, returning the final [`Response`] and the trace of
    /// [`Decision`]s the algorithm made along the way.
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
            call.respond(
                client
                    .call(routed.ctx, routed.request, routed.decision)
                    .await,
            )
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
                            Step::CallLlm(call) => in_flight.push(serve(*call)),
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

        // Return whatever the algorithm produced — buffered or streamed. The caller
        // decides whether to aggregate the response or forward its chunks.
        final_response
            .map(|response| (trace, response))
            .ok_or_else(|| "run: stream ended without a final response".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use switchyard_protocol::{completion_text, text_request, text_response};

    /// Mock client that echoes back the target name it was called with.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            // Echo back the model the algorithm routed to (the decision's selection).
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
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
            ctx: Context,
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
            driver.info(ctx.clone(), decision.clone()).await?;
            driver
                .call_llm_target(ctx, &target, request, decision)
                .await
        }
    }

    /// Build a shared `TestAlgo` over the given target set.
    fn orch(target_set: LlmTargetSet) -> Arc<dyn Algorithm> {
        Arc::new(TestAlgo { target_set })
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi".to_string()),
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
                llm_client: has_client.then(|| Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>),
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    /// Client that serves a call as a token stream — its `call` returns
    /// [`LlmResponse::Stream`] replaying `chunks` in order (as `Ok` items).
    struct StreamingClient {
        chunks: Vec<LlmResponseChunk>,
    }

    #[async_trait]
    impl RoutedLlmClient for StreamingClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            _decision: Arc<dyn Decision>,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            let stream = futures::stream::iter(self.chunks.clone().into_iter().map(Ok)).boxed();
            Ok(Response {
                llm_response: LlmResponse::Stream(stream),
                metadata: None,
            })
        }
    }

    /// Build a single-target algo whose one target streams `chunks`.
    fn streaming_orch(chunks: Vec<LlmResponseChunk>) -> Arc<dyn Algorithm> {
        let target = LlmTarget {
            semantic_name: "stream/model".to_string(),
            llm_client: Some(Arc::new(StreamingClient { chunks }) as Arc<dyn RoutedLlmClient>),
        };
        orch(LlmTargetSet::new(vec![target]))
    }

    #[tokio::test]
    async fn run_returns_a_streamed_response_the_caller_aggregates(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A streaming client -> its chunks flow through the promise and `ReturnToAgent`,
        // and `run` returns the live stream untouched for the caller to fold.
        let orch = streaming_orch(vec![
            LlmResponseChunk::MessageStart {
                id: Some("m1".to_string()),
                model: Some("stream/model".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "hel".to_string(),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "lo".to_string(),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            },
        ]);
        let (trace, response) = orch.run(Context::default(), request()).await?;
        // `run` handed back the live stream; the caller folds it to a buffered aggregate.
        let agg = response.llm_response.into_agg().await?;
        assert_eq!(completion_text(&agg), "hello");
        assert_eq!(agg.model.as_deref(), Some("stream/model"));
        assert_eq!(trace.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn aggregating_a_streamed_response_propagates_a_mid_stream_error(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // `run` succeeds and returns the stream; the in-band `Error` chunk surfaces only
        // when the caller aggregates it.
        let orch = streaming_orch(vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "partial".to_string(),
            },
            LlmResponseChunk::Error {
                message: "upstream exploded".to_string(),
            },
        ]);
        let (_, response) = orch.run(Context::default(), request()).await?;
        match response.llm_response.into_agg().await {
            Ok(_) => Err("expected a mid-stream error, got an aggregate".into()),
            Err(err) => {
                assert!(err.to_string().contains("upstream exploded"));
                Ok(())
            }
        }
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
                        llm_response: LlmResponse::Agg(text_response(
                            None,
                            "fulfilled".to_string(),
                        )),
                        metadata: None,
                    }))?;
                }
                Step::Decision(decision) => {
                    assert_eq!(decision.selected_model(), "offload/model");
                }
                Step::ReturnToAgent(response) => {
                    final_completion = Some(
                        response
                            .llm_response
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default(),
                    );
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
                    let result = client
                        .call(routed.ctx, routed.request, routed.decision)
                        .await;
                    call.respond(result)?;
                }
                Step::Decision(_) => {}
                Step::ReturnToAgent(response) => {
                    final_completion = Some(
                        response
                            .llm_response
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default(),
                    );
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
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "direct/model"
        );
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
        impl RoutedLlmClient for BarrierClient {
            async fn call(
                &self,
                _ctx: Context,
                _request: Request,
                decision: Arc<dyn Decision>,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                self.barrier.wait().await;
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(
                        None,
                        decision.selected_model().to_string(),
                    )),
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
                    .map(|(_, response)| {
                        response
                            .llm_response
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default()
                    })
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

    #[tokio::test]
    async fn dropping_the_stream_cancels_the_algorithm_task(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;
        use tokio::sync::mpsc;

        // Sets a flag when dropped, so we can observe whether the algorithm task was
        // cancelled/dropped.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        struct StuckAlgo {
            started: mpsc::UnboundedSender<()>,
            dropped: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Algorithm for StuckAlgo {
            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let _guard = DropGuard(self.dropped.clone());
                let _ = self.started.send(());
                // Await forever without ever touching the driver.
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let algo: Arc<dyn Algorithm> = Arc::new(StuckAlgo {
            started: started_tx,
            dropped: dropped.clone(),
        });

        let stream = algo.run_stream(Context::default(), request());
        started_rx.recv().await.ok_or("task never started")?;
        drop(stream);
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after dropping the stream"
        );
        Ok(())
    }

    #[tokio::test]
    async fn create_run_task_panic_surfaces_as_a_stream_error(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // An algorithm whose task panics must surface an `Err` step to the stream
        // consumer, not abort the process from an unobserved detached task.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        let stream = algo.run_stream(Context::default(), request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Err(err) => {
                    assert!(err.to_string().contains("panicked"));
                    saw_error = true;
                }
                Ok(_) => return Err("expected the panic to surface as an error step".into()),
            }
        }

        assert!(saw_error, "expected an error step from the panicked task");
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_an_error_when_the_algorithm_task_panics(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // The panic surfaces as an `Err` step inside `run_stream`; `run` propagates it
        // via `?`, so the caller gets an `Err` rather than a hang or a silent panic.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        match algo.run(Context::default(), request()).await {
            Ok(_) => Err("expected run to surface the algorithm panic as an error".into()),
            Err(err) => {
                assert!(err.to_string().contains("panicked"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn cancelling_run_cancels_the_algorithm_task() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;
        use tokio::sync::mpsc;

        // Sets a flag when dropped, so we can observe whether the algorithm task was
        // cancelled once the `run` future driving it is dropped.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        struct StuckAlgo {
            started: mpsc::UnboundedSender<()>,
            dropped: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Algorithm for StuckAlgo {
            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let _guard = DropGuard(self.dropped.clone());
                let _ = self.started.send(());
                // Hang forever without ever touching the driver, so only cancellation
                // (not a dropped step channel) can stop this task.
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let algo: Arc<dyn Algorithm> = Arc::new(StuckAlgo {
            started: started_tx,
            dropped: dropped.clone(),
        });

        // Drive `run` on its own task, wait until the algorithm task is up, then cancel
        // `run` — dropping its future (and the `run_stream` stream it holds).
        let run_task = tokio::spawn(async move { algo.run(Context::default(), request()).await });
        started_rx.recv().await.ok_or("task never started")?;
        run_task.abort();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after cancelling run"
        );
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
        impl RoutedLlmClient for ProbeClient {
            async fn call(
                &self,
                _ctx: Context,
                _request: Request,
                decision: Arc<dyn Decision>,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
                self.max.fetch_max(now, Ordering::SeqCst);
                let _ = self.entered.send(());
                // Block until the test opens the gate.
                let permit = self.gate.acquire().await.map_err(|e| e.to_string())?;
                drop(permit);
                self.current.fetch_sub(1, Ordering::SeqCst);
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(
                        None,
                        decision.selected_model().to_string(),
                    )),
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
                ctx: Context,
                driver: Driver,
                _request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let calls = (0..self.n).map(|i| {
                    let driver = driver.clone();
                    let ctx = ctx.clone();
                    let target = self.target.clone();
                    async move {
                        let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                            model: format!("m{i}"),
                        });
                        driver
                            .call_llm_target(ctx, &target, request(), decision)
                            .await
                    }
                });
                futures::future::join_all(calls).await;
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, "done".to_string())),
                    metadata: None,
                })
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
        }) as Arc<dyn RoutedLlmClient>;
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
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "done"
        );
        assert_eq!(
            max.load(Ordering::SeqCst),
            CAP,
            "peak in-flight should equal the cap"
        );
        Ok(())
    }
}

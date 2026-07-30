// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`Algorithm`] trait and its [`Driver`] — the orchestration contract every
//! routing/optimization algorithm implements, and the offload channel it makes model
//! calls and publishes [`Decision`]s over. See the crate root for the narrative model.

use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tracing::Instrument;

/// The request/response protocol types, re-exported from [`switchyard_protocol`].
/// [`LlmRequest`] is the normalized request; [`AggLlmResponse`] is the buffered response;
/// [`LlmResponseChunk`] is one streaming event; [`LlmResponse`] is the streamed response
/// (a live [`LlmResponseStream`] or the terminal aggregate).
use switchyard_protocol::{
    Context, Decision, LlmClientError, Request, Response, RoutedLlmClient, Signals, Usage,
};

use super::driver::{DriverRequest, DriverStep, TypeErasedDriver};
use crate::{DriverError, LibsyError, Result, observability};

/// A boxed, `Send` stream of [`Step`]s — the output of
/// [`Algorithm::run_stream`]. Boxed so the trait method that produces it keeps
/// `Arc<dyn Algorithm>` object-safe.
pub type StepStream = Pin<Box<dyn Stream<Item = Result<Step>> + Send>>;
/// A boxed, `Send` stream of [`DecisionOnlyStep`]s — the output of
/// [`Algorithm::run_decision_only_stream`]. The decision-only counterpart of
/// [`StepStream`]: it ends on a route to serve rather than on a served answer.
pub type DecisionOnlyStepStream = Pin<Box<dyn Stream<Item = Result<DecisionOnlyStep>> + Send>>;

/// Either step stream, as produced by [`Algorithm::run_stream_inner`].
///
/// The two run modes share one implementation, so the variant is fixed by the
/// `decision_only` flag the run was started with rather than chosen per step.
pub enum AnyStepStream {
    /// A served run: every call is performed and the stream ends with an answer.
    Step(StepStream),
    /// A decision-only run: the stream ends with the route for the caller to serve.
    DecisionOnly(DecisionOnlyStepStream),
}

/// One completed model call observed at the algorithm offload boundary.
#[derive(Clone, Debug)]
pub struct LlmCallObservation {
    pub selected_model: String,
    /// Routing tier attached to the selected model, when present.
    pub tier: Option<String>,
    /// Whether this was the routed backend call rather than classifier or judge overhead.
    pub is_routed: bool,
    pub is_success: bool,
    /// Time spent waiting for the model call to resolve.
    pub duration: Duration,
    /// Normalized usage for a buffered successful response.
    pub usage: Option<Usage>,
}

/// One request-scoped observation emitted by the algorithm runner.
#[derive(Clone, Debug)]
pub enum RunObservation {
    /// A completed model call.
    LlmCall(LlmCallObservation),
    /// Routing time recorded by the `switchyard.routing_overhead_ms` metric.
    RoutingOverhead(Duration),
}

/// Request-scoped callback for algorithm-run observations.
pub type RunObserver = Arc<dyn Fn(RunObservation) + Send + Sync>;

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
    pub fn respond(self, result: Result<Response>) -> Result<()> {
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
    // How long the call that served this run took. We need this to calculate routing overhead.
    routed_call: Arc<Mutex<Option<Duration>>>,
    observer: Option<RunObserver>,
    // Whether this run concludes with a handoff instead of serving its final call.
    decision_only: bool,
}

impl Driver {
    /// Build an empty driver with its step channel ready. Created per call by
    /// [`run_stream`](Algorithm::run_stream), which fixes the run's `decision_only` mode.
    pub(crate) fn new(decision_only: bool) -> Self {
        Self::with_observer(decision_only, None)
    }

    fn with_observer(decision_only: bool, observer: Option<RunObserver>) -> Self {
        Self {
            driver: TypeErasedDriver::new(),
            routed_call: Arc::new(Mutex::new(None)),
            observer,
            decision_only,
        }
    }

    /// Whether this run concludes with a handoff rather than serving its final call.
    ///
    /// Fixed for the whole run by the entry point that started it, and applied by
    /// [`final_decision`](Self::final_decision). An algorithm reads it only when the mode
    /// changes what it does *before* concluding.
    pub fn decision_only(&self) -> bool {
        self.decision_only
    }

    /// How long the call that served this run took, if one has succeeded.
    pub(crate) fn routed_call_duration(&self) -> Option<Duration> {
        *self.routed_call.lock()
    }

    /// Records routing overhead (how long Switchyard added to the call) once
    /// per successful run. Called by `observe_run`.
    pub(crate) fn observe_routing_overhead(&self, duration: Duration) {
        if let Some(observer) = &self.observer {
            observer(RunObservation::RoutingOverhead(duration));
        }
    }

    /// Offload a model call: publish `routed` as a [`Step::CallLlm`] and await the
    /// consumer's [`Response`]. The call's context travels inside
    /// [`routed.ctx`](RoutedRequest::ctx). Errors if the stream is closed or the call failed.
    /// The await is wrapped in a `libsy.llm_call` span measuring *fulfillment* as
    /// the algorithm observes it (host queueing/serving included; a streamed
    /// response resolves when its stream handle arrives); latency, outcome, and
    /// token usage are recorded when it resolves. The provider call itself gets a
    /// `libsy.client_call` span when [`Algorithm::run`] serves it.
    #[tracing::instrument(
        target = "libsy",
        name = "libsy.llm_call",
        skip_all,
        fields(
            algorithm = observability::algorithm_label(&routed.ctx),
            selected_model = routed.decision.selected_model(),
            openinference.span.kind = "CHAIN",
            outcome = tracing::field::Empty,
            error = tracing::field::Empty,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            total_tokens = tracing::field::Empty,
            reasoning_tokens = tracing::field::Empty,
        )
    )]
    pub async fn call_llm(&self, routed: RoutedRequest) -> Result<Response> {
        let algorithm = observability::algorithm_label(&routed.ctx).to_string();
        let selected_model = routed.decision.selected_model().to_string();
        let tier = routed.decision.routing_tier().map(str::to_string);
        let is_routed = routed.decision.is_routed_call();
        let started = Instant::now();
        let result = self
            .driver
            .fulfill_request::<RoutedRequest, Response>(routed.ctx.clone(), routed)
            .await;
        let elapsed = started.elapsed();
        observability::record_llm_call(
            &algorithm,
            &selected_model,
            tier.as_deref(),
            is_routed,
            elapsed,
            &result,
            &tracing::Span::current(),
        );
        if let Some(observer) = &self.observer {
            observer(RunObservation::LlmCall(LlmCallObservation {
                selected_model,
                tier,
                is_routed,
                is_success: result.is_ok(),
                duration: elapsed,
                usage: result
                    .as_ref()
                    .ok()
                    .and_then(|response| response.llm_response.as_agg())
                    .map(|response| response.usage.clone()),
            }));
        }
        // Classifier and judge calls are routing overhead.
        // And don't record time for failed calls.
        if is_routed && result.is_ok() {
            *self.routed_call.lock() = Some(elapsed);
        }
        result
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
    ) -> Result<Response> {
        self.call_llm(RoutedRequest {
            request,
            decision,
            default_client: target.llm_client.clone(),
            ctx,
        })
        .await
    }

    /// Conclude a run on `decision`: either serve the final call here, or hand it back.
    pub async fn final_decision(
        &self,
        ctx: Context,
        target: &LlmTarget,
        request: Request,
        decision: Arc<dyn Decision>,
        response: &mut Option<Response>,
    ) -> Result<ResponseOrDecision> {
        if self.decision_only() {
            Ok(ResponseOrDecision::Decision((
                decision,
                Box::new(request),
                response.take().map(Box::new),
            )))
        } else {
            let res = match response.take() {
                Some(res) => res,
                None => {
                    self.call_llm_target(ctx.clone(), target, request, decision.clone())
                        .await?
                }
            };
            Ok(ResponseOrDecision::Response(Box::new(res)))
        }
    }

    /// Publish a routing [`Decision`] as a [`Step::Decision`] on the stream.
    /// Each successfully published decision is counted and logged with its
    /// reasoning; a decision the stream never accepted is not recorded.
    pub async fn info(&self, ctx: Context, decision: Arc<dyn Decision>) -> Result<()> {
        self.driver.info(ctx.clone(), decision.clone()).await?;
        observability::record_decision(&ctx, decision.as_ref());
        Ok(())
    }

    /// Emit the terminal step: [`Step::ReturnToAgent`] on `Ok`, or an `Err` stream
    /// item on failure. Internal: called once by [`run_stream`](Algorithm::run_stream)
    /// when the algorithm finishes.
    pub(crate) async fn finish(
        &self,
        ctx: Context,
        result: Result<ResponseOrDecision>,
    ) -> Result<()> {
        match result {
            Ok(terminal) => self.driver.done(ctx, terminal).await,
            Err(err) => self.driver.fail(ctx, err).await,
        }
    }

    /// Transform the raw driver stream into a stream of [`Step`]s. Internal: the
    /// consumer stream is taken (once) by [`run_stream`](Algorithm::run_stream). A
    /// payload that does not match the expected type for its step becomes an `Err` item.
    pub(crate) fn stream(&self) -> impl Stream<Item = Result<Step>> + use<> {
        self.driver.stream().map(|item| match item? {
            DriverStep::Request(req) => Ok(Step::CallLlm(Box::new(CallLlmRequest::new(req)))),
            DriverStep::Info(payload) => payload
                .downcast::<Arc<dyn Decision>>()
                .map(|decision| Step::Decision(*decision))
                .map_err(|_| {
                    DriverError::TypeMismatch {
                        expected: "Arc<dyn Decision>",
                    }
                    .into()
                }),
            // The run task rejects a terminal its mode cannot deliver before publishing it
            // (see `check_response_type`), so the mismatch arm is a decode-boundary guard.
            DriverStep::Done(payload) => match payload.downcast::<ResponseOrDecision>() {
                Ok(terminal) => match *terminal {
                    ResponseOrDecision::Response(response) => Ok(Step::ReturnToAgent(response)),
                    ResponseOrDecision::Decision(_) => Err(LibsyError::AlgorithmError {
                        message: DECISION_IN_SERVED_RUN.to_string(),
                    }),
                },
                Err(_) => Err(DriverError::TypeMismatch {
                    expected: "ResponseOrDecision",
                }
                .into()),
            },
        })
    }

    pub(crate) fn decision_only_stream(
        &self,
    ) -> impl Stream<Item = Result<DecisionOnlyStep>> + use<> {
        self.driver.stream().map(|item| match item? {
            DriverStep::Request(req) => Ok(DecisionOnlyStep::CallLlm(Box::new(
                CallLlmRequest::new(req),
            ))),
            DriverStep::Info(payload) => payload
                .downcast::<Arc<dyn Decision>>()
                .map(|decision| DecisionOnlyStep::Decision(*decision))
                .map_err(|_| {
                    DriverError::TypeMismatch {
                        expected: "Arc<dyn Decision>",
                    }
                    .into()
                }),
            // The run task rejects a terminal its mode cannot deliver before publishing it
            // (see `check_response_type`), so the mismatch arm is a decode-boundary guard.
            DriverStep::Done(payload) => match payload.downcast::<ResponseOrDecision>() {
                Ok(terminal) => match *terminal {
                    ResponseOrDecision::Response(_) => Err(LibsyError::AlgorithmError {
                        message: RESPONSE_IN_DECISION_ONLY_RUN.to_string(),
                    }),
                    ResponseOrDecision::Decision((decision, request, response)) => Ok(
                        DecisionOnlyStep::ReturnToAgent((decision, request, response)),
                    ),
                },
                Err(_) => Err(DriverError::TypeMismatch {
                    expected: "ResponseOrDecision",
                }
                .into()),
            },
        })
    }
}

impl Default for Driver {
    fn default() -> Self {
        Self::new(false)
    }
}

/// What an algorithm concludes a run with
pub enum ResponseOrDecision {
    /// A served run's answer to the request.
    Response(Box<Response>),
    /// A decision-only run's route for the caller to serve — the [`DecidedCall`].
    Decision((Arc<dyn Decision>, Box<Request>, Option<Box<Response>>)),
}

/// A terminal a decision-only run cannot hand back: it has no route to the agent.
const RESPONSE_IN_DECISION_ONLY_RUN: &str =
    "algorithm returned a response result, which the decision-only step stream cannot serve";
/// A terminal a served run cannot hand back: its caller expects an answer, not a route.
const DECISION_IN_SERVED_RUN: &str =
    "algorithm returned a decision-only result, which the step stream cannot serve";

/// Check the return variant matches the run mode
fn check_response_type(
    terminal: ResponseOrDecision,
    decision_only: bool,
) -> Result<ResponseOrDecision> {
    let message = match (&terminal, decision_only) {
        (ResponseOrDecision::Response(_), true) => RESPONSE_IN_DECISION_ONLY_RUN,
        (ResponseOrDecision::Decision(_), false) => DECISION_IN_SERVED_RUN,
        _ => return Ok(terminal),
    };
    Err(LibsyError::AlgorithmError {
        message: message.to_string(),
    })
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

/// One item in the stream returned by [`Algorithm::run_decision_only_stream`].
pub enum DecisionOnlyStep {
    /// The algorithm needs this model call performed. The host serves it (optionally
    /// via [`RoutedRequest::default_client`]) and fulfills it with
    /// [`CallLlmRequest::respond`]. Boxed: it is by far the largest variant.
    CallLlm(Box<CallLlmRequest>),
    /// A routing decision the algorithm made, published via [`Driver::info`] as it
    /// happens (rather than collected into a trace returned at the end).
    Decision(Arc<dyn Decision>),
    /// The algorithm finished with its final decision
    ReturnToAgent(DecidedCall),
}

/// The final routing decision of a decision-only run: the decision to act on, the request
/// to serve it with, and that call's response when the algorithm already made it.
pub type DecidedCall = (Arc<dyn Decision>, Box<Request>, Option<Box<Response>>);

/// One step of a run as the driving loop sees it.
///
/// [`Step`] and [`DecisionOnlyStep`] differ only in what their terminal variant carries, so
/// both convert into this and one loop can drive either.
enum StepKind {
    CallLlm(Box<CallLlmRequest>),
    Decision(Arc<dyn Decision>),
    Terminal(ResponseOrDecision),
}

impl From<Step> for StepKind {
    fn from(step: Step) -> Self {
        match step {
            Step::CallLlm(call) => StepKind::CallLlm(call),
            Step::Decision(decision) => StepKind::Decision(decision),
            Step::ReturnToAgent(response) => {
                StepKind::Terminal(ResponseOrDecision::Response(response))
            }
        }
    }
}

impl From<DecisionOnlyStep> for StepKind {
    fn from(step: DecisionOnlyStep) -> Self {
        match step {
            DecisionOnlyStep::CallLlm(call) => StepKind::CallLlm(call),
            DecisionOnlyStep::Decision(decision) => StepKind::Decision(decision),
            DecisionOnlyStep::ReturnToAgent(decided) => {
                StepKind::Terminal(ResponseOrDecision::Decision(decided))
            }
        }
    }
}

/// Drive a step stream to completion: serve every offloaded call with its target's default
/// client, collect the decisions published along the way, and return the terminal step.
///
/// Generic over the step shape so the served-response and decision-only runs share one
/// loop; the [`StepKind`] conversion normalizes their terminal steps into a
/// [`ResponseOrDecision`].
async fn drive<T: Into<StepKind>>(
    stream: impl Stream<Item = Result<T>>,
) -> Result<(Vec<Arc<dyn Decision>>, ResponseOrDecision)> {
    // Serve one offloaded call with its target's default client. A failed *model*
    // call is forwarded to the algorithm via `respond`; this errors only on an
    // infrastructure failure (no default client, or the promise was dropped).
    // `serve` makes the one API call libsy itself performs, so it gets its
    // own `libsy.client_call` span.
    #[tracing::instrument(
        target = "libsy",
        name = "libsy.client_call",
        skip_all,
        fields(
            algorithm = observability::algorithm_label(&call.get_routed().ctx),
            switchyard.algorithm = observability::algorithm_label(&call.get_routed().ctx),
            switchyard.routing.tier = tracing::field::Empty,
            selected_model = call.get_decision().selected_model(),
            otel.kind = "client",
            otel.name = %format_args!("chat {}", call.get_decision().selected_model()),
            openinference.span.kind = "LLM",
            gen_ai.operation.name = "chat",
            gen_ai.request.model = call.get_decision().selected_model(),
            gen_ai.request.stream = tracing::field::Empty,
            gen_ai.request.temperature = tracing::field::Empty,
            gen_ai.request.top_p = tracing::field::Empty,
            gen_ai.request.top_k = tracing::field::Empty,
            gen_ai.request.max_tokens = tracing::field::Empty,
            gen_ai.request.reasoning.level = tracing::field::Empty,
            gen_ai.output.type = tracing::field::Empty,
            gen_ai.conversation.id = tracing::field::Empty,
            server.address = tracing::field::Empty,
            server.port = tracing::field::Empty,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
            gen_ai.usage.reasoning.output_tokens = tracing::field::Empty,
            outcome = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
            error.type = tracing::field::Empty,
            error = tracing::field::Empty,
        )
    )]
    async fn serve(call: CallLlmRequest) -> Result<()> {
        let span = tracing::Span::current();
        observability::record_gen_ai_request(&span, &call.get_routed().request.llm_request);
        if let Some(tier) = call.get_decision().routing_tier() {
            span.record("switchyard.routing.tier", tier);
        }
        if let Some(session_id) = call
            .get_routed()
            .request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_id.as_deref())
        {
            span.record("gen_ai.conversation.id", session_id);
        }
        let routed = call.get_routed().clone();
        let target = routed.decision.selected_model().to_string();
        let client = routed
            .default_client
            .clone()
            .ok_or_else(|| LibsyError::MissingClient {
                target: target.clone(),
            })?;
        let result = client
            .call(routed.ctx, routed.request, routed.decision)
            .await
            .map_err(|source| LibsyError::client_call(target, source));
        let result = observability::observe_client_call(result);
        // An algorithm may abandon a call it no longer needs. In that case we just return ok.
        // TODO Ideally we would signal the caller so that the actual client request can be canceld
        match call.respond(result) {
            Err(LibsyError::Driver(DriverError::ResponseDropped)) => Ok(()),
            result => result,
        }
    }

    tokio::pin!(stream);

    let mut trace: Vec<Arc<dyn Decision>> = Vec::new();
    let mut in_flight = futures::stream::FuturesUnordered::new();
    let mut terminal: Option<ResponseOrDecision> = None;

    loop {
        tokio::select! {
            Some(result) = in_flight.next() => match result {
                Ok(()) => {}, // CallLlm completed successfully
                Err(err) => return Err(err), // CallLlm failed, propagate the error
            },
            step = stream.next() => {
                match step {
                    None => break, // stream has ended, no more steps
                    Some(item) => match item?.into() {
                        StepKind::CallLlm(call) => in_flight.push(serve(*call)),
                        StepKind::Decision(decision) => trace.push(decision),
                        StepKind::Terminal(step) => {
                            terminal = Some(step);
                            break;
                        }
                    }
                }
            },
        }
    }
    terminal
        .map(|terminal| (trace, terminal))
        .ok_or(LibsyError::MissingFinalResponse)
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
    pub fn get_target(&self, name: &str) -> Result<LlmTarget> {
        self.targets
            .iter()
            .find(|t| t.semantic_name == name)
            .cloned()
            .ok_or_else(|| LibsyError::TargetNotFound {
                target: name.to_string(),
            })
    }

    /// The named target, or the first one this request is not barred from when it has been
    /// excluded (see [`Context::exclude_target`]). Errors if every target is excluded.
    pub fn resolve_target(&self, name: &str, ctx: &Context) -> Result<LlmTarget> {
        let target = self.get_target(name)?;
        if !ctx.is_excluded(&target.semantic_name) {
            return Ok(target);
        }
        self.targets
            .iter()
            .find(|t| !ctx.is_excluded(&t.semantic_name))
            .cloned()
            .ok_or(LibsyError::AllTargetsExcluded)
    }

    /// The first target's client that can serve `count_tokens` (an Anthropic
    /// upstream), or `None` when no target has one. Used by an algorithm's
    /// [`count_tokens_client`](crate::Algorithm::count_tokens_client).
    pub fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.targets.iter().find_map(|target| {
            target
                .llm_client
                .as_ref()
                .filter(|client| client.supports_count_tokens())
                .cloned()
        })
    }
}

/// Matches the cap the Python stack uses. Dropping a live session's entry costs one
/// rediscovered overflow, so the victim choice does not need to be exact.
const MAX_EVICTION_SESSIONS: usize = 1_024;

/// Per-session record of the targets that overflowed their context window.
///
/// A conversation only grows, so a target that could not fit one turn will not fit a
/// later one; remembering it lets the next turn skip a call certain to fail. Requests
/// without a session id are not tracked — there is nothing to remember them by.
#[derive(Default)]
pub(crate) struct SessionEvictions {
    sessions: Mutex<HashMap<String, HashSet<String>>>,
}

impl SessionEvictions {
    /// The targets `session` has already overflowed; empty for an untracked request.
    fn evicted_in(&self, session: Option<&str>) -> Vec<String> {
        let Some(session) = session else {
            return Vec::new();
        };
        self.sessions
            .lock()
            .get(session)
            .map(|targets| targets.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Remembers that `target` overflowed in `session`, tracking at most
    /// [`MAX_EVICTION_SESSIONS`] sessions.
    fn record(&self, session: Option<&str>, target: &str) {
        let Some(session) = session else { return };
        let mut sessions = self.sessions.lock();
        if sessions.len() >= MAX_EVICTION_SESSIONS
            && !sessions.contains_key(session)
            && let Some(oldest) = sessions.keys().next().cloned()
        {
            sessions.remove(&oldest);
        }
        sessions
            .entry(session.to_string())
            .or_default()
            .insert(target.to_string());
    }
}

/// How many of `targets` this request is still allowed to reach.
fn eligible_targets(targets: &LlmTargetSet, ctx: &Context) -> usize {
    targets
        .targets()
        .iter()
        .filter(|t| !ctx.is_excluded(&t.semantic_name))
        .count()
}

/// Bars the targets `session` has already overflowed from this request, so routing does
/// not select one that is certain to fail again.
pub(crate) fn exclude_evicted(
    ctx: &mut Context,
    targets: &LlmTargetSet,
    evictions: &SessionEvictions,
    session: Option<&str>,
) {
    for target in evictions.evicted_in(session) {
        // Never seed the pool empty: a later turn may be small enough to serve, and the
        // caller should get the upstream's answer rather than a routing error.
        if eligible_targets(targets, ctx) <= 1 {
            break;
        }
        ctx.exclude_target(target);
    }
}

/// Calls `target`, falling back to the next eligible target in `targets` whenever one
/// overflows its context window, until a call succeeds or every target has been tried.
///
/// Routing is deliberately not re-run: the fallback replaces the target in place, so the
/// caller's request-side work and retained state still see exactly one turn.
/// `fallback_decision` builds the [`Decision`] published for a `from -> to` hop, and each
/// overflow is recorded against `session` so later turns skip that target outright.
///
/// Set `final_decision` only when this call's answer is the one returned to the agent: it
/// concludes the run through [`Driver::final_decision`], which honours decision-only mode.
/// A side call — a judge or classifier consultation — passes `false` and is always served,
/// even on a decision-only run, because deciding is what it is for.
///
/// `response` carries a response the caller already obtained; on a decision-only run it is
/// handed back with the decision instead of `target` being called. It is meaningful only
/// alongside `final_decision`; a side call has no answer to hand back and passes `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn call_llm_with_overflow_fallback(
    mut ctx: Context,
    driver: &Driver,
    targets: &LlmTargetSet,
    mut target: LlmTarget,
    mut decision: Arc<dyn Decision>,
    request: Request,
    mut response: Option<Response>,
    session: Option<&str>,
    evictions: &SessionEvictions,
    fallback_decision: impl Fn(&LlmTarget, &LlmTarget) -> Arc<dyn Decision>,
    final_decision: bool,
) -> Result<ResponseOrDecision> {
    loop {
        // The response stays owned here and is only lent to each attempt: `Response` is
        // not `Clone`, so moving it in would leave the next attempt with nothing.
        // `final_decision` takes it only when it concludes the run.
        let result = match final_decision {
            true => {
                driver
                    .final_decision(
                        ctx.clone(),
                        &target,
                        request.clone(),
                        decision.clone(),
                        &mut response,
                    )
                    .await
            }
            false => driver
                .call_llm_target(ctx.clone(), &target, request.clone(), decision.clone())
                .await
                .map(|r| ResponseOrDecision::Response(Box::new(r))),
        };
        let Err(error) = result else { return result };
        let LibsyError::ClientCall {
            target: failed,
            source: LlmClientError::ContextWindowExceeded { .. },
        } = &error
        else {
            return Err(error);
        };
        // A target already excluded means the pool is spent; surface the client error
        // so the caller still sees a context overflow rather than an internal failure.
        if !ctx.exclude_target(failed) {
            return Err(error);
        }
        evictions.record(session, failed);
        let Ok(next) = targets.resolve_target(&target.semantic_name, &ctx) else {
            return Err(error);
        };
        decision = fallback_decision(&target, &next);
        target = next;
        driver.info(ctx.clone(), decision.clone()).await?;
    }
}
struct AlgoInner<A: Algorithm + ?Sized> {
    algo: Arc<A>,
}
impl<A: Algorithm + ?Sized> AlgoInner<A> {
    fn new(algo: Arc<A>) -> Self {
        Self { algo }
    }
    /// Process a request to completion, returning a stream of [`Step`]s.
    /// Each [`Step::CallLlm`] is an offloaded model call the consumer must serve.
    /// The stream ends with a [`Step::ReturnToAgent`] on success, or an `Err` item on failure.
    /// Report each model call to `observer`.
    fn run_stream_inner(
        &self,
        ctx: Context,
        request: Request,
        decision_only: bool,
        observer: Option<RunObserver>,
    ) -> AnyStepStream {
        // Stamp the algorithm's telemetry label into the request context; the
        // context rides on every driver call, so its telemetry is attributed.
        let mut ctx = ctx;
        ctx.values.insert(
            observability::ALGORITHM_KEY.to_string(),
            self.algo.name().to_string(),
        );
        let driver = Driver::with_observer(decision_only, observer);
        let task_driver = driver.clone();
        let task_ctx = ctx.clone();
        // Take the consumer stream before the task starts: the driver refuses to publish a
        // step until a consumer exists. The two shapes read the same driver, so exactly one
        // of them may be taken. Boxing here erases the two opaque stream types to the one
        // the variant holds.
        let steps = if decision_only {
            AnyStepStream::DecisionOnly(Box::pin(task_driver.decision_only_stream()))
        } else {
            AnyStepStream::Step(Box::pin(task_driver.stream()))
        };
        // One `libsy.run` span covers the whole algorithm task; the driver's
        // `libsy.llm_call` spans and decision logs nest inside it via `tracing`'s
        // contextual parenting.
        let span = observability::run_span(self.algo.name(), &request);
        let observed_driver = task_driver.clone();
        let algo_task = self
            .algo
            .clone()
            .create_run_task(task_ctx.clone(), task_driver, request);
        let handle = tokio::spawn(
            async move {
                observability::observe_run(task_ctx.clone(), observed_driver, async move {
                    //let terminal = self.algo.clone().create_run_task(task_ctx, task_driver, request).await?;
                    let terminal = algo_task.await?;
                    check_response_type(terminal, decision_only)
                })
                .await
            }
            .instrument(span),
        );
        // Dropping the stream aborts the algorithm task, so it doesn't keep running after the
        let abort_guard = AbortOnDrop(handle.abort_handle());

        let finish_driver = driver.clone();
        let finish_ctx = ctx;

        // awaits create_run_task append terminal to driver and appends err to step stream
        fn tail<T: Send + 'static>(
            handle: tokio::task::JoinHandle<Result<ResponseOrDecision>>,
            driver: Driver,
            ctx: Context,
        ) -> impl Stream<Item = Result<T>> + Send {
            futures::stream::once(async move {
                let result = match handle.await {
                    Ok(response) => response,
                    Err(source) => Err(LibsyError::AlgorithmTask { source }),
                };
                driver.finish(ctx, result).await
            })
            .filter_map(|finish_result| async move { finish_result.err().map(Err) })
        }

        // merge step stream and tail
        fn merge<T: Send + 'static>(
            steps: impl Stream<Item = Result<T>> + Send + 'static,
            tail: impl Stream<Item = Result<T>> + Send + 'static,
            guard: AbortOnDrop,
        ) -> Pin<Box<dyn Stream<Item = Result<T>> + Send>> {
            Box::pin(futures::stream::select(steps, tail).map(move |step| {
                // link abort guard to stream
                let _keep_alive = &guard;
                step
            }))
        }

        match steps {
            AnyStepStream::Step(steps) => AnyStepStream::Step(merge(
                steps,
                tail(handle, finish_driver, finish_ctx),
                abort_guard,
            )),
            AnyStepStream::DecisionOnly(steps) => AnyStepStream::DecisionOnly(merge(
                steps,
                tail(handle, finish_driver, finish_ctx),
                abort_guard,
            )),
        }
    }

    /// Process a request to completion, returning a stream of [`Step`]s.
    fn run_stream(
        &self,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> StepStream {
        match self.run_stream_inner(ctx, request, false, observer) {
            AnyStepStream::Step(stream) => stream,
            AnyStepStream::DecisionOnly(_) => futures::stream::once(async move {
                Err(LibsyError::AlgorithmError {
                    message: "run_stream_inner with decision_only=false should return StepStream"
                        .to_string(),
                })
            })
            .boxed(),
        }
    }

    /// Process a request up to its final routing decision as [`DecisionOnlyStep`]s.
    fn run_decision_only_stream(
        &self,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> DecisionOnlyStepStream {
        match self.run_stream_inner(ctx, request, true, observer) {
            AnyStepStream::DecisionOnly(stream) => stream,
            AnyStepStream::Step(_) => futures::stream::once(async move {
                Err(LibsyError::AlgorithmError {
                    message: "run_stream_inner with decision_only=true should return \
                                  DecisionOnlyStepStream"
                        .to_string(),
                })
            })
            .boxed(),
        }
    }

    /// Process a request to completion, serving every offloaded call, and return the
    /// terminal [`ResponseOrDecision`] plus the trace of [`Decision`]s made along the way.
    ///
    /// Both stream shapes drive identically — only their terminal step differs, and the
    /// shared driving loop normalizes that — so this is one match over the two.
    async fn run_inner(
        &self,
        ctx: Context,
        request: Request,
        decision_only: bool,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, ResponseOrDecision)> {
        match decision_only {
            true => drive(self.run_decision_only_stream(ctx, request, observer)).await,
            false => drive(self.run_stream(ctx, request, observer)).await,
        }
    }

    /// Process a request to completion, returning the final [`Response`] and the trace of
    /// [`Decision`]s the algorithm made along the way.
    async fn run(
        &self,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
        self.run_observed(ctx, request, observer).await
    }

    /// Process a request to completion while reporting each model call to `observer`.
    async fn run_observed(
        &self,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
        let (trace, response) = self.run_inner(ctx, request, false, observer).await?;
        match response {
            ResponseOrDecision::Response(response) => Ok((trace, *response)),
            ResponseOrDecision::Decision(_) => Err(LibsyError::AlgorithmError {
                message: DECISION_IN_SERVED_RUN.to_string(),
            }),
        }
    }

    /// Process a request up to its final routing decision *without* serving that call:
    /// returns the decision, the request to serve it with, and any response the algorithm
    /// already obtained, plus the trace of decisions made along the way.
    ///
    /// Only the routed call is left unmade: the decision still binds the algorithm's
    /// retained state — session affinity latches this session to the target it chose —
    /// exactly as a served run would. Deciding commits to a route, it does not preview one.
    async fn decide(
        &self,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, DecidedCall)> {
        let (trace, response) = self.run_inner(ctx, request, true, observer).await?;
        match response {
            ResponseOrDecision::Decision(decided) => Ok((trace, decided)),
            ResponseOrDecision::Response(_) => Err(LibsyError::AlgorithmError {
                message: RESPONSE_IN_DECISION_ONLY_RUN.to_string(),
            }),
        }
    }
}

/// An optimization strategy. Implement [`create_run_task`](Self::create_run_task);
/// callers drive it with the provided [`run`](Self::run) (serve calls, get the answer)
/// or [`run_stream`](Self::run_stream) (drive the [`Step`] stream yourself).
///
/// Methods take `self: Arc<Self>`: one algorithm (`Arc<dyn Algorithm>`) is shared across
/// requests and run concurrently, so it owns its thread-safety and any shared state.
#[async_trait]
pub trait Algorithm: Send + Sync + 'static {
    /// Stable, low-cardinality name identifying this algorithm — the
    /// `algorithm` attribute on every span, metric, and log line the crate
    /// emits for its runs (see the crate docs' Observability section).
    fn name(&self) -> &str;

    /// Run one request to completion: make model calls with [`Driver::call_llm_target`],
    /// publish [`Decision`]s with [`Driver::info`], and conclude on the winning target with
    /// [`Driver::final_decision`].
    /// The method an algorithm implements; [`run`](Self::run) / [`run_stream`](Self::run_stream)
    /// drive it. `ctx` carries the request's cross-cutting values (today: the
    /// algorithm's telemetry label in [`Context::values`]).
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<ResponseOrDecision>;

    /// Feed the algorithm agentic-stack events (tool results, budgets, etc.). The
    /// reference algorithms ignore signals; a stateful algorithm updates its own
    /// (interior-mutable) state. Takes `self: Arc<Self>` like the other run methods.
    #[allow(unused_variables)]
    async fn process_signals(self: Arc<Self>, signals: Signals) -> Result<()> {
        Ok(())
    }

    /// The client [`count_tokens`](Self::count_tokens) forwards to: the first of
    /// this algorithm's targets whose client can count tokens (an Anthropic
    /// upstream). The default is `None` — an algorithm with no Anthropic target
    /// does not support token counting.
    ///
    /// CAVEAT: this picks the *first* Anthropic target, not a routed one —
    /// count_tokens is a direct passthrough, so it does not run the routing
    /// cascade. For a route with several Anthropic tiers "first" is arbitrary;
    /// choosing which tier count_tokens should reflect is deferred.
    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        None
    }

    /// Count the tokens `request` would use — a **direct passthrough** to this
    /// algorithm's Anthropic target (via
    /// [`count_tokens_client`](Self::count_tokens_client)), **not** a routed
    /// call. Token counting is a pre-flight estimate with no routing decision,
    /// so it deliberately bypasses the classifier cascade (which runs only for
    /// completions via [`run`](Self::run)). Returns the upstream's JSON
    /// verbatim. Errors when the algorithm has no Anthropic target.
    async fn count_tokens(&self, request: Request) -> Result<serde_json::Value> {
        let client = self
            .count_tokens_client()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "no target supports count_tokens (needs an Anthropic upstream)"
                    .to_string(),
            })?;
        client
            .count_tokens(request)
            .await
            .map_err(|source| LibsyError::client_call("count_tokens", source))
    }

    /// Process a request to completion, returning a stream of [`Step`]s.
    fn run_stream(
        self: Arc<Self>,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> StepStream {
        AlgoInner::new(self).run_stream(ctx, request, observer)
    }

    /// Process a request up to its final routing decision as [`DecisionOnlyStep`]s.
    fn run_decision_only_stream(
        self: Arc<Self>,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> DecisionOnlyStepStream {
        AlgoInner::new(self).run_decision_only_stream(ctx, request, observer)
    }

    /// Process a request to completion, returning the final [`Response`] and the trace of
    /// [`Decision`]s the algorithm made along the way.
    async fn run(
        self: Arc<Self>,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
        AlgoInner::new(self).run(ctx, request, observer).await
    }

    /// Process a request up to its final routing decision *without* serving that call:
    /// returns the decision, the request to serve it with, and any response the algorithm
    /// already obtained, plus the trace of decisions made along the way.
    ///
    /// Only the routed call is left unmade: the decision still binds the algorithm's
    /// retained state — session affinity latches this session to the target it chose —
    /// exactly as a served run would. Deciding commits to a route, it does not preview one.
    async fn decide(
        self: Arc<Self>,
        ctx: Context,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<(Vec<Arc<dyn Decision>>, DecidedCall)> {
        AlgoInner::new(self).decide(ctx, request, observer).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use switchyard_protocol::{
        LlmResponse, LlmResponseChunk, completion_text, text_request, text_response,
    };

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct TestError(&'static str);

    fn test_error(message: &'static str) -> LibsyError {
        LibsyError::external("test", TestError(message))
    }

    /// Mock client that echoes back the target name it was called with.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
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
        fn name(&self) -> &str {
            "test"
        }

        async fn create_run_task(
            self: Arc<Self>,
            ctx: Context,
            driver: Driver,
            request: Request,
        ) -> Result<ResponseOrDecision> {
            let target = self
                .target_set
                .targets()
                .first()
                .ok_or(LibsyError::NoTargets)?
                .clone();
            let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                model: target.semantic_name.clone(),
            });
            driver.info(ctx.clone(), decision.clone()).await?;
            driver
                .final_decision(ctx, &target, request, decision, &mut None)
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

    #[tokio::test]
    async fn observed_run_reports_one_successful_routed_call() -> Result<()> {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observed = observations.clone();
        let observer: RunObserver = Arc::new(move |observation| observed.lock().push(observation));
        let (_, response) = orch(target_set(&[("direct/model", true)]))
            .run(Context::default(), request(), Some(observer))
            .await?;
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("direct/model".to_string())
        );
        let observations = observations.lock();
        assert_eq!(observations.len(), 2);
        let RunObservation::LlmCall(observation) = &observations[0] else {
            return Err(test_error("expected an LLM call observation"));
        };
        assert_eq!(observation.selected_model, "direct/model");
        assert!(observation.is_routed);
        assert!(observation.is_success);
        assert!(observation.usage.is_some());
        assert!(matches!(
            observations[1],
            RunObservation::RoutingOverhead(_)
        ));
        Ok(())
    }

    #[test]
    fn target_lookup_returns_the_missing_target() {
        let error = target_set(&[]).get_target("missing").err();
        assert!(matches!(
            error,
            Some(LibsyError::TargetNotFound { target }) if target == "missing"
        ));
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
        ) -> std::result::Result<Response, LlmClientError> {
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
    async fn run_returns_a_streamed_response_the_caller_aggregates() -> Result<()> {
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
        let (trace, response) = orch.run(Context::default(), request(), None).await?;
        // `run` handed back the live stream; the caller folds it to a buffered aggregate.
        let agg = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::external("aggregating response stream", error))?;
        assert_eq!(completion_text(&agg), "hello");
        assert_eq!(agg.model.as_deref(), Some("stream/model"));
        assert_eq!(trace.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn aggregating_a_streamed_response_propagates_a_mid_stream_error() -> Result<()> {
        // `run` succeeds and returns the stream; the in-band `Error` chunk surfaces only
        // when the caller aggregates it.
        let orch = streaming_orch(vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "partial".to_string(),
            },
            LlmResponseChunk::StreamError {
                message: "upstream exploded".to_string(),
            },
        ]);
        let (_, response) = orch.run(Context::default(), request(), None).await?;
        match response.llm_response.into_agg().await {
            Ok(_) => panic!("expected a mid-stream error, got an aggregate"),
            Err(err) => {
                assert!(err.to_string().contains("upstream exploded"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn run_offloads_via_promise_then_returns_to_agent() -> Result<()> {
        // A client-less target -> its call is offloaded via a promise the
        // orchestrator surfaces as a `CallLlm` step for us to fulfill.
        let stream = orch(target_set(&[("offload/model", false)])).run_stream(
            Context::default(),
            request(),
            None,
        );
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
            final_completion.ok_or_else(|| test_error("no ReturnToAgent step"))?,
            "fulfilled"
        );
        Ok(())
    }

    #[tokio::test]
    async fn client_backed_target_offloads_with_a_default_client() -> Result<()> {
        // Every call now offloads to the stream; a client-backed target rides its
        // client along as `default_client` so the consumer can serve it by default.
        let stream = orch(target_set(&[("direct/model", true)])).run_stream(
            Context::default(),
            request(),
            None,
        );
        tokio::pin!(stream);

        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallLlm(call) => {
                    let routed = call.get_routed().clone();
                    let client = routed
                        .default_client
                        .clone()
                        .ok_or_else(|| test_error("expected a default client"))?;
                    let target = routed.decision.selected_model().to_string();
                    let result = client
                        .call(routed.ctx, routed.request, routed.decision)
                        .await
                        .map_err(|error| LibsyError::client_call(target, error));
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
        assert_eq!(
            final_completion.ok_or_else(|| test_error("no ReturnToAgent"))?,
            "direct/model"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_the_response_when_all_targets_have_clients() -> Result<()> {
        // Every target has a client, so run serves every call via the
        // default client and returns the trace + final response.
        let (trace, response) = orch(target_set(&[("direct/model", true)]))
            .run(Context::default(), request(), None)
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
    async fn concluding_through_final_decision_honours_the_run_mode() -> Result<()> {
        // `TestAlgo` never mentions the mode: it just ends on `Driver::final_decision`.
        // That alone is enough to support both entry points — `run` gets the served
        // response, `decide` gets the route with the call left unmade.
        let algo = orch(target_set(&[("direct/model", true)]));

        let (_, response) = algo
            .clone()
            .run(Context::default(), request(), None)
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "direct/model"
        );

        let (_, (decision, _, served)) = algo.decide(Context::default(), request(), None).await?;
        assert_eq!(decision.selected_model(), "direct/model");
        assert!(served.is_none(), "the final call was handed back, not made");
        Ok(())
    }

    #[tokio::test]
    async fn decide_errors_when_the_algorithm_cannot_hand_the_call_back() -> Result<()> {
        // `Noop` answers without ever routing a call, so it builds its response directly
        // instead of concluding through `final_decision` — it has no route to hand back.
        // The mismatch is caught where the terminal payload is decoded, so `decide` never
        // sees a response to mislabel.
        let algo: Arc<dyn Algorithm> = Arc::new(crate::Noop {});
        let error = algo
            .decide(Context::default(), request(), None)
            .await
            .err()
            .ok_or_else(|| test_error("expected a decision-only mismatch"))?;
        assert!(matches!(
            error,
            LibsyError::AlgorithmError { message } if message.contains("decision-only step stream")
        ));
        Ok(())
    }

    #[tokio::test]
    async fn the_driver_reports_the_mode_the_run_was_started_in() -> Result<()> {
        // The mode is fixed by the entry point and read back by the algorithm, so a
        // composition cannot disagree with the stream shape it is being driven as.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ModeRecording(Arc<AtomicBool>);

        #[async_trait]
        impl Algorithm for ModeRecording {
            fn name(&self) -> &str {
                "mode_recording"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                driver: Driver,
                _request: Request,
            ) -> Result<ResponseOrDecision> {
                self.0.store(driver.decision_only(), Ordering::SeqCst);
                Err(test_error("stop after recording the mode"))
            }
        }

        for expected in [false, true] {
            let seen = Arc::new(AtomicBool::new(!expected));
            let algo: Arc<dyn Algorithm> = Arc::new(ModeRecording(seen.clone()));
            // Both entry points fail here by design; only the recorded mode matters.
            let _ = AlgoInner::new(algo)
                .run_inner(Context::default(), request(), expected, None)
                .await;
            assert_eq!(seen.load(Ordering::SeqCst), expected);
        }
        Ok(())
    }

    #[tokio::test]
    async fn concluding_with_an_answer_already_obtained_does_not_call_again() -> Result<()> {
        // The target has no client, so any outbound call fails: the run can only succeed by
        // concluding on the response the algorithm already had.
        struct ConcludesWithServedAnswer(LlmTargetSet);

        #[async_trait]
        impl Algorithm for ConcludesWithServedAnswer {
            fn name(&self) -> &str {
                "concludes-with-served-answer"
            }

            async fn create_run_task(
                self: Arc<Self>,
                ctx: Context,
                driver: Driver,
                request: Request,
            ) -> Result<ResponseOrDecision> {
                let target = self.0.get_target("offload/model")?;
                let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                    model: target.semantic_name.clone(),
                });
                let mut served = Some(Response {
                    llm_response: LlmResponse::Agg(text_response(None, "already answered")),
                    metadata: None,
                });
                driver
                    .final_decision(ctx, &target, request, decision, &mut served)
                    .await
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(ConcludesWithServedAnswer(target_set(&[(
            "offload/model",
            false,
        )])));
        let (_trace, response) = algo.run(Context::default(), request(), None).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "already answered"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_errors_when_a_target_lacks_a_client() -> Result<()> {
        // A client-less target has no default client to serve its offloaded call, so
        // driving it to completion errors.
        let error = orch(target_set(&[("offload/model", false)]))
            .run(Context::default(), request(), None)
            .await
            .err()
            .ok_or_else(|| test_error("expected a missing-client error"))?;
        assert!(matches!(
            error,
            LibsyError::MissingClient { target } if target == "offload/model"
        ));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 12)]
    async fn requests_are_processed_in_parallel() -> Result<()> {
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
            ) -> std::result::Result<Response, LlmClientError> {
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
                algo.run(Context::default(), request(), None)
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
            let completion = tokio::time::timeout(Duration::from_secs(5), handle)
                .await
                .map_err(|error| LibsyError::external("waiting for test task", error))?
                .map_err(|source| LibsyError::AlgorithmTask { source })??;
            assert_eq!(completion, "m");
        }
        Ok(())
    }

    #[tokio::test]
    async fn offload_error_propagates_back_to_the_algorithm() -> Result<()> {
        // A client-less target offloads its call; we fulfill the promise with an
        // Err, which must flow back through `call_llm_target` into the algorithm and
        // out as an error step — not a response.
        let stream = orch(target_set(&[("offload/model", false)])).run_stream(
            Context::default(),
            request(),
            None,
        );
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(Step::CallLlm(call)) => {
                    call.respond(Err(test_error("upstream model call failed")))?;
                }
                Ok(Step::Decision(_)) => {}
                Ok(Step::ReturnToAgent(..)) => {
                    return Err(test_error(
                        "expected the offload error to propagate, got a response",
                    ));
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
    async fn dropping_the_stream_cancels_the_algorithm_task() -> Result<()> {
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
            fn name(&self) -> &str {
                "stuck"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<ResponseOrDecision> {
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

        let stream = algo.run_stream(Context::default(), request(), None);
        started_rx
            .recv()
            .await
            .ok_or_else(|| test_error("task never started"))?;
        drop(stream);
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after dropping the stream"
        );
        Ok(())
    }

    #[tokio::test]
    async fn create_run_task_panic_surfaces_as_a_stream_error() -> Result<()> {
        // An algorithm whose task panics must surface an `Err` step to the stream
        // consumer, not abort the process from an unobserved detached task.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            fn name(&self) -> &str {
                "panicky"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<ResponseOrDecision> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        let stream = algo.run_stream(Context::default(), request(), None);
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Err(err) => {
                    assert!(matches!(err, LibsyError::AlgorithmTask { .. }));
                    saw_error = true;
                }
                Ok(_) => return Err(test_error("expected the panic to surface as an error step")),
            }
        }

        assert!(saw_error, "expected an error step from the panicked task");
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_an_error_when_the_algorithm_task_panics() -> Result<()> {
        // The panic surfaces as an `Err` step inside `run_stream`; `run` propagates it
        // via `?`, so the caller gets an `Err` rather than a hang or a silent panic.
        struct Panicky;

        #[async_trait]
        impl Algorithm for Panicky {
            fn name(&self) -> &str {
                "panicky"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<ResponseOrDecision> {
                panic!("boom");
            }
        }

        let algo: Arc<dyn Algorithm> = Arc::new(Panicky);
        match algo.run(Context::default(), request(), None).await {
            Ok(_) => Err(test_error(
                "expected run to surface the algorithm panic as an error",
            )),
            Err(err) => {
                assert!(matches!(err, LibsyError::AlgorithmTask { .. }));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn cancelling_run_cancels_the_algorithm_task() -> Result<()> {
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
            fn name(&self) -> &str {
                "stuck"
            }

            async fn create_run_task(
                self: Arc<Self>,
                _ctx: Context,
                _driver: Driver,
                _request: Request,
            ) -> Result<ResponseOrDecision> {
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
        let run_task =
            tokio::spawn(async move { algo.run(Context::default(), request(), None).await });
        started_rx
            .recv()
            .await
            .ok_or_else(|| test_error("task never started"))?;
        run_task.abort();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            dropped.load(Ordering::SeqCst),
            "algorithm task was NOT cancelled after cancelling run"
        );
        Ok(())
    }

    // --- first-wins hedging: `run` must not wait on losing speculative calls -------------

    /// The loser: signals it has started serving (so the winner can then win with the
    /// loser's serve guaranteed in flight), then finishes late (`Some(delay)`) or never
    /// (`None`).
    struct LoserClient {
        started: Arc<tokio::sync::Notify>,
        delay: Option<std::time::Duration>,
    }

    #[async_trait]
    impl RoutedLlmClient for LoserClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            self.started.notify_one();
            match self.delay {
                Some(delay) => tokio::time::sleep(delay).await,
                None => std::future::pending::<()>().await,
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
                metadata: None,
            })
        }
    }

    /// The winner: waits until the loser's serve has started, then echoes immediately, so
    /// the loser's serve is guaranteed in flight when the winner wins.
    struct GatedEchoClient {
        gate: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl RoutedLlmClient for GatedEchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            self.gate.notified().await;
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
                metadata: None,
            })
        }
    }

    /// Offloads two targets concurrently and returns the first to resolve, dropping the
    /// loser's call (first-wins hedging).
    struct Hedge {
        winner: LlmTarget,
        loser: LlmTarget,
        /// Work the algorithm does after the race, before it concludes.
        post_select_delay: Option<std::time::Duration>,
    }

    #[async_trait]
    impl Algorithm for Hedge {
        fn name(&self) -> &str {
            "hedge"
        }

        async fn create_run_task(
            self: Arc<Self>,
            ctx: Context,
            driver: Driver,
            request: Request,
        ) -> Result<ResponseOrDecision> {
            let dec_w: Arc<dyn Decision> = Arc::new(TestDecision {
                model: self.winner.semantic_name.clone(),
            });
            let dec_l: Arc<dyn Decision> = Arc::new(TestDecision {
                model: self.loser.semantic_name.clone(),
            });
            let win = driver.call_llm_target(ctx.clone(), &self.winner, request.clone(), dec_w);
            let lose = driver.call_llm_target(ctx, &self.loser, request, dec_l);
            // First to resolve wins; `select!` drops the losing future (and its promise).
            let winner = tokio::select! {
                res = win => res,
                res = lose => res,
            };
            if let Some(delay) = self.post_select_delay {
                tokio::time::sleep(delay).await;
            }
            Ok(ResponseOrDecision::Response(Box::new(winner?)))
        }
    }

    /// Builds a hedging algo whose winner is gated behind the loser starting, and whose
    /// loser finishes after `loser_delay` (or never, when `None`).
    fn hedge(loser_delay: Option<std::time::Duration>) -> Arc<dyn Algorithm> {
        hedge_concluding_after(loser_delay, None)
    }

    /// A hedge that keeps working for `post_select_delay` after the race, so a late loser
    /// resolves into its dropped promise while the run is still in progress.
    fn hedge_concluding_after(
        loser_delay: Option<std::time::Duration>,
        post_select_delay: Option<std::time::Duration>,
    ) -> Arc<dyn Algorithm> {
        let started = Arc::new(tokio::sync::Notify::new());
        let winner = LlmTarget {
            semantic_name: "winner".to_string(),
            llm_client: Some(Arc::new(GatedEchoClient {
                gate: started.clone(),
            })),
        };
        let loser = LlmTarget {
            semantic_name: "loser".to_string(),
            llm_client: Some(Arc::new(LoserClient {
                started,
                delay: loser_delay,
            })),
        };
        Arc::new(Hedge {
            winner,
            loser,
            post_select_delay,
        })
    }

    #[tokio::test]
    async fn run_returns_the_winner_without_a_late_loser_overwriting_it() -> Result<()> {
        // The loser responds 50ms after the winner has already won. `run` must return the
        // winner, not the loser's `respond`-to-a-dropped-receiver error.
        let (_trace, response) = hedge(Some(std::time::Duration::from_millis(50)))
            .run(Context::default(), request(), None)
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "winner"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_late_loser_resolving_into_a_dropped_promise_does_not_fail_the_run() -> Result<()> {
        // Abandoning a call is how an algorithm hedges, so the response the loser resolves
        // into a dropped promise is discarded rather than failing a run the winner already
        // answered. Unlike the test above, this hedge keeps working after the race, so that
        // discarded response reaches the consumer well before the terminal step instead of
        // racing it.
        let algorithm = hedge_concluding_after(
            Some(std::time::Duration::from_millis(20)),
            Some(std::time::Duration::from_millis(200)),
        );
        let (_trace, response) = algorithm.run(Context::default(), request(), None).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "winner"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_returns_the_winner_without_hanging_on_a_pending_loser() -> Result<()> {
        // The loser never resolves. `run` must return the winner promptly, not hang
        // waiting for the in-flight loser.
        let run = hedge(None).run(Context::default(), request(), None);
        let (_trace, response) = tokio::time::timeout(std::time::Duration::from_secs(1), run)
            .await
            .map_err(|error| LibsyError::external("waiting for pending loser", error))??;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "winner"
        );
        Ok(())
    }

    #[tokio::test]
    async fn run_surfaces_a_terminal_error_with_many_calls_in_flight() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A large fan-out (10 matched the old, now-removed concurrency cap). The terminal
        // error must still reach the caller with all of these calls pending.
        const N: usize = 10;

        // Enters each call; once all N are in flight, signals, then pends forever.
        struct EnterThenPend {
            started: Arc<AtomicUsize>,
            all_started: Arc<tokio::sync::Notify>,
            n: usize,
        }

        #[async_trait]
        impl RoutedLlmClient for EnterThenPend {
            async fn call(
                &self,
                _ctx: Context,
                _request: Request,
                _decision: Arc<dyn Decision>,
            ) -> std::result::Result<Response, LlmClientError> {
                if self.started.fetch_add(1, Ordering::SeqCst) + 1 == self.n {
                    self.all_started.notify_one();
                }
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        // Fans out N calls, then errors as soon as all N are in flight — exercising a
        // terminal failure emitted while the offloaded calls are still pending.
        struct FanOutThenError {
            target: LlmTarget,
            all_started: Arc<tokio::sync::Notify>,
            n: usize,
        }

        #[async_trait]
        impl Algorithm for FanOutThenError {
            fn name(&self) -> &str {
                "fan_out_then_error"
            }

            async fn create_run_task(
                self: Arc<Self>,
                ctx: Context,
                driver: Driver,
                request: Request,
            ) -> Result<ResponseOrDecision> {
                let offloads = futures::future::join_all((0..self.n).map(|i| {
                    let decision: Arc<dyn Decision> = Arc::new(TestDecision {
                        model: format!("m{i}"),
                    });
                    driver.call_llm_target(ctx.clone(), &self.target, request.clone(), decision)
                }));
                tokio::select! {
                    _ = offloads => Err(test_error("offloads unexpectedly completed")),
                    _ = self.all_started.notified() => {
                        Err(test_error("terminal error while calls pending"))
                    }
                }
            }
        }

        let all_started = Arc::new(tokio::sync::Notify::new());
        let target = LlmTarget {
            semantic_name: "pending".to_string(),
            llm_client: Some(Arc::new(EnterThenPend {
                started: Arc::new(AtomicUsize::new(0)),
                all_started: all_started.clone(),
                n: N,
            })),
        };
        let algo: Arc<dyn Algorithm> = Arc::new(FanOutThenError {
            target,
            all_started,
            n: N,
        });

        // With the cap gone, `run` keeps polling the stream even with N calls in flight, so
        // the terminal error surfaces promptly instead of hanging.
        let run = algo.run(Context::default(), request(), None);
        let result = tokio::time::timeout(std::time::Duration::from_millis(500), run)
            .await
            .map_err(|error| {
                LibsyError::external("waiting for terminal error with full call cap", error)
            })?;
        match result {
            Ok(_) => Err(test_error("expected the terminal error, got a response")),
            Err(err) => {
                assert!(
                    err.to_string()
                        .contains("terminal error while calls pending")
                );
                Ok(())
            }
        }
    }
}

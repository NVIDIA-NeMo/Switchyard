// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drive a libsy algorithm to completion, making the model calls it offloads.
//!
//! [`switchyard_libsy::Algorithm::run_stream`] is the whole libsy API: it yields a stream of
//! steps and expects its consumer to serve every offloaded model call. [`run()`] is that
//! consumer — it drives the stream with [`switchyard_libsy::drive`], hands routing-time calls to
//! a [`RoutedLlmClient`], and serves the terminal routing outcome.
//!
//! libsy owns the stream mechanics; what this module adds is per-target request preparation,
//! ordered candidate fallback, and the `libsy.client_call` span around each candidate. Each
//! completion candidate exhausts its backend retry budget before fallback advances. Routing
//! calls stop on the first candidate's failure. A timeout stops either kind of call.
//!
//! A Responses continuation can refer to state through `previous_response_id` or `conversation`.
//! [`ClientRouter`] sends native Responses state back to its provider; for Chat or Anthropic
//! targets it retains the canonical history needed to materialize the continuation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::{StreamExt, TryStreamExt, stream};
use http::StatusCode;
use parking_lot::Mutex;
use serde_json::{Value, json};
use switchyard_libsy::{
    Algorithm, CallModel, LibsyError, OutcomeMetadata, Result, RoutingOutcome, RuntimeModels, drive,
};
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStream, Message,
    ModelId, Request, Response, ResponseAccumulator, RoutedLlmClient, RoutingFallbackReason,
    WireFormat,
};
use switchyard_translation::prepare_request_for_target;

use crate::observation::{LlmCallObservation, RunObservation, RunObserver};
use crate::{metrics, observability};

/// Run one request to completion, serving every offloaded model call with `client`.
///
/// Returns the model selected by the algorithm and the final [`Response`]. `observer`, when
/// present, receives metadata for the routing outcome, each completed routing or answer call,
/// and the routing overhead.
///
/// `clients` resolves each offloaded call to the client for the target the algorithm
/// selected — an algorithm may route among targets served by different providers, so this is
/// a per-call lookup, not one client for the whole run. Use
/// [`ClientRouter::single`](ClientRouter::single) when one client serves every target.
///
/// Routing calls are buffered; a client failure stops the request before the algorithm continues.
/// Once routing completes, non-timeout failures may try the outcome's ordered fallback candidates.
pub async fn run(
    algorithm: Arc<dyn Algorithm>,
    clients: ClientRouter,
    request: Request,
    models: Arc<RuntimeModels>,
    observer: Option<RunObserver>,
) -> Result<(ModelId, Response)> {
    let algorithm_name = algorithm.name().to_string();
    let run_started = Instant::now();
    let routing_clients = clients.clone();
    // This says if we have an observer, put Some(..) in routing_observations.
    // No observer means we don't want any routing_observations.
    let routing_observations = observer.as_ref().map(|_| Arc::new(Mutex::new(Vec::new())));
    let outcome = match clients.stored_state_owner(&request) {
        Some(owner) => Ok(continue_on(owner, &algorithm_name, request)),
        None => {
            drive(algorithm, request, models, {
                let routing_observations = routing_observations.clone();
                move |call| serve(routing_clients.clone(), call, routing_observations.clone())
            })
            .await
        }
    };
    let answered_model = outcome
        .as_ref()
        .ok()
        .and_then(|outcome| outcome.response.as_ref())
        .and_then(Response::served_model);
    emit_routing_observations(&observer, &routing_observations, answered_model);
    let outcome = outcome?;
    let overhead = run_started.elapsed();
    metrics::record_routing_overhead(&algorithm_name, overhead);

    let selected_model_id = outcome.selected_model_id()?.clone();
    if let Some(observer) = &observer
        && let Some(metadata) = outcome.metadata
    {
        observer(RunObservation::Outcome(metadata));
    }
    let result = if let Some(response) = outcome.response {
        let result = clients.remember_state_owner(&outcome.request, response);
        let served_model = result
            .as_ref()
            .ok()
            .and_then(Response::served_model)
            .unwrap_or(&selected_model_id);
        let served_model = served_model.clone();
        metrics::observe_routed_request(&algorithm_name, &served_model, None, result)
    } else {
        let observe = |observation| {
            if let Some(observer) = &observer {
                observer(RunObservation::AnswerCall(observation));
            }
        };
        call_first_available(
            &clients,
            &algorithm_name,
            &outcome.request,
            &outcome.selected_model_ids,
            &observe,
        )
        .await
        .and_then(|response| clients.remember_state_owner(&outcome.request, response))
    };
    if let Some(observer) = &observer {
        observer(RunObservation::RoutingOverhead(overhead));
    }
    result.map(|response| (selected_model_id, response))
}

/// Run an algorithm to a routing decision without serving its terminal completion.
///
/// Routing-time calls are served normally. The returned request is prepared only for the selected
/// target; use [`run`] to execute selected and fallback candidates.
pub async fn decide(
    algorithm: Arc<dyn Algorithm>,
    clients: ClientRouter,
    request: Request,
    models: Arc<RuntimeModels>,
) -> Result<RoutingOutcome> {
    let routing_clients = clients.clone();
    let mut outcome = match clients.stored_state_owner(&request) {
        Some(owner) => continue_on(owner, algorithm.name(), request),
        None => {
            drive(algorithm, request, models, move |call| {
                serve(routing_clients.clone(), call, None)
            })
            .await?
        }
    };
    let selected_model_id = outcome.selected_model_id()?.clone();
    outcome.request = clients.prepare_completion_request(outcome.request, &selected_model_id);
    outcome.response = outcome
        .response
        .map(|response| clients.remember_state_owner(&outcome.request, response))
        .transpose()?;
    Ok(outcome)
}

/// Emits completed routing calls after the outcome reveals whether one response became the answer.
fn emit_routing_observations(
    observer: &Option<RunObserver>,
    observations: &Option<Arc<Mutex<Vec<LlmCallObservation>>>>,
    answered_model: Option<&ModelId>,
) {
    let (Some(observer), Some(observations)) = (observer, observations) else {
        return;
    };
    let mut answer_observed = false;
    for observation in observations.lock().drain(..) {
        if !answer_observed && answered_model == Some(&observation.selected_model) {
            answer_observed = true;
            observer(RunObservation::AnswerCall(observation));
        } else {
            observer(RunObservation::LlmCall(observation));
        }
    }
}

/// Serve one offloaded call and fulfill its promise.
///
/// A client failure stops the driver before the algorithm can issue another call.
async fn serve(
    clients: ClientRouter,
    call: CallModel,
    observations: Option<Arc<Mutex<Vec<LlmCallObservation>>>>,
) -> Result<()> {
    let observe = |observation| {
        if let Some(observations) = &observations {
            observations.lock().push(observation);
        }
    };
    let target = call.models.first().ok_or(LibsyError::NoTargets)?;
    let request = clients.prepare_routing_request(call.request.clone(), target);
    match call_one(
        &clients,
        target,
        request,
        &call.algorithm,
        &observe,
        0,
        call.models.len(),
        true,
    )
    .await
    {
        Ok(response) => call.respond(Ok(response)),
        Err(error) => call.fail(error),
    }
}

/// Try candidates in order until one succeeds or a failure stops fallback.
async fn call_first_available(
    clients: &ClientRouter,
    algorithm: &str,
    request: &Request,
    models: &[ModelId],
    observe: &(dyn Fn(LlmCallObservation) + Send + Sync),
) -> Result<Response> {
    for (index, target) in models.iter().enumerate() {
        let request = clients.prepare_completion_request(request.clone(), target);
        match call_one(
            clients,
            target,
            request,
            algorithm,
            observe,
            index,
            models.len(),
            false,
        )
        .await
        {
            Ok(response) => return Ok(response),
            Err(error) if index + 1 == models.len() => return Err(error),
            Err(error) => match fallback_reason(&error) {
                Some(reason) => tracing::info!(
                    from = %target,
                    to = %models[index + 1],
                    reason = reason.as_str(),
                    "model call failed; trying next candidate"
                ),
                None => return Err(error),
            },
        }
    }
    Err(LibsyError::NoTargets)
}

/// Call one candidate model and record its observation and span.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    target = "libsy",
    name = "libsy.client_call",
    skip_all,
    fields(
        algorithm = algorithm,
        switchyard.algorithm = algorithm,
        switchyard.candidate = index + 1,
        switchyard.candidate_count = count,
        selected_model = %model_id,
        otel.kind = "client",
        otel.name = %format_args!("chat {model_id}"),
        openinference.span.kind = "LLM",
        gen_ai.operation.name = "chat",
        gen_ai.request.model = %model_id,
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
async fn call_one(
    clients: &ClientRouter,
    model_id: &ModelId,
    request: Request,
    algorithm: &str,
    observe: &(dyn Fn(LlmCallObservation) + Send + Sync),
    // index is for span log
    index: usize,
    // count is for span log
    count: usize,
    buffer: bool,
) -> Result<Response> {
    let span = tracing::Span::current();
    observability::record_gen_ai_request(&span, &request.llm_request);
    if let Some(session_id) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.session_id.as_deref())
    {
        span.record("gen_ai.conversation.id", session_id);
    }
    // Resolved before the clock starts: picking the client is Switchyard's work, not
    // the provider's, so it belongs in the routing overhead.
    let client = clients.route(model_id);
    let started = Instant::now();
    let result = async {
        let mut response = client?.call(request).await?;
        if buffer && let LlmResponse::Stream(chunks) = response.llm_response {
            response.llm_response = LlmResponse::Stream(buffer_routing_stream(chunks).await?);
        }
        Ok(response)
    }
    .await
    .map_err(|source| LibsyError::client_call(model_id.clone(), source));
    let duration = started.elapsed();

    let result = result.map(|mut response| {
        response.set_served_model(model_id);
        response
    });
    let result = observability::observe_client_call(result);
    let result = if !buffer {
        metrics::observe_routed_request(algorithm, model_id, Some(duration), result)
    } else {
        result
    };
    observe(LlmCallObservation {
        selected_model: model_id.clone(),
        is_success: result.is_ok(),
        duration,
        usage: result
            .as_ref()
            .ok()
            .and_then(|response| response.llm_response.as_agg())
            .map(|response| response.usage.clone()),
    });
    result
}

// Preserve signed provider events while checking the complete routing response.
async fn buffer_routing_stream(
    mut chunks: LlmResponseStream,
) -> std::result::Result<LlmResponseStream, LlmClientError> {
    let mut events = Vec::new();
    while let Some(event) = chunks.try_next().await? {
        for chunk in event.normalized() {
            match chunk {
                LlmResponseChunk::DecodeError { message } => {
                    return Err(LlmClientError::ResponseTranslation(message.clone()));
                }
                LlmResponseChunk::StreamError { message } => {
                    return Err(LlmClientError::UpstreamHttp {
                        status: StatusCode::BAD_GATEWAY,
                        body: message.clone(),
                    });
                }
                _ => {}
            }
        }
        events.push(event);
    }
    Ok(stream::iter(events.into_iter().map(Ok)).boxed())
}

/// Whether a failed candidate is worth routing around.
fn fallback_reason(error: &LibsyError) -> Option<RoutingFallbackReason> {
    let LibsyError::ClientCall { source, .. } = error else {
        return None;
    };
    match source {
        LlmClientError::ContextWindowExceeded { .. } => Some(RoutingFallbackReason::ContextWindow),
        LlmClientError::Transport { .. } => Some(RoutingFallbackReason::Unavailable),
        // A policy denial can be specific to one provider. Preserve its HTTP
        // error, but allow another candidate to serve the request.
        LlmClientError::UpstreamHttp { status, body }
            if *status == StatusCode::BAD_REQUEST
                && serde_json::from_str::<serde_json::Value>(body).is_ok_and(|value| {
                    value["error"]["code"].as_str() == Some("content_policy_violation")
                }) =>
        {
            Some(RoutingFallbackReason::Unavailable)
        }
        LlmClientError::UpstreamHttp { status, .. }
            if matches!(
                *status,
                StatusCode::FORBIDDEN | StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS
            ) || status.is_server_error() =>
        {
            Some(RoutingFallbackReason::Unavailable)
        }
        _ => None,
    }
}

/// Route to `owner` without fallbacks and record why the algorithm did not run.
fn continue_on(owner: StateOwner, algorithm: &str, mut request: Request) -> RoutingOutcome {
    tracing::debug!(
        target = %owner.model,
        trigger = owner.trigger,
        "routing Responses continuation to the model that holds its state"
    );
    if let Some(history) = owner.history {
        let mut messages = Vec::with_capacity(history.len + request.llm_request.messages.len());
        history.extend(&mut messages);
        messages.append(&mut request.llm_request.messages);
        request.llm_request.messages = messages;
    }
    let mut outcome = RoutingOutcome::route_to(owner.model, Vec::new(), request);
    outcome.metadata = Some(OutcomeMetadata::new(
        algorithm.to_string(),
        Some(json!({"source": "stored_state", "trigger": owner.trigger})),
    ));
    outcome
}

/// The model that served the stored state and the request field that refers to it.
struct StateOwner {
    model: ModelId,
    trigger: &'static str,
    history: Option<Arc<MessageHistory>>,
}

struct MessageHistory {
    parent: Option<Arc<Self>>,
    segment: Arc<[Message]>,
    len: usize,
}

impl MessageHistory {
    fn new(parent: Option<Arc<Self>>, segment: Arc<[Message]>) -> Self {
        let len = parent.as_ref().map_or(0, |history| history.len) + segment.len();
        Self {
            parent,
            segment,
            len,
        }
    }

    fn extend(&self, messages: &mut Vec<Message>) {
        let mut segments = Vec::new();
        let mut history = Some(self);
        while let Some(node) = history {
            segments.push(node.segment.as_ref());
            history = node.parent.as_deref();
        }
        for segment in segments.into_iter().rev() {
            messages.extend_from_slice(segment);
        }
    }
}

#[derive(Clone)]
struct StoredState {
    model: ModelId,
    /// `None` means the provider owns the state and only routing must be pinned.
    history: Option<Arc<MessageHistory>>,
}

#[derive(Clone)]
struct CanonicalInput {
    parent: Option<Arc<MessageHistory>>,
    messages: Arc<[Message]>,
}

/// Stored Responses IDs and conversation IDs, with their provider or canonical state.
///
/// Keep earlier response IDs because a client can continue from any stored response.
/// Reject new IDs at capacity rather than forgetting where an earlier response is stored.
#[derive(Default)]
struct StateOwners {
    by_id: HashMap<String, StoredState>,
}

const MAX_STATE_OWNERS: usize = 65_536;

impl StateOwners {
    fn owner(&self, id: &str) -> Option<&StoredState> {
        self.by_id.get(id)
    }

    fn remember(
        &mut self,
        response_id: Option<&str>,
        conversation_id: Option<&str>,
        model: &ModelId,
        history: Option<Arc<MessageHistory>>,
    ) -> std::result::Result<(), LlmClientError> {
        let materialized = history.is_some();
        let ids = [
            response_id,
            conversation_id.filter(|id| Some(*id) != response_id),
        ];
        let mut new_ids = 0;
        for id in ids.into_iter().flatten() {
            match self.by_id.get(id) {
                Some(owner) if owner.model != *model && !materialized => {
                    return Err(LlmClientError::ResponseStateConflict);
                }
                None => new_ids += 1,
                _ => {}
            }
        }
        if self.by_id.len() + new_ids > MAX_STATE_OWNERS {
            return Err(LlmClientError::ResponseStateLimitExceeded {
                limit: MAX_STATE_OWNERS,
            });
        }
        let state = StoredState {
            model: model.clone(),
            history,
        };
        if let Some(id) = response_id {
            if materialized {
                self.by_id.insert(id.to_owned(), state.clone());
            } else {
                self.by_id
                    .entry(id.to_owned())
                    .or_insert_with(|| state.clone());
            }
        }
        if let Some(id) = conversation_id {
            self.by_id.entry(id.to_owned()).or_insert(state);
        }
        Ok(())
    }
}

/// Read the Responses `conversation` ID from a string or an object.
fn conversation_id(fields: &serde_json::Map<String, Value>) -> Option<&str> {
    let conversation = fields.get("conversation")?;
    conversation
        .as_str()
        .or_else(|| conversation.get("id").and_then(Value::as_str))
}

/// Resolves a routed call's selected model to the client that serves it.
///
/// An algorithm routes among named targets; which provider each target lives on is the
/// host's concern, and two targets in one run may sit on different providers. A router owns
/// that mapping. It is *not* itself a client: it hands back a [`RoutedLlmClient`] and the
/// caller makes the call.
///
/// Cloning is cheap — the mapping is shared, so one router can serve every request.
#[derive(Clone)]
pub struct ClientRouter {
    inner: Arc<ClientRouting>,
}

struct ClientRouting {
    routing: Routing,
    target_prompts: HashMap<ModelId, String>,
    routing_answer_target: Option<ModelId>,
    /// Provider-owned routing pins and materialized cross-format history.
    state_owners: Mutex<StateOwners>,
    /// Native Responses state only needs a routing pin when answer targets span clients.
    track_provider_state: bool,
}

enum Routing {
    /// One client serves every model.
    Single(Arc<dyn RoutedLlmClient>),
    /// Each model is served by the client configured for it.
    ByModel(HashMap<ModelId, Arc<dyn RoutedLlmClient>>),
}

impl ClientRouter {
    /// Build a router over `model name -> client`, for targets spread across providers.
    pub fn new(by_model: HashMap<ModelId, Arc<dyn RoutedLlmClient>>) -> Self {
        Self::new_with_target_prompts(by_model, HashMap::new(), None)
    }

    /// Build a router with target prompts used by [`run`] and [`decide`].
    ///
    /// `routing_answer_target` identifies a routing-time call whose response may become the
    /// answer. Pass `None` when routing only selects a later completion target. Prompt keys and
    /// `routing_answer_target` use the resolved model IDs stored in `by_model`.
    /// This constructor treats every model in `by_model` as a possible completion target.
    /// Use [`Self::new_with_completion_targets`] to exclude classifier-only models from the
    /// comparison of completion clients.
    pub fn new_with_target_prompts(
        by_model: HashMap<ModelId, Arc<dyn RoutedLlmClient>>,
        target_prompts: HashMap<ModelId, String>,
        routing_answer_target: Option<ModelId>,
    ) -> Self {
        let completion_targets = by_model.keys().cloned().collect::<Vec<_>>();
        Self::new_with_completion_targets(
            by_model,
            target_prompts,
            routing_answer_target,
            &completion_targets,
        )
    }

    /// Build a router with an explicit list of models that can answer requests.
    ///
    /// `by_model` contains every callable model, including classifiers. Only the model IDs in
    /// `completion_targets` determine whether native Responses state needs a routing pin.
    /// Cross-format state is recorded lazily when a Responses request uses a Chat or Anthropic
    /// target. `target_prompts` and `routing_answer_target` have the same meaning as in
    /// [`Self::new_with_target_prompts`].
    pub fn new_with_completion_targets(
        by_model: HashMap<ModelId, Arc<dyn RoutedLlmClient>>,
        target_prompts: HashMap<ModelId, String>,
        routing_answer_target: Option<ModelId>,
        completion_targets: &[ModelId],
    ) -> Self {
        let mut clients = completion_targets
            .iter()
            .filter_map(|model| by_model.get(model));
        let spans_providers = clients
            .next()
            .is_some_and(|first| clients.any(|client| !Arc::ptr_eq(first, client)));
        Self {
            inner: Arc::new(ClientRouting {
                routing: Routing::ByModel(by_model),
                target_prompts,
                routing_answer_target,
                state_owners: Mutex::default(),
                track_provider_state: spans_providers,
            }),
        }
    }

    /// A router that serves every model with one client — the single-provider case.
    ///
    /// [`TranslatingLlmClient`](crate::TranslatingLlmClient) already maps model names to
    /// backends internally and rejects ones it does not know, so enumerating them here would
    /// only duplicate that.
    pub fn single(client: Arc<dyn RoutedLlmClient>) -> Self {
        Self {
            inner: Arc::new(ClientRouting {
                routing: Routing::Single(client),
                target_prompts: HashMap::new(),
                routing_answer_target: None,
                state_owners: Mutex::default(),
                track_provider_state: false,
            }),
        }
    }

    /// The client that serves `model`.
    ///
    /// Errors with [`LlmClientError::Configuration`] when the router maps models and has no
    /// entry for this one, rather than silently sending the call to another provider.
    pub fn route(
        &self,
        model: &ModelId,
    ) -> std::result::Result<&Arc<dyn RoutedLlmClient>, LlmClientError> {
        match &self.inner.routing {
            Routing::Single(client) => Ok(client),
            Routing::ByModel(by_model) => {
                by_model
                    .get(model)
                    .ok_or_else(|| LlmClientError::Configuration {
                        message: format!("no llm client is configured for model {model:?}"),
                    })
            }
        }
    }

    /// Return the recorded model for the requested response or conversation ID.
    fn stored_state_owner(&self, request: &Request) -> Option<StateOwner> {
        let fields = &request.llm_request.extensions.fields;
        let (trigger, id) = match fields.get("previous_response_id").and_then(Value::as_str) {
            Some(id) => ("previous_response_id", id),
            None => ("conversation", conversation_id(fields)?),
        };
        let state = self.inner.state_owners.lock().owner(id)?.clone();
        Some(StateOwner {
            model: state.model,
            trigger,
            history: state.history,
        })
    }

    fn canonical_input(&self, request: &Request) -> CanonicalInput {
        let parent = request
            .llm_request
            .extensions
            .fields
            .get("previous_response_id")
            .and_then(Value::as_str)
            .and_then(|id| self.inner.state_owners.lock().owner(id)?.history.clone());
        let parent_len = parent.as_ref().map_or(0, |history| history.len);
        let (parent, messages) = match request.llm_request.messages.get(parent_len..) {
            Some(messages) => (parent, messages.to_vec()),
            None => (None, request.llm_request.messages.clone()),
        };
        CanonicalInput {
            parent,
            messages: Arc::from(messages),
        }
    }

    /// Record provider-owned Responses state or canonical history for a cross-format response.
    fn remember_state_owner(&self, request: &Request, mut response: Response) -> Result<Response> {
        let Some(model) = response.served_model().cloned() else {
            return Ok(response);
        };
        let fields = &request.llm_request.extensions.fields;
        let store = fields.get("store").and_then(Value::as_bool) != Some(false);
        let conversation = conversation_id(fields).map(str::to_owned);
        let responses_format = WireFormat::OpenAiResponses.into();
        let responses_request = request
            .llm_request
            .preservation
            .requests
            .contains_key(&responses_format)
            || fields.contains_key("previous_response_id")
            || fields.contains_key("conversation");
        if !responses_request && !self.inner.track_provider_state {
            return Ok(response);
        }
        let canonical_input = responses_request.then(|| self.canonical_input(request));
        response.llm_response = match response.llm_response {
            LlmResponse::Agg(agg) => {
                if let Some(body) = agg.preservation.responses.get(&responses_format) {
                    if self.inner.track_provider_state {
                        self.remember_response(body, &model, store, conversation.as_deref())
                            .map_err(|error| LibsyError::client_call(model.clone(), error))?;
                    }
                } else if let Some(input) = &canonical_input {
                    self.remember_canonical_response(&agg, &model, store, input)
                        .map_err(|error| LibsyError::client_call(model.clone(), error))?;
                }
                LlmResponse::Agg(agg)
            }
            LlmResponse::Stream(stream) => {
                let router = self.clone();
                LlmResponse::Stream(Box::pin(futures_util::stream::try_unfold(
                    (
                        stream,
                        router,
                        model,
                        conversation,
                        canonical_input,
                        ResponseAccumulator::new(),
                        false,
                        true,
                    ),
                    move |(
                        mut stream,
                        router,
                        model,
                        conversation,
                        canonical_input,
                        mut accumulator,
                        mut native_responses,
                        mut valid,
                    )| async move {
                        let Some(event) = stream.next().await else {
                            if let Some(input) = &canonical_input
                                && !native_responses
                                && valid
                            {
                                router.remember_canonical_response(
                                    &accumulator.finish(),
                                    &model,
                                    store,
                                    input,
                                )?;
                            }
                            return Ok(None);
                        };
                        let event = event?;
                        if let Some(preserved) = event.preservation()
                            && preserved.source().as_str() == WireFormat::OpenAiResponses.as_str()
                        {
                            native_responses = true;
                            if router.inner.track_provider_state
                                && let Some(body) = preserved.raw().get("response")
                            {
                                router.remember_response(
                                    body,
                                    &model,
                                    store,
                                    conversation.as_deref(),
                                )?;
                            }
                        }
                        if responses_request {
                            for chunk in event.normalized() {
                                if matches!(
                                    chunk,
                                    LlmResponseChunk::DecodeError { .. }
                                        | LlmResponseChunk::StreamError { .. }
                                ) {
                                    valid = false;
                                } else {
                                    accumulator.push(chunk.clone());
                                }
                            }
                        }
                        Ok(Some((
                            event,
                            (
                                stream,
                                router,
                                model,
                                conversation,
                                canonical_input,
                                accumulator,
                                native_responses,
                                valid,
                            ),
                        )))
                    },
                )))
            }
        };
        Ok(response)
    }

    /// Check both IDs from a response or stream event before recording either. The response's
    /// `store` value overrides the request value; `false` skips only the response ID.
    fn remember_response(
        &self,
        body: &Value,
        model: &ModelId,
        store: bool,
        conversation: Option<&str>,
    ) -> std::result::Result<(), LlmClientError> {
        let mut owners = self.inner.state_owners.lock();
        let response_id = body
            .get("id")
            .and_then(Value::as_str)
            .filter(|_| body.get("store").and_then(Value::as_bool).unwrap_or(store));
        let conversation = body.as_object().and_then(conversation_id).or(conversation);
        if let Err(error) = owners.remember(response_id, conversation, model, None) {
            if matches!(error, LlmClientError::ResponseStateLimitExceeded { .. }) {
                tracing::warn!(%error, "Responses state owner capacity reached; state was not retained");
            } else {
                tracing::error!(%error, "could not record Responses state owner");
                return Err(error);
            }
        }
        Ok(())
    }

    fn remember_canonical_response(
        &self,
        response: &AggLlmResponse,
        model: &ModelId,
        store: bool,
        input: &CanonicalInput,
    ) -> std::result::Result<(), LlmClientError> {
        let response_id = response.id.as_deref().filter(|_| store);
        if response_id.is_none() {
            return Ok(());
        }
        let mut segment = input.messages.to_vec();
        segment.extend(response.outputs.iter().map(|output| Message {
            role: output.role,
            content: output.content.clone(),
        }));
        let history = Arc::new(MessageHistory::new(
            input.parent.clone(),
            Arc::from(segment),
        ));
        let result =
            self.inner
                .state_owners
                .lock()
                .remember(response_id, None, model, Some(history));
        if let Err(error) = result {
            if matches!(error, LlmClientError::ResponseStateLimitExceeded { .. }) {
                tracing::warn!(%error, "cross-format Responses state capacity reached; history was not retained");
                return Ok(());
            }
            tracing::error!(%error, "could not record cross-format Responses state");
            return Err(error);
        }
        Ok(())
    }

    /// Prepare a completion candidate with its configured target prompt.
    fn prepare_completion_request(&self, mut request: Request, target: &ModelId) -> Request {
        let prompt = self.inner.target_prompts.get(target).map(String::as_str);
        prepare_request_for_target(&mut request.llm_request, target, prompt);
        request
    }

    /// Prepare a routing call, adding a target prompt only when it generates a candidate answer.
    fn prepare_routing_request(&self, mut request: Request, target: &ModelId) -> Request {
        let prompt = if self.inner.routing_answer_target.as_ref() == Some(target) {
            self.inner.target_prompts.get(target).map(String::as_str)
        } else {
            None
        };
        prepare_request_for_target(&mut request.llm_request, target, prompt);
        request
    }
}

impl FromIterator<(ModelId, Arc<dyn RoutedLlmClient>)> for ClientRouter {
    fn from_iter<I: IntoIterator<Item = (ModelId, Arc<dyn RoutedLlmClient>)>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use async_trait::async_trait;
    use futures::StreamExt;
    use http::StatusCode;
    use switchyard_libsy::{Driver, RoutingOutcome};
    use switchyard_protocol::{
        ContentBlock, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Message, Role,
        ToolResult, completion_text, text_request, text_response,
    };
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
    use switchyard_protocol::Category;

    struct CandidateAlgorithm {}

    struct AnsweredAlgorithm {
        model: ModelId,
    }

    #[async_trait]
    impl Algorithm for CandidateAlgorithm {
        fn name(&self) -> &str {
            "candidate_test"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            let models = driver.models_for(&Category::Any);
            let selected_model = models.first().cloned().ok_or(LibsyError::NoTargets)?;
            Ok(RoutingOutcome::route_to(
                selected_model,
                models.iter().skip(1).cloned().collect(),
                request,
            ))
        }
    }

    #[async_trait]
    impl Algorithm for AnsweredAlgorithm {
        fn name(&self) -> &str {
            "answered_test"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            let response = driver
                .call_model(request.clone(), vec![self.model.clone()])
                .await?;
            Ok(RoutingOutcome::answered(
                self.model.clone(),
                request,
                response,
            ))
        }
    }

    #[derive(Clone, Copy)]
    enum FirstOutcome {
        ContextWindow,
        ContentPolicy,
        Unauthorized,
        StreamSuccess,
        MidStreamError,
    }

    struct CandidateClient {
        calls: Mutex<Vec<ModelId>>,
        requests: Mutex<Vec<Request>>,
        first: FirstOutcome,
    }

    #[async_trait]
    impl RoutedLlmClient for CandidateClient {
        async fn call(&self, request: Request) -> std::result::Result<Response, LlmClientError> {
            let model = request.model_id().unwrap_or_default();
            self.calls.lock().push(model.clone());
            self.requests.lock().push(request);
            if model == "weak" {
                return match self.first {
                    FirstOutcome::ContextWindow => Err(LlmClientError::ContextWindowExceeded {
                        model,
                        message: "too long".to_string(),
                    }),
                    FirstOutcome::ContentPolicy => Err(LlmClientError::UpstreamHttp {
                        status: StatusCode::BAD_REQUEST,
                        body: r#"{"error":{"code":"content_policy_violation","message":"request blocked by content policy","type":"invalid_request_error"},"metadata":{"documentation_section":"context window"}}"#.to_string(),
                    }),
                    FirstOutcome::Unauthorized => Err(LlmClientError::UpstreamHttp {
                        status: StatusCode::UNAUTHORIZED,
                        body: "unauthorized".to_string(),
                    }),
                    FirstOutcome::StreamSuccess => Ok(stream_response(vec![
                        LlmResponseChunk::TextDelta {
                            index: 0,
                            text: "streamed".to_string(),
                        },
                        LlmResponseChunk::MessageStop {
                            reason: Some("stop".to_string()),
                        },
                    ])),
                    FirstOutcome::MidStreamError => Ok(stream_response(vec![
                        LlmResponseChunk::TextDelta {
                            index: 0,
                            text: "partial".to_string(),
                        },
                        LlmResponseChunk::StreamError {
                            message: "stream failed".to_string(),
                        },
                    ])),
                };
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(Some(model.to_string()), model)),
                metadata: None,
                upstream_headers: http::HeaderMap::new(),
            })
        }
    }

    fn stream_response(chunks: Vec<LlmResponseChunk>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(
                futures::stream::iter(
                    chunks
                        .into_iter()
                        .map(|chunk| Ok(LlmResponseStreamEvent::from(chunk))),
                )
                .boxed(),
            ),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hello".to_string()),
            raw_request: None,
            metadata: None,
        }
    }

    fn instruction_text(request: &Request) -> Vec<&str> {
        request
            .llm_request
            .instructions
            .iter()
            .flat_map(|instruction| &instruction.content)
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    async fn run_candidates(
        first: FirstOutcome,
    ) -> (Arc<CandidateClient>, Result<(ModelId, Response)>) {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first,
        });
        let algorithm = Arc::new(CandidateAlgorithm {});
        let models = to_category_map(&["weak", "strong"]);
        let result = run(
            algorithm,
            ClientRouter::single(client.clone()),
            request(),
            models,
            None,
        )
        .await;
        (client, result)
    }

    fn to_category_map(names: &[&str]) -> Arc<RuntimeModels> {
        Arc::new(RuntimeModels::new(
            [(
                Category::Any,
                names.iter().map(|name| ModelId::from(*name)).collect(),
            )]
            .into(),
        ))
    }

    #[tokio::test]
    async fn answered_outcome_does_not_make_a_second_model_call() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&observations);
        let observer: RunObserver = Arc::new(move |event| observed.lock().push(event));

        let (selected, response) = run(
            Arc::new(AnsweredAlgorithm {
                model: "weak".into(),
            }),
            ClientRouter::single(client.clone()),
            request(),
            Arc::new(RuntimeModels::default()),
            Some(observer),
        )
        .await?;

        assert_eq!(selected, "weak");
        assert_eq!(response.served_model().map(ModelId::as_str), Some("weak"));
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        let observations = observations.lock();
        assert!(matches!(observations[0], RunObservation::AnswerCall(_)));
        assert!(matches!(
            observations[1],
            RunObservation::Outcome(ref metadata) if metadata.algorithm == "answered_test"
        ));
        assert!(matches!(
            observations[2],
            RunObservation::RoutingOverhead(_)
        ));
        assert_eq!(observations.len(), 3);
        Ok(())
    }

    #[test]
    fn answer_observation_keeps_call_order() {
        let pending = Some(Arc::new(Mutex::new(
            ["answer", "judge"]
                .map(|model| LlmCallObservation {
                    selected_model: model.into(),
                    is_success: true,
                    duration: std::time::Duration::ZERO,
                    usage: None,
                })
                .into(),
        )));
        let emitted = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&emitted);
        let observer: RunObserver = Arc::new(move |event| captured.lock().push(event));
        let answer = ModelId::from("answer");

        emit_routing_observations(&Some(observer), &pending, Some(&answer));

        assert!(matches!(
            &emitted.lock()[..],
            [RunObservation::AnswerCall(answer), RunObservation::LlmCall(judge)]
                if answer.selected_model == "answer" && judge.selected_model == "judge"
        ));
    }

    #[tokio::test]
    async fn responses_state_tracking_uses_upstream_storage_and_format() -> Result<()> {
        for (responses, store, conversation) in [
            (true, true, None),
            (true, false, None),
            (true, false, Some(json!("conv_plain"))),
            (true, false, Some(json!({"id": "conv_object"}))),
            (false, true, None),
        ] {
            for stream in [false, true] {
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .respond_with(move |request: &wiremock::Request| {
                        let body: Value = serde_json::from_slice(&request.body).expect("request JSON");
                        let response = if responses {
                            json!({
                                "id": "resp_seed", "object": "response", "model": "weak",
                                "status": "completed", "output": [], "store": body["store"],
                                "conversation": body["conversation"],
                            })
                        } else if stream {
                            json!({"id": "resp_seed", "model": "weak", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})
                        } else {
                            json!({"id": "resp_seed", "model": "weak", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
                        };
                        if stream {
                            let event = if responses { json!({"type": "response.completed", "response": response}) } else { response };
                            ResponseTemplate::new(200)
                                .insert_header("content-type", "text/event-stream")
                                .set_body_string(if responses {
                                    format!("event: response.completed\ndata: {event}\n\n")
                                } else {
                                    format!("data: {event}\n\ndata: [DONE]\n\n")
                                })
                        } else {
                            ResponseTemplate::new(200).set_body_json(response)
                        }
                    })
                    .mount(&server).await;
                let mut clients = HashMap::new();
                for model in ["weak", "strong"] {
                    let config = HttpBackendConfig {
                        base_url: server.uri(),
                        api_key: None,
                        forward_auth: false,
                        extra_headers: BTreeMap::new(),
                        extra_body: BTreeMap::from([("store".to_string(), json!(store))]),
                        reasoning_effort: None,
                        max_retries: 0,
                        timeout: None,
                    };
                    let backend = if responses {
                        Backend::OpenAiResponses(config)
                    } else {
                        Backend::OpenAiChat(config)
                    };
                    let client: Arc<dyn RoutedLlmClient> = Arc::new(
                        TranslatingLlmClient::new(&[ModelConfig::new(model, backend, None)])
                            .map_err(|error| LibsyError::external("building test client", error))?,
                    );
                    clients.insert(ModelId::from(model), client);
                }
                let clients = ClientRouter::new(clients);
                let mut seed = request();
                seed.llm_request.stream = stream;
                if let Some(conversation) = &conversation {
                    seed.llm_request
                        .extensions
                        .fields
                        .insert("conversation".to_string(), conversation.clone());
                }
                let (_, response) = run(
                    Arc::new(switchyard_libsy::Passthrough),
                    clients.clone(),
                    seed,
                    to_category_map(&["weak"]),
                    None,
                )
                .await?;
                response
                    .llm_response
                    .into_agg()
                    .await
                    .map_err(|error| LibsyError::client_call("weak", error))?;
                let mut continuations = vec![(
                    "previous_response_id",
                    json!("resp_seed"),
                    responses && store,
                )];
                if let Some(conversation) = conversation.clone() {
                    continuations.push(("conversation", conversation, true));
                }
                for (field, id, pinned) in continuations {
                    let mut follow = request();
                    follow
                        .llm_request
                        .extensions
                        .fields
                        .insert(field.to_string(), id);
                    let outcome = decide(
                        Arc::new(switchyard_libsy::Passthrough),
                        clients.clone(),
                        follow,
                        to_category_map(&["strong"]),
                    )
                    .await?;
                    assert_eq!(
                        outcome.selected_model_id()?.as_str(),
                        if pinned { "weak" } else { "strong" },
                        "responses={responses}, store={store}, stream={stream}, field={field}"
                    );
                }
                let requests = server.received_requests().await.expect("request recording");
                assert_eq!(requests.len(), 1);
                assert_eq!(
                    serde_json::from_slice::<Value>(&requests[0].body).expect("request JSON")["store"],
                    store
                );
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn responses_stored_tool_continuation_materializes_for_anthropic() -> Result<()> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(|request: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&request.body).expect("request JSON");
                let blocks = body["messages"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|message| message["content"].as_array())
                    .flatten()
                    .collect::<Vec<_>>();
                let has_result = blocks.iter().any(|block| block["type"] == "tool_result");
                let has_call = blocks.iter().any(|block| block["type"] == "tool_use");
                let has_thanks = blocks
                    .iter()
                    .any(|block| block["type"] == "text" && block["text"] == "thanks")
                    || body["messages"].as_array().is_some_and(|messages| {
                        messages
                            .iter()
                            .any(|message| message["content"] == "thanks")
                    });
                if has_result && !has_call {
                    return ResponseTemplate::new(400).set_body_json(json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "tool_result has no corresponding tool_use"
                        }
                    }));
                }
                let response = if has_thanks {
                    json!({
                        "id": "msg_third", "type": "message", "role": "assistant",
                        "model": "weak", "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn", "usage": {"input_tokens": 6, "output_tokens": 1}
                    })
                } else if has_result {
                    json!({
                        "id": "msg_follow", "type": "message", "role": "assistant",
                        "model": "weak", "content": [{"type": "text", "text": "sunny"}],
                        "stop_reason": "end_turn", "usage": {"input_tokens": 4, "output_tokens": 1}
                    })
                } else {
                    json!({
                        "id": "msg_seed", "type": "message", "role": "assistant",
                        "model": "weak",
                        "content": [{
                            "type": "tool_use", "id": "toolu_weather", "name": "get_weather",
                            "input": {"city": "Paris"}
                        }],
                        "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 1}
                    })
                };
                ResponseTemplate::new(200).set_body_json(response)
            })
            .mount(&server)
            .await;

        let client: Arc<dyn RoutedLlmClient> = Arc::new(
            TranslatingLlmClient::new(&[ModelConfig::new(
                "weak",
                Backend::Anthropic(HttpBackendConfig {
                    base_url: server.uri(),
                    api_key: None,
                    forward_auth: false,
                    extra_headers: BTreeMap::new(),
                    extra_body: BTreeMap::new(),
                    reasoning_effort: None,
                    max_retries: 0,
                    timeout: None,
                }),
                None,
            )])
            .map_err(|error| LibsyError::external("building test client", error))?,
        );
        let clients = ClientRouter::new(HashMap::from([(ModelId::from("weak"), client)]));
        let models = to_category_map(&["weak"]);

        let seed = Request {
            llm_request: switchyard_translation::decode_request(
                WireFormat::OpenAiResponses,
                &json!({
                    "model": "route",
                    "input": "Call get_weather for Paris",
                    "tools": [{
                        "type": "function", "name": "get_weather",
                        "parameters": {"type": "object"}
                    }],
                    "store": true
                }),
            )
            .map_err(|error| LibsyError::external("decoding seed request", error))?,
            raw_request: None,
            metadata: None,
        };
        let (_, response) = run(
            Arc::new(switchyard_libsy::Passthrough),
            clients.clone(),
            seed,
            models.clone(),
            None,
        )
        .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .and_then(|agg| agg.id.as_deref()),
            Some("msg_seed")
        );

        let follow = Request {
            llm_request: switchyard_translation::decode_request(
                WireFormat::OpenAiResponses,
                &json!({
                    "model": "route",
                    "previous_response_id": "msg_seed",
                    "input": [{
                        "type": "function_call_output",
                        "call_id": "toolu_weather",
                        "output": "Paris is sunny"
                    }],
                    "tools": [{
                        "type": "function", "name": "get_weather",
                        "parameters": {"type": "object"}
                    }],
                    "store": true
                }),
            )
            .map_err(|error| LibsyError::external("decoding continuation request", error))?,
            raw_request: None,
            metadata: None,
        };
        let (_, response) = run(
            Arc::new(switchyard_libsy::Passthrough),
            clients.clone(),
            follow,
            models.clone(),
            None,
        )
        .await?;
        assert_eq!(
            completion_text(
                response
                    .llm_response
                    .as_agg()
                    .expect("buffered Anthropic response"),
            ),
            "sunny"
        );

        let third = Request {
            llm_request: switchyard_translation::decode_request(
                WireFormat::OpenAiResponses,
                &json!({
                    "model": "route",
                    "previous_response_id": "msg_follow",
                    "input": "thanks",
                    "store": true
                }),
            )
            .map_err(|error| LibsyError::external("decoding chained request", error))?,
            raw_request: None,
            metadata: None,
        };
        let history = clients
            .stored_state_owner(&third)
            .and_then(|owner| owner.history)
            .expect("chained canonical history");
        assert_eq!(history.len, 4);
        assert_eq!(history.segment.len(), 2);
        assert!(history.parent.is_some());
        let (_, response) = run(
            Arc::new(switchyard_libsy::Passthrough),
            clients,
            third,
            models,
            None,
        )
        .await?;
        assert_eq!(
            completion_text(
                response
                    .llm_response
                    .as_agg()
                    .expect("buffered Anthropic response"),
            ),
            "done"
        );

        let requests = server.received_requests().await.expect("request recording");
        assert_eq!(requests.len(), 3);
        let follow: Value = serde_json::from_slice(&requests[1].body)
            .map_err(|error| LibsyError::external("decoding captured request", error))?;
        assert_eq!(follow["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(follow["messages"][2]["content"][0]["type"], "tool_result");
        let third: Value = serde_json::from_slice(&requests[2].body)
            .map_err(|error| LibsyError::external("decoding captured request", error))?;
        assert_eq!(third["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(third["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(third["messages"][3]["content"], "sunny");
        assert_eq!(third["messages"][4]["content"][0]["text"], "thanks");
        Ok(())
    }

    #[tokio::test]
    async fn streamed_cross_format_state_is_recorded_only_after_completion() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let clients = ClientRouter::new(HashMap::from([(
            ModelId::from("weak"),
            client as Arc<dyn RoutedLlmClient>,
        )]));
        let mut seed = request();
        seed.llm_request.messages = vec![Message::text(Role::User, "inspect")];
        seed.llm_request
            .preservation
            .requests
            .insert(WireFormat::OpenAiResponses.into(), json!({}));
        let mut response = stream_response(vec![
            LlmResponseChunk::MessageStart {
                id: Some("msg_stream".to_string()),
                model: Some("weak".to_string()),
            },
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: Some("toolu_stream".to_string()),
                name: Some("lookup".to_string()),
                arguments_delta: Some("{}".to_string()),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("tool_use".to_string()),
            },
        ]);
        response.set_served_model(&ModelId::from("weak"));
        let response = clients.remember_state_owner(&seed, response)?;

        let mut follow = request();
        follow
            .llm_request
            .extensions
            .fields
            .insert("previous_response_id".to_string(), json!("msg_stream"));
        follow.llm_request.messages = vec![Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "toolu_stream".to_string(),
                content: vec![ContentBlock::Text {
                    text: "ready".to_string(),
                }],
                is_error: None,
            })],
        }];
        assert!(clients.stored_state_owner(&follow).is_none());
        response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::client_call("weak", error))?;

        let outcome = continue_on(
            clients
                .stored_state_owner(&follow)
                .expect("completed stream state"),
            "passthrough",
            follow,
        );
        assert_eq!(outcome.request.llm_request.messages.len(), 3);
        assert!(matches!(
            outcome.request.llm_request.messages[1].content.as_slice(),
            [ContentBlock::ToolCall(call)] if call.id == "toolu_stream"
        ));
        assert!(matches!(
            outcome.request.llm_request.messages[2].content.as_slice(),
            [ContentBlock::ToolResult(result)] if result.tool_call_id == "toolu_stream"
        ));
        Ok(())
    }

    #[test]
    fn cross_format_response_with_store_false_is_not_retained() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let clients = ClientRouter::new(HashMap::from([(
            ModelId::from("weak"),
            client as Arc<dyn RoutedLlmClient>,
        )]));
        let mut seed = request();
        seed.llm_request
            .extensions
            .fields
            .insert("store".to_string(), json!(false));
        seed.llm_request
            .preservation
            .requests
            .insert(WireFormat::OpenAiResponses.into(), json!({}));
        let mut agg = text_response(Some("weak".to_string()), "done".to_string());
        agg.id = Some("msg_not_stored".to_string());
        let mut response = Response {
            llm_response: LlmResponse::Agg(agg),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        response.set_served_model(&ModelId::from("weak"));
        clients.remember_state_owner(&seed, response)?;

        let mut follow = request();
        follow
            .llm_request
            .extensions
            .fields
            .insert("previous_response_id".to_string(), json!("msg_not_stored"));
        assert!(clients.stored_state_owner(&follow).is_none());
        Ok(())
    }

    #[tokio::test]
    async fn state_capacity_does_not_fail_completed_responses() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let clients = ClientRouter::new(HashMap::from([(
            ModelId::from("weak"),
            client as Arc<dyn RoutedLlmClient>,
        )]));
        let model = ModelId::from("weak");
        clients.inner.state_owners.lock().by_id = (0..MAX_STATE_OWNERS)
            .map(|id| {
                (
                    format!("existing_{id}"),
                    StoredState {
                        model: model.clone(),
                        history: None,
                    },
                )
            })
            .collect();

        let mut request = request();
        request
            .llm_request
            .preservation
            .requests
            .insert(WireFormat::OpenAiResponses.into(), json!({}));
        let mut agg = text_response(Some("weak".to_string()), "done".to_string());
        agg.id = Some("overflow_buffered".to_string());
        let mut buffered = Response {
            llm_response: LlmResponse::Agg(agg),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        buffered.set_served_model(&model);
        clients.remember_state_owner(&request, buffered)?;

        let mut streamed = stream_response(vec![
            LlmResponseChunk::MessageStart {
                id: Some("overflow_streamed".to_string()),
                model: Some("weak".to_string()),
            },
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "done".to_string(),
            },
            LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            },
        ]);
        streamed.set_served_model(&model);
        clients
            .remember_state_owner(&request, streamed)?
            .llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::client_call("weak", error))?;

        assert_eq!(
            clients.inner.state_owners.lock().by_id.len(),
            MAX_STATE_OWNERS
        );
        Ok(())
    }

    #[test]
    fn state_owners_preserve_existing_records_at_capacity()
    -> std::result::Result<(), LlmClientError> {
        let mut owners = StateOwners::default();
        let model = ModelId::from("model/a");
        for id in 0..MAX_STATE_OWNERS - 1 {
            owners.remember(Some(&format!("resp_{id}")), None, &model, None)?;
        }
        assert!(matches!(
            owners.remember(Some("new_response"), Some("new_conversation"), &model, None,),
            Err(LlmClientError::ResponseStateLimitExceeded {
                limit: MAX_STATE_OWNERS
            })
        ));
        assert!(owners.owner("new_response").is_none());
        assert!(owners.owner("new_conversation").is_none());
        owners.remember(Some("last_id"), Some("last_id"), &model, None)?;
        owners.remember(Some("last_id"), Some("resp_0"), &model, None)?;
        assert!(matches!(
            owners.remember(Some("over_limit"), None, &model, None),
            Err(LlmClientError::ResponseStateLimitExceeded { .. })
        ));
        assert!(matches!(
            owners.remember(Some("resp_0"), None, &ModelId::from("model/b"), None,),
            Err(LlmClientError::ResponseStateConflict)
        ));
        assert_eq!(
            owners.owner("resp_0").map(|state| &state.model),
            Some(&model)
        );
        assert_eq!(owners.by_id.len(), MAX_STATE_OWNERS);
        Ok(())
    }

    #[test]
    fn concurrent_state_owners_do_not_exceed_capacity() {
        let model = ModelId::from("model/a");
        let owners = Arc::new(Mutex::new(StateOwners {
            by_id: (0..MAX_STATE_OWNERS - 16)
                .map(|id| {
                    (
                        format!("existing_{id}"),
                        StoredState {
                            model: model.clone(),
                            history: None,
                        },
                    )
                })
                .collect(),
        }));
        let barrier = Arc::new(std::sync::Barrier::new(32));
        let accepted = std::thread::scope(|scope| {
            let handles = (0..32)
                .map(|id| {
                    let owners = owners.clone();
                    let barrier = barrier.clone();
                    let model = model.clone();
                    scope.spawn(move || {
                        barrier.wait();
                        let result = owners.lock().remember(
                            Some(&format!("resp_{id}")),
                            Some(&format!("conv_{id}")),
                            &model,
                            None,
                        );
                        match result {
                            Ok(()) => Some(id),
                            Err(LlmClientError::ResponseStateLimitExceeded { .. }) => None,
                            other => panic!("unexpected state tracking result: {other:?}"),
                        }
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().expect("request thread panicked"))
                .collect::<Vec<_>>()
        });
        let owners = owners.lock();
        assert_eq!(accepted.len(), 8);
        assert_eq!(owners.by_id.len(), MAX_STATE_OWNERS);
        assert_eq!(
            owners.owner("existing_0").map(|state| &state.model),
            Some(&model)
        );
        for id in accepted {
            assert_eq!(
                owners
                    .owner(&format!("resp_{id}"))
                    .map(|state| &state.model),
                Some(&model)
            );
            assert_eq!(
                owners
                    .owner(&format!("conv_{id}"))
                    .map(|state| &state.model),
                Some(&model)
            );
        }
    }

    #[tokio::test]
    async fn each_fallback_candidate_receives_only_its_own_prompt() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::ContextWindow,
        });
        let routed_client: Arc<dyn RoutedLlmClient> = client.clone();
        let clients = ClientRouter::new_with_target_prompts(
            HashMap::from([
                (ModelId::from("weak"), Arc::clone(&routed_client)),
                (ModelId::from("strong"), routed_client),
            ]),
            HashMap::from([
                ("weak".into(), "weak prompt".to_string()),
                ("strong".into(), "strong prompt".to_string()),
            ]),
            None,
        );

        run(
            Arc::new(CandidateAlgorithm {}),
            clients,
            request(),
            to_category_map(&["weak", "strong"]),
            None,
        )
        .await?;

        let calls = client.requests.lock();
        assert_eq!(calls.len(), 2);
        assert_eq!(instruction_text(&calls[0]), ["weak prompt"]);
        assert_eq!(instruction_text(&calls[1]), ["strong prompt"]);
        Ok(())
    }

    #[tokio::test]
    async fn decision_prompts_a_routing_response_target() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let routed_client: Arc<dyn RoutedLlmClient> = client.clone();
        let clients = ClientRouter::new_with_target_prompts(
            HashMap::from([(ModelId::from("answer"), routed_client)]),
            HashMap::from([("answer".into(), "answer prompt".to_string())]),
            Some("answer".into()),
        );

        decide(
            Arc::new(AnsweredAlgorithm {
                model: "answer".into(),
            }),
            clients,
            request(),
            Arc::new(RuntimeModels::default()),
        )
        .await?;

        let calls = client.requests.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(instruction_text(&calls[0]), ["answer prompt"]);
        Ok(())
    }

    #[tokio::test]
    async fn decision_prepares_only_the_selected_request() -> Result<()> {
        let client = Arc::new(CandidateClient {
            calls: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            first: FirstOutcome::StreamSuccess,
        });
        let routed_client: Arc<dyn RoutedLlmClient> = client;
        let clients = ClientRouter::new_with_target_prompts(
            HashMap::from([
                (ModelId::from("weak"), Arc::clone(&routed_client)),
                (ModelId::from("strong"), routed_client),
            ]),
            HashMap::from([
                (ModelId::from("weak"), "weak prompt".to_string()),
                (ModelId::from("strong"), "strong prompt".to_string()),
            ]),
            None,
        );

        let outcome = decide(
            Arc::new(CandidateAlgorithm {}),
            clients,
            request(),
            to_category_map(&["weak", "strong"]),
        )
        .await?;

        assert_eq!(
            outcome.selected_model_ids,
            [ModelId::from("weak"), ModelId::from("strong")]
        );
        assert_eq!(instruction_text(&outcome.request), ["weak prompt"]);
        Ok(())
    }

    #[test]
    fn fallback_only_accepts_context_policy_and_unavailable_failures() {
        let error = |source| LibsyError::client_call("target", source);
        assert_eq!(
            fallback_reason(&error(LlmClientError::ContextWindowExceeded {
                model: "target".into(),
                message: "too long".to_string(),
            })),
            Some(RoutingFallbackReason::ContextWindow)
        );
        for status in [
            StatusCode::FORBIDDEN,
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::from_u16(599).expect("599 is a valid status"),
        ] {
            assert_eq!(
                fallback_reason(&error(LlmClientError::UpstreamHttp {
                    status,
                    body: "failed".to_string(),
                })),
                Some(RoutingFallbackReason::Unavailable)
            );
        }
        for (body, expected) in [
            (
                r#"{"error":{"code":"content_policy_violation"}}"#,
                Some(RoutingFallbackReason::Unavailable),
            ),
            (
                r#"{"error":{"code":"invalid_request_error"},"metadata":"content_policy_violation"}"#,
                None,
            ),
            (r#"{"error":{"message":"content_policy_violation"}}"#, None),
            ("content_policy_violation", None),
        ] {
            assert_eq!(
                fallback_reason(&error(LlmClientError::UpstreamHttp {
                    status: StatusCode::BAD_REQUEST,
                    body: body.to_string(),
                })),
                expected,
                "{body}"
            );
        }
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::from_u16(499).expect("499 is a valid status"),
            StatusCode::from_u16(600).expect("600 is a valid status"),
        ] {
            assert_eq!(
                fallback_reason(&error(LlmClientError::UpstreamHttp {
                    status,
                    body: "failed".to_string(),
                })),
                None
            );
        }
    }

    #[tokio::test]
    async fn candidate_failures_follow_the_fallback_policy() -> Result<()> {
        // Context overflow is retryable across candidates.
        let (client, result) = run_candidates(FirstOutcome::ContextWindow).await;
        let (_, response) = result?;
        assert_eq!(
            &*client.calls.lock(),
            &[ModelId::from("weak"), "strong".into()]
        );
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(|response| response.model.as_deref()),
            Some(Some("strong"))
        );
        assert_eq!(response.served_model().map(ModelId::as_str), Some("strong"));

        // A provider-specific policy denial also permits another candidate.
        let (client, result) = run_candidates(FirstOutcome::ContentPolicy).await;
        let (_, response) = result?;
        assert_eq!(
            &*client.calls.lock(),
            &[ModelId::from("weak"), "strong".into()]
        );
        assert_eq!(response.served_model().map(ModelId::as_str), Some("strong"));

        // Authentication failure is not retryable, so the second candidate is untouched.
        let (client, result) = run_candidates(FirstOutcome::Unauthorized).await;
        assert!(matches!(
            result,
            Err(LibsyError::ClientCall {
                source: LlmClientError::UpstreamHttp {
                    status: StatusCode::UNAUTHORIZED,
                    ..
                },
                ..
            })
        ));
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        Ok(())
    }

    #[tokio::test]
    async fn retry_budget_is_exhausted_before_falling_through() -> Result<()> {
        let server = MockServer::start().await;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |request: &wiremock::Request| {
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).unwrap_or(serde_json::Value::Null);
                let model = body["model"].as_str().unwrap_or_default().to_string();
                observed_calls.lock().push(model.clone());
                if model == "weak" {
                    ResponseTemplate::new(503)
                        .insert_header("retry-after", "0")
                        .set_body_string("unavailable")
                } else {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "id": "answer",
                        "model": "strong",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"
                        }],
                        "usage": {}
                    }))
                }
            })
            .mount(&server)
            .await;

        let backend = || {
            Backend::OpenAiChat(HttpBackendConfig {
                base_url: format!("{}/v1", server.uri()),
                api_key: None,
                forward_auth: false,
                extra_headers: BTreeMap::new(),
                extra_body: BTreeMap::new(),
                reasoning_effort: None,
                max_retries: 2,
                timeout: None,
            })
        };
        let client = Arc::new(
            TranslatingLlmClient::new(&[
                ModelConfig::new("weak", backend(), None),
                ModelConfig::new("strong", backend(), None),
            ])
            .map_err(|error| LibsyError::external("building test client", error))?,
        );
        let algorithm = Arc::new(CandidateAlgorithm {});
        run(
            algorithm,
            ClientRouter::single(client),
            request(),
            to_category_map(&["weak", "strong"]),
            None,
        )
        .await?;

        assert_eq!(&*calls.lock(), &["weak", "weak", "weak", "strong"]);
        Ok(())
    }

    #[tokio::test]
    async fn streams_are_outside_the_candidate_fallback_boundary() -> Result<()> {
        // Receiving a stream handle is a successful call and ends candidate selection.
        let (client, result) = run_candidates(FirstOutcome::StreamSuccess).await;
        let (_, response) = result?;
        assert_eq!(response.served_model().map(ModelId::as_str), Some("weak"));
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|source| LibsyError::client_call("weak", source))?;
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        assert_eq!(completion_text(&aggregate), "streamed");

        // A later in-stream failure surfaces during aggregation without trying another model.
        let (client, result) = run_candidates(FirstOutcome::MidStreamError).await;
        let (_, response) = result?;
        let error = response
            .llm_response
            .into_agg()
            .await
            .expect_err("the stream should fail during aggregation");
        assert!(error.to_string().contains("stream failed"));
        assert_eq!(&*client.calls.lock(), &[ModelId::from("weak")]);
        Ok(())
    }

    // Verbatim shape of an SGLang admission rejection on a streaming session:
    // HTTP 200, and the overflow arrives as the first SSE event.
    const OVERFLOW_SSE: &str = "data: {\"error\":{\"code\":400,\"message\":\"Input length (600013 tokens) exceeds the maximum allowed length (536826 tokens). Use a shorter input or enable --allow-auto-truncate.\",\"object\":\"error\",\"param\":null,\"type\":\"BAD_REQUEST\"},\"model\":\"weak\"}\n\ndata: [DONE]\n\n";

    // A normal one-turn streamed answer, as the second candidate serves it.
    const STRONG_SSE: &str = "data: {\"id\":\"answer\",\"model\":\"strong\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"answer\",\"model\":\"strong\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

    /// Runs a streaming request through the weak→strong candidate loop, serving
    /// `weak_sse` from the first candidate and [`STRONG_SSE`] from the second.
    /// The server is returned so response streams stay readable after the call.
    async fn run_streaming_candidates(
        weak_sse: &'static str,
    ) -> (
        MockServer,
        Arc<Mutex<Vec<String>>>,
        Result<(ModelId, Response)>,
    ) {
        let server = MockServer::start().await;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed_calls = Arc::clone(&calls);
        Mock::given(method("POST"))
            .respond_with(move |request: &wiremock::Request| {
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).unwrap_or(serde_json::Value::Null);
                let model = body["model"].as_str().unwrap_or_default().to_string();
                observed_calls.lock().push(model.clone());
                let sse = if model == "weak" {
                    weak_sse
                } else {
                    STRONG_SSE
                };
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse)
            })
            .mount(&server)
            .await;

        let backend = || {
            Backend::OpenAiChat(HttpBackendConfig {
                base_url: format!("{}/v1", server.uri()),
                api_key: None,
                forward_auth: false,
                extra_headers: BTreeMap::new(),
                extra_body: BTreeMap::new(),
                reasoning_effort: None,
                max_retries: 0,
                timeout: None,
            })
        };
        let client = Arc::new(
            TranslatingLlmClient::new(&[
                ModelConfig::new("weak", backend(), None),
                ModelConfig::new("strong", backend(), None),
            ])
            .expect("building test client"),
        );
        let algorithm = Arc::new(CandidateAlgorithm {});
        let mut llm_request = text_request(Some("auto".to_string()), "hello".to_string());
        llm_request.stream = true;
        let request = Request {
            llm_request,
            raw_request: None,
            metadata: None,
        };
        let result = run(
            algorithm,
            ClientRouter::single(client),
            request,
            to_category_map(&["weak", "strong"]),
            None,
        )
        .await;
        (server, calls, result)
    }

    #[tokio::test]
    async fn first_stream_event_overflow_falls_back_to_the_next_candidate() -> Result<()> {
        // Nothing has reached the caller when the first event arrives, so an overflow
        // there falls through to the next candidate exactly as its buffered twin does.
        let (_server, calls, result) = run_streaming_candidates(OVERFLOW_SSE).await;
        let (_, response) = result?;
        assert_eq!(&*calls.lock(), &["weak", "strong"]);
        assert_eq!(response.served_model().map(ModelId::as_str), Some("strong"));
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|source| LibsyError::client_call("strong", source))?;
        assert_eq!(completion_text(&aggregate), "ok");

        // Only a classified overflow re-routes; any other in-band error event still
        // streams through to the consumer without trying another candidate.
        let (_server, calls, result) = run_streaming_candidates(
            "data: {\"error\":{\"code\":500,\"message\":\"engine exploded\",\"object\":\"error\"},\"model\":\"weak\"}\n\ndata: [DONE]\n\n",
        )
        .await;
        let (_, response) = result?;
        assert_eq!(&*calls.lock(), &["weak"]);
        let error = response
            .llm_response
            .into_agg()
            .await
            .expect_err("the in-band error should surface to the consumer");
        assert!(error.to_string().contains("engine exploded"));

        // Once content has streamed, a retry would repeat delivered output, so a later
        // overflow event keeps today's semantics and surfaces during aggregation.
        let (_server, calls, result) = run_streaming_candidates(
            "data: {\"id\":\"turn\",\"model\":\"weak\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\ndata: {\"error\":{\"code\":400,\"message\":\"Input length (600013 tokens) exceeds the maximum allowed length (536826 tokens).\",\"object\":\"error\",\"type\":\"BAD_REQUEST\"},\"model\":\"weak\"}\n\ndata: [DONE]\n\n",
        )
        .await;
        let (_, response) = result?;
        assert_eq!(&*calls.lock(), &["weak"]);
        let error = response
            .llm_response
            .into_agg()
            .await
            .expect_err("the mid-stream overflow should surface to the consumer");
        assert!(
            error
                .to_string()
                .contains("exceeds the maximum allowed length")
        );

        // A stream that ends before any event is not a classified failure; it still
        // aggregates to an empty response.
        let (_server, calls, result) = run_streaming_candidates("data: [DONE]\n\n").await;
        let (_, response) = result?;
        assert_eq!(&*calls.lock(), &["weak"]);
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|source| LibsyError::client_call("weak", source))?;
        assert_eq!(completion_text(&aggregate), "");
        Ok(())
    }
}

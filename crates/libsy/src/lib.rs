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
//!   like ordinary calls — publishes its [`Decision`](switchyard_protocol::Decision)s with [`Driver::info`], and
//!   returns the final [`Response`](switchyard_protocol::Response). The provided
//!   [`run_stream`](Algorithm::run_stream) drives that on its own task and hands
//!   back a stream of [`Step`]s; [`run`](Algorithm::run) runs
//!   it to completion with the targets' default clients.
//! - An [`LlmTarget`] names a routing target by its [`semantic_name`](LlmTarget::semantic_name).
//!   Every call is *offloaded* to the request's stream as a [`Step::CallLlm`]; the
//!   target's [`RoutedLlmClient`](switchyard_protocol::RoutedLlmClient), if any, rides along as
//!   [`RoutedRequest::default_client`] so the host can serve it by default or
//!   override it (see below).
//!
//! ## Running a request
//!
//! Hold the algorithm as `Arc<dyn Algorithm>` and call one of two provided methods:
//!
//! - [`run`](Algorithm::run) — run to completion, serving each
//!   offloaded call via its [`RoutedRequest::default_client`], and return the decision
//!   trace plus the final [`Response`](switchyard_protocol::Response). The simplest integration; use it when the
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
//! ## Observability
//!
//! The provided run methods instrument every algorithm from the outside — at the
//! [`Decision`](switchyard_protocol::Decision) hook and the offload boundary — so algorithms carry no telemetry
//! code. Each run gets a `libsy.run` tracing span (correlation ids from
//! [`Metadata`](switchyard_protocol::Metadata) attached) with a child `libsy.llm_call` span per model call
//! (fulfillment time as the algorithm observes it) plus a `libsy.client_call`
//! span around the actual API call when [`run`](Algorithm::run) serves it;
//! each [`Driver::info`] decision is logged with its reasoning; and OpenTelemetry
//! metrics record run/call counts, latency, and published decisions, keyed by
//! [`Algorithm::name`] plus `selected_model` and `outcome`.
//! Metrics use the global meter provider and spans/logs the `tracing` facade: a
//! host that installs an OTel SDK and a `tracing` subscriber (bridged with
//! `tracing-opentelemetry` for OTLP spans) gets the full signal set; with
//! neither installed, everything is a no-op.
//!
//! ## Algorithms
//!
//! Concrete algorithms live in [`algorithms`]:
//!
//! [`algorithms::Random`] provides uniform or weighted random routing.
//!
//! [`algorithms::LlmTaskClassifier`] uses one model to classify and route to its selected target.

mod core;
pub use core::algorithm::{
    Algorithm, CallLlmRequest, Driver, LlmCallObservation, LlmTarget, LlmTargetSet, RoutedRequest,
    RunObservation, RunObserver, Step, StepStream,
};
pub use core::classifier::{Classification, Classifier, Score};
pub use core::processor::{Event, Processor};
pub use core::state::{State, StateValue};

mod error;
pub use error::{DriverError, LibsyError, Result};

mod algorithms;
pub use algorithms::llm_class::{LlmTaskClassifier, TaskClassifierConfig};
pub use algorithms::noop::{Noop, NoopDecision};
pub use algorithms::passthrough::{Passthrough, PassthroughDecision};
pub use algorithms::rand::{Random, RandomClassifier, RandomDecision};
pub use algorithms::stage::{LlmFallback, StageRouter, StageRouterConfig};
pub use algorithms::util::affinity::AffinityRouter;
pub use algorithms::util::escalation::EscalationJudgeConfig;
pub use algorithms::util::prompts::{SystemPromptProcessor, TargetPrompts, append_note};
pub use algorithms::util::subagent::SubagentOverride;
pub use algorithms::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSignals};

// Stage-router scoring and tier selection — the shared signal-driven routing
// core (scorer, picker, and the `StageClassifier`).
pub use algorithms::util::stage::{
    CodingAgentDimensions, DECISION_SOURCE_KEY, DecisionSource, HandoffNoteConfig, PickOutcome,
    PickerMode, ScoreResult, StageClassifier, StageTargets, Tier, dimensions_from_signal,
    pick_tier, score_signal,
};

mod observability;

/// Registers process-wide compatibility gauges with the global meter provider.
///
/// Hosts should call this after installing their OpenTelemetry meter provider.
pub fn initialize_metrics() {
    observability::initialize_metrics();
}

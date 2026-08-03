// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]

//! # switchyard-libsy
//!
//! Provider-neutral routing and multi-model orchestration for LLM applications.
//! An [`Algorithm`] chooses one or more semantic [`LlmTarget`]s; each target's
//! [`RoutedLlmClient`](switchyard_protocol::RoutedLlmClient) performs model I/O.
//! This separation lets the same algorithm run in a proxy, gateway, or agent runtime.
//!
//! ## Setup
//!
//! ```toml
//! [dependencies]
//! switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
//! switchyard-llm-client = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
//! switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```
//!
//! ## Quick start
//!
//! This complete `src/main.rs` sends one request through [`Passthrough`] to an
//! OpenAI-compatible backend. Set `LLM_BASE_URL`, `LLM_MODEL`, and optionally
//! `LLM_API_KEY` before running it.
//!
//! ```no_run
//! use std::collections::BTreeMap;
//! use std::error::Error;
//! use std::sync::Arc;
//!
//! use switchyard_libsy::{Algorithm, LlmTarget, Passthrough};
//! use switchyard_llm_client::{
//!     Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient,
//! };
//! use switchyard_protocol::{Context, Request, completion_text, text_request};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//!     let model = std::env::var("LLM_MODEL")?;
//!     let client = Arc::new(TranslatingLlmClient::new(&[ModelConfig::new(
//!         model.clone(),
//!         Backend::OpenAiChat(HttpBackendConfig {
//!             base_url: std::env::var("LLM_BASE_URL")?,
//!             api_key: std::env::var("LLM_API_KEY").ok(),
//!             extra_headers: BTreeMap::new(),
//!             extra_body: BTreeMap::new(),
//!             max_retries: 2,
//!         }),
//!         None,
//!     )])?);
//!     let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(LlmTarget {
//!         semantic_name: model,
//!         llm_client: Some(client),
//!     }));
//!     let request = Request {
//!         llm_request: text_request(None, "Explain tail latency in one sentence."),
//!         ..Request::default()
//!     };
//!
//!     let (_decisions, response) = algorithm.run(Context::default(), request).await?;
//!     let response = response.llm_response.into_agg().await?;
//!     println!("{}", completion_text(&response));
//!     Ok(())
//! }
//! ```
//!
//! ## Built-in algorithms
//!
//! | Algorithm | Purpose |
//! |---|---|
//! | [`Passthrough`] | Always call one configured target. |
//! | [`Random`] | Select among any number of targets using uniform or weighted routing. |
//! | [`LlmTaskClassifier`] | Ask a judge model to choose an efficient or capable target. |
//! | [`StageRouter`] | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |
//!
//! [`Noop`] is a test helper, not a production routing algorithm.
//!
//! ## Core concepts
//!
//! - [`Algorithm`] owns routing policy and can make one or more model calls per request.
//! - [`LlmTarget`] gives an algorithm a semantic target name and optional default client.
//! - [`Request`](switchyard_protocol::Request) and
//!   [`Response`](switchyard_protocol::Response) carry the provider-neutral conversation.
//! - [`Decision`](switchyard_protocol::Decision) records each selected target and its reasoning.
//!
//! ## Execution modes
//!
//! Use [`Algorithm::run`] when targets have default clients. Use
//! [`Algorithm::run_stream`] when the host owns model transport and needs to fulfill each
//! [`Step::CallLlm`] itself. Both return the same final response and decision trace.
//!
//! ## Operational notes
//!
//! Algorithm instances are shared with `Arc` and may serve concurrent requests; the full
//! implementor contract is documented on [`Algorithm`]. Runs emit `tracing` and
//! OpenTelemetry signals through host-installed global providers; see [`initialize_metrics`]
//! when compatibility gauges must exist before the first request.

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

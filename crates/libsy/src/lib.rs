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
//! switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
//! ```
//!
//! ## Quick start
//!
//! This complete `src/main.rs` constructs a uniform random router over two semantic
//! targets. Add a client to each target before calling [`Algorithm::run`], or let the
//! host fulfill calls through [`Algorithm::run_stream`].
//!
//! ```
//! use std::sync::Arc;
//!
//! use switchyard_libsy::{Algorithm, LlmTarget, LlmTargetSet, Random};
//!
//! fn main() -> switchyard_libsy::Result<()> {
//!     let target = |name: &str| LlmTarget {
//!         semantic_name: name.into(),
//!         llm_client: None,
//!     };
//!     let targets = LlmTargetSet::new(vec![target("fast"), target("strong")]);
//!     let router: Arc<dyn Algorithm> = Arc::new(Random::new(targets, None, None)?);
//!     println!("{}", router.name());
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
//! ## How it fits together
//!
//! [`LlmTarget`] pairs a semantic routing name with an optional
//! [`RoutedLlmClient`](switchyard_protocol::RoutedLlmClient). An [`Algorithm`] selects
//! targets and records decisions. Use [`Algorithm::run`] with target-owned clients, or
//! [`Algorithm::run_stream`] when the host owns model transport. Request, response, usage,
//! and streaming data are defined by [`switchyard-protocol`](switchyard_protocol).

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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

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
/// Hosts should call this after installing their OpenTelemetry meter provider. Other
/// libsy metrics are created when recorded and do not require initialization. Spans
/// and structured logs require the host to install a `tracing` subscriber separately.
pub fn initialize_metrics() {
    observability::initialize_metrics();
}

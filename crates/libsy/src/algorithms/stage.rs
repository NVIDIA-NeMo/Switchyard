// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signal-driven stage routing for coding agents.
//!
//! [`StageRouter`] is the assembled algorithm: a [`FallThrough`] pre-wired with
//! the tool-signal processor that reads each turn's tool results and the
//! [`StageClassifier`] that scores them onto the capable/efficient tiers. The cascade
//! is an internal detail — callers drive the algorithm, not its parts.
//!
//! Signals do not decide every turn. An under-threshold turn abstains and falls
//! through to the optional [`LlmTaskClassifier`] — the capability route's judge,
//! joined in unchanged — and then to the picker's default tier. The judge is
//! asked per turn and its verdict is never pinned to the session.
//!
use std::sync::Arc;

use async_trait::async_trait;

use super::fall_through::{DefaultTarget, FallThrough};
use super::llm_class::{LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig};
use super::util::prompts::{SystemPromptProcessor, TargetPrompts};
use super::util::stage::{
    DecisionSource, HandoffNoteConfig, PickerMode, StageClassifier, StageTargets,
    record_decision_source,
};
use super::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSignalProcessor};
use crate::core::algorithm::{Algorithm, Driver, LlmTarget, LlmTargetSet};
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{Context, Request, Response};

/// Telemetry name for a router this module assembles.
const STAGE_ROUTER: &str = "stage_router";

/// Attributes a turn to the classifier it wraps, when that classifier decides it.
///
/// The classifiers themselves are composition-agnostic and write no state; only
/// this router knows where each sits in its cascade.
struct SourceStamp {
    inner: Arc<dyn Classifier<State>>,
    source: DecisionSource,
}

#[async_trait]
impl Classifier<State> for SourceStamp {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        self.inner.routing_tier(selected_model)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let (classification, served) = self.inner.score(state, request, driver).await?;
        // An abstaining classifier passes the turn on, so it is not its to claim.
        if matches!(&classification, Classification::Scores(scores) if !scores.is_empty()) {
            record_decision_source(state, self.source);
        }
        Ok((classification, served))
    }
}

/// The capability judge a stage router falls through to.
pub struct LlmFallback {
    /// Target the judge model is called through. It is not a routing
    /// destination, so it does not belong in the router's target set.
    pub judge_target: LlmTarget,
    /// Judge configuration. `recent_turn_window` is worth setting to this router's
    /// `recent_window` so the judge reads the same span the signal scorer scored.
    /// Note: `session_affinity` and `message_hash_fallback` have no effect here —
    /// the judge runs as a cascade classifier, not a standalone algorithm.
    pub config: TaskClassifierConfig,
}

/// How a stage router scores turns, and what it hands the model it picks.
pub struct StageRouterConfig {
    /// Tier a turn falls open to when the scorer is not confident.
    pub mode: PickerMode,
    /// How much corroboration a decisive pick needs, in `[0.0, 1.0]`.
    pub confidence_threshold: f64,
    /// Trailing tool results the signals are computed over. `None` uses
    /// [`DEFAULT_RECENT_WINDOW`].
    pub recent_window: Option<usize>,
    /// Note handed to the model on a signal-driven escalation, and on a
    /// hand-back to the efficient tier when a de-escalation note is configured.
    pub handoff_notes: Option<HandoffNoteConfig>,
    /// System prompts keyed by target, handed over on every turn that target
    /// serves. Empty by default.
    pub tier_prompts: TargetPrompts,
    /// Capability judge consulted on turns the signals leave undecided — the
    /// judge's own target, plus the same configuration the standalone capability
    /// route takes.
    pub llm_fallback: Option<LlmFallback>,
}

impl StageRouterConfig {
    /// The signal-only configuration: no notes, no per-tier prompts, no judge.
    /// Set the optional fields to add them.
    pub fn new(mode: PickerMode, confidence_threshold: f64) -> Self {
        Self {
            mode,
            confidence_threshold,
            recent_window: None,
            handoff_notes: None,
            tier_prompts: TargetPrompts::default(),
            llm_fallback: None,
        }
    }
}

/// Routes coding-agent turns between a capable and an efficient tier: tool signals
/// decide first, an optional capability judge takes the turns they cannot, and
/// the picker's default tier closes the cascade so a turn is never left unrouted.
pub struct StageRouter {
    route: FallThrough<State>,
}

impl StageRouter {
    /// Routes between the `capable` and `efficient` targets. The
    /// judge, when configured, is called through its own target and is not a
    /// routing destination.
    ///
    /// Errors if either threshold in `config` is outside `[0.0, 1.0]`.
    pub fn new(
        capable: LlmTarget,
        efficient: LlmTarget,
        config: StageRouterConfig,
    ) -> Result<Self> {
        Ok(Self {
            route: build_route(capable, efficient, config)?,
        })
    }
}

#[async_trait]
impl Algorithm for StageRouter {
    fn name(&self) -> &str {
        STAGE_ROUTER
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.route.execute(ctx, driver, request).await
    }
}

/// Wires the cascade the wrapper drives.
fn build_route(
    capable: LlmTarget,
    efficient: LlmTarget,
    config: StageRouterConfig,
) -> Result<FallThrough<State>> {
    if !(0.0..=1.0).contains(&config.confidence_threshold) {
        return Err(LibsyError::AlgorithmError {
            message: format!(
                "confidence_threshold must be between 0 and 1, got {}",
                config.confidence_threshold
            ),
        });
    }
    // The tiers are a fixed pair; their targets are whatever the deployment calls
    // them, and the classifier scores onto those names.
    let targets = StageTargets::new(
        capable.semantic_name.clone(),
        efficient.semantic_name.clone(),
    );
    // The picker's mode fixes the fallback tier up front, so the terminal
    // classifier is a constant rather than a per-turn lookup.
    let fall_open = targets.name(config.mode.default_tier()).to_string();

    let mut classifier = StageClassifier::new(targets, config.mode, config.confidence_threshold);
    if let Some(notes) = config.handoff_notes {
        classifier = classifier.with_handoff_notes(notes);
    }
    let signals = ToolSignalProcessor {
        recent_window: config.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
    };

    let target_set = LlmTargetSet::new(vec![capable.clone(), efficient.clone()]);
    let mut router = FallThrough::<State>::new_with_state(target_set)
        .with_name(STAGE_ROUTER)
        .with_processor(Arc::new(signals))
        .with_classifier(Arc::new(classifier));
    if let Some(fallback) = config.llm_fallback {
        // The capability judge takes its tiers in the same order the capability
        // route passes them: efficient first, capable second.
        router = router.with_classifier(Arc::new(SourceStamp {
            inner: Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                judge_target: fallback.judge_target,
                efficient_target: efficient,
                capable_target: capable,
                config: fallback.config,
            })?),
            source: DecisionSource::LlmClassifier,
        }));
    }
    // Nothing behind this, so the turn lands on the picker's default tier —
    // including when the judge could not tell.
    router = router.with_classifier(Arc::new(SourceStamp {
        inner: Arc::new(DefaultTarget::new(fall_open)),
        source: DecisionSource::FallOpen,
    }));
    // Runs on the post-decision hook, so it applies to the target the cascade
    // settled on, whichever classifier picked it. With no prompts configured it
    // is a no-op, so there is nothing to branch on.
    router = router.with_processor(Arc::new(SystemPromptProcessor::new(config.tier_prompts)));
    Ok(router)
}

#[cfg(test)]
#[path = "stage_tests.rs"]
mod tests;

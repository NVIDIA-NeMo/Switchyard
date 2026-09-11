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
//! joined in unchanged — and then to the picker's default tier, or to an
//! override a decider ahead of stage set in its place.
//!
use std::sync::Arc;

use async_trait::async_trait;

use super::fall_through::FallThrough;
use super::llm_class::{LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig};
use super::util::prompts::prepend_system_prompt;
use super::util::stage::{
    DecisionSource, HandoffNoteConfig, PickerMode, StageClassifier, Tier, fall_open_tier,
    record_decision_source, record_routing_decision,
};
use super::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSemantics, ToolSignalProcessor};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{Category, Request, Response};

/// Telemetry name for a router this module assembles.
const STAGE_ROUTER: &str = "stage_router";

struct TierPromptProcessor {
    capable: Option<String>,
    efficient: Option<String>,
}

#[async_trait]
impl Processor<State> for TierPromptProcessor {
    async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
        if let Event::Decision {
            request, category, ..
        } = event
        {
            let prompt = match category {
                Some(Category::Capable) => self.capable.as_deref(),
                Some(Category::Efficient) => self.efficient.as_deref(),
                _ => None,
            };
            if let Some(prompt) = prompt {
                prepend_system_prompt(request, prompt);
            }
        }
        Ok(())
    }
}

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
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let (classification, served) = self.inner.score(state, request, driver).await?;
        // An abstaining classifier passes the turn on, so it is not its to claim.
        if let Some(winner) = classification.argmax(false)? {
            record_decision_source(state, self.source);
            record_routing_decision(self.source, &winner.target);
            driver.set_evidence_if_empty(serde_json::json!({
                "source": self.source.as_str(),
            }));
        }
        Ok((classification, served))
    }
}

/// Closes the cascade at zero confidence: a fallback, not a judgement.
struct FallOpen {
    default_tier: Tier,
}

#[async_trait]
impl Classifier<State> for FallOpen {
    async fn score(
        &self,
        state: &mut State,
        _request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let tier = fall_open_tier(state).unwrap_or(self.default_tier);
        let category = match tier {
            Tier::Capable => Category::Capable,
            Tier::Efficient => Category::Efficient,
        };
        let target = driver.first_model_for(&category)?.clone();
        Ok((
            Classification::Scores(vec![Score {
                target,
                confidence: 0.0,
                category: Some(category),
            }]),
            None,
        ))
    }
}

/// The capability judge a stage router falls through to.
pub struct LlmFallback {
    /// Judge configuration. `recent_turn_window` is worth setting to this router's
    /// `recent_window` so the judge reads the same span the signal scorer scored.
    /// Note: `classify_trigger = new_session` and `message_hash_fallback` have no effect here —
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
    /// Exact tool-name semantics added to the built-in coding vocabulary.
    pub tool_semantics: ToolSemantics,
    /// Note handed to the model on a signal-driven escalation, and on a
    /// hand-back to the efficient tier when a de-escalation note is configured.
    pub handoff_notes: Option<HandoffNoteConfig>,
    /// System prompt handed to the runtime capable model.
    pub capable_system_prompt: Option<String>,
    /// System prompt handed to the runtime efficient model.
    pub efficient_system_prompt: Option<String>,
    /// Capability judge consulted on turns the signals leave undecided. It uses
    /// the runtime judge model and the standalone capability route's settings.
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
            tool_semantics: ToolSemantics::default(),
            handoff_notes: None,
            capable_system_prompt: None,
            efficient_system_prompt: None,
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
    /// Routes between the runtime `capable` and `efficient` models. The judge,
    /// when configured, is called through the runtime `judge` model.
    ///
    /// Errors if either threshold in `config` is outside `[0.0, 1.0]`.
    pub fn new(config: StageRouterConfig) -> Result<Self> {
        Ok(Self {
            route: build_stage_route(config)?,
        })
    }
}

#[async_trait]
impl Algorithm for StageRouter {
    fn name(&self) -> &str {
        STAGE_ROUTER
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> Result<crate::RoutingOutcome> {
        self.route.execute(driver, request).await
    }
}

/// Wires the cascade the wrapper drives. Exposed so a composition above can
/// stack a prelude onto it.
pub(crate) fn build_stage_route(config: StageRouterConfig) -> Result<FallThrough<State>> {
    if !(0.0..=1.0).contains(&config.confidence_threshold) {
        return Err(LibsyError::AlgorithmError {
            message: format!(
                "confidence_threshold must be between 0 and 1, got {}",
                config.confidence_threshold
            ),
        });
    }
    config.tool_semantics.validate()?;
    let default_tier = config.mode.default_tier();
    let fall_open = FallOpen { default_tier };

    let mut classifier = StageClassifier::new(config.mode, config.confidence_threshold);
    if let Some(notes) = config.handoff_notes {
        classifier = classifier.with_handoff_notes(notes);
    }
    let signals = ToolSignalProcessor {
        recent_window: config.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
        tool_semantics: config.tool_semantics,
    };
    let mut router = FallThrough::<State>::new_with_state()
        .with_name(STAGE_ROUTER)
        .with_processor(Arc::new(signals))
        .with_classifier(Arc::new(classifier));
    if let Some(fallback) = config.llm_fallback {
        router = router.with_classifier(Arc::new(SourceStamp {
            inner: Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                config: fallback.config,
            })?),
            source: DecisionSource::LlmClassifier,
        }));
    }
    // Nothing behind this, so no turn is left unrouted.
    router = router.with_classifier(Arc::new(SourceStamp {
        inner: Arc::new(fall_open),
        source: DecisionSource::FallOpen,
    }));
    if config.capable_system_prompt.is_some() || config.efficient_system_prompt.is_some() {
        router = router.with_processor(Arc::new(TierPromptProcessor {
            capable: config.capable_system_prompt,
            efficient: config.efficient_system_prompt,
        }));
    }
    Ok(router)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;

    use super::*;
    use crate::algorithms::util::stage::{DECISION_SOURCE_KEY, clear_fall_open, set_fall_open};
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::state::StateValue;
    use crate::core::testing::{empty_driver, test_drive_with_models};
    use switchyard_protocol::{Category, ModelId};

    /// A classifier that always picks `target`, standing in for a cascade member.
    struct Fixed(&'static str);

    #[async_trait]
    impl Classifier<State> for Fixed {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((
                Classification::Scores(vec![Score {
                    target: ModelId::from(self.0),
                    confidence: 1.0,
                    category: None,
                }]),
                None,
            ))
        }
    }

    /// A classifier that never decides.
    struct Abstains;

    #[async_trait]
    impl Classifier<State> for Abstains {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((Classification::Ambiguous(vec![]), None))
        }
    }

    async fn stamped(inner: Arc<dyn Classifier<State>>) -> Result<Option<String>> {
        let stamp = SourceStamp {
            inner,
            source: DecisionSource::LlmClassifier,
        };
        let mut state = State::default();
        stamp
            .score(&mut state, &mut Request::default(), &empty_driver())
            .await?;
        Ok(match state.extra.get(DECISION_SOURCE_KEY) {
            Some(StateValue::String(source)) => Some(source.clone()),
            _ => None,
        })
    }

    #[tokio::test]
    async fn a_deciding_classifier_is_credited_with_the_turn() -> Result<()> {
        assert_eq!(
            stamped(Arc::new(Fixed("strong"))).await?.as_deref(),
            Some("llm-classifier")
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_abstaining_classifier_claims_nothing() -> Result<()> {
        // It passed the turn on, so the next classifier is the one that decided.
        assert_eq!(stamped(Arc::new(Abstains)).await?, None);
        Ok(())
    }

    fn config() -> StageRouterConfig {
        StageRouterConfig::new(PickerMode::EfficientFirst, 0.5)
    }

    fn runtime_models() -> HashMap<Category, Vec<ModelId>> {
        runtime_models_for("strong", "weak")
    }

    fn runtime_models_for(capable: &str, efficient: &str) -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Judge, vec![ModelId::from(JUDGE)]),
            (Category::Efficient, vec![ModelId::from(efficient)]),
            (Category::Capable, vec![ModelId::from(capable)]),
            (
                Category::Any,
                vec![ModelId::from(capable), ModelId::from(efficient)],
            ),
        ]
        .into()
    }

    #[test]
    fn rejects_an_out_of_range_confidence_threshold() {
        let mut config = config();
        config.confidence_threshold = 1.5;
        assert!(matches!(
            StageRouter::new(config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_an_out_of_range_judge_threshold() {
        let mut config = config();
        config.llm_fallback = Some(LlmFallback {
            config: TaskClassifierConfig {
                base_threshold: -0.1,
                ..Default::default()
            },
        });
        assert!(matches!(
            StageRouter::new(config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn builds() -> Result<()> {
        let router = StageRouter::new(config())?;
        assert_eq!(router.name(), STAGE_ROUTER);
        Ok(())
    }

    // ── routing integration tests ────────────────────────────────────────────

    const ESCALATION: &str = "the previous model was stalling; pick up the diagnosis";
    fn recording_router(config: StageRouterConfig) -> Result<Arc<StageRouter>> {
        Ok(Arc::new(StageRouter::new(config)?))
    }

    fn config_with_notes() -> StageRouterConfig {
        let mut c = config();
        c.handoff_notes = Some(HandoffNoteConfig::new(ESCALATION, None, true));
        c
    }

    fn config_with_judge(recorder: &Arc<Recorder>, p_solve: f64) -> StageRouterConfig {
        *recorder.judge_p_solve.lock() = p_solve;
        let mut c = config();
        c.llm_fallback = Some(LlmFallback {
            config: TaskClassifierConfig {
                base_threshold: 0.5,
                recent_turn_window: Some(3),
                ..Default::default()
            },
        });
        c
    }

    /// Stands in for a decider ahead of stage: sets the override once, clears it
    /// once, and leaves the turns between alone.
    #[derive(Default)]
    struct TierDecider {
        requests: Mutex<u32>,
    }

    #[async_trait]
    impl Processor<State> for TierDecider {
        async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
            if matches!(event, Event::Request { .. }) {
                let mut requests = self.requests.lock();
                *requests += 1;
                match *requests {
                    1 => set_fall_open(state, Tier::Efficient),
                    4 => clear_fall_open(state),
                    _ => {}
                }
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn an_override_replaces_the_picker_default_and_leaves_the_signals_alone() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        // The picker would fall open to "strong"; the override says "weak".
        let config = StageRouterConfig::new(PickerMode::CapableFirst, 0.5);
        let route: Arc<dyn Algorithm> =
            Arc::new(build_stage_route(config)?.with_processor(Arc::new(TierDecider::default())));

        let models = runtime_models();
        for is_turn_failed in [false, false, true, false] {
            test_drive_with_models(
                route.clone(),
                turn_request(is_turn_failed),
                models.clone(),
                recorder.serve(),
            )
            .await?;
        }

        let routed = recorder.routed();
        assert_eq!(
            routed[0].target, "weak",
            "an undecided turn takes the override"
        );
        assert_eq!(
            routed[1].target, "weak",
            "which outlives the turn that set it"
        );
        assert_eq!(
            routed[2].target, "strong",
            "a critical failure still reaches the signals"
        );
        assert_eq!(
            routed[3].target, "strong",
            "clearing restores the picker default"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_signal_driven_escalation_hands_the_note_to_the_model() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_notes())?;

        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        test_drive_with_models(
            router.clone(),
            turn_request(true),
            runtime_models_for("runtime-strong", "runtime-weak"),
            recorder.serve(),
        )
        .await?;

        let calls = recorder.routed();
        assert_eq!(calls[0].target, "weak");
        assert_eq!(calls[1].target, "runtime-strong");
        assert!(
            !calls[0].messages.iter().any(|t| t.contains(ESCALATION)),
            "steady-state turn should carry no note: {:?}",
            calls[0].messages
        );
        assert!(
            calls[1]
                .messages
                .last()
                .is_some_and(|t| t.ends_with(ESCALATION)),
            "escalating turn should carry the note last: {:?}",
            calls[1].messages
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_judge_decides_a_turn_the_signals_leave_undecided() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.1))?;

        let (selected_model, _) = test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        let calls = recorder.calls.lock();
        assert!(
            calls.iter().any(|call| call.target == JUDGE),
            "the judge should be recorded as a routing side call"
        );
        assert!(
            calls.iter().any(|call| call.target == "strong"),
            "the selected target should be recorded as an answer call"
        );
        drop(calls);
        assert_eq!(selected_model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn a_decisive_signal_never_reaches_the_judge() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.9))?;

        test_drive_with_models(
            router.clone(),
            turn_request(true),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert!(
            !recorder.calls.lock().iter().any(|c| c.target == JUDGE),
            "a resolved turn should not pay for a judge call"
        );
        assert_eq!(recorder.routed()[0].target, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn the_judges_verdict_is_not_pinned_to_the_session() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.1))?;

        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        *recorder.judge_p_solve.lock() = 0.9;
        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        let routed = recorder.routed();
        assert_eq!(routed[0].target, "strong");
        assert_eq!(routed[1].target, "weak");
        assert_eq!(
            recorder
                .calls
                .lock()
                .iter()
                .filter(|c| c.target == JUDGE)
                .count(),
            2,
            "each undecided turn is its own question"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_judge_that_cannot_tell_lands_on_the_picker_default() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 42.0))?;

        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(recorder.routed()[0].target, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn the_judge_reads_the_window_it_was_configured_with() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.9))?;

        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        let judged = recorder
            .calls
            .lock()
            .iter()
            .find(|c| c.target == JUDGE)
            .map(|c| c.messages.join("|"));
        let Some(judged) = judged else {
            panic!("the judge was never called");
        };
        assert!(
            judged.contains("fix the build"),
            "the judge should see the opening task: {judged}"
        );
        Ok(())
    }
}

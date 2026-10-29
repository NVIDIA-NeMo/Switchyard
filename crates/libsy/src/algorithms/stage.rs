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
use super::util::affinity::ClassifyTrigger;
use super::util::prompts::prepend_system_prompt;
use super::util::stage::{
    DecisionSource, HandoffNoteConfig, PickerMode, StageClassifier, Tier, fall_open_tier,
    record_routing_decision, selected_tier,
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
impl Processor for TierPromptProcessor {
    async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
        if let Event::Decision {
            request,
            selected_model_id,
            category,
            driver,
        } = event
        {
            let prompt = match selected_tier(category.as_ref(), selected_model_id, driver) {
                Some(Tier::Capable) => self.capable.as_deref(),
                Some(Tier::Efficient) => self.efficient.as_deref(),
                None => None,
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
    inner: Arc<dyn Classifier>,
    source: DecisionSource,
}

#[async_trait]
impl Classifier for SourceStamp {
    fn max_pool_targets(&self) -> Option<usize> {
        self.inner.max_pool_targets()
    }

    async fn classify(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(crate::ClassifierOutput, Option<Response>)> {
        let (decision, response) = self.score(state, request, driver).await?;
        if self.source == DecisionSource::LlmClassifier
            && let Some(pool) = driver.pool()
            && pool.target_count() > 2
            && let Some(winner) = decision.argmax(false)?
        {
            return Ok((
                crate::ClassifierOutput::Position {
                    value: pool.position(&winner.target)?,
                    confidence: winner.confidence,
                    ambiguous: false,
                },
                response,
            ));
        }
        Ok((crate::ClassifierOutput::Decision(decision), response))
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let (classification, served) = self.inner.score(state, request, driver).await?;
        // An abstaining classifier passes the turn on, so it is not its to claim.
        if let Some(winner) = classification.argmax(false)? {
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
impl Classifier for FallOpen {
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
    /// Judge configuration. Its `recent_turn_window` counts trailing messages, while
    /// this router's `recent_window` counts tool results.
    /// Note: `classify_trigger` and `message_hash_fallback` have no effect here. The judge
    /// runs as a cascade classifier on every ambiguous turn, not as a standalone algorithm,
    /// so no verdict is retained.
    pub config: TaskClassifierConfig,
}

/// How a stage router scores turns, and what it hands the model it picks.
pub struct StageRouterConfig {
    /// Tier a turn falls open to when the scorer is not confident.
    pub mode: PickerMode,
    /// How much corroboration a decisive pick needs, in `[0.0, 1.0]`.
    pub confidence_threshold: f64,
    /// Threshold after an outer classifier selects capable-first. Defaults to `0.0`.
    pub capable_first_threshold: f64,
    /// Trailing tool results the signals are computed over. `None` uses
    /// [`DEFAULT_RECENT_WINDOW`].
    pub recent_window: Option<usize>,
    /// Exact tool-name semantics added to the built-in coding vocabulary.
    pub tool_semantics: ToolSemantics,
    /// Requests to keep on capable after an escalation. `0` disables the hold;
    /// `-1` pins after switching to this tier.
    pub capable_hold_turns: i64,
    /// Requests to keep on efficient after a switch. `None` defaults to `-1`
    /// under composite's capable-first policy, otherwise `0`. `Some(0)` disables
    /// the hold; `Some(-1)` pins until the session state is cleared.
    pub efficient_hold_turns: Option<i64>,
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
            capable_first_threshold: 0.0,
            recent_window: None,
            tool_semantics: ToolSemantics::default(),
            capable_hold_turns: 2,
            efficient_hold_turns: None,
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
    route: FallThrough,
}

impl StageRouter {
    /// Routes between the runtime `capable` and `efficient` models. The judge,
    /// when configured, is called through the runtime `judge` model.
    ///
    /// Errors if either threshold in `config` is outside `[0.0, 1.0]`, or if the
    /// hold counts are outside `-1..=u32::MAX`. Also rejects a
    /// fallback judge configured to retain verdicts: it runs on every ambiguous
    /// turn, so `classify_trigger` and `message_hash_fallback` would do nothing.
    pub fn new(config: StageRouterConfig) -> Result<Self> {
        if let Some(fallback) = &config.llm_fallback
            && (fallback.config.classify_trigger != ClassifyTrigger::EveryRequest
                || fallback.config.message_hash_fallback)
        {
            return Err(LibsyError::AlgorithmError {
                message: "classify_trigger and message_hash_fallback have no effect on the \
                          stage fallback judge, which runs on every ambiguous turn; remove \
                          them, or use a composite route to judge once per user turn"
                    .to_string(),
            });
        }
        Ok(Self {
            route: build_stage_route(config)?,
        })
    }
    /// Uses the existing custom-schema classifier for an N-target fallback.
    pub fn new_with_custom_fallback(
        config: StageRouterConfig,
        fallback: super::llm_class::CustomClassifierConfig,
        fail_open: bool,
    ) -> Result<Self> {
        if config.llm_fallback.is_some() {
            return Err(LibsyError::AlgorithmError {
                message: "configure only one stage fallback judge".into(),
            });
        }
        if fallback.classify_trigger != ClassifyTrigger::EveryRequest
            || fallback.message_hash_fallback
        {
            return Err(LibsyError::AlgorithmError {
                message: "stage fallback must classify every request without retention".into(),
            });
        }
        let route = build_stage_route_with_fallback(
            config,
            Some(LlmTaskClassifier::custom_classifier(&fallback, fail_open)?),
        )?;
        tracing::info!(fail_open, "Stage categorical fallback configured");
        Ok(Self { route })
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
pub(crate) fn build_stage_route(config: StageRouterConfig) -> Result<FallThrough> {
    build_stage_route_with_fallback(config, None)
}

fn build_stage_route_with_fallback(
    config: StageRouterConfig,
    custom_fallback: Option<Arc<dyn Classifier>>,
) -> Result<FallThrough> {
    for (name, threshold) in [
        ("confidence_threshold", config.confidence_threshold),
        ("capable_first_threshold", config.capable_first_threshold),
    ] {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!("{name} must be between 0 and 1, got {threshold}"),
            });
        }
    }
    config.tool_semantics.validate()?;
    for (name, turns) in [
        ("capable_hold_turns", Some(config.capable_hold_turns)),
        ("efficient_hold_turns", config.efficient_hold_turns),
    ] {
        let Some(turns) = turns else { continue };
        if !(-1..=i64::from(u32::MAX)).contains(&turns) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "{name} must be -1 or between 0 and {}, got {turns}",
                    u32::MAX
                ),
            });
        }
    }
    let default_tier = config.mode.default_tier();
    let fall_open = FallOpen { default_tier };

    let mut classifier = StageClassifier::new(config.mode, config.confidence_threshold)
        .with_capable_first_threshold(config.capable_first_threshold)
        .with_capable_hold_turns(config.capable_hold_turns);
    if let Some(turns) = config.efficient_hold_turns {
        classifier = classifier.with_efficient_hold_turns(turns);
    }
    if let Some(notes) = config.handoff_notes {
        classifier = classifier.with_handoff_notes(notes);
    }
    let signals = ToolSignalProcessor {
        recent_window: config.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
        tool_semantics: config.tool_semantics,
    };
    let classifier = Arc::new(classifier.with_signals(signals));
    let mut router = FallThrough::new_with_state()
        .with_name(STAGE_ROUTER)
        .with_processor(classifier.clone())
        .with_classifier(classifier);
    if let Some(fallback) = config.llm_fallback {
        router = router.with_classifier(Arc::new(SourceStamp {
            inner: Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                config: fallback.config,
            })?),
            source: DecisionSource::LlmClassifier,
        }));
    }
    if let Some(inner) = custom_fallback {
        router = router.with_classifier(Arc::new(SourceStamp {
            inner,
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
    use crate::{CapabilityJudgeConfig, LlmCapabilityConfig};
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;

    use super::*;
    use crate::algorithms::util::stage::{clear_fall_open, set_fall_open};
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::testing::test_drive_with_models;
    use switchyard_protocol::{Category, ModelId};

    /// A classifier that always picks `target`, standing in for a cascade member.
    struct Fixed(&'static str);

    #[async_trait]
    impl Classifier for Fixed {
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
    impl Classifier for Abstains {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((Classification::Ambiguous(vec![]), None))
        }
    }

    async fn stamped(inner: Arc<dyn Classifier>) -> Result<Option<String>> {
        let stamp = SourceStamp {
            inner,
            source: DecisionSource::LlmClassifier,
        };
        let (driver, mut rx) = Driver::new("test", Arc::new(crate::RuntimeModels::default()));
        stamp
            .score(&mut State::default(), &mut Request::default(), &driver)
            .await?;
        driver
            .finish(Ok(crate::RoutingOutcome::route_to(
                "strong".into(),
                Vec::new(),
                Request::default(),
            )))
            .await?;
        let Some(Ok(crate::Step::Done(outcome))) = rx.recv().await else {
            panic!("expected Done step");
        };
        Ok(outcome
            .metadata
            .and_then(|metadata| metadata.evidence)
            .and_then(|evidence| evidence.get("source")?.as_str().map(str::to_string)))
    }

    #[tokio::test]
    async fn n_target_judge_returns_an_anchor_for_final_pool_resolution() -> Result<()> {
        let targets = vec![
            "weak".into(),
            "mid-low".into(),
            "mid-high".into(),
            "strong".into(),
        ];
        let mut groups = runtime_models();
        groups.insert(Category::Any, targets.clone());
        let models = crate::RuntimeModels::new(groups)
            .with_pool(crate::TargetPool::new(targets, vec![0.0, 0.2, 0.8, 1.0])?)?;
        let (driver, _) = Driver::new("test", Arc::new(models));
        for (target, anchor) in [
            ("weak", 0.0),
            ("mid-low", 0.2),
            ("mid-high", 0.8),
            ("strong", 1.0),
        ] {
            let stamp = SourceStamp {
                inner: Arc::new(Fixed(target)),
                source: DecisionSource::LlmClassifier,
            };
            let (output, _) = stamp
                .classify(&mut State::default(), &mut Request::default(), &driver)
                .await?;
            assert!(
                matches!(output, crate::ClassifierOutput::Position { value, confidence: 1.0, ambiguous: false } if value == anchor)
            );
            assert_eq!(output.resolve(&driver)?.unwrap().target.as_str(), target);
        }
        let stamp = SourceStamp {
            inner: Arc::new(Abstains),
            source: DecisionSource::LlmClassifier,
        };
        let (output, _) = stamp
            .classify(&mut State::default(), &mut Request::default(), &driver)
            .await?;
        assert!(output.resolve(&driver)?.is_none());
        let stamp = SourceStamp {
            inner: Arc::new(Fixed("not-in-pool")),
            source: DecisionSource::LlmClassifier,
        };
        assert!(
            stamp
                .classify(&mut State::default(), &mut Request::default(), &driver)
                .await
                .is_err()
        );
        Ok(())
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
                judge: CapabilityJudgeConfig::Llm(LlmCapabilityConfig {
                    base_threshold: -0.1,
                    ..LlmCapabilityConfig::default()
                }),
                ..Default::default()
            },
        });
        assert!(matches!(
            StageRouter::new(config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_retaining_triggers_on_the_fallback_judge() {
        for (trigger, message_hash_fallback) in [
            (ClassifyTrigger::UserTurn, false),
            (ClassifyTrigger::NewSession, false),
            (ClassifyTrigger::UserTurn, true),
        ] {
            let mut config = config();
            config.llm_fallback = Some(LlmFallback {
                config: TaskClassifierConfig {
                    classify_trigger: trigger,
                    message_hash_fallback,
                    ..Default::default()
                },
            });
            assert!(
                matches!(
                    StageRouter::new(config),
                    Err(LibsyError::AlgorithmError { .. })
                ),
                "{trigger:?} with message_hash_fallback = {message_hash_fallback}"
            );
        }
    }

    #[test]
    fn builds() -> Result<()> {
        let router = StageRouter::new(config())?;
        assert_eq!(router.name(), STAGE_ROUTER);
        Ok(())
    }

    #[test]
    fn hold_turns_accept_pins_and_reject_invalid_counts() {
        for capable in [true, false] {
            for turns in [-2, -1, 0, 1, i64::from(u32::MAX), i64::from(u32::MAX) + 1] {
                let mut config = config();
                if capable {
                    config.capable_hold_turns = turns;
                } else {
                    config.efficient_hold_turns = Some(turns);
                }
                assert_eq!(
                    StageRouter::new(config).is_ok(),
                    (-1..=i64::from(u32::MAX)).contains(&turns)
                );
            }
        }
    }

    /// A clean write selects efficient without ending finite capable holds early.
    fn production_request() -> Request {
        let mut request = turn_request(false);
        if let switchyard_protocol::ContentBlock::ToolCall(call) =
            &mut request.llm_request.messages[1].content[0]
        {
            call.name = "Write".to_string();
        }
        request
    }

    #[tokio::test]
    async fn standalone_pickers_default_to_no_efficient_hold() -> Result<()> {
        for mode in [PickerMode::EfficientFirst, PickerMode::CapableFirst] {
            let mut config = StageRouterConfig::new(mode, 0.5);
            config.capable_hold_turns = 0;
            let router = recording_router(config)?;
            let recorder = Arc::new(Recorder::default());
            for request in [turn_request(true), production_request(), turn_request(true)] {
                test_drive_with_models(router.clone(), request, runtime_models(), recorder.serve())
                    .await?;
            }
            let actual: Vec<_> = recorder
                .routed()
                .into_iter()
                .map(|call| call.target)
                .collect();
            assert_eq!(actual, ["strong", "weak", "strong"]);
        }
        Ok(())
    }

    /// Both pickers honor both holds; the switch request is not one of the held turns.
    #[tokio::test]
    async fn holds_expire_or_pin_after_switching_in_both_pickers() -> Result<()> {
        for mode in [PickerMode::EfficientFirst, PickerMode::CapableFirst] {
            for capable in [true, false] {
                for turns in [0, 1, 2, -1] {
                    let mut config = StageRouterConfig::new(mode, 0.5);
                    config.capable_hold_turns = if capable { turns } else { 0 };
                    config.efficient_hold_turns = Some(if capable { 0 } else { turns });
                    let router = recording_router(config)?;
                    let recorder = Arc::new(Recorder::default());
                    let (switch, opposing, held_target, other_target) = if capable {
                        (turn_request(true), production_request(), "strong", "weak")
                    } else {
                        (production_request(), turn_request(true), "weak", "strong")
                    };
                    let mut requests = vec![opposing.clone(), switch];
                    requests.extend(std::iter::repeat_n(opposing, 4));
                    for request in requests {
                        test_drive_with_models(
                            router.clone(),
                            request,
                            runtime_models(),
                            recorder.serve(),
                        )
                        .await?;
                    }
                    let actual: Vec<_> = recorder
                        .routed()
                        .into_iter()
                        .map(|call| call.target)
                        .collect();
                    let mut expected = vec![other_target, held_target];
                    expected.extend((0..4).map(|i| {
                        if turns == -1 || i < turns {
                            held_target
                        } else {
                            other_target
                        }
                    }));
                    assert_eq!(
                        actual, expected,
                        "{mode:?}, capable={capable}, turns={turns}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Initial selection does not pin. Once switched, pins survive missing signals,
    /// passing tests, and critical errors, and remain scoped to one session/agent.
    #[tokio::test]
    async fn pins_survive_signals_and_remain_isolated() -> Result<()> {
        for mode in [PickerMode::EfficientFirst, PickerMode::CapableFirst] {
            for capable in [true, false] {
                let mut config = StageRouterConfig::new(mode, 0.5);
                config.capable_hold_turns = -1;
                config.efficient_hold_turns = Some(-1);
                let router = recording_router(config)?;
                let recorder = Arc::new(Recorder::default());
                let (initial, switch, held, other) = if capable {
                    (production_request(), turn_request(true), "strong", "weak")
                } else {
                    (turn_request(true), production_request(), "weak", "strong")
                };
                let mut no_signals = turn_request(false);
                no_signals.llm_request.messages.truncate(1);
                let mut passing = turn_request(false);
                if let switchyard_protocol::ContentBlock::ToolResult(result) =
                    &mut passing.llm_request.messages[2].content[0]
                {
                    result.content = vec![switchyard_protocol::ContentBlock::Text {
                        text: "test result: ok. 1 passed; 0 failed".into(),
                    }];
                }
                let mut child = initial.clone();
                child.metadata.as_mut().unwrap().agent_id = Some("child".into());
                let mut other_session = initial.clone();
                other_session.metadata.as_mut().unwrap().session_id = Some("other".into());
                let mut final_request = initial.clone();
                final_request.metadata.as_mut().unwrap().session_final = Some(true);
                let requests = [
                    initial.clone(),
                    switch,
                    no_signals,
                    passing,
                    initial.clone(),
                    child,
                    other_session,
                    final_request,
                    initial,
                ];
                for request in requests {
                    test_drive_with_models(
                        router.clone(),
                        request,
                        runtime_models(),
                        recorder.serve(),
                    )
                    .await?;
                }
                let actual: Vec<_> = recorder
                    .routed()
                    .into_iter()
                    .map(|call| call.target)
                    .collect();
                assert_eq!(
                    actual,
                    [other, held, held, held, held, other, other, held, other]
                );
            }
        }
        Ok(())
    }

    /// Judge-driven switches must arm the same pin as signal-driven switches.
    #[tokio::test]
    async fn judge_switches_arm_pins_and_held_turns_skip_the_judge() -> Result<()> {
        for capable in [true, false] {
            let recorder = Arc::new(Recorder::default());
            let mut config = config_with_judge(&recorder, if capable { 1.0 } else { 0.0 });
            config.capable_hold_turns = -1;
            config.efficient_hold_turns = Some(-1);
            let router = recording_router(config)?;
            for p_solve in [
                if capable { 1.0 } else { 0.0 },
                if capable { 0.0 } else { 1.0 },
                if capable { 1.0 } else { 0.0 },
            ] {
                *recorder.judge_p_solve.lock() = p_solve;
                test_drive_with_models(
                    router.clone(),
                    turn_request(false),
                    runtime_models(),
                    recorder.serve(),
                )
                .await?;
            }
            let actual: Vec<_> = recorder
                .routed()
                .into_iter()
                .map(|call| call.target)
                .collect();
            assert_eq!(
                actual,
                if capable {
                    ["weak", "strong", "strong"]
                } else {
                    ["strong", "weak", "weak"]
                }
            );
            assert_eq!(recorder.judge_calls(), 2);
        }
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
                judge: CapabilityJudgeConfig::Llm(LlmCapabilityConfig {
                    base_threshold: 0.5,
                    ..LlmCapabilityConfig::default()
                }),
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
    impl Processor for TierDecider {
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

    #[test]
    fn custom_fallback_rejects_conflicts_and_invalid_contracts() {
        use crate::{CustomClassifierConfig, CustomClassifierPolicy};
        use serde_json::json;
        let fallback = |prompt: &str, schema, selector: &str| {
            CustomClassifierConfig::new(
                prompt,
                schema,
                CustomClassifierPolicy::target_selector(selector),
            )
        };
        for (prompt, schema, selector, expected) in [
            (
                "route",
                json!({"type":"invalid"}),
                "/target",
                "valid JSON Schema",
            ),
            ("route", json!({"type":"object"}), "target", "JSON Pointer"),
            (
                " ",
                json!({"type":"object"}),
                "/target",
                "prompt must not be empty",
            ),
        ] {
            let error = StageRouter::new_with_custom_fallback(
                config(),
                fallback(prompt, schema, selector),
                true,
            )
            .err()
            .expect("invalid contract");
            assert!(error.to_string().contains(expected), "{error}");
        }
        let mut config = config();
        config.llm_fallback = Some(LlmFallback {
            config: TaskClassifierConfig::default(),
        });
        let error = StageRouter::new_with_custom_fallback(
            config,
            fallback("route", json!({"type":"object"}), "/target"),
            true,
        )
        .err()
        .expect("conflicting fallbacks");
        assert!(error.to_string().contains("only one stage fallback"));
    }

    #[tokio::test]
    async fn binary_fallback_rejects_n_target_pools_before_any_call() -> Result<()> {
        for count in [2, 3, 4] {
            let recorder = Arc::new(Recorder::default());
            let router = Arc::new(StageRouter::new(config_with_judge(&recorder, 0.9))?);
            let mut targets = vec![ModelId::from("weak"), ModelId::from("strong")];
            for i in 2..count {
                targets.insert(1, format!("middle-{i}").into());
            }
            let mut groups = runtime_models();
            groups.insert(Category::Any, targets.clone());
            let models = crate::RuntimeModels::new(groups)
                .with_pool(crate::TargetPool::new(targets, vec![])?)?;
            let result =
                test_drive_with_models(router, turn_request(false), models, recorder.serve()).await;
            if count > 2 {
                assert!(
                    result
                        .err()
                        .expect("binary judge must reject N targets")
                        .to_string()
                        .contains("supports at most 2 pool targets")
                );
                assert_eq!(recorder.judge_calls(), 0);
                assert!(recorder.calls.lock().is_empty());
            } else {
                result?;
                assert_eq!(recorder.judge_calls(), 1);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn three_targets_route_neutral_signals_to_the_middle_and_preserve_override_holds()
    -> Result<()> {
        use crate::{RuntimeModels, TargetPool};
        let mut groups = runtime_models();
        groups
            .get_mut(&Category::Any)
            .unwrap()
            .push("middle".into());
        let models = RuntimeModels::new(groups).with_pool(TargetPool::new(
            vec!["weak".into(), "middle".into(), "strong".into()],
            vec![],
        )?)?;
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(StageRouter::new(config())?);
        for (failed, expected) in [
            (false, "middle"),
            (true, "strong"),
            (false, "strong"),
            (false, "strong"),
            (false, "middle"),
        ] {
            let (target, _) = test_drive_with_models(
                router.clone(),
                turn_request(failed),
                models.clone(),
                recorder.serve(),
            )
            .await?;
            assert_eq!(target, expected);
        }
        assert_eq!(recorder.judge_calls(), 0);
        Ok(())
    }
}

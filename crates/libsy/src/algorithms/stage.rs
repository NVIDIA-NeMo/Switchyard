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
    record_decision_source, record_routing_decision,
};
use super::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSignalProcessor};
use crate::core::algorithm::{Algorithm, Driver, LlmTarget, LlmTargetSet};
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{Request, Response};

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
    fn routing_tier(&self, selected_model_id: &str) -> Option<&'static str> {
        self.inner.routing_tier(selected_model_id)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let (classification, served) = self.inner.score(state, request, driver).await?;
        // An abstaining classifier passes the turn on, so it is not its to claim.
        if let Some(winner) = classification.argmax(false)? {
            record_decision_source(state, self.source);
            record_routing_decision(self.source, &winner.target);
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
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.route.execute(driver, request).await
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
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, LlmRequest, Message, Role, ToolCall, ToolResult, WireFormat,
    };

    use super::*;
    use crate::algorithms::util::stage::DECISION_SOURCE_KEY;
    use crate::core::algorithm::LlmTarget;
    use crate::core::classifier::Score;
    use crate::core::state::StateValue;
    use crate::core::testing::{Serve, reply, test_drive};
    use switchyard_protocol::{Decision, Metadata, Response};

    fn tier_target(name: &str) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
        }
    }

    /// A classifier that always picks `target`, standing in for a cascade member.
    struct Fixed(&'static str);

    #[async_trait]
    impl Classifier<State> for Fixed {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((
                Classification::Scores(vec![Score {
                    target: self.0.to_string(),
                    confidence: 1.0,
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
            _driver: Option<&Driver>,
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
            .score(&mut state, &mut Request::default(), None)
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

    #[test]
    fn rejects_an_out_of_range_confidence_threshold() {
        let mut config = config();
        config.confidence_threshold = 1.5;
        assert!(matches!(
            StageRouter::new(tier_target("strong"), tier_target("weak"), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn rejects_an_out_of_range_judge_threshold() {
        let mut config = config();
        config.llm_fallback = Some(LlmFallback {
            judge_target: LlmTarget {
                semantic_name: "judge".to_string(),
            },
            config: TaskClassifierConfig {
                base_threshold: -0.1,
                ..Default::default()
            },
        });
        assert!(matches!(
            StageRouter::new(tier_target("strong"), tier_target("weak"), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn builds_over_both_tiers() -> Result<()> {
        let router = StageRouter::new(tier_target("strong"), tier_target("weak"), config())?;
        assert_eq!(router.name(), STAGE_ROUTER);
        Ok(())
    }

    // ── routing integration tests ────────────────────────────────────────────

    const ESCALATION: &str = "the previous model was stalling; pick up the diagnosis";
    const JUDGE: &str = "judge";

    #[derive(Clone, Debug)]
    struct Call {
        target: String,
        messages: Vec<String>,
        is_answer_call: bool,
    }

    /// Records what each target receives.
    #[derive(Default)]
    struct Recorder {
        calls: Mutex<Vec<Call>>,
        judge_p_solve: Mutex<f64>,
    }

    impl Recorder {
        fn routed(&self) -> Vec<Call> {
            self.calls
                .lock()
                .iter()
                .filter(|call| call.is_answer_call)
                .cloned()
                .collect()
        }

        /// Serves every call, recording it. The judge target gets a structured verdict
        /// back so the fallback classifier has an answer without a real model.
        fn serve(self: &Arc<Self>) -> impl Serve {
            let recorder = Arc::clone(self);
            move |decision: Decision, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    let target = decision.selected_model_id().to_string();
                    recorder.calls.lock().push(Call {
                        target: target.clone(),
                        messages: request
                            .llm_request
                            .messages
                            .iter()
                            .filter_map(|message| message.text_content("|"))
                            .collect(),
                        is_answer_call: decision.is_answer_call(),
                    });
                    let completion = if target == JUDGE {
                        let p_solve = *recorder.judge_p_solve.lock();
                        format!(
                            r#"{{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":{p_solve}}}"#
                        )
                    } else {
                        target
                    };
                    Ok(reply(completion))
                }
            }
        }
    }

    fn recording_target(name: &str) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
        }
    }

    fn recording_router(config: StageRouterConfig) -> Result<Arc<StageRouter>> {
        Ok(Arc::new(StageRouter::new(
            recording_target("strong"),
            recording_target("weak"),
            config,
        )?))
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
            judge_target: recording_target(JUDGE),
            config: TaskClassifierConfig {
                base_threshold: 0.5,
                recent_turn_window: Some(3),
                ..Default::default()
            },
        });
        c
    }

    fn turn_request(failed: bool) -> Request {
        let content = if failed {
            "fatal runtime error: out of memory"
        } else {
            "ok"
        };
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![
                    Message::text(Role::User, "fix the build"),
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call_1".to_string(),
                            name: "Bash".to_string(),
                            arguments: json!({"command": "cargo test"}),
                        })],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "call_1".to_string(),
                            content: vec![ContentBlock::Text {
                                text: content.to_string(),
                            }],
                            is_error: Some(failed),
                        })],
                    },
                ],
                ..LlmRequest::default()
            },
            raw_request: Some(json!({
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "fix the build"},
                    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": content},
                ],
            })),
            metadata: Some(Metadata {
                wire_format: Some(WireFormat::OpenAiChat),
                session_id: Some("session-1".to_string()),
                ..Default::default()
            }),
        }
    }

    #[tokio::test]
    async fn a_signal_driven_escalation_hands_the_note_to_the_model() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_notes())?;

        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;
        test_drive(router.clone(), turn_request(true), recorder.serve()).await?;

        let calls = recorder.routed();
        assert_eq!(calls[0].target, "weak");
        assert_eq!(calls[1].target, "strong");
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

        let (trace, _) = test_drive(router.clone(), turn_request(false), recorder.serve()).await?;

        let calls = recorder.calls.lock();
        assert!(
            calls
                .iter()
                .any(|call| call.target == JUDGE && !call.is_answer_call),
            "the judge should be recorded as a routing side call"
        );
        assert!(
            calls
                .iter()
                .any(|call| call.target == "strong" && call.is_answer_call),
            "the selected target should be recorded as an answer call"
        );
        drop(calls);
        assert!(
            trace
                .last()
                .and_then(|decision| decision.reasoning())
                .is_some_and(|reasoning| reasoning.contains("routing tier: strong"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_decisive_signal_never_reaches_the_judge() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.9))?;

        test_drive(router.clone(), turn_request(true), recorder.serve()).await?;

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

        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;
        *recorder.judge_p_solve.lock() = 0.9;
        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;

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

        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;

        assert_eq!(recorder.routed()[0].target, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn the_judge_reads_the_window_it_was_configured_with() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = recording_router(config_with_judge(&recorder, 0.9))?;

        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;

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

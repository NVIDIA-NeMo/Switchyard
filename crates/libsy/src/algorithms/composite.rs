// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing that stacks a judge over a stage router.
//!
//! The judge runs as a [`Processor`]: it sets configuration the stage router reads,
//! and picks no target itself.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use super::fall_through::FallThrough;
use super::llm_class::{
    CustomClassifierConfig, LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig,
};
use super::stage::{StageRouterConfig, build_stage_route};
use super::util::affinity::{ClassifyTrigger, evict_if_full, has_new_user_turn, retention_key};
use super::util::stage::{Tier, fall_open_tier, set_fall_open};
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::classifier::Classifier;
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{Category, Request};

const COMPOSITE: &str = "composite";

/// Sets the stage router's fall-open tier from a judge verdict.
///
/// Retains the tier per routing identity, so it survives requests that carry no
/// session ID when `message_hash_fallback` is on. The retained tier is replayed
/// into state on every request so the cascade below reads it.
struct TierSetter {
    judge: Arc<dyn Classifier>,
    trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    default_tier: Tier,
    tiers: Mutex<HashMap<RoutingIdentity, Tier>>,
}

impl TierSetter {
    fn tier_for(category: Option<Category>) -> Option<Tier> {
        match category {
            Some(Category::Capable) => Some(Tier::Capable),
            Some(Category::Efficient) => Some(Tier::Efficient),
            _ => None,
        }
    }

    /// Two requests for one identity can both pass this and both judge, since a
    /// judge call sits between here and [`retain`](Self::retain). The later wins.
    fn is_due(&self, identity: Option<&RoutingIdentity>, state: &State, request: &Request) -> bool {
        match self.trigger {
            ClassifyTrigger::UserTurn => has_new_user_turn(&request.llm_request.messages),
            // Unkeyed requests with no carried tier cannot be told apart, so every one
            // is a new session.
            ClassifyTrigger::NewSession => {
                identity.is_none_or(|identity| !self.tiers.lock().contains_key(identity))
                    && fall_open_tier(state).is_none()
            }
            // Rejected by the constructor, and only in the enum for the standalone route.
            ClassifyTrigger::EveryRequest => true,
        }
    }

    fn retain(&self, identity: RoutingIdentity, tier: Tier) {
        let mut tiers = self.tiers.lock();
        // A user turn re-decides, so it overwrites. A session keeps its first verdict,
        // matching how affinity retains an assignment.
        let writable = self.trigger == ClassifyTrigger::UserTurn || !tiers.contains_key(&identity);
        if writable {
            evict_if_full(&mut tiers);
            tiers.insert(identity, tier);
        }
    }
}

#[async_trait]
impl Processor for TierSetter {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        let Event::Request { request, driver } = event else {
            return Ok(());
        };
        let identity = retention_key(request, self.message_hash_fallback);
        if self.is_due(identity.as_ref(), state, request) {
            let (classification, _) = self.judge.score(state, request, driver).await?;
            if let Some(winner) = classification.argmax(false)?
                && let Some(tier) = Self::tier_for(winner.category)
            {
                set_fall_open(state, tier);
                if let Some(identity) = identity {
                    self.retain(identity, tier);
                }
                return Ok(());
            }
            if self.trigger == ClassifyTrigger::NewSession
                && let Some(identity) = identity.as_ref()
            {
                // A failed opening judgment is still the session's one judgment.
                self.retain(identity.clone(), self.default_tier);
            }
        }
        // Either not this request's turn or the judge had no verdict, so the last
        // tier stands rather than dropping back to the picker default.
        if let Some(tier) = identity
            .and_then(|identity| self.tiers.lock().get(&identity).copied())
            .or_else(|| fall_open_tier(state))
        {
            set_fall_open(state, tier);
            driver.set_evidence_if_empty(serde_json::json!({"source": "retained"}));
        }
        Ok(())
    }
}

/// Classifier settings for the composite router's retained tier.
pub enum CompositeJudgeConfig {
    /// Selects a tier from a capability verdict.
    Capability(TaskClassifierConfig),
    /// Selects a tier using a custom schema and policy, retaining the fallback on failure.
    Custom(CustomClassifierConfig),
}

/// A judge stacked over a stage router.
pub struct CompositeRouterConfig {
    /// Judge settings, including how often `classify_trigger` runs it.
    pub judge: CompositeJudgeConfig,
    /// Serves the turns, with the tier the judge picked as its fall-open default.
    pub stage: StageRouterConfig,
}

/// Runs a stage router with a tier the judge picks.
pub struct CompositeRouter {
    route: FallThrough,
}

impl CompositeRouter {
    /// Stacks the judge over a stage router across the same tier pair.
    ///
    /// Errors on a configuration either algorithm rejects and on `every_request`.
    ///
    /// A stage router carrying its own judge is allowed, but that judge sits ahead
    /// of the fall-open tier and so answers most of the turns this one set a tier for.
    pub fn new(config: CompositeRouterConfig) -> Result<Self> {
        let (trigger, message_hash_fallback) = match &config.judge {
            CompositeJudgeConfig::Capability(judge) => {
                (judge.classify_trigger, judge.message_hash_fallback)
            }
            CompositeJudgeConfig::Custom(config) => {
                (config.classify_trigger, config.message_hash_fallback)
            }
        };
        if trigger == ClassifyTrigger::EveryRequest {
            return Err(LibsyError::AlgorithmError {
                message: "composite: classify_trigger must be user_turn or new_session".to_string(),
            });
        }
        let judge: Arc<dyn Classifier> = match config.judge {
            CompositeJudgeConfig::Capability(judge) => {
                // Composite owns the trigger and retention; standalone affinity never runs.
                Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    config: TaskClassifierConfig {
                        classify_trigger: ClassifyTrigger::EveryRequest,
                        message_hash_fallback: false,
                        ..judge
                    },
                })?)
            }
            CompositeJudgeConfig::Custom(config) => {
                if matches!(config.policy, crate::CustomClassifierPolicy::HandoffNote(_)) {
                    return Err(LibsyError::AlgorithmError {
                        message: "composite requires a tier-selecting policy; handoff_note fixes the target".into(),
                    });
                }
                LlmTaskClassifier::custom_classifier(&config, true)?
            }
        };
        let setter = TierSetter {
            judge,
            trigger,
            message_hash_fallback,
            default_tier: config.stage.mode.default_tier(),
            tiers: Mutex::new(HashMap::new()),
        };
        let route = build_stage_route(config.stage)?
            .with_name(COMPOSITE)
            .with_processor(Arc::new(setter));
        Ok(Self { route })
    }
}

#[async_trait]
impl Algorithm for CompositeRouter {
    fn name(&self) -> &str {
        COMPOSITE
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> Result<crate::RoutingOutcome> {
        self.route.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{CapabilityJudgeConfig, LlmCapabilityConfig};
    use std::collections::HashMap;
    use std::sync::Arc;

    use switchyard_protocol::{Category, LlmClientError, Message, ModelId, Role};

    use super::*;
    use crate::algorithms::util::stage::PickerMode;
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::testing::test_drive_with_models;

    fn runtime_models() -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Judge, vec![ModelId::from(JUDGE)]),
            (Category::Efficient, vec![ModelId::from("weak")]),
            (Category::Capable, vec![ModelId::from("strong")]),
            (
                Category::Any,
                vec![ModelId::from("strong"), ModelId::from("weak")],
            ),
        ]
        .into()
    }

    fn user_turn_request() -> Request {
        let mut request = turn_request(false);
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, "now rewrite the parser"));
        request
    }

    /// The same request shape with no session ID, so only the hash can key it.
    fn unkeyed(mut request: Request) -> Request {
        if let Some(metadata) = request.metadata.as_mut() {
            metadata.session_id = None;
        }
        request
    }

    fn hash_keyed_router() -> Result<Arc<CompositeRouter>> {
        Ok(Arc::new(CompositeRouter::new(CompositeRouterConfig {
            judge: CompositeJudgeConfig::Capability(TaskClassifierConfig {
                judge: CapabilityJudgeConfig::Llm(LlmCapabilityConfig {
                    base_threshold: 0.5,
                    ..LlmCapabilityConfig::default()
                }),
                classify_trigger: ClassifyTrigger::UserTurn,
                message_hash_fallback: true,
                ..Default::default()
            }),
            stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        })?))
    }

    fn router() -> Result<Arc<CompositeRouter>> {
        router_with_trigger(ClassifyTrigger::UserTurn)
    }

    fn router_with_trigger(classify_trigger: ClassifyTrigger) -> Result<Arc<CompositeRouter>> {
        Ok(Arc::new(CompositeRouter::new(CompositeRouterConfig {
            judge: CompositeJudgeConfig::Capability(TaskClassifierConfig {
                judge: CapabilityJudgeConfig::Llm(LlmCapabilityConfig {
                    base_threshold: 0.5,
                    ..LlmCapabilityConfig::default()
                }),
                classify_trigger,
                ..Default::default()
            }),
            stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        })?))
    }

    #[test]
    fn rejects_every_request_as_a_trigger() {
        for judge in [
            CompositeJudgeConfig::Capability(TaskClassifierConfig::default()),
            CompositeJudgeConfig::Custom(CustomClassifierConfig::new(
                "Choose a tier",
                serde_json::json!({"type": "object"}),
                crate::CustomClassifierPolicy::target_selector("/decision/target"),
            )),
        ] {
            let error = CompositeRouter::new(CompositeRouterConfig {
                judge,
                stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
            })
            .err()
            .expect("every_request must be rejected for both classifier modes");
            assert!(matches!(error, LibsyError::AlgorithmError { message }
                if message == "composite: classify_trigger must be user_turn or new_session"));
        }
    }

    fn custom_judge(trigger: ClassifyTrigger) -> CompositeJudgeConfig {
        let mut config = CustomClassifierConfig::new(
            "Select a tier",
            serde_json::json!({"type": "object"}),
            crate::CustomClassifierPolicy::target_selector("/decision/target"),
        );
        config.classify_trigger = trigger;
        CompositeJudgeConfig::Custom(config)
    }

    #[test]
    fn fixed_hint_targets_cannot_be_used_as_tier_judges() -> Result<()> {
        let policy = crate::HandoffNotePolicy::new(
            ModelId::from("target"),
            "/note",
            [("retry".into(), "Inspect the failure.".into())].into(),
        )?;
        let mut config = CustomClassifierConfig::new(
            "Choose an optional note",
            serde_json::json!({"type": "object"}),
            crate::CustomClassifierPolicy::HandoffNote(policy),
        );
        config.classify_trigger = ClassifyTrigger::UserTurn;
        let error = CompositeRouter::new(CompositeRouterConfig {
            judge: CompositeJudgeConfig::Custom(config),
            stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        })
        .err()
        .expect("a tier picker cannot preserve the hint policy's fixed target");
        assert!(error.to_string().contains("handoff_note fixes the target"));
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_session_judgment_retains_the_configured_stage_default() -> Result<()> {
        for custom in [false, true] {
            for (mode, expected) in [
                (PickerMode::EfficientFirst, "weak"),
                (PickerMode::CapableFirst, "strong"),
            ] {
                let judge = if custom {
                    custom_judge(ClassifyTrigger::NewSession)
                } else {
                    CompositeJudgeConfig::Capability(TaskClassifierConfig {
                        classify_trigger: ClassifyTrigger::NewSession,
                        ..Default::default()
                    })
                };
                let router = Arc::new(CompositeRouter::new(CompositeRouterConfig {
                    judge,
                    stage: StageRouterConfig::new(mode, 0.5),
                })?);
                let calls = Arc::new(Mutex::new(0));
                let serve = {
                    let calls = calls.clone();
                    move |target: ModelId, _request: Request| {
                        let completion = if target.as_str() == JUDGE {
                            *calls.lock() += 1;
                            "not JSON".to_string()
                        } else {
                            target.to_string()
                        };
                        async move { Ok(crate::core::testing::reply(completion)) }
                    }
                };
                for request in [
                    user_turn_request(),
                    turn_request(false),
                    user_turn_request(),
                ] {
                    let (selected, _) = test_drive_with_models(
                        router.clone(),
                        request,
                        runtime_models(),
                        serve.clone(),
                    )
                    .await?;
                    assert_eq!(selected.as_str(), expected);
                }
                assert_eq!(
                    *calls.lock(),
                    1,
                    "new_session must retain a failed judgment"
                );
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_custom_judge_sets_the_tier_and_retains_it_on_failure() -> Result<()> {
        let router = Arc::new(CompositeRouter::new(CompositeRouterConfig {
            judge: custom_judge(ClassifyTrigger::UserTurn),
            stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        })?);
        let replies = Arc::new(Mutex::new(
            [
                Some(r#"{"decision":{"target":"capable"}}"#),
                None,
                Some("not JSON"),
                Some(r#"{"decision":{"target":"efficient"}}"#),
            ]
            .into_iter(),
        ));
        let serve = {
            let replies = replies.clone();
            move |target: ModelId, _request: Request| {
                let completion = if target.as_str() == JUDGE {
                    replies
                        .lock()
                        .next()
                        .expect("unexpected judge call")
                        .map(str::to_string)
                } else {
                    Some(target.to_string())
                };
                async move {
                    completion
                        .map(crate::core::testing::reply)
                        .ok_or_else(|| LlmClientError::General("judge unavailable".into()))
                }
            }
        };
        for (request, expected) in [
            (user_turn_request(), "strong"),
            (turn_request(false), "strong"),
            (user_turn_request(), "strong"),
            (user_turn_request(), "strong"),
            (user_turn_request(), "weak"),
        ] {
            let (selected, _) =
                test_drive_with_models(router.clone(), request, runtime_models(), serve.clone())
                    .await?;
            assert_eq!(selected.as_str(), expected);
        }
        assert!(replies.lock().next().is_none());
        Ok(())
    }

    /// A route may point its judge at the same target it serves capable turns on.
    /// The tier then cannot be recovered by looking the served model up in the
    /// runtime groups — one id, two categories — so it comes off the verdict itself.
    #[tokio::test]
    async fn a_judge_sharing_the_capable_model_still_latches_the_tier() -> Result<()> {
        let models: HashMap<Category, Vec<ModelId>> = [
            (Category::Judge, vec![ModelId::from("strong")]),
            (Category::Capable, vec![ModelId::from("strong")]),
            (Category::Efficient, vec![ModelId::from("weak")]),
            (
                Category::Any,
                vec![ModelId::from("strong"), ModelId::from("weak")],
            ),
        ]
        .into();
        // The judge runs first as a request-side processor, so the opening call is its own.
        let calls = Arc::new(Mutex::new(0u32));
        let serve = {
            let calls = Arc::clone(&calls);
            move |target: ModelId, _request: Request| {
                let calls = Arc::clone(&calls);
                async move {
                    let mut calls = calls.lock();
                    *calls += 1;
                    let completion = if *calls == 1 {
                        // p_solve below the threshold: the judge does not trust the
                        // efficient tier, so the verdict is capable.
                        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.1}"#.to_string()
                    } else {
                        target.to_string()
                    };
                    Ok(crate::core::testing::reply(completion))
                }
            }
        };
        let router = router()?;

        test_drive_with_models(
            router.clone(),
            user_turn_request(),
            models.clone(),
            serve.clone(),
        )
        .await?;
        // A tool step is not a user turn, so nothing re-judges and the latched tier decides.
        let (selected, _) =
            test_drive_with_models(router, turn_request(false), models, serve).await?;

        assert_eq!(selected, ModelId::from("strong"));
        Ok(())
    }

    #[tokio::test]
    async fn a_session_without_an_id_keys_on_the_message_hash() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.1;
        let router = hash_keyed_router()?;

        test_drive_with_models(
            router.clone(),
            unkeyed(user_turn_request()),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        test_drive_with_models(
            router.clone(),
            unkeyed(turn_request(false)),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        assert_eq!(
            recorder.judge_calls(),
            1,
            "a tool step is not a user turn, session id or not"
        );
        assert_eq!(
            recorder.routed()[1].target,
            "strong",
            "and the tier survives the tool step"
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_judge_sets_the_tier_once_a_turn_and_the_signals_run_within_it() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.1;
        let router = router()?;

        test_drive_with_models(
            router.clone(),
            user_turn_request(),
            runtime_models(),
            recorder.serve(),
        )
        .await?;
        test_drive_with_models(
            router.clone(),
            turn_request(false),
            runtime_models(),
            recorder.serve(),
        )
        .await?;

        let routed = recorder.routed();
        assert_eq!(
            routed[0].target, "strong",
            "a quiet turn falls open to the verdict"
        );
        assert_eq!(
            routed[1].target, "strong",
            "which holds across the tool steps after it"
        );
        assert_eq!(
            recorder.judge_calls(),
            1,
            "a tool step is not a new user turn"
        );
        Ok(())
    }

    #[tokio::test]
    async fn caller_state_carries_the_judged_tier_without_a_session_id() -> Result<()> {
        for trigger in [ClassifyTrigger::UserTurn, ClassifyTrigger::NewSession] {
            let recorder = Arc::new(Recorder::default());
            *recorder.judge_p_solve.lock() = 0.1;
            let router = router_with_trigger(trigger)?;
            let run = |request, state| {
                crate::core::testing::test_drive_with_state(
                    router.clone(),
                    request,
                    runtime_models(),
                    state,
                    recorder.serve(),
                )
            };

            let first = run(unkeyed(user_turn_request()), State::default()).await?;
            assert_eq!(first.selected_model_id()?, &ModelId::from("strong"));

            let continued = run(unkeyed(turn_request(false)), first.state).await?;
            assert_eq!(continued.selected_model_id()?, &ModelId::from("strong"));
            assert_eq!(
                continued.metadata.and_then(|m| m.evidence),
                Some(serde_json::json!({"source": "retained"}))
            );
            assert_eq!(recorder.judge_calls(), 1, "{trigger:?}");
        }
        Ok(())
    }
}

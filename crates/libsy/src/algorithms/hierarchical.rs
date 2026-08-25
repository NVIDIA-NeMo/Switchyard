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
use super::llm_class::{LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig};
use super::stage::{StageRouterConfig, build_stage_route};
use super::util::affinity::{ClassifyTrigger, evict_if_full, has_new_user_turn, retention_key};
use super::util::stage::{StageTargets, Tier, set_fall_open};
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::classifier::Classifier;
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{ModelId, Request};

const HIERARCHICAL: &str = "hierarchical";

/// Sets the stage router's fall-open tier from a judge verdict.
///
/// Retains the tier per routing identity, so it survives requests that carry no
/// session ID when `message_hash_fallback` is on. The retained tier is replayed
/// into state on every request so the cascade below reads it.
struct TierSetter {
    judge: Arc<dyn Classifier<State>>,
    targets: StageTargets,
    trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    tiers: Mutex<HashMap<RoutingIdentity, Tier>>,
}

impl TierSetter {
    /// Two requests for one identity can both pass this and both judge, since a
    /// judge call sits between here and [`retain`](Self::retain). The later wins.
    fn is_due(&self, identity: Option<&RoutingIdentity>, request: &Request) -> bool {
        match self.trigger {
            ClassifyTrigger::UserTurn => has_new_user_turn(&request.llm_request.messages),
            // Unkeyed requests cannot be told apart, so every one is a new session.
            ClassifyTrigger::NewSession => {
                identity.is_none_or(|identity| !self.tiers.lock().contains_key(identity))
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
impl Processor<State> for TierSetter {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
        let Event::Request { request, driver } = event else {
            return Ok(());
        };
        let identity = retention_key(request, self.message_hash_fallback);
        if self.is_due(identity.as_ref(), request) {
            let (classification, _) = self.judge.score(state, request, driver).await?;
            if let Some(winner) = classification.argmax(false)?
                && let Some(tier) = self.targets.tier_for(&winner.target)
            {
                set_fall_open(state, tier);
                if let Some(identity) = identity {
                    self.retain(identity, tier);
                }
                return Ok(());
            }
        }
        // Either not this request's turn or the judge had no verdict, so the last
        // tier stands rather than dropping back to the picker default.
        if let Some(tier) = identity.and_then(|identity| self.tiers.lock().get(&identity).copied())
        {
            set_fall_open(state, tier);
        }
        Ok(())
    }
}

/// A judge stacked over a stage router.
pub struct HierarchicalRouterConfig {
    /// Target the judge is called through. Not a routing destination.
    pub judge_target: ModelId,
    /// Judge settings, including how often `classify_trigger` runs it.
    pub judge: TaskClassifierConfig,
    /// Serves the turns, with the tier the judge picked as its fall-open default.
    pub stage: StageRouterConfig,
}

/// Runs a stage router with a tier the judge picks.
pub struct HierarchicalRouter {
    route: FallThrough<State>,
}

impl HierarchicalRouter {
    /// Stacks the judge over a stage router across the same tier pair.
    ///
    /// Errors on a configuration either algorithm rejects and on `every_request`.
    ///
    /// A stage router carrying its own judge is allowed, but that judge sits ahead
    /// of the fall-open tier and so answers most of the turns this one set a tier for.
    pub fn new(
        capable: ModelId,
        efficient: ModelId,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        if config.judge.classify_trigger == ClassifyTrigger::EveryRequest {
            return Err(LibsyError::AlgorithmError {
                message: "hierarchical: classify_trigger must be user_turn or new_session"
                    .to_string(),
            });
        }
        let trigger = config.judge.classify_trigger;
        let message_hash_fallback = config.judge.message_hash_fallback;
        // Only the judge's Classifier face is used, so its own affinity never runs.
        // Leaving these set would apply the standalone route's pairing rules to a
        // trigger this router implements itself.
        let judge_config = TaskClassifierConfig {
            classify_trigger: ClassifyTrigger::EveryRequest,
            message_hash_fallback: false,
            ..config.judge
        };
        let judge = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: config.judge_target,
            efficient_target: efficient.clone(),
            capable_target: capable.clone(),
            config: judge_config,
        })?;
        let setter = TierSetter {
            judge: Arc::new(judge),
            targets: StageTargets::new(capable.clone(), efficient.clone()),
            trigger,
            message_hash_fallback,
            tiers: Mutex::new(HashMap::new()),
        };
        let route = build_stage_route(capable, efficient, config.stage)?
            .with_name(HIERARCHICAL)
            .with_processor(Arc::new(setter));
        Ok(Self { route })
    }
}

#[async_trait]
impl Algorithm for HierarchicalRouter {
    fn name(&self) -> &str {
        HIERARCHICAL
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
    use std::sync::Arc;

    use switchyard_protocol::{Message, Role};

    use super::*;
    use crate::algorithms::util::stage::PickerMode;
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::testing::test_drive;

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

    fn hash_keyed_router() -> Result<Arc<HierarchicalRouter>> {
        Ok(Arc::new(HierarchicalRouter::new(
            ModelId::from("strong"),
            ModelId::from("weak"),
            HierarchicalRouterConfig {
                judge_target: ModelId::from(JUDGE),
                judge: TaskClassifierConfig {
                    base_threshold: 0.5,
                    classify_trigger: ClassifyTrigger::UserTurn,
                    message_hash_fallback: true,
                    ..Default::default()
                },
                stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
            },
        )?))
    }

    fn router() -> Result<Arc<HierarchicalRouter>> {
        Ok(Arc::new(HierarchicalRouter::new(
            ModelId::from("strong"),
            ModelId::from("weak"),
            HierarchicalRouterConfig {
                judge_target: ModelId::from(JUDGE),
                judge: TaskClassifierConfig {
                    base_threshold: 0.5,
                    classify_trigger: ClassifyTrigger::UserTurn,
                    ..Default::default()
                },
                stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
            },
        )?))
    }

    #[test]
    fn rejects_every_request_as_a_trigger() {
        let config = HierarchicalRouterConfig {
            judge_target: ModelId::from(JUDGE),
            judge: TaskClassifierConfig::default(),
            stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        };
        assert!(matches!(
            HierarchicalRouter::new(ModelId::from("strong"), ModelId::from("weak"), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[tokio::test]
    async fn a_session_without_an_id_keys_on_the_message_hash() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.1;
        let router = hash_keyed_router()?;

        test_drive(
            router.clone(),
            unkeyed(user_turn_request()),
            recorder.serve(),
        )
        .await?;
        test_drive(
            router.clone(),
            unkeyed(turn_request(false)),
            recorder.serve(),
        )
        .await?;

        assert_eq!(
            recorder.judge_calls(),
            1,
            "new_session judges once without a session id"
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

        test_drive(router.clone(), user_turn_request(), recorder.serve()).await?;
        test_drive(router.clone(), turn_request(false), recorder.serve()).await?;

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
}

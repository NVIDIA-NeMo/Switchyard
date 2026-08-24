// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing that stacks one algorithm on top of another.
//!
//! The algorithm above runs as a [`Prelude`]: it sets configuration the one below
//! reads, and decides nothing itself.

use std::sync::Arc;

use async_trait::async_trait;

use super::fall_through::FallThrough;
use super::llm_class::{LlmClassifierConfig, LlmTaskClassifier, TaskClassifierConfig};
use super::stage::{StageRouterConfig, build_stage_route};
use super::util::affinity::{ClassifyTrigger, has_new_user_turn};
use super::util::stage::{StageTargets, fall_open_tier, set_fall_open};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::Classifier;
use crate::core::prelude::Prelude;
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{ModelId, Request};

const HIERARCHICAL: &str = "hierarchical";

/// Sets the stage router's fall-open tier from a judge verdict at each user turn.
struct TierPicker {
    judge: Arc<dyn Classifier<State>>,
    targets: StageTargets,
    trigger: ClassifyTrigger,
}

impl TierPicker {
    fn due(&self, state: &State, request: &Request) -> bool {
        match self.trigger {
            ClassifyTrigger::EveryRequest => true,
            ClassifyTrigger::UserTurn => has_new_user_turn(&request.llm_request.messages),
            // The tier it set is the record that it has already run.
            ClassifyTrigger::NewSession => fall_open_tier(state).is_none(),
        }
    }
}

#[async_trait]
impl Prelude<State> for TierPicker {
    async fn run(&self, state: &mut State, request: &mut Request, driver: &Driver) -> Result<()> {
        if !self.due(state, request) {
            return Ok(());
        }
        let (classification, _) = self.judge.score(state, request, Some(driver)).await?;
        if let Some(winner) = classification.argmax(false)?
            && let Some(tier) = self.targets.tier_for(&winner.target)
        {
            set_fall_open(state, tier);
        }
        Ok(())
    }
}

/// The judge that picks a user turn's tier.
pub struct TierClassifier {
    /// Target the judge is called through. Not a routing destination.
    pub judge_target: ModelId,
    /// Judge configuration, including how often `classify_trigger` runs it.
    pub config: TaskClassifierConfig,
}

/// An algorithm stacked on top of a stage router.
pub struct HierarchicalRouterConfig {
    /// Runs at each user turn and sets the tier below it.
    pub classifier: TierClassifier,
    /// Serves every request.
    pub stage: StageRouterConfig,
}

/// Runs a stage router with a tier the classifier picks once per user turn.
pub struct HierarchicalRouter {
    route: FallThrough<State>,
}

impl HierarchicalRouter {
    /// Stacks the classifier over a stage router across the same tier pair.
    ///
    /// Errors on a stage or judge configuration either algorithm rejects, and on a
    /// stage router carrying its own judge, which would decide the turns this
    /// classifier set a tier for.
    pub fn new(
        capable: ModelId,
        efficient: ModelId,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        if config.stage.llm_fallback.is_some() {
            return Err(LibsyError::AlgorithmError {
                message: "hierarchical: the stage router cannot also carry a judge".to_string(),
            });
        }
        let trigger = config.classifier.config.classify_trigger;
        let judge = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: config.classifier.judge_target,
            efficient_target: efficient.clone(),
            capable_target: capable.clone(),
            config: config.classifier.config,
        })?;
        let picker = TierPicker {
            judge: Arc::new(judge),
            targets: StageTargets::new(capable.clone(), efficient.clone()),
            trigger,
        };
        let route = build_stage_route(capable, efficient, config.stage)?
            .with_name(HIERARCHICAL)
            .with_prelude(Arc::new(picker));
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

    use parking_lot::Mutex;
    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, LlmRequest, Message, Metadata, Role, ToolCall, ToolResult, WireFormat,
    };

    use super::*;
    use crate::algorithms::stage::LlmFallback;
    use crate::algorithms::util::stage::PickerMode;
    use crate::core::testing::{Serve, reply, test_drive};

    const JUDGE: &str = "judge";

    #[derive(Default)]
    struct Recorder {
        targets: Mutex<Vec<String>>,
        judge_p_solve: Mutex<f64>,
    }

    impl Recorder {
        fn routed(&self) -> Vec<String> {
            self.targets
                .lock()
                .iter()
                .filter(|target| *target != JUDGE)
                .cloned()
                .collect()
        }

        fn judge_calls(&self) -> usize {
            self.targets
                .lock()
                .iter()
                .filter(|target| *target == JUDGE)
                .count()
        }

        fn serve(self: &Arc<Self>) -> impl Serve {
            let recorder = Arc::clone(self);
            move |target: ModelId, _request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    let target = target.to_string();
                    recorder.targets.lock().push(target.clone());
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
            raw_request: None,
            metadata: Some(Metadata {
                wire_format: Some(WireFormat::OpenAiChat),
                session_id: Some("session-1".to_string()),
                ..Default::default()
            }),
        }
    }

    fn user_turn_request() -> Request {
        let mut request = turn_request(false);
        request
            .llm_request
            .messages
            .push(Message::text(Role::User, "now rewrite the parser"));
        request
    }

    fn router() -> Result<Arc<HierarchicalRouter>> {
        Ok(Arc::new(HierarchicalRouter::new(
            ModelId::from("strong"),
            ModelId::from("weak"),
            HierarchicalRouterConfig {
                classifier: TierClassifier {
                    judge_target: ModelId::from(JUDGE),
                    config: TaskClassifierConfig {
                        base_threshold: 0.5,
                        classify_trigger: ClassifyTrigger::UserTurn,
                        ..Default::default()
                    },
                },
                stage: StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
            },
        )?))
    }

    #[test]
    fn rejects_a_stage_router_that_carries_its_own_judge() {
        let mut stage = StageRouterConfig::new(PickerMode::EfficientFirst, 0.5);
        stage.llm_fallback = Some(LlmFallback {
            judge_target: ModelId::from(JUDGE),
            config: TaskClassifierConfig::default(),
        });
        let config = HierarchicalRouterConfig {
            classifier: TierClassifier {
                judge_target: ModelId::from(JUDGE),
                config: TaskClassifierConfig::default(),
            },
            stage,
        };
        assert!(matches!(
            HierarchicalRouter::new(ModelId::from("strong"), ModelId::from("weak"), config),
            Err(LibsyError::AlgorithmError { .. })
        ));
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
            routed[0], "strong",
            "a quiet turn falls open to the verdict"
        );
        assert_eq!(
            routed[1], "strong",
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

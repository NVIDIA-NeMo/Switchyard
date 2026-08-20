// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct parent routing with an optional delegated-work classifier cascade.

use std::sync::Arc;

use switchyard_protocol::{ModelId, Request};

use super::fall_through::{DefaultTarget, FallThrough};
use super::util::affinity::AffinityRouter;
use super::util::subagent::{SubagentGate, SubagentOverride};
use super::util::turn_pin::ClassifyTrigger;
use crate::core::algorithm::{self, Algorithm, Driver};
use crate::core::classifier::Classifier;
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};

/// Routes parent traffic directly and optionally classifies delegated sub-agent work.
pub struct Passthrough {
    parent_target: ModelId,
    route: FallThrough<State>,
}

/// Runtime components for classifying and retaining delegated sub-agent work.
pub struct PassthroughSubagentConfig {
    /// Targets the delegated-work classifier may select.
    pub targets: Vec<ModelId>,
    /// Classifier invoked for the first request from each identified child.
    pub classifier: Arc<dyn Classifier<State>>,
    /// Child target used when `classifier` abstains.
    pub default_target: ModelId,
    /// Controls whether each child is classified once or on every request.
    pub classify_trigger: ClassifyTrigger,
    /// Unsupported for child routing because child identity must come from harness metadata.
    pub message_hash_fallback: bool,
}

/// Complete construction settings for [`Passthrough`].
pub struct PassthroughConfig {
    /// Target used for parent and harness-maintenance traffic.
    pub parent_target: ModelId,
    /// Optional delegated-work decision gate.
    pub subagent: Option<PassthroughSubagentConfig>,
}

impl Passthrough {
    /// Creates direct parent routing, optionally with a decision gate for sub-agents.
    ///
    /// When configured, the sub-agent classifier runs once per identified child. Its first
    /// decision is retained by `session + agent`; an abstaining classifier uses the child
    /// default. Root and harness-maintenance traffic continue to the parent target.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured child default is not a child target.
    pub fn new(config: PassthroughConfig) -> Result<Self> {
        let parent_target = config.parent_target;
        let route = match config.subagent {
            None => FallThrough::new_with_state(vec![parent_target.clone()])
                .with_name("passthrough")
                .with_classifier(Arc::new(DefaultTarget::new(parent_target.clone()))),
            Some(subagent) => {
                algorithm::ensure_model_is_target(&subagent.targets, &subagent.default_target)?;
                if subagent.message_hash_fallback {
                    return Err(LibsyError::AlgorithmError {
                        message: "sub-agent routing cannot use message_hash_fallback".to_string(),
                    });
                }
                let mut targets = subagent.targets;
                if !targets.contains(&parent_target) {
                    targets.push(parent_target.clone());
                }
                let mut route = FallThrough::new_with_state(targets).with_name("passthrough");
                match subagent.classify_trigger {
                    ClassifyTrigger::EveryRequest => {}
                    ClassifyTrigger::NewSession => {
                        let affinity = Arc::new(AffinityRouter::for_subagents());
                        route = route
                            .with_processor(affinity.clone())
                            .with_classifier(affinity);
                    }
                    ClassifyTrigger::UserTurn => {
                        return Err(LibsyError::AlgorithmError {
                            message: "sub-agent routing cannot use classify_trigger = user_turn"
                                .to_string(),
                        });
                    }
                }
                route
                    .with_classifier(Arc::new(SubagentGate::new(subagent.classifier)))
                    .with_classifier(Arc::new(SubagentOverride::new(subagent.default_target)))
                    .with_classifier(Arc::new(DefaultTarget::new(parent_target.clone())))
            }
        };

        Ok(Self {
            parent_target,
            route,
        })
    }
}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let mut outcome = self.route.execute(driver, request).await?;
        // Parent traffic preserves passthrough's no-fallback contract. Child traffic may
        // fall back only within the child target set, never into the parent route.
        if outcome.selected_model_id == self.parent_target {
            outcome.fallback_models.clear();
        } else {
            outcome
                .fallback_models
                .retain(|target| *target != self.parent_target);
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;

    use super::{Passthrough, PassthroughConfig, PassthroughSubagentConfig};
    use crate::core::algorithm::Algorithm;
    use crate::core::classifier::{Classification, Classifier, Score};
    use crate::core::testing::{echo, reply, test_drive};
    use crate::{
        ClassifyTrigger, CustomClassifierConfig, CustomClassifierPolicy, Driver,
        LlmClassifierConfig, LlmTaskClassifier, State,
    };
    use switchyard_protocol::{
        ContentBlock, InstructionBlock, Message, Metadata, ModelId, Request, Response, Role,
        completion_text, text_request,
    };

    struct AlternatingClassifier {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Classifier<State> for AlternatingClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> crate::Result<(Classification, Option<Response>)> {
            let target = if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                "worker"
            } else {
                "reviewer"
            };
            Ok((
                Classification::Scores(vec![Score {
                    confidence: 1.0,
                    target: ModelId::from(target),
                }]),
                None,
            ))
        }
    }

    struct AbstainingClassifier;

    #[async_trait]
    impl Classifier<State> for AbstainingClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> crate::Result<(Classification, Option<Response>)> {
            Ok((Classification::Scores(Vec::new()), None))
        }
    }

    fn request(metadata: Option<Metadata>) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata,
        }
    }

    fn child(agent_id: &str) -> Request {
        request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some(agent_id.to_string()),
            is_subagent: true,
            is_delegated_work: true,
            ..Metadata::default()
        }))
    }

    fn configured(classifier: Arc<dyn Classifier<State>>) -> crate::Result<Arc<Passthrough>> {
        Ok(Arc::new(Passthrough::new(PassthroughConfig {
            parent_target: ModelId::from("parent"),
            subagent: Some(PassthroughSubagentConfig {
                targets: vec![ModelId::from("worker"), ModelId::from("reviewer")],
                classifier,
                default_target: ModelId::from("worker"),
                classify_trigger: ClassifyTrigger::NewSession,
                message_hash_fallback: false,
            }),
        })?))
    }

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(PassthroughConfig {
            parent_target: ModelId::from(MODEL_ID),
            subagent: None,
        })?);
        let (selected_model, response) = test_drive(algorithm, request, echo()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            MODEL_ID
        );
        assert_eq!(selected_model, MODEL_ID);
        Ok(())
    }

    #[tokio::test]
    async fn parent_traffic_keeps_the_passthrough_target() -> crate::Result<()> {
        let router = configured(Arc::new(AlternatingClassifier {
            calls: AtomicUsize::new(0),
        }))?;

        let (selected, _) = test_drive(router, request(None), echo()).await?;

        assert_eq!(selected, "parent");
        Ok(())
    }

    #[tokio::test]
    async fn classifier_decision_is_retained_per_child() -> crate::Result<()> {
        let classifier = Arc::new(AlternatingClassifier {
            calls: AtomicUsize::new(0),
        });
        let router = configured(classifier.clone())?;

        let (first, _) = test_drive(router.clone(), child("child-1"), echo()).await?;
        let (same_child, _) = test_drive(router.clone(), child("child-1"), echo()).await?;
        let (sibling, _) = test_drive(router, child("child-2"), echo()).await?;

        assert_eq!(first, "worker");
        assert_eq!(same_child, "worker");
        assert_eq!(sibling, "reviewer");
        assert_eq!(classifier.calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[tokio::test]
    async fn an_abstaining_child_classifier_uses_the_child_default() -> crate::Result<()> {
        let router = configured(Arc::new(AbstainingClassifier))?;

        let (selected, _) = test_drive(router, child("child-1"), echo()).await?;

        assert_eq!(selected, "worker");
        Ok(())
    }

    #[tokio::test]
    async fn maintenance_traffic_ignores_an_existing_child_pin() -> crate::Result<()> {
        let router = configured(Arc::new(AlternatingClassifier {
            calls: AtomicUsize::new(0),
        }))?;
        test_drive(router.clone(), child("child-1"), echo()).await?;
        let maintenance = request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("child-1".to_string()),
            is_subagent: true,
            is_delegated_work: false,
            ..Metadata::default()
        }));

        let (selected, _) = test_drive(router, maintenance, echo()).await?;

        assert_eq!(selected, "parent");
        Ok(())
    }

    #[tokio::test]
    async fn custom_classifier_receives_only_the_delegated_prompt() -> crate::Result<()> {
        let classifier = LlmTaskClassifier::new(LlmClassifierConfig::Custom {
            judge_target: ModelId::from("judge"),
            targets: vec![
                ("worker".to_string(), ModelId::from("worker")),
                ("reviewer".to_string(), ModelId::from("reviewer")),
            ],
            default_target: "worker".to_string(),
            config: CustomClassifierConfig::new(
                "classify the delegated task",
                json!({
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "enum": ["worker", "reviewer"]}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }),
                CustomClassifierPolicy::target_selector("/target"),
            ),
        })?;
        let router = configured(Arc::new(classifier))?;
        let mut request = child("child-1");
        request.llm_request.instructions = vec![InstructionBlock {
            role: Role::System,
            content: Message::text(Role::System, "child system instructions").content,
        }];
        request.llm_request.messages = vec![
            Message::text(Role::User, "harness context"),
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "<system-reminder>tool context</system-reminder>".to_string(),
                    },
                    ContentBlock::Text {
                        text: "review this parser".to_string(),
                    },
                ],
            },
        ];
        let calls = Arc::new(Mutex::new(Vec::new()));
        let served_calls = calls.clone();

        let (selected, _) = test_drive(router, request, move |target, request| {
            let calls = served_calls.clone();
            async move {
                let completion = if target == "judge" {
                    r#"{"target":"reviewer"}"#
                } else {
                    "child answer"
                };
                calls.lock().push((target, request));
                Ok(reply(completion))
            }
        })
        .await?;

        assert_eq!(selected, "reviewer");
        let calls = calls.lock();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "judge");
        assert_eq!(
            calls[0].1.llm_request.instructions[0].content,
            Message::text(Role::System, "classify the delegated task").content
        );
        assert_eq!(
            calls[0].1.llm_request.messages,
            vec![Message::text(Role::User, "review this parser")]
        );
        assert_eq!(calls[1].0, "reviewer");
        assert_eq!(calls[1].1.llm_request.instructions.len(), 1);
        assert_eq!(calls[1].1.llm_request.messages.len(), 2);
        Ok(())
    }
}

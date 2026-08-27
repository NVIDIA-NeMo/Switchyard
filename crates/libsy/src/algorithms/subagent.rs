// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delegated sub-agent routing around an arbitrary parent algorithm.

use std::sync::Arc;

use switchyard_protocol::{Metadata, ModelId, Request};

use super::fall_through::{DefaultTarget, FallThrough};
use super::util::affinity::{AffinityRouter, ClassifyTrigger};
use super::util::subagent::SubagentGate;
use crate::core::algorithm::{self, Algorithm, Driver};
use crate::core::classifier::Classifier;
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};

/// Runtime components for delegated sub-agent routing.
pub struct SubagentRouterConfig {
    /// Targets the delegated-work classifier may select.
    pub targets: Vec<ModelId>,
    /// Classifier invoked for delegated work according to `classify_trigger`.
    pub classifier: Arc<dyn Classifier<State>>,
    /// Child target used when `classifier` abstains.
    pub default_target: ModelId,
    /// Controls whether each child is classified once or on every request.
    pub classify_trigger: ClassifyTrigger,
    /// Unsupported for child routing because child identity must come from harness metadata.
    pub message_hash_fallback: bool,
}

impl SubagentRouterConfig {
    /// Routes delegated work directly to one fixed target.
    pub fn fixed_target(target: impl Into<ModelId>) -> Self {
        let target = target.into();
        Self {
            targets: vec![target.clone()],
            classifier: Arc::new(DefaultTarget::new(target.clone())),
            default_target: target,
            classify_trigger: ClassifyTrigger::EveryRequest,
            message_hash_fallback: false,
        }
    }
}

/// Routes delegated work independently while preserving the parent algorithm for other traffic.
pub struct SubagentRouter {
    parent: Arc<dyn Algorithm>,
    subagent: FallThrough<State>,
}

impl SubagentRouter {
    /// Wraps `parent` with the configured delegated-work route.
    ///
    /// # Errors
    ///
    /// Returns an error when the child default is not a child target or when the affinity
    /// settings cannot identify delegated children safely.
    pub fn new(parent: Arc<dyn Algorithm>, config: SubagentRouterConfig) -> Result<Self> {
        algorithm::ensure_model_is_target(&config.targets, &config.default_target)?;
        if config.message_hash_fallback {
            return Err(LibsyError::AlgorithmError {
                message: "sub-agent routing cannot use message_hash_fallback".to_string(),
            });
        }

        let mut subagent = match config.classify_trigger {
            ClassifyTrigger::EveryRequest => {
                FallThrough::new_with_state(config.targets).with_name("subagent")
            }
            ClassifyTrigger::NewSession => {
                let affinity = Arc::new(AffinityRouter::for_subagents());
                FallThrough::new_with_state(config.targets)
                    .with_name("subagent")
                    .with_processor(affinity.clone())
                    .with_classifier(affinity)
            }
            ClassifyTrigger::UserTurn => {
                return Err(LibsyError::AlgorithmError {
                    message: "sub-agent routing cannot use classify_trigger = user_turn"
                        .to_string(),
                });
            }
        };
        subagent = subagent
            .with_classifier(Arc::new(SubagentGate::new(config.classifier)))
            .with_classifier(Arc::new(DefaultTarget::new(config.default_target)));

        Ok(Self { parent, subagent })
    }
}

#[async_trait::async_trait]
impl Algorithm for SubagentRouter {
    fn name(&self) -> &str {
        self.parent.name()
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        if request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work)
        {
            self.subagent.execute(driver, request).await
        } else {
            self.parent.clone().route(driver, request).await
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, InstructionBlock, Message, Metadata, ModelId, Request, Response, Role,
        text_request,
    };

    use super::{SubagentRouter, SubagentRouterConfig};
    use crate::algorithms::passthrough::Passthrough;
    use crate::core::algorithm::Algorithm;
    use crate::core::classifier::{Classification, Classifier, Score};
    use crate::core::testing::{echo, reply, test_drive};
    use crate::{
        ClassifyTrigger, CustomClassifierConfig, CustomClassifierPolicy, Driver,
        LlmClassifierConfig, LlmTaskClassifier, State,
    };

    struct ScriptedClassifier {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Classifier<State> for ScriptedClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> crate::Result<(Classification, Option<Response>)> {
            let scores = match self.calls.fetch_add(1, Ordering::Relaxed) {
                0 => vec![Score {
                    confidence: 1.0,
                    target: ModelId::from("worker"),
                }],
                1 => vec![Score {
                    confidence: 1.0,
                    target: ModelId::from("reviewer"),
                }],
                _ => Vec::new(),
            };
            Ok((Classification::Scores(scores), None))
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

    fn parent() -> Arc<dyn Algorithm> {
        Arc::new(Passthrough::new("parent"))
    }

    fn configured(classifier: Arc<dyn Classifier<State>>) -> crate::Result<Arc<SubagentRouter>> {
        Ok(Arc::new(SubagentRouter::new(
            parent(),
            SubagentRouterConfig {
                targets: vec![ModelId::from("worker"), ModelId::from("reviewer")],
                classifier,
                default_target: ModelId::from("worker"),
                classify_trigger: ClassifyTrigger::NewSession,
                message_hash_fallback: false,
            },
        )?))
    }

    #[tokio::test]
    async fn routes_parent_and_children_with_affinity_and_default() -> crate::Result<()> {
        let classifier = Arc::new(ScriptedClassifier {
            calls: AtomicUsize::new(0),
        });
        let router = configured(classifier.clone())?;

        let (parent, _) = test_drive(router.clone(), request(None), echo()).await?;
        let (first, _) = test_drive(router.clone(), child("child-1"), echo()).await?;
        let (same_child, _) = test_drive(router.clone(), child("child-1"), echo()).await?;
        let (sibling, _) = test_drive(router.clone(), child("child-2"), echo()).await?;
        let (defaulted, _) = test_drive(router.clone(), child("child-3"), echo()).await?;
        let maintenance = request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("child-1".to_string()),
            is_subagent: true,
            is_delegated_work: false,
            ..Metadata::default()
        }));
        let (maintenance, _) = test_drive(router, maintenance, echo()).await?;

        assert_eq!(parent, "parent");
        assert_eq!(first, "worker");
        assert_eq!(same_child, "worker");
        assert_eq!(sibling, "reviewer");
        assert_eq!(defaulted, "worker");
        assert_eq!(maintenance, "parent");
        assert_eq!(classifier.calls.load(Ordering::Relaxed), 3);
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

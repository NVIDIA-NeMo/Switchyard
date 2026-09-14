// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delegated sub-agent routing around an arbitrary parent algorithm.

use std::sync::Arc;

use switchyard_protocol::{Category, Metadata, Request};

use super::fall_through::FallThrough;
use super::util::affinity::{AffinityRouter, ClassifyTrigger};
use super::util::subagent::SubagentGate;
use crate::algorithms::llm_class::DefaultCategoryClassifier;
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::Classifier;
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};

/// Runtime components for delegated sub-agent routing.
pub struct SubagentRouterConfig {
    /// Classifier invoked for delegated work according to `classify_trigger`.
    pub classifier: Arc<dyn Classifier<State>>,
    /// Child model category used when `classifier` abstains.
    pub default_target: Category,
    /// Controls whether each child is classified once or on every request.
    pub classify_trigger: ClassifyTrigger,
    /// Unsupported for child routing because child identity must come from harness metadata.
    pub message_hash_fallback: bool,
}

impl SubagentRouterConfig {
    /// Routes all delegated work to the first model in the sub-agent `Any` category.
    pub fn fixed_target() -> Self {
        Self {
            classifier: Arc::new(DefaultCategoryClassifier(Category::Any)),
            default_target: Category::Any,
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
    /// Returns an error when the affinity settings cannot identify delegated children safely.
    pub fn new(parent: Arc<dyn Algorithm>, config: SubagentRouterConfig) -> Result<Self> {
        if config.message_hash_fallback {
            return Err(LibsyError::AlgorithmError {
                message: "sub-agent routing cannot use message_hash_fallback".to_string(),
            });
        }

        let mut subagent = match config.classify_trigger {
            ClassifyTrigger::EveryRequest => FallThrough::new_with_state().with_name("subagent"),
            ClassifyTrigger::NewSession => {
                let affinity = Arc::new(AffinityRouter::for_subagents());
                FallThrough::new_with_state()
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
            .with_classifier(Arc::new(DefaultCategoryClassifier(config.default_target)));

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
            // Delegated work routes over the sub-agent's own models, never the parent's.
            self.subagent.execute(driver.for_subagent()?, request).await
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
    use switchyard_protocol::{Category, Metadata, ModelId, Request, Response, text_request};

    use super::{SubagentRouter, SubagentRouterConfig};
    use crate::algorithms::passthrough::Passthrough;
    use crate::core::classifier::{Classification, Classifier, Score};
    use crate::core::testing::{echo, test_drive_with_models};
    use crate::{ClassifyTrigger, Driver, RuntimeModels, State};

    struct ScriptedClassifier {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Classifier<State> for ScriptedClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            driver: &Driver,
        ) -> crate::Result<(Classification, Option<Response>)> {
            let category = match self.calls.fetch_add(1, Ordering::Relaxed) {
                0 => Some(Category::Capable),
                1 => Some(Category::Efficient),
                _ => None,
            };
            let scores = match category {
                Some(category) => vec![Score {
                    confidence: 1.0,
                    target: driver.first_model_for(&category)?.clone(),
                    category: Some(category),
                }],
                None => Vec::new(),
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

    fn configured(classifier: Arc<dyn Classifier<State>>) -> crate::Result<Arc<SubagentRouter>> {
        Ok(Arc::new(SubagentRouter::new(
            Arc::new(Passthrough),
            SubagentRouterConfig {
                classifier,
                default_target: Category::Capable,
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

        // The parent and its children route over separate model groups.
        let models = RuntimeModels::new([(Category::Any, vec![ModelId::from("parent")])].into())
            .with_subagent(
                [
                    (
                        Category::Any,
                        vec![ModelId::from("worker"), ModelId::from("reviewer")],
                    ),
                    (Category::Capable, vec![ModelId::from("worker")]),
                    (Category::Efficient, vec![ModelId::from("reviewer")]),
                ]
                .into(),
            );
        let (selected_parent, _) =
            test_drive_with_models(router.clone(), request(None), models.clone(), echo()).await?;
        let (first, _) =
            test_drive_with_models(router.clone(), child("child-1"), models.clone(), echo())
                .await?;
        let (same_child, _) =
            test_drive_with_models(router.clone(), child("child-1"), models.clone(), echo())
                .await?;
        let (sibling, _) =
            test_drive_with_models(router.clone(), child("child-2"), models.clone(), echo())
                .await?;
        let (defaulted, _) =
            test_drive_with_models(router.clone(), child("child-3"), models.clone(), echo())
                .await?;
        let maintenance = request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("child-1".to_string()),
            is_subagent: true,
            is_delegated_work: false,
            ..Metadata::default()
        }));
        let (maintenance, _) =
            test_drive_with_models(router, maintenance, models.clone(), echo()).await?;

        assert_eq!(selected_parent, "parent");
        assert_eq!(first, "worker");
        assert_eq!(same_child, "worker");
        assert_eq!(sibling, "reviewer");
        assert_eq!(defaulted, "worker");
        assert_eq!(maintenance, "parent");
        assert_eq!(classifier.calls.load(Ordering::Relaxed), 3);

        let fixed = Arc::new(SubagentRouter::new(
            Arc::new(Passthrough),
            SubagentRouterConfig::fixed_target(),
        )?);
        let (fixed, _) = test_drive_with_models(fixed, child("fixed"), models, echo()).await?;
        assert_eq!(fixed, "worker");
        Ok(())
    }
}

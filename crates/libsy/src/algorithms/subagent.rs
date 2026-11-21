// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delegated sub-agent routing around an arbitrary parent algorithm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use switchyard_protocol::{Category, Metadata, Request};

use super::fall_through::FallThrough;
use super::util::affinity::{AffinityRouter, ClassifyTrigger};
use super::util::subagent::SubagentGate;
use crate::algorithms::llm_class::DefaultCategoryClassifier;
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::Classifier;
use crate::{LibsyError, Result, RoutingOutcome};

/// Runtime components for delegated sub-agent routing.
pub struct SubagentRouterConfig {
    /// Classifier invoked for delegated work according to `classify_trigger`.
    pub classifier: Arc<dyn Classifier>,
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
    subagent: FallThrough,
    /// Whether the one-time warning about a harness without sub-agent identity has been emitted.
    identity_warning_emitted: AtomicBool,
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

        Ok(Self {
            parent,
            subagent,
            identity_warning_emitted: AtomicBool::new(false),
        })
    }

    /// Warns once when a harness that cannot identify its children reaches this route.
    ///
    /// Such a build sends only the session id, so its sub-agent requests look like the
    /// parent's and route through the parent algorithm. Without this the route reports
    /// itself as configured while never routing any delegated work.
    fn warn_if_subagent_identity_unsupported(&self, request: &Request) {
        let unsupported = request
            .metadata
            .as_ref()
            .is_some_and(|metadata| metadata.subagent_identity_unsupported);
        if !unsupported || self.identity_warning_emitted.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::warn!(
            target: "libsy",
            "this route has sub-agent routing but the calling harness does not send \
             sub-agent identity, so its delegated requests route through the parent route; \
             a harness upgrade may be required"
        );
    }
}

#[async_trait::async_trait]
impl Algorithm for SubagentRouter {
    fn name(&self) -> &str {
        self.parent.name()
    }

    fn supports_host_state(&self) -> bool {
        self.parent.supports_host_state() && !self.subagent.tracks_answers()
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        if request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work)
        {
            // Delegated work routes over the sub-agent's own models, never the parent's.
            let subagent_driver = driver.for_subagent()?;
            if subagent_driver.is_host_managed() {
                let full_state = subagent_driver.take_state();
                let Some(agent_id) = request
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.agent_id.as_deref())
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                else {
                    subagent_driver.set_state(crate::State::default());
                    let outcome = self
                        .subagent
                        .execute(subagent_driver.clone(), request)
                        .await?;
                    subagent_driver.set_state(full_state);
                    return Ok(outcome);
                };
                let prefix = format!("subagent.{agent_id}.");
                let child_state = crate::State {
                    extra: full_state
                        .extra
                        .iter()
                        .filter_map(|(key, value)| {
                            key.strip_prefix(&prefix)
                                .map(|key| (key.to_string(), value.clone()))
                        })
                        .collect(),
                    ..crate::State::default()
                };
                subagent_driver.set_state(child_state);
                let outcome = self
                    .subagent
                    .execute(subagent_driver.clone(), request)
                    .await?;
                let child_after = subagent_driver.take_state();
                let mut merged = full_state;
                merged.extra.retain(|key, _| !key.starts_with(&prefix));
                merged.extra.extend(
                    child_after
                        .extra
                        .into_iter()
                        .map(|(key, value)| (format!("{prefix}{key}"), value)),
                );
                subagent_driver.set_state(merged);
                Ok(outcome)
            } else {
                self.subagent.execute(subagent_driver, request).await
            }
        } else {
            self.warn_if_subagent_identity_unsupported(&request);
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
    impl Classifier for ScriptedClassifier {
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

    fn configured(classifier: Arc<dyn Classifier>) -> crate::Result<Arc<SubagentRouter>> {
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

    fn without_subagent_identity() -> Request {
        request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            subagent_identity_unsupported: true,
            ..Metadata::default()
        }))
    }

    #[tokio::test]
    async fn harness_without_subagent_identity_warns_once_and_routes_through_the_parent()
    -> crate::Result<()> {
        // Without child identity the request is indistinguishable from the parent's, so it
        // keeps routing through the parent algorithm; the warning fires once per route.
        let router = configured(Arc::new(ScriptedClassifier {
            calls: AtomicUsize::new(0),
        }))?;
        let models = RuntimeModels::new([(Category::Any, vec![ModelId::from("parent")])].into())
            .with_subagent([(Category::Any, vec![ModelId::from("worker")])].into());
        for _ in 0..2 {
            let request = without_subagent_identity();
            let (selected, _) =
                test_drive_with_models(router.clone(), request, models.clone(), echo()).await?;
            assert_eq!(selected, "parent");
        }
        assert!(router.identity_warning_emitted.load(Ordering::Relaxed));
        Ok(())
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
    #[tokio::test]
    async fn caller_state_preserves_subagent_affinity_and_clears_on_final() -> crate::Result<()> {
        use crate::{Algorithm, SessionKey};
        use std::collections::HashMap;
        use switchyard_protocol::{ContentBlock, Message, Role};

        struct AnswerTrackingChild;

        #[async_trait]
        impl Classifier for AnswerTrackingChild {
            fn tracks_answers(&self) -> bool {
                true
            }

            async fn score(
                &self,
                _state: &mut State,
                _request: &mut Request,
                _driver: &Driver,
            ) -> crate::Result<(Classification, Option<Response>)> {
                Ok((Classification::Scores(Vec::new()), None))
            }
        }

        let answer_child = SubagentRouter::new(
            Arc::new(Passthrough),
            SubagentRouterConfig {
                classifier: Arc::new(AnswerTrackingChild),
                default_target: Category::Efficient,
                classify_trigger: ClassifyTrigger::NewSession,
                message_hash_fallback: false,
            },
        )?;
        assert!(!answer_child.supports_host_state());

        let classifier_b = Arc::new(ScriptedClassifier {
            calls: AtomicUsize::new(0),
        });
        let make_stateful_pod =
            |classifier: Arc<dyn Classifier>| -> crate::Result<Arc<SubagentRouter>> {
                let parent = Arc::new(crate::StageRouter::new(crate::StageRouterConfig::new(
                    crate::PickerMode::EfficientFirst,
                    0.5,
                ))?);
                Ok(Arc::new(SubagentRouter::new(
                    parent,
                    SubagentRouterConfig {
                        classifier,
                        default_target: Category::Efficient,
                        classify_trigger: ClassifyTrigger::NewSession,
                        message_hash_fallback: false,
                    },
                )?))
            };
        let pod_a = make_stateful_pod(Arc::new(ScriptedClassifier {
            calls: AtomicUsize::new(0),
        }))?;
        let pod_b = make_stateful_pod(classifier_b.clone())?;
        let models = RuntimeModels::new(
            [
                (
                    Category::Any,
                    vec![ModelId::from("parent_strong"), ModelId::from("parent_fast")],
                ),
                (Category::Capable, vec![ModelId::from("parent_strong")]),
                (Category::Efficient, vec![ModelId::from("parent_fast")]),
            ]
            .into(),
        )
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

        let tool_turn = |meta: Metadata, output: &str, is_error: bool| Request {
            llm_request: switchyard_protocol::LlmRequest {
                messages: vec![
                    Message::new(
                        Role::User,
                        vec![ContentBlock::Text {
                            text: "fix build".to_string(),
                        }],
                    ),
                    Message::new(
                        Role::Assistant,
                        vec![ContentBlock::ToolCall(switchyard_protocol::ToolCall {
                            id: "call-1".to_string(),
                            name: "Bash".to_string(),
                            arguments: serde_json::json!({"command": "cargo test"}),
                        })],
                    ),
                    Message::new(
                        Role::Tool,
                        vec![ContentBlock::ToolResult(switchyard_protocol::ToolResult {
                            tool_call_id: "call-1".to_string(),
                            content: vec![ContentBlock::Text {
                                text: output.to_string(),
                            }],
                            is_error: Some(is_error),
                        })],
                    ),
                ],
                ..Default::default()
            },
            metadata: Some(meta),
            ..Default::default()
        };

        // Key external records by SessionKey so parent and child records stay separate.
        let mut store: HashMap<SessionKey, State> = HashMap::new();
        let route_keyed = |algo: Arc<SubagentRouter>,
                           req: Request,
                           models: RuntimeModels,
                           mut store: HashMap<SessionKey, State>| async move {
            let (key, session_final) = SessionKey::from_request(&req).unwrap();
            let loaded = store.get(&key).cloned().unwrap_or_default();
            let outcome =
                crate::core::testing::test_drive_with_host_state(algo, req, models, loaded, echo())
                    .await?;
            let selected = outcome.selected_model_id()?.clone();
            if session_final {
                store.retain(|stored, _| stored.session_id != key.session_id);
            } else if outcome.state.extra.is_empty() {
                store.remove(&key);
            } else {
                store.insert(key, outcome.state);
            }
            Ok::<_, crate::LibsyError>((selected, store))
        };

        let root_meta = Metadata {
            session_id: Some("session-1".to_string()),
            ..Metadata::default()
        };
        let (root_open, next_store) = route_keyed(
            pod_a.clone(),
            request(Some(root_meta.clone())),
            models.clone(),
            store,
        )
        .await?;
        store = next_store;
        assert_eq!(root_open, ModelId::from("parent_fast"));

        // Root escalates on a critical error, arming its 2-turn capable hold.
        let (root_escalated, next_store) = route_keyed(
            pod_a.clone(),
            tool_turn(
                root_meta.clone(),
                "fatal runtime error: out of memory",
                true,
            ),
            models.clone(),
            store,
        )
        .await?;
        store = next_store;
        assert_eq!(root_escalated, ModelId::from("parent_strong"));

        let (first_sel, next_store) =
            route_keyed(pod_a.clone(), child("child-1"), models.clone(), store).await?;
        store = next_store;
        assert_eq!(first_sel, ModelId::from("worker"));

        let (sibling_sel, next_store) =
            route_keyed(pod_a.clone(), child("child-2"), models.clone(), store).await?;
        store = next_store;
        assert_eq!(sibling_sel, ModelId::from("reviewer"));

        // Non-delegated child maintenance uses child-1's SessionKey, keeping root's hold
        // turns intact while preserving child-1's delegated affinity in the same record.
        let maintenance = tool_turn(
            Metadata {
                session_id: Some("session-1".to_string()),
                agent_id: Some("child-1".to_string()),
                is_subagent: true,
                is_delegated_work: false,
                agent_kind: Some("compact".to_string()),
                ..Metadata::default()
            },
            "summary complete",
            false,
        );
        let (maint_sel, next_store) =
            route_keyed(pod_b.clone(), maintenance, models.clone(), store).await?;
        store = next_store;
        assert_eq!(maint_sel, ModelId::from("parent_fast"));

        // Both of root's capable hold turns remain untouched on pod_b.
        for _ in 0..2 {
            let (held_sel, next_store) = route_keyed(
                pod_b.clone(),
                tool_turn(root_meta.clone(), "ok", false),
                models.clone(),
                store,
            )
            .await?;
            store = next_store;
            assert_eq!(held_sel, ModelId::from("parent_strong"));
        }

        // child-1's delegated affinity also remains intact without re-classifying.
        let (resumed_sel, next_store) =
            route_keyed(pod_b.clone(), child("child-1"), models.clone(), store).await?;
        store = next_store;
        assert_eq!(resumed_sel, ModelId::from("worker"));
        assert_eq!(classifier_b.calls.load(Ordering::Relaxed), 0);

        // An unidentifiable delegated child routes ephemerally without mutating caller state.
        let root_key = SessionKey {
            session_id: "session-1".to_string(),
            agent_id: None,
        };
        let before_unkeyed = store.get(&root_key).cloned().unwrap();
        let unkeyed_child = request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            is_subagent: true,
            is_delegated_work: true,
            ..Metadata::default()
        }));
        let unkeyed_outcome = crate::core::testing::test_drive_with_host_state(
            pod_a,
            unkeyed_child,
            models.clone(),
            before_unkeyed.clone(),
            echo(),
        )
        .await?;
        assert_eq!(unkeyed_outcome.state.extra, before_unkeyed.extra);

        // session_final clears every stored record belonging to the finalized session.
        let final_child = request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("child-1".to_string()),
            is_subagent: true,
            is_delegated_work: true,
            session_final: Some(true),
            ..Metadata::default()
        }));
        let (_, store) = route_keyed(pod_b, final_child, models, store).await?;
        assert!(store.is_empty());
        Ok(())
    }
}

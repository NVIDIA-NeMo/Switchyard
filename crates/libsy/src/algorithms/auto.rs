// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named request signals resolved through one model pool.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{Category, ModelId, Request};

use super::advisor_gate::{AdvisorGate, AdvisorGateConfig};
use super::llm_class::{LlmTaskClassifier, TaskClassifierConfig};
use super::util::affinity::{evict_if_full, starts_user_turn};
use super::util::stage::{PickerMode, StageClassifier, record_routing_decision};
use super::util::tool_signals::{DEFAULT_RECENT_WINDOW, ToolSemantics, ToolSignalProcessor};
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::pool::{self, PoolContext};
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};

/// When a task signal consults its judge. Retention stores evidence, never a model pin.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskSignalTrigger {
    #[default]
    /// Consult on each request.
    EveryRequest,
    /// Consult at a new user turn, retaining evidence between turns.
    UserTurn,
    /// Consult once per explicit session or child identity.
    NewSession,
    /// Consult only when tool evidence abstains.
    OnDemand,
}

/// Judge settings and the lifetime of its signal.
pub struct TaskSignalSettings {
    /// Judge contract and runtime settings.
    pub classifier: TaskClassifierConfig,
    /// When to refresh the estimate.
    pub trigger: TaskSignalTrigger,
}

/// Tool history settings. Tier policies belong to the standalone stage router.
pub struct ToolLoopSignalConfig {
    /// Minimum corroboration for a decisive tool signal.
    pub confidence_threshold: f64,
    /// Number of recent tool results; uses the stage default when unset.
    pub recent_window: Option<usize>,
    /// Additional tool names and their effects.
    pub tool_semantics: ToolSemantics,
}

/// A review gate applied after pool selection.
pub struct PoolReviewConfig {
    /// Reviewer target, separate from the task judge.
    pub judge: ModelId,
    /// Review budget, trigger, prompts, and error policy.
    pub gate: AdvisorGateConfig,
}

/// One route's signals and its deployment's shared catalog.
pub struct AutoRouterConfig {
    /// Models and capability tags declared once for the deployment.
    pub pool: PoolContext,
    /// Optional task solve estimate.
    pub task: Option<TaskSignalSettings>,
    /// Optional tool-history evidence.
    pub tool_loop: Option<ToolLoopSignalConfig>,
    /// Optional answer review after selection.
    pub review: Option<PoolReviewConfig>,
}

#[derive(Clone, Default)]
struct TaskEvidence {
    score: Option<Score>,
    evidence: Option<Value>,
}

/// Collects signals, resolves their needs, then optionally reviews the selected answer.
pub struct AutoRouter {
    pool: PoolContext,
    task: Option<(Arc<dyn Classifier>, TaskSignalTrigger)>,
    tool_loop: Option<(ToolSignalProcessor, StageClassifier)>,
    review: Option<(AdvisorGate, ModelId)>,
    task_evidence: Mutex<HashMap<RoutingIdentity, TaskEvidence>>,
    used: Mutex<HashMap<RoutingIdentity, Vec<ModelId>>>,
}

impl AutoRouter {
    /// Builds a route, rejecting a missing default or required capability.
    pub fn new(config: AutoRouterConfig) -> Result<Self> {
        if config.pool.models.is_empty()
            || !config
                .pool
                .models
                .iter()
                .any(|m| m.provides.contains(&config.pool.default))
        {
            return Err(LibsyError::AlgorithmError {
                message: "pool must provide its default capability".into(),
            });
        }
        if config.task.is_some() || config.tool_loop.is_some() {
            for tier in [Category::Efficient, Category::Capable] {
                if !config
                    .pool
                    .models
                    .iter()
                    .any(|m| m.provides.contains(&tier))
                {
                    return Err(LibsyError::AlgorithmError {
                        message: format!("configured signals require pool capability {tier:?}"),
                    });
                }
            }
        }
        let task = config
            .task
            .map(|task| -> Result<_> {
                Ok((
                    LlmTaskClassifier::probability_signal(task.classifier)?,
                    task.trigger,
                ))
            })
            .transpose()?;
        let tool_loop = config
            .tool_loop
            .map(|tool| {
                tool.tool_semantics.validate()?;
                if !(0.0..=1.0).contains(&tool.confidence_threshold)
                    || tool.recent_window == Some(0)
                {
                    return Err(LibsyError::AlgorithmError {
                        message: "invalid tool_loop confidence or recent window".into(),
                    });
                }
                Ok((
                    ToolSignalProcessor {
                        recent_window: tool.recent_window.unwrap_or(DEFAULT_RECENT_WINDOW),
                        tool_semantics: tool.tool_semantics,
                    },
                    StageClassifier::new(PickerMode::EfficientFirst, tool.confidence_threshold)
                        .defer_decision_reporting(),
                ))
            })
            .transpose()?;
        let review = config
            .review
            .map(|review| -> Result<_> { Ok((AdvisorGate::new(review.gate)?, review.judge)) })
            .transpose()?;
        Ok(Self {
            pool: config.pool,
            task,
            tool_loop,
            review,
            task_evidence: Mutex::default(),
            used: Mutex::default(),
        })
    }

    async fn task_score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
        other_decided: bool,
    ) -> Result<TaskEvidence> {
        let Some((task, trigger)) = &self.task else {
            return Ok(TaskEvidence::default());
        };
        if *trigger == TaskSignalTrigger::OnDemand && other_decided {
            return Ok(TaskEvidence::default());
        }
        let identity = RoutingIdentity::from_request(request);
        let retained = identity
            .as_ref()
            .and_then(|id| self.task_evidence.lock().get(id).cloned());
        let due = match trigger {
            TaskSignalTrigger::EveryRequest | TaskSignalTrigger::OnDemand => true,
            TaskSignalTrigger::UserTurn => {
                retained.is_none()
                    || (!request.llm_request.is_compaction && starts_user_turn(request))
            }
            TaskSignalTrigger::NewSession => retained.is_none(),
        };
        if !due {
            return Ok(retained.unwrap_or_default());
        }
        let (classification, _) = task.score(state, request, driver).await?;
        let value = TaskEvidence {
            score: classification.argmax(false)?,
            evidence: driver.evidence(),
        };
        if matches!(
            trigger,
            TaskSignalTrigger::UserTurn | TaskSignalTrigger::NewSession
        ) && let Some(id) = identity
        {
            let mut memory = self.task_evidence.lock();
            evict_if_full(&mut memory);
            memory.insert(id, value.clone());
        }
        Ok(value)
    }
}

#[async_trait]
impl Algorithm for AutoRouter {
    fn name(&self) -> &str {
        "auto"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        let identity = RoutingIdentity::from_request(&request);
        let final_turn = request.metadata.as_ref().and_then(|m| m.session_final) == Some(true);
        let mut state = State::default();
        let mut signals = Vec::new();
        if let Some((processor, classifier)) = &self.tool_loop {
            processor
                .process(
                    &mut state,
                    Event::Request {
                        request: &mut request,
                        driver: &driver,
                    },
                )
                .await?;
            let (classification, _) = classifier.score(&mut state, &mut request, &driver).await?;
            if let Some(score) = classification.argmax(false)? {
                signals.push((score, driver.evidence()));
            }
        }
        let task = self
            .task_score(&mut state, &mut request, &driver, !signals.is_empty())
            .await?;
        let raw_task = task.score.as_ref().map(|s| s.confidence);
        if let Some(score) = task.score {
            let threshold = task
                .evidence
                .as_ref()
                .and_then(|e| e.get("threshold"))
                .and_then(Value::as_f64)
                .unwrap_or(0.5);
            signals.push((pool::probability_need(score, threshold), task.evidence));
        }
        let scores: Vec<_> = signals.iter().map(|(score, _)| score.clone()).collect();
        let winner = Classification::Scores(scores.clone()).argmax(false)?;
        let evidence = signals
            .into_iter()
            .find_map(|(score, evidence)| (Some(&score) == winner.as_ref()).then_some(evidence))
            .flatten()
            .unwrap_or_else(|| json!({"source": "pool_default"}));
        let used = identity
            .as_ref()
            .and_then(|id| self.used.lock().get(id).cloned())
            .unwrap_or_default();
        let selected = pool::select_with_history(&scores, &self.pool, &used)?;
        record_routing_decision(
            evidence
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("pool_default"),
            driver.provider_model(&selected),
        );
        driver.set_evidence(json!({"source": "pool", "decision": evidence, "task_probability": raw_task, "needs": winner.as_ref().map(|s| s.needs.iter().map(Category::as_str).collect::<Vec<_>>())}));
        let mut candidates = vec![selected.clone()];
        candidates.extend(
            self.pool
                .models
                .iter()
                .filter(|m| m.target != selected)
                .map(|m| m.target.clone()),
        );
        let outcome = if let Some((review, judge)) = &self.review {
            review
                .review_selected(&driver, request, candidates, judge)
                .await?
        } else {
            RoutingOutcome::route_to(selected, candidates[1..].to_vec(), request)
        };
        if let Some(id) = identity {
            let mut used = self.used.lock();
            let mut memory = self.task_evidence.lock();
            if final_turn {
                used.remove(&id);
                memory.remove(&id);
            } else {
                evict_if_full(&mut used);
                let models = used.entry(id).or_default();
                let selected = outcome.selected_model_id()?.clone();
                if !models.contains(&selected) {
                    models.push(selected);
                }
            }
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::llm_class::{CapabilityJudgeConfig, LlmCapabilityConfig};
    use crate::algorithms::util::tier_fixtures::{JUDGE, Recorder, turn_request};
    use crate::core::pool::PoolModel;
    use crate::core::testing::{reply, test_drive_with_models};
    use switchyard_protocol::{Message, Metadata, Role};

    fn pool() -> PoolContext {
        PoolContext {
            models: vec![
                PoolModel {
                    target: "strong".into(),
                    provides: vec![Category::Capable],
                },
                PoolModel {
                    target: "weak".into(),
                    provides: vec![Category::Efficient],
                },
            ],
            default: Category::Capable,
        }
    }

    fn models() -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Capable, vec!["strong".into()]),
            (Category::Efficient, vec!["weak".into()]),
            (Category::Any, vec!["strong".into(), "weak".into()]),
            (Category::Judge, vec![JUDGE.into(), "reviewer".into()]),
        ]
        .into()
    }

    fn config(trigger: TaskSignalTrigger) -> AutoRouterConfig {
        AutoRouterConfig {
            pool: pool(),
            task: Some(TaskSignalSettings {
                classifier: TaskClassifierConfig {
                    judge: CapabilityJudgeConfig::Llm(LlmCapabilityConfig {
                        base_threshold: 0.9,
                        ..LlmCapabilityConfig::default()
                    }),
                    ..TaskClassifierConfig::default()
                },
                trigger,
            }),
            tool_loop: Some(ToolLoopSignalConfig {
                confidence_threshold: 0.5,
                recent_window: None,
                tool_semantics: ToolSemantics::default(),
            }),
            review: None,
        }
    }

    #[tokio::test]
    async fn retained_task_evidence_allows_a_later_tool_override() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.95;
        let router = Arc::new(AutoRouter::new(config(TaskSignalTrigger::UserTurn))?);
        let (first, _) = test_drive_with_models(
            router.clone(),
            turn_request(false),
            models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(first, "weak");
        let (next, _) = test_drive_with_models(
            router.clone(),
            turn_request(true),
            models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(next, "strong");
        assert_eq!(recorder.judge_calls(), 1);
        let mut user = turn_request(false);
        user.llm_request
            .messages
            .push(Message::text(Role::User, "another task"));
        test_drive_with_models(router, user, models(), recorder.serve()).await?;
        assert_eq!(recorder.judge_calls(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn on_demand_skips_the_judge_when_tool_evidence_decides() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        let router = Arc::new(AutoRouter::new(config(TaskSignalTrigger::OnDemand))?);
        let (selected, _) = test_drive_with_models(
            router.clone(),
            turn_request(true),
            models(),
            recorder.serve(),
        )
        .await?;
        assert_eq!(selected, "strong");
        assert!(recorder.judge_calls() == 0);
        *recorder.judge_p_solve.lock() = 0.95;
        let (selected, _) =
            test_drive_with_models(router, turn_request(false), models(), recorder.serve()).await?;
        assert_eq!(selected, "weak");
        assert_eq!(recorder.judge_calls(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn retained_task_evidence_isolated_by_session_and_child_identity() -> Result<()> {
        let recorder = Arc::new(Recorder::default());
        *recorder.judge_p_solve.lock() = 0.95;
        let router = Arc::new(AutoRouter::new(config(TaskSignalTrigger::NewSession))?);
        for (session, agent) in [("a", None), ("b", None), ("a", Some("child")), ("a", None)] {
            let mut request = turn_request(false);
            request.metadata = Some(Metadata {
                session_id: Some(session.into()),
                agent_id: agent.map(str::to_owned),
                is_subagent: agent.is_some(),
                ..Metadata::default()
            });
            test_drive_with_models(router.clone(), request, models(), recorder.serve()).await?;
        }
        assert_eq!(recorder.judge_calls(), 3);
        Ok(())
    }

    #[tokio::test]
    async fn review_uses_the_pool_selected_executor_and_its_own_judge() -> Result<()> {
        let mut config = config(TaskSignalTrigger::EveryRequest);
        config.review = Some(PoolReviewConfig {
            judge: "reviewer".into(),
            gate: AdvisorGateConfig::default(),
        });
        let router = Arc::new(AutoRouter::new(config)?);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        let (selected, _) = test_drive_with_models(router, turn_request(true), models(), move |model: ModelId, _request: Request| {
            let seen = seen.clone();
            async move {
                seen.lock().push(model.to_string());
                Ok(reply(match model.as_str() {
                    JUDGE => r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.95}"#,
                    "reviewer" => "APPROVE",
                    "strong" => "done",
                    other => panic!("unexpected executor or reviewer: {other}"),
                }))
            }
        }).await?;
        assert_eq!(selected, "strong");
        assert_eq!(*calls.lock(), [JUDGE, "strong", "reviewer"]);
        Ok(())
    }

    #[tokio::test]
    async fn review_redo_keeps_the_selected_executor_and_feedback() -> Result<()> {
        let mut config = config(TaskSignalTrigger::EveryRequest);
        config.task = None;
        config.review = Some(PoolReviewConfig {
            judge: "reviewer".into(),
            gate: AdvisorGateConfig::default(),
        });
        let router = Arc::new(AutoRouter::new(config)?);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        let (selected, _) = test_drive_with_models(
            router,
            turn_request(true),
            models(),
            move |model: ModelId, request: Request| {
                let seen = seen.clone();
                async move {
                    let mut calls = seen.lock();
                    calls.push(model.to_string());
                    if calls.len() == 3 {
                        assert!(
                            request
                                .llm_request
                                .messages
                                .last()
                                .unwrap()
                                .text_content("\n")
                                .unwrap()
                                .contains("run tests")
                        );
                    }
                    Ok(reply(if model == "reviewer" {
                        "REDO\nrun tests"
                    } else {
                        "done"
                    }))
                }
            },
        )
        .await?;
        assert_eq!(selected, "strong");
        assert_eq!(*calls.lock(), ["strong", "reviewer", "strong"]);
        Ok(())
    }
    #[tokio::test]
    async fn review_confirmations_share_one_budget_and_refund_unattempted_checks() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for first_redo in [false, true] {
            let mut config = config(TaskSignalTrigger::EveryRequest);
            config.task = None;
            config.tool_loop = None;
            config.review = Some(PoolReviewConfig {
                judge: "reviewer".into(),
                gate: AdvisorGateConfig {
                    max_reviews: 3,
                    confirmations: 2,
                    ..AdvisorGateConfig::default()
                },
            });
            let router = Arc::new(AutoRouter::new(config)?);
            let reviews = Arc::new(AtomicUsize::new(0));
            for _ in 0..2 {
                let count = reviews.clone();
                test_drive_with_models(
                    router.clone(),
                    turn_request(false),
                    models(),
                    move |model: ModelId, _: Request| {
                        let count = count.clone();
                        async move {
                            Ok(reply(if model == "reviewer" {
                                if count.fetch_add(1, Ordering::SeqCst) == 0 && first_redo {
                                    "REDO\nrun tests"
                                } else {
                                    "APPROVE"
                                }
                            } else {
                                "done"
                            }))
                        }
                    },
                )
                .await?;
            }
            assert_eq!(
                reviews.load(Ordering::SeqCst),
                if first_redo { 3 } else { 2 }
            );
        }
        Ok(())
    }
}

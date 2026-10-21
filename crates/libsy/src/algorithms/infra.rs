// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional efficient-to-capable spillover after task routing.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use serde::Deserialize;
use switchyard_protocol::{Category, InfraSignals, ModelId, ModelInfraSignals, Request};

use crate::{Algorithm, Driver, LibsyError, Result, RoutingOutcome};

/// Deployment-calibrated work scores, not predicted response latency.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InfraRouterConfig {
    /// Deadline for the host lookup. Expiry preserves the task decision.
    pub probe_timeout_ms: u64,
    /// Maximum observation age at the final routing decision.
    pub max_signal_age_ms: u64,
    /// Require at least this many in-flight requests on the efficient candidate.
    pub min_active_requests: u64,
    /// Protect the capable candidate when it has more in-flight requests than this.
    pub max_spillover_active_requests: u64,
    /// Relative prefill costs, ordered efficient then capable.
    pub prefill_costs: [f64; 2],
    /// Extra work per in-flight request, expressed in equivalent prefill tokens.
    pub request_penalty_tokens: f64,
    /// Required fractional score reduction, in [0, 1).
    pub min_improvement: f64,
}

impl InfraRouterConfig {
    fn validate(&self) -> Result<()> {
        if self.probe_timeout_ms == 0
            || self.max_signal_age_ms == 0
            || self.min_active_requests == 0
            || self
                .prefill_costs
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || !self.request_penalty_tokens.is_finite()
            || self.request_penalty_tokens <= 0.0
            || !(0.0..1.0).contains(&self.min_improvement)
        {
            return Err(LibsyError::AlgorithmError {
                message: "infra requires positive age, source load, costs and request penalty, and min_improvement in [0, 1)".into(),
            });
        }
        Ok(())
    }

    fn score(&self, signal: &ModelInfraSignals, cost: f64, now: Instant) -> Option<f64> {
        if now.checked_duration_since(signal.sampled_at)?
            > Duration::from_millis(self.max_signal_age_ms)
        {
            return None;
        }
        let work = signal.effective_prefill_tokens? as f64
            + signal.active_prefill_tokens? as f64
            + signal.active_requests? as f64 * self.request_penalty_tokens;
        let score = work * cost;
        score.is_finite().then_some(score)
    }
}

/// Runs task routing first, then optionally moves an efficient choice to the capable tier.
///
/// The host supplies trusted observations through a read-only call after task routing. This policy does not
/// select or reserve a worker. Task affinity stays in the inner algorithm, so a
/// temporary spillover does not latch the session onto the capable model.
pub struct InfraRouter {
    inner: Arc<dyn Algorithm>,
    config: InfraRouterConfig,
}

impl InfraRouter {
    /// Wraps a decision-only tier router. Both tiers must contain exactly one model.
    pub fn new(inner: Arc<dyn Algorithm>, config: InfraRouterConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { inner, config })
    }

    fn evaluate(
        &self,
        signals: &InfraSignals,
        efficient: &ModelId,
        capable: &ModelId,
    ) -> (&'static str, Option<(f64, f64)>) {
        let (Some(source), Some(destination)) = (signals.get(efficient), signals.get(capable))
        else {
            return ("signals_unavailable", None);
        };
        let now = Instant::now();
        let (Some(source_score), Some(destination_score)) = (
            self.config.score(source, self.config.prefill_costs[0], now),
            self.config
                .score(destination, self.config.prefill_costs[1], now),
        ) else {
            return ("stale_or_incomplete_signals", None);
        };
        let scores = Some((source_score, destination_score));
        if source.active_requests.unwrap() < self.config.min_active_requests {
            return ("source_not_busy", scores);
        }
        if destination.active_requests.unwrap() > self.config.max_spillover_active_requests {
            return ("destination_busy", scores);
        }
        if destination_score < source_score * (1.0 - self.config.min_improvement) {
            ("lower_work_score", scores)
        } else {
            ("insufficient_improvement", scores)
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for InfraRouter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let delegated = request
            .metadata
            .as_ref()
            .is_some_and(|m| m.is_subagent_work());
        let mut outcome = self.inner.clone().route(driver.clone(), request).await?;
        // Sub-agent routing has separate target groups and capability requirements.
        if outcome.response.is_some() || delegated {
            return Ok(outcome);
        }
        let ([efficient], [capable]) = (
            driver.models_for(&Category::Efficient),
            driver.models_for(&Category::Capable),
        ) else {
            return Err(LibsyError::AlgorithmError {
                message: "infra requires exactly one efficient and one capable model".into(),
            });
        };
        if outcome.selected_model_id()? != efficient || efficient == capable {
            return Ok(outcome);
        }
        let lookup = tokio::time::timeout(
            Duration::from_millis(self.config.probe_timeout_ms),
            driver.call_infra(
                outcome.request.clone(),
                vec![efficient.clone(), capable.clone()],
            ),
        )
        .await;
        let (reason, scores) = match lookup {
            Ok(Ok(signals)) => self.evaluate(&signals, efficient, capable),
            Ok(Err(_)) => ("signals_unavailable", None),
            Err(_) => ("probe_timeout", None),
        };
        let spillover = reason == "lower_work_score";
        driver.annotate_evidence(
            "infra",
            serde_json::json!({
                "task_model": efficient,
                "alternate_model": capable,
                "spillover": spillover,
                "reason": reason,
                "source_score": scores.map(|s| s.0),
                "destination_score": scores.map(|s| s.1),
            }),
        );
        tracing::info!(
            task_model = %efficient, alternate_model = %capable, spillover, reason,
            source_score = scores.map(|s| s.0), destination_score = scores.map(|s| s.1),
            "infrastructure routing decision"
        );
        if spillover {
            outcome.selected_model_ids.retain(|model| model != capable);
            outcome.selected_model_ids.insert(0, capable.clone());
            outcome.request.llm_request.model = Some(capable.to_string());
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Call, RuntimeModels};
    use switchyard_protocol::text_request;

    struct TaskChoice(&'static str);

    #[async_trait::async_trait]
    impl Algorithm for TaskChoice {
        fn name(&self) -> &str {
            "task"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> Result<RoutingOutcome> {
            driver.set_evidence(serde_json::json!({"task_reason": "classified"}));
            Ok(RoutingOutcome::route_to(self.0.into(), vec![], request))
        }
    }

    fn config() -> InfraRouterConfig {
        InfraRouterConfig {
            probe_timeout_ms: 10,
            max_signal_age_ms: 1000,
            min_active_requests: 8,
            max_spillover_active_requests: 1,
            prefill_costs: [1.0, 4.0],
            request_penalty_tokens: 4096.0,
            min_improvement: 0.2,
        }
    }

    #[tokio::test]
    async fn spillover_requires_a_fresh_and_better_capable_candidate() {
        for case in [
            "busy",
            "quiet",
            "destination_busy",
            "cold",
            "missing",
            "incomplete",
            "stale",
            "future",
            "timeout",
            "capable_task",
        ] {
            let mut signals: InfraSignals = [("small", 8), ("large", 0)]
                .into_iter()
                .map(|(id, active)| {
                    (
                        id.into(),
                        ModelInfraSignals {
                            sampled_at: Instant::now(),
                            effective_prefill_tokens: Some(1000),
                            active_prefill_tokens: Some(0),
                            active_requests: Some(active),
                        },
                    )
                })
                .collect();
            let source = signals.get_mut(&ModelId::from("small")).unwrap();
            match case {
                "quiet" => source.active_requests = Some(0),
                "incomplete" => source.active_requests = None,
                "stale" => source.sampled_at -= Duration::from_secs(2),
                "future" => source.sampled_at += Duration::from_secs(2),
                "missing" => signals.clear(),
                "destination_busy" => {
                    signals
                        .get_mut(&ModelId::from("large"))
                        .unwrap()
                        .active_requests = Some(2)
                }
                "cold" => {
                    signals
                        .get_mut(&ModelId::from("large"))
                        .unwrap()
                        .effective_prefill_tokens = Some(20_000)
                }
                _ => {}
            }
            let task_model = if case == "capable_task" {
                "large"
            } else {
                "small"
            };
            let algorithm =
                Arc::new(InfraRouter::new(Arc::new(TaskChoice(task_model)), config()).unwrap());
            let models = RuntimeModels::new(
                [
                    (Category::Efficient, vec!["small".into()]),
                    (Category::Capable, vec!["large".into()]),
                ]
                .into(),
            );
            let request = Request {
                llm_request: text_request(None, "hello"),
                ..Default::default()
            };
            let expected = if matches!(case, "busy" | "capable_task") {
                "large"
            } else {
                "small"
            };
            let outcome = crate::drive(algorithm, request.clone(), Arc::new(models), move |call| {
                let signals = signals.clone();
                async move {
                    let Call::Infra(call) = call else {
                        panic!("unexpected model call")
                    };
                    assert_ne!(case, "capable_task");
                    assert_eq!(call.request.llm_request.model.as_deref(), Some("small"));
                    assert_eq!(
                        call.candidates,
                        vec![ModelId::from("small"), ModelId::from("large")]
                    );
                    call.respond(async move {
                        if case == "timeout" {
                            std::future::pending::<()>().await;
                        }
                        signals
                    })
                    .await
                }
            })
            .await
            .unwrap();
            assert_eq!(
                outcome.selected_model_id().unwrap().as_str(),
                expected,
                "{case}"
            );
            let mut actual_request = outcome.request.llm_request;
            assert_eq!(actual_request.model.take().as_deref(), Some(expected));
            assert_eq!(actual_request, request.llm_request);
            let evidence = outcome.metadata.unwrap().evidence.unwrap();
            assert_eq!(evidence["task_reason"], "classified");
            if case != "capable_task" {
                assert_eq!(evidence["infra"]["spillover"], case == "busy");
            }
        }
    }
}

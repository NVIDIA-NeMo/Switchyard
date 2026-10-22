// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional queue-based moves from the efficient model to the capable model.

use crate::{LibsyError, Result, RoutingOutcome, RuntimeModels};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
use switchyard_protocol::{Category, InfraSignals, ModelInfraSignals};

/// Measured service times for a representative, bounded workload on one efficient worker.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueCalibration {
    /// Number of background requests served per measured batch.
    pub batch_size: u64,
    /// Duration of a background batch, measured under the expected contention.
    pub batch_time_ms: f64,
    /// Duration of the foreground request without a background queue.
    pub request_time_ms: f64,
}

/// Queue thresholds and measured service times. No production defaults are assumed.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InfraPolicyConfig {
    /// Maximum age of every required worker observation.
    pub max_signal_age_ms: u64,
    /// Every efficient worker must have at least this many waiting requests.
    pub min_waiting_requests: u64,
    /// Stop moving requests when observed running or locally tracked capable work reaches this limit.
    pub max_capable_inflight: u64,
    /// Limit concurrent moved requests through the host.
    pub max_spillovers: u64,
    /// Required predicted saving over the capable model's service time.
    pub min_saving_ms: f64,
    /// All efficient workers, with their measured batch and foreground service times.
    pub efficient_workers: BTreeMap<String, QueueCalibration>,
    /// All capable workers, with their measured foreground service time in milliseconds.
    pub capable_workers: BTreeMap<String, f64>,
}

/// Applies a queue policy after task routing; the host owns collection and request lifetimes.
pub struct InfraPolicy {
    config: InfraPolicyConfig,
}

impl InfraPolicy {
    /// Validate the policy once at configuration load.
    pub fn new(config: InfraPolicyConfig) -> Result<Self> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if config.max_signal_age_ms == 0
            || config.min_waiting_requests == 0
            || config.max_capable_inflight == 0
            || config.max_spillovers == 0
            || !config.min_saving_ms.is_finite()
            || config.min_saving_ms < 0.0
            || config.efficient_workers.is_empty()
            || config.capable_workers.is_empty()
            || config.efficient_workers.iter().any(|(id, c)| {
                id.is_empty()
                    || c.batch_size == 0
                    || !positive(c.batch_time_ms)
                    || !positive(c.request_time_ms)
                    || config.capable_workers.contains_key(id)
            })
            || config
                .capable_workers
                .iter()
                .any(|(id, ms)| id.is_empty() || !positive(*ms))
        {
            return Err(LibsyError::AlgorithmError { message: "infra requires positive limits and service times, distinct nonempty worker IDs, and a finite nonnegative saving margin".into() });
        }
        Ok(Self { config })
    }

    fn fresh<T>(
        &self,
        model: &ModelInfraSignals,
        expected: &BTreeMap<String, T>,
        now: Instant,
    ) -> bool {
        expected.len() == model.workers.len()
            && expected.keys().all(|id| {
                model.workers.get(id).is_some_and(|worker| {
                    now.checked_duration_since(worker.sampled_at)
                        .is_some_and(|age| {
                            age <= Duration::from_millis(self.config.max_signal_age_ms)
                        })
                })
            })
    }

    fn evaluate(
        &self,
        source: &ModelInfraSignals,
        destination: &ModelInfraSignals,
    ) -> (bool, &'static str, Option<(f64, f64)>) {
        let now = Instant::now();
        if !self.fresh(source, &self.config.efficient_workers, now)
            || !self.fresh(destination, &self.config.capable_workers, now)
        {
            return (false, "unknown_or_stale_signals", None);
        }
        if source
            .workers
            .values()
            .any(|w| w.waiting < self.config.min_waiting_requests)
        {
            return (false, "small_workers_not_all_queued", None);
        }
        let running = destination
            .workers
            .values()
            .try_fold(0u64, |sum, w| sum.checked_add(w.running));
        if destination.workers.values().any(|w| w.waiting > 0)
            || running.is_none_or(|n| {
                n.max(destination.local_inflight) >= self.config.max_capable_inflight
            })
            || destination.local_spillovers >= self.config.max_spillovers
        {
            return (false, "protect_capable_capacity", None);
        }
        // Use the fastest small-worker estimate: EPP may still find a better small worker.
        let small_ms = source
            .workers
            .iter()
            .map(|(id, worker)| {
                let c = &self.config.efficient_workers[id];
                worker.waiting.div_ceil(c.batch_size) as f64 * c.batch_time_ms + c.request_time_ms
            })
            .fold(f64::INFINITY, f64::min);
        let large_ms = self
            .config
            .capable_workers
            .values()
            .copied()
            .fold(0.0, f64::max);
        if !small_ms.is_finite() {
            return (false, "invalid_queue_estimate", None);
        }
        let move_request = small_ms > large_ms + self.config.min_saving_ms;
        (
            move_request,
            if move_request {
                "queue_cost_exceeds_capable_service"
            } else {
                "insufficient_predicted_saving"
            },
            Some((small_ms, large_ms)),
        )
    }

    /// Adjust a decision-only task result. Returns true only when the model changes.
    ///
    /// Hosts must serialize this check with their local in-flight increment, then decrement
    /// when the response ends or is cancelled. Worker selection remains in the serving layer.
    pub fn apply(
        &self,
        outcome: &mut RoutingOutcome,
        models: &RuntimeModels,
        signals: &InfraSignals,
    ) -> Result<bool> {
        if outcome.response.is_some()
            || outcome
                .request
                .metadata
                .as_ref()
                .is_some_and(|m| m.is_subagent_work())
        {
            return Ok(false);
        }
        let ([efficient], [capable]) = (
            models.models_for(&Category::Efficient),
            models.models_for(&Category::Capable),
        ) else {
            return Err(LibsyError::AlgorithmError {
                message: "infra requires one efficient and one capable model".into(),
            });
        };
        if outcome.selected_model_id()? != efficient || efficient == capable {
            return Ok(false);
        }
        let (spillover, reason, estimates) = match (signals.get(efficient), signals.get(capable)) {
            (Some(source), Some(destination)) => self.evaluate(source, destination),
            _ => (false, "unknown_or_stale_signals", None),
        };
        if let Some(metadata) = &mut outcome.metadata {
            let evidence = metadata
                .evidence
                .get_or_insert_with(|| serde_json::json!({}));
            if let Some(object) = evidence.as_object_mut() {
                object.insert(
                    "infra".into(),
                    serde_json::json!({
                        "task_model": efficient, "alternate_model": capable, "spillover": spillover,
                        "reason": reason, "estimated_small_ms": estimates.map(|v| v.0),
                        "estimated_capable_ms": estimates.map(|v| v.1),
                    }),
                );
            }
        }
        tracing::info!(task_model = %efficient, alternate_model = %capable, spillover, reason,
            estimated_small_ms = estimates.map(|v| v.0), estimated_capable_ms = estimates.map(|v| v.1),
            "queue routing decision");
        if spillover {
            outcome.selected_model_ids.retain(|id| id != capable);
            outcome.selected_model_ids.insert(0, capable.clone());
            outcome.request.llm_request.model = Some(capable.to_string());
        }
        Ok(spillover)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{ModelId, Request, WorkerLoad, text_request};

    fn config() -> InfraPolicyConfig {
        InfraPolicyConfig {
            max_signal_age_ms: 1000,
            min_waiting_requests: 4,
            max_capable_inflight: 2,
            max_spillovers: 2,
            min_saving_ms: 500.0,
            efficient_workers: ["a", "b"]
                .into_iter()
                .map(|id| {
                    (
                        id.into(),
                        QueueCalibration {
                            batch_size: 4,
                            batch_time_ms: 2000.0,
                            request_time_ms: 1000.0,
                        },
                    )
                })
                .collect(),
            capable_workers: [("c".into(), 4000.0)].into(),
        }
    }

    #[test]
    fn queue_policy_requires_all_workers_and_preserves_task_choices() {
        for case in [
            "busy",
            "one_quiet",
            "capable_queued",
            "capable_running",
            "local_limit",
            "spillover_limit",
            "missing",
            "stale",
            "future",
            "no_saving",
            "capable_task",
        ] {
            let model = |ids: &[&str], waiting| ModelInfraSignals {
                workers: ids
                    .iter()
                    .map(|id| {
                        (
                            id.to_string(),
                            WorkerLoad {
                                sampled_at: Instant::now(),
                                running: 0,
                                waiting,
                            },
                        )
                    })
                    .collect(),
                ..Default::default()
            };
            let mut source = model(&["a", "b"], 8);
            let mut destination = model(&["c"], 0);
            match case {
                "one_quiet" => source.workers.get_mut("b").unwrap().waiting = 0,
                "capable_queued" => destination.workers.get_mut("c").unwrap().waiting = 1,
                "capable_running" => destination.workers.get_mut("c").unwrap().running = 2,
                "local_limit" => destination.local_inflight = 2,
                "spillover_limit" => destination.local_spillovers = 2,
                "missing" => {
                    source.workers.remove("a");
                }
                "stale" => {
                    source.workers.get_mut("a").unwrap().sampled_at -= Duration::from_secs(2)
                }
                "future" => {
                    destination.workers.get_mut("c").unwrap().sampled_at += Duration::from_secs(2)
                }
                "no_saving" => source.workers.get_mut("b").unwrap().waiting = 4,
                _ => {}
            }
            let signals = [
                (ModelId::from("small"), source),
                (ModelId::from("large"), destination),
            ]
            .into();
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
            let mut outcome = RoutingOutcome::route_to(
                if case == "capable_task" {
                    "large"
                } else {
                    "small"
                }
                .into(),
                vec![],
                request.clone(),
            );
            let moved = InfraPolicy::new(config())
                .unwrap()
                .apply(&mut outcome, &models, &signals)
                .unwrap();
            assert_eq!(moved, case == "busy", "{case}");
            let expected = if matches!(case, "busy" | "capable_task") {
                "large"
            } else {
                "small"
            };
            assert_eq!(
                outcome.selected_model_id().unwrap().as_str(),
                expected,
                "{case}"
            );
            outcome.request.llm_request.model = None;
            assert_eq!(outcome.request.llm_request, request.llm_request);
        }
        let mut invalid = config();
        invalid
            .efficient_workers
            .get_mut("a")
            .unwrap()
            .batch_time_ms = f64::NAN;
        assert!(InfraPolicy::new(invalid).is_err());
    }
}

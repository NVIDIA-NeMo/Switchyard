// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional queue-based moves from the efficient model to the capable model.

use crate::{LibsyError, Result, RoutingOutcome, RuntimeModels};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};
use switchyard_protocol::{Category, InfraSignals, ModelInfraSignals};

/// Queue and capacity limits for moving requests to the capable model.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InfraPolicyConfig {
    /// Maximum age of every required worker observation.
    pub max_signal_age_ms: u64,
    /// Every efficient worker must have at least this many waiting requests.
    pub min_waiting_requests: u64,
    /// Stop moving requests when observed running or locally tracked capable work reaches this limit.
    pub max_capable_inflight: u64,
    /// Complete set of efficient worker IDs in the host's metrics inventory.
    pub efficient_workers: BTreeSet<String>,
    /// Complete set of capable worker IDs in the host's metrics inventory.
    pub capable_workers: BTreeSet<String>,
}

/// Applies a queue policy after task routing; the host owns collection and request lifetimes.
pub struct InfraPolicy {
    config: InfraPolicyConfig,
}

impl InfraPolicy {
    /// Validate the policy once at configuration load.
    pub fn new(config: InfraPolicyConfig) -> Result<Self> {
        if config.max_signal_age_ms == 0
            || config.min_waiting_requests == 0
            || config.max_capable_inflight == 0
            || config.efficient_workers.is_empty()
            || config.capable_workers.is_empty()
            || !config
                .efficient_workers
                .is_disjoint(&config.capable_workers)
            || config
                .efficient_workers
                .iter()
                .chain(&config.capable_workers)
                .any(String::is_empty)
        {
            return Err(LibsyError::AlgorithmError {
                message: "infra requires positive limits and distinct nonempty worker IDs".into(),
            });
        }
        Ok(Self { config })
    }

    fn fresh(&self, model: &ModelInfraSignals, expected: &BTreeSet<String>, now: Instant) -> bool {
        expected.len() == model.workers.len()
            && expected.iter().all(|id| {
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
    ) -> (bool, &'static str) {
        let now = Instant::now();
        if !self.fresh(source, &self.config.efficient_workers, now)
            || !self.fresh(destination, &self.config.capable_workers, now)
        {
            return (false, "unknown_or_stale_signals");
        }
        if source
            .workers
            .values()
            .any(|w| w.waiting < self.config.min_waiting_requests)
        {
            return (false, "small_workers_not_all_queued");
        }
        let running = destination
            .workers
            .values()
            .try_fold(0u64, |sum, w| sum.checked_add(w.running));
        if destination.workers.values().any(|w| w.waiting > 0)
            || running.is_none_or(|n| {
                n.max(destination.local_inflight) >= self.config.max_capable_inflight
            })
        {
            return (false, "protect_capable_capacity");
        }
        (true, "small_workers_queued")
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
        let (spillover, reason) = match (signals.get(efficient), signals.get(capable)) {
            (Some(source), Some(destination)) => self.evaluate(source, destination),
            _ => (false, "unknown_or_stale_signals"),
        };
        tracing::info!(task_model = %efficient, alternate_model = %capable, spillover, reason,
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
            efficient_workers: ["a".into(), "b".into()].into(),
            capable_workers: ["c".into()].into(),
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
            "missing",
            "stale",
            "future",
            "below_threshold",
            "at_threshold",
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
                "missing" => {
                    source.workers.remove("a");
                }
                "stale" => {
                    source.workers.get_mut("a").unwrap().sampled_at -= Duration::from_secs(2)
                }
                "future" => {
                    destination.workers.get_mut("c").unwrap().sampled_at += Duration::from_secs(2)
                }
                "below_threshold" => source.workers.get_mut("b").unwrap().waiting = 3,
                "at_threshold" => source.workers.values_mut().for_each(|w| w.waiting = 4),
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
            assert_eq!(moved, matches!(case, "busy" | "at_threshold"), "{case}");
            let expected = if matches!(case, "busy" | "at_threshold" | "capable_task") {
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
        invalid.min_waiting_requests = 0;
        assert!(InfraPolicy::new(invalid).is_err());
    }
}

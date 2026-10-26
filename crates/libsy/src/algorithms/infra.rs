// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional queue-based moves from the efficient model to the capable model.

use crate::{LibsyError, Result, RoutingOutcome, RuntimeModels};
use serde::Deserialize;
use std::time::{Duration, Instant};
use switchyard_protocol::{Category, InfraSignals, ModelId};

/// Queue and capacity limits for moving requests to the capable model.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InfraPolicy {
    /// Maximum age of the pool observation.
    pub max_signal_age_ms: u64,
    /// Every efficient worker must have at least this many waiting requests.
    pub min_waiting_requests: u64,
    /// Stop moving requests when observed running or locally tracked capable work reaches this limit.
    pub max_capable_inflight: u64,
}

impl InfraPolicy {
    /// Validate the policy once at configuration load.
    pub fn validate(&self) -> Result<()> {
        if self.max_signal_age_ms == 0
            || self.min_waiting_requests == 0
            || self.max_capable_inflight == 0
        {
            return Err(LibsyError::AlgorithmError {
                message: "infra requires positive limits".into(),
            });
        }
        Ok(())
    }

    fn evaluate(
        &self,
        signals: &InfraSignals,
        source: &ModelId,
        destination: &ModelId,
    ) -> Option<(bool, &'static str)> {
        let source = signals.get(source)?.load.as_ref()?;
        let destination = signals.get(destination)?;
        let load = destination.load.as_ref()?;
        let now = Instant::now();
        if [source, load].iter().any(|pool| {
            now.checked_duration_since(pool.sampled_at)
                .is_none_or(|age| age > Duration::from_millis(self.max_signal_age_ms))
        }) {
            return None;
        }
        if source.min_waiting < self.min_waiting_requests {
            return Some((false, "small_pool_not_queued"));
        }
        if load.max_waiting > 0
            || load.running.max(destination.local_inflight) >= self.max_capable_inflight
        {
            return Some((false, "protect_capable_capacity"));
        }
        Some((true, "small_pool_queued"))
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
        let (spillover, reason) = self
            .evaluate(signals, efficient, capable)
            .unwrap_or((false, "unknown_or_stale_signals"));
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
    use switchyard_protocol::{ModelInfraSignals, PoolLoad, Request, text_request};

    #[test]
    fn queue_policy_preserves_task_choices() {
        let policy = InfraPolicy {
            max_signal_age_ms: 1000,
            min_waiting_requests: 4,
            max_capable_inflight: 2,
        };
        policy.validate().unwrap();
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
            let model = |waiting| ModelInfraSignals {
                load: Some(PoolLoad {
                    sampled_at: Instant::now(),
                    running: 0,
                    min_waiting: waiting,
                    max_waiting: waiting,
                }),
                ..Default::default()
            };
            let mut source = model(8);
            let mut destination = model(0);
            match case {
                "one_quiet" => source.load.as_mut().unwrap().min_waiting = 0,
                "capable_queued" => destination.load.as_mut().unwrap().max_waiting = 1,
                "capable_running" => destination.load.as_mut().unwrap().running = 2,
                "local_limit" => destination.local_inflight = 2,
                "missing" => source.load = None,
                "stale" => source.load.as_mut().unwrap().sampled_at -= Duration::from_secs(2),
                "future" => destination.load.as_mut().unwrap().sampled_at += Duration::from_secs(2),
                "below_threshold" => source.load.as_mut().unwrap().min_waiting = 3,
                "at_threshold" => source.load.as_mut().unwrap().min_waiting = 4,
                _ => {}
            }
            let signals = [
                (ModelId::from("small"), source),
                (ModelId::from("large"), destination),
            ]
            .into();
            let initial = if case == "capable_task" {
                "large"
            } else {
                "small"
            };
            let mut outcome = RoutingOutcome::route_to(initial.into(), vec![], request.clone());
            let moved = matches!(case, "busy" | "at_threshold");
            assert_eq!(
                policy.apply(&mut outcome, &models, &signals).unwrap(),
                moved,
                "{case}"
            );
            assert_eq!(
                outcome.selected_model_id().unwrap().as_str(),
                if moved { "large" } else { initial },
                "{case}"
            );
            outcome.request.llm_request.model = None;
            assert_eq!(outcome.request.llm_request, request.llm_request);
        }
        let mut invalid = policy;
        invalid.min_waiting_requests = 0;
        assert!(invalid.validate().is_err());
    }
}

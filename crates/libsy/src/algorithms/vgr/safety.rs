// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The two controls that stop a verification-gated route without reconfiguring it.
//!
//! Both resolve the same way when they fire: the turn escalates to the capable
//! tier without producing an attempt. That is the direction this router already
//! fails in, so neither control can make a turn less safe — only more expensive.
//!
//! [`KillSwitch`] is the operator's manual stop, flippable while the route is
//! serving. [`CircuitBreaker`] is the automatic one, for a local endpoint that
//! has stopped answering: without it every request pays a fresh failed call, and
//! the decision budget is spent on a tier that cannot answer.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::LibsyError;

/// An operator's runtime stop for a verification-gated route.
///
/// Cloneable, and every clone observes the same flag, so an operator can hold a
/// handle while the router holds another. Engaging it takes effect on the next
/// turn: no local attempt is produced and no verifier is consulted.
///
/// Distinct from [`ServingMode::Off`](super::config::ServingMode::Off), which is
/// the same behaviour chosen at construction. This one can be flipped without
/// rebuilding the route, which is what an incident needs.
#[derive(Clone, Debug, Default)]
pub struct KillSwitch(Arc<AtomicBool>);

impl KillSwitch {
    /// A switch that is not engaged.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops local commits from this point on.
    pub fn engage(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Allows local commits again.
    pub fn release(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    /// Whether local commits are currently stopped.
    pub fn is_engaged(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// How the breaker is tuned.
#[derive(Clone, Copy, Debug)]
pub struct BreakerConfig {
    /// Consecutive local-tier failures that open the circuit.
    pub threshold: u32,
    /// How long the circuit stays open before one trial request is allowed.
    pub cooldown: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            threshold: 5,
            cooldown: Duration::from_secs(30),
        }
    }
}

/// Mutable breaker state, held behind one lock so the two fields cannot disagree.
#[derive(Debug, Default)]
struct BreakerState {
    /// Local-tier failures since the last success.
    consecutive_failures: u32,
    /// When the circuit opened, if it is open.
    opened_at: Option<Instant>,
}

/// Stops calling a local endpoint that has failed repeatedly.
///
/// Counts only *consecutive* failures, so an endpoint that is merely lossy never
/// trips: one success resets the count. After the cooldown the circuit admits a
/// single trial request, and a failure of that trial re-opens it immediately
/// rather than starting the count over.
#[derive(Debug)]
pub(super) struct CircuitBreaker {
    config: BreakerConfig,
    state: Mutex<BreakerState>,
}

impl CircuitBreaker {
    /// A closed breaker with the given tuning.
    pub(super) fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BreakerState::default()),
        }
    }

    /// Whether the local tier should be skipped for this turn.
    ///
    /// Expiring the cooldown is a write, so this is not a read-only predicate:
    /// the trial request is admitted by the same call that observes the
    /// expiry, which is what keeps exactly one trial in flight per cooldown.
    pub(super) fn is_open(&self) -> bool {
        let mut state = self.state.lock();
        let Some(opened_at) = state.opened_at else {
            return false;
        };
        if opened_at.elapsed() < self.config.cooldown {
            return true;
        }
        // Half-open: admit one trial, but leave the count one short of the
        // threshold so that trial failing re-opens the circuit at once.
        state.opened_at = None;
        state.consecutive_failures = self.config.threshold.saturating_sub(1);
        false
    }

    /// Records a local-tier call that answered.
    pub(super) fn record_success(&self) {
        let mut state = self.state.lock();
        state.consecutive_failures = 0;
        state.opened_at = None;
    }

    /// Records a local-tier call that failed, opening the circuit at the threshold.
    pub(super) fn record_failure(&self) {
        let mut state = self.state.lock();
        state.consecutive_failures += 1;
        if state.consecutive_failures >= self.config.threshold {
            state.opened_at = Some(Instant::now());
        }
    }
}

/// Whether a failed local call says anything about the endpoint's health.
///
/// A context-window overflow is a property of the request, not of the tier: the
/// endpoint answered correctly that this request does not fit. Counting it would
/// let a run of oversized requests open the circuit on a healthy endpoint.
pub(super) fn indicates_endpoint_failure(error: &LibsyError) -> bool {
    !matches!(
        error,
        LibsyError::ClientCall {
            source: switchyard_protocol::LlmClientError::ContextWindowExceeded { .. },
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmClientError, ModelId};

    /// A breaker that opens on two failures, with a cooldown short enough to wait out.
    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(BreakerConfig {
            threshold: 2,
            cooldown: Duration::from_millis(20),
        })
    }

    #[test]
    fn a_lossy_endpoint_never_trips_the_breaker() {
        // Only consecutive failures count, so alternating outcomes must not open
        // the circuit however long they go on.
        let breaker = breaker();
        for _ in 0..10 {
            breaker.record_failure();
            breaker.record_success();
        }
        assert!(!breaker.is_open());
    }

    #[test]
    fn the_circuit_opens_at_the_threshold_and_admits_one_trial_after_the_cooldown() {
        let breaker = breaker();
        breaker.record_failure();
        assert!(!breaker.is_open(), "one failure is below the threshold");
        breaker.record_failure();
        assert!(breaker.is_open());

        std::thread::sleep(Duration::from_millis(30));
        // The trial is admitted once...
        assert!(!breaker.is_open());
        // ...and a single failure of that trial re-opens the circuit, rather
        // than the count starting over from zero.
        breaker.record_failure();
        assert!(breaker.is_open());
    }

    #[test]
    fn a_successful_trial_closes_the_circuit() {
        let breaker = breaker();
        breaker.record_failure();
        breaker.record_failure();
        std::thread::sleep(Duration::from_millis(30));
        assert!(!breaker.is_open());
        breaker.record_success();
        breaker.record_failure();
        assert!(!breaker.is_open(), "the count restarted from the success");
    }

    #[test]
    fn an_oversized_request_is_not_evidence_about_the_endpoint() {
        let overflow = LibsyError::ClientCall {
            target: ModelId::new("local"),
            source: LlmClientError::ContextWindowExceeded {
                model: ModelId::new("local"),
                message: "too long".to_string(),
            },
        };
        assert!(!indicates_endpoint_failure(&overflow));
        assert!(indicates_endpoint_failure(&LibsyError::NoTargets));
    }

    #[test]
    fn a_kill_switch_is_shared_by_its_clones() {
        // The operator's handle and the router's handle must be the same flag.
        let operator = KillSwitch::new();
        let router = operator.clone();
        assert!(!router.is_engaged());
        operator.engage();
        assert!(router.is_engaged());
        operator.release();
        assert!(!router.is_engaged());
    }
}

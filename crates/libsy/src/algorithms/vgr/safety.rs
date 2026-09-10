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

/// Mutable breaker state, held behind one lock so the fields cannot disagree.
#[derive(Debug, Default)]
struct BreakerState {
    /// Local-tier failures since the last success.
    consecutive_failures: u32,
    /// When the circuit opened, if it is open.
    opened_at: Option<Instant>,
    /// Whether a half-open trial has been admitted and has not reported back.
    ///
    /// Without this, every caller arriving after the cooldown expires would be
    /// admitted at once, so a dead endpoint would be hit by the whole concurrent
    /// load rather than by one probe.
    trial_in_flight: bool,
}

/// Stops calling an endpoint that has failed repeatedly.
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

    /// Whether the endpoint should be skipped for this turn.
    ///
    /// Admitting the half-open trial is a write, so this is not a read-only
    /// predicate: the same call that observes the cooldown expiring claims the
    /// trial, and every concurrent caller keeps seeing an open circuit until
    /// that trial reports back through [`CircuitBreaker::record_success`] or
    /// [`CircuitBreaker::record_failure`].
    pub(super) fn is_open(&self) -> bool {
        let mut state = self.state.lock();
        if state.opened_at.is_none() {
            return false;
        }
        // The circuit stays open while a trial is out, however long ago it
        // started: one probe at a time, not one per cooldown period.
        if state.trial_in_flight {
            return true;
        }
        let cooled = state
            .opened_at
            .is_some_and(|opened_at| opened_at.elapsed() >= self.config.cooldown);
        if !cooled {
            return true;
        }
        state.trial_in_flight = true;
        false
    }

    /// Records a local-tier call that answered.
    pub(super) fn record_success(&self) {
        let mut state = self.state.lock();
        state.consecutive_failures = 0;
        state.opened_at = None;
        state.trial_in_flight = false;
    }

    /// Records a local-tier call that failed, opening the circuit at the threshold.
    ///
    /// A failure while open is a failed half-open trial, which restarts the
    /// cooldown rather than admitting another probe immediately.
    pub(super) fn record_failure(&self) {
        let mut state = self.state.lock();
        state.consecutive_failures += 1;
        state.trial_in_flight = false;
        if state.consecutive_failures >= self.config.threshold {
            state.opened_at = Some(Instant::now());
        }
    }
}

/// Whether a failed local call says anything about the endpoint's health.
///
/// Only transport failures, timeouts, and gateway-unavailable responses count.
/// Request, capability, configuration, and other completed client failures prove
/// that the endpoint answered; cancellation drops the call future before this
/// predicate can record an outcome.
pub(super) fn indicates_endpoint_failure(error: &LibsyError) -> bool {
    indicates_transport_unavailability(error)
}

/// Whether a turn-judge failure proves that its local endpoint is unavailable.
///
/// Unlike the terminal commit gate, in-flight verification is fail-open on a
/// judge error. Only transport, timeout, and gateway-unavailable failures are
/// strong enough to override that and latch cloud.
pub(super) fn indicates_transport_unavailability(error: &LibsyError) -> bool {
    match error {
        LibsyError::CircuitOpen { .. } | LibsyError::VgrTiersUnavailable { .. } => true,
        LibsyError::ClientCall { source, .. } => match source {
            switchyard_protocol::LlmClientError::Transport { .. }
            | switchyard_protocol::LlmClientError::Timeout { .. } => true,
            switchyard_protocol::LlmClientError::UpstreamHttp { status, .. } => matches!(
                *status,
                http::StatusCode::BAD_GATEWAY
                    | http::StatusCode::SERVICE_UNAVAILABLE
                    | http::StatusCode::GATEWAY_TIMEOUT
            ),
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmClientError, ModelId};

    fn client_call(source: LlmClientError) -> LibsyError {
        LibsyError::ClientCall {
            target: ModelId::new("local"),
            source,
        }
    }

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
    fn only_one_caller_is_admitted_per_half_open_trial() {
        // The point of half-open is to probe with a single request. If every
        // caller arriving after the cooldown were admitted, a dead endpoint
        // would take the full concurrent load once per cooldown.
        let breaker = breaker();
        breaker.record_failure();
        breaker.record_failure();
        std::thread::sleep(Duration::from_millis(30));

        assert!(!breaker.is_open(), "the first caller takes the trial");
        for _ in 0..5 {
            assert!(breaker.is_open(), "a trial is already in flight");
        }

        // The trial reporting back is what releases the circuit either way.
        breaker.record_success();
        assert!(!breaker.is_open());
    }

    #[test]
    fn a_failed_trial_restarts_the_cooldown_rather_than_admitting_another() {
        let breaker = breaker();
        breaker.record_failure();
        breaker.record_failure();
        std::thread::sleep(Duration::from_millis(30));
        assert!(!breaker.is_open());

        breaker.record_failure();
        // Re-opened, and the fresh cooldown has not elapsed.
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
    fn request_and_configuration_failures_are_not_endpoint_failures() {
        let failures = [
            client_call(LlmClientError::ContextWindowExceeded {
                model: ModelId::new("local"),
                message: "too long".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::BAD_REQUEST,
                body: "invalid client request".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::UNAUTHORIZED,
                body: "invalid credential".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::FORBIDDEN,
                body: "request forbidden".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::REQUEST_TIMEOUT,
                body: "client request timeout".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::TOO_MANY_REQUESTS,
                body: "rate limited".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::BAD_REQUEST,
                body: "image input is not supported by this model".to_string(),
            }),
            client_call(LlmClientError::InvalidRequest {
                message: "unsupported request".to_string(),
            }),
            client_call(LlmClientError::RequestEncoding(
                "unsupported capability".to_string(),
            )),
            client_call(LlmClientError::Configuration {
                message: "missing target configuration".to_string(),
            }),
            LibsyError::NoTargets,
        ];

        for failure in failures {
            assert!(
                !indicates_endpoint_failure(&failure),
                "{failure} must not affect endpoint health"
            );
        }
    }

    #[test]
    fn transport_failures_are_endpoint_failures() {
        let failures = [
            client_call(LlmClientError::Transport {
                source: std::io::Error::other("connection refused").into(),
            }),
            client_call(LlmClientError::Timeout {
                source: std::io::Error::other("request timed out").into(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::BAD_GATEWAY,
                body: "bad gateway".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::SERVICE_UNAVAILABLE,
                body: "unavailable".to_string(),
            }),
            client_call(LlmClientError::UpstreamHttp {
                status: http::StatusCode::GATEWAY_TIMEOUT,
                body: "gateway timeout".to_string(),
            }),
        ];

        for failure in failures {
            assert!(
                indicates_endpoint_failure(&failure),
                "{failure} must affect endpoint health"
            );
        }
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

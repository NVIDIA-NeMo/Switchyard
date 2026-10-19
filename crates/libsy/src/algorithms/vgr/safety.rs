// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use http::StatusCode;
use parking_lot::Mutex;
use switchyard_protocol::LlmClientError;

use crate::LibsyError;

/// Operator-controlled stop for local attempts.
#[derive(Clone, Debug, Default)]
pub struct KillSwitch(Arc<AtomicBool>);

impl KillSwitch {
    /// Creates a released switch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops local attempts for subsequent requests.
    pub fn engage(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Allows local attempts again.
    pub fn release(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    /// Reports whether local attempts are stopped.
    pub fn is_engaged(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Circuit-breaker configuration.
#[derive(Clone, Copy, Debug)]
pub struct BreakerConfig {
    /// Consecutive endpoint failures required to open.
    pub threshold: u32,
    /// Delay before one half-open trial is admitted.
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

#[derive(Default)]
struct BreakerState {
    failures: u32,
    opened_at: Option<Instant>,
    trial_in_flight: bool,
}

pub(super) struct CircuitBreaker {
    config: BreakerConfig,
    state: Mutex<BreakerState>,
}

impl CircuitBreaker {
    pub(super) fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BreakerState::default()),
        }
    }

    pub(super) fn attempt(&self) -> Option<Attempt<'_>> {
        let mut state = self.state.lock();
        let recovery = if let Some(opened_at) = state.opened_at {
            if state.trial_in_flight || opened_at.elapsed() < self.config.cooldown {
                return None;
            }
            state.trial_in_flight = true;
            true
        } else {
            false
        };
        Some(Attempt {
            breaker: self,
            recovery,
            completed: false,
        })
    }

    pub(super) fn success(&self) {
        *self.state.lock() = BreakerState::default();
    }

    pub(super) fn failure(&self) {
        let mut state = self.state.lock();
        state.failures = state.failures.saturating_add(1);
        state.trial_in_flight = false;
        if state.failures >= self.config.threshold {
            state.opened_at = Some(Instant::now());
        }
    }
}

/// Releases a recovery trial if its local call is cancelled or abandoned.
pub(super) struct Attempt<'a> {
    breaker: &'a CircuitBreaker,
    recovery: bool,
    completed: bool,
}

impl Attempt<'_> {
    pub(super) fn success(mut self) {
        self.completed = true;
        self.breaker.success();
    }

    pub(super) fn failure(mut self) {
        self.completed = true;
        self.breaker.failure();
    }
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        if self.recovery && !self.completed {
            self.breaker.state.lock().trial_in_flight = false;
        }
    }
}

/// Errors for which retrying on the capable tier is safe.
pub(super) fn fallback_eligible(error: &LibsyError) -> bool {
    matches!(
        error,
        LibsyError::ClientCall { source, .. } if match source {
            LlmClientError::ContextWindowExceeded { .. }
                | LlmClientError::Transport { .. }
                | LlmClientError::Timeout { .. }
                | LlmClientError::TemporarilyUnavailable => true,
            LlmClientError::UpstreamHttp { status, .. } =>
                matches!(
                    *status,
                    StatusCode::FORBIDDEN
                        | StatusCode::REQUEST_TIMEOUT
                        | StatusCode::TOO_MANY_REQUESTS
                ) || status.is_server_error(),
            _ => false,
        }
    )
}

/// Only endpoint unavailability contributes to breaker health.
pub(super) fn endpoint_failure(error: &LibsyError) -> bool {
    matches!(
        error,
        LibsyError::ClientCall { source, .. } if match source {
            LlmClientError::Transport { .. }
            | LlmClientError::Timeout { .. }
            | LlmClientError::TemporarilyUnavailable => true,
            LlmClientError::UpstreamHttp { status, .. } => matches!(
                *status,
                StatusCode::BAD_GATEWAY
                    | StatusCode::SERVICE_UNAVAILABLE
                    | StatusCode::GATEWAY_TIMEOUT
            ),
            _ => false,
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_trials_are_exclusive_and_can_be_retried_after_failure_or_drop() {
        let breaker = CircuitBreaker::new(BreakerConfig {
            threshold: 1,
            cooldown: Duration::ZERO,
        });
        breaker.failure();
        let trial = breaker.attempt().expect("recovery trial");
        assert!(breaker.attempt().is_none());
        drop(trial);
        breaker.attempt().expect("released trial").failure();
        breaker.attempt().expect("retry after failure").success();
        assert!(breaker.state.lock().opened_at.is_none());
        assert!(breaker.attempt().is_some());
    }

    #[test]
    fn recovery_waits_for_the_cooldown() {
        let breaker = CircuitBreaker::new(BreakerConfig {
            threshold: 1,
            cooldown: Duration::from_secs(30),
        });
        breaker.failure();
        assert!(breaker.attempt().is_none());
    }
}

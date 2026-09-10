// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local-tier stop controls and failure classification.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use http::StatusCode;
use parking_lot::Mutex;
use switchyard_protocol::LlmClientError;

use crate::LibsyError;

/// Shared operator-controlled stop for local attempts.
#[derive(Clone, Debug, Default)]
pub struct KillSwitch(Arc<AtomicBool>);

impl KillSwitch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn engage(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn release(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    pub fn is_engaged(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Consecutive-failure breaker tuning.
#[derive(Clone, Copy, Debug)]
pub struct BreakerConfig {
    pub threshold: u32,
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

#[derive(Debug, Default)]
struct BreakerState {
    failures: u32,
    opened_at: Option<Instant>,
    trial_in_flight: bool,
}

/// Circuit breaker shared by concurrent requests to the local endpoint.
#[derive(Debug)]
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

    pub(super) fn is_open(&self) -> bool {
        let mut state = self.state.lock();
        let Some(opened_at) = state.opened_at else {
            return false;
        };
        if state.trial_in_flight || opened_at.elapsed() < self.config.cooldown {
            return true;
        }
        state.trial_in_flight = true;
        false
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

/// Errors for which retrying on the capable tier is safe.
pub(super) fn fallback_eligible(error: &LibsyError) -> bool {
    matches!(
        error,
        LibsyError::ClientCall { source, .. } if match source {
            LlmClientError::ContextWindowExceeded { .. }
                | LlmClientError::Transport { .. }
                | LlmClientError::Timeout { .. } => true,
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
            LlmClientError::Transport { .. } | LlmClientError::Timeout { .. } => true,
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

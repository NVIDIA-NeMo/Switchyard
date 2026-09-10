// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operator configuration for a verification-gated route.

use std::time::Duration;

use switchyard_protocol::ModelId;

use super::safety::{BreakerConfig, KillSwitch};
use crate::{LibsyError, Result};

/// Required attestation for live local commits.
pub const ACTIVE_APPROVAL: &str = "prospective-validation-and-canary-approved";

/// Authority granted to VGR decisions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ServingMode {
    /// Skip VGR and always use the capable tier.
    #[default]
    Off,
    /// Serve decisions in an isolated evaluation.
    Evaluate,
    /// Compute decisions but always serve the capable tier.
    Shadow,
    /// Serve decisions after explicit operator approval.
    Active {
        /// Exact operator attestation required by [`ACTIVE_APPROVAL`].
        approval: String,
    },
}

/// Model targets used by one VGR route.
#[derive(Clone, Debug)]
pub struct Targets {
    /// Tier that produces the candidate attempt.
    pub local: ModelId,
    /// Tier used when the candidate is not licensed.
    pub cloud: ModelId,
    /// Local verifier, defaulting to `local`.
    pub judge: Option<ModelId>,
    /// Optional capable-tier confirmation model.
    pub cloud_judge: Option<ModelId>,
}

/// Complete runtime configuration.
#[derive(Clone, Debug)]
pub struct VgrConfig {
    /// Completion and verifier targets.
    pub targets: Targets,
    /// Authority granted to routing decisions.
    pub mode: ServingMode,
    /// End-to-end budget for attempt generation and verification.
    pub deadline: Duration,
    /// Optional live stop controlled by the operator.
    pub kill_switch: Option<KillSwitch>,
    /// Local endpoint breaker tuning.
    pub breaker: BreakerConfig,
    /// Whether a cheap typing call selects a verification regime.
    pub task_typing: bool,
}

impl VgrConfig {
    /// Creates an off-by-default route between local and capable tiers.
    pub fn new(local: ModelId, cloud: ModelId) -> Self {
        Self {
            targets: Targets {
                local,
                cloud,
                judge: None,
                cloud_judge: None,
            },
            mode: ServingMode::Off,
            deadline: Duration::from_secs(30),
            kill_switch: None,
            breaker: BreakerConfig::default(),
            task_typing: true,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.deadline.is_zero() || self.breaker.threshold == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "vgr deadline and breaker threshold must be non-zero".into(),
            });
        }
        if let ServingMode::Active { approval } = &self.mode
            && approval != ACTIVE_APPROVAL
        {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "vgr active mode requires approval attestation {ACTIVE_APPROVAL:?}"
                ),
            });
        }
        Ok(())
    }

    pub(super) fn judge(&self) -> &ModelId {
        self.targets.judge.as_ref().unwrap_or(&self.targets.local)
    }
}

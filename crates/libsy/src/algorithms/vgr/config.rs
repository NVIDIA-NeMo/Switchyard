// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use switchyard_protocol::ModelId;

use super::safety::{BreakerConfig, KillSwitch};
use crate::{LibsyError, Result};

/// Required attestation for live local commits.
pub const ACTIVE_APPROVAL: &str = "prospective-validation-and-canary-approved";

/// Authority granted to VGR decisions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ServingMode {
    /// Always serves cloud without producing an attempt.
    #[default]
    Off,
    /// Serves decisions, for isolated measurement.
    Evaluate,
    /// Decides but always serves cloud.
    Shadow,
    /// Serves decisions after operator approval.
    Active {
        /// Operator approval attestation.
        approval: String,
    },
}

/// Targets used by a VGR route.
#[derive(Clone, Debug)]
pub struct Targets {
    /// Tier that produces the candidate attempt.
    pub local: ModelId,
    /// Tier used when the candidate is not licensed.
    pub cloud: ModelId,
    /// Local verifier, defaulting to `local`.
    pub judge: Option<ModelId>,
    /// Capable-tier verifier; unset removes the cloud confirmation rungs.
    pub cloud_judge: Option<ModelId>,
}

/// VGR runtime configuration.
#[derive(Clone, Debug)]
pub struct VgrConfig {
    /// Completion and verifier targets.
    pub targets: Targets,
    /// Authority granted to routing decisions.
    pub mode: ServingMode,
    /// Budget for one turn's attempt and verification.
    pub deadline: Duration,
    /// Optional live stop controlled by the operator.
    pub kill_switch: Option<KillSwitch>,
    /// Local endpoint breaker tuning.
    pub breaker: BreakerConfig,
    /// Whether a cheap typing call selects a verification regime.
    pub task_typing: bool,
    /// Whether the local tier accepts image content.
    pub local_supports_images: bool,
    /// Lets a long agentic run with earlier tool errors commit on a capable-tier
    /// confirmation once it ends in this many consecutive clean tool results.
    pub confirmed_recovery_min_clean_tail: Option<u32>,
    /// Tells the capable tier, once per user turn, that it inherits unverified
    /// tool-using work.
    pub agentic_handoff: bool,
    /// Condenses local history at handoff. Requires `agentic_handoff`.
    pub compact_handoff: bool,
    /// Wall-clock budget for the local tier within one user turn, including the
    /// client's tool execution.
    pub local_turn_budget: Option<Duration>,
    /// Holds an escalated session on the capable tier past later user turns.
    pub latch_escalation: bool,
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
            local_supports_images: false,
            confirmed_recovery_min_clean_tail: None,
            agentic_handoff: false,
            compact_handoff: false,
            local_turn_budget: None,
            latch_escalation: false,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.compact_handoff && !self.agentic_handoff {
            return Err(LibsyError::AlgorithmError {
                message: "compact_handoff requires agentic_handoff".into(),
            });
        }
        if self.targets.local == self.targets.cloud {
            return Err(LibsyError::AlgorithmError {
                message: "vgr requires distinct local and cloud model IDs".into(),
            });
        }
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

    pub(super) fn confirmed_min_clean_tail(&self) -> Option<i32> {
        self.confirmed_recovery_min_clean_tail
            .map(|tail| i32::try_from(tail.max(1)).unwrap_or(i32::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_requires_handoff_for_direct_callers() {
        let mut config = VgrConfig::new("local".into(), "cloud".into());
        assert!(!config.compact_handoff);
        config.compact_handoff = true;
        assert!(super::super::Vgr::new(config.clone()).is_err());
        config.agentic_handoff = true;
        assert!(super::super::Vgr::new(config).is_ok());
    }

    #[test]
    fn direct_callers_must_use_distinct_local_and_cloud_ids() {
        assert!(super::super::Vgr::new(VgrConfig::new("same".into(), "same".into())).is_err());
        assert!(super::super::Vgr::new(VgrConfig::new("local".into(), "cloud".into())).is_ok());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use switchyard_protocol::ModelId;

use super::safety::{BreakerConfig, KillSwitch};
use crate::{LibsyError, Result};

/// Required attestation for serving efficient-tier answers in production.
pub const ACTIVE_APPROVAL: &str = "prospective-validation-and-canary-approved";

/// Authority granted to VGR decisions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ServingMode {
    /// Always serves the capable tier without producing an attempt.
    #[default]
    Off,
    /// Serves decisions, for isolated measurement.
    Evaluate,
    /// Decides but always serves the capable tier.
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
    pub efficient: ModelId,
    /// Tier used when the candidate is not licensed.
    pub capable: ModelId,
    /// Efficient verifier, defaulting to `efficient`.
    pub judge: Option<ModelId>,
    /// Capable-tier verifier; unset removes the capable confirmation rungs.
    pub capable_judge: Option<ModelId>,
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
    /// Efficient endpoint breaker tuning.
    pub breaker: BreakerConfig,
    /// Whether a cheap typing call selects a verification regime.
    pub task_typing: bool,
    /// Whether the efficient tier accepts image content.
    pub efficient_supports_images: bool,
    /// Lets a long agentic run with earlier tool errors commit on a capable-tier
    /// confirmation once it ends in this many consecutive clean tool results.
    pub confirmed_recovery_min_clean_tail: Option<u32>,
    /// Tells the capable tier, once per user turn, that it inherits unverified
    /// tool-using work.
    pub agentic_handoff: bool,
    /// Condenses history from the efficient tier at handoff. Requires `agentic_handoff`.
    pub compact_handoff: bool,
    /// Wall-clock budget for the efficient tier within one user turn, including the
    /// client's tool execution.
    pub efficient_turn_budget: Option<Duration>,
    /// Holds an escalated session on the capable tier past later user turns.
    /// Shadow mode retains only the current turn's hold.
    pub latch_escalation: bool,
}

impl VgrConfig {
    /// Creates an off-by-default route between efficient and capable tiers.
    pub fn new(efficient: ModelId, capable: ModelId) -> Self {
        Self {
            targets: Targets {
                efficient,
                capable,
                judge: None,
                capable_judge: None,
            },
            mode: ServingMode::Off,
            deadline: Duration::from_secs(30),
            kill_switch: None,
            breaker: BreakerConfig::default(),
            task_typing: true,
            efficient_supports_images: false,
            confirmed_recovery_min_clean_tail: None,
            agentic_handoff: false,
            compact_handoff: false,
            efficient_turn_budget: None,
            latch_escalation: false,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.compact_handoff && !self.agentic_handoff {
            return Err(LibsyError::AlgorithmError {
                message: "compact_handoff requires agentic_handoff".into(),
            });
        }
        if self.targets.efficient == self.targets.capable {
            return Err(LibsyError::AlgorithmError {
                message: "vgr requires distinct efficient and capable model IDs".into(),
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
        self.targets
            .judge
            .as_ref()
            .unwrap_or(&self.targets.efficient)
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
        let mut config = VgrConfig::new("efficient".into(), "capable".into());
        assert!(!config.compact_handoff);
        config.compact_handoff = true;
        assert!(super::super::Vgr::new(config.clone()).is_err());
        config.agentic_handoff = true;
        assert!(super::super::Vgr::new(config).is_ok());
    }

    #[test]
    fn direct_callers_must_use_distinct_efficient_and_capable_ids() {
        assert!(super::super::Vgr::new(VgrConfig::new("same".into(), "same".into())).is_err());
        assert!(
            super::super::Vgr::new(VgrConfig::new("efficient".into(), "capable".into())).is_ok()
        );
    }
}

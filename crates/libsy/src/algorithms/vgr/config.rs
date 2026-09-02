// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operator configuration for a verification-gated route.
//!
//! Everything an operator chooses lives here: which tiers to route between,
//! which verifiers to consult, how long the decision may take, and — most
//! consequentially — whether decisions are allowed to route traffic at all.

use std::sync::Arc;
use std::time::Duration;

use switchyard_protocol::ModelId;

use super::policy::Policy;

/// How much authority a decision has over the traffic it decides.
///
/// The ladder exists because a routing decision and *acting* on that decision
/// are separable, and the gap between them is where a new router earns trust: a
/// deployment can measure what VGR would have done for as long as it likes
/// before letting it do anything.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ServingMode {
    /// Decisions are not made. Every request goes to the capable tier.
    ///
    /// The default, so a route that is configured but not yet consciously
    /// enabled cannot commit anything locally.
    #[default]
    Off,
    /// Decisions are made and reported, and the *decided* route is served.
    ///
    /// For isolated measurement only: readiness gates are not applied, so this
    /// serves routes a production deployment would refuse.
    Evaluate,
    /// Decisions are made and reported, but the capable tier is always served.
    ///
    /// The safe observation mode — full decision records at no routing risk.
    Shadow,
    /// Decisions route traffic, subject to the readiness gates.
    ///
    /// Requires an explicit attestation string, so enabling live local commits
    /// is a deliberate act rather than a config default someone inherited.
    Active {
        /// The operator's recorded approval to serve local commits.
        approval: String,
    },
}

/// An operator-supplied sandboxed checker.
///
/// The checker executes a task's own tests and reports whether they passed,
/// which is the only ground truth in the evidence stack. It runs in this
/// process rather than through the routing host, because it never leaves the
/// local machine and has no reason to make a round trip.
///
/// Implementations must be self-limiting: the router bounds the whole decision
/// by a deadline, but cannot terminate work it did not spawn.
pub trait Checker: Send + Sync {
    /// Runs the task's checks against an attempt, reporting whether they passed.
    ///
    /// `None` means the checks could not be run to a conclusion — a timeout, a
    /// sandbox failure, a missing manifest. It is not a failure verdict, but it
    /// commits nothing either.
    fn check(&self, task_text: &str, attempt: &str) -> Option<bool>;
}

/// The tiers a verification-gated route moves between.
#[derive(Clone, Debug)]
pub struct Targets {
    /// The tier that produces the attempt being verified.
    pub local: ModelId,
    /// The tier a request escalates to when the attempt is not licensed.
    pub cloud: ModelId,
    /// The tier that answers verification questions.
    ///
    /// Defaults to the local tier, since the readout and deliberation rungs are
    /// deliberately cheap and local.
    pub judge: Option<ModelId>,
    /// The tier that answers the cloud confirmation rungs, when configured.
    ///
    /// Leaving this unset removes those rungs, which the decision rules treat as
    /// signals never produced rather than as indeterminate ones.
    pub cloud_judge: Option<ModelId>,
}

/// A complete verification-gated route.
#[derive(Clone)]
pub struct VgrConfig {
    /// The tiers to route between and consult.
    pub targets: Targets,
    /// The decision constants. Defaults to the current policy.
    pub policy: Policy,
    /// How much authority decisions have.
    pub mode: ServingMode,
    /// An operator-configured sandboxed checker, when one is deployed.
    pub checker: Option<Arc<dyn Checker>>,
    /// Whether the operator validated the checker against a pinned manifest.
    ///
    /// Separate from [`VgrConfig::checker`] because possessing a checker and
    /// having verified it is trustworthy are different claims, and only the
    /// second one satisfies the readiness gate.
    pub checker_validated: bool,
    /// Whether the operator declares this surface enforces a schema-validated
    /// terse final-answer format, on which typed agreement is checkable.
    pub structured_answer: bool,
    /// Whether an escalation carries the local attempt forward as context.
    ///
    /// Per-route rather than global: carrying the attempt was measured to help
    /// substantially on research-style work and to hurt on conversational work.
    pub speculation_carry: bool,
    /// The budget for the whole decision, verification included.
    ///
    /// Exceeding it does not fail the request; it ends evidence gathering, and
    /// whatever was not established stays unestablished — which escalates.
    pub deadline: Duration,
}

impl VgrConfig {
    /// A route between two tiers, with everything else at its default.
    pub fn new(local: ModelId, cloud: ModelId) -> Self {
        Self {
            targets: Targets {
                local,
                cloud,
                judge: None,
                cloud_judge: None,
            },
            policy: Policy::CURRENT,
            mode: ServingMode::Off,
            checker: None,
            checker_validated: false,
            structured_answer: false,
            speculation_carry: false,
            deadline: Duration::from_secs(30),
        }
    }

    /// The tier that answers verification questions, defaulting to local.
    pub(super) fn judge_target(&self) -> &ModelId {
        self.targets.judge.as_ref().unwrap_or(&self.targets.local)
    }
}

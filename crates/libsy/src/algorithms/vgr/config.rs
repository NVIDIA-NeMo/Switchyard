// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operator configuration for a verification-gated route.
//!
//! Everything an operator chooses lives here: which tiers to route between,
//! which verifiers to consult, how long the decision may take, and — most
//! consequentially — whether decisions are allowed to route traffic at all.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use switchyard_protocol::ModelId;

use super::policy::Policy;
use super::safety::{BreakerConfig, KillSwitch};
use crate::{LibsyError, Result};

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

/// One invocation of an operator-supplied sandboxed checker.
///
/// The absolute deadline and remaining budget describe the same decision
/// deadline. The manifest identity is pinned by [`ValidatedChecker`], not chosen
/// by a request or by the checker implementation.
#[derive(Clone, Copy, Debug)]
pub struct CheckerRequest<'a> {
    /// The task whose locally produced attempt is being checked.
    pub task_text: &'a str,
    /// The locally produced attempt to check.
    pub attempt: &'a str,
    /// Absolute monotonic deadline for the check.
    pub deadline: Instant,
    /// Budget remaining when the checker was invoked.
    pub remaining: Duration,
    /// Operator-authenticated identity of the immutable check manifest.
    pub manifest_identity: &'a str,
}

/// An asynchronous, cancellation-safe sandboxed checker.
///
/// Dropping the returned future must cancel the check or terminate its isolated
/// worker. VGR also enforces [`CheckerRequest::deadline`] around every call.
#[async_trait]
pub trait Checker: Send + Sync {
    /// Runs the task's checks against an attempt, reporting whether they passed.
    ///
    /// `None` means the checks could not be run to a conclusion — a timeout, a
    /// sandbox failure, a missing manifest. It is not a failure verdict, but it
    /// commits nothing either.
    async fn check(&self, request: CheckerRequest<'_>) -> Option<bool>;
}

/// A checker bound to validation evidence for one pinned manifest.
///
/// Keeping the checker handle and authenticated manifest identity in one value
/// makes it impossible for route configuration to claim checker validation
/// without also supplying the checker that was validated.
#[derive(Clone)]
pub struct ValidatedChecker {
    checker: Arc<dyn Checker>,
    manifest_identity: Arc<str>,
}

impl ValidatedChecker {
    /// Binds a checker to an operator-authenticated immutable manifest identity.
    ///
    /// The host authenticates the identity before construction. Empty,
    /// whitespace-padded, or control-character identities are rejected so the
    /// pinned value is unambiguous when passed to the checker.
    pub fn new(
        checker: Arc<dyn Checker>,
        authenticated_manifest_identity: impl Into<String>,
    ) -> Result<Self> {
        let identity = authenticated_manifest_identity.into();
        if identity.is_empty()
            || identity.trim() != identity
            || identity.chars().any(char::is_control)
        {
            return Err(LibsyError::AlgorithmError {
                message: "vgr checker requires an authenticated manifest identity".to_string(),
            });
        }
        Ok(Self {
            checker,
            manifest_identity: Arc::from(identity),
        })
    }

    /// Runs the bound checker with its pinned manifest identity.
    pub(super) async fn check(
        &self,
        task_text: &str,
        attempt: &str,
        deadline: Instant,
    ) -> Option<bool> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return None;
        }
        let request = CheckerRequest {
            task_text,
            attempt,
            deadline,
            remaining,
            manifest_identity: &self.manifest_identity,
        };
        tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            self.checker.check(request),
        )
        .await
        .ok()
        .flatten()
    }

    pub(super) fn manifest_identity(&self) -> &str {
        &self.manifest_identity
    }
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
    /// Whether the local tier accepts image content.
    ///
    /// False by default because an undeclared local capability must not turn an
    /// image-bearing request into a failed speculative call.
    pub local_supports_images: bool,
    /// The decision constants. Defaults to the current policy.
    pub policy: Policy,
    /// How much authority decisions have.
    pub mode: ServingMode,
    /// A checker bound to its authenticated pinned-manifest evidence.
    pub checker: Option<ValidatedChecker>,
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
    /// An operator handle that stops local commits without rebuilding the route.
    ///
    /// Unset means no runtime stop exists, which is not the same as one that is
    /// never engaged: the operator must hold a handle for there to be anything
    /// to engage.
    pub kill_switch: Option<KillSwitch>,
    /// Tuning for the local-endpoint circuit breaker.
    pub breaker: BreakerConfig,
    /// Whether the router types the request before deriving capabilities.
    ///
    /// Typing costs one cheap local call per turn and is what makes the
    /// answer and conversational regimes reachable at all; abstaining selects
    /// the default regime, which is more conservative rather than weaker.
    pub task_typing: bool,
    /// Whether a session that escalated stays on the capable tier.
    ///
    /// Off by default: holding a session on the capable tier is a cost decision
    /// an operator makes, not one this router should make for them.
    pub latch_escalation: bool,
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
            local_supports_images: false,
            policy: Policy::CURRENT,
            mode: ServingMode::Off,
            checker: None,
            structured_answer: false,
            speculation_carry: false,
            deadline: Duration::from_secs(30),
            kill_switch: None,
            breaker: BreakerConfig::default(),
            task_typing: true,
            latch_escalation: false,
        }
    }

    /// The tier that answers verification questions, defaulting to local.
    pub(super) fn judge_target(&self) -> &ModelId {
        match self.targets.judge.as_ref() {
            Some(judge) => judge,
            None => &self.targets.local,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopChecker;

    #[async_trait]
    impl Checker for NoopChecker {
        async fn check(&self, _request: CheckerRequest<'_>) -> Option<bool> {
            None
        }
    }

    #[test]
    fn checker_validation_binds_a_handle_to_one_manifest() -> Result<()> {
        let checker = ValidatedChecker::new(
            Arc::new(NoopChecker),
            "sha256:0123456789abcdef0123456789abcdef",
        )?;

        assert_eq!(
            checker.manifest_identity(),
            "sha256:0123456789abcdef0123456789abcdef"
        );
        Ok(())
    }

    #[test]
    fn checker_validation_rejects_ambiguous_manifest_identity() {
        for identity in ["", " manifest", "manifest ", "manifest\nother"] {
            assert!(
                ValidatedChecker::new(Arc::new(NoopChecker), identity).is_err(),
                "{identity:?} must not become checker evidence"
            );
        }
    }
}

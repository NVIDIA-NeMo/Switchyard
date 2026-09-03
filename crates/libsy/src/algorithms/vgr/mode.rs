// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a deployment actually serves, given what the rules decided.
//!
//! Deciding and serving are separate steps, and this is the second one. Keeping
//! them apart is what lets a deployment record complete decisions — including
//! the local commits it is not yet willing to make — while serving something
//! more conservative.

use super::config::ServingMode;
use super::decide::{Decision, Route};

/// The route a deployment serves for a decision.
///
/// Only [`ServingMode::Active`] can serve a local commit in production, and it
/// serves the readiness-gated route rather than the raw decision.
/// [`ServingMode::Evaluate`] serves the *ungated* decision, which is why it is
/// restricted to isolated measurement: it will serve routes the gates exist to
/// refuse.
pub(super) fn serve_route(mode: &ServingMode, decision: &Decision) -> Route {
    match mode {
        // No decision authority at all, so nothing local is served.
        ServingMode::Off | ServingMode::Shadow => Route::Cloud,
        ServingMode::Evaluate => decision.route,
        ServingMode::Active { .. } => decision.effective_route,
    }
}

/// The attestation an operator must record to serve live local commits.
pub const ACTIVE_APPROVAL: &str = "prospective-validation-and-canary-approved";

/// Whether a mode is configured coherently.
///
/// Only [`ServingMode::Active`] can be misconfigured: it is the one mode that
/// lets a decision commit locally, so it must carry the operator's recorded
/// approval verbatim. A typo yields an error at construction rather than a
/// route that silently serves cloud forever.
pub(super) fn validate(mode: &ServingMode) -> Result<(), String> {
    match mode {
        ServingMode::Active { approval } if approval != ACTIVE_APPROVAL => Err(format!(
            "vgr active mode requires the approval attestation {ACTIVE_APPROVAL:?}"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::Branch;
    use super::super::decide::ReadinessGate;
    use super::super::rules::Signals;
    use super::*;

    /// A decision with the given decided and readiness-gated routes.
    fn decision(route: Route, effective_route: Route) -> Decision {
        Decision {
            route,
            effective_route,
            readiness_gate: (effective_route != route)
                .then_some(ReadinessGate::SecureCheckerMissing),
            branch: Branch::CodingNoChecks,
            signals: Signals::default(),
            policy_version: "test",
        }
    }

    #[test]
    fn only_active_mode_can_serve_a_local_commit_in_production() {
        // A route configured but never consciously enabled must not commit.
        assert_eq!(ServingMode::default(), ServingMode::Off);
        let committed = decision(Route::Local, Route::Local);
        assert_eq!(serve_route(&ServingMode::Off, &committed), Route::Cloud);
        assert_eq!(serve_route(&ServingMode::Shadow, &committed), Route::Cloud);
        assert_eq!(
            serve_route(
                &ServingMode::Active {
                    approval: ACTIVE_APPROVAL.into()
                },
                &committed
            ),
            Route::Local
        );
    }

    #[test]
    fn active_mode_serves_the_gated_route_and_evaluate_serves_the_ungated_one() {
        // A local decision the readiness gates refuse: active must not serve it,
        // and evaluate deliberately does, which is why it is measurement-only.
        let gated = decision(Route::Local, Route::Cloud);
        assert_eq!(
            serve_route(
                &ServingMode::Active {
                    approval: ACTIVE_APPROVAL.into()
                },
                &gated
            ),
            Route::Cloud
        );
        assert_eq!(serve_route(&ServingMode::Evaluate, &gated), Route::Local);
    }

    #[test]
    fn active_mode_requires_the_approval_attestation_verbatim() {
        assert!(validate(&ServingMode::Off).is_ok());
        assert!(validate(&ServingMode::Shadow).is_ok());
        assert!(
            validate(&ServingMode::Active {
                approval: ACTIVE_APPROVAL.into()
            })
            .is_ok()
        );
        // A near miss is a misconfiguration, not an approval.
        assert!(
            validate(&ServingMode::Active {
                approval: "approved".into()
            })
            .is_err()
        );
        assert!(
            validate(&ServingMode::Active {
                approval: "vgr-active-serving-approved".into()
            })
            .is_err(),
            "the retired approval token must fail closed"
        );
        assert!(
            validate(&ServingMode::Active {
                approval: String::new()
            })
            .is_err()
        );
    }
}

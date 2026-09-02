// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for one verification-gated turn.
//!
//! Driven through the shared mocked-`Driver` scaffold every other algorithm
//! uses, so what is asserted is the behavior a deployment would see: which tier
//! served the turn, which calls were paid for, and what the capable tier was
//! given when the attempt was rejected.

use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{
    ContentBlock, FormatId, LlmResponse, ModelId, PreservationMetadata, Request, Response,
    WireFormat, text_request, text_response,
};

use super::super::config::{Checker, ServingMode, VgrConfig};
use super::super::mode::ACTIVE_APPROVAL;
use crate::core::testing::{ServeResult, test_drive};

const LOCAL: &str = "local-tier";
const CLOUD: &str = "cloud-tier";

/// A request carrying one user turn.
fn request(text: &str) -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), text),
        raw_request: None,
        metadata: None,
    }
}

/// An active route between the two tiers, with verification enabled.
fn active() -> VgrConfig {
    VgrConfig {
        mode: ServingMode::Active {
            approval: ACTIVE_APPROVAL.into(),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    }
}

/// A buffered reply that also reports a readout probability for "yes".
fn reply_with_readout(text: &str, p_yes: f64) -> Response {
    let mut agg = text_response(None, text.to_string());
    let mut preservation = PreservationMetadata::default();
    preservation.responses.insert(
        FormatId::from(WireFormat::OpenAiChat),
        serde_json::json!({"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "yes", "logprob": p_yes.ln()},
            {"token": "no", "logprob": (1.0 - p_yes).ln()}
        ]}]}}]}),
    );
    agg.preservation = preservation;
    Response {
        llm_response: LlmResponse::Agg(agg),
        metadata: None,
    }
}

/// A plain buffered reply.
fn reply(text: &str) -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, text.to_string())),
        metadata: None,
    }
}

/// Records every target called, in order, so cost can be asserted.
#[derive(Clone, Default)]
struct CallLog(Arc<Mutex<Vec<String>>>);

impl CallLog {
    fn targets(&self) -> Vec<String> {
        self.0.lock().clone()
    }
    fn record(&self, target: &ModelId) {
        self.0.lock().push(target.to_string());
    }
}

/// The text of a served response.
async fn served_text(response: Response) -> String {
    let agg = response
        .llm_response
        .into_agg()
        .await
        .expect("response buffers");
    agg.first_output()
        .map(|output| {
            output
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_confident_readout_commits_the_local_attempt_without_a_cloud_call() {
    // The point of the cheap rung: a commit reachable from the readout alone
    // pays for the attempt and the readout, and nothing else.
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));

    let (target, response) = test_drive(route, request("what is the capital?"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = if log.targets().len() == 1 {
                Ok(reply("Paris."))
            } else {
                Ok(reply_with_readout("yes", 0.97))
            };
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(served_text(response).await, "Paris.");
    // The attempt and one readout. No cloud verifier was configured, and none
    // was needed.
    assert_eq!(log.targets(), vec![LOCAL, LOCAL]);
}

#[tokio::test]
async fn a_weak_readout_escalates_to_the_capable_tier() {
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let (target, _) = test_drive(route, request("what is the capital?"), |t: ModelId, _r| async move {
        let result: ServeResult = if t == *LOCAL {
            // The attempt, then a readout well under every bar.
            Ok(reply_with_readout("Paris.", 0.05))
        } else {
            Ok(reply("cloud answer"))
        };
        result
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
}

#[tokio::test]
async fn an_unreadable_verifier_escalates_rather_than_failing_the_request() {
    // A verifier that produces nothing usable is indeterminate evidence, which
    // escalates. It must not surface as an error to the caller.
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let (target, _) = test_drive(route, request("hello"), |t: ModelId, _r| async move {
        let result: ServeResult = if t == *LOCAL {
            // No preserved logprobs at all, so the readout cannot be scored,
            // and the deliberating verifier hedges.
            Ok(reply("an attempt\nMaybe? I cannot say."))
        } else {
            Ok(reply("cloud answer"))
        };
        result
    })
    .await
    .expect("routes without erroring");

    assert_eq!(target, ModelId::from(CLOUD));
}

#[tokio::test]
async fn off_mode_serves_cloud_without_producing_an_attempt() {
    // Off spends nothing: there is no decision to inform, so the local tier is
    // never called at all.
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(
        super::super::Vgr::new(VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD)))
            .expect("builds"),
    );

    let (target, _) = test_drive(route, request("hello"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("cloud answer"));
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![CLOUD]);
}

#[tokio::test]
async fn shadow_mode_produces_the_attempt_but_serves_cloud() {
    // The observation mode still does the work, so a deployment can measure
    // what would have happened; it just does not act on it.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        mode: ServingMode::Shadow,
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route, request("hello"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply_with_readout("an attempt", 0.99));
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    // The local attempt was produced and verified, despite not being served.
    assert!(log.targets().iter().any(|t| t == LOCAL));
}

#[tokio::test]
async fn active_mode_requires_the_approval_attestation() {
    let config = VgrConfig {
        mode: ServingMode::Active {
            approval: "sure".into(),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    assert!(super::super::Vgr::new(config).is_err());
}

#[tokio::test]
async fn an_escalation_carries_the_rejected_attempt_as_unverified_reference() {
    // The local tier's work is not discarded, but the capable tier is told
    // plainly that it was never verified.
    let carried = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&carried);
    let config = VgrConfig {
        speculation_carry: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route, request("solve this"), move |t: ModelId, r: Request| {
        let seen = Arc::clone(&seen);
        async move {
            if t == *CLOUD {
                let text: String = r
                    .llm_request
                    .messages
                    .iter()
                    .flat_map(|m| m.content.iter())
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                *seen.lock() = text;
            }
            let result: ServeResult = if t == *LOCAL {
                Ok(reply_with_readout("my draft solution", 0.01))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    let carried = carried.lock().clone();
    assert!(carried.contains("my draft solution"), "attempt not carried");
    assert!(carried.contains("NOT verified"), "not labelled unverified");
    // The original request survives alongside the carried attempt.
    assert!(carried.contains("solve this"));
}

#[tokio::test]
async fn without_speculation_carry_the_capable_tier_sees_the_original_request() {
    let carried = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&carried);
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));

    test_drive(route, request("solve this"), move |t: ModelId, r: Request| {
        let seen = Arc::clone(&seen);
        async move {
            if t == *CLOUD {
                *seen.lock() = format!("{:?}", r.llm_request.messages);
            }
            let result: ServeResult = if t == *LOCAL {
                Ok(reply_with_readout("my draft solution", 0.01))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        }
    })
    .await
    .expect("routes");

    assert!(!carried.lock().contains("my draft solution"));
}

/// A checker that reports a fixed verdict.
struct FixedChecker(Option<bool>);

impl Checker for FixedChecker {
    fn check(&self, _task_text: &str, _attempt: &str) -> Option<bool> {
        self.0
    }
}

#[tokio::test]
async fn a_checker_pass_commits_and_spends_nothing_on_verifiers() {
    // The sandboxed checker is ground truth: it supersedes every other rung on
    // its branch, so no verifier call is made at all.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        checker: Some(Arc::new(FixedChecker(Some(true)))),
        checker_validated: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route, request("fix the build"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("a patch"));
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(LOCAL));
    // Only the attempt was paid for.
    assert_eq!(log.targets(), vec![LOCAL]);
}

#[tokio::test]
async fn a_checker_that_cannot_run_commits_nothing() {
    // An indeterminate checker is not a pass. The branch has no other rung to
    // fall back on, so the turn escalates.
    let config = VgrConfig {
        checker: Some(Arc::new(FixedChecker(None))),
        checker_validated: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route, request("fix the build"), |t: ModelId, _r| async move {
        let result: ServeResult = if t == *LOCAL {
            Ok(reply("a patch"))
        } else {
            Ok(reply("cloud answer"))
        };
        result
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
}

#[tokio::test]
async fn a_validated_checker_is_required_before_a_coding_commit_is_served() {
    // The readiness gate: a coding attempt may be decided local, but without a
    // validated checker the deployment is not allowed to serve that decision.
    let config = VgrConfig {
        checker: Some(Arc::new(FixedChecker(Some(true)))),
        checker_validated: false,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route, request("fix the build"), |t: ModelId, _r| async move {
        let result: ServeResult = if t == *LOCAL {
            Ok(reply("a patch"))
        } else {
            Ok(reply("cloud answer"))
        };
        result
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
}

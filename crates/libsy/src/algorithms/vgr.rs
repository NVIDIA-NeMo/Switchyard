// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verification-gated routing: capability derivation and branch selection.
//!
//! The router decides whether to commit a locally produced attempt or escalate
//! it, based on evidence about that specific attempt. This module holds the
//! decision core: everything that is pure and does no I/O, so that what the
//! router concludes is separable from the calls that gathered the evidence.
//!
//! Three stages, in order. [`derive_capabilities`] builds the [`Capabilities`]
//! the decision core is allowed to see. [`select_branch`] picks the verification
//! regime those capabilities license. [`decide::decide_from_signals`] applies
//! that regime's rule to the evidence and returns a decision. The rules
//! themselves live in [`rules`], and the constants they read in [`policy`].
//!
//! # Trust model
//!
//! [`Capabilities`] are **never client-declared**, and never read from
//! client-supplied structure such as tool schemas or third-party classifier
//! output. [`derive_capabilities`] builds them from exactly four sources:
//!
//! 1. **Operator route configuration** — whether a checker is configured.
//! 2. **The router's own typing of the request** — a [`TaskType`] the router
//!    produced itself. Anything outside the known set is an abstention.
//! 3. **Locally produced attempt evidence** — the attempt this router generated,
//!    and only that attempt.
//! 4. **Request-derived tool summaries** — the native runtime deliberately
//!    trusts normalized transcript tool results as Host evidence. A client that
//!    can submit conversation history can therefore authorize an agentic commit
//!    by reporting a clean tool record.
//!
//! The derivation lattice is **monotone**: no input reachable by a client
//! selects a weaker verification regime than the default one. Evidence found in
//! the attempt can only harden the branch. Unsupported content and incomplete
//! context fail closed, to capabilities that select [`Branch::Unknown`].

#![allow(dead_code)]

use std::sync::Arc;

use switchyard_protocol::Request;

use crate::core::algorithm::{Algorithm, Driver};
use crate::core::state::State;
use crate::{Result, RoutingOutcome};

use self::config::VgrConfig;
use self::fall_through::FallThrough;
use crate::algorithms::fall_through;
use crate::algorithms::util::affinity::AffinityRouter;

// Unix process groups and Windows Job Objects preserve the same cancellation
// contract, while platform mutation stamps detect edit-and-restore attempts.
#[cfg(any(unix, windows))]
pub mod checker;
pub mod config;
mod decide;
mod matching;
pub mod mode;
mod policy;
mod readout;
mod render;
mod rules;
mod rungs;
mod runtime;
pub mod safety;
mod telemetry;
mod text;

#[cfg(test)]
mod conformance_tests;
#[cfg(test)]
mod decide_tests;
#[cfg(test)]
mod matching_tests;
#[cfg(test)]
mod tests;

/// A verification-gated route.
///
/// Calls the local tier, gathers evidence about the answer it produced, and
/// either releases that answer or escalates to the capable tier. Composed on the
/// shared fall-through shell like every other algorithm here, with a single
/// classifier: the whole decision is one unit of work, not a cascade of
/// independent recommendations.
pub struct Vgr {
    route: FallThrough<State>,
    local: switchyard_protocol::ModelId,
    cloud: switchyard_protocol::ModelId,
    cloud_breaker: Arc<safety::CircuitBreaker>,
}

impl Vgr {
    /// Builds a verification-gated route.
    ///
    /// Errors when the serving mode is configured incoherently — the one
    /// setting whose misconfiguration would otherwise be silent.
    pub fn new(config: VgrConfig) -> Result<Self> {
        mode::validate(&config.mode)
            .map_err(|message| crate::LibsyError::AlgorithmError { message })?;
        let local = config.targets.local.clone();
        let cloud = config.targets.cloud.clone();
        // A local turn must re-enter VGR so every proposed tool call is verified.
        // Retain only cloud to keep an escalation stable through the current user turn.
        let turn_affinity = Arc::new(
            AffinityRouter::new()
                .with_release_on_user_turn()
                .with_latch_only([cloud.clone()]),
        );
        let latch = config.latch_escalation.then(|| {
            // Retaining only the capable tier is what makes this an escalation
            // latch rather than plain affinity: a local commit leaves the
            // session free to be verified afresh next turn, while an escalation
            // sticks. Registered ahead of the verification classifier, so a
            // latched session does not even pay for the local attempt.
            Arc::new(AffinityRouter::new().with_latch_only([config.targets.cloud.clone()]))
        });
        let local_breaker = Arc::new(safety::CircuitBreaker::new(config.breaker));
        let cloud_breaker = Arc::new(safety::CircuitBreaker::new(config.breaker));
        let classifier = Arc::new(runtime::VgrClassifier {
            config,
            local_breaker,
            cloud_breaker: cloud_breaker.clone(),
        });

        let mut route = FallThrough::new_with_state().with_name("vgr");
        if let Some(latch) = latch {
            route = route.with_processor(latch.clone()).with_classifier(latch);
        }
        route = route
            .with_processor(turn_affinity.clone())
            .with_classifier(turn_affinity);
        Ok(Self {
            route: route.with_classifier(classifier),
            local,
            cloud,
            cloud_breaker,
        })
    }
}

#[async_trait::async_trait]
impl Algorithm for Vgr {
    fn name(&self) -> &str {
        "vgr"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let mut outcome = self.route.execute(driver.clone(), request).await?;
        let selected = outcome.selected_model_id()?.clone();
        telemetry::annotate_retained_route(
            &mut outcome.request,
            if selected == self.local {
                decide::Route::Local
            } else {
                decide::Route::Cloud
            },
        );
        if selected == self.cloud {
            // A cloud decision is terminal. Falling backward to local would
            // bypass the verification decision that selected cloud.
            outcome.selected_model_ids.truncate(1);
            if outcome.response.is_none() {
                outcome.response = Some(
                    runtime::complete_cloud(
                        &driver,
                        &outcome.request,
                        &self.cloud,
                        &self.cloud_breaker,
                        None,
                    )
                    .await?,
                );
            }
        } else if selected == self.local && outcome.response.is_none() {
            // Local may fail forward to cloud on the host's eligible-failure
            // policy. Current local commits carry their buffered response, but
            // retain the directional contract if that implementation changes.
            outcome.selected_model_ids = vec![self.local.clone(), self.cloud.clone()];
        }
        Ok(outcome)
    }
}

/// The verification regime a request's capabilities license.
///
/// Ordered by the priority [`select_branch`] applies, strongest evidence first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Branch {
    /// An operator-configured sandboxed checker can execute tests.
    Checks,
    /// Code or test activity, with no checker configured.
    CodingNoChecks,
    /// A single-turn request with a typed final answer to agree against.
    Answer,
    /// Conversational traffic.
    Chat,
    /// An agentic session with an operator-declared prior.
    AgenticRecognized,
    /// An agentic session verified from evidence alone.
    AgenticVerified,
    /// Anything else with an attempt to verify.
    DefaultVerified,
    /// Nothing to verify.
    Unknown,
}

/// A task type the router itself produced by typing the request.
///
/// This is never a client-declared field. Any value the router's typing step
/// cannot resolve to one of these is an abstention, represented as `None`, which
/// derives to the default verification regime rather than a weaker one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskType {
    /// Code-producing work.
    Coding,
    /// Tool-operating work.
    Agentic,
    /// Answer-seeking work with a checkable final answer.
    Answer,
    /// Conversational work.
    Chat,
}

/// A reported tool-error count together with its provenance.
///
/// Only [`ToolErrorCount::Host`] — evidence the routing deployment trusts — can
/// authorize a commit. An untrusted report may veto a commit when it reports
/// errors, but never authorize one when it reports none.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolErrorCount {
    /// A count trusted by the routing host.
    Host(i32),
    /// A count from any other provenance.
    Untrusted(i32),
}

impl ToolErrorCount {
    /// Returns the reported number of tool errors.
    pub fn count(self) -> i32 {
        match self {
            Self::Host(count) | Self::Untrusted(count) => count,
        }
    }

    /// Whether the routing host trusts this count.
    pub fn is_host(self) -> bool {
        matches!(self, Self::Host(_))
    }

    /// Low-cardinality provenance label for telemetry.
    pub fn source_label(self) -> &'static str {
        match self {
            Self::Host(_) => "host",
            Self::Untrusted(_) => "untrusted",
        }
    }
}

/// The complete input the decision core sees.
///
/// Every field is derived, never client-declared — see the module-level trust
/// model. An all-default `Capabilities` selects [`Branch::Unknown`], which is the
/// fail-closed outcome.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Capabilities {
    /// The latest user instruction — the only outside datum branching consults.
    pub task_text: Option<String>,
    /// The attempt's final answer, when the request was typed as answer-seeking
    /// and is single-turn.
    pub final_answer: Option<String>,
    /// The judged view of the request and attempt, as rendered for a verifier.
    pub transcript: Option<String>,
    /// An operator-configured sandboxed checker is available for this route.
    pub has_checks: bool,
    /// Typed as coding, or the attempt shows code or test activity.
    pub is_coding: bool,
    /// Typed as conversational, or typed answer-seeking on a multi-turn request.
    pub is_chat: bool,
    /// An operator-declared prior for this surface. Never derived, so
    /// [`derive_capabilities`] always leaves it unset.
    pub prior_local: Option<f64>,
    /// No confident type, but there is an attempt to verify.
    pub default_verified: bool,
    /// The locally produced attempt text. Scoped to the attempt only: request
    /// content must never mint evidence.
    pub attempt: Option<String>,
    /// The operator declared that this surface enforces a schema-validated terse
    /// final-answer format. Never derived.
    pub structured_answer: bool,
    /// Typed as agentic, or the attempt shows tool activity.
    pub is_agentic: bool,
    /// A tool-error count derived from normalized tool-result history.
    pub tool_errors: Option<ToolErrorCount>,
    /// Total tool results in the execution log summarized with `tool_errors`.
    pub tool_results: Option<i32>,
    /// Whether the final result in that execution log was clean.
    pub tool_tail_clean: Option<bool>,
}

/// Selects the verification regime a request's capabilities license.
///
/// Structural and priority-ordered: the first regime whose evidence is present
/// wins. Pure — no I/O, no dependence on anything but its argument.
pub fn select_branch(caps: &Capabilities) -> Branch {
    if caps.has_checks {
        return Branch::Checks;
    }
    if caps.is_coding {
        return Branch::CodingNoChecks;
    }
    if caps.final_answer.is_some() && caps.task_text.is_some() {
        return Branch::Answer;
    }
    if caps.is_chat {
        return Branch::Chat;
    }
    if caps.transcript.is_some() && caps.prior_local.is_some() {
        return Branch::AgenticRecognized;
    }
    if caps.is_agentic && caps.transcript.is_some() {
        return Branch::AgenticVerified;
    }
    if caps.default_verified && caps.transcript.is_some() {
        return Branch::DefaultVerified;
    }
    Branch::Unknown
}

/// Builds [`Capabilities`] from the three trusted sources, failing closed.
///
/// `task_type` must be the router's own typing output, never a client field;
/// `None` is an abstention and derives to the default regime. `attempt` is the
/// text this router generated locally. `tool_errors` carries a tool-execution
/// error count together with its provenance.
///
/// Returns capabilities selecting [`Branch::Unknown`] when the request carries
/// content the router cannot faithfully judge, when there is no user text or no
/// attempt to verify, or when the judged view would silently drop a requirement
/// the verifier needs to see. In each of those cases there is nothing to commit,
/// so the request must not take a local-committing regime.
pub fn derive_capabilities(
    request: &Request,
    attempt: &str,
    checker_configured: bool,
    task_type: Option<TaskType>,
    tool_errors: Option<ToolErrorCount>,
    tool_results: Option<i32>,
    tool_tail_clean: Option<bool>,
) -> Capabilities {
    let (turns, unsupported) = text::turns(request);
    if unsupported {
        // Media or unrecognized provider blocks the router cannot faithfully
        // judge: fail closed with no task text at all.
        return Capabilities::default();
    }

    let task_text = text::user_task_text(&turns);
    if task_text.trim().is_empty() || attempt.trim().is_empty() {
        // Request validation runs before every regime, the checker included: a
        // request with no user text, or no local attempt, has nothing to commit.
        return Capabilities {
            task_text: (!task_text.is_empty()).then(|| task_text.clone()),
            ..Default::default()
        };
    }

    // Beyond this point every regime carries the same derived evidence; only the
    // regime-selecting flags and the judged view differ.
    let observed = Capabilities {
        task_text: Some(task_text.clone()),
        attempt: Some(attempt.to_string()),
        tool_errors,
        tool_results,
        tool_tail_clean,
        ..Default::default()
    };

    if checker_configured {
        return Capabilities {
            transcript: Some(render::render_session(&turns, attempt)),
            has_checks: true,
            // The checker is the evidence; a tool-error count plays no part.
            tool_errors: None,
            tool_results: None,
            tool_tail_clean: None,
            ..observed
        };
    }

    if task_type == Some(TaskType::Answer) && !text::has_assistant_turn(&turns) {
        return Capabilities {
            final_answer: Some(attempt.to_string()),
            transcript: Some(render::render_session(&turns, attempt)),
            tool_errors: None,
            tool_results: None,
            tool_tail_clean: None,
            ..observed
        };
    }

    let has_tool_trajectory = text::has_tool_trajectory(request);
    let agentic_trajectory = task_type == Some(TaskType::Agentic)
        || (has_tool_trajectory && !matches!(task_type, Some(TaskType::Answer | TaskType::Chat)));
    if agentic_trajectory {
        if !render::agentic_context_complete(&turns) {
            return Capabilities {
                task_text: Some(task_text),
                ..Default::default()
            };
        }
        return Capabilities {
            transcript: Some(render::render_agentic_view(request, &turns, attempt)),
            is_agentic: true,
            ..observed
        };
    }

    if text::observed_hardening(attempt) || task_type == Some(TaskType::Coding) {
        if !render::coding_context_complete(&turns) {
            // A coding conversation whose earlier user constraints cannot all be
            // represented is not judged at all — never judge against partial
            // requirements.
            return Capabilities {
                task_text: Some(task_text),
                ..Default::default()
            };
        }
        return Capabilities {
            transcript: Some(render::render_coding_view(&turns, attempt)),
            is_coding: true,
            ..observed
        };
    }

    if !render::session_context_complete(&turns, attempt) {
        // A judged view that would silently drop a system or user requirement
        // never judges.
        return Capabilities {
            task_text: Some(task_text),
            ..Default::default()
        };
    }

    let transcript = Some(render::render_session(&turns, attempt));
    if task_type == Some(TaskType::Agentic) || text::observed_tool_activity(attempt) {
        return Capabilities {
            transcript,
            is_agentic: true,
            ..observed
        };
    }
    if matches!(task_type, Some(TaskType::Chat | TaskType::Answer)) {
        // A typed answer reaching here is multi-turn answer-seeking: the answer
        // regime's instruments are context-blind, so they are invalid here.
        return Capabilities {
            transcript,
            is_chat: true,
            ..observed
        };
    }
    Capabilities {
        transcript,
        default_verified: true,
        ..observed
    }
}

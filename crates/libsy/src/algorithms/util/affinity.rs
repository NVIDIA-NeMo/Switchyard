// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model affinity as a single SDK component.
//!
//! [`AffinityRouter`] retains the first model chosen for a request's stable identity and
//! forces that model on later requests sharing the identity. It is one object that plays
//! both SDK roles, so registering it as a processor and a classifier cannot drift apart:
//!
//! - As a [`Processor`] it *writes* the assignment: [`Event::Decision`] carries the request
//!   whose chosen model is retained, with the first assignment for an identity winning.
//! - As a [`Classifier`] it *reads* the assignment: it scores the retained model with
//!   confidence `1.0`, or returns no scores to abstain.
//!
//! Identity is derived from correlation metadata: by default, a request is keyed by its
//! session, and a sub-agent is keyed more finely by `session + agent` — a subset of its
//! session. [`AffinityRouter::for_subagents`] narrows affinity to explicitly identified
//! child agents, leaving root traffic to later classifiers on every turn.

use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use parking_lot::Mutex;
use switchyard_protocol::{Request, Role};

use crate::core::algorithm::{Driver, RoutingIdentity};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::processor::{Event, Processor};

/// Upper bound on retained assignments, keeping the process-local map from growing
/// without limit; the oldest entry is evicted once the bound is reached.
const MAX_ASSIGNMENTS: usize = 4096;

/// Retains a model per request identity and forces it on later matching requests.
///
/// Register the same instance as both a processor and a classifier; the two roles share
/// the retained assignments through this instance's interior storage. Decision events carry
/// their originating request, so concurrent turns cannot bind one request's identity to
/// another request's selected model.
///
/// [`with_latch_only`](Self::with_latch_only) narrows *which* models are retained — a
/// decision for any other model routes normally but is not latched (the escalation latch:
/// retain only the strong tier, never the weak one).
#[derive(Default)]
pub struct AffinityRouter {
    /// When set, only these models are retained; a decision for any other model is not latched.
    latch_only: Option<HashSet<String>>,
    /// Whether root-session requests should abstain instead of being retained.
    subagents_only: bool,
    /// In absence of headers, use the message hash based fallback key to do task based routing
    message_hash_fallback: bool,
    /// Retained assignments, shared across this router's processor and classifier roles.
    ///
    /// Held on the instance so the two roles share one process-local map through a
    /// single registered [`Arc`](std::sync::Arc); bounded by [`MAX_ASSIGNMENTS`].
    assignments: Mutex<HashMap<RoutingIdentity, String>>,
}

impl AffinityRouter {
    /// Creates a router that latches every decision.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a router that retains assignments only for explicitly identified child agents.
    ///
    /// Root-agent requests always abstain, so a later classifier selects them on every turn.
    pub fn for_subagents() -> Self {
        Self {
            subagents_only: true,
            ..Self::default()
        }
    }

    /// Uses the first user-message text as a fallback key when metadata has no session.
    pub fn with_message_hash_fallback(mut self) -> Self {
        self.message_hash_fallback = true;
        self
    }

    /// Restricts latching to `models`; a decision for any other model routes but is not
    /// retained.
    pub fn with_latch_only(mut self, models: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.latch_only = Some(models.into_iter().map(Into::into).collect());
        self
    }

    /// Whether a decision for `model` should be retained.
    fn should_latch(&self, model: &str) -> bool {
        self.latch_only
            .as_ref()
            .is_none_or(|set| set.contains(model))
    }

    /// Derives the stable identity this router should retain for `request`.
    fn affinity_key(&self, request: &Request) -> Option<RoutingIdentity> {
        if let Some(identity) = RoutingIdentity::from_request(request) {
            return match identity {
                RoutingIdentity::Session(_) if self.subagents_only => None,
                identity => Some(identity),
            };
        }

        // If headers are not present and we are not a subagent, use the message hash based fallback key to do task based routing
        let is_subagent = request
            .metadata
            .as_ref()
            .is_some_and(|metadata| metadata.is_subagent);
        (!self.subagents_only && !is_subagent && self.message_hash_fallback)
            .then(|| {
                first_user_message_hash(request).map(|hash| {
                    tracing::debug!(affinity_key = %hash, "affinity using message hash fallback");
                    RoutingIdentity::Session(hash)
                })
            })
            .flatten()
    }
}

#[async_trait]
impl<S> Processor<S> for AffinityRouter
where
    S: Send + 'static,
{
    async fn process(&self, _state: &mut S, event: Event<'_>) -> crate::Result<()> {
        if let Event::Decision { request, decision } = event
            && let Some(key) = self.affinity_key(request)
        {
            let model = decision.selected_model();
            let mut assignments = self.assignments.lock();
            if self.should_latch(model) && !assignments.contains_key(&key) {
                evict_if_full(&mut assignments);
                assignments.insert(key, model.to_string());
            }
        }
        Ok(())
    }
}

/// Hashes the first user message so later turns retain the initial task's affinity.
/// For benchmarking purpose with harnesses, task instructions are added as a user prompt to the request so we hash the initial user message.
/// TODO: Have not considered multi-modal payloads yet. That needs to be handled separately.
fn first_user_message_hash(request: &Request) -> Option<String> {
    let message = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)?;
    let mut hasher = DefaultHasher::new();
    message.text_content("")?.hash(&mut hasher);
    Some(format!("{:016x}", hasher.finish()))
}

#[async_trait]
impl<S> Classifier<S> for AffinityRouter
where
    S: Send + 'static,
{
    fn target_unavailable(&self, request: &Request, target: &str) {
        let Some(key) = self.affinity_key(request) else {
            return;
        };
        let mut assignments = self.assignments.lock();
        if assignments
            .get(&key)
            .is_some_and(|assigned| assigned == target)
        {
            assignments.remove(&key);
        }
    }

    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> crate::Result<(Classification, Option<switchyard_protocol::Response>)> {
        let Some(key) = self.affinity_key(request) else {
            return Ok((Classification::Scores(Vec::new()), None));
        };
        let assigned = self.assignments.lock().get(&key).cloned();
        Ok((
            Classification::Scores(match assigned {
                Some(target) => vec![Score {
                    confidence: 1.0,
                    target,
                }],
                None => Vec::new(),
            }),
            None,
        ))
    }
}

/// Evicts one arbitrary assignment when the map has reached [`MAX_ASSIGNMENTS`].
fn evict_if_full(assignments: &mut HashMap<RoutingIdentity, String>) {
    if assignments.len() >= MAX_ASSIGNMENTS
        && let Some(evicted) = assignments.keys().next().cloned()
    {
        assignments.remove(&evicted);
    }
}

#[cfg(test)]
#[path = "affinity_tests.rs"]
mod tests;

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gate's mutable ledger: the per-scope review budget behind a
//! reserve/refund interface, plus the per-conversation stall latch — the
//! gate's only shared mutable state, behind one lock.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use parking_lot::Mutex;
use switchyard_protocol::{Request, Role};

use super::BENCH_SESSION_HEADER;

/// Failed consults tolerated per scope before the gate stops consulting;
/// failures refund the budget, so this cap is what bounds a down advisor.
const MAX_FAILED_CONSULTS: u32 = 3;
/// Bounds tracked budget scopes and stall keys; a scope dropped at the bound
/// re-arms like a process restart (rare, harmless).
const MAX_TRACKED_SCOPES: usize = 1_024;

/// Review budget scope; resolution and precedence live in [`budget_scope`].
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) enum ScopeKey {
    Instance,
    Client(String),
    Session(String),
}

/// Per-scope review ledger.
#[derive(Default)]
struct ScopeState {
    reviews: u32,
    failed_consults: u32,
    exhaustion_logged: bool,
}

/// Shared mutable gate state; locked briefly, never across an await.
#[derive(Default)]
struct GateState {
    scopes: HashMap<ScopeKey, ScopeState>,
    stall_fired: HashSet<u64>,
}

/// Per-scope review budget plus the stall checkpoint's per-conversation latch.
pub(super) struct ReviewBudget {
    max_reviews: u32,
    state: Mutex<GateState>,
}

impl ReviewBudget {
    /// A fresh ledger allowing `max_reviews` reviews per scope.
    pub(super) fn new(max_reviews: u32) -> Self {
        Self {
            max_reviews,
            state: Mutex::new(GateState::default()),
        }
    }

    /// Whether the scope's budget or failure cap is spent; logs once per scope.
    pub(super) fn check_exhausted(&self, scope: &ScopeKey) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.scopes.get_mut(scope) else {
            return false;
        };
        let exhausted =
            entry.reviews >= self.max_reviews || entry.failed_consults >= MAX_FAILED_CONSULTS;
        if exhausted && !entry.exhaustion_logged {
            entry.exhaustion_logged = true;
            tracing::info!(
                target: "libsy",
                scope = ?scope,
                "advisor gate: review budget spent; passing through"
            );
        }
        exhausted
    }

    /// Atomically re-checks exhaustion and reserves one review, so concurrent
    /// same-scope requests cannot overdraw `max_reviews`.
    pub(super) fn try_reserve(&self, scope: &ScopeKey) -> bool {
        let mut state = self.state.lock();
        if state.scopes.len() >= MAX_TRACKED_SCOPES && !state.scopes.contains_key(scope) {
            let evict = state
                .scopes
                .keys()
                .find(|key| **key != ScopeKey::Instance)
                .cloned();
            if let Some(key) = evict {
                state.scopes.remove(&key);
            }
        }
        let max_reviews = self.max_reviews;
        let entry = state.scopes.entry(scope.clone()).or_default();
        if entry.reviews >= max_reviews || entry.failed_consults >= MAX_FAILED_CONSULTS {
            return false;
        }
        entry.reviews += 1;
        true
    }

    /// Returns a reserved review after a failed consult and counts the
    /// failure (fail-open and fail-closed paths alike).
    pub(super) fn refund_failure(&self, scope: &ScopeKey) {
        let mut state = self.state.lock();
        let entry = state.scopes.entry(scope.clone()).or_default();
        entry.reviews = entry.reviews.saturating_sub(1);
        entry.failed_consults += 1;
    }

    /// Drops a completed session's ledger entry; the instance scope persists.
    pub(super) fn evict_scope(&self, scope: &ScopeKey) {
        if *scope == ScopeKey::Instance {
            return;
        }
        self.state.lock().scopes.remove(scope);
    }

    /// Atomically latches the stall checkpoint for a conversation key; true
    /// only for the call that set the latch.
    pub(super) fn try_mark_stall_fired(&self, key: u64) -> bool {
        let mut state = self.state.lock();
        if state.stall_fired.contains(&key) {
            return false;
        }
        if state.stall_fired.len() >= MAX_TRACKED_SCOPES {
            let drop = state.stall_fired.iter().next().copied();
            if let Some(key) = drop {
                state.stall_fired.remove(&key);
            }
        }
        state.stall_fired.insert(key)
    }
}

/// Resolves the review budget scope: the benchmark harness header, then the
/// host-resolved session id, then one shared instance scope.
pub(super) fn budget_scope(request: &Request) -> ScopeKey {
    let metadata = request.metadata.as_ref();
    if let Some(value) = metadata
        .and_then(|metadata| metadata.http_headers.as_ref())
        .and_then(|headers| headers.get(BENCH_SESSION_HEADER))
        .and_then(|value| value.to_str().ok())
        && !value.is_empty()
    {
        return ScopeKey::Client(value.to_string());
    }
    if let Some(id) = metadata.and_then(|metadata| metadata.session_id.as_deref())
        && !id.is_empty()
    {
        return ScopeKey::Session(id.to_string());
    }
    ScopeKey::Instance
}

/// Latches the stall checkpoint per conversation: hash of the first user
/// message's text, which is constant across a session's turns.
pub(super) fn stall_key(request: &Request) -> u64 {
    let text = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)
        .and_then(|message| message.text_content("\n"))
        .unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> ScopeKey {
        ScopeKey::Session("s1".to_string())
    }

    #[test]
    fn reserving_spends_the_budget() {
        let budget = ReviewBudget::new(1);
        assert!(!budget.check_exhausted(&scope()));
        assert!(budget.try_reserve(&scope()));
        assert!(budget.check_exhausted(&scope()));
        assert!(!budget.try_reserve(&scope()));
    }

    #[test]
    fn a_refund_reopens_the_budget_until_the_failure_cap() {
        let budget = ReviewBudget::new(1);
        for _ in 0..MAX_FAILED_CONSULTS {
            assert!(budget.try_reserve(&scope()));
            budget.refund_failure(&scope());
        }
        // Every review was refunded, so it is the failure cap that is spent.
        assert!(budget.check_exhausted(&scope()));
        assert!(!budget.try_reserve(&scope()));
    }

    #[test]
    fn eviction_rearms_a_session_scope_but_never_the_instance() {
        let budget = ReviewBudget::new(1);
        assert!(budget.try_reserve(&scope()));
        budget.evict_scope(&scope());
        assert!(budget.try_reserve(&scope()));

        assert!(budget.try_reserve(&ScopeKey::Instance));
        budget.evict_scope(&ScopeKey::Instance);
        assert!(!budget.try_reserve(&ScopeKey::Instance));
    }

    #[test]
    fn the_stall_latch_admits_one_caller_per_key() {
        let budget = ReviewBudget::new(1);
        assert!(budget.try_mark_stall_fired(7));
        assert!(!budget.try_mark_stall_fired(7));
        assert!(budget.try_mark_stall_fired(8));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-session memory of targets that exceeded their context window.

use std::collections::{HashMap, HashSet};

use parking_lot::Mutex;
use switchyard_protocol::{ModelId, Request};

/// Bounds process-local history. Dropping an entry costs one rediscovered overflow.
const MAX_IDENTITIES: usize = 1_024;

/// Overflow history keyed by conversation.
#[derive(Default)]
pub(crate) struct SessionOverflows {
    by_identity: Mutex<HashMap<String, HashSet<ModelId>>>,
}

impl SessionOverflows {
    /// The candidates `identity` has not already overflowed, in the order given.
    /// Never empty while `candidates` is non-empty: a later turn may fit again, so the
    /// caller should get the upstream's answer rather than a routing error.
    pub(crate) fn eligible(&self, identity: Option<&str>, candidates: &[ModelId]) -> Vec<ModelId> {
        let Some(identity) = identity else {
            return candidates.to_vec();
        };
        let history = self.by_identity.lock();
        let Some(overflowed) = history.get(identity) else {
            return candidates.to_vec();
        };
        let eligible: Vec<ModelId> = candidates
            .iter()
            .filter(|candidate| !overflowed.contains(*candidate))
            .cloned()
            .collect();
        if eligible.is_empty() {
            candidates.to_vec()
        } else {
            eligible
        }
    }

    /// Remembers that `target` overflowed for `identity`.
    pub(crate) fn record(&self, identity: Option<&str>, target: &ModelId) {
        let Some(identity) = identity else { return };
        let mut history = self.by_identity.lock();
        if history.len() >= MAX_IDENTITIES
            && !history.contains_key(identity)
            && let Some(victim) = history.keys().next().cloned()
        {
            history.remove(&victim);
        }
        history
            .entry(identity.to_string())
            .or_default()
            .insert(target.clone());
    }
}

/// A root request keyed by session, a child request by session and agent. A child missing
/// either ID is untracked rather than sharing its parent's history.
pub(crate) fn identity(request: &Request) -> Option<String> {
    let metadata = request.metadata.as_ref()?;
    let session = metadata.session_id.as_deref().filter(|id| !id.is_empty())?;
    if metadata.is_subagent {
        let agent = metadata.agent_id.as_deref().filter(|id| !id.is_empty())?;
        Some(format!("{session}/{agent}"))
    } else {
        Some(session.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{Metadata, text_request};

    fn model(name: &str) -> ModelId {
        ModelId::from(name)
    }

    fn request(metadata: Metadata) -> Request {
        Request {
            llm_request: text_request(None, "hi"),
            raw_request: None,
            metadata: Some(metadata),
        }
    }

    #[test]
    fn an_overflowed_target_is_skipped_for_the_rest_of_the_session() {
        let history = SessionOverflows::default();
        let candidates = vec![model("weak"), model("strong")];

        assert_eq!(history.eligible(Some("s1"), &candidates), candidates);
        history.record(Some("s1"), &model("weak"));
        assert_eq!(
            history.eligible(Some("s1"), &candidates),
            vec![model("strong")]
        );
    }

    #[test]
    fn history_is_scoped_to_one_session() {
        let history = SessionOverflows::default();
        let candidates = vec![model("weak"), model("strong")];
        history.record(Some("s1"), &model("weak"));

        assert_eq!(history.eligible(Some("s2"), &candidates), candidates);
        assert_eq!(history.eligible(None, &candidates), candidates);
    }

    #[test]
    fn the_candidate_pool_is_never_emptied() {
        let history = SessionOverflows::default();
        let candidates = vec![model("weak")];
        history.record(Some("s1"), &model("weak"));

        assert_eq!(history.eligible(Some("s1"), &candidates), candidates);
    }

    #[test]
    fn a_subagent_does_not_share_its_parents_history() {
        let parent = request(Metadata {
            session_id: Some("s1".into()),
            ..Default::default()
        });
        let child = request(Metadata {
            session_id: Some("s1".into()),
            agent_id: Some("a1".into()),
            is_subagent: true,
            ..Default::default()
        });

        assert_eq!(identity(&parent).as_deref(), Some("s1"));
        assert_eq!(identity(&child).as_deref(), Some("s1/a1"));
    }

    #[test]
    fn tracking_is_bounded() {
        let history = SessionOverflows::default();
        for n in 0..MAX_IDENTITIES + 10 {
            history.record(Some(&format!("s{n}")), &model("weak"));
        }

        assert!(history.by_identity.lock().len() <= MAX_IDENTITIES);
    }
}

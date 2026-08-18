// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remembers which targets exceeded their context window, per conversation.
//!
//! A conversation only grows, so a target that could not fit one turn will not fit a later
//! one. The host reports them on the next request so routing does not pick one again.

use std::collections::HashMap;

use parking_lot::Mutex;
use switchyard_protocol::{ModelId, Request};

/// Bounds process-local history. Dropping an entry costs one rediscovered overflow, so the
/// victim choice does not need to be exact.
const MAX_CONVERSATIONS: usize = 1_024;

#[derive(Default)]
pub(crate) struct Overflows {
    by_conversation: Mutex<HashMap<String, Vec<ModelId>>>,
}

impl Overflows {
    pub(crate) fn get(&self, conversation: &str) -> Vec<ModelId> {
        self.by_conversation
            .lock()
            .get(conversation)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn record(&self, conversation: &str, target: &ModelId) {
        let mut history = self.by_conversation.lock();
        if history.len() >= MAX_CONVERSATIONS
            && !history.contains_key(conversation)
            && let Some(victim) = history.keys().next().cloned()
        {
            history.remove(&victim);
        }
        let overflowed = history.entry(conversation.to_string()).or_default();
        if !overflowed.contains(target) {
            overflowed.push(target.clone());
        }
    }

    pub(crate) fn forget(&self, conversation: &str) {
        self.by_conversation.lock().remove(conversation);
    }
}

/// A root request is one conversation, a sub-agent is its own, so a child does not inherit
/// its parent's history. Untracked when either id is missing.
pub(crate) fn conversation(request: &Request) -> Option<String> {
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

    fn request(metadata: Metadata) -> Request {
        Request {
            llm_request: text_request(None, "hi"),
            raw_request: None,
            metadata: Some(metadata),
            ineligible_targets: Vec::new(),
        }
    }

    #[test]
    fn overflows_are_remembered_per_conversation() {
        let history = Overflows::default();
        history.record("s1", &ModelId::from("weak"));

        assert_eq!(history.get("s1"), [ModelId::from("weak")]);
        assert!(history.get("s2").is_empty());

        history.forget("s1");
        assert!(history.get("s1").is_empty());
    }

    #[test]
    fn a_subagent_is_its_own_conversation() {
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

        assert_eq!(conversation(&parent).as_deref(), Some("s1"));
        assert_eq!(conversation(&child).as_deref(), Some("s1/a1"));
    }

    #[test]
    fn tracking_is_bounded() {
        let history = Overflows::default();
        for n in 0..MAX_CONVERSATIONS + 10 {
            history.record(&format!("s{n}"), &ModelId::from("weak"));
        }

        assert!(history.by_conversation.lock().len() <= MAX_CONVERSATIONS);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public affinity extension-point contract tests.

use libsy::affinity::{Affinity, AffinityKey, AffinityState};
use libsy::{Metadata, Request};
use switchyard_protocol::text_request;

/// Example policy implemented outside the `libsy` crate.
#[derive(Default)]
struct TaskAffinity {
    state: AffinityState,
}

impl Affinity for TaskAffinity {
    fn key(&self, request: &Request) -> Option<AffinityKey> {
        let metadata = request.metadata.as_ref()?;
        Some(AffinityKey::new(
            "task",
            vec![metadata.session_id.clone()?, metadata.task_id.clone()?],
        ))
    }

    fn state(&self) -> &AffinityState {
        &self.state
    }
}

#[test]
fn external_policy_can_reuse_the_public_affinity_state() {
    let affinity = TaskAffinity::default();
    let request = Request {
        llm_request: text_request(Some("auto".to_string()), "hi"),
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some("session-1".to_string()),
            task_id: Some("task-1".to_string()),
            ..Metadata::default()
        }),
    };

    assert_eq!(affinity.retain(&request, "model-a".to_string()), "model-a");
    assert_eq!(affinity.assignment(&request).as_deref(), Some("model-a"));
}

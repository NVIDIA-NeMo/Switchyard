// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decision questions for assessing a user request or the agent's progress.

use std::collections::BTreeMap;

use serde::Deserialize;
use switchyard_protocol::{DecisionQuestion, Request};

use crate::algorithms::util::affinity::starts_user_turn;

/// The bundled rubric, also usable as a template for a custom TOML file.
pub const DEFAULT_RUBRIC_TOML: &str = include_str!("task_modeling/default.toml");

/// Each TOML key names a signal in the decision response.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rubric {
    /// Questions about the user's goal and requirements.
    pub user: BTreeMap<String, DecisionQuestion>,
    /// Questions about the latest agent action and remaining work.
    pub agent: BTreeMap<String, DecisionQuestion>,
}

impl Default for Rubric {
    fn default() -> Self {
        toml::from_str(DEFAULT_RUBRIC_TOML).expect("bundled task modeling rubric must parse")
    }
}

impl Rubric {
    /// Selects the questions for this request using libsy's user-turn detection.
    pub fn questions_for(&self, request: &Request) -> &BTreeMap<String, DecisionQuestion> {
        if starts_user_turn(request) {
            &self.user
        } else {
            &self.agent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{Message, Role, text_request};

    #[test]
    fn uses_configured_questions_at_each_boundary() {
        let custom: Rubric = toml::from_str(
            r#"
            [user.needs_patch]
            instructions = "Is a patch requested?"
            kind = { type = "boolean", data = {} }
            [agent.tests_passed]
            instructions = "Did the requested tests pass?"
            kind = { type = "boolean", data = {} }
            "#,
        )
        .unwrap();
        for (rubric, user_key, agent_key) in [
            (Rubric::default(), "deliverable", "progress"),
            (custom, "needs_patch", "tests_passed"),
        ] {
            let mut request = Request {
                llm_request: text_request(None, "Fix the parser"),
                raw_request: None,
                metadata: None,
            };
            assert!(rubric.questions_for(&request).contains_key(user_key));
            request
                .llm_request
                .messages
                .push(Message::text(Role::Tool, "parser test failed"));
            let questions = rubric.questions_for(&request);
            assert!(questions.contains_key(agent_key));
            assert!(!questions.contains_key(user_key));
        }
    }
}

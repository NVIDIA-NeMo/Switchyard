// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build a decision request about the current task and interpret its answers.
//!
//! ```no_run
//! # use switchyard_libsy::{Driver, Result, task_modeling::Rubric};
//! # use switchyard_protocol::{ModelId, Request};
//! # async fn example(driver: Driver, request: Request, decision_model: ModelId) -> Result<()> {
//! let rubric = Rubric::default();
//! let decision_request = rubric.questions(&request);
//! let decision_response = driver.call_decision(decision_request, decision_model).await?;
//! let assessment = rubric.interpret(&request, decision_response)?;
//! let signals = assessment.signals();
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use switchyard_protocol::{
    DecisionAnswer, DecisionQuestion, DecisionRequest, DecisionResponse, Request,
};

use crate::algorithms::util::affinity::starts_user_turn;
use crate::{LibsyError, Result};

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

/// A completed assessment with the provider's typed answers and usage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "boundary",
    content = "decision_response",
    rename_all = "snake_case"
)]
pub enum TaskAssessment {
    /// Signals about the incoming user goal.
    User(DecisionResponse),
    /// Signals about the state of the agent's work.
    Agent(DecisionResponse),
}

impl TaskAssessment {
    /// Answers keyed by the signal names in the originating rubric.
    pub fn signals(&self) -> &BTreeMap<String, DecisionAnswer> {
        match self {
            Self::User(response) | Self::Agent(response) => &response.answers,
        }
    }
}

impl Default for Rubric {
    fn default() -> Self {
        toml::from_str(DEFAULT_RUBRIC_TOML).expect("bundled task modeling rubric must parse")
    }
}

impl Rubric {
    /// Builds a decision request from the normalized conversation. The caller selects the model.
    pub fn questions(&self, request: &Request) -> DecisionRequest {
        let questions = if starts_user_turn(request) {
            &self.user
        } else {
            &self.agent
        };
        DecisionRequest {
            model: None,
            input: json!({ "messages": request.llm_request.messages }),
            questions: questions.clone(),
        }
    }

    /// Interprets a decision response against the same request and rubric used to build it.
    pub fn interpret(
        &self,
        request: &Request,
        decision_response: DecisionResponse,
    ) -> Result<TaskAssessment> {
        let (questions, assessment) = if starts_user_turn(request) {
            (&self.user, TaskAssessment::User(decision_response))
        } else {
            (&self.agent, TaskAssessment::Agent(decision_response))
        };
        if !questions.keys().eq(assessment.signals().keys()) {
            return Err(LibsyError::AlgorithmError {
                message:
                    "decision response keys do not match the task modeling rubric for this boundary"
                        .into(),
            });
        }
        Ok(assessment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{
        BooleanEstimate, DecisionValue, Message, Probability, Role, text_request,
    };

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
        let mut request = Request {
            llm_request: text_request(None, "Fix the parser"),
            ..Default::default()
        };
        assert!(
            Rubric::default()
                .questions(&request)
                .questions
                .contains_key("deliverable")
        );
        for (user_turn, key, value) in [
            (
                true,
                "needs_patch",
                DecisionValue::Boolean(BooleanEstimate::ProbabilityTrue(Probability(0.91))),
            ),
            (false, "tests_passed", DecisionValue::Refusal),
        ] {
            let decision_request = custom.questions(&request);
            assert_eq!(decision_request.questions.keys().collect::<Vec<_>>(), [key]);
            assert_eq!(
                decision_request.input["messages"],
                json!(request.llm_request.messages)
            );

            let decision_response = DecisionResponse {
                id: None,
                model: None,
                usage: Default::default(),
                answers: BTreeMap::from([(
                    key.into(),
                    DecisionAnswer {
                        value: value.clone(),
                        provider_confidence: None,
                    },
                )]),
            };
            let assessment = custom
                .interpret(&request, decision_response.clone())
                .unwrap();
            assert_eq!(matches!(assessment, TaskAssessment::User(_)), user_turn);
            assert_eq!(assessment.signals()[key].value, value);

            if user_turn {
                request
                    .llm_request
                    .messages
                    .push(Message::text(Role::Tool, "parser test failed"));
                assert!(custom.interpret(&request, decision_response).is_err());
            }
        }
    }
}

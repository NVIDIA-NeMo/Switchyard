// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability routing with a judge-backed classifier.
//!
//! The classifier judges the full inbound request and emits one decisive target for a
//! [`FallThrough`](super::FallThrough) cascade. Invalid, abstained, or unavailable judge output
//! always selects the capable target.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{LlmRequest, Message, OutputParams, Role};

use super::util::{Judge, JudgeClassifier, JudgeConfig, JudgePolicy};
use crate::{
    Classification, Classifier, Driver, LibsyError, LlmTarget, Request, Result, Score, State,
};

// TODO: As a first implementation, keeping the prompt and schema paths hardcoded. Add a way to dynamically load and parse user passed prompt and schema.
const PROMPT_TEMPLATE: &str = include_str!("../prompts/capability-classifier/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/capability-classifier/schema.json");
// TODO: There can be more knobs to tune the classifier after its verdict is parsed. Add more later.
const THRESHOLD: f64 = 0.5;
const RECENT_MESSAGE_WINDOW: usize = 5;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
/// Parsed judge output. Only confidence, abstention, and solve probability affect v0 routing.
struct TaskClassifierVerdict {
    #[serde(rename = "recommended_route")]
    _recommended_route: String,
    p_solve: f64,
    confidence: f64,
    abstain: bool,
    #[serde(rename = "capability_boundary")]
    _capability_boundary: String,
    #[serde(rename = "primary_rule")]
    _primary_rule: String,
    #[serde(rename = "crux")]
    _crux: String,
}

impl TaskClassifierVerdict {
    /// Reject non-finite or out-of-range probabilities before the policy can route efficiently.
    fn is_valid(&self) -> bool {
        self.p_solve.is_finite()
            && (0.0..=1.0).contains(&self.p_solve)
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

fn task_context_messages(messages: &[Message]) -> Vec<Message> {
    let initial_instruction = messages
        .iter()
        .enumerate()
        .find(|(_, message)| message.role == Role::User);
    let recent_start = messages.len().saturating_sub(RECENT_MESSAGE_WINDOW);
    let mut context = Vec::with_capacity(RECENT_MESSAGE_WINDOW + 1);
    if let Some((_, instruction)) = initial_instruction {
        context.push(instruction.clone());
    }
    context.extend(
        messages
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                *index >= recent_start
                    && Some(*index) != initial_instruction.map(|(index, _)| index)
            })
            .map(|(_, message)| message.clone()),
    );
    context
}

struct CapabilityJudge {
    config: JudgeConfig,
}

impl CapabilityJudge {
    fn new(config: JudgeConfig) -> Self {
        Self { config }
    }
}

impl Judge for CapabilityJudge {
    type Verdict = TaskClassifierVerdict;

    fn build_request(&self, _state: &State, request: &Request) -> Request {
        // Keep the task's initial instruction plus a bounded recent context for each judgment.
        let mut messages = Vec::with_capacity(RECENT_MESSAGE_WINDOW + 2);
        messages.push(Message::text(
            Role::System,
            self.config.system_prompt.clone(),
        ));
        messages.extend(task_context_messages(&request.llm_request.messages));
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                messages,
                output: OutputParams {
                    response_format: self.config.response_schema.clone(),
                    ..OutputParams::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

struct TaskClassifierPolicy {
    efficient_target: String,
    capable_target: String,
}

impl TaskClassifierPolicy {
    fn new(efficient_target: impl Into<String>, capable_target: impl Into<String>) -> Self {
        Self {
            efficient_target: efficient_target.into(),
            capable_target: capable_target.into(),
        }
    }
}

impl JudgePolicy for TaskClassifierPolicy {
    type Verdict = TaskClassifierVerdict;

    fn classify(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Judge output is untrusted. Anything incomplete, invalid, abstained, or below the
        // threshold routes to the capable target rather than risking an underpowered route.
        let target = match verdict {
            Some(verdict)
                if verdict.is_valid() && !verdict.abstain && verdict.p_solve >= THRESHOLD =>
            {
                &self.efficient_target
            }
            _ => &self.capable_target,
        };
        Classification::Scores(vec![Score {
            target: target.clone(),
            confidence: 1.0,
        }])
    }
}

/// A full-request capability classifier configured with the packaged prompt and schema.
pub struct LlmClassifier {
    classifier: JudgeClassifier<CapabilityJudge, TaskClassifierPolicy>,
}

impl LlmClassifier {
    /// Creates a classifier that selects `efficient_target` only above the fixed solve threshold.
    pub fn new(
        judge_target: LlmTarget,
        efficient_target: impl Into<String>,
        capable_target: impl Into<String>,
    ) -> Result<Self> {
        let judge = CapabilityJudge::new(Self::load_judge_config()?);
        Ok(Self {
            classifier: JudgeClassifier::new(
                judge,
                judge_target,
                TaskClassifierPolicy::new(efficient_target, capable_target),
            ),
        })
    }

    pub fn load_system_prompt() -> Result<String> {
        Ok(Self::load_judge_config()?.system_prompt)
    }

    pub fn load_response_schema() -> Result<Value> {
        Self::load_judge_config()?
            .response_schema
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "capability classifier has no response schema".to_string(),
            })
    }

    /// Loads the judge configuration from the packaged prompt and schema.
    /// TODO: Move towards more generic loading and parsing of config when we multiple prompts and schemas to handle for same algorithm
    fn load_judge_config() -> Result<JudgeConfig> {
        // The response schema is rendered into the prompt and sent as structured-output metadata.
        // One asset therefore defines both the instruction contract and provider enforcement.
        let response_schema: Value =
            serde_json::from_str(SCHEMA_TEMPLATE).map_err(|error| LibsyError::AlgorithmError {
                message: format!("capability response schema is invalid: {error}"),
            })?;
        let prompt_schema = response_schema
            .pointer("/json_schema/schema")
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "capability response schema has no json_schema.schema".to_string(),
            })?;
        let prompt_schema = serde_json::to_string_pretty(prompt_schema).map_err(|error| {
            LibsyError::AlgorithmError {
                message: format!("capability prompt schema could not be rendered: {error}"),
            }
        })?;
        Ok(JudgeConfig {
            system_prompt: PROMPT_TEMPLATE.replace("{{RESPONSE_SCHEMA}}", &prompt_schema),
            response_schema: Some(response_schema),
        })
    }
}

#[async_trait]
impl Classifier for LlmClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Classification> {
        self.classifier.score(state, request, driver).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use switchyard_protocol::{text_response, LlmClientError};

    use crate::{
        Algorithm, Context, LlmResponse, LlmTargetSet, Response, RoutedLlmClient, SharedState,
    };

    fn policy() -> TaskClassifierPolicy {
        TaskClassifierPolicy::new("efficient", "capable")
    }

    fn verdict(
        p_solve: f64,
        confidence: f64,
        abstain: bool,
        capability_boundary: &str,
        primary_rule: &str,
    ) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            _recommended_route: "efficient".to_string(),
            p_solve,
            confidence,
            abstain,
            _capability_boundary: capability_boundary.to_string(),
            _primary_rule: primary_rule.to_string(),
            _crux: "test crux".to_string(),
        }
    }

    fn selected(
        policy: &TaskClassifierPolicy,
        verdict: Option<&TaskClassifierVerdict>,
    ) -> Result<String> {
        policy
            .classify(verdict)
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "policy abstained".to_string(),
            })
    }

    #[derive(Default)]
    struct PerRequestClient {
        calls: Mutex<Vec<String>>,
    }

    impl PerRequestClient {
        fn calls(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    #[async_trait]
    impl RoutedLlmClient for PerRequestClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn crate::Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            let model = decision.selected_model().to_string();
            self.calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(model.clone());
            let completion = if model == "judge" {
                r#"{"recommended_route":"efficient","p_solve":0.9,"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#.to_string()
            } else {
                format!("answer from {model}")
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, completion)),
                metadata: request.metadata,
            })
        }
    }

    #[tokio::test]
    async fn classifier_judges_each_request_without_affinity() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let routed_client: Arc<dyn RoutedLlmClient> = client.clone();
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(routed_client.clone()),
        };
        let judge = target("judge");
        let router = Arc::new(
            super::super::FallThrough::new(LlmTargetSet::new(vec![
                target("efficient"),
                target("capable"),
            ]))
            .with_classifier(Arc::new(LlmClassifier::new(
                judge,
                "efficient",
                "capable",
            )?)),
        );
        let request = || Request {
            llm_request: switchyard_protocol::text_request(
                Some("auto".to_string()),
                "classify this task",
            ),
            raw_request: None,
            metadata: None,
        };

        router
            .clone()
            .run(Context::<SharedState>::default(), request())
            .await?;
        router
            .clone()
            .run(Context::<SharedState>::default(), request())
            .await?;

        assert_eq!(
            client.calls(),
            vec!["judge", "efficient", "judge", "efficient"]
        );
        Ok(())
    }

    #[test]
    fn threshold_policy_uses_the_fixed_threshold() -> Result<()> {
        let policy = policy();
        let at_threshold = verdict(0.5, 0.0, false, "supported", "SUP-1");
        let below_threshold = verdict(0.49, 1.0, false, "supported", "SUP-1");
        assert_eq!(selected(&policy, Some(&at_threshold))?, "efficient");
        assert_eq!(selected(&policy, Some(&below_threshold))?, "capable");
        Ok(())
    }

    #[test]
    fn invalid_or_abstained_verdict_routes_capable() -> Result<()> {
        let policy = policy();
        let invalid_probability = verdict(1.1, 1.0, false, "supported", "SUP-1");
        let abstained = verdict(1.0, 1.0, true, "supported", "SUP-1");
        assert_eq!(selected(&policy, Some(&invalid_probability))?, "capable");
        assert_eq!(selected(&policy, Some(&abstained))?, "capable");
        assert_eq!(selected(&policy, None)?, "capable");
        Ok(())
    }

    #[test]
    fn capability_judge_builds_a_structured_request() -> Result<()> {
        let judge = CapabilityJudge::new(LlmClassifier::load_judge_config()?);
        let request = Request {
            llm_request: LlmRequest {
                model: Some("inbound".to_string()),
                messages: vec![
                    Message::text(Role::System, "client instructions"),
                    Message::text(Role::User, "initial task"),
                    Message::text(Role::Assistant, "old response"),
                    Message::text(Role::User, "old follow-up"),
                    Message::text(Role::Assistant, "recent 1"),
                    Message::text(Role::User, "recent 2"),
                    Message::text(Role::Assistant, "recent 3"),
                    Message::text(Role::User, "recent 4"),
                    Message::text(Role::Assistant, "recent 5"),
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        let judge_request = judge.build_request(&State::default(), &request);

        assert_eq!(judge_request.llm_request.model, request.llm_request.model);
        assert_eq!(
            judge_request.llm_request.messages.len(),
            RECENT_MESSAGE_WINDOW + 2
        );
        let contents = judge_request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>();
        assert!(contents.contains(&"initial task".to_string()));
        assert!(contents.contains(&"recent 1".to_string()));
        assert!(contents.contains(&"recent 5".to_string()));
        assert!(!contents.contains(&"client instructions".to_string()));
        assert!(!contents.contains(&"old response".to_string()));
        assert!(!contents.contains(&"old follow-up".to_string()));
        assert_eq!(
            judge_request.llm_request.output.response_format,
            judge.config.response_schema
        );
        Ok(())
    }

    #[test]
    fn prompt_includes_concrete_rules_and_schema() -> Result<()> {
        let prompt = LlmClassifier::load_system_prompt()?;
        assert!(prompt.contains("SUP-1 [supported]"));
        assert!(!prompt.contains("{{CAPABILITY_RULES}}"));
        assert!(!prompt.contains("{{PRIMARY_RULE_VALUES}}"));
        assert!(!prompt.contains("{{RESPONSE_SCHEMA}}"));
        assert!(prompt.contains("\"type\": \"object\""));
        assert!(!prompt.contains("\"json_schema\""));
        assert!(!prompt.contains("\"CapabilityClassifierDecision\""));
        let schema = LlmClassifier::load_response_schema()?;
        let rule_values = schema
            .pointer("/json_schema/schema/properties/primary_rule/enum")
            .and_then(Value::as_array)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "rendered response schema has no primary rule enum".to_string(),
            })?;
        assert!(rule_values
            .iter()
            .any(|value| value.as_str() == Some("SUP-1")));
        assert!(rule_values
            .iter()
            .any(|value| value.as_str() == Some("none")));
        Ok(())
    }
}

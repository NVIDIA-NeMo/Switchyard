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
    /// Reject out-of-range probabilities before the policy can route efficiently. Range
    /// containment also rejects NaN and the infinities, which compare false against both bounds.
    fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.p_solve) && (0.0..=1.0).contains(&self.confidence)
    }
}

/// Keeps client instructions, the initial task, and bounded recent dialogue for the judge.
fn trim_messages(messages: &[Message], recent_turn_window: usize) -> Vec<Message> {
    let mut system = Vec::new();
    let mut first_user = None;
    let mut first_user_idx = None;
    for (idx, message) in messages.iter().enumerate() {
        match message.role {
            Role::System | Role::Developer => system.push(message.clone()),
            Role::User if first_user.is_none() => {
                first_user = Some(message.clone());
                first_user_idx = Some(idx);
            }
            _ => {}
        }
    }
    let Some(first_user) = first_user else {
        return system;
    };
    let tail = messages
        .iter()
        .enumerate()
        .filter(|(idx, message)| {
            *idx > first_user_idx.unwrap_or(0)
                && !matches!(message.role, Role::System | Role::Developer)
        })
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    if recent_turn_window == 0 {
        let mut out = system;
        out.push(first_user);
        if let Some(last_user) = tail.iter().rev().find(|message| message.role == Role::User) {
            out.push(last_user.clone());
        }
        return out;
    }
    let mut out = system;
    out.push(first_user);
    let start = tail.len().saturating_sub(recent_turn_window);
    out.extend_from_slice(&tail[start..]);
    out
}

struct CapabilityJudge {
    config: JudgeConfig,
}

impl Judge for CapabilityJudge {
    type Verdict = TaskClassifierVerdict;

    fn build_request(&self, _state: &State, request: &Request) -> Request {
        // The judge owns the leading system prompt; client instructions and task context follow.
        let mut messages = trim_messages(&request.llm_request.messages, RECENT_MESSAGE_WINDOW);
        messages.insert(
            0,
            Message::text(Role::System, self.config.system_prompt.clone()),
        );
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
    /// Lowest `p_solve` that still routes to the efficient target.
    threshold: f64,
}

impl TaskClassifierPolicy {
    fn new(
        efficient_target: impl Into<String>,
        capable_target: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            efficient_target: efficient_target.into(),
            capable_target: capable_target.into(),
            threshold,
        }
    }
}

impl JudgePolicy for TaskClassifierPolicy {
    type Verdict = TaskClassifierVerdict;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Judge output is untrusted. Anything incomplete, invalid, abstained, or below the
        // threshold routes to the capable target rather than risking an underpowered route.
        let target = match verdict {
            Some(verdict)
                if verdict.is_valid() && !verdict.abstain && verdict.p_solve >= self.threshold =>
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
pub struct LlmTaskClassifier {
    classifier: JudgeClassifier<CapabilityJudge, TaskClassifierPolicy>,
    efficient_target: String,
    capable_target: String,
}

impl LlmTaskClassifier {
    /// Selects `efficient_target` when the judge's `p_solve` reaches `threshold`, and
    /// `capable_target` otherwise. Errors if `threshold` is outside `[0.0, 1.0]`.
    pub fn new(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        threshold: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!("threshold must be between 0 and 1, got {threshold}"),
            });
        }
        let judge = CapabilityJudge {
            config: Self::load_judge_config()?,
        };
        let efficient_target = efficient_target.semantic_name;
        let capable_target = capable_target.semantic_name;
        Ok(Self {
            classifier: JudgeClassifier::new(
                judge,
                judge_target,
                TaskClassifierPolicy::new(
                    efficient_target.clone(),
                    capable_target.clone(),
                    threshold,
                ),
            ),
            efficient_target,
            capable_target,
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
impl Classifier<State> for LlmTaskClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        if self.efficient_target == self.capable_target {
            None
        } else if selected_model == self.efficient_target {
            Some("weak")
        } else if selected_model == self.capable_target {
            Some("strong")
        } else {
            None
        }
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<Classification> {
        self.classifier.score(state, request, driver).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::*;
    use switchyard_protocol::{completion_text, text_response, LlmClientError};

    use crate::{Algorithm, Context, LlmResponse, LlmTargetSet, Response, RoutedLlmClient};

    const TEST_THRESHOLD: f64 = 0.5;

    fn policy() -> TaskClassifierPolicy {
        TaskClassifierPolicy::new("efficient", "capable", TEST_THRESHOLD)
    }

    /// A verdict whose non-routing fields are fixed — only the three the policy reads vary.
    fn verdict(p_solve: f64, confidence: f64, abstain: bool) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            _recommended_route: "efficient".to_string(),
            p_solve,
            confidence,
            abstain,
            _capability_boundary: "supported".to_string(),
            _primary_rule: "SUP-1".to_string(),
            _crux: "test crux".to_string(),
        }
    }

    fn selected(
        policy: &TaskClassifierPolicy,
        verdict: Option<&TaskClassifierVerdict>,
    ) -> Result<String> {
        policy
            .to_classification(verdict)
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
            self.calls.lock().clone()
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
            self.calls.lock().push(model.clone());
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

    struct UnreachableJudgeClient;

    #[async_trait]
    impl RoutedLlmClient for UnreachableJudgeClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn crate::Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            let model = decision.selected_model().to_string();
            if model == "judge" {
                return Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::other("judge unreachable")),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {model}"))),
                metadata: request.metadata,
            })
        }
    }

    fn router(client: Arc<dyn RoutedLlmClient>) -> Result<Arc<super::super::FallThrough<State>>> {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let targets = LlmTargetSet::new(vec![target("efficient"), target("capable")]);
        let efficient = targets.get_target("efficient")?;
        let capable = targets.get_target("capable")?;
        Ok(Arc::new(
            super::super::FallThrough::<State>::new_with_state(targets).with_classifier(Arc::new(
                LlmTaskClassifier::new(target("judge"), efficient, capable, TEST_THRESHOLD)?,
            )),
        ))
    }

    fn classify_request() -> Request {
        Request {
            llm_request: switchyard_protocol::text_request(
                Some("auto".to_string()),
                "classify this task",
            ),
            raw_request: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn an_unreachable_judge_routes_capable_instead_of_failing_the_request() -> Result<()> {
        let router = router(Arc::new(UnreachableJudgeClient))?;

        let (trace, response) = router.run(Context::default(), classify_request()).await?;

        assert_eq!(trace.last().map(|d| d.selected_model()), Some("capable"));
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from capable".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn classifier_judges_each_request_without_affinity() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let router = router(client.clone())?;
        let request = classify_request;

        router.clone().run(Context::default(), request()).await?;
        router.clone().run(Context::default(), request()).await?;

        assert_eq!(
            client.calls(),
            vec!["judge", "efficient", "judge", "efficient"]
        );
        Ok(())
    }

    #[test]
    fn the_threshold_boundary_is_inclusive() -> Result<()> {
        let policy = policy();
        let at_threshold = verdict(0.5, 0.0, false);
        let below_threshold = verdict(0.49, 1.0, false);
        assert_eq!(selected(&policy, Some(&at_threshold))?, "efficient");
        assert_eq!(selected(&policy, Some(&below_threshold))?, "capable");
        Ok(())
    }

    #[test]
    fn the_threshold_moves_the_routing_boundary() -> Result<()> {
        let borderline = verdict(0.5, 1.0, false);
        let strict = TaskClassifierPolicy::new("efficient", "capable", 0.9);
        let lenient = TaskClassifierPolicy::new("efficient", "capable", 0.1);
        assert_eq!(selected(&strict, Some(&borderline))?, "capable");
        assert_eq!(selected(&lenient, Some(&borderline))?, "efficient");
        Ok(())
    }

    #[test]
    fn an_out_of_range_threshold_is_rejected() -> Result<()> {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: None,
        };
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                LlmTaskClassifier::new(target("judge"), target("e"), target("c"), bad).is_err(),
                "threshold {bad} should be rejected"
            );
        }
        LlmTaskClassifier::new(target("judge"), target("e"), target("c"), 0.0)?;
        LlmTaskClassifier::new(target("judge"), target("e"), target("c"), 1.0)?;
        Ok(())
    }

    #[test]
    fn invalid_or_abstained_verdict_routes_capable() -> Result<()> {
        let policy = policy();
        let invalid_probability = verdict(1.1, 1.0, false);
        let abstained = verdict(1.0, 1.0, true);
        assert_eq!(selected(&policy, Some(&invalid_probability))?, "capable");
        assert_eq!(selected(&policy, Some(&abstained))?, "capable");
        assert_eq!(selected(&policy, None)?, "capable");
        Ok(())
    }

    #[test]
    fn capability_judge_builds_a_structured_request() -> Result<()> {
        let judge = CapabilityJudge {
            config: LlmTaskClassifier::load_judge_config()?,
        };
        let request = Request {
            llm_request: LlmRequest {
                model: Some("inbound".to_string()),
                messages: vec![
                    Message::text(Role::System, "client instructions"),
                    Message::text(Role::Developer, "client developer instructions"),
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
            RECENT_MESSAGE_WINDOW + 4
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
        assert!(contents.contains(&"client instructions".to_string()));
        assert!(contents.contains(&"client developer instructions".to_string()));
        assert!(!contents.contains(&"old response".to_string()));
        assert!(!contents.contains(&"old follow-up".to_string()));
        assert_eq!(
            judge_request.llm_request.output.response_format,
            judge.config.response_schema
        );
        Ok(())
    }

    fn sample_value(spec: &Value) -> Value {
        if let Some(first) = spec
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
        {
            return first.clone();
        }
        match spec.get("type").and_then(Value::as_str) {
            Some("number") => serde_json::json!(0.5),
            Some("boolean") => serde_json::json!(false),
            _ => serde_json::json!("sample"),
        }
    }

    fn schema_shaped_verdict(schema: &Value) -> Result<String> {
        let properties = schema
            .pointer("/json_schema/schema/properties")
            .and_then(Value::as_object)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "packaged schema declares no properties".to_string(),
            })?;
        Ok(Value::Object(
            properties
                .iter()
                .map(|(name, spec)| (name.clone(), sample_value(spec)))
                .collect(),
        )
        .to_string())
    }

    /// Built from the schema so a property added there fails here rather than silently
    /// rejecting every production verdict.
    #[test]
    fn every_schema_property_round_trips_through_the_judge_parser() -> Result<()> {
        let config = LlmTaskClassifier::load_judge_config()?;
        let schema = config
            .response_schema
            .as_ref()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "packaged judge config has no response schema".to_string(),
            })?;
        let reply = schema_shaped_verdict(schema)?;
        let judge = CapabilityJudge {
            config: config.clone(),
        };

        let verdict = judge.parse(&text_response(None, reply))?;

        assert!(verdict.is_valid());
        assert!(!verdict.abstain);
        Ok(())
    }

    #[test]
    fn prompt_includes_concrete_rules_and_schema() -> Result<()> {
        let config = LlmTaskClassifier::load_judge_config()?;
        let prompt = config.system_prompt;
        assert!(prompt.contains("SUP-1 [supported]"));
        assert!(!prompt.contains("{{CAPABILITY_RULES}}"));
        assert!(!prompt.contains("{{PRIMARY_RULE_VALUES}}"));
        assert!(!prompt.contains("{{RESPONSE_SCHEMA}}"));
        assert!(prompt.contains("\"type\": \"object\""));
        assert!(!prompt.contains("\"json_schema\""));
        assert!(!prompt.contains("\"CapabilityClassifierDecision\""));
        let rule_values = config
            .response_schema
            .as_ref()
            .and_then(|schema| schema.pointer("/json_schema/schema/properties/primary_rule/enum"))
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

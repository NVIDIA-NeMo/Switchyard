// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task-level capability routing with a judge-backed classifier.
//!
//! The algorithm owns a [`FallThrough`](super::FallThrough) cascade. Its classifier judges the
//! full inbound request and selects one decisive target. Invalid, abstained, or unavailable judge
//! output always selects the capable target.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{LlmRequest, Message, OutputParams, Role};

use super::util::{AffinityRouter, Judge, JudgeClassifier, JudgeConfig, JudgePolicy};
use super::FallThrough;
use crate::{
    Algorithm, Classification, Classifier, Context, Driver, LibsyError, LlmTarget, LlmTargetSet,
    Request, Response, Result, RoutedLlmClient, Score, State,
};

// TODO: As a first implementation, keeping the prompt and schema paths hardcoded. Add a way to dynamically load and parse user passed prompt and schema.
const PROMPT_TEMPLATE: &str = include_str!("../prompts/capability-classifier/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/capability-classifier/schema.json");
/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "llm_task_classifier";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
/// Parsed judge output. Only confidence, abstention, and solve probability affect v0 routing.
/// These fields are parsed from the judge output and used to route the request.
/// For supporting a new schema, we need to add a new Verdict struct and parse the new
struct TaskClassifierVerdict {
    #[serde(rename = "recommended_route")]
    _recommended_route: String,
    p_solve: f64,
    confidence: f64,
    abstain: bool,
    capability_boundary: String,
    #[serde(rename = "primary_rule")]
    _primary_rule: String,
    #[serde(rename = "crux")]
    _crux: String,
}

impl TaskClassifierVerdict {
    /// Reject out-of-range probabilities before the policy can route efficiently. Range
    /// containment also rejects NaN and the infinities, which compare false against both bounds.
    fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.p_solve)
            && (0.0..=1.0).contains(&self.confidence)
            && matches!(
                self.capability_boundary.as_str(),
                "supported" | "uncertain" | "unsupported" | "unmatched"
            )
    }

    /// Whether this verdict needs the elevated capability threshold.
    /// When capability boundary is "uncertain", "unsupported", or "unmatched", we need to use the elevated capability threshold to route the request to weak model
    /// This is to ensure that we are not routing too many requests to the weak model.
    fn is_capability_elevated(&self) -> bool {
        matches!(
            self.capability_boundary.as_str(),
            "uncertain" | "unsupported" | "unmatched"
        )
    }
}

/// The judge is responsible for any kind of llm judge based calls
/// Example: A judge can be a capability classifier, a escalation classifier etc
/// Builds capability-specific judge requests from shared classifier configuration.
struct CapabilityJudge {
    config: JudgeConfig,
}

impl Judge for CapabilityJudge {
    type Verdict = TaskClassifierVerdict;

    /// For different judges, the request building logic can be different
    /// To have a single interface for all judges, we may make a common request building logic here.
    fn build_request(&self, _state: &State, request: &Request) -> Request {
        // For task-based routing, classify only the newest user message with the judge prompt.
        // For any turn window size setting, use TaskClassifierConfig to define the window size.
        let mut messages = request
            .llm_request
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .cloned()
            .into_iter()
            .collect::<Vec<_>>();
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
    config: TaskClassifierConfig,
}

impl TaskClassifierPolicy {
    fn new(
        efficient_target: impl Into<String>,
        capable_target: impl Into<String>,
        config: TaskClassifierConfig,
    ) -> Self {
        Self {
            efficient_target: efficient_target.into(),
            capable_target: capable_target.into(),
            config,
        }
    }

    /// Returns the required solve probability for one validated verdict.
    fn threshold(&self, verdict: &TaskClassifierVerdict) -> f64 {
        if verdict.is_capability_elevated() {
            self.config
                .capability_elevated_floor
                .unwrap_or(self.config.base_threshold)
        } else {
            self.config.base_threshold
        }
    }
}

impl JudgePolicy for TaskClassifierPolicy {
    type Verdict = TaskClassifierVerdict;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Judge output is untrusted. Anything incomplete, invalid, abstained, or below its
        // configured confidence or capability threshold routes to the capable target.
        let target = match verdict {
            Some(verdict)
                if verdict.is_valid()
                    && !verdict.abstain
                    && verdict.confidence >= self.config.min_confidence
                    && verdict.p_solve >= self.threshold(verdict) =>
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

#[derive(Clone, Debug, Default, Deserialize)]
/// Thresholds that control capability classifier routing.
pub struct TaskClassifierConfig {
    /// Lowest solve probability that routes a supported task to the efficient target.
    pub base_threshold: f64,
    /// Lowest judge confidence that permits efficient routing.
    #[serde(default)]
    pub min_confidence: f64,
    /// Higher solve-probability floor for uncertain, unmatched, and unsupported tasks.
    #[serde(default)]
    pub capability_elevated_floor: Option<f64>,
    /// Enables session affinity before the judge-backed classifier.
    #[serde(default)]
    pub session_affinity: bool,
    /// Uses the first user message as the SessionKey for sticky routing when session metadata is unavailable.
    #[serde(default)]
    pub message_hash_fallback: bool,
}

impl TaskClassifierConfig {
    /// Validates routing thresholds before the classifier is constructed.
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.base_threshold) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "base_threshold must be between 0 and 1, got {}",
                    self.base_threshold
                ),
            });
        }
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "min_confidence must be between 0 and 1, got {}",
                    self.min_confidence
                ),
            });
        }
        if let Some(floor) = self.capability_elevated_floor {
            if !(0.0..=1.0).contains(&floor) {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "capability_elevated_floor must be between 0 and 1, got {floor}"
                    ),
                });
            }
            if floor <= self.base_threshold {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "capability_elevated_floor must be greater than base_threshold, got {floor}"
                    ),
                });
            }
        }
        if self.message_hash_fallback && !self.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        Ok(())
    }
}

struct TaskClassifier {
    classifier: JudgeClassifier<CapabilityJudge, TaskClassifierPolicy>,
    efficient_target: String,
    capable_target: String,
}

/// A task-level capability routing algorithm with an internal fall-through cascade.
pub struct LlmTaskClassifier {
    route: FallThrough<State>,
    classifier: Arc<TaskClassifier>,
}

impl LlmTaskClassifier {
    /// Routes according to `config` and returns errors for invalid thresholds.
    pub fn new(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        config: TaskClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        let judge_config = Self::load_judge_config()?;
        let targets = LlmTargetSet::new(vec![efficient_target.clone(), capable_target.clone()]);
        let session_affinity = config.session_affinity;
        let message_hash_fallback = config.message_hash_fallback;
        let classifier = Arc::new(TaskClassifier {
            classifier: JudgeClassifier::new(
                CapabilityJudge {
                    config: judge_config,
                },
                judge_target,
                TaskClassifierPolicy::new(
                    efficient_target.semantic_name.clone(),
                    capable_target.semantic_name.clone(),
                    config,
                ),
            ),
            efficient_target: efficient_target.semantic_name,
            capable_target: capable_target.semantic_name,
        });

        // The cascade is an internal detail: callers drive the algorithm, not its parts.
        // Affinity comes first so a retained assignment short-circuits the judge call.
        let mut route = FallThrough::<State>::new_with_state(targets).with_name(ALGORITHM_NAME);
        if session_affinity {
            let affinity = if message_hash_fallback {
                AffinityRouter::new().with_message_hash_fallback()
            } else {
                AffinityRouter::new()
            };
            // Both roles must share one `Arc` so the classifier reads what the processor wrote.
            let affinity = Arc::new(affinity);
            route = route
                .with_processor(affinity.clone())
                .with_classifier(affinity);
        }
        Ok(Self {
            route: route.with_classifier(classifier.clone()),
            classifier,
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
impl Classifier<State> for TaskClassifier {
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

#[async_trait]
impl Classifier<State> for LlmTaskClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        self.classifier.routing_tier(selected_model)
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

#[async_trait]
impl Algorithm for LlmTaskClassifier {
    fn name(&self) -> &str {
        "llm_task_classifier"
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.route.count_tokens_client()
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.route.execute(ctx, driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use super::*;
    use switchyard_protocol::{completion_text, text_response, LlmClientError, Metadata};

    use crate::{Algorithm, Context, LlmResponse, Response, RoutedLlmClient};

    const TEST_THRESHOLD: f64 = 0.5;

    fn test_config(base_threshold: f64) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold,
            min_confidence: 0.0,
            capability_elevated_floor: None,
            session_affinity: false,
            message_hash_fallback: false,
        }
    }

    fn policy() -> TaskClassifierPolicy {
        TaskClassifierPolicy::new("efficient", "capable", test_config(TEST_THRESHOLD))
    }

    /// A verdict whose non-routing fields are fixed — only the three the policy reads vary.
    fn verdict(p_solve: f64, confidence: f64, abstain: bool) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            _recommended_route: "efficient".to_string(),
            p_solve,
            confidence,
            abstain,
            capability_boundary: "supported".to_string(),
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

    fn router(client: Arc<dyn RoutedLlmClient>) -> Result<Arc<LlmTaskClassifier>> {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        Ok(Arc::new(LlmTaskClassifier::new(
            target("judge"),
            target("efficient"),
            target("capable"),
            test_config(TEST_THRESHOLD),
        )?))
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

    fn classify_session_request() -> Request {
        Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
            ..classify_request()
        }
    }

    fn classify_follow_up_request() -> Request {
        let mut request = classify_request();
        request
            .llm_request
            .messages
            .push(Message::text(Role::Assistant, "I will add the test."));
        request.llm_request.messages.push(Message::text(
            Role::User,
            "Now run the test suite and report the result.",
        ));
        request
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

    #[tokio::test]
    async fn classifier_config_enables_session_affinity() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(
            target("judge"),
            target("efficient"),
            target("capable"),
            TaskClassifierConfig {
                session_affinity: true,
                ..test_config(TEST_THRESHOLD)
            },
        )?);

        router
            .clone()
            .run(Context::default(), classify_session_request())
            .await?;
        router
            .clone()
            .run(Context::default(), classify_session_request())
            .await?;

        assert_eq!(client.calls(), vec!["judge", "efficient", "efficient"]);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_reuses_message_hash_affinity_for_a_follow_up() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(
            target("judge"),
            target("efficient"),
            target("capable"),
            TaskClassifierConfig {
                session_affinity: true,
                message_hash_fallback: true,
                ..test_config(TEST_THRESHOLD)
            },
        )?);

        router
            .clone()
            .run(Context::default(), classify_request())
            .await?;
        router
            .clone()
            .run(Context::default(), classify_follow_up_request())
            .await?;

        assert_eq!(client.calls(), vec!["judge", "efficient", "efficient"]);
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
        let strict = TaskClassifierPolicy::new("efficient", "capable", test_config(0.9));
        let lenient = TaskClassifierPolicy::new("efficient", "capable", test_config(0.1));
        assert_eq!(selected(&strict, Some(&borderline))?, "capable");
        assert_eq!(selected(&lenient, Some(&borderline))?, "efficient");
        Ok(())
    }

    #[test]
    fn invalid_classifier_config_is_rejected() -> Result<()> {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: None,
        };
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                LlmTaskClassifier::new(
                    target("judge"),
                    target("e"),
                    target("c"),
                    test_config(bad),
                )
                .is_err(),
                "base threshold {bad} should be rejected"
            );
        }
        for config in [
            TaskClassifierConfig {
                base_threshold: 0.5,
                min_confidence: 1.1,
                capability_elevated_floor: None,
                session_affinity: false,
                message_hash_fallback: false,
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                min_confidence: 0.0,
                capability_elevated_floor: Some(0.5),
                session_affinity: false,
                message_hash_fallback: false,
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                min_confidence: 0.0,
                capability_elevated_floor: None,
                session_affinity: false,
                message_hash_fallback: true,
            },
        ] {
            assert!(
                LlmTaskClassifier::new(target("judge"), target("e"), target("c"), config).is_err()
            );
        }
        LlmTaskClassifier::new(target("judge"), target("e"), target("c"), test_config(0.0))?;
        LlmTaskClassifier::new(target("judge"), target("e"), target("c"), test_config(1.0))?;
        Ok(())
    }

    #[test]
    fn invalid_or_abstained_verdict_routes_capable() -> Result<()> {
        let policy = policy();
        let invalid_probability = verdict(1.1, 1.0, false);
        let abstained = verdict(1.0, 1.0, true);
        let invalid_boundary = TaskClassifierVerdict {
            capability_boundary: "unknown".to_string(),
            ..verdict(1.0, 1.0, false)
        };
        assert_eq!(selected(&policy, Some(&invalid_probability))?, "capable");
        assert_eq!(selected(&policy, Some(&abstained))?, "capable");
        assert_eq!(selected(&policy, Some(&invalid_boundary))?, "capable");
        assert_eq!(selected(&policy, None)?, "capable");
        Ok(())
    }

    #[test]
    fn elevated_capability_floor_is_a_targeted_safety_brake() -> Result<()> {
        let policy = TaskClassifierPolicy::new(
            "efficient",
            "capable",
            TaskClassifierConfig {
                capability_elevated_floor: Some(0.45),
                ..test_config(0.25)
            },
        );
        let supported = verdict(0.30, 1.0, false);
        let elevated = TaskClassifierVerdict {
            capability_boundary: "uncertain".to_string(),
            ..verdict(0.30, 1.0, false)
        };
        let strong_elevated = TaskClassifierVerdict {
            capability_boundary: "unsupported".to_string(),
            ..verdict(0.50, 1.0, false)
        };

        assert_eq!(selected(&policy, Some(&supported))?, "efficient");
        assert_eq!(selected(&policy, Some(&elevated))?, "capable");
        assert_eq!(selected(&policy, Some(&strong_elevated))?, "efficient");
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
        assert_eq!(judge_request.llm_request.messages.len(), 2);
        let contents = judge_request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>();
        assert!(contents.contains(&"recent 4".to_string()));
        assert!(!contents.contains(&"initial task".to_string()));
        assert!(!contents.contains(&"recent 5".to_string()));
        assert!(!contents.contains(&"client instructions".to_string()));
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

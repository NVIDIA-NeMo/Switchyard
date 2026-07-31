// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task-level capability routing with a judge-backed classifier.
//!
//! The algorithm owns a [`FallThrough`] cascade. Its classifier judges the
//! full inbound request and selects one decisive target. Invalid, abstained, or unavailable judge
//! output always selects the capable target.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use switchyard_protocol::{LlmRequest, Message, OutputParams, Role, SimpleDecision};

use super::fall_through::{DefaultTarget, FallThrough};
use super::util::affinity::AffinityRouter;
use super::util::escalation::{self, EscalationJudge, EscalationJudgeConfig, EscalationPolicy};
use super::util::llm_judge::{self, Judge, JudgeClassifier, JudgeConfig, JudgePolicy};
use crate::core::algorithm::{Algorithm, Driver, LlmTarget, LlmTargetSet};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};
use switchyard_protocol::{
    AggLlmResponse, Context, LlmResponse, Request, Response, RoutedLlmClient,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/capability-classifier/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/capability-classifier/schema.json");
/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "llm_task_classifier";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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

    /// Whether the capability boundary requires the elevated routing threshold.
    fn is_capability_elevated(&self) -> bool {
        matches!(
            self.capability_boundary.as_str(),
            "uncertain" | "unsupported" | "unmatched"
        )
    }
}

/// Keeps client instructions, the opening task, and the last `recent_turn_window`
/// turns after it. A window of `0` keeps the instructions and the task alone.
///
/// Selects by reference and clones only what survives — a coding-agent
/// conversation carries every tool result, so cloning it whole to keep a window
/// would copy the transcript on each judged turn.
fn trim_messages(messages: &[Message], recent_turn_window: usize) -> Vec<Message> {
    let is_instruction = |message: &Message| matches!(message.role, Role::System | Role::Developer);
    let mut kept: Vec<&Message> = messages.iter().filter(|m| is_instruction(m)).collect();
    let Some(task) = messages.iter().position(|m| m.role == Role::User) else {
        return kept.into_iter().cloned().collect();
    };
    kept.push(&messages[task]);

    let tail: Vec<&Message> = messages[task + 1..]
        .iter()
        .filter(|m| !is_instruction(m))
        .collect();
    kept.extend(&tail[tail.len().saturating_sub(recent_turn_window)..]);
    kept.into_iter().cloned().collect()
}

struct CapabilityJudge {
    config: JudgeConfig,
    recent_turn_window: Option<usize>,
}

impl Judge for CapabilityJudge {
    type Verdict = TaskClassifierVerdict;

    fn build_request(&self, _state: &State, request: &Request) -> Request {
        // Task-based routing judges the newest user message alone. A configured
        // window widens that to the surrounding conversation.
        let mut messages = match self.recent_turn_window {
            Some(window) => trim_messages(&request.llm_request.messages, window),
            None => request
                .llm_request
                .messages
                .iter()
                .rev()
                .find(|message| message.role == Role::User)
                .cloned()
                .into_iter()
                .collect::<Vec<_>>(),
        };
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
        // Judge output is untrusted, so only a complete, valid, non-abstained verdict that
        // clears the configured confidence decides. Anything else is "I could not tell" —
        // reported as ambiguous so the composition around this classifier chooses the
        // fallback, rather than this policy silently imposing one.
        let Some(verdict) = verdict
            .filter(|v| v.is_valid() && !v.abstain && v.confidence >= self.config.min_confidence)
        else {
            return Classification::Ambiguous(vec![]);
        };
        // A usable verdict below the capability threshold is still a decision: the judge
        // does not trust the efficient tier with this task.
        let target = if verdict.p_solve >= self.threshold(verdict) {
            &self.efficient_target
        } else {
            &self.capable_target
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
    /// Trailing conversation turns the judge sees on top of the client
    /// instructions and the opening task.
    ///
    /// `None` (the default) judges the newest user message alone — the task, with
    /// no history. `Some(n)` widens that to the client instructions, the opening
    /// task, and the last `n` turns after it.
    #[serde(default)]
    pub recent_turn_window: Option<usize>,
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

// ── Escalation classifier ──────────────────────────────────────────────────

/// Session-state key holding the consecutive-escalate streak.
const STREAK_KEY: &str = "escalation_streak";

fn streak(state: &State) -> u32 {
    match state.extra.get(STREAK_KEY) {
        Some(StateValue::Count(n)) => *n,
        _ => 0,
    }
}

fn decisive(target: &str) -> Classification {
    Classification::Scores(vec![Score {
        target: target.to_string(),
        confidence: 1.0,
    }])
}

fn assistant_message(response: &AggLlmResponse) -> Message {
    Message {
        role: Role::Assistant,
        content: response
            .first_output()
            .map(|output| output.content.clone())
            .unwrap_or_default(),
    }
}

/// Calls the efficient model, judges its response, and latches to capable once the streak
/// confirms. Returns the efficient response directly when not escalating so the caller does
/// not pay for a second model call.
struct EscalationClassifier {
    judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    capable: LlmTarget,
    efficient: LlmTarget,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        if self.capable.semantic_name == self.efficient.semantic_name {
            None
        } else if selected_model == self.capable.semantic_name {
            Some("strong")
        } else if selected_model == self.efficient.semantic_name {
            Some("weak")
        } else {
            None
        }
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "escalation classifier requires a driver".into(),
            });
        };

        // A confirmed session stays capable without a judge call.
        if streak(state) >= self.confirmations {
            return Ok((decisive(&self.capable.semantic_name), None));
        }

        // Call efficient model and buffer the response so the judge can read it.
        // `Classifier::score` takes no `ctx`, so inner calls use Context::default() and their
        // spans carry algorithm="" rather than the algorithm name. Known gap shared with the
        // task classifier's judge consultation.
        let efficient_response = driver
            .call_llm_target(
                Context::default(),
                &self.efficient,
                request.clone(),
                Arc::new(SimpleDecision {
                    selected_model: self.efficient.semantic_name.clone(),
                    reasoning: Some("escalation classifier: efficient tier".into()),
                }),
            )
            .await?;
        let agg = efficient_response
            .llm_response
            .into_agg()
            .await
            .map_err(|e| LibsyError::AlgorithmError {
                message: format!("failed to aggregate efficient response: {e}"),
            })?;
        // Append the efficient reply so the judge reads this turn's completed trajectory.
        let mut judge_request = request.clone();
        judge_request
            .llm_request
            .messages
            .push(assistant_message(&agg));
        let efficient_response = Response {
            llm_response: if request.llm_request.stream {
                LlmResponse::Stream(agg.into_stream())
            } else {
                LlmResponse::Agg(agg)
            },
            metadata: efficient_response.metadata,
        };

        let (classification, _) = self
            .judge
            .score(state, &mut judge_request, Some(driver))
            .await?;

        let held = streak(state);
        let best = classification.argmax(false)?;
        let (escalate, pending) = match &best {
            Some(score) if score.target == self.capable.semantic_name => (true, held + 1),
            Some(_) => (false, 0),
            None => (false, held),
        };
        state
            .extra
            .insert(STREAK_KEY.to_string(), StateValue::Count(pending));

        if escalate && pending >= self.confirmations {
            // Streak confirmed: drop the efficient response, caller will serve capable.
            return Ok((decisive(&self.capable.semantic_name), None));
        }

        Ok((
            decisive(&self.efficient.semantic_name),
            Some(efficient_response),
        ))
    }
}

/// A task-level capability routing algorithm with an internal fall-through cascade.
pub struct LlmTaskClassifier {
    route: FallThrough<State>,
    /// The active classifier — either capability-based or escalation-based.
    inner: Arc<dyn Classifier<State>>,
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
                    recent_turn_window: config.recent_turn_window,
                },
                judge_target.clone(),
                TaskClassifierPolicy::new(
                    efficient_target.semantic_name.clone(),
                    capable_target.semantic_name.clone(),
                    config,
                ),
            ),
            efficient_target: efficient_target.semantic_name.clone(),
            capable_target: capable_target.semantic_name.clone(),
        });
        let inner: Arc<dyn Classifier<State>> = classifier.clone();

        // Affinity comes first so a retained assignment short-circuits the judge call.
        // Note: when this classifier is embedded inside another cascade (e.g. StageRouter)
        // the affinity processor never fires — only the inner score() is called.
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
        // The judge abstains when it cannot tell; the capable tier catches those turns
        // rather than letting the cascade come back empty-handed.
        let capable_fallback = DefaultTarget::new(classifier.capable_target.clone());
        Ok(Self {
            route: route
                .with_classifier(inner.clone())
                .with_classifier(Arc::new(capable_fallback)),
            inner,
        })
    }

    /// Constructs an escalation variant that calls the efficient model each turn, judges its
    /// response, and latches to the capable tier once the streak confirms.
    ///
    /// Every unlatched turn calls the efficient model, buffers its reply, and consults the
    /// trajectory judge. Once `config.confirmations` consecutive escalate verdicts accumulate
    /// the session latches to the capable tier for its remainder. A judge outage always stays
    /// efficient.
    pub fn new_with_escalation(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        config: EscalationJudgeConfig,
    ) -> Result<Self> {
        let capable_name = capable_target.semantic_name.clone();
        let efficient_name = efficient_target.semantic_name.clone();
        let confirmations = config.confirmations;
        let esc = Arc::new(EscalationClassifier {
            judge: escalation::build_judge(judge_target, capable_name, efficient_name, config)?,
            capable: capable_target.clone(),
            efficient: efficient_target.clone(),
            confirmations,
        });
        let inner: Arc<dyn Classifier<State>> = esc.clone();
        let targets = LlmTargetSet::new(vec![capable_target, efficient_target]);
        Ok(Self {
            route: FallThrough::<State>::new_with_state(targets)
                .with_name(ALGORITHM_NAME)
                .with_classifier(esc),
            inner,
        })
    }

    /// Loads the judge configuration from the packaged prompt and schema.
    fn load_judge_config() -> Result<JudgeConfig> {
        llm_judge::load_judge_config(PROMPT_TEMPLATE, SCHEMA_TEMPLATE)
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
    ) -> Result<(Classification, Option<Response>)> {
        self.classifier.score(state, request, driver).await
    }
}

#[async_trait]
impl Classifier<State> for LlmTaskClassifier {
    fn routing_tier(&self, selected_model: &str) -> Option<&'static str> {
        self.inner.routing_tier(selected_model)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        self.inner.score(state, request, driver).await
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
    use serde_json::Value;

    use super::*;
    use switchyard_protocol::{
        LlmClientError, Metadata, completion_text, text_request, text_response,
    };

    use crate::core::algorithm::Algorithm;
    use switchyard_protocol::{Context, LlmResponse, Response, RoutedLlmClient};

    const TEST_THRESHOLD: f64 = 0.5;

    fn test_config(base_threshold: f64) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold,
            min_confidence: 0.0,
            capability_elevated_floor: None,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
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
            decision: Arc<dyn Decision>,
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
            decision: Arc<dyn Decision>,
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
            llm_request: text_request(Some("auto".to_string()), "classify this task"),
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
                recent_turn_window: None,
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
                recent_turn_window: None,
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                min_confidence: 0.0,
                capability_elevated_floor: Some(0.5),
                session_affinity: false,
                message_hash_fallback: false,
                recent_turn_window: None,
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                min_confidence: 0.0,
                capability_elevated_floor: None,
                session_affinity: false,
                message_hash_fallback: true,
                recent_turn_window: None,
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
    fn an_unusable_verdict_abstains() -> Result<()> {
        // Invalid, abstained, unintelligible, or absent: the judge could not tell,
        // so it declines to decide and leaves the fallback to whoever composed the
        // cascade.
        let policy = policy();
        let invalid_boundary = TaskClassifierVerdict {
            capability_boundary: "unknown".to_string(),
            ..verdict(1.0, 1.0, false)
        };
        let unusable = [
            Some(verdict(1.1, 1.0, false)),
            Some(verdict(1.0, 1.0, true)),
            Some(invalid_boundary),
            None,
        ];
        for verdict in unusable {
            let classification = policy.to_classification(verdict.as_ref());
            assert!(matches!(classification, Classification::Ambiguous(_)));
            assert!(classification.argmax(false)?.is_none());
            assert!(classification.argmax(true)?.is_none());
        }
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

    /// The text of each message a judge with `recent_turn_window` would be sent.
    /// The no-window case is covered by `capability_judge_builds_a_structured_request`.
    fn judged_contents(recent_turn_window: usize) -> Result<Vec<String>> {
        let judge = CapabilityJudge {
            config: LlmTaskClassifier::load_judge_config()?,
            recent_turn_window: Some(recent_turn_window),
        };
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message::text(Role::System, "client instructions"),
                    Message::text(Role::User, "initial task"),
                    Message::text(Role::Assistant, "old response"),
                    Message::text(Role::User, "old follow-up"),
                    Message::text(Role::Assistant, "recent 1"),
                    Message::text(Role::User, "recent 2"),
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        Ok(judge
            .build_request(&State::default(), &request)
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect())
    }

    #[test]
    fn a_window_widens_the_judge_to_the_surrounding_conversation() -> Result<()> {
        // Client instructions and the opening task, plus the last two turns.
        let contents = judged_contents(2)?;
        assert!(contents.contains(&"client instructions".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(contents.contains(&"recent 1".to_string()));
        assert!(contents.contains(&"recent 2".to_string()));
        assert!(!contents.contains(&"old response".to_string()));
        Ok(())
    }

    #[test]
    fn a_zero_window_keeps_only_the_instructions_and_the_task() -> Result<()> {
        let contents = judged_contents(0)?;
        assert!(contents.contains(&"client instructions".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(!contents.contains(&"recent 2".to_string()));
        Ok(())
    }

    #[test]
    fn capability_judge_builds_a_structured_request() -> Result<()> {
        let judge = CapabilityJudge {
            config: LlmTaskClassifier::load_judge_config()?,
            recent_turn_window: None,
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
            recent_turn_window: None,
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
        assert!(
            rule_values
                .iter()
                .any(|value| value.as_str() == Some("SUP-1"))
        );
        assert!(
            rule_values
                .iter()
                .any(|value| value.as_str() == Some("none"))
        );
        Ok(())
    }

    // ── with_escalation tests ──────────────────────────────────────────────

    use std::collections::VecDeque;

    use switchyard_protocol::LlmClientError as ClientError;

    use switchyard_protocol::Decision;

    /// Serves each call with the next queued reply.
    struct QueuedClient {
        replies: Mutex<VecDeque<String>>,
    }

    impl QueuedClient {
        fn new(replies: impl IntoIterator<Item = &'static str>) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(replies.into_iter().map(String::from).collect()),
            })
        }
    }

    #[async_trait]
    impl RoutedLlmClient for QueuedClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            _decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, ClientError> {
            let reply = self
                .replies
                .lock()
                .pop_front()
                .unwrap_or_else(|| "unexpected call".to_string());
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, reply)),
                metadata: request.metadata,
            })
        }
    }

    /// Builds a router with escalation enabled (`confirmations=1` latches on the first verdict).
    fn escalation_router(
        client: Arc<dyn RoutedLlmClient>,
        judge_client: Arc<dyn RoutedLlmClient>,
    ) -> Result<Arc<LlmTaskClassifier>> {
        let target = |name: &str, c: Arc<dyn RoutedLlmClient>| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(c),
        };
        Ok(Arc::new(LlmTaskClassifier::new_with_escalation(
            target("judge", judge_client),
            target("efficient", client.clone()),
            target("capable", client),
            EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
        )?))
    }

    #[tokio::test]
    async fn escalation_router_serves_efficient_when_judge_declines() -> Result<()> {
        // Judge: no escalation. Expect the efficient response to be returned directly.
        let judge_client = QueuedClient::new([r#"{"escalate":false,"reason":"progressing"}"#]);
        let model_client = QueuedClient::new(["efficient answer"]);
        let router = escalation_router(model_client, judge_client)?;
        let request = classify_request();

        let (trace, response) = router.run(Context::default(), request).await?;

        // The efficient model is the serving target, and the response comes from its call.
        assert_eq!(trace.last().map(|d| d.selected_model()), Some("efficient"));
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("efficient answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn escalation_router_upgrades_to_capable_when_judge_escalates() -> Result<()> {
        // Judge: escalate. After the efficient call, the streak confirms and capable is served.
        let judge_client = QueuedClient::new([r#"{"escalate":true,"reason":"stuck in a loop"}"#]);
        // Efficient is called first (by the classifier), then capable is called by FallThrough.
        let model_client = QueuedClient::new(["efficient draft", "capable answer"]);
        let router = escalation_router(model_client, judge_client)?;
        let request = classify_request();

        let (trace, response) = router.run(Context::default(), request).await?;

        assert_eq!(trace.last().map(|d| d.selected_model()), Some("capable"));
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn escalation_router_stays_capable_after_latch() -> Result<()> {
        // First turn: judge escalates and the streak latches.
        // Second turn: judge is not called again; capable is served directly.
        let judge_client = QueuedClient::new([r#"{"escalate":true,"reason":"stuck"}"#]);
        let model_client = QueuedClient::new(["efficient draft", "capable t1", "capable t2"]);
        let router = escalation_router(model_client, judge_client)?;

        let session_request = classify_session_request();
        router
            .clone()
            .run(Context::default(), session_request.clone())
            .await?;
        let (trace, _) = router
            .clone()
            .run(Context::default(), session_request)
            .await?;

        assert_eq!(trace.last().map(|d| d.selected_model()), Some("capable"));
        Ok(())
    }
}

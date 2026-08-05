// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Judge-backed capability, escalation, and custom-policy routing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use switchyard_protocol::{Message, Role, SimpleDecision};

use super::fall_through::{DefaultTarget, FallThrough};
use super::util::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS;
use super::util::affinity::AffinityRouter;
use super::util::classifier_contract::{ClassifierContract, ClassifierContractConfig};
use super::util::escalation::{self, EscalationJudge, EscalationJudgeConfig, EscalationPolicy};
use super::util::llm_judge::{
    ClassifierInput, JsonSchemaDecoder, JudgeClassifier, JudgePolicy, JudgeRuntimeConfig,
    SerdeDecoder, StructuredJudge,
};
use super::util::target_selector::TargetSelectorPolicy;
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
    crux: String,
    primary_rule: String,
    capability_boundary: String,
    p_solve: f64,
}

impl TaskClassifierVerdict {
    /// Rejects malformed or internally inconsistent verdicts before policy evaluation.
    fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.p_solve)
            && !self.crux.trim().is_empty()
            && matches!(
                (
                    self.primary_rule.as_str(),
                    self.capability_boundary.as_str()
                ),
                ("SUP-1" | "SUP-2" | "SUP-3" | "SUP-4" | "SUP-5", "supported")
                    | ("UNC-1" | "UNC-2", "uncertain")
                    | ("LIM-1" | "LIM-2", "unsupported")
                    | ("none", "unmatched")
            )
    }

    /// Returns the number of threshold steps assigned to this capability boundary.
    fn boundary_steps(&self) -> Option<u8> {
        match self.capability_boundary.as_str() {
            "supported" => Some(0),
            "uncertain" | "unmatched" => Some(1),
            "unsupported" => Some(2),
            _ => None,
        }
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

/// Keeps the opening task and the latest user follow-up when they differ.
fn task_messages(messages: &[Message]) -> Vec<Message> {
    let mut user_messages = messages.iter().filter(|message| message.role == Role::User);
    let Some(opening_task) = user_messages.next() else {
        return Vec::new();
    };
    match user_messages.next_back() {
        Some(latest_follow_up) => vec![opening_task.clone(), latest_follow_up.clone()],
        None => vec![opening_task.clone()],
    }
}

/// Selects the task messages shown to capability and custom-schema classifiers.
struct TaskInput {
    recent_turn_window: Option<usize>,
}

impl ClassifierInput for TaskInput {
    fn build_messages(&self, _state: &State, request: &Request) -> Vec<Message> {
        // The default preserves the whole-task anchor and latest user update. A
        // configured window widens that to the surrounding conversation.
        match self.recent_turn_window {
            Some(window) => trim_messages(&request.llm_request.messages, window),
            None => task_messages(&request.llm_request.messages),
        }
    }
}

type CapabilityJudge = StructuredJudge<TaskInput, SerdeDecoder<TaskClassifierVerdict>>;

struct TaskClassifierPolicy {
    efficient_target: String,
    capable_target: String,
    base_threshold: f64,
    threshold_step: f64,
}

impl TaskClassifierPolicy {
    fn new(
        efficient_target: impl Into<String>,
        capable_target: impl Into<String>,
        config: &TaskClassifierConfig,
    ) -> Self {
        Self {
            efficient_target: efficient_target.into(),
            capable_target: capable_target.into(),
            base_threshold: config.base_threshold,
            threshold_step: config.threshold_step,
        }
    }

    /// Returns the required solve probability for one validated verdict.
    fn threshold(&self, verdict: &TaskClassifierVerdict) -> Option<f64> {
        Some(self.base_threshold + f64::from(verdict.boundary_steps()?) * self.threshold_step)
    }
}

impl JudgePolicy for TaskClassifierPolicy {
    type Verdict = TaskClassifierVerdict;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        // Judge output is untrusted. An absent, invalid, or inconsistent verdict is
        // ambiguous so the surrounding router applies its configured fallback.
        let Some(verdict) = verdict.filter(|verdict| verdict.is_valid()) else {
            return Classification::Ambiguous(vec![]);
        };
        // A usable verdict below the capability threshold is still a decision: the judge
        // does not trust the efficient tier with this task.
        let Some(threshold) = self.threshold(verdict) else {
            return Classification::Ambiguous(vec![]);
        };
        let target = if verdict.p_solve >= threshold
            || (threshold - verdict.p_solve).abs() <= f64::EPSILON
        {
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

#[derive(Clone, Debug)]
/// Settings that control capability classifier prompting and routing.
pub struct TaskClassifierConfig {
    /// Lowest solve probability that routes a supported task to the efficient target.
    pub base_threshold: f64,
    /// Amount added per capability-boundary step.
    ///
    /// Supported verdicts use `base_threshold`, uncertain and unmatched verdicts use one
    /// step, and unsupported verdicts use two steps.
    pub threshold_step: f64,
    /// Enables session affinity before the judge-backed classifier.
    pub session_affinity: bool,
    /// Uses the first user message as the SessionKey for sticky routing when session metadata is unavailable.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns the judge sees on top of the client
    /// instructions and the opening task.
    ///
    /// `None` (the default) judges the opening task and latest user follow-up.
    /// `Some(n)` widens that to the client instructions, the opening task, and
    /// the last `n` turns after it.
    pub recent_turn_window: Option<usize>,
    /// Prompt and verdict contract settings for the classifier judge.
    pub contract: ClassifierContractConfig,
    /// Maximum completion tokens available to the classifier verdict.
    pub max_output_tokens: u64,
}

/// Flat serialized shape that maps prompt settings into the runtime contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskClassifierConfigWire {
    base_threshold: f64,
    #[serde(default)]
    threshold_step: f64,
    #[serde(default)]
    session_affinity: bool,
    #[serde(default)]
    message_hash_fallback: bool,
    #[serde(default)]
    recent_turn_window: Option<usize>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default = "default_judge_max_output_tokens")]
    max_output_tokens: u64,
}

impl<'de> Deserialize<'de> for TaskClassifierConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TaskClassifierConfigWire::deserialize(deserializer)?;
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = wire.prompt {
            contract = contract.with_prompt(prompt);
        }
        Ok(Self {
            base_threshold: wire.base_threshold,
            threshold_step: wire.threshold_step,
            session_affinity: wire.session_affinity,
            message_hash_fallback: wire.message_hash_fallback,
            recent_turn_window: wire.recent_turn_window,
            contract,
            max_output_tokens: wire.max_output_tokens,
        })
    }
}

const fn default_judge_max_output_tokens() -> u64 {
    DEFAULT_JUDGE_MAX_OUTPUT_TOKENS
}

impl Default for TaskClassifierConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.0,
            threshold_step: 0.0,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
            contract: ClassifierContractConfig::default(),
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        }
    }
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
        if !self.threshold_step.is_finite() || self.threshold_step < 0.0 {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "threshold_step must be finite and greater than or equal to 0, got {}",
                    self.threshold_step
                ),
            });
        }
        let unsupported_threshold = self.base_threshold + 2.0 * self.threshold_step;
        if unsupported_threshold > 1.0 && unsupported_threshold - 1.0 > f64::EPSILON {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "base_threshold + 2 * threshold_step must be at most 1, got {unsupported_threshold}"
                ),
            });
        }
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        if self.message_hash_fallback && !self.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        Ok(())
    }
}

/// Policy that maps a custom classifier verdict to a routing target.
#[derive(Clone, Debug)]
pub enum CustomClassifierPolicy {
    /// Resolves a JSON Pointer and treats its string value as a configured target label.
    TargetSelector {
        /// JSON Pointer evaluated against each schema-validated verdict.
        selector: String,
    },
}

impl CustomClassifierPolicy {
    /// Creates a policy that selects a target label through a JSON Pointer.
    pub fn target_selector(selector: impl Into<String>) -> Self {
        Self::TargetSelector {
            selector: selector.into(),
        }
    }
}

/// Settings for a classifier whose JSON Schema and target-selection policy are user supplied.
#[derive(Clone, Debug)]
pub struct CustomClassifierConfig {
    /// System prompt sent to the classifier judge.
    pub prompt: String,
    /// Inner JSON Schema placed inside the provider's structured-output wrapper.
    pub response_schema: Value,
    /// Deterministic policy applied after the verdict passes schema validation.
    pub policy: CustomClassifierPolicy,
    /// Enables session affinity before the judge-backed classifier.
    pub session_affinity: bool,
    /// Uses the first user message when session metadata is unavailable.
    pub message_hash_fallback: bool,
    /// Trailing conversation turns shown to the classifier judge.
    pub recent_turn_window: Option<usize>,
    /// Maximum completion tokens available to the classifier verdict.
    pub max_output_tokens: u64,
}

impl CustomClassifierConfig {
    /// Creates a custom-schema classifier contract with conservative runtime defaults.
    pub fn new(
        prompt: impl Into<String>,
        response_schema: Value,
        policy: CustomClassifierPolicy,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            response_schema,
            policy,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        if self.message_hash_fallback && !self.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        Ok(())
    }
}

enum CustomPolicyRuntime {
    TargetSelector(TargetSelectorPolicy),
}

impl JudgePolicy for CustomPolicyRuntime {
    type Verdict = Value;

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
        match self {
            Self::TargetSelector(policy) => policy.to_classification(verdict),
        }
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

/// Routes requests through a capability, escalation, or custom classifier mode.
pub struct LlmTaskClassifier {
    route: FallThrough<State>,
    /// Classifier used when this router is embedded in another cascade.
    inner: Arc<dyn Classifier<State>>,
}

struct ClassifierRouteConfig {
    default_target: String,
    session_affinity: bool,
    message_hash_fallback: bool,
}

/// Complete construction settings for one LLM classifier mode.
#[non_exhaustive]
pub enum LlmClassifierConfig {
    /// Routes between efficient and capable targets from a task-level verdict.
    Capability {
        /// Target that produces classifier verdicts.
        judge_target: LlmTarget,
        /// Target used when the efficient tier can handle the task.
        efficient_target: LlmTarget,
        /// Target used when the task needs the capable tier.
        capable_target: LlmTarget,
        /// Capability classifier settings.
        config: TaskClassifierConfig,
    },
    /// Judges efficient responses and escalates after a confirmed streak.
    Escalation {
        /// Target that produces escalation verdicts.
        judge_target: LlmTarget,
        /// Target called before each escalation decision.
        efficient_target: LlmTarget,
        /// Target used after escalation is confirmed.
        capable_target: LlmTarget,
        /// Prompt and verdict contract settings for the escalation judge.
        contract: ClassifierContractConfig,
        /// Escalation policy settings.
        config: EscalationJudgeConfig,
        /// Maximum completion tokens available to the escalation verdict.
        max_output_tokens: u64,
    },
    /// Routes among named targets using a user-supplied schema and policy.
    Custom {
        /// Target that produces classifier verdicts.
        judge_target: LlmTarget,
        /// User-facing labels paired with their resolved routing targets.
        targets: Vec<(String, LlmTarget)>,
        /// Label selected when the judge does not produce a usable verdict.
        default_target: String,
        /// Custom classifier settings.
        config: CustomClassifierConfig,
    },
}

impl LlmTaskClassifier {
    /// Builds the classifier mode described by `config`.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected mode's targets, contract, policy, or runtime
    /// settings are invalid.
    pub fn new(config: LlmClassifierConfig) -> Result<Self> {
        match config {
            LlmClassifierConfig::Capability {
                judge_target,
                efficient_target,
                capable_target,
                config,
            } => Self::build_capability(judge_target, efficient_target, capable_target, config),
            LlmClassifierConfig::Escalation {
                judge_target,
                efficient_target,
                capable_target,
                contract,
                config,
                max_output_tokens,
            } => Self::build_escalation(
                judge_target,
                efficient_target,
                capable_target,
                contract,
                config,
                max_output_tokens,
            ),
            LlmClassifierConfig::Custom {
                judge_target,
                targets,
                default_target,
                config,
            } => Self::build_custom(judge_target, targets, default_target, config),
        }
    }

    fn build_capability(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        config: TaskClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        let contract = Self::load_capability_contract(&config.contract)?;
        let targets = LlmTargetSet::new(vec![efficient_target.clone(), capable_target.clone()]);
        let session_affinity = config.session_affinity;
        let message_hash_fallback = config.message_hash_fallback;
        let classifier = Arc::new(TaskClassifier {
            classifier: JudgeClassifier::new(
                StructuredJudge::new(
                    TaskInput {
                        recent_turn_window: config.recent_turn_window,
                    },
                    contract,
                    SerdeDecoder::new(),
                    JudgeRuntimeConfig::new(config.max_output_tokens)?,
                ),
                judge_target.clone(),
                TaskClassifierPolicy::new(
                    efficient_target.semantic_name.clone(),
                    capable_target.semantic_name.clone(),
                    &config,
                ),
            ),
            efficient_target: efficient_target.semantic_name.clone(),
            capable_target: capable_target.semantic_name.clone(),
        });
        let inner: Arc<dyn Classifier<State>> = classifier.clone();
        Self::from_classifier(
            targets,
            inner,
            ClassifierRouteConfig {
                default_target: classifier.capable_target.clone(),
                session_affinity,
                message_hash_fallback,
            },
        )
    }

    fn build_custom(
        judge_target: LlmTarget,
        targets: Vec<(String, LlmTarget)>,
        default_target: String,
        config: CustomClassifierConfig,
    ) -> Result<Self> {
        config.validate()?;
        if targets.len() < 2 {
            return Err(LibsyError::AlgorithmError {
                message: "custom classifier requires at least two targets".to_string(),
            });
        }

        let mut labels = BTreeSet::new();
        let mut semantic_names = BTreeSet::new();
        let mut target_map = BTreeMap::new();
        let mut resolved_targets = Vec::with_capacity(targets.len());
        for (label, target) in targets {
            if label.trim().is_empty() || label.trim() != label {
                return Err(LibsyError::AlgorithmError {
                    message: "custom classifier target labels must be non-empty and have no surrounding whitespace"
                        .to_string(),
                });
            }
            if !labels.insert(label.clone()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!("custom classifier target label {label:?} is duplicated"),
                });
            }
            if !semantic_names.insert(target.semantic_name.clone()) {
                return Err(LibsyError::AlgorithmError {
                    message: format!(
                        "custom classifier resolved target {:?} is duplicated",
                        target.semantic_name
                    ),
                });
            }
            target_map.insert(label, target.semantic_name.clone());
            resolved_targets.push(target);
        }
        let default_semantic_name =
            target_map
                .get(&default_target)
                .cloned()
                .ok_or_else(|| LibsyError::AlgorithmError {
                    message: format!(
                        "default_target {default_target:?} must be one of the configured targets"
                    ),
                })?;

        let CustomClassifierConfig {
            prompt,
            response_schema,
            policy,
            session_affinity,
            message_hash_fallback,
            recent_turn_window,
            max_output_tokens,
        } = config;
        let contract = ClassifierContract::from_inner_schema(&prompt, response_schema)?;
        let policy = match policy {
            CustomClassifierPolicy::TargetSelector { selector } => {
                CustomPolicyRuntime::TargetSelector(TargetSelectorPolicy::new(
                    selector, target_map,
                )?)
            }
        };
        let classifier: Arc<dyn Classifier<State>> = Arc::new(JudgeClassifier::new(
            StructuredJudge::new(
                TaskInput { recent_turn_window },
                contract,
                JsonSchemaDecoder::new(),
                JudgeRuntimeConfig::new(max_output_tokens)?,
            ),
            judge_target,
            policy,
        ));

        Self::from_classifier(
            LlmTargetSet::new(resolved_targets),
            classifier,
            ClassifierRouteConfig {
                default_target: default_semantic_name,
                session_affinity,
                message_hash_fallback,
            },
        )
    }

    fn build_escalation(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        contract_config: ClassifierContractConfig,
        config: EscalationJudgeConfig,
        max_output_tokens: u64,
    ) -> Result<Self> {
        let capable_name = capable_target.semantic_name.clone();
        let efficient_name = efficient_target.semantic_name.clone();
        let confirmations = config.confirmations;
        let esc = Arc::new(EscalationClassifier {
            judge: escalation::build_judge(
                judge_target,
                capable_name,
                efficient_name,
                &contract_config,
                config,
                max_output_tokens,
            )?,
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

    /// Loads the packaged capability-classifier contract.
    fn load_capability_contract(config: &ClassifierContractConfig) -> Result<ClassifierContract> {
        ClassifierContract::from_config(config, PROMPT_TEMPLATE, SCHEMA_TEMPLATE)
    }

    /// Keeps affinity and fallback ordering identical across judge-backed modes.
    fn from_classifier(
        targets: LlmTargetSet,
        inner: Arc<dyn Classifier<State>>,
        config: ClassifierRouteConfig,
    ) -> Result<Self> {
        targets.get_target(&config.default_target)?;
        if config.message_hash_fallback && !config.session_affinity {
            return Err(LibsyError::AlgorithmError {
                message: "message_hash_fallback requires session_affinity".to_string(),
            });
        }
        // Affinity comes first so a retained assignment short-circuits the judge call.
        // Note: when this classifier is embedded inside another cascade (e.g. StageRouter)
        // the affinity processor never fires — only the inner score() is called.
        let mut route = FallThrough::<State>::new_with_state(targets).with_name(ALGORITHM_NAME);
        if config.session_affinity {
            let affinity = if config.message_hash_fallback {
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
        let fallback = DefaultTarget::new(config.default_target);
        Ok(Self {
            route: route
                .with_classifier(inner.clone())
                .with_classifier(Arc::new(fallback)),
            inner,
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
        LlmClientError, LlmRequest, Metadata, completion_text, text_request, text_response,
    };

    use crate::algorithms::util::llm_judge::Judge;
    use crate::core::algorithm::Algorithm;
    use switchyard_protocol::{Context, LlmResponse, Response, RoutedLlmClient};

    const TEST_THRESHOLD: f64 = 0.5;

    fn test_config(base_threshold: f64) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold,
            ..TaskClassifierConfig::default()
        }
    }

    fn policy() -> TaskClassifierPolicy {
        TaskClassifierPolicy::new("efficient", "capable", &test_config(TEST_THRESHOLD))
    }

    fn verdict(
        p_solve: f64,
        capability_boundary: &str,
        primary_rule: &str,
    ) -> TaskClassifierVerdict {
        TaskClassifierVerdict {
            crux: "test crux".to_string(),
            primary_rule: primary_rule.to_string(),
            capability_boundary: capability_boundary.to_string(),
            p_solve,
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
        judge_max_output_tokens: Mutex<Vec<Option<u64>>>,
        judge_system_prompts: Mutex<Vec<String>>,
    }

    impl PerRequestClient {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }

        fn judge_max_output_tokens(&self) -> Vec<Option<u64>> {
            self.judge_max_output_tokens.lock().clone()
        }

        fn judge_system_prompts(&self) -> Vec<String> {
            self.judge_system_prompts.lock().clone()
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
                self.judge_max_output_tokens
                    .lock()
                    .push(request.llm_request.output.max_output_tokens);
                self.judge_system_prompts.lock().extend(
                    request
                        .llm_request
                        .messages
                        .first()
                        .and_then(|message| message.text_content("\n")),
                );
                r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#.to_string()
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
            LlmClassifierConfig::Capability {
                judge_target: target("judge"),
                efficient_target: target("efficient"),
                capable_target: target("capable"),
                config: test_config(TEST_THRESHOLD),
            },
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
    async fn classifier_config_sets_the_judge_completion_cap() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: target("judge"),
            efficient_target: target("efficient"),
            capable_target: target("capable"),
            config: TaskClassifierConfig {
                max_output_tokens: 512,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        router.run(Context::default(), classify_request()).await?;

        assert_eq!(client.judge_max_output_tokens(), vec![Some(512)]);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_overrides_the_packaged_prompt() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: target("judge"),
            efficient_target: target("efficient"),
            capable_target: target("capable"),
            config: TaskClassifierConfig {
                contract: ClassifierContractConfig::default()
                    .with_prompt("Custom capability rubric."),
                ..test_config(TEST_THRESHOLD)
            },
        })?);

        router.run(Context::default(), classify_request()).await?;

        let prompts = client.judge_system_prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], "Custom capability rubric.");
        Ok(())
    }

    #[tokio::test]
    async fn classifier_config_enables_session_affinity() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: target("judge"),
            efficient_target: target("efficient"),
            capable_target: target("capable"),
            config: TaskClassifierConfig {
                session_affinity: true,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

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
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Capability {
            judge_target: target("judge"),
            efficient_target: target("efficient"),
            capable_target: target("capable"),
            config: TaskClassifierConfig {
                session_affinity: true,
                message_hash_fallback: true,
                recent_turn_window: None,
                ..test_config(TEST_THRESHOLD)
            },
        })?);

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
        let at_threshold = verdict(0.5, "supported", "SUP-1");
        let below_threshold = verdict(0.49, "supported", "SUP-1");
        assert_eq!(selected(&policy, Some(&at_threshold))?, "efficient");
        assert_eq!(selected(&policy, Some(&below_threshold))?, "capable");
        Ok(())
    }

    #[test]
    fn the_threshold_moves_the_routing_boundary() -> Result<()> {
        let borderline = verdict(0.5, "supported", "SUP-1");
        let strict = TaskClassifierPolicy::new("efficient", "capable", &test_config(0.9));
        let lenient = TaskClassifierPolicy::new("efficient", "capable", &test_config(0.1));
        assert_eq!(selected(&strict, Some(&borderline))?, "capable");
        assert_eq!(selected(&lenient, Some(&borderline))?, "efficient");
        Ok(())
    }

    #[test]
    fn classifier_config_rejects_unknown_fields() {
        let error = serde_json::from_value::<TaskClassifierConfig>(serde_json::json!({
            "base_threshold": 0.5,
            "classifier_magic": true,
        }))
        .expect_err("unknown classifier fields must be rejected");

        assert!(
            error
                .to_string()
                .contains("unknown field `classifier_magic`"),
            "{error}"
        );
    }

    #[test]
    fn invalid_classifier_config_is_rejected() -> Result<()> {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: None,
        };
        for bad in [1.5, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    judge_target: target("judge"),
                    efficient_target: target("e"),
                    capable_target: target("c"),
                    config: test_config(bad),
                })
                .is_err(),
                "base threshold {bad} should be rejected"
            );
        }
        for config in [
            TaskClassifierConfig {
                base_threshold: 0.5,
                threshold_step: -0.1,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.8,
                threshold_step: 0.11,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                message_hash_fallback: true,
                ..TaskClassifierConfig::default()
            },
            TaskClassifierConfig {
                base_threshold: 0.5,
                max_output_tokens: 0,
                ..TaskClassifierConfig::default()
            },
        ] {
            assert!(
                LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                    judge_target: target("judge"),
                    efficient_target: target("e"),
                    capable_target: target("c"),
                    config,
                })
                .is_err()
            );
        }
        for base_threshold in [0.0, 1.0] {
            LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                judge_target: target("judge"),
                efficient_target: target("e"),
                capable_target: target("c"),
                config: test_config(base_threshold),
            })?;
        }
        Ok(())
    }

    #[test]
    fn an_unusable_verdict_is_ambiguous() -> Result<()> {
        let policy = policy();
        let inconsistent_rule = TaskClassifierVerdict {
            capability_boundary: "uncertain".to_string(),
            ..verdict(1.0, "supported", "SUP-1")
        };
        let empty_crux = TaskClassifierVerdict {
            crux: "  ".to_string(),
            ..verdict(1.0, "supported", "SUP-1")
        };
        let unusable = [
            Some(verdict(1.1, "supported", "SUP-1")),
            Some(inconsistent_rule),
            Some(empty_crux),
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
    fn capability_boundaries_apply_monotonic_threshold_steps() -> Result<()> {
        let policy = TaskClassifierPolicy::new(
            "efficient",
            "capable",
            &TaskClassifierConfig {
                threshold_step: 0.1,
                ..test_config(0.4)
            },
        );

        assert_eq!(
            selected(&policy, Some(&verdict(0.4, "supported", "SUP-2")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.49, "uncertain", "UNC-1")))?,
            "capable"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.5, "uncertain", "UNC-1")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.5, "unmatched", "none")))?,
            "efficient"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.59, "unsupported", "LIM-1")))?,
            "capable"
        );
        assert_eq!(
            selected(&policy, Some(&verdict(0.6, "unsupported", "LIM-1")))?,
            "efficient"
        );
        Ok(())
    }

    /// The text of each message a judge with `recent_turn_window` would be sent.
    /// The no-window case is covered by `capability_judge_builds_a_structured_request`.
    fn capability_judge(recent_turn_window: Option<usize>) -> Result<CapabilityJudge> {
        Ok(StructuredJudge::new(
            TaskInput { recent_turn_window },
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)?,
        ))
    }

    fn judged_contents(recent_turn_window: usize) -> Result<Vec<String>> {
        let judge = capability_judge(Some(recent_turn_window))?;
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
        let judge = capability_judge(None)?;
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
        assert_eq!(judge_request.llm_request.messages.len(), 3);
        let contents = judge_request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>();
        assert!(contents.contains(&"recent 4".to_string()));
        assert!(contents.contains(&"initial task".to_string()));
        assert!(!contents.contains(&"recent 5".to_string()));
        assert!(!contents.contains(&"client instructions".to_string()));
        assert_eq!(
            judge_request.llm_request.output.response_format,
            Some(judge.contract().response_format().clone())
        );
        assert_eq!(
            judge_request.llm_request.output.max_output_tokens,
            Some(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)
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
        let contract =
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?;
        let schema = contract.response_format();
        let reply = schema_shaped_verdict(schema)?;
        let judge: CapabilityJudge = StructuredJudge::new(
            TaskInput {
                recent_turn_window: None,
            },
            contract,
            SerdeDecoder::new(),
            JudgeRuntimeConfig::new(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS)?,
        );

        let verdict = judge.parse(&text_response(None, reply))?;

        assert!(verdict.is_valid());
        assert!((0.0..=1.0).contains(&verdict.p_solve));
        Ok(())
    }

    #[test]
    fn packaged_prompt_keeps_the_schema_in_the_structured_request() -> Result<()> {
        let contract =
            LlmTaskClassifier::load_capability_contract(&ClassifierContractConfig::default())?;
        let prompt = contract.system_prompt();
        let schema_name = contract
            .response_format()
            .pointer("/json_schema/name")
            .and_then(Value::as_str)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "packaged response schema has no name".to_string(),
            })?;
        assert_eq!(schema_name, "CapabilityClassifierDecision");
        assert!(prompt.contains("SUP-1 [supported]"));
        assert!(prompt.contains("SUP-5 [supported]"));
        assert!(!prompt.contains("{{RESPONSE_SCHEMA}}"));
        assert!(!prompt.contains("\"type\": \"object\""));
        assert!(!prompt.contains("\"json_schema\""));
        assert!(!prompt.contains(schema_name));
        let rule_values = contract
            .response_format()
            .pointer("/json_schema/schema/properties/primary_rule/enum")
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
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: target("judge", judge_client),
                efficient_target: target("efficient", client.clone()),
                capable_target: target("capable", client),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations: 1,
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
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
    async fn escalation_config_overrides_the_packaged_prompt() -> Result<()> {
        let client = Arc::new(PerRequestClient::default());
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: target("judge"),
            efficient_target: target("efficient"),
            capable_target: target("capable"),
            contract: ClassifierContractConfig::default().with_prompt("Custom trajectory rubric."),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        })?);

        router.run(Context::default(), classify_request()).await?;

        let prompts = client.judge_system_prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0], "Custom trajectory rubric.");
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

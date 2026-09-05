// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-neutral algorithm configuration and construction.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;

use libsy::{
    AdvisorGate, AdvisorGateConfig, Algorithm, ClassifierContractConfig, ClassifierResponseFormat,
    ClassifyTrigger, CompositeRouter, CompositeRouterConfig, CustomClassifierConfig,
    CustomClassifierPolicy, EscalationJudgeConfig, GateTrigger, HandoffNoteConfig,
    LlmClassifierConfig, LlmFallback, LlmTaskClassifier, Noop, Passthrough, PickerMode, Random,
    StageRouter, StageRouterConfig, SubagentRouter, SubagentRouterConfig, TargetPrompts,
    TaskClassifierConfig,
};
use serde::Deserialize;
use switchyard_protocol::ModelId;

/// Error returned when an algorithm description cannot be constructed.
#[derive(Debug)]
pub struct AlgorithmConfigError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl AlgorithmConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    fn with_source(message: impl Into<String>, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl Display for AlgorithmConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for AlgorithmConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

type AlgorithmResult<T> = Result<T, AlgorithmConfigError>;

/// How a custom classifier turns the judge's JSON verdict into a target.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClassifierPolicyConfig {
    /// Reads the target name straight out of the judge's verdict.
    TargetSelector {
        /// JSON Pointer to the name, such as `/decision/target`.
        selector: String,
    },
}

/// Which of the three `llm_classifier` behaviors a route uses.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierMode {
    /// Judges the request first, then serves the strong or weak target.
    Capability,
    /// Serves the weak target first and judges the finished turn, moving to the
    /// strong target once the session latches.
    Escalation,
    /// Judges against your own JSON schema and routes to any configured target.
    Custom,
}

impl ClassifierPolicyConfig {
    fn into_libsy(self) -> CustomClassifierPolicy {
        match self {
            Self::TargetSelector { selector } => CustomClassifierPolicy::target_selector(selector),
        }
    }
}

#[derive(Clone, Debug)]
enum LlmClassifierModeConfig {
    Capability(CapabilityClassifierRouteConfig),
    Escalation(EscalationClassifierRouteConfig),
    Custom(CustomClassifierRouteConfig),
}

#[derive(Clone, Debug)]
struct CapabilityClassifierRouteConfig {
    strong_target: String,
    weak_target: String,
    base_threshold: f64,
    threshold_step: f64,
    classify_trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    judge_char_budget: usize,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
}

#[derive(Clone, Debug)]
struct EscalationClassifierRouteConfig {
    strong_target: String,
    weak_target: String,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
    judge: EscalationJudgeConfig,
}

#[derive(Clone, Debug)]
struct CustomClassifierRouteConfig {
    classifier_target: String,
    targets: Vec<String>,
    default_target: String,
    prompt: String,
    response_schema: String,
    policy: ClassifierPolicyConfig,
    classify_trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    judge_char_budget: usize,
    max_output_tokens: u64,
}

/// Settings for an `llm_classifier` route. Which fields are required depends on
/// the [`ClassifierMode`]; using a field from the wrong mode is an error.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmClassifierRouteConfig {
    /// Target the judge is called through. Never a routing destination itself.
    pub classifier_target: String,
    /// Mode to run. Defaults to escalation when `escalation` is set, otherwise capability.
    pub mode: Option<ClassifierMode>,
    /// Capability and escalation modes: the capable tier.
    pub strong_target: Option<String>,
    /// Capability and escalation modes: the efficient tier.
    pub weak_target: Option<String>,
    /// Capability mode: lowest solve probability that still routes to the weak
    /// target, from 0 to 1.
    pub base_threshold: Option<f64>,
    /// Capability mode: how much to raise the threshold when the judge is
    /// uncertain. Added once for an uncertain verdict and twice for unsupported.
    pub threshold_step: Option<f64>,
    /// How often the judge runs: every request, once per user turn, or once per session.
    pub classify_trigger: ClassifyTrigger,
    /// Reuses the session's target by hashing the first user message when no
    /// session ID is available. Needs `classify_trigger = "new_session"`.
    pub message_hash_fallback: bool,
    /// How many trailing turns the judge sees. Unset shows it the opening task
    /// and the latest user follow-up only.
    pub recent_turn_window: Option<usize>,
    /// Most characters a windowed judge payload may use. The window narrows from the
    /// oldest turn until it fits, so one large tool result cannot decide the judge's
    /// cost. Ignored without `recent_turn_window`.
    #[serde(default = "default_judge_char_budget")]
    pub judge_char_budget: usize,
    /// Replaces the packaged judge prompt. Required in custom mode.
    pub prompt: Option<String>,
    /// How the judge is asked for structured output. Use `json_object` when the
    /// provider cannot do JSON Schema.
    pub response_format_type: ClassifierResponseFormat,
    /// Most completion tokens the judge verdict may use.
    #[serde(default = "default_classifier_max_output_tokens")]
    pub max_output_tokens: u64,
    /// Escalation mode: how many escalate verdicts latch the session, and how
    /// much of the transcript the judge sees.
    pub escalation: Option<EscalationJudgeConfig>,
    /// Custom mode: the target names the policy may pick from.
    pub targets: Option<Vec<String>>,
    /// Custom mode: target used when the judge fails or its verdict cannot be routed.
    pub default_target: Option<String>,
    /// Custom mode: JSON Schema the verdict must match, written as a string.
    pub response_schema: Option<String>,
    /// Custom mode: how to read the chosen target out of the verdict.
    pub policy: Option<ClassifierPolicyConfig>,
}

/// Routing policy applied only to delegated sub-agent work, nested inside a
/// `passthrough` or `stage_router` route.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SubagentRouteConfig {
    /// Sends all sub-agent work to one target.
    Passthrough {
        /// Target that serves sub-agent requests.
        target: String,
    },
    /// Judges each sub-agent request. Only [`ClassifierMode::Custom`] is supported here.
    LlmClassifier(Box<LlmClassifierRouteConfig>),
}

impl SubagentRouteConfig {
    fn routing_target_names(&self) -> Vec<&str> {
        match self {
            Self::Passthrough { target } => vec![target],
            Self::LlmClassifier(classifier) => classifier
                .targets
                .iter()
                .flatten()
                .map(String::as_str)
                .collect(),
        }
    }

    fn classifier_target_name(&self) -> Option<&str> {
        match self {
            Self::LlmClassifier(classifier) => Some(&classifier.classifier_target),
            Self::Passthrough { .. } => None,
        }
    }
}

/// A routing algorithm described by configured target names.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AlgorithmSpec {
    /// Replies `OK` without calling any model. Useful for smoke tests.
    Noop {},
    /// Splits traffic across several targets.
    Random {
        /// Target names to choose from.
        targets: Vec<String>,
        /// Relative weights in `targets` order. Equal weights when unset.
        weights: Option<Vec<f64>>,
        /// Makes the sequence of choices repeatable.
        seed: Option<u64>,
    },
    /// Sends every request to one target.
    Passthrough {
        /// Target that serves the request.
        target: String,
        /// Separate policy for delegated sub-agent work.
        #[serde(default)]
        subagents: Option<SubagentRouteConfig>,
    },
    /// Asks a judge model which target should serve the request.
    LlmClassifier {
        /// Judge and tier settings, written directly in the route table.
        #[serde(flatten)]
        config: LlmClassifierRouteConfig,
    },
    /// Picks a tier per turn by scoring signals from recent tool results.
    StageRouter {
        #[serde(flatten)]
        tiers: StageTierConfig,
        /// Tier to use when the signals are not confident.
        picker: PickerMode,
        /// Judge consulted for turns the tool signals cannot decide.
        #[serde(default)]
        classifier: Option<StageClassifierConfig>,
        /// Separate policy for delegated sub-agent work.
        #[serde(default)]
        subagents: Option<SubagentRouteConfig>,
    },
    /// A judge picks the tier at each user turn; a stage router runs the turns within it.
    Composite {
        /// Judge that picks the tier. Called through its own target.
        classifier: StageClassifierConfig,
        /// The stage router the judge hands off to.
        stage: StageTierConfig,
        /// Separate policy for delegated sub-agent work.
        #[serde(default)]
        subagents: Option<SubagentRouteConfig>,
    },
    /// Serves every turn from one target, and has a second model review some of
    /// those turns before the caller sees them.
    Advisor {
        /// Serves every client-visible turn.
        executor_target: String,
        /// Reviews gated turns. Never a routing destination.
        advisor_target: String,
        /// Replaces the built-in APPROVE/REDO reviewer prompt.
        #[serde(default)]
        reviewer_system_prompt: Option<String>,
        /// Replaces the built-in text put in front of a REDO plan.
        #[serde(default)]
        redo_feedback_prefix: Option<String>,
        /// What fires a review.
        #[serde(default)]
        gate_trigger: AdvisorTriggerConfig,
        /// Regular expression for the `pattern` trigger. Required by it, and unused otherwise.
        #[serde(default)]
        gate_trigger_pattern: Option<String>,
        /// How many reviews one session may spend.
        #[serde(default = "default_max_reviews")]
        max_reviews: u32,
        /// Reviews a turn after this many assistant turns, as a mid-task
        /// checkpoint. Zero turns the checkpoint off.
        #[serde(default)]
        gate_stall_turns: u32,
        /// Tool results a turn needs before it can be reviewed. Skips early chatty turns.
        #[serde(default)]
        gate_min_tool_results: u32,
        /// Most output tokens one review may use.
        #[serde(default = "default_advisor_max_tokens")]
        advisor_max_tokens: u64,
        /// Sampling temperature for reviews. Left off the request when unset.
        #[serde(default)]
        advisor_temperature: Option<f64>,
        /// Size cap on the transcript sent to the advisor. Longer transcripts
        /// are trimmed from the middle.
        #[serde(default = "default_transcript_max_chars")]
        transcript_max_chars: usize,
        /// Lets the turn through when the advisor fails, instead of erroring.
        #[serde(default = "default_fail_open")]
        fail_open: bool,
    },
    /// Routes using a checkpoint-backed prefill classifier.
    PrefillRouter {
        /// Target names in checkpoint output order.
        targets: Vec<String>,
        /// Tensor-only router checkpoint path.
        checkpoint: PathBuf,
        /// PyTorch device used for encoder inference, such as `cpu`, `cuda`, or
        /// `cuda:0`. Auto-detected when omitted.
        device: Option<String>,
        /// Directory where Hugging Face caches the downloaded encoder and tokenizer.
        cache_dir: Option<PathBuf>,
        /// Maximum tokenized encoder input length; longer prompts are truncated.
        max_length: Option<usize>,
        /// Maximum prompts per encoder forward pass.
        batch_size: Option<usize>,
    },
}

/// What fires an advisor route's review.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorTriggerConfig {
    /// The executor's first turn without tool calls.
    #[default]
    NoToolCall,
    /// The first turn whose text matches `gate_trigger_pattern`.
    Pattern,
}

/// The judge a `stage_router` route falls through to, and how it routes.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageClassifierConfig {
    /// Target the judge is called through. Not a routing destination.
    pub target: String,
    /// Lowest solve probability that still routes to the efficient tier, from 0 to 1.
    pub base_threshold: f64,
    /// How much to raise the threshold when the judge is uncertain. Added once
    /// for an uncertain verdict and twice for unsupported.
    #[serde(default)]
    pub threshold_step: f64,
    /// How often the judge runs. `new_session` has no effect here.
    #[serde(default)]
    pub classify_trigger: ClassifyTrigger,
    /// Reuses the session's target by hashing the first user message when no
    /// session ID is available.
    #[serde(default)]
    pub message_hash_fallback: bool,
    /// How many trailing turns the judge sees. Unset shows it the opening task
    /// and the latest user follow-up only.
    #[serde(default)]
    pub recent_turn_window: Option<usize>,
    /// Most characters a windowed judge payload may use. Ignored without a window.
    #[serde(default = "default_judge_char_budget")]
    pub judge_char_budget: usize,
    /// Replaces the packaged judge prompt.
    #[serde(default)]
    pub prompt: Option<String>,
    /// How the judge is asked for structured output. Use `json_object` when the
    /// provider cannot do JSON Schema.
    #[serde(default)]
    pub response_format_type: ClassifierResponseFormat,
    /// Most completion tokens the judge verdict may use.
    #[serde(default = "default_classifier_max_output_tokens")]
    pub max_output_tokens: u64,
}

/// The tier pair and scoring settings shared by every stage-router-backed route.
///
/// `deny_unknown_fields` here is what rejects a typo on a flattened `stage_router`
/// route, since the enum's own `deny_unknown_fields` does not apply through a flatten.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTierConfig {
    /// The capable tier.
    pub capable_target: String,
    /// The efficient tier.
    pub efficient_target: String,
    /// How much agreement a decisive pick needs, from 0 to 1.
    pub confidence_threshold: f64,
    /// How many trailing tool results the signals are scored over.
    #[serde(default)]
    pub recent_turn_window: Option<usize>,
    /// Notes handed to a tier when the router switches to it.
    #[serde(default)]
    pub handoff_notes: Option<HandoffNoteConfig>,
    /// System prompt handed to the capable tier.
    #[serde(default)]
    pub capable_system_prompt: Option<String>,
    /// System prompt handed to the efficient tier.
    #[serde(default)]
    pub efficient_system_prompt: Option<String>,
}

impl StageClassifierConfig {
    fn task_classifier_config(&self) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold: self.base_threshold,
            threshold_step: self.threshold_step,
            classify_trigger: self.classify_trigger,
            message_hash_fallback: self.message_hash_fallback,
            recent_turn_window: self.recent_turn_window,
            judge_char_budget: self.judge_char_budget,
            contract: classifier_contract(self.prompt.as_deref())
                .with_response_format_type(self.response_format_type),
            max_output_tokens: self.max_output_tokens,
        }
    }
}

impl AlgorithmSpec {
    /// Completion targets in algorithm order; judge-only targets are excluded.
    pub fn routing_target_names(&self) -> Vec<&str> {
        match self {
            Self::Noop { .. } => Vec::new(),
            Self::Random { targets, .. } => targets.iter().map(String::as_str).collect(),
            Self::Passthrough {
                target, subagents, ..
            } => {
                let mut names = vec![target.as_str()];
                if let Some(subagents) = subagents {
                    names.extend(subagents.routing_target_names());
                }
                names
            }
            Self::LlmClassifier { config, .. } => {
                match config.mode.unwrap_or(if config.escalation.is_some() {
                    ClassifierMode::Escalation
                } else {
                    ClassifierMode::Capability
                }) {
                    ClassifierMode::Capability => config
                        .weak_target
                        .iter()
                        .chain(&config.strong_target)
                        .map(String::as_str)
                        .collect(),
                    ClassifierMode::Escalation => config
                        .strong_target
                        .iter()
                        .chain(&config.weak_target)
                        .map(String::as_str)
                        .collect(),
                    ClassifierMode::Custom => config
                        .targets
                        .iter()
                        .flatten()
                        .map(String::as_str)
                        .collect(),
                }
            }
            Self::StageRouter {
                tiers, subagents, ..
            } => {
                let mut names = vec![
                    tiers.capable_target.as_str(),
                    tiers.efficient_target.as_str(),
                ];
                if let Some(subagents) = subagents {
                    names.extend(subagents.routing_target_names());
                }
                names
            }
            Self::Composite {
                stage, subagents, ..
            } => {
                let mut names = vec![
                    stage.capable_target.as_str(),
                    stage.efficient_target.as_str(),
                ];
                if let Some(subagents) = subagents {
                    names.extend(subagents.routing_target_names());
                }
                names
            }
            // The advisor is judge-only: reviews go through its own client,
            // so it is not a completion (or count_tokens) destination.
            Self::Advisor {
                executor_target, ..
            } => vec![executor_target],
            Self::PrefillRouter { targets, .. } => targets.iter().map(String::as_str).collect(),
        }
    }

    /// Every target the algorithm may call, including judge-only targets.
    ///
    /// [`routing_target_names`](Self::routing_target_names) covers completion destinations;
    /// a classifier also calls its judge, and that call needs a client too.
    pub fn callable_target_names(&self) -> Vec<&str> {
        let mut names = self.routing_target_names();
        match self {
            Self::LlmClassifier { config, .. } => names.push(&config.classifier_target),
            Self::Passthrough {
                subagents: Some(subagents),
                ..
            } => names.extend(subagents.classifier_target_name()),
            Self::StageRouter {
                classifier,
                subagents,
                ..
            } => {
                if let Some(classifier) = classifier {
                    names.push(&classifier.target);
                }
                if let Some(subagents) = subagents {
                    names.extend(subagents.classifier_target_name());
                }
            }
            Self::Composite {
                classifier,
                subagents,
                ..
            } => {
                names.push(&classifier.target);
                if let Some(subagents) = subagents {
                    names.extend(subagents.classifier_target_name());
                }
            }
            Self::Advisor { advisor_target, .. } => names.push(advisor_target),
            _ => {}
        }
        names
    }
    /// Builds this algorithm after resolving configured target names.
    pub fn build(
        &self,
        context: &str,
        targets: &BTreeMap<String, ModelId>,
    ) -> AlgorithmResult<Arc<dyn Algorithm>> {
        build_algorithm(context, self, targets)
    }
}
impl LlmClassifierRouteConfig {
    fn classifier_mode(&self, route_name: &str) -> AlgorithmResult<LlmClassifierModeConfig> {
        let Self {
            classifier_target,
            mode,
            strong_target,
            weak_target,
            base_threshold,
            threshold_step,
            classify_trigger,
            message_hash_fallback,
            recent_turn_window,
            judge_char_budget,
            prompt,
            response_format_type,
            max_output_tokens,
            escalation,
            targets,
            default_target,
            response_schema,
            policy,
        } = self;

        let selected_mode = match (mode, escalation.is_some()) {
            (Some(mode), _) => *mode,
            (None, true) => ClassifierMode::Escalation,
            (None, false) => ClassifierMode::Capability,
        };

        match selected_mode {
            ClassifierMode::Capability => {
                if escalation.is_some() {
                    return Err(classifier_field_error(
                        route_name,
                        "escalation",
                        "capability",
                    ));
                }
                reject_custom_fields(
                    route_name,
                    "capability",
                    targets,
                    default_target,
                    response_schema,
                    policy,
                )?;
                Ok(LlmClassifierModeConfig::Capability(
                    CapabilityClassifierRouteConfig {
                        strong_target: required_classifier_field(
                            route_name,
                            "strong_target",
                            strong_target,
                        )?,
                        weak_target: required_classifier_field(
                            route_name,
                            "weak_target",
                            weak_target,
                        )?,
                        base_threshold: required_classifier_field(
                            route_name,
                            "base_threshold",
                            base_threshold,
                        )?,
                        threshold_step: threshold_step.unwrap_or_default(),
                        classify_trigger: *classify_trigger,
                        message_hash_fallback: *message_hash_fallback,
                        recent_turn_window: *recent_turn_window,
                        judge_char_budget: *judge_char_budget,
                        prompt: prompt.clone(),
                        response_format_type: *response_format_type,
                        max_output_tokens: *max_output_tokens,
                    },
                ))
            }
            ClassifierMode::Escalation => {
                reject_custom_fields(
                    route_name,
                    "escalation",
                    targets,
                    default_target,
                    response_schema,
                    policy,
                )?;
                if *classify_trigger != ClassifyTrigger::EveryRequest {
                    return Err(AlgorithmConfigError::new(format!(
                        "llm_classifier route {route_name} mode escalation cannot use classify_trigger"
                    )));
                }
                if mode.is_some()
                    && (base_threshold.is_some()
                        || threshold_step.is_some()
                        || *message_hash_fallback
                        || recent_turn_window.is_some())
                {
                    return Err(AlgorithmConfigError::new(format!(
                        "llm_classifier route {route_name} mode escalation cannot use capability routing settings"
                    )));
                }
                Ok(LlmClassifierModeConfig::Escalation(
                    EscalationClassifierRouteConfig {
                        strong_target: required_classifier_field(
                            route_name,
                            "strong_target",
                            strong_target,
                        )?,
                        weak_target: required_classifier_field(
                            route_name,
                            "weak_target",
                            weak_target,
                        )?,
                        prompt: prompt.clone(),
                        response_format_type: *response_format_type,
                        max_output_tokens: *max_output_tokens,
                        judge: required_classifier_field(route_name, "escalation", escalation)?,
                    },
                ))
            }
            ClassifierMode::Custom => {
                if strong_target.is_some()
                    || weak_target.is_some()
                    || base_threshold.is_some()
                    || threshold_step.is_some()
                    || escalation.is_some()
                    || *response_format_type != ClassifierResponseFormat::JsonSchema
                {
                    return Err(AlgorithmConfigError::new(format!(
                        "llm_classifier route {route_name} mode custom cannot use capability or escalation fields and response_format_type must be 'json_schema'"
                    )));
                }
                Ok(LlmClassifierModeConfig::Custom(
                    CustomClassifierRouteConfig {
                        classifier_target: classifier_target.clone(),
                        targets: required_classifier_field(route_name, "targets", targets)?,
                        default_target: required_classifier_field(
                            route_name,
                            "default_target",
                            default_target,
                        )?,
                        prompt: required_classifier_field(route_name, "prompt", prompt)?,
                        response_schema: required_classifier_field(
                            route_name,
                            "response_schema",
                            response_schema,
                        )?,
                        policy: required_classifier_field(route_name, "policy", policy)?,
                        classify_trigger: *classify_trigger,
                        message_hash_fallback: *message_hash_fallback,
                        recent_turn_window: *recent_turn_window,
                        judge_char_budget: *judge_char_budget,
                        max_output_tokens: *max_output_tokens,
                    },
                ))
            }
        }
    }
}

fn reject_custom_fields(
    route_name: &str,
    mode: &str,
    targets: &Option<Vec<String>>,
    default_target: &Option<String>,
    response_schema: &Option<String>,
    policy: &Option<ClassifierPolicyConfig>,
) -> AlgorithmResult<()> {
    if targets.is_some()
        || default_target.is_some()
        || response_schema.is_some()
        || policy.is_some()
    {
        return Err(AlgorithmConfigError::new(format!(
            "llm_classifier route {route_name} mode {mode} cannot use custom classifier fields"
        )));
    }
    Ok(())
}

fn classifier_field_error(route_name: &str, field: &str, mode: &str) -> AlgorithmConfigError {
    AlgorithmConfigError::new(format!(
        "llm_classifier route {route_name} mode {mode} cannot use {field}"
    ))
}

fn required_classifier_field<T: Clone>(
    route_name: &str,
    field: &str,
    value: &Option<T>,
) -> AlgorithmResult<T> {
    value.clone().ok_or_else(|| {
        AlgorithmConfigError::new(format!(
            "llm_classifier route {route_name} requires {field}"
        ))
    })
}

fn build_subagent_router_config(
    route_name: &str,
    config: &SubagentRouteConfig,
    targets: &BTreeMap<String, ModelId>,
) -> AlgorithmResult<SubagentRouterConfig> {
    match config {
        SubagentRouteConfig::Passthrough { target } => Ok(SubagentRouterConfig::fixed_target(
            resolve_target_model_id(route_name, target, targets)?,
        )),
        SubagentRouteConfig::LlmClassifier(config) => {
            let LlmClassifierModeConfig::Custom(config) = config.classifier_mode(route_name)?
            else {
                return Err(AlgorithmConfigError::new(format!(
                    "route {route_name}: subagents llm_classifier only supports mode custom"
                )));
            };
            let judge_target =
                resolve_target_model_id(route_name, &config.classifier_target, targets)?;
            let resolved_targets = config
                .targets
                .iter()
                .map(|name| {
                    resolve_target_model_id(route_name, name, targets)
                        .map(|target| (name.clone(), target))
                })
                .collect::<AlgorithmResult<Vec<_>>>()?;
            let default_target = resolved_targets
                .iter()
                .find(|(name, _)| *name == config.default_target)
                .map(|(_, target)| target.clone())
                .ok_or_else(|| {
                    AlgorithmConfigError::new(format!(
                        "route {route_name}: subagents llm_classifier default_target {:?} must be one of its configured targets",
                        config.default_target
                    ))
                })?;
            let response_schema =
                serde_json::from_str(&config.response_schema).map_err(|error| {
                    AlgorithmConfigError::with_source(
                        format!(
                            "route {route_name}: subagents llm_classifier response_schema is invalid JSON: {error}"
                        ),
                        error,
                    )
                })?;
            let mut classifier_config = CustomClassifierConfig::new(
                config.prompt,
                response_schema,
                config.policy.into_libsy(),
            );
            classifier_config.recent_turn_window = config.recent_turn_window;
            classifier_config.judge_char_budget = config.judge_char_budget;
            classifier_config.max_output_tokens = config.max_output_tokens;
            let subagent_targets = resolved_targets
                .iter()
                .map(|(_, target)| target.clone())
                .collect();
            let classifier = Arc::new(
                LlmTaskClassifier::new(LlmClassifierConfig::Custom {
                    judge_target,
                    targets: resolved_targets,
                    default_target: config.default_target,
                    config: classifier_config,
                })
                .map_err(|error| {
                    AlgorithmConfigError::with_source(
                        format!("route {route_name}: subagents llm_classifier: {error}"),
                        error,
                    )
                })?,
            );
            Ok(SubagentRouterConfig {
                targets: subagent_targets,
                classifier,
                default_target,
                classify_trigger: config.classify_trigger,
                message_hash_fallback: config.message_hash_fallback,
            })
        }
    }
}

fn attach_subagent_router(
    route_name: &str,
    parent: Arc<dyn Algorithm>,
    config: Option<&SubagentRouteConfig>,
    targets: &BTreeMap<String, ModelId>,
) -> AlgorithmResult<Arc<dyn Algorithm>> {
    let Some(config) = config else {
        return Ok(parent);
    };
    let config = build_subagent_router_config(route_name, config, targets)?;
    let algorithm = SubagentRouter::new(parent, config).map_err(|error| {
        AlgorithmConfigError::with_source(
            format!("route {route_name}: subagent routing: {error}"),
            error,
        )
    })?;
    Ok(Arc::new(algorithm))
}

fn build_algorithm(
    route_name: &str,
    config: &AlgorithmSpec,
    targets: &BTreeMap<String, ModelId>,
) -> AlgorithmResult<Arc<dyn Algorithm>> {
    match config {
        AlgorithmSpec::Noop { .. } => Ok(Arc::new(Noop {})),
        AlgorithmSpec::Random {
            targets: names,
            weights,
            seed,
            ..
        } => {
            let target_set =
                resolve_targets(route_name, names.iter().map(String::as_str), targets)?;
            let algorithm = Random::new(target_set, weights.clone(), *seed).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("random route {route_name}: {error}"),
                    error,
                )
            })?;
            Ok(Arc::new(algorithm))
        }
        AlgorithmSpec::Passthrough {
            target, subagents, ..
        } => {
            let parent_target = resolve_target_model_id(route_name, target, targets)?;
            let algorithm = Passthrough::new(parent_target);
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::LlmClassifier {
            config: classifier_config,
            ..
        } => {
            let classifier =
                resolve_target_model_id(route_name, &classifier_config.classifier_target, targets)?;
            let mode = classifier_config.classifier_mode(route_name)?;
            let algorithm = match mode {
                LlmClassifierModeConfig::Capability(config) => {
                    let strong =
                        resolve_target_model_id(route_name, &config.strong_target, targets)?;
                    let weak = resolve_target_model_id(route_name, &config.weak_target, targets)?;
                    let classifier_config = TaskClassifierConfig {
                        base_threshold: config.base_threshold,
                        threshold_step: config.threshold_step,
                        classify_trigger: config.classify_trigger,
                        message_hash_fallback: config.message_hash_fallback,
                        recent_turn_window: config.recent_turn_window,
                        judge_char_budget: config.judge_char_budget,
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        max_output_tokens: config.max_output_tokens,
                    };
                    LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                        judge_target: classifier,
                        efficient_target: weak,
                        capable_target: strong,
                        config: classifier_config,
                    })
                }
                LlmClassifierModeConfig::Escalation(config) => {
                    let strong =
                        resolve_target_model_id(route_name, &config.strong_target, targets)?;
                    let weak = resolve_target_model_id(route_name, &config.weak_target, targets)?;
                    LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
                        judge_target: classifier,
                        efficient_target: weak,
                        capable_target: strong,
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        config: config.judge,
                        max_output_tokens: config.max_output_tokens,
                    })
                }
                LlmClassifierModeConfig::Custom(config) => {
                    let resolved_targets = config
                        .targets
                        .iter()
                        .map(|name| {
                            resolve_target_model_id(route_name, name, targets)
                                .map(|target| (name.clone(), target))
                        })
                        .collect::<AlgorithmResult<Vec<_>>>()?;
                    let response_schema = serde_json::from_str(&config.response_schema).map_err(
                        |error| {
                            AlgorithmConfigError::with_source(
                                format!(
                                    "llm_classifier route {route_name}: response_schema is invalid JSON: {error}"
                                ),
                                error,
                            )
                        },
                    )?;
                    let mut classifier_config = CustomClassifierConfig::new(
                        config.prompt,
                        response_schema,
                        config.policy.into_libsy(),
                    );
                    classifier_config.classify_trigger = config.classify_trigger;
                    classifier_config.message_hash_fallback = config.message_hash_fallback;
                    classifier_config.recent_turn_window = config.recent_turn_window;
                    classifier_config.max_output_tokens = config.max_output_tokens;
                    LlmTaskClassifier::new(LlmClassifierConfig::Custom {
                        judge_target: classifier,
                        targets: resolved_targets,
                        default_target: config.default_target,
                        config: classifier_config,
                    })
                }
            }
            .map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("llm_classifier route {route_name}: {error}"),
                    error,
                )
            })?;
            Ok(Arc::new(algorithm))
        }
        AlgorithmSpec::StageRouter {
            tiers,
            picker,
            classifier,
            subagents,
            ..
        } => {
            let StageTierConfig {
                capable_target,
                efficient_target,
                confidence_threshold,
                recent_turn_window,
                handoff_notes,
                capable_system_prompt,
                efficient_system_prompt,
            } = tiers;
            if matches!(picker, PickerMode::CapableFirst) {
                tracing::warn!(
                    "stage_router route {route_name} uses picker \"capable_first\", which is experimental: published thresholds and routing results all come from \"efficient_first\", so there is no calibrated confidence_threshold for it and no measured accuracy or cost. Use \"efficient_first\" unless you are running your own calibration."
                );
            }
            let capable = resolve_target_model_id(route_name, capable_target, targets)?;
            let efficient = resolve_target_model_id(route_name, efficient_target, targets)?;
            let mut config = StageRouterConfig::new(*picker, *confidence_threshold);
            config.recent_window = *recent_turn_window;
            config.handoff_notes = handoff_notes.clone();
            config.tier_prompts = tier_prompts(
                &capable,
                capable_system_prompt.as_deref(),
                &efficient,
                efficient_system_prompt.as_deref(),
            );
            // The judge is called through its own target, so it is not a routing
            // destination and stays out of the tier pair.
            config.llm_fallback = classifier
                .as_ref()
                .map(|classifier| {
                    resolve_target_model_id(route_name, &classifier.target, targets).map(
                        |judge_target| LlmFallback {
                            judge_target,
                            config: classifier.task_classifier_config(),
                        },
                    )
                })
                .transpose()?;
            let algorithm = StageRouter::new(capable, efficient, config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("stage_router route {route_name}: {error}"),
                    error,
                )
            })?;
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::Composite {
            classifier,
            stage,
            subagents,
        } => {
            let capable = resolve_target_model_id(route_name, &stage.capable_target, targets)?;
            let efficient = resolve_target_model_id(route_name, &stage.efficient_target, targets)?;
            let judge_target = resolve_target_model_id(route_name, &classifier.target, targets)?;
            let mut stage_config =
                StageRouterConfig::new(PickerMode::EfficientFirst, stage.confidence_threshold);
            stage_config.recent_window = stage.recent_turn_window;
            stage_config.handoff_notes = stage.handoff_notes.clone();
            stage_config.tier_prompts = tier_prompts(
                &capable,
                stage.capable_system_prompt.as_deref(),
                &efficient,
                stage.efficient_system_prompt.as_deref(),
            );
            let config = CompositeRouterConfig {
                judge_target,
                judge: classifier.task_classifier_config(),
                stage: stage_config,
            };
            let algorithm = CompositeRouter::new(capable, efficient, config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("composite route {route_name}: {error}"),
                    error,
                )
            })?;
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::Advisor {
            executor_target,
            advisor_target,
            reviewer_system_prompt,
            redo_feedback_prefix,
            gate_trigger,
            gate_trigger_pattern,
            max_reviews,
            gate_stall_turns,
            gate_min_tool_results,
            advisor_max_tokens,
            advisor_temperature,
            transcript_max_chars,
            fail_open,
            ..
        } => {
            let executor = resolve_target_model_id(route_name, executor_target, targets)?;
            let advisor = resolve_target_model_id(route_name, advisor_target, targets)?;
            // A pattern set under the default trigger would be silently
            // ignored; reject the misconfiguration instead.
            if *gate_trigger == AdvisorTriggerConfig::NoToolCall && gate_trigger_pattern.is_some() {
                return Err(AlgorithmConfigError::new(format!(
                    "advisor route {route_name}: gate_trigger_pattern requires \
                     gate_trigger = \"pattern\""
                )));
            }
            let mut config = AdvisorGateConfig::default();
            if let Some(prompt) = reviewer_system_prompt {
                config.reviewer_system_prompt = prompt.clone();
            }
            if let Some(prefix) = redo_feedback_prefix {
                config.redo_feedback_prefix = prefix.clone();
            }
            config.gate_trigger = match gate_trigger {
                AdvisorTriggerConfig::NoToolCall => GateTrigger::NoToolCall,
                AdvisorTriggerConfig::Pattern => {
                    GateTrigger::Pattern(gate_trigger_pattern.clone().unwrap_or_default())
                }
            };
            config.max_reviews = *max_reviews;
            config.gate_stall_turns = *gate_stall_turns;
            config.gate_min_tool_results = *gate_min_tool_results;
            config.advisor_max_tokens = *advisor_max_tokens;
            config.advisor_temperature = *advisor_temperature;
            config.transcript_max_chars = *transcript_max_chars;
            config.fail_open = *fail_open;
            let algorithm = AdvisorGate::new(executor, advisor, config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("advisor route {route_name}: {error}"),
                    error,
                )
            })?;
            Ok(Arc::new(algorithm))
        }
        AlgorithmSpec::PrefillRouter {
            targets: names,
            checkpoint,
            device,
            cache_dir,
            max_length,
            batch_size,
        } => {
            #[cfg(feature = "prefill-router")]
            {
                let targets =
                    resolve_targets(route_name, names.iter().map(String::as_str), targets)?;
                let mut config = prefill_router::PrefillRouterConfig::new(targets, checkpoint);
                config.device.clone_from(device);
                config.cache_dir.clone_from(cache_dir);
                if let Some(max_length) = max_length {
                    config.max_length = *max_length;
                }
                if let Some(batch_size) = batch_size {
                    config.batch_size = *batch_size;
                }
                let algorithm = config.build().map_err(|error| {
                    AlgorithmConfigError::with_source(
                        format!("prefill_router route {route_name}: {error}"),
                        error,
                    )
                })?;
                Ok(Arc::new(algorithm))
            }

            #[cfg(not(feature = "prefill-router"))]
            {
                let _ = (
                    names, checkpoint, device, cache_dir, max_length, batch_size, targets,
                );
                Err(AlgorithmConfigError::new(format!(
                    "prefill_router route {route_name} requires the `prefill-router` Cargo feature"
                )))
            }
        }
    }
}

const fn default_max_reviews() -> u32 {
    1
}

const fn default_advisor_max_tokens() -> u64 {
    2048
}

const fn default_transcript_max_chars() -> usize {
    200_000
}

const fn default_fail_open() -> bool {
    true
}

fn classifier_contract(prompt: Option<&str>) -> ClassifierContractConfig {
    prompt.map_or_else(ClassifierContractConfig::default, |prompt| {
        ClassifierContractConfig::default().with_prompt(prompt)
    })
}

fn default_classifier_max_output_tokens() -> u64 {
    TaskClassifierConfig::default().max_output_tokens
}

fn default_judge_char_budget() -> usize {
    TaskClassifierConfig::default().judge_char_budget
}

/// Keys each configured system prompt by the target it belongs to.
fn tier_prompts(
    capable: &str,
    capable_prompt: Option<&str>,
    efficient: &str,
    efficient_prompt: Option<&str>,
) -> TargetPrompts {
    let mut prompts = TargetPrompts::default();
    if let Some(prompt) = capable_prompt {
        prompts = prompts.with(capable, prompt);
    }
    if let Some(prompt) = efficient_prompt {
        prompts = prompts.with(efficient, prompt);
    }
    prompts
}

fn resolve_targets<'a>(
    route_name: &str,
    names: impl IntoIterator<Item = &'a str>,
    targets: &BTreeMap<String, ModelId>,
) -> AlgorithmResult<Vec<ModelId>> {
    names
        .into_iter()
        .map(|name| resolve_target_model_id(route_name, name, targets))
        .collect()
}

fn resolve_target_model_id(
    route_name: &str,
    name: &str,
    targets: &BTreeMap<String, ModelId>,
) -> AlgorithmResult<ModelId> {
    targets.get(name).cloned().ok_or_else(|| {
        AlgorithmConfigError::new(format!(
            "route {route_name} references unknown target {name}"
        ))
    })
}

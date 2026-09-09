// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-neutral algorithm configuration and construction.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;

use libsy::{
    AdvisorGate, AdvisorGateConfig, Algorithm, ClassifierContractConfig, ClassifierResponseFormat,
    ClassifyTrigger, CompositeRouter, CompositeRouterConfig, CustomClassifierConfig,
    CustomClassifierPolicy, EscalationJudgeConfig, GateTrigger, HandoffNoteConfig,
    LlmClassifierConfig, LlmFallback, LlmTaskClassifier, Noop, Passthrough, PickerMode, Random,
    StageRouter, StageRouterConfig, SubagentRouter, SubagentRouterConfig, TaskClassifierConfig,
    ToolSemantics,
};
use serde::Deserialize;
use switchyard_protocol::{Category, ModelId};

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
    classifier_target: String,
    strong_target: String,
    weak_target: String,
    base_threshold: f64,
    threshold_step: f64,
    classify_trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
}

#[derive(Clone, Debug)]
struct EscalationClassifierRouteConfig {
    classifier_target: String,
    strong_target: String,
    weak_target: String,
    prompt: Option<String>,
    response_format_type: ClassifierResponseFormat,
    max_output_tokens: u64,
    judge: EscalationJudgeConfig,
}

#[derive(Clone, Debug)]
struct CustomClassifierRouteConfig {
    models: CategoryModelConfig,
    default_target: Category,
    prompt: String,
    response_schema: String,
    policy: ClassifierPolicyConfig,
    classify_trigger: ClassifyTrigger,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
    max_output_tokens: u64,
}

/// Runtime model groups for a custom classifier.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CategoryModelConfig {
    /// All completion targets in fallback order.
    pub any: Vec<String>,
    /// Ordered candidates for judge calls.
    pub judge: Vec<String>,
    /// Ordered capable-tier models.
    pub capable: Vec<String>,
    /// Ordered efficient-tier models.
    pub efficient: Vec<String>,
}

impl CategoryModelConfig {
    fn validate(&self, route_name: &str, default_target: Category) -> AlgorithmResult<()> {
        for (category, models) in [(Category::Any, &self.any), (Category::Judge, &self.judge)] {
            if models.is_empty() {
                return Err(AlgorithmConfigError::new(format!(
                    "llm_classifier route {route_name} models.{} must contain at least one target",
                    category.as_str()
                )));
            }
        }
        if self.get(default_target).is_empty() {
            return Err(AlgorithmConfigError::new(format!(
                "llm_classifier route {route_name} models.{} must contain at least one target because it is the default_target",
                default_target.as_str()
            )));
        }
        Ok(())
    }

    fn get(&self, category: Category) -> &Vec<String> {
        match category {
            Category::Any => &self.any,
            Category::Judge => &self.judge,
            Category::Capable => &self.capable,
            Category::Efficient => &self.efficient,
        }
    }

    fn all_names(&self) -> Vec<&str> {
        self.any
            .iter()
            .chain(&self.judge)
            .chain(&self.capable)
            .chain(&self.efficient)
            .map(String::as_str)
            .collect()
    }
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
    /// Custom mode: runtime model groups.
    pub models: Option<CategoryModelConfig>,
    /// Custom mode: category used when the judge fails or its verdict cannot be routed.
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
                .models
                .as_ref()
                .map(CategoryModelConfig::all_names)
                .unwrap_or_default(),
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
    /// Picks a routing strategy automatically, preset with recommended knobs.
    /// Currently a `stage_router` with `picker = "efficient_first"` and
    /// `confidence_threshold = 0.5`; change `build_algorithm`'s `Auto` arm to
    /// repoint it at a different algorithm or preset.
    Auto {
        /// The capable tier.
        capable_target: String,
        /// The efficient tier.
        efficient_target: String,
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
    /// Exact tool-name semantics added to the built-in stage vocabulary.
    #[serde(default)]
    pub tool_semantics: ToolSemantics,
    /// Notes handed to a tier when the router switches to it.
    #[serde(default)]
    pub handoff_notes: Option<HandoffNoteConfig>,
}

impl StageClassifierConfig {
    fn task_classifier_config(&self) -> TaskClassifierConfig {
        TaskClassifierConfig {
            base_threshold: self.base_threshold,
            threshold_step: self.threshold_step,
            classify_trigger: self.classify_trigger,
            message_hash_fallback: self.message_hash_fallback,
            recent_turn_window: self.recent_turn_window,
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
            Self::LlmClassifier { config, .. } => match config.classifier_mode() {
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
                    .models
                    .as_ref()
                    .map(CategoryModelConfig::all_names)
                    .unwrap_or_default(),
            },
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
            Self::Auto {
                capable_target,
                efficient_target,
            } => vec![capable_target.as_str(), efficient_target.as_str()],
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
            Self::LlmClassifier { config, .. }
                if !matches!(config.classifier_mode(), ClassifierMode::Custom) =>
            {
                names.push(&config.classifier_target)
            }
            Self::StageRouter {
                classifier: Some(classifier),
                ..
            } => names.push(&classifier.target),
            Self::Composite { classifier, .. } => {
                names.push(&classifier.target);
            }
            Self::Advisor { advisor_target, .. } => names.push(advisor_target),
            _ => {}
        }
        names
    }

    /// Target names grouped as the runtime [`Driver`](libsy::Driver) expects them.
    pub(crate) fn runtime_model_names(
        &self,
        route_name: &str,
    ) -> AlgorithmResult<RuntimeModelNames> {
        let parent = match self {
            Self::Noop { .. } => HashMap::new(),
            Self::Random { targets, .. } | Self::PrefillRouter { targets, .. } => {
                category_models([(Category::Any, targets.clone())])
            }
            Self::Passthrough { target, .. } => {
                category_models([(Category::Any, vec![target.clone()])])
            }
            Self::LlmClassifier { config } => {
                classifier_runtime_model_names(config.validated_classifier_mode(route_name)?)
            }
            Self::StageRouter {
                tiers, classifier, ..
            } => {
                let mut models = category_models([
                    (Category::Capable, vec![tiers.capable_target.clone()]),
                    (Category::Efficient, vec![tiers.efficient_target.clone()]),
                    (
                        Category::Any,
                        vec![tiers.capable_target.clone(), tiers.efficient_target.clone()],
                    ),
                ]);
                if let Some(classifier) = classifier {
                    models.insert(Category::Judge, vec![classifier.target.clone()]);
                }
                models
            }
            Self::Auto {
                capable_target,
                efficient_target,
            } => category_models([
                (Category::Capable, vec![capable_target.clone()]),
                (Category::Efficient, vec![efficient_target.clone()]),
                (
                    Category::Any,
                    vec![capable_target.clone(), efficient_target.clone()],
                ),
            ]),
            Self::Composite {
                classifier, stage, ..
            } => category_models([
                (Category::Judge, vec![classifier.target.clone()]),
                (Category::Capable, vec![stage.capable_target.clone()]),
                (Category::Efficient, vec![stage.efficient_target.clone()]),
                (
                    Category::Any,
                    vec![stage.capable_target.clone(), stage.efficient_target.clone()],
                ),
            ]),
            Self::Advisor {
                executor_target,
                advisor_target,
                ..
            } => category_models([
                (Category::Any, vec![executor_target.clone()]),
                (Category::Judge, vec![advisor_target.clone()]),
            ]),
        };

        let subagents = match self {
            Self::Passthrough { subagents, .. }
            | Self::StageRouter { subagents, .. }
            | Self::Composite { subagents, .. } => subagents.as_ref(),
            _ => None,
        };
        // Sub-agent groups stay separate from the parent's. Merging them would let a
        // sub-agent's `capable` resolve to the parent's, and would put models only the
        // sub-agents were given into the parent's `Any` fallback list.
        let subagent = subagents
            .map(|subagents| subagent_runtime_model_names(subagents, route_name))
            .transpose()?;
        Ok(RuntimeModelNames { parent, subagent })
    }

    /// Response target and routing-only dependency for routers that answer while routing.
    pub(crate) fn routing_response_and_dependency(&self) -> Option<(&str, &str)> {
        match self {
            Self::LlmClassifier { config, .. }
                if matches!(config.classifier_mode(), ClassifierMode::Escalation) =>
            {
                Some((
                    config.weak_target.as_deref()?,
                    config.classifier_target.as_str(),
                ))
            }
            Self::Advisor {
                executor_target,
                advisor_target,
                ..
            } => Some((executor_target, advisor_target)),
            Self::Noop { .. }
            | Self::Random { .. }
            | Self::Passthrough { .. }
            | Self::LlmClassifier { .. }
            | Self::StageRouter { .. }
            | Self::Auto { .. }
            | Self::Composite { .. }
            | Self::PrefillRouter { .. } => None,
        }
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

/// One route's target names, grouped by category and by routing scope.
pub(crate) struct RuntimeModelNames {
    /// Groups the algorithm itself routes over.
    pub(crate) parent: HashMap<Category, Vec<String>>,
    /// Groups delegated sub-agent work routes over, when the route has a `subagents` table.
    pub(crate) subagent: Option<HashMap<Category, Vec<String>>>,
}

fn category_models(
    entries: impl IntoIterator<Item = (Category, Vec<String>)>,
) -> HashMap<Category, Vec<String>> {
    entries.into_iter().collect()
}

fn custom_runtime_model_names(config: &CategoryModelConfig) -> HashMap<Category, Vec<String>> {
    [
        Category::Any,
        Category::Judge,
        Category::Capable,
        Category::Efficient,
    ]
    .into_iter()
    .map(|category| (category, config.get(category).clone()))
    .collect()
}

fn classifier_runtime_model_names(
    config: LlmClassifierModeConfig,
) -> HashMap<Category, Vec<String>> {
    match config {
        LlmClassifierModeConfig::Capability(config) => category_models([
            (Category::Judge, vec![config.classifier_target]),
            (Category::Efficient, vec![config.weak_target.clone()]),
            (Category::Capable, vec![config.strong_target.clone()]),
            (
                Category::Any,
                vec![config.weak_target, config.strong_target],
            ),
        ]),
        LlmClassifierModeConfig::Escalation(config) => category_models([
            (Category::Judge, vec![config.classifier_target]),
            (Category::Efficient, vec![config.weak_target.clone()]),
            (Category::Capable, vec![config.strong_target.clone()]),
            (
                Category::Any,
                vec![config.strong_target, config.weak_target],
            ),
        ]),
        LlmClassifierModeConfig::Custom(config) => custom_runtime_model_names(&config.models),
    }
}

fn subagent_runtime_model_names(
    config: &SubagentRouteConfig,
    route_name: &str,
) -> AlgorithmResult<HashMap<Category, Vec<String>>> {
    match config {
        SubagentRouteConfig::Passthrough { target } => {
            Ok(category_models([(Category::Any, vec![target.clone()])]))
        }
        SubagentRouteConfig::LlmClassifier(config) => {
            let LlmClassifierModeConfig::Custom(config) =
                config.validated_classifier_mode(route_name)?
            else {
                return Err(AlgorithmConfigError::new(format!(
                    "route {route_name}: subagents llm_classifier only supports mode custom"
                )));
            };
            Ok(custom_runtime_model_names(&config.models))
        }
    }
}
impl LlmClassifierRouteConfig {
    fn classifier_mode(&self) -> ClassifierMode {
        let default = if self.escalation.is_some() {
            ClassifierMode::Escalation
        } else {
            ClassifierMode::Capability
        };
        self.mode.unwrap_or(default)
    }

    fn validated_classifier_mode(
        &self,
        route_name: &str,
    ) -> AlgorithmResult<LlmClassifierModeConfig> {
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
            prompt,
            response_format_type,
            max_output_tokens,
            escalation,
            models,
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
                    models,
                    default_target,
                    response_schema,
                    policy,
                )?;
                Ok(LlmClassifierModeConfig::Capability(
                    CapabilityClassifierRouteConfig {
                        classifier_target: classifier_target.clone(),
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
                    models,
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
                        classifier_target: classifier_target.clone(),
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
                if !classifier_target.is_empty()
                    || strong_target.is_some()
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
                let models = required_classifier_field(route_name, "models", models)?;
                let default_target = required_classifier_field(
                    route_name,
                    "default_target",
                    default_target,
                )?
                .parse()
                .map_err(|error| {
                    AlgorithmConfigError::new(format!(
                        "llm_classifier route {route_name} has invalid default_target: {error}"
                    ))
                })?;
                if default_target == Category::Judge {
                    return Err(AlgorithmConfigError::new(format!(
                        "llm_classifier route {route_name} default_target cannot be judge"
                    )));
                }
                models.validate(route_name, default_target)?;
                Ok(LlmClassifierModeConfig::Custom(
                    CustomClassifierRouteConfig {
                        models,
                        default_target,
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
    models: &Option<CategoryModelConfig>,
    default_target: &Option<String>,
    response_schema: &Option<String>,
    policy: &Option<ClassifierPolicyConfig>,
) -> AlgorithmResult<()> {
    if models.is_some() || default_target.is_some() || response_schema.is_some() || policy.is_some()
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
        SubagentRouteConfig::Passthrough { target } => {
            resolve_target_model_id(route_name, target, targets)?;
            Ok(SubagentRouterConfig::fixed_target())
        }
        SubagentRouteConfig::LlmClassifier(config) => {
            let LlmClassifierModeConfig::Custom(config) =
                config.validated_classifier_mode(route_name)?
            else {
                return Err(AlgorithmConfigError::new(format!(
                    "route {route_name}: subagents llm_classifier only supports mode custom"
                )));
            };
            for name in config.models.all_names() {
                resolve_target_model_id(route_name, name, targets)?;
            }
            if config.models.get(config.default_target).is_empty() {
                return Err(AlgorithmConfigError::new(format!(
                    "route {route_name}: subagents llm_classifier has no model for default category {}",
                    config.default_target.as_str()
                )));
            }
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
            classifier_config.max_output_tokens = config.max_output_tokens;
            let classifier = Arc::new(
                LlmTaskClassifier::new(LlmClassifierConfig::Custom {
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
                classifier,
                default_target: config.default_target,
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
        AlgorithmSpec::Random { weights, seed, .. } => {
            let algorithm = Random::new(weights.clone(), *seed).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("random route {route_name}: {error}"),
                    error,
                )
            })?;
            Ok(Arc::new(algorithm))
        }
        AlgorithmSpec::Passthrough { subagents, .. } => {
            let algorithm = Passthrough;
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::LlmClassifier {
            config: classifier_config,
            ..
        } => {
            let mode = classifier_config.validated_classifier_mode(route_name)?;
            let algorithm = match mode {
                LlmClassifierModeConfig::Capability(config) => {
                    let classifier_config = TaskClassifierConfig {
                        base_threshold: config.base_threshold,
                        threshold_step: config.threshold_step,
                        classify_trigger: config.classify_trigger,
                        message_hash_fallback: config.message_hash_fallback,
                        recent_turn_window: config.recent_turn_window,
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        max_output_tokens: config.max_output_tokens,
                    };
                    LlmTaskClassifier::new(LlmClassifierConfig::Capability {
                        config: classifier_config,
                    })
                }
                LlmClassifierModeConfig::Escalation(config) => {
                    LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
                        contract: classifier_contract(config.prompt.as_deref())
                            .with_response_format_type(config.response_format_type),
                        config: config.judge,
                        max_output_tokens: config.max_output_tokens,
                    })
                }
                LlmClassifierModeConfig::Custom(config) => {
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
                confidence_threshold,
                recent_turn_window,
                tool_semantics,
                handoff_notes,
                ..
            } = tiers;
            if matches!(picker, PickerMode::CapableFirst) {
                tracing::warn!(
                    "stage_router route {route_name} uses picker \"capable_first\", which is experimental: published thresholds and routing results all come from \"efficient_first\", so there is no calibrated confidence_threshold for it and no measured accuracy or cost. Use \"efficient_first\" unless you are running your own calibration."
                );
            }
            let mut config = StageRouterConfig::new(*picker, *confidence_threshold);
            config.recent_window = *recent_turn_window;
            config.tool_semantics = tool_semantics.clone();
            config.handoff_notes = handoff_notes.clone();
            // The judge is called through its own target, so it is not a routing
            // destination and stays out of the tier pair.
            config.llm_fallback = classifier.as_ref().map(|classifier| LlmFallback {
                config: classifier.task_classifier_config(),
            });
            let algorithm = StageRouter::new(config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("stage_router route {route_name}: {error}"),
                    error,
                )
            })?;
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::Auto { .. } => {
            let config = StageRouterConfig::new(PickerMode::EfficientFirst, 0.5);
            let algorithm = StageRouter::new(config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("auto route {route_name}: {error}"),
                    error,
                )
            })?;
            Ok(Arc::new(algorithm))
        }
        AlgorithmSpec::Composite {
            classifier,
            stage,
            subagents,
        } => {
            let mut stage_config =
                StageRouterConfig::new(PickerMode::EfficientFirst, stage.confidence_threshold);
            stage_config.recent_window = stage.recent_turn_window;
            stage_config.tool_semantics = stage.tool_semantics.clone();
            stage_config.handoff_notes = stage.handoff_notes.clone();
            let config = CompositeRouterConfig {
                judge: classifier.task_classifier_config(),
                stage: stage_config,
            };
            let algorithm = CompositeRouter::new(config).map_err(|error| {
                AlgorithmConfigError::with_source(
                    format!("composite route {route_name}: {error}"),
                    error,
                )
            })?;
            let parent: Arc<dyn Algorithm> = Arc::new(algorithm);
            attach_subagent_router(route_name, parent, subagents.as_ref(), targets)
        }
        AlgorithmSpec::Advisor {
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
            let algorithm = AdvisorGate::new(config).map_err(|error| {
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

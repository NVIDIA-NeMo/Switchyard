// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signal stage_router profile implemented as a profile-owned Rust runtime.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use switchyard_components::dimension_collector::{
    extract_tool_signals_with_window, ToolResultSignal, DEFAULT_RECENT_WINDOW,
};
use switchyard_components::stats::{usage_from_body, TokenUsage};
use switchyard_components::StatsAccumulator;
use switchyard_core::{ChatRequest, ChatResponse, LlmTarget, LlmTargetId, Result, SwitchyardError};

use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::{
    decision_for_stage_router_routing, profile_config, DecisionContext, FeatureFreshness, Profile,
    ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse, RelaySnapshot, RoutingDecision,
    RoutingMetadata,
};

const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.7;
const DEFAULT_CLASSIFIER_TIMEOUT_SECS: f64 = 30.0;
const DEFAULT_CLASSIFIER_RECENT_TURN_WINDOW: usize = 3;
const CLASSIFIER_MAX_TOKENS: u32 = 4096;
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const STAGE_ROUTER_PROFILE_TYPE: &str = "stage_router";

const SEVERITY_CRITICAL: f32 = 1.0;
const CLEAN_TESTS_MIN_TURN_DEPTH: u32 = 10;
const CLEAN_TESTS_MAX_WRITES: u32 = 1;
const PURE_BASH_NORM: f64 = 8.0;

const CLASSIFIER_SYSTEM_PROMPT: &str = include_str!("stage_router/prompts/classifier.md");

const DEFAULT_WEIGHTS: &[(&str, f64)] = &[
    ("severity", 0.80),
    ("stuck_exploring", 0.70),
    ("no_progress", 0.60),
    ("tests_passed", -0.80),
    ("planning_active", -0.70),
    ("write_intensity", -0.40),
    ("edit_intensity", -0.30),
    ("recent_write_intensity", -0.30),
    ("pure_bash_intensity", -0.30),
    ("no_error_streak_intensity", -0.20),
];

/// Default picker mode names the tier used when the scorer is ambiguous.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRouterPickerMode {
    /// Default to capable unless the scorer/classifier confidently picks efficient.
    CapableFirst,
    /// Default to efficient unless the scorer/classifier confidently picks capable.
    EfficientFirst,
}

impl StageRouterPickerMode {
    fn default_tier(self) -> StageRouterTier {
        match self {
            Self::CapableFirst => StageRouterTier::Capable,
            Self::EfficientFirst => StageRouterTier::Efficient,
        }
    }
}

/// Optional LLM classifier invoked for low-confidence scorer outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRouterClassifierConfig {
    /// OpenAI-compatible classifier model.
    pub model: String,
    /// API key used for the classifier call.
    pub api_key: String,
    /// OpenAI-compatible base URL. Defaults to OpenAI's `/v1` endpoint.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Per-call timeout in seconds.
    #[serde(default = "default_classifier_timeout_secs")]
    pub timeout_secs: f64,
    /// Number of trailing request messages rendered into the classifier prompt.
    #[serde(default = "default_classifier_recent_turn_window")]
    pub recent_turn_window: usize,
    /// Maximum tokens allowed for the classifier response.
    #[serde(default = "default_classifier_max_tokens")]
    pub max_tokens: u32,
    /// Optional system prompt override for the classifier request.
    #[serde(default)]
    pub system_prompt: Option<String>,
}

/// Config for a capable/efficient signal stage_router profile.
#[profile_config("stage_router")]
pub struct StageRouterProfileConfig {
    /// Capable target served by this profile.
    #[profile_target]
    pub capable: LlmTarget,
    /// Efficient target served by this profile.
    #[profile_target]
    pub efficient: LlmTarget,
    /// Target used for one retry after context-window overflow.
    pub fallback_target_on_evict: LlmTargetId,
    /// Picker mode controlling the low-confidence default tier.
    #[serde(default = "default_picker")]
    pub picker: StageRouterPickerMode,
    /// Scorer confidence threshold in `[0.0, 1.0]`.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    /// Sliding window for `recent_*` tool-result signal counts.
    #[serde(default = "default_signal_recent_window")]
    pub signal_recent_window: usize,
    /// Optional LLM classifier for ambiguous scorer outputs.
    #[serde(default)]
    pub classifier: Option<StageRouterClassifierConfig>,
    /// Whether to emit stats for this profile.
    #[serde(default = "default_enable_stats")]
    pub enable_stats: bool,
}

impl ProfileConfig for StageRouterProfileConfig {
    type Runtime = StageRouterProfile;

    /// Builds the runtime profile using native target backends.
    fn build(&self) -> Result<Self::Runtime> {
        self.validate()?;
        Ok(StageRouterProfile {
            capable_backend: native_target_backend(self.capable.clone())?,
            efficient_backend: native_target_backend(self.efficient.clone())?,
            fallback_target_on_evict: self.fallback_target_on_evict.clone(),
            picker: self.picker,
            confidence_threshold: self.confidence_threshold,
            signal_recent_window: self.signal_recent_window,
            classifier: self
                .classifier
                .as_ref()
                .map(StageRouterTierClassifier::new)
                .transpose()?,
            stats: profile_stats_accumulator(),
            enable_stats: self.enable_stats,
        })
    }
}

impl StageRouterProfileConfig {
    fn validate(&self) -> Result<()> {
        if self.capable.id == self.efficient.id {
            return Err(SwitchyardError::InvalidConfig(
                "stage_router capable and efficient targets must have distinct target ids".to_string(),
            ));
        }
        if !self.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.confidence_threshold)
        {
            return Err(SwitchyardError::InvalidConfig(format!(
                "confidence_threshold must be finite and in [0.0, 1.0], got {:?}",
                self.confidence_threshold
            )));
        }
        if self.signal_recent_window == 0 {
            return Err(SwitchyardError::InvalidConfig(
                "signal_recent_window must be at least 1".to_string(),
            ));
        }
        if self.fallback_target_on_evict != self.capable.id
            && self.fallback_target_on_evict != self.efficient.id
        {
            return Err(SwitchyardError::InvalidConfig(format!(
                "fallback_target_on_evict={} must match one of [{}, {}]",
                self.fallback_target_on_evict, self.efficient.id, self.capable.id
            )));
        }
        if let Some(classifier) = &self.classifier {
            classifier.validate()?;
        }
        Ok(())
    }
}

impl StageRouterClassifierConfig {
    fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(SwitchyardError::InvalidConfig(
                "classifier.model must not be empty".to_string(),
            ));
        }
        if !self.timeout_secs.is_finite() || self.timeout_secs <= 0.0 {
            return Err(SwitchyardError::InvalidConfig(format!(
                "classifier.timeout_secs must be finite and > 0.0, got {:?}",
                self.timeout_secs
            )));
        }
        if self.max_tokens == 0 {
            return Err(SwitchyardError::InvalidConfig(
                "classifier.max_tokens must be greater than 0".to_string(),
            ));
        }
        if let Some(system_prompt) = &self.system_prompt {
            if system_prompt.trim().is_empty() {
                return Err(SwitchyardError::InvalidConfig(
                    "classifier.system_prompt must not be empty".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Capable/efficient stage_router profile runtime.
pub struct StageRouterProfile {
    capable_backend: TargetBackend,
    efficient_backend: TargetBackend,
    fallback_target_on_evict: LlmTargetId,
    picker: StageRouterPickerMode,
    confidence_threshold: f64,
    signal_recent_window: usize,
    classifier: Option<StageRouterTierClassifier>,
    stats: StatsAccumulator,
    enable_stats: bool,
}

/// Processed stage_router request with profile-owned decision state.
pub struct StageRouterProcessedRequest {
    /// Routed input prepared for the selected backend.
    pub profile_input: ProfileInput,
    /// Selected routing decision for this request.
    pub decision: StageRouterDecision,
}

/// Named side of a stage_router decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRouterTier {
    /// Capable tier.
    Capable,
    /// Efficient tier.
    Efficient,
}

impl StageRouterTier {
    /// Stable lowercase label used by stats.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Capable => "capable",
            Self::Efficient => "efficient",
        }
    }
}

/// Source that produced a stage_router decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageRouterDecisionSource {
    /// Hard override fired.
    Override,
    /// Dimension scorer crossed `confidence_threshold`.
    Dimensions,
    /// LLM classifier returned a usable verdict.
    #[serde(rename = "llm-classifier")]
    LlmClassifier,
    /// Classifier was absent or failed; picker default tier was used.
    FallOpen,
    /// A context-window overflow retried the configured fallback target.
    ContextOverflowFallback,
}

impl StageRouterDecisionSource {
    /// Stable lowercase label used in stats JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::Dimensions => "dimensions",
            Self::LlmClassifier => "llm-classifier",
            Self::FallOpen => "fall_open",
            Self::ContextOverflowFallback => "context_overflow_fallback",
        }
    }
}

/// StageRouter routing decision with the selected target and scorer metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageRouterDecision {
    /// Selected capable/efficient side.
    pub tier: StageRouterTier,
    /// Selected target id.
    pub selected_target: LlmTargetId,
    /// Selected upstream model.
    pub selected_model: switchyard_core::ModelId,
    /// Client-provided model before routing.
    pub original_model: Option<String>,
    /// Decision source for observability.
    pub source: StageRouterDecisionSource,
    /// Linear scorer value in `[-1.0, 1.0]`.
    pub score: f64,
    /// Router confidence when the decision source produced one.
    pub confidence: Option<f64>,
}

impl StageRouterProfile {
    async fn route_request(&self, mut input: ProfileInput) -> Result<StageRouterProcessedRequest> {
        let signal = extract_tool_signals_with_window(&input.request, self.signal_recent_window);
        let original_model = input.request.model().map(std::borrow::ToOwned::to_owned);
        let decision = self.pick(&input.request, &signal, original_model).await?;
        input.request.set_model(decision.selected_model.as_str());
        self.record_decision_source(decision.source)?;
        Ok(StageRouterProcessedRequest {
            profile_input: input,
            decision,
        })
    }

    /// Routes an immutable Decision snapshot without dispatching the selected target.
    pub(crate) async fn process_decision_snapshot(
        &self,
        mut input: ProfileInput,
        snapshot: Option<&RelaySnapshot>,
    ) -> Result<(StageRouterProcessedRequest, Option<FeatureFreshness>)> {
        let original_model = input.request.model().map(std::borrow::ToOwned::to_owned);
        let (decision, freshness) = match self.signal_from_relay_snapshot(snapshot) {
            Some(signal) => (
                self.pick(&input.request, &signal, original_model).await?,
                Some(FeatureFreshness::Fresh),
            ),
            None => (
                self.decision_for_tier(
                    self.picker.default_tier(),
                    original_model,
                    StageRouterDecisionSource::FallOpen,
                    0.0,
                    None,
                )?,
                None,
            ),
        };
        input.request.set_model(decision.selected_model.as_str());
        self.record_decision_source(decision.source)?;
        Ok((
            StageRouterProcessedRequest {
                profile_input: input,
                decision,
            },
            freshness,
        ))
    }

    // Projects router-neutral Relay history into StageRouter's existing feature model.
    fn signal_from_relay_snapshot(
        &self,
        snapshot: Option<&RelaySnapshot>,
    ) -> Option<ToolResultSignal> {
        let snapshot = snapshot.filter(|snapshot| !snapshot.messages.is_empty())?;
        let request = ChatRequest::openai_chat(json!({
            "model": "stage_router-relay-snapshot",
            "messages": snapshot.messages,
        }));
        let mut signal = extract_tool_signals_with_window(&request, self.signal_recent_window);
        signal.turn_depth = u32::try_from(snapshot.turn_depth).unwrap_or(u32::MAX);
        Some(signal)
    }

    // Routing decision flow:
    // 1. Hard overrides choose capable for critical failures or efficient for clean tests.
    // 2. Dimensions scoring chooses by score sign when confidence clears the threshold.
    // 3. Low-confidence requests use the optional LLM classifier when configured.
    // 4. Missing or failed classifier output falls open to the picker default tier.
    async fn pick(
        &self,
        request: &ChatRequest,
        signal: &ToolResultSignal,
        original_model: Option<String>,
    ) -> Result<StageRouterDecision> {
        if let Some(tier) = apply_overrides(signal) {
            return self.decision_for_tier(
                tier,
                original_model,
                StageRouterDecisionSource::Override,
                0.0,
                Some(1.0),
            );
        }

        let score = score_signal(signal);
        if score.confidence >= self.confidence_threshold {
            let tier = if score.score > 0.0 {
                StageRouterTier::Capable
            } else {
                StageRouterTier::Efficient
            };
            return self.decision_for_tier(
                tier,
                original_model,
                StageRouterDecisionSource::Dimensions,
                score.score,
                Some(score.confidence),
            );
        }

        if let Some(classifier) = &self.classifier {
            if let Some(tier) = classifier
                .classify(request, signal, self.stats_handle())
                .await
            {
                return self.decision_for_tier(
                    tier,
                    original_model,
                    StageRouterDecisionSource::LlmClassifier,
                    score.score,
                    None,
                );
            }
        }

        self.decision_for_tier(
            self.picker.default_tier(),
            original_model,
            StageRouterDecisionSource::FallOpen,
            score.score,
            Some(score.confidence),
        )
    }

    fn decision_for_tier(
        &self,
        tier: StageRouterTier,
        original_model: Option<String>,
        source: StageRouterDecisionSource,
        score: f64,
        confidence: Option<f64>,
    ) -> Result<StageRouterDecision> {
        let backend = self.backend_for_tier(tier);
        let target = backend.target();
        Ok(StageRouterDecision {
            tier,
            selected_target: target.id.clone(),
            selected_model: target.model.clone(),
            original_model,
            source,
            score,
            confidence,
        })
    }

    fn fallback_decision(&self, decision: &StageRouterDecision) -> Result<StageRouterDecision> {
        let backend = self.backend_for_target(&self.fallback_target_on_evict)?;
        let target = backend.target();
        Ok(StageRouterDecision {
            tier: self.tier_for_target(&target.id)?,
            selected_target: target.id.clone(),
            selected_model: target.model.clone(),
            original_model: decision.original_model.clone(),
            source: StageRouterDecisionSource::ContextOverflowFallback,
            score: decision.score,
            confidence: decision.confidence,
        })
    }

    fn retry_processed_request(
        &self,
        processed: &StageRouterProcessedRequest,
    ) -> Result<StageRouterProcessedRequest> {
        let decision = self.fallback_decision(&processed.decision)?;
        let mut profile_input = processed.profile_input.clone();
        profile_input
            .request
            .set_model(decision.selected_model.as_str());
        Ok(StageRouterProcessedRequest {
            profile_input,
            decision,
        })
    }

    fn backend_for_tier(&self, tier: StageRouterTier) -> &TargetBackend {
        match tier {
            StageRouterTier::Capable => &self.capable_backend,
            StageRouterTier::Efficient => &self.efficient_backend,
        }
    }

    fn backend_for_target(&self, target_id: &LlmTargetId) -> Result<&TargetBackend> {
        if *target_id == self.capable_backend.target().id {
            Ok(&self.capable_backend)
        } else if *target_id == self.efficient_backend.target().id {
            Ok(&self.efficient_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "stage_router selected target {target_id} that is not configured for this profile"
            )))
        }
    }

    /// Returns selected target metadata for a decision-only integration.
    pub(crate) fn target_for_decision(&self, decision: &StageRouterDecision) -> Result<&LlmTarget> {
        self.backend_for_target(&decision.selected_target)
            .map(TargetBackend::target)
    }

    fn tier_for_target(&self, target_id: &LlmTargetId) -> Result<StageRouterTier> {
        if *target_id == self.capable_backend.target().id {
            Ok(StageRouterTier::Capable)
        } else if *target_id == self.efficient_backend.target().id {
            Ok(StageRouterTier::Efficient)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "stage_router target {target_id} is not configured for this profile"
            )))
        }
    }

    async fn call_selected(
        &self,
        processed: &StageRouterProcessedRequest,
    ) -> (Result<ChatResponse>, f64) {
        let started_at = Instant::now();
        let backend = match self.backend_for_target(&processed.decision.selected_target) {
            Ok(backend) => backend,
            Err(error) => return (Err(error), 0.0),
        };
        let result = backend.call(&processed.profile_input.request).await;
        let latency_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        (result, latency_ms)
    }

    fn stats_handle(&self) -> Option<&StatsAccumulator> {
        self.enable_stats.then_some(&self.stats)
    }

    fn record_decision_source(&self, source: StageRouterDecisionSource) -> Result<()> {
        if let Some(stats) = self.stats_handle() {
            stats.record_routing_decision(STAGE_ROUTER_PROFILE_TYPE, source.as_str())?;
        }
        Ok(())
    }

    fn record_success(
        &self,
        decision: &StageRouterDecision,
        response: &ChatResponse,
        total_latency_ms: f64,
        backend_latency_ms: f64,
    ) -> Result<()> {
        if let Some(stats) = self.stats_handle() {
            stats.record_success(
                decision.selected_model.as_str(),
                Some(backend_latency_ms),
                Some(decision.tier.as_str()),
            )?;
            let routing_overhead_ms = (total_latency_ms - backend_latency_ms).max(0.0);
            let usage = response.body().map(usage_from_body).unwrap_or_default();
            stats.record_usage_after_success_attribution(
                decision.selected_model.as_str(),
                usage,
                Some(total_latency_ms),
                Some(routing_overhead_ms),
                Some(decision.tier.as_str()),
            )?;
        }
        Ok(())
    }

    fn record_error(&self, decision: &StageRouterDecision) -> Result<()> {
        if let Some(stats) = self.stats_handle() {
            stats.record_error(
                decision.selected_model.as_str(),
                Some(decision.tier.as_str()),
            )?;
        }
        Ok(())
    }

    fn routing_metadata(&self, decision: &StageRouterDecision) -> RoutingMetadata {
        RoutingMetadata {
            selected_model: Some(decision.selected_model.to_string()),
            selected_tier: Some(decision.tier.as_str().to_string()),
            confidence: decision.confidence,
            router_version: Some("stage_router:v1".to_string()),
            tolerance: Some(self.confidence_threshold),
            rationale: Some(format!(
                "stage_router source={}; score={}; selected {}",
                decision.source.as_str(),
                decision.score,
                decision.tier.as_str()
            )),
        }
    }
}

#[async_trait]
impl ProfileHooks for StageRouterProfile {
    type ProcessedRequest = StageRouterProcessedRequest;

    /// Extracts signals, picks a tier, and rewrites the request model.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        self.route_request(input).await
    }

    /// Leaves the backend response unchanged after stage_router routing completes.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for StageRouterProfile {
    /// Executes stage_router routing with one context-window fallback retry.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile_started_at = Instant::now();
        let processed = self.process(input).await?;
        let (first_result, first_backend_latency_ms) = self.call_selected(&processed).await;
        match first_result {
            Ok(response) => {
                let total_latency_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;
                self.record_success(
                    &processed.decision,
                    &response,
                    total_latency_ms,
                    first_backend_latency_ms,
                )?;
                let response = self.rprocess(&processed, response).await?;
                return Ok(ProfileResponse::with_routing_metadata(
                    response,
                    self.routing_metadata(&processed.decision),
                ));
            }
            Err(SwitchyardError::ContextWindowExceeded { .. }) => {
                let retry = self.retry_processed_request(&processed)?;
                let (retry_result, retry_backend_latency_ms) = self.call_selected(&retry).await;
                match retry_result {
                    Ok(response) => {
                        let total_latency_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;
                        self.record_success(
                            &retry.decision,
                            &response,
                            total_latency_ms,
                            retry_backend_latency_ms,
                        )?;
                        let response = self.rprocess(&retry, response).await?;
                        return Ok(ProfileResponse::with_routing_metadata(
                            response,
                            self.routing_metadata(&retry.decision),
                        ));
                    }
                    Err(SwitchyardError::ContextWindowExceeded { target_id, .. }) => {
                        self.record_error(&retry.decision)?;
                        return Err(SwitchyardError::ContextPoolExhausted {
                            last_target_id: target_id,
                            reason: "all attempted targets returned context-window overflow"
                                .to_string(),
                        });
                    }
                    Err(error) => {
                        self.record_error(&retry.decision)?;
                        return Err(error);
                    }
                }
            }
            Err(error) => {
                self.record_error(&processed.decision)?;
                return Err(error);
            }
        }
    }

    /// Routes from an exact Relay snapshot without dispatching the selected target.
    async fn decide(&self, context: DecisionContext) -> Result<RoutingDecision> {
        decision_for_stage_router_routing(self, context).await
    }
}

struct ScoreResult {
    score: f64,
    confidence: f64,
}

struct CodingAgentDimensions {
    severity: f64,
    no_error_streak_intensity: f64,
    write_intensity: f64,
    edit_intensity: f64,
    recent_write_intensity: f64,
    planning_active: f64,
    pure_bash_intensity: f64,
    stuck_exploring: f64,
    no_progress: f64,
    tests_passed: f64,
}

impl CodingAgentDimensions {
    fn value(&self, name: &str) -> f64 {
        match name {
            "severity" => self.severity,
            "no_error_streak_intensity" => self.no_error_streak_intensity,
            "write_intensity" => self.write_intensity,
            "edit_intensity" => self.edit_intensity,
            "recent_write_intensity" => self.recent_write_intensity,
            "planning_active" => self.planning_active,
            "pure_bash_intensity" => self.pure_bash_intensity,
            "stuck_exploring" => self.stuck_exploring,
            "no_progress" => self.no_progress,
            "tests_passed" => self.tests_passed,
            _ => 0.0,
        }
    }
}

fn score_signal(signal: &ToolResultSignal) -> ScoreResult {
    let dimensions = dimensions_from_signal(signal);
    let raw = DEFAULT_WEIGHTS
        .iter()
        .map(|(name, weight)| dimensions.value(name) * weight)
        .sum::<f64>();
    let score = raw.clamp(-1.0, 1.0);
    ScoreResult {
        score,
        confidence: score.abs(),
    }
}

fn dimensions_from_signal(signal: &ToolResultSignal) -> CodingAgentDimensions {
    let total_tool_ops = signal.write_count + signal.edit_count + signal.read_count;
    let recent_tool_ops =
        signal.recent_write_count + signal.recent_edit_count + signal.recent_read_count;
    let stuck = signal.turn_depth >= 8 && signal.write_count <= 1 && signal.read_count >= 5;
    let no_progress = signal.turn_depth > 60 && signal.write_count == 0;

    CodingAgentDimensions {
        severity: f64::from(signal.severity),
        no_error_streak_intensity: saturating(f64::from(signal.no_error_streak), 3.0),
        write_intensity: ratio(signal.write_count, total_tool_ops),
        edit_intensity: ratio(signal.edit_count, total_tool_ops),
        recent_write_intensity: ratio(signal.recent_write_count, recent_tool_ops),
        planning_active: if signal.recent_todowrite_count > 0 {
            1.0
        } else {
            0.0
        },
        pure_bash_intensity: saturating(f64::from(signal.pure_bash_streak), PURE_BASH_NORM),
        stuck_exploring: if stuck { 1.0 } else { 0.0 },
        no_progress: if no_progress { 1.0 } else { 0.0 },
        tests_passed: if signal.tests_passed && signal.write_count >= 3 {
            1.0
        } else {
            0.0
        },
    }
}

fn apply_overrides(signal: &ToolResultSignal) -> Option<StageRouterTier> {
    if signal.severity >= SEVERITY_CRITICAL {
        return Some(StageRouterTier::Capable);
    }
    if signal.tests_passed
        && signal.turn_depth >= CLEAN_TESTS_MIN_TURN_DEPTH
        && signal.write_count <= CLEAN_TESTS_MAX_WRITES
    {
        return Some(StageRouterTier::Efficient);
    }
    None
}

fn saturating(value: f64, scale: f64) -> f64 {
    if value <= 0.0 {
        0.0
    } else {
        1.0 - (-value / scale).exp()
    }
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        f64::from(numerator) / f64::from(denominator)
    }
}

struct StageRouterTierClassifier {
    config: StageRouterClassifierConfig,
    client: reqwest::Client,
    disable_reasoning: bool,
}

impl StageRouterTierClassifier {
    fn new(config: &StageRouterClassifierConfig) -> Result<Self> {
        config.validate()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(config.timeout_secs))
            .build()
            .map_err(|error| {
                SwitchyardError::InvalidConfig(format!(
                    "failed to build stage_router classifier HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            config: config.clone(),
            client,
            disable_reasoning: model_accepts_reasoning_hint(config.model.as_str()),
        })
    }

    async fn classify(
        &self,
        request: &ChatRequest,
        signal: &ToolResultSignal,
        stats: Option<&StatsAccumulator>,
    ) -> Option<StageRouterTier> {
        let started_at = Instant::now();
        let response = match self
            .client
            .post(chat_completions_url(self.config.base_url.as_deref()))
            .bearer_auth(&self.config.api_key)
            .json(&self.request_body(request, signal))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(error = %error, "stage_router classifier call failed; falling open");
                record_classifier_error(stats, self.config.model.as_str());
                return None;
            }
        };

        if !response.status().is_success() {
            tracing::warn!(
                status = %response.status(),
                "stage_router classifier returned error status; falling open"
            );
            record_classifier_error(stats, self.config.model.as_str());
            return None;
        }

        let body = match response.json::<Value>().await {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(error = %error, "stage_router classifier returned invalid JSON; falling open");
                record_classifier_error(stats, self.config.model.as_str());
                return None;
            }
        };
        record_classifier_usage(
            stats,
            self.config.model.as_str(),
            usage_from_body(&body),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
        let tier = parse_classifier_tier(&body);
        if tier.is_none() {
            record_classifier_error(stats, self.config.model.as_str());
        }
        tier
    }

    fn request_body(&self, request: &ChatRequest, signal: &ToolResultSignal) -> Value {
        let system_prompt = self
            .config
            .system_prompt
            .as_deref()
            .unwrap_or(CLASSIFIER_SYSTEM_PROMPT);
        let mut body = json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarise_signal(request, signal, self.config.recent_turn_window)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": self.config.max_tokens,
        });
        if self.disable_reasoning {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        body
    }
}

fn record_classifier_usage(
    stats: Option<&StatsAccumulator>,
    model: &str,
    usage: TokenUsage,
    latency_ms: f64,
) {
    if let Some(stats) = stats {
        if let Err(error) = stats.record_classifier_usage(model, usage, Some(latency_ms)) {
            tracing::debug!(error = %error, "failed to record stage_router classifier usage");
        }
    }
}

fn record_classifier_error(stats: Option<&StatsAccumulator>, model: &str) {
    if let Some(stats) = stats {
        if let Err(error) = stats.record_classifier_error(model) {
            tracing::debug!(error = %error, "failed to record stage_router classifier error");
        }
    }
}

fn parse_classifier_tier(body: &Value) -> Option<StageRouterTier> {
    let content = body
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()?;
    let payload = serde_json::from_str::<Value>(content).ok()?;
    match payload.get("tier").and_then(Value::as_str) {
        Some("capable") => Some(StageRouterTier::Capable),
        Some("efficient") => Some(StageRouterTier::Efficient),
        _ => None,
    }
}

fn summarise_signal(
    request: &ChatRequest,
    signal: &ToolResultSignal,
    recent_window: usize,
) -> String {
    let state_line = format!(
        "State: turn_depth={}, severity={:.1}, writes={}, edits={}, reads={}, todowrites={}, recent_writes={}, recent_edits={}, recent_reads={}, pure_bash_streak={}, no_error_streak={}, tests_passed={}",
        signal.turn_depth,
        signal.severity,
        signal.write_count,
        signal.edit_count,
        signal.read_count,
        signal.todowrite_count,
        signal.recent_write_count,
        signal.recent_edit_count,
        signal.recent_read_count,
        signal.pure_bash_streak,
        signal.no_error_streak,
        signal.tests_passed,
    );
    let recent_messages = recent_messages(request, recent_window);
    if recent_messages.is_empty() {
        return format!("Decide STRONG or WEAK for the next call. {state_line}");
    }

    let mut lines = vec![
        "Decide STRONG or WEAK for the next call.".to_string(),
        state_line,
        "Recent turns (most recent last):".to_string(),
    ];
    lines.extend(recent_messages.iter().map(format_message));
    lines.join("\n")
}

fn recent_messages(request: &ChatRequest, recent_window: usize) -> Vec<Value> {
    if recent_window == 0 {
        return Vec::new();
    }
    let Some(messages) = request
        .body()
        .as_object()
        .and_then(|body| body.get("messages"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    messages
        .iter()
        .skip(messages.len().saturating_sub(recent_window))
        .cloned()
        .collect()
}

fn format_message(message: &Value) -> String {
    let Some(object) = message.as_object() else {
        let rendered = message.to_string();
        return format!("[?] {}", truncate(&rendered, 400));
    };
    let role = object.get("role").and_then(Value::as_str).unwrap_or("?");
    let body = match object.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(format_content_block)
            .collect::<Vec<_>>()
            .join(" "),
        Some(other) => other.to_string(),
        None => "(empty)".to_string(),
    };
    format!("[{role}] {}", truncate(&body, 400))
}

fn format_content_block(block: &Value) -> Option<String> {
    let object = block.as_object()?;
    match object.get("type").and_then(Value::as_str) {
        Some("tool_use") => Some(format!(
            "<tool_use:{}>",
            object.get("name").and_then(Value::as_str).unwrap_or("?")
        )),
        Some("tool_result") => Some(format!(
            "<tool_result:{}>",
            truncate(
                &object
                    .get("content")
                    .map(Value::to_string)
                    .unwrap_or_default(),
                120
            )
        )),
        Some("text") => Some(
            object
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
        _ => None,
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn chat_completions_url(base_url: Option<&str>) -> String {
    format!(
        "{}/chat/completions",
        base_url
            .unwrap_or(DEFAULT_OPENAI_BASE_URL)
            .trim_end_matches('/')
    )
}

fn model_accepts_reasoning_hint(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    !["anthropic", "bedrock", "claude"]
        .iter()
        .any(|tag| lowered.contains(tag))
}

fn default_picker() -> StageRouterPickerMode {
    StageRouterPickerMode::CapableFirst
}

fn default_confidence_threshold() -> f64 {
    DEFAULT_CONFIDENCE_THRESHOLD
}

fn default_signal_recent_window() -> usize {
    DEFAULT_RECENT_WINDOW
}

fn default_classifier_timeout_secs() -> f64 {
    DEFAULT_CLASSIFIER_TIMEOUT_SECS
}

fn default_classifier_recent_turn_window() -> usize {
    DEFAULT_CLASSIFIER_RECENT_TURN_WINDOW
}

fn default_classifier_max_tokens() -> u32 {
    CLASSIFIER_MAX_TOKENS
}

fn default_enable_stats() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;
    use switchyard_core::{BackendFormat, ModelId};

    use crate::backend::ProfileBackend;
    use crate::{RelayIdentityKey, RequestMetadata};

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct ObservedCall {
        backend: &'static str,
        body: Value,
    }

    #[derive(Clone, Debug)]
    enum BackendAction {
        Ok,
        ContextOverflow,
    }

    struct TestBackend {
        name: &'static str,
        target_id: String,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
        actions: Mutex<Vec<BackendAction>>,
    }

    #[async_trait]
    impl ProfileBackend for TestBackend {
        async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.calls
                .lock()
                .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))?
                .push(ObservedCall {
                    backend: self.name,
                    body: request.body().clone(),
                });
            let action = self
                .actions
                .lock()
                .map_err(|_| SwitchyardError::Other("actions mutex poisoned".to_string()))?
                .pop()
                .unwrap_or(BackendAction::Ok);
            match action {
                BackendAction::Ok => Ok(ChatResponse::openai_completion(json!({
                    "served_by": self.name,
                    "model": request.model(),
                    "usage": {
                        "prompt_tokens": 13,
                        "completion_tokens": 5,
                    },
                }))),
                BackendAction::ContextOverflow => Err(SwitchyardError::ContextWindowExceeded {
                    target_id: self.target_id.clone(),
                    model: request.model().unwrap_or("").to_string(),
                    message: "prompt is too long".to_string(),
                }),
            }
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    fn target_backend(
        target: &LlmTarget,
        name: &'static str,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
        actions: Vec<BackendAction>,
    ) -> TargetBackend {
        let mut actions = actions;
        actions.reverse();
        TargetBackend::new(
            target.clone(),
            Arc::new(TestBackend {
                name,
                target_id: target.id.to_string(),
                calls,
                actions: Mutex::new(actions),
            }),
        )
    }

    fn observed(calls: &Arc<Mutex<Vec<ObservedCall>>>) -> Result<Vec<ObservedCall>> {
        calls
            .lock()
            .map(|calls| calls.clone())
            .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))
    }

    fn profile_input(request: ChatRequest) -> ProfileInput {
        ProfileInput {
            request,
            metadata: RequestMetadata::default(),
        }
    }

    fn profile(
        capable: LlmTarget,
        efficient: LlmTarget,
        picker: StageRouterPickerMode,
        confidence_threshold: f64,
        efficient_actions: Vec<BackendAction>,
        capable_actions: Vec<BackendAction>,
    ) -> Result<(StageRouterProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let profile = StageRouterProfile {
            capable_backend: target_backend(
                &capable,
                "capable-backend",
                calls.clone(),
                capable_actions,
            ),
            efficient_backend: target_backend(&efficient, "efficient-backend", calls.clone(), efficient_actions),
            fallback_target_on_evict: capable.id.clone(),
            picker,
            confidence_threshold,
            signal_recent_window: DEFAULT_RECENT_WINDOW,
            classifier: None,
            stats: StatsAccumulator::new(),
            enable_stats: true,
        };
        Ok((profile, calls))
    }

    fn relay_snapshot(messages: Vec<Value>, turn_depth: u64) -> RelaySnapshot {
        RelaySnapshot {
            identity: RelayIdentityKey::session_only("session-1"),
            event_count: messages.len() as u64,
            messages,
            turn_depth,
        }
    }

    #[tokio::test]
    async fn stage_router_decision_cold_state_uses_configured_default_without_classifier_or_target_call(
    ) -> Result<()> {
        for (picker, expected_tier, expected_model) in [
            (
                StageRouterPickerMode::CapableFirst,
                StageRouterTier::Capable,
                "frontier/model",
            ),
            (
                StageRouterPickerMode::EfficientFirst,
                StageRouterTier::Efficient,
                "cheap/model",
            ),
        ] {
            let (mut profile, calls) = profile(
                target("capable", "frontier/model")?,
                target("efficient", "cheap/model")?,
                picker,
                1.0,
                vec![BackendAction::Ok],
                vec![BackendAction::Ok],
            )?;
            profile.classifier = Some(StageRouterTierClassifier::new(&StageRouterClassifierConfig {
                model: "unreachable-classifier".to_string(),
                api_key: "test-key".to_string(),
                base_url: Some("http://127.0.0.1:1/v1".to_string()),
                timeout_secs: 0.01,
                recent_turn_window: 1,
                max_tokens: CLASSIFIER_MAX_TOKENS,
                system_prompt: None,
            })?);

            let (processed, freshness) = profile
                .process_decision_snapshot(
                    profile_input(ChatRequest::openai_chat(json!({
                        "model": "smart-stage_router",
                    }))),
                    None,
                )
                .await?;

            assert_eq!(freshness, None);
            assert_eq!(processed.decision.tier, expected_tier);
            assert_eq!(processed.decision.source, StageRouterDecisionSource::FallOpen);
            assert_eq!(processed.decision.confidence, None);
            assert_eq!(processed.decision.score, 0.0);
            assert_eq!(
                processed.profile_input.request.model(),
                Some(expected_model)
            );
            assert!(observed(&calls)?.is_empty());
            assert_eq!(
                profile
                    .stats
                    .snapshot()?
                    .routing_decisions
                    .get("stage_router")
                    .and_then(|sources| sources.get("fall_open")),
                Some(&1)
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_decision_turn_only_snapshot_remains_cold() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.0,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;
        let snapshot = relay_snapshot(Vec::new(), 42);

        let (processed, freshness) = profile
            .process_decision_snapshot(
                profile_input(ChatRequest::openai_chat(json!({
                    "model": "smart-stage_router",
                }))),
                Some(&snapshot),
            )
            .await?;

        assert_eq!(freshness, None);
        assert_eq!(processed.decision.tier, StageRouterTier::Efficient);
        assert_eq!(processed.decision.source, StageRouterDecisionSource::FallOpen);
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_decision_projects_fresh_relay_history_without_target_dispatch() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.7,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;
        let snapshot = relay_snapshot(
            vec![json!({
                "role": "tool",
                "tool_call_id": "call-oom",
                "content": "process failed: out of memory",
            })],
            3,
        );

        let (processed, freshness) = profile
            .process_decision_snapshot(
                profile_input(ChatRequest::openai_chat(json!({
                    "model": "smart-stage_router",
                }))),
                Some(&snapshot),
            )
            .await?;

        assert_eq!(freshness, Some(FeatureFreshness::Fresh));
        assert_eq!(processed.decision.tier, StageRouterTier::Capable);
        assert_eq!(processed.decision.source, StageRouterDecisionSource::Override);
        assert_eq!(processed.decision.confidence, Some(1.0));
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_decision_fresh_clean_signal_uses_dimensions() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::CapableFirst,
            0.0,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;
        let snapshot = relay_snapshot(
            vec![json!({
                "role": "tool",
                "tool_call_id": "call-tests",
                "content": "all tests passed",
            })],
            1,
        );

        let (processed, freshness) = profile
            .process_decision_snapshot(
                profile_input(ChatRequest::openai_chat(json!({
                    "model": "smart-stage_router",
                }))),
                Some(&snapshot),
            )
            .await?;

        assert_eq!(freshness, Some(FeatureFreshness::Fresh));
        assert_eq!(processed.decision.tier, StageRouterTier::Efficient);
        assert_eq!(processed.decision.source, StageRouterDecisionSource::Dimensions);
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[test]
    fn stage_router_decision_snapshot_turn_depth_saturates_to_u32() -> Result<()> {
        let (profile, _calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.7,
            Vec::new(),
            Vec::new(),
        )?;
        let snapshot = relay_snapshot(
            vec![json!({"role": "user", "content": "continue"})],
            u64::MAX,
        );

        let signal = profile
            .signal_from_relay_snapshot(Some(&snapshot))
            .ok_or_else(|| SwitchyardError::Other("expected fresh signal".to_string()))?;

        assert_eq!(signal.turn_depth, u32::MAX);
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_normal_run_still_extracts_request_signals_and_dispatches() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.7,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "smart-stage_router",
                "messages": [{"role": "user", "content": "continue"}],
            }))))
            .await?;

        assert_eq!(
            response
                .routing_metadata
                .as_ref()
                .and_then(|metadata| metadata.selected_tier.as_deref()),
            Some("efficient")
        );
        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].backend, "efficient-backend");
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_routes_critical_tool_errors_to_capable() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.7,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "smart-stage_router",
                "messages": [
                    {"role": "assistant", "tool_calls": [{
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{\"command\":\"python test.py\"}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "Out of memory"}
                ],
            }))))
            .await?;

        let routing_metadata = response
            .routing_metadata
            .as_ref()
            .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
        assert_eq!(
            routing_metadata.selected_model.as_deref(),
            Some("frontier/model")
        );
        assert_eq!(routing_metadata.selected_tier.as_deref(), Some("capable"));
        assert_eq!(routing_metadata.confidence, Some(1.0));
        assert_eq!(
            routing_metadata.router_version.as_deref(),
            Some("stage_router:v1")
        );
        let response = response.response;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].backend, "capable-backend");
        assert_eq!(calls[0].body["model"], "frontier/model");
        match response {
            ChatResponse::OpenAiCompletion(body) => {
                assert_eq!(body.body()["served_by"], "capable-backend");
            }
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }

        let snapshot = profile.stats.snapshot()?;
        assert_eq!(
            snapshot
                .routing_decisions
                .get("stage_router")
                .and_then(|sources| sources.get("override")),
            Some(&1)
        );
        assert_eq!(
            snapshot
                .models
                .get("frontier/model")
                .and_then(|model| model.tier.as_deref()),
            Some("capable")
        );
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_threshold_zero_accepts_neutral_scorer_as_efficient() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::CapableFirst,
            0.0,
            vec![BackendAction::Ok],
            vec![BackendAction::Ok],
        )?;

        let processed = profile
            .process(profile_input(ChatRequest::openai_chat(json!({
                "model": "smart-stage_router",
                "messages": [{"role": "user", "content": "continue"}],
            }))))
            .await?;

        assert_eq!(processed.decision.tier, StageRouterTier::Efficient);
        assert_eq!(processed.decision.source, StageRouterDecisionSource::Dimensions);
        assert_eq!(processed.profile_input.request.model(), Some("cheap/model"));
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn stage_router_retries_configured_fallback_after_context_overflow() -> Result<()> {
        let (profile, calls) = profile(
            target("capable", "frontier/model")?,
            target("efficient", "cheap/model")?,
            StageRouterPickerMode::EfficientFirst,
            0.7,
            vec![BackendAction::ContextOverflow],
            vec![BackendAction::Ok],
        )?;

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "smart-stage_router",
                "messages": [{"role": "user", "content": "continue"}],
            }))))
            .await?;

        let routing_metadata = response
            .routing_metadata
            .as_ref()
            .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
        assert_eq!(
            routing_metadata.selected_model.as_deref(),
            Some("frontier/model")
        );
        assert_eq!(routing_metadata.selected_tier.as_deref(), Some("capable"));
        assert!(routing_metadata
            .rationale
            .as_deref()
            .is_some_and(|reason| reason.contains("source=context_overflow_fallback")));
        let response = response.response;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].backend, "efficient-backend");
        assert_eq!(calls[0].body["model"], "cheap/model");
        assert_eq!(calls[1].backend, "capable-backend");
        assert_eq!(calls[1].body["model"], "frontier/model");
        match response {
            ChatResponse::OpenAiCompletion(body) => {
                assert_eq!(body.body()["served_by"], "capable-backend");
            }
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }

        let snapshot = profile.stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(
            snapshot
                .models
                .get("frontier/model")
                .and_then(|model| model.tier.as_deref()),
            Some("capable")
        );
        assert_eq!(
            snapshot
                .routing_decisions
                .get("stage_router")
                .and_then(|sources| sources.get("fall_open")),
            Some(&1)
        );
        Ok(())
    }

    #[test]
    fn stage_router_config_rejects_unknown_fallback_target() -> Result<()> {
        let config = StageRouterProfileConfig {
            capable: target("capable", "frontier/model")?,
            efficient: target("efficient", "cheap/model")?,
            fallback_target_on_evict: LlmTargetId::new("ghost")?,
            picker: StageRouterPickerMode::CapableFirst,
            confidence_threshold: 0.7,
            signal_recent_window: DEFAULT_RECENT_WINDOW,
            classifier: None,
            enable_stats: true,
        };

        let error = config
            .validate()
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "expected validation error".to_string());
        assert!(error.contains("fallback_target_on_evict"));
        Ok(())
    }

    #[test]
    fn classifier_request_uses_reasoning_hint_for_vllm_models() -> Result<()> {
        let classifier = StageRouterTierClassifier::new(&StageRouterClassifierConfig {
            model: "nvidia/deepseek-ai/deepseek-v4-flash".to_string(),
            api_key: "test-key".to_string(),
            base_url: None,
            timeout_secs: 1.0,
            recent_turn_window: 2,
            max_tokens: CLASSIFIER_MAX_TOKENS,
            system_prompt: None,
        })?;

        let body = classifier.request_body(
            &ChatRequest::openai_chat(json!({
                "model": "smart-stage_router",
                "messages": [{"role": "user", "content": "hi"}],
            })),
            &ToolResultSignal::default(),
        );

        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(body["max_tokens"], CLASSIFIER_MAX_TOKENS);
        Ok(())
    }
}

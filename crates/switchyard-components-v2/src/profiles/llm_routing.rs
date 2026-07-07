// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM classifier routing as a profile-owned runtime.

use std::collections::BTreeSet;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use switchyard_components::stats::usage_from_body;
use switchyard_components::StatsAccumulator;
use switchyard_core::{
    BackendFormat, ChatRequest, ChatResponse, LlmTarget, LlmTargetId, Result, SwitchyardError,
};

use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::stats::{record_usage_or_tap_stream, UsageAttribution};
use crate::{
    profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse,
    RoutingMetadata,
};

const TIER_STRONG: &str = "strong";
const TIER_WEAK: &str = "weak";
const PROFILE_GENERAL: &str = "general";
const PROFILE_CODING_AGENT: &str = "coding_agent";
const PROFILE_OPENCLAW: &str = "openclaw";
const DEFAULT_CLASSIFIER_MAX_TOKENS: u64 = 4096;
const DEFAULT_ALIGNMENT_MIN_CONFIDENCE: f64 = 0.85;
const DEFAULT_CLASSIFIER_TOOL_NAME: &str = "select_route";

/// Config for the flatter LLM classifier profile.
#[profile_config("llm-routing")]
pub struct LlmRoutingProfileConfig {
    /// Strong target served by this profile.
    #[profile_target]
    pub strong: LlmTarget,
    /// Weak target served by this profile.
    #[profile_target]
    pub weak: LlmTarget,
    /// OpenAI-compatible classifier target used for tool-call route selection.
    #[profile_target]
    pub classifier: LlmTarget,
    /// Target used for one retry after context-window overflow.
    #[serde(default = "default_fallback_target_on_evict")]
    pub fallback_target_on_evict: LlmTargetId,
    /// Classifier policy profile: `general`, `coding_agent`, or `openclaw`.
    #[serde(default = "default_profile_name")]
    pub profile_name: String,
    /// Confidence floor below which routing falls back to the profile default.
    #[serde(default)]
    pub classifier_min_confidence: f64,
    /// Whether classifier failures fall open to the default tier.
    #[serde(default = "default_classifier_fail_open")]
    pub classifier_fail_open: bool,
    /// Number of recent turns included in the classifier request summary.
    #[serde(default = "default_recent_turn_window")]
    pub classifier_recent_turn_window: usize,
    /// Maximum tokens allowed for the classifier tool-call response.
    #[serde(default = "default_classifier_max_tokens")]
    pub classifier_max_tokens: u64,
    /// Confidence required before the classifier recommendation can bump the policy tier.
    #[serde(default = "default_alignment_min_confidence")]
    pub alignment_min_confidence: f64,
    /// Default backend tier used for abstain, low-confidence, and fail-open decisions.
    #[serde(default)]
    pub default_tier: Option<String>,
    /// Optional mapping from classifier route tiers to backend tiers.
    #[serde(default)]
    pub tier_mapping: Option<LlmRoutingTierMapping>,
    /// Optional system prompt override for the classifier request.
    #[serde(default)]
    pub classifier_system_prompt: Option<String>,
    /// Optional function/tool name override for classifier route selection.
    #[serde(default)]
    pub classifier_tool_name: Option<String>,
    /// Optional function/tool description override for classifier route selection.
    #[serde(default)]
    pub classifier_tool_description: Option<String>,
    /// Optional JSON schema override for classifier tool arguments.
    #[serde(default)]
    pub classifier_tool_parameters: Option<Value>,
}

/// Mapping from classifier route tiers to configured backend tiers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmRoutingTierMapping {
    /// Backend tier for `recommended_tier: simple`.
    pub simple: String,
    /// Backend tier for `recommended_tier: medium`.
    pub medium: String,
    /// Backend tier for `recommended_tier: complex`.
    pub complex: String,
    /// Backend tier for `recommended_tier: reasoning`.
    pub reasoning: String,
}

impl ProfileConfig for LlmRoutingProfileConfig {
    type Runtime = LlmRoutingProfile;

    /// Builds the runtime profile using profile-local targets and backends.
    fn build(&self) -> Result<Self::Runtime> {
        validate_config(self)?;
        let policy = ClassifierPolicy::from_name(&self.profile_name)?;
        let strong_target = llm_routing_target(self.strong.clone());
        let weak_target = llm_routing_target(self.weak.clone());
        let classifier_target = llm_routing_classifier_target(self.classifier.clone());
        Ok(LlmRoutingProfile {
            policy,
            strong_backend: native_target_backend(strong_target)?,
            weak_backend: native_target_backend(weak_target)?,
            classifier_backend: native_target_backend(classifier_target.clone())?,
            classifier_target,
            fallback_target_on_evict: self.fallback_target_on_evict.clone(),
            classifier_min_confidence: self.classifier_min_confidence,
            classifier_fail_open: self.classifier_fail_open,
            classifier_recent_turn_window: self.classifier_recent_turn_window,
            classifier_max_tokens: self.classifier_max_tokens,
            alignment_min_confidence: self.alignment_min_confidence,
            default_tier: resolve_default_tier(self.default_tier.as_deref())?,
            tier_mapping: self
                .tier_mapping
                .clone()
                .unwrap_or_else(|| policy.default_tier_mapping()),
            classifier_tool: LlmRoutingClassifierTool::from_config(policy, self),
            stats: profile_stats_accumulator(),
        })
    }
}

/// LLM classifier profile runtime.
pub struct LlmRoutingProfile {
    policy: ClassifierPolicy,
    strong_backend: TargetBackend,
    weak_backend: TargetBackend,
    classifier_backend: TargetBackend,
    classifier_target: LlmTarget,
    fallback_target_on_evict: LlmTargetId,
    classifier_min_confidence: f64,
    classifier_fail_open: bool,
    classifier_recent_turn_window: usize,
    classifier_max_tokens: u64,
    alignment_min_confidence: f64,
    default_tier: String,
    tier_mapping: LlmRoutingTierMapping,
    classifier_tool: LlmRoutingClassifierTool,
    stats: StatsAccumulator,
}

/// Processed LLM-routing request with the profile-owned route decision.
pub struct LlmRoutingProcessedRequest {
    /// Routed input prepared for the selected backend.
    pub profile_input: ProfileInput,
    /// Selected LLM-routing decision for this request.
    pub decision: LlmRoutingDecision,
}

/// Serializable decision record used by Rust callers and tests.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LlmRoutingDecision {
    /// Concrete tier label selected for backend dispatch.
    pub tier: String,
    /// Decision source (`policy_tier`, `low_confidence`, `abstain`, etc.).
    pub source: String,
    /// Human-readable reason for the tier selection.
    pub reason: String,
    /// Policy tier computed from classifier features.
    pub policy_tier: Option<String>,
    /// Classifier-emitted recommended tier.
    pub llm_recommended_tier: Option<String>,
    /// Classifier confidence, when a classifier output was available.
    pub confidence: Option<f64>,
    /// Selected v2 target id.
    pub selected_target: String,
    /// Selected upstream model.
    pub selected_model: String,
    /// Raw classifier signals as JSON for audit/debugging.
    pub signals: Option<Value>,
}

impl LlmRoutingProfile {
    async fn classify(&self, request: &ChatRequest) -> Result<ClassifierSignals> {
        let classifier_request = ChatRequest::openai_chat(json!({
            "model": self.classifier_target.model.as_str(),
            "messages": [
                {"role": "system", "content": self.classifier_tool.system_prompt.as_str()},
                {"role": "user", "content": summarize_request(
                    request,
                    self.classifier_recent_turn_window,
                )},
            ],
            "temperature": 0,
            "max_tokens": self.classifier_max_tokens,
            "tools": [self.classifier_tool.definition()],
            "tool_choice": self.classifier_tool.choice(),
        }));

        let started_at = Instant::now();
        let response = self.classifier_backend.call(&classifier_request).await?;
        let latency_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let raw = classifier_tool_arguments(&response, &self.classifier_tool.name)?;
        let signals = self.policy.parse_signals(&raw)?;
        validate_signals(&signals)?;
        let usage = response.body().map(usage_from_body).unwrap_or_default();
        self.stats.record_classifier_usage(
            self.classifier_target.model.as_str(),
            usage,
            Some(latency_ms),
        )?;
        Ok(signals)
    }

    async fn route_request(&self, mut input: ProfileInput) -> Result<LlmRoutingProcessedRequest> {
        normalize_reasoning_effort(&mut input.request);
        let decision = match self.classify(&input.request).await {
            Ok(signals) => self.select(&signals)?,
            Err(error) => {
                self.stats
                    .record_classifier_error(self.classifier_target.model.as_str())?;
                if self.classifier_fail_open {
                    let signals = ClassifierSignals::abstain(self.policy);
                    self.default_decision(
                        "classifier_error_fall_open",
                        "classifier failed and classifier_fail_open routed to the default tier",
                        &signals,
                    )?
                } else {
                    return Err(classifier_error(error));
                }
            }
        };
        input.request.set_model(&decision.selected_model);
        Ok(LlmRoutingProcessedRequest {
            profile_input: input,
            decision,
        })
    }

    fn select(&self, signals: &ClassifierSignals) -> Result<LlmRoutingDecision> {
        let selected = if signals.abstain {
            self.default_decision("abstain", "classifier abstained", signals)?
        } else if signals.confidence < self.classifier_min_confidence {
            self.default_decision(
                "low_confidence",
                &format!(
                    "classifier confidence {:.3} < min_confidence {:.3}",
                    signals.confidence, self.classifier_min_confidence
                ),
                signals,
            )?
        } else {
            let mut policy_tier = self.policy.policy_tier(signals);
            let mut source = "policy_tier";
            let mut reason = format!("policy_tier() returned {:?}", policy_tier.as_str());
            if self.policy.align_with_llm_recommendation()
                && signals.confidence >= self.alignment_min_confidence
                && matches!(policy_tier, RouteTier::Simple | RouteTier::Medium)
                && matches!(
                    signals.recommended_tier,
                    RouteTier::Complex | RouteTier::Reasoning
                )
            {
                policy_tier = signals.recommended_tier;
                source = "llm_alignment_bump";
                reason = format!(
                    "LLM alignment bump to {:?} at confidence {:.3}",
                    policy_tier.as_str(),
                    signals.confidence
                );
            }
            let mut tier = self.tier_for(policy_tier);
            if self.policy.escalate_on_tool_planning()
                && tier != self.default_tier
                && self.policy.requires_tool_planning(signals)
            {
                tier = self.default_tier.clone();
                source = "tool_planning_escalation";
                reason = format!(
                    "tool planning escalation from {:?} to {:?}",
                    self.tier_for(policy_tier),
                    tier
                );
            }
            self.decision(LlmRoutingDecisionInput {
                tier,
                source,
                reason,
                policy_tier: Some(policy_tier),
                llm_recommended_tier: Some(signals.recommended_tier),
                confidence: Some(signals.confidence),
                signals: Some(signals.raw.clone()),
            })?
        };
        Ok(selected)
    }

    fn default_decision(
        &self,
        source: &'static str,
        reason: &str,
        signals: &ClassifierSignals,
    ) -> Result<LlmRoutingDecision> {
        self.decision(LlmRoutingDecisionInput {
            tier: self.default_tier.clone(),
            source,
            reason: reason.to_string(),
            policy_tier: Some(self.policy.policy_tier(signals)),
            llm_recommended_tier: Some(signals.recommended_tier),
            confidence: Some(signals.confidence),
            signals: Some(signals.raw.clone()),
        })
    }

    fn decision(&self, input: LlmRoutingDecisionInput) -> Result<LlmRoutingDecision> {
        let backend = self.backend_for_tier(&input.tier)?;
        Ok(LlmRoutingDecision {
            tier: input.tier,
            source: input.source.to_string(),
            reason: input.reason,
            policy_tier: input.policy_tier.map(|tier| tier.as_str().to_string()),
            llm_recommended_tier: input
                .llm_recommended_tier
                .map(|tier| tier.as_str().to_string()),
            confidence: input.confidence,
            selected_target: backend.target().id.to_string(),
            selected_model: backend.target().model.to_string(),
            signals: input.signals,
        })
    }

    fn backend_for_tier(&self, tier: &str) -> Result<&TargetBackend> {
        match tier {
            TIER_STRONG => Ok(&self.strong_backend),
            TIER_WEAK => Ok(&self.weak_backend),
            other => Err(SwitchyardError::InvalidConfig(format!(
                "llm-routing selected unknown tier {other:?}"
            ))),
        }
    }

    fn tier_for(&self, tier: RouteTier) -> String {
        self.tier_mapping.tier_for(tier).to_string()
    }

    fn backend_for_target(&self, target_id: &LlmTargetId) -> Result<&TargetBackend> {
        if *target_id == self.strong_backend.target().id {
            Ok(&self.strong_backend)
        } else if *target_id == self.weak_backend.target().id {
            Ok(&self.weak_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "llm-routing selected target {target_id} that is not configured for this profile"
            )))
        }
    }

    fn tier_for_target(&self, target_id: &LlmTargetId) -> Result<&'static str> {
        if *target_id == self.strong_backend.target().id {
            Ok(TIER_STRONG)
        } else if *target_id == self.weak_backend.target().id {
            Ok(TIER_WEAK)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "llm-routing target {target_id} is not configured for this profile"
            )))
        }
    }

    fn fallback_processed_request(
        &self,
        processed: &LlmRoutingProcessedRequest,
    ) -> Result<LlmRoutingProcessedRequest> {
        let backend = self.backend_for_target(&self.fallback_target_on_evict)?;
        let target = backend.target();
        let mut profile_input = processed.profile_input.clone();
        profile_input.request.set_model(target.model.as_str());
        let mut decision = processed.decision.clone();
        decision.tier = self.tier_for_target(&target.id)?.to_string();
        decision.source = "context_overflow_fallback".to_string();
        decision.reason = format!(
            "selected target {} exceeded its context window; retried fallback target {}",
            processed.decision.selected_target, target.id
        );
        decision.selected_target = target.id.to_string();
        decision.selected_model = target.model.to_string();
        Ok(LlmRoutingProcessedRequest {
            profile_input,
            decision,
        })
    }

    async fn call_selected(
        &self,
        processed: &LlmRoutingProcessedRequest,
    ) -> (Result<ChatResponse>, f64) {
        let started_at = Instant::now();
        let backend = match self.backend_for_target_id_str(&processed.decision.selected_target) {
            Ok(backend) => backend,
            Err(error) => return (Err(error), 0.0),
        };
        let result = backend.call(&processed.profile_input.request).await;
        let latency_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        (result, latency_ms)
    }

    fn backend_for_target_id_str(&self, target_id: &str) -> Result<&TargetBackend> {
        if target_id == self.strong_backend.target().id.as_str() {
            Ok(&self.strong_backend)
        } else if target_id == self.weak_backend.target().id.as_str() {
            Ok(&self.weak_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "llm-routing selected target {target_id:?} that is not configured for this profile"
            )))
        }
    }

    fn record_success(
        &self,
        decision: &LlmRoutingDecision,
        response: ChatResponse,
        started_at: Instant,
        backend_latency_ms: f64,
    ) -> Result<ChatResponse> {
        self.stats.record_success(
            decision.selected_model.as_str(),
            Some(backend_latency_ms),
            Some(decision.tier.as_str()),
        )?;
        Ok(record_usage_or_tap_stream(
            response,
            UsageAttribution::new(
                self.stats.clone(),
                decision.selected_model.as_str(),
                Some(decision.tier.clone()),
                started_at,
                backend_latency_ms,
            ),
        ))
    }

    fn record_error(&self, decision: &LlmRoutingDecision) -> Result<()> {
        self.stats.record_error(
            decision.selected_model.as_str(),
            Some(decision.tier.as_str()),
        )
    }

    fn routing_metadata(&self, decision: &LlmRoutingDecision) -> RoutingMetadata {
        RoutingMetadata {
            selected_model: Some(decision.selected_model.clone()),
            selected_tier: Some(decision.tier.clone()),
            confidence: decision.confidence,
            router_version: Some(format!("llm-routing:{}:v1", self.policy.as_str())),
            tolerance: Some(self.classifier_min_confidence),
            rationale: Some(decision.reason.clone()),
        }
    }
}

#[async_trait]
impl ProfileHooks for LlmRoutingProfile {
    type ProcessedRequest = LlmRoutingProcessedRequest;

    /// Runs classifier routing and returns a prepared backend request.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        self.route_request(input).await
    }

    /// Leaves the backend response unchanged after LLM routing completes.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for LlmRoutingProfile {
    /// Executes LLM classifier routing with one context-window fallback retry.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile_started_at = Instant::now();
        let processed = self.process(input).await?;
        let (first_result, first_backend_latency_ms) = self.call_selected(&processed).await;
        match first_result {
            Ok(response) => {
                let response = self.record_success(
                    &processed.decision,
                    response,
                    profile_started_at,
                    first_backend_latency_ms,
                )?;
                let response = self.rprocess(&processed, response).await?;
                return Ok(ProfileResponse::with_routing_metadata(
                    response,
                    self.routing_metadata(&processed.decision),
                ));
            }
            Err(SwitchyardError::ContextWindowExceeded { .. }) => {
                let retry = self.fallback_processed_request(&processed)?;
                let (retry_result, retry_backend_latency_ms) = self.call_selected(&retry).await;
                match retry_result {
                    Ok(response) => {
                        let response = self.record_success(
                            &retry.decision,
                            response,
                            profile_started_at,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteTier {
    Simple,
    Medium,
    Complex,
    Reasoning,
}

struct LlmRoutingDecisionInput {
    tier: String,
    source: &'static str,
    reason: String,
    policy_tier: Option<RouteTier>,
    llm_recommended_tier: Option<RouteTier>,
    confidence: Option<f64>,
    signals: Option<Value>,
}

impl RouteTier {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "simple" => Some(Self::Simple),
            "medium" => Some(Self::Medium),
            "complex" => Some(Self::Complex),
            "reasoning" => Some(Self::Reasoning),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Medium => "medium",
            Self::Complex => "complex",
            Self::Reasoning => "reasoning",
        }
    }
}

#[derive(Clone, Debug)]
struct LlmRoutingClassifierTool {
    name: String,
    description: String,
    parameters: Value,
    system_prompt: String,
}

impl LlmRoutingClassifierTool {
    fn from_config(policy: ClassifierPolicy, config: &LlmRoutingProfileConfig) -> Self {
        Self {
            name: config
                .classifier_tool_name
                .clone()
                .unwrap_or_else(default_classifier_tool_name),
            description: config
                .classifier_tool_description
                .clone()
                .unwrap_or_else(|| policy.default_tool_description().to_string()),
            parameters: config
                .classifier_tool_parameters
                .clone()
                .unwrap_or_else(|| policy.schema()),
            system_prompt: config
                .classifier_system_prompt
                .clone()
                .unwrap_or_else(|| policy.default_system_prompt().to_string()),
        }
    }

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name.as_str(),
                "description": self.description.as_str(),
                "parameters": self.parameters.clone(),
                "strict": true,
            },
        })
    }

    fn choice(&self) -> Value {
        json!({
            "type": "function",
            "function": {"name": self.name.as_str()},
        })
    }
}

impl LlmRoutingTierMapping {
    fn tier_for(&self, tier: RouteTier) -> &str {
        match tier {
            RouteTier::Simple => &self.simple,
            RouteTier::Medium => &self.medium,
            RouteTier::Complex => &self.complex,
            RouteTier::Reasoning => &self.reasoning,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClassifierPolicy {
    General,
    CodingAgent,
    OpenClaw,
}

impl ClassifierPolicy {
    fn from_name(name: &str) -> Result<Self> {
        match name {
            PROFILE_GENERAL => Ok(Self::General),
            PROFILE_CODING_AGENT => Ok(Self::CodingAgent),
            PROFILE_OPENCLAW => Ok(Self::OpenClaw),
            other => Err(SwitchyardError::InvalidConfig(format!(
                "unknown LLM classifier profile {other:?}; expected general, coding_agent, or openclaw"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::General => PROFILE_GENERAL,
            Self::CodingAgent => PROFILE_CODING_AGENT,
            Self::OpenClaw => PROFILE_OPENCLAW,
        }
    }

    fn default_recommended_tier(self) -> RouteTier {
        RouteTier::Medium
    }

    fn default_tier_mapping(self) -> LlmRoutingTierMapping {
        match self {
            Self::General => LlmRoutingTierMapping {
                simple: TIER_WEAK.to_string(),
                medium: TIER_STRONG.to_string(),
                complex: TIER_STRONG.to_string(),
                reasoning: TIER_STRONG.to_string(),
            },
            Self::CodingAgent | Self::OpenClaw => LlmRoutingTierMapping {
                simple: TIER_WEAK.to_string(),
                medium: TIER_WEAK.to_string(),
                complex: TIER_STRONG.to_string(),
                reasoning: TIER_STRONG.to_string(),
            },
        }
    }

    fn escalate_on_tool_planning(self) -> bool {
        matches!(self, Self::CodingAgent | Self::OpenClaw)
    }

    fn align_with_llm_recommendation(self) -> bool {
        matches!(self, Self::CodingAgent | Self::OpenClaw)
    }

    fn default_system_prompt(self) -> &'static str {
        match self {
            Self::General => include_str!("llm_routing/prompts/general.md"),
            Self::CodingAgent => include_str!("llm_routing/prompts/coding_agent.md"),
            Self::OpenClaw => include_str!("llm_routing/prompts/openclaw.md"),
        }
    }

    fn default_tool_description(self) -> &'static str {
        match self {
            Self::General => "Select the backend tier for a general LLM request.",
            Self::CodingAgent => "Select the backend tier for a coding-agent request.",
            Self::OpenClaw => "Select the backend tier for an OpenClaw assistant request.",
        }
    }

    fn schema(self) -> Value {
        match self {
            Self::General => general_schema(),
            Self::CodingAgent => coding_agent_schema(),
            Self::OpenClaw => openclaw_schema(),
        }
    }

    fn parse_signals(self, raw: &str) -> Result<ClassifierSignals> {
        let value: Value = serde_json::from_str(raw).map_err(|error| {
            SwitchyardError::Backend(format!("classifier returned invalid JSON: {error}"))
        })?;
        ClassifierSignals::from_value(self, value)
    }

    fn policy_tier(self, signals: &ClassifierSignals) -> RouteTier {
        match self {
            Self::General => general_policy_tier(signals),
            Self::CodingAgent => coding_agent_policy_tier(signals),
            Self::OpenClaw => openclaw_policy_tier(signals),
        }
    }

    fn requires_tool_planning(self, signals: &ClassifierSignals) -> bool {
        match self {
            Self::General => bool_field(&signals.raw, "tool_planning_required"),
            Self::CodingAgent => {
                let turn_type = str_field(&signals.raw, "turn_type");
                let scope = str_field(&signals.raw, "code_modification_scope");
                let tool_count = u64_field(&signals.raw, "tool_call_count_estimate");
                let scope_is_modifying = matches!(
                    scope,
                    Some("function" | "file" | "multi_file" | "cross_module")
                );
                turn_type == Some("planning") || (tool_count >= 3 && scope_is_modifying)
            }
            Self::OpenClaw => {
                u64_field(&signals.raw, "tool_call_count_estimate") >= 2
                    || str_field(&signals.raw, "turn_type") == Some("tool_orchestration")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ClassifierSignals {
    raw: Value,
    recommended_tier: RouteTier,
    confidence: f64,
    abstain: bool,
}

impl ClassifierSignals {
    fn from_value(policy: ClassifierPolicy, raw: Value) -> Result<Self> {
        let recommended_tier = str_field(&raw, "recommended_tier")
            .and_then(RouteTier::parse)
            .ok_or_else(|| {
                SwitchyardError::Backend(
                    "classifier JSON missing valid recommended_tier".to_string(),
                )
            })?;
        let confidence = raw
            .get("confidence")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                SwitchyardError::Backend("classifier JSON missing numeric confidence".to_string())
            })?;
        let abstain = raw.get("abstain").and_then(Value::as_bool).unwrap_or(false);
        let signals = Self {
            raw,
            recommended_tier,
            confidence,
            abstain,
        };
        validate_policy_fields(policy, &signals)?;
        Ok(signals)
    }

    fn abstain(policy: ClassifierPolicy) -> Self {
        let recommended_tier = policy.default_recommended_tier();
        let base = json!({
            "recommended_tier": recommended_tier.as_str(),
            "confidence": 0.0,
            "abstain": true,
        });
        let raw = match policy {
            ClassifierPolicy::General => merge_json_objects(
                base,
                json!({
                    "task_type": "other",
                    "complexity": "medium",
                    "reasoning_depth": "light",
                    "tool_planning_required": false,
                    "precision_requirement": "medium",
                    "context_dependency": "conversation",
                    "structured_output_risk": "medium",
                    "reason_code": "ambiguous",
                }),
            ),
            ClassifierPolicy::CodingAgent => merge_json_objects(
                base,
                json!({
                    "turn_type": "exploration",
                    "code_modification_scope": "none",
                    "tool_call_count_estimate": 0,
                    "requires_codebase_context": false,
                }),
            ),
            ClassifierPolicy::OpenClaw => merge_json_objects(
                base,
                json!({
                    "turn_type": "chitchat",
                    "tool_call_count_estimate": 0,
                    "memory_dependency": "none",
                    "external_action_required": false,
                    "precision_requirement": "low",
                    "ambiguity": "low",
                    "channel_kind": "casual",
                }),
            ),
        };
        Self {
            raw,
            recommended_tier,
            confidence: 0.0,
            abstain: true,
        }
    }
}

fn merge_json_objects(mut left: Value, right: Value) -> Value {
    let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) else {
        return left;
    };
    for (key, value) in right {
        left.insert(key.clone(), value.clone());
    }
    Value::Object(left.clone())
}

fn validate_signals(signals: &ClassifierSignals) -> Result<()> {
    if signals.confidence.is_finite() && (0.0..=1.0).contains(&signals.confidence) {
        return Ok(());
    }
    Err(SwitchyardError::Backend(format!(
        "classifier confidence must be finite and in [0.0, 1.0], got {:?}",
        signals.confidence
    )))
}

fn validate_policy_fields(policy: ClassifierPolicy, signals: &ClassifierSignals) -> Result<()> {
    let required = match policy {
        ClassifierPolicy::General => &[
            "task_type",
            "complexity",
            "reasoning_depth",
            "tool_planning_required",
            "precision_requirement",
            "context_dependency",
            "structured_output_risk",
            "reason_code",
        ][..],
        ClassifierPolicy::CodingAgent => &[
            "turn_type",
            "code_modification_scope",
            "tool_call_count_estimate",
            "requires_codebase_context",
        ][..],
        ClassifierPolicy::OpenClaw => &[
            "turn_type",
            "tool_call_count_estimate",
            "memory_dependency",
            "external_action_required",
            "precision_requirement",
            "ambiguity",
            "channel_kind",
        ][..],
    };
    for field in required {
        if signals.raw.get(*field).is_none() {
            return Err(SwitchyardError::Backend(format!(
                "classifier JSON missing required field {field:?}"
            )));
        }
    }
    Ok(())
}

fn general_policy_tier(signals: &ClassifierSignals) -> RouteTier {
    let mut scores = [0_i32; 4];
    match str_field(&signals.raw, "complexity") {
        Some("simple") => scores[0] += 2,
        Some("medium") => scores[1] += 2,
        Some("complex") => scores[2] += 2,
        Some("reasoning") => scores[3] += 2,
        _ => {}
    }
    match str_field(&signals.raw, "reasoning_depth") {
        Some("deep") => scores[3] += 2,
        Some("multi_step") => scores[2] += 1,
        _ => {}
    }
    if bool_field(&signals.raw, "tool_planning_required") {
        scores[2] += 1;
    }
    if str_field(&signals.raw, "precision_requirement") == Some("high") {
        scores[2] += 1;
    }
    if str_field(&signals.raw, "structured_output_risk") == Some("high") {
        scores[2] += 1;
    }
    argmax_tier(scores)
}

fn coding_agent_policy_tier(signals: &ClassifierSignals) -> RouteTier {
    let mut scores = [0_i32; 4];
    match str_field(&signals.raw, "turn_type") {
        Some("chitchat" | "clarification" | "summarize") => scores[0] += 2,
        Some("exploration" | "explanation" | "edit") => scores[1] += 2,
        Some("planning" | "debug") => scores[2] += 2,
        _ => {}
    }
    match str_field(&signals.raw, "code_modification_scope") {
        Some("none" | "single_line") => scores[0] += 1,
        Some("function" | "file") => scores[1] += 1,
        Some("multi_file" | "cross_module") => scores[2] += 2,
        _ => {}
    }
    let tool_count = u64_field(&signals.raw, "tool_call_count_estimate");
    if tool_count >= 4 {
        scores[2] += 1;
    } else if tool_count >= 2 {
        scores[1] += 1;
    }
    if bool_field(&signals.raw, "requires_codebase_context") {
        scores[1] += 1;
    }
    argmax_tier(scores)
}

fn openclaw_policy_tier(signals: &ClassifierSignals) -> RouteTier {
    let mut scores = [0_i32; 4];
    match str_field(&signals.raw, "turn_type") {
        Some("chitchat" | "lookup" | "memory_recall" | "clarification") => scores[0] += 2,
        Some("planning" | "explanation") => scores[1] += 2,
        Some("tool_orchestration" | "action") => scores[2] += 2,
        _ => {}
    }
    let tool_count = u64_field(&signals.raw, "tool_call_count_estimate");
    if tool_count >= 4 {
        scores[2] += 1;
    } else if tool_count >= 1 {
        scores[1] += 1;
    }
    match str_field(&signals.raw, "memory_dependency") {
        Some("heavy") => scores[2] += 2,
        Some("light") => scores[1] += 1,
        _ => {}
    }
    if bool_field(&signals.raw, "external_action_required")
        && str_field(&signals.raw, "precision_requirement") == Some("high")
    {
        scores[2] += 2;
    } else if bool_field(&signals.raw, "external_action_required") {
        scores[1] += 1;
    }
    match str_field(&signals.raw, "ambiguity") {
        Some("high") => scores[2] += 1,
        Some("medium") => scores[1] += 1,
        _ => {}
    }
    if str_field(&signals.raw, "channel_kind") == Some("casual") {
        scores[0] += 1;
    } else {
        scores[2] += 1;
    }
    argmax_tier(scores)
}

fn argmax_tier(scores: [i32; 4]) -> RouteTier {
    let best = scores.into_iter().max().unwrap_or(0);
    for (index, score) in scores.into_iter().enumerate().rev() {
        if score == best {
            return match index {
                0 => RouteTier::Simple,
                1 => RouteTier::Medium,
                2 => RouteTier::Complex,
                _ => RouteTier::Reasoning,
            };
        }
    }
    RouteTier::Medium
}

fn classifier_tool_arguments(response: &ChatResponse, tool_name: &str) -> Result<String> {
    let body = response.body().ok_or_else(|| {
        SwitchyardError::Backend("classifier returned a streaming response".to_string())
    })?;
    let tool_calls = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            SwitchyardError::Backend("classifier response did not include tool_calls".to_string())
        })?;
    let Some(tool_call) = tool_calls.iter().find(|tool_call| {
        tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            == Some(tool_name)
    }) else {
        return Err(SwitchyardError::Backend(format!(
            "classifier response did not call required tool {tool_name:?}"
        )));
    };
    let arguments = tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .ok_or_else(|| {
            SwitchyardError::Backend("classifier tool call omitted arguments".to_string())
        })?;
    match arguments {
        Value::String(raw) if !raw.trim().is_empty() => Ok(raw.trim().to_string()),
        Value::Object(_) => serde_json::to_string(arguments).map_err(|error| {
            SwitchyardError::Backend(format!(
                "classifier tool arguments could not be serialized: {error}"
            ))
        }),
        _ => Err(SwitchyardError::Backend(
            "classifier tool arguments must be a non-empty JSON string or object".to_string(),
        )),
    }
}

fn summarize_request(request: &ChatRequest, recent_turn_window: usize) -> String {
    let body = request.body();
    let summary_body = body
        .as_object()
        .map(|object| condense_body(object, recent_turn_window))
        .unwrap_or_else(|| body.clone());
    let payload = json!({
        "request_type": request.request_type(),
        "body": summary_body,
    });
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    const MAX_CHARS: usize = 16_000;
    if text.len() <= MAX_CHARS {
        text
    } else {
        format!(
            "{}...<truncated>",
            truncate_at_char_boundary(&text, MAX_CHARS - 32)
        )
    }
}

fn truncate_at_char_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn condense_body(body: &Map<String, Value>, recent_turn_window: usize) -> Value {
    let mut out = Map::new();
    for (key, value) in body {
        if !matches!(key.as_str(), "messages" | "tools" | "input" | "tool_choice") {
            out.insert(key.clone(), value.clone());
        }
    }
    if let Some(Value::Array(tools)) = body.get("tools") {
        out.insert(
            "tools".to_string(),
            Value::Array(tools.iter().map(condense_tool).collect()),
        );
    }
    if let Some(Value::Array(messages)) = body.get("messages") {
        out.insert(
            "messages".to_string(),
            Value::Array(trim_messages(messages, recent_turn_window)),
        );
    }
    if let Some(input) = body.get("input") {
        match input {
            Value::Array(items) => {
                out.insert(
                    "input".to_string(),
                    Value::Array(trim_messages(items, recent_turn_window)),
                );
            }
            Value::String(_) => {
                out.insert("input".to_string(), input.clone());
            }
            _ => {}
        }
    }
    Value::Object(out)
}

fn condense_tool(tool: &Value) -> Value {
    let Some(object) = tool.as_object() else {
        return tool.clone();
    };
    let mut out = object.clone();
    if let Some(Value::Object(function)) = object.get("function") {
        let mut slim = function.clone();
        slim.remove("parameters");
        out.insert("function".to_string(), Value::Object(slim));
    }
    out.remove("input_schema");
    Value::Object(out)
}

fn trim_messages(messages: &[Value], recent_turn_window: usize) -> Vec<Value> {
    let mut system = Vec::new();
    let mut first_user = None;
    let mut first_user_idx = None;
    for (idx, message) in messages.iter().enumerate() {
        let role = message
            .as_object()
            .and_then(|object| object.get("role"))
            .and_then(Value::as_str);
        match role {
            Some("system" | "developer") => system.push(message.clone()),
            Some("user") if first_user.is_none() => {
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
                && message
                    .as_object()
                    .and_then(|object| object.get("role"))
                    .and_then(Value::as_str)
                    .is_some_and(|role| !matches!(role, "system" | "developer"))
        })
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    if recent_turn_window == 0 {
        let mut out = system;
        out.push(first_user);
        if let Some(last_user) = tail.iter().rev().find(|message| {
            message
                .as_object()
                .and_then(|object| object.get("role"))
                .and_then(Value::as_str)
                == Some("user")
        }) {
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

fn base_properties(extra: Vec<(&'static str, Value)>) -> Map<String, Value> {
    let mut properties = Map::from_iter([
        (
            "recommended_tier".to_string(),
            enum_schema(&["simple", "medium", "complex", "reasoning"]),
        ),
        ("confidence".to_string(), json!({"type": "number"})),
        ("abstain".to_string(), json!({"type": "boolean"})),
    ]);
    for (name, value) in extra {
        properties.insert(name.to_string(), value);
    }
    properties
}

fn object_schema(properties: Map<String, Value>) -> Value {
    let required = properties
        .keys()
        .cloned()
        .map(Value::String)
        .collect::<Vec<_>>();
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn enum_schema(values: &[&str]) -> Value {
    json!({
        "type": "string",
        "enum": values,
    })
}

fn general_schema() -> Value {
    object_schema(base_properties(vec![
        (
            "task_type",
            enum_schema(&[
                "chat",
                "summarization",
                "extraction",
                "translation",
                "coding",
                "debugging",
                "math",
                "planning",
                "creative_writing",
                "agentic_task",
                "research",
                "data_analysis",
                "other",
            ]),
        ),
        (
            "complexity",
            enum_schema(&["simple", "medium", "complex", "reasoning"]),
        ),
        (
            "reasoning_depth",
            enum_schema(&["none", "light", "multi_step", "deep"]),
        ),
        ("tool_planning_required", json!({"type": "boolean"})),
        (
            "precision_requirement",
            enum_schema(&["low", "medium", "high"]),
        ),
        (
            "context_dependency",
            enum_schema(&["latest_message", "conversation", "external_context"]),
        ),
        (
            "structured_output_risk",
            enum_schema(&["low", "medium", "high"]),
        ),
        (
            "reason_code",
            enum_schema(&[
                "simple_qa",
                "summarization",
                "extraction",
                "translation",
                "coding_simple",
                "coding_complex",
                "debugging",
                "math_reasoning",
                "tool_agentic",
                "long_context",
                "structured_output",
                "creative_generation",
                "research_synthesis",
                "ambiguous",
                "other",
            ]),
        ),
    ]))
}

fn coding_agent_schema() -> Value {
    object_schema(base_properties(vec![
        (
            "turn_type",
            enum_schema(&[
                "chitchat",
                "planning",
                "exploration",
                "edit",
                "debug",
                "explanation",
                "clarification",
                "summarize",
            ]),
        ),
        (
            "code_modification_scope",
            enum_schema(&[
                "none",
                "single_line",
                "function",
                "file",
                "multi_file",
                "cross_module",
            ]),
        ),
        ("tool_call_count_estimate", json!({"type": "integer"})),
        ("requires_codebase_context", json!({"type": "boolean"})),
    ]))
}

fn openclaw_schema() -> Value {
    object_schema(base_properties(vec![
        (
            "turn_type",
            enum_schema(&[
                "chitchat",
                "lookup",
                "memory_recall",
                "planning",
                "tool_orchestration",
                "action",
                "explanation",
                "clarification",
            ]),
        ),
        ("tool_call_count_estimate", json!({"type": "integer"})),
        (
            "memory_dependency",
            enum_schema(&["none", "light", "heavy"]),
        ),
        ("external_action_required", json!({"type": "boolean"})),
        (
            "precision_requirement",
            enum_schema(&["low", "medium", "high"]),
        ),
        ("ambiguity", enum_schema(&["low", "medium", "high"])),
        ("channel_kind", enum_schema(&["casual", "deliberate"])),
    ]))
}

fn str_field<'a>(value: &'a Value, name: &str) -> Option<&'a str> {
    value.get(name).and_then(Value::as_str)
}

fn bool_field(value: &Value, name: &str) -> bool {
    value.get(name).and_then(Value::as_bool).unwrap_or(false)
}

fn u64_field(value: &Value, name: &str) -> u64 {
    value.get(name).and_then(Value::as_u64).unwrap_or(0)
}

fn classifier_error(error: SwitchyardError) -> SwitchyardError {
    SwitchyardError::Backend(format!(
        "LLM classifier failed to produce valid route signals: {error}"
    ))
}

fn validate_config(config: &LlmRoutingProfileConfig) -> Result<()> {
    ClassifierPolicy::from_name(&config.profile_name)?;
    if !(config.classifier_min_confidence.is_finite()
        && (0.0..=1.0).contains(&config.classifier_min_confidence))
    {
        return Err(SwitchyardError::InvalidConfig(format!(
            "classifier_min_confidence must be finite and in [0.0, 1.0], got {:?}",
            config.classifier_min_confidence
        )));
    }
    if !(config.alignment_min_confidence.is_finite()
        && (0.0..=1.0).contains(&config.alignment_min_confidence))
    {
        return Err(SwitchyardError::InvalidConfig(format!(
            "alignment_min_confidence must be finite and in [0.0, 1.0], got {:?}",
            config.alignment_min_confidence
        )));
    }
    if config.classifier_max_tokens == 0 {
        return Err(SwitchyardError::InvalidConfig(
            "classifier_max_tokens must be greater than 0".to_string(),
        ));
    }
    if let Some(default_tier) = &config.default_tier {
        validate_backend_tier("default_tier", default_tier)?;
    }
    if let Some(tier_mapping) = &config.tier_mapping {
        validate_backend_tier("tier_mapping.simple", &tier_mapping.simple)?;
        validate_backend_tier("tier_mapping.medium", &tier_mapping.medium)?;
        validate_backend_tier("tier_mapping.complex", &tier_mapping.complex)?;
        validate_backend_tier("tier_mapping.reasoning", &tier_mapping.reasoning)?;
    }
    if let Some(tool_name) = &config.classifier_tool_name {
        validate_classifier_tool_name(tool_name)?;
    }
    if let Some(tool_parameters) = &config.classifier_tool_parameters {
        validate_classifier_tool_parameters(tool_parameters)?;
    }
    if config.fallback_target_on_evict != config.strong.id
        && config.fallback_target_on_evict != config.weak.id
    {
        return Err(SwitchyardError::InvalidConfig(format!(
            "fallback_target_on_evict={} must match one of [{}, {}]",
            config.fallback_target_on_evict, config.weak.id, config.strong.id
        )));
    }
    if config.classifier.format != BackendFormat::OpenAi {
        return Err(SwitchyardError::InvalidConfig(
            "llm-routing classifier target must use format: openai".to_string(),
        ));
    }
    Ok(())
}

fn validate_backend_tier(field: &str, value: &str) -> Result<()> {
    match value {
        TIER_STRONG | TIER_WEAK => Ok(()),
        other => Err(SwitchyardError::InvalidConfig(format!(
            "{field} must be {TIER_STRONG:?} or {TIER_WEAK:?}, got {other:?}"
        ))),
    }
}

fn validate_classifier_tool_name(value: &str) -> Result<()> {
    let valid = !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'));
    if valid {
        Ok(())
    } else {
        Err(SwitchyardError::InvalidConfig(format!(
            "classifier_tool_name must be non-empty and contain only ASCII letters, numbers, '_' or '-', got {value:?}"
        )))
    }
}

fn validate_classifier_tool_parameters(value: &Value) -> Result<()> {
    validate_strict_object_schema("classifier_tool_parameters", value)
}

fn validate_strict_object_schema(path: &str, value: &Value) -> Result<()> {
    if value.get("type").and_then(Value::as_str) != Some("object") {
        return Err(SwitchyardError::InvalidConfig(format!(
            "{path} must be a strict JSON schema object with type: object"
        )));
    }
    if value.get("additionalProperties").and_then(Value::as_bool) != Some(false) {
        return Err(SwitchyardError::InvalidConfig(format!(
            "{path} must set additionalProperties: false when strict tool calls are enabled"
        )));
    }
    let properties = value
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "{path} must define object properties for strict tool calls"
            ))
        })?;
    let required = value
        .get("required")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "{path} must list all properties in required for strict tool calls"
            ))
        })?;
    let required = required
        .iter()
        .map(|item| {
            item.as_str().ok_or_else(|| {
                SwitchyardError::InvalidConfig(format!(
                    "{path}.required must contain only property names"
                ))
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let property_names = properties
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if required != property_names {
        return Err(SwitchyardError::InvalidConfig(format!(
            "{path}.required must contain exactly every key from properties for strict tool calls"
        )));
    }
    for (name, schema) in properties {
        validate_nested_strict_schema(&format!("{path}.properties.{name}"), schema)?;
    }
    Ok(())
}

fn validate_nested_strict_schema(path: &str, value: &Value) -> Result<()> {
    if value.get("type").and_then(Value::as_str) == Some("object")
        || value.get("properties").is_some()
    {
        validate_strict_object_schema(path, value)?;
    }
    if let Some(items) = value.get("items") {
        validate_nested_strict_schema(&format!("{path}.items"), items)?;
    }
    for combinator in ["anyOf", "oneOf", "allOf"] {
        if let Some(schemas) = value.get(combinator).and_then(Value::as_array) {
            for (index, schema) in schemas.iter().enumerate() {
                validate_nested_strict_schema(&format!("{path}.{combinator}[{index}]"), schema)?;
            }
        }
    }
    Ok(())
}

fn resolve_default_tier(value: Option<&str>) -> Result<String> {
    let value = value.unwrap_or(TIER_STRONG);
    validate_backend_tier("default_tier", value)?;
    Ok(value.to_string())
}

fn default_profile_name() -> String {
    PROFILE_CODING_AGENT.to_string()
}

fn default_classifier_fail_open() -> bool {
    true
}

fn default_recent_turn_window() -> usize {
    4
}

fn default_classifier_max_tokens() -> u64 {
    DEFAULT_CLASSIFIER_MAX_TOKENS
}

fn default_alignment_min_confidence() -> f64 {
    DEFAULT_ALIGNMENT_MIN_CONFIDENCE
}

fn default_classifier_tool_name() -> String {
    DEFAULT_CLASSIFIER_TOOL_NAME.to_string()
}

fn default_fallback_target_on_evict() -> LlmTargetId {
    LlmTargetId::from_static(TIER_STRONG)
}

fn llm_routing_target(mut target: LlmTarget) -> LlmTarget {
    apply_deepseek_defaults(&mut target);
    target
}

fn llm_routing_classifier_target(mut target: LlmTarget) -> LlmTarget {
    if target.extra_body.is_none() && model_accepts_reasoning_hint(target.model.as_str()) {
        target.extra_body = Some(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    }
    apply_deepseek_defaults(&mut target);
    target
}

fn apply_deepseek_defaults(target: &mut LlmTarget) {
    let model = target.model.as_str().to_ascii_lowercase();
    if target.extra_body.is_none() && model.contains("deepseek-v4") {
        target.extra_body = Some(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    }
    if target.extra_headers.is_empty() && model.contains("deepseek") {
        target
            .extra_headers
            .insert("X-Inference-Priority".to_string(), "batch".to_string());
    }
}

fn model_accepts_reasoning_hint(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    !["anthropic", "bedrock", "claude"]
        .iter()
        .any(|tag| lowered.contains(tag))
}

fn normalize_reasoning_effort(request: &mut ChatRequest) {
    const VALID: &[&str] = &["low", "medium", "high", "max"];
    let Value::Object(body) = request.body_mut() else {
        return;
    };
    let Some(effort) = body.get("reasoning_effort").and_then(Value::as_str) else {
        return;
    };
    if VALID.contains(&effort) {
        return;
    }
    body.insert(
        "reasoning_effort".to_string(),
        Value::String("high".to_string()),
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures_util::StreamExt;
    use serde_json::json;
    use switchyard_core::{ModelId, StreamEvent};

    use crate::backend::ProfileBackend;

    use super::*;

    // Stub backend used only to satisfy the profile struct; `record_success`
    // never dispatches through it.
    struct UnusedBackend;

    #[async_trait]
    impl ProfileBackend for UnusedBackend {
        async fn call(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Err(SwitchyardError::Other(
                "backend should not be called".to_string(),
            ))
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    fn stub_backend(target: &LlmTarget) -> TargetBackend {
        TargetBackend::new(target.clone(), Arc::new(UnusedBackend))
    }

    fn config(
        strong: &LlmTarget,
        weak: &LlmTarget,
        classifier: &LlmTarget,
    ) -> LlmRoutingProfileConfig {
        LlmRoutingProfileConfig {
            strong: strong.clone(),
            weak: weak.clone(),
            classifier: classifier.clone(),
            fallback_target_on_evict: strong.id.clone(),
            profile_name: PROFILE_GENERAL.to_string(),
            classifier_min_confidence: 0.0,
            classifier_fail_open: true,
            classifier_recent_turn_window: 4,
            classifier_max_tokens: 256,
            alignment_min_confidence: DEFAULT_ALIGNMENT_MIN_CONFIDENCE,
            default_tier: None,
            tier_mapping: None,
            classifier_system_prompt: None,
            classifier_tool_name: None,
            classifier_tool_description: None,
            classifier_tool_parameters: None,
        }
    }

    fn profile(stats: StatsAccumulator) -> Result<LlmRoutingProfile> {
        let strong = target("strong", "frontier/model")?;
        let weak = target("weak", "cheap/model")?;
        let classifier = target("classifier", "cls/model")?;
        let config = config(&strong, &weak, &classifier);
        let policy = ClassifierPolicy::from_name(&config.profile_name)?;
        Ok(LlmRoutingProfile {
            policy,
            strong_backend: stub_backend(&strong),
            weak_backend: stub_backend(&weak),
            classifier_backend: stub_backend(&classifier),
            classifier_target: classifier,
            fallback_target_on_evict: config.fallback_target_on_evict.clone(),
            classifier_min_confidence: config.classifier_min_confidence,
            classifier_fail_open: config.classifier_fail_open,
            classifier_recent_turn_window: config.classifier_recent_turn_window,
            classifier_max_tokens: config.classifier_max_tokens,
            alignment_min_confidence: config.alignment_min_confidence,
            default_tier: resolve_default_tier(config.default_tier.as_deref())?,
            tier_mapping: config
                .tier_mapping
                .clone()
                .unwrap_or_else(|| policy.default_tier_mapping()),
            classifier_tool: LlmRoutingClassifierTool::from_config(policy, &config),
            stats,
        })
    }

    fn weak_decision() -> LlmRoutingDecision {
        LlmRoutingDecision {
            tier: TIER_WEAK.to_string(),
            source: "unit_test".to_string(),
            reason: "unit_test".to_string(),
            policy_tier: None,
            llm_recommended_tier: None,
            confidence: None,
            selected_target: "weak".to_string(),
            selected_model: "cheap/model".to_string(),
            signals: None,
        }
    }

    #[tokio::test]
    async fn record_success_taps_streaming_usage_on_consumption() -> Result<()> {
        let stats = StatsAccumulator::new();
        let profile = profile(stats.clone())?;

        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::Json(json!({
                "choices": [{"delta": {"content": "hi"}}],
            }))),
            Ok(StreamEvent::Json(json!({
                "choices": [],
                "usage": {"prompt_tokens": 23, "completion_tokens": 12},
            }))),
        ];
        let response = ChatResponse::OpenAiStream(Box::pin(futures_util::stream::iter(events)));

        let response = profile.record_success(&weak_decision(), response, Instant::now(), 4.0)?;

        // Success is attributed synchronously; streaming usage waits for consume.
        let before = stats.snapshot()?;
        assert_eq!(before.total_requests, 1);
        assert_eq!(before.total_tokens.prompt, 0);

        let mut stream = match response {
            ChatResponse::OpenAiStream(stream) => stream,
            _ => return Err(SwitchyardError::Other("expected stream response".into())),
        };
        let mut forwarded = 0;
        while let Some(event) = stream.next().await {
            event?;
            forwarded += 1;
        }
        assert_eq!(forwarded, 2, "every event must still reach the client");

        let after = stats.snapshot()?;
        assert_eq!(after.total_tokens.prompt, 23);
        assert_eq!(after.total_tokens.completion, 12);
        let tier = after
            .tiers
            .get(TIER_WEAK)
            .ok_or_else(|| SwitchyardError::Other("weak tier stats should be present".into()))?;
        assert_eq!(tier.calls, 1);
        assert_eq!(tier.model, "cheap/model");
        Ok(())
    }

    #[tokio::test]
    async fn record_success_records_buffered_usage_immediately() -> Result<()> {
        let stats = StatsAccumulator::new();
        let profile = profile(stats.clone())?;

        let response = ChatResponse::openai_completion(json!({
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }));
        let _response = profile.record_success(&weak_decision(), response, Instant::now(), 4.0)?;

        let snapshot = stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_tokens.prompt, 7);
        assert_eq!(snapshot.total_tokens.completion, 2);
        Ok(())
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile configuration and runtime for learned prefill-probe routing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_components::stats::usage_from_body;
use switchyard_components::StatsAccumulator;
use switchyard_core::{ChatResponse, LlmTarget, Result, SwitchyardError};

use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::{
    profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse,
    RoutingMetadata,
};

use super::artifact::InferenceArtifact;
use super::policy::CostAwareRoutingPolicy;
use super::scorer::{HiddenStateProbeScorer, ProbeScorer};

const LEARNED_SCORE_THRESHOLD: f64 = 0.5;
const TIER_STRONG: &str = "strong";
const TIER_WEAK: &str = "weak";

/// Learned routing policy applied to the mapped weak and strong checkpoint heads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum PrefillProbeRoutingPolicyConfig {
    /// Balances predicted correctness against normalized completion cost.
    CostAware {
        /// Correctness weight in `[0, 1]`; this is the profile's routing knob.
        lambda: f64,
        /// Non-negative weak-target cost in the same units as `strong_cost`.
        weak_cost: f64,
        /// Non-negative strong-target cost in the same units as `weak_cost`.
        strong_cost: f64,
    },
}

/// Config for routing with learned prompt hidden-state features.
///
/// The probe target produces hidden states but is never selected for completion
/// inference. The scorer returns `1.0` for weak and `0.0` for strong, and this
/// profile applies a fixed decision threshold of `0.5`.
#[profile_config("prefill-probe")]
pub struct PrefillProbeProfileConfig {
    /// Probe target used only to produce prompt hidden states.
    #[profile_target]
    pub probe: LlmTarget,
    /// Strong completion target selected by score `0.0` or probe failure.
    #[profile_target]
    pub strong: LlmTarget,
    /// Artifact output head corresponding to the strong completion target.
    pub strong_checkpoint_head: String,
    /// Weak completion target selected by score `1.0`.
    #[profile_target]
    pub weak: LlmTarget,
    /// Artifact output head corresponding to the weak completion target.
    pub weak_checkpoint_head: String,
    /// Directory shared with vLLM's `ExampleHiddenStatesConnector`.
    pub hidden_states_dir: String,
    /// Directory containing `router.json` and `router.safetensors`.
    pub inference_artifact_dir: String,
    /// Policy that maps the two selected correctness probabilities to a binary score.
    pub routing_policy: PrefillProbeRoutingPolicyConfig,
}

impl ProfileConfig for PrefillProbeProfileConfig {
    type Runtime = PrefillProbeProfile;

    /// Validates configuration and builds the complete learned routing runtime.
    fn build(&self) -> Result<Self::Runtime> {
        let policy = match self.routing_policy {
            PrefillProbeRoutingPolicyConfig::CostAware {
                lambda,
                weak_cost,
                strong_cost,
            } => CostAwareRoutingPolicy::new(lambda, weak_cost, strong_cost)?,
        };
        let artifact =
            InferenceArtifact::load(&self.inference_artifact_dir, self.probe.model.as_str())?;
        let weak_head_index = checkpoint_head_index(
            &artifact,
            "weak_checkpoint_head",
            &self.weak_checkpoint_head,
        )?;
        let strong_head_index = checkpoint_head_index(
            &artifact,
            "strong_checkpoint_head",
            &self.strong_checkpoint_head,
        )?;
        if weak_head_index == strong_head_index {
            return Err(SwitchyardError::InvalidConfig(format!(
                "weak_checkpoint_head and strong_checkpoint_head must map to distinct outputs; both map to `{}`",
                self.weak_checkpoint_head,
            )));
        }

        let base_url = self
            .probe
            .endpoint
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8000/v1".to_string());
        let artifact = Arc::new(artifact);
        let scorer = HiddenStateProbeScorer::new(
            base_url,
            self.probe.model.as_str(),
            self.hidden_states_dir.as_str(),
            artifact,
            weak_head_index,
            strong_head_index,
            policy,
        );

        Ok(PrefillProbeProfile {
            strong_backend: native_target_backend(self.strong.clone())?,
            weak_backend: native_target_backend(self.weak.clone())?,
            score_threshold: LEARNED_SCORE_THRESHOLD,
            scorer: Arc::new(scorer),
            stats: profile_stats_accumulator(),
            decision_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

fn checkpoint_head_index(
    artifact: &InferenceArtifact,
    field: &str,
    checkpoint_head: &str,
) -> Result<usize> {
    artifact
        .output_names()
        .iter()
        .position(|name| name == checkpoint_head)
        .ok_or_else(|| {
            SwitchyardError::InvalidConfig(format!(
                "{field} `{checkpoint_head}` is not present in artifact output_names {:?}",
                artifact.output_names(),
            ))
        })
}

/// One strong/weak decision emitted by the prefill-probe router.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PrefillProbeDecision {
    /// ID of the selected completion target.
    pub selected_target: String,
    /// Model string sent to the selected completion backend.
    pub selected_model: String,
    /// Selected tier, either `strong` or `weak`.
    pub tier: &'static str,
    /// Binary weak-preference score: `0.0` for strong and `1.0` for weak.
    pub score: f64,
}

/// Request prepared for a completion backend with its routing decision.
pub struct PrefillProbeProcessedRequest {
    /// Routed input with its model rewritten to the selected completion model.
    pub profile_input: ProfileInput,
    /// Decision used to select the completion backend.
    pub decision: PrefillProbeDecision,
}

/// Runtime for learned prefill-probe strong/weak routing.
pub struct PrefillProbeProfile {
    strong_backend: TargetBackend,
    weak_backend: TargetBackend,
    score_threshold: f64,
    scorer: Arc<dyn ProbeScorer>,
    stats: StatsAccumulator,
    // Successful scores are reused by the first string-valued user instruction.
    decision_cache: Arc<Mutex<HashMap<u64, f64>>>,
}

impl PrefillProbeProfile {
    // Hashes the first user instruction represented as a string.
    fn instruction_key(request: &switchyard_core::ChatRequest) -> Option<u64> {
        let messages = request.body().as_object()?.get("messages")?.as_array()?;
        let content = messages.iter().find_map(|message| {
            (message.get("role").and_then(serde_json::Value::as_str) == Some("user"))
                .then(|| message.get("content").and_then(serde_json::Value::as_str))
                .flatten()
        })?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        content.hash(&mut hasher);
        Some(hasher.finish())
    }

    // Scores an uncached instruction and explicitly bypasses threshold routing on failure.
    async fn route(&self, mut input: ProfileInput) -> Result<PrefillProbeProcessedRequest> {
        let key = Self::instruction_key(&input.request);
        let cached = key.and_then(|key| self.decision_cache.lock().get(&key).copied());
        let (score, strong_fallback) = if let Some(score) = cached {
            (score, false)
        } else {
            match self.scorer.score(&input.request).await {
                Ok(score) => {
                    if let Some(key) = key {
                        self.decision_cache.lock().insert(key, score);
                    }
                    (score, false)
                }
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        fallback_tier = TIER_STRONG,
                        score = 0.0,
                        "prefill probe failed; using uncached strong fallback"
                    );
                    (0.0, true)
                }
            }
        };

        let (selected_backend, tier) = if strong_fallback {
            (&self.strong_backend, TIER_STRONG)
        } else if score >= self.score_threshold {
            (&self.weak_backend, TIER_WEAK)
        } else {
            (&self.strong_backend, TIER_STRONG)
        };
        input
            .request
            .set_model(selected_backend.target().model.as_str());

        Ok(PrefillProbeProcessedRequest {
            profile_input: input,
            decision: PrefillProbeDecision {
                selected_target: selected_backend.target().id.to_string(),
                selected_model: selected_backend.target().model.to_string(),
                tier,
                score,
            },
        })
    }

    fn backend_for(&self, decision: &PrefillProbeDecision) -> Result<&TargetBackend> {
        if decision.selected_target == self.strong_backend.target().id.as_str() {
            Ok(&self.strong_backend)
        } else if decision.selected_target == self.weak_backend.target().id.as_str() {
            Ok(&self.weak_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "prefill probe selected target {} that is not configured for this profile",
                decision.selected_target,
            )))
        }
    }

    fn record_success(
        &self,
        decision: &PrefillProbeDecision,
        response: &ChatResponse,
        total_latency_ms: f64,
        backend_latency_ms: f64,
    ) -> Result<()> {
        self.stats.record_success(
            decision.selected_model.as_str(),
            Some(backend_latency_ms),
            Some(decision.tier),
        )?;
        let routing_overhead_ms = (total_latency_ms - backend_latency_ms).max(0.0);
        let usage = response.body().map(usage_from_body).unwrap_or_default();
        self.stats.record_usage_after_success_attribution(
            decision.selected_model.as_str(),
            usage,
            Some(total_latency_ms),
            Some(routing_overhead_ms),
            Some(decision.tier),
        )?;
        Ok(())
    }

    fn record_error(&self, decision: &PrefillProbeDecision) -> Result<()> {
        self.stats
            .record_error(decision.selected_model.as_str(), Some(decision.tier))
    }

    fn routing_metadata(&self, decision: &PrefillProbeDecision) -> RoutingMetadata {
        RoutingMetadata {
            selected_model: Some(decision.selected_model.clone()),
            selected_tier: Some(decision.tier.to_string()),
            confidence: None,
            router_version: Some("prefill-probe:v1".to_string()),
            tolerance: Some(LEARNED_SCORE_THRESHOLD),
            rationale: Some(format!(
                "binary weak-preference score {} selected {}",
                decision.score, decision.tier,
            )),
        }
    }
}

#[async_trait]
impl ProfileHooks for PrefillProbeProfile {
    type ProcessedRequest = PrefillProbeProcessedRequest;

    /// Scores the prompt and returns a request prepared for the selected backend.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        self.route(input).await
    }

    /// Leaves the selected backend response unchanged.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for PrefillProbeProfile {
    /// Routes one request, calls the selected completion backend, and records the outcome.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile_started_at = Instant::now();
        let processed = self.process(input).await?;
        let decision = &processed.decision;
        let backend = self.backend_for(decision)?;
        let backend_started_at = Instant::now();
        let response = match backend.call(&processed.profile_input.request).await {
            Ok(response) => response,
            Err(error) => {
                self.record_error(decision)?;
                return Err(error);
            }
        };
        let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
        let total_latency_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;
        self.record_success(decision, &response, total_latency_ms, backend_latency_ms)?;
        let response = self.rprocess(&processed, response).await?;
        Ok(ProfileResponse::with_routing_metadata(
            response,
            self.routing_metadata(decision),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::{json, Value};
    use switchyard_core::{BackendFormat, ChatRequest, LlmTargetId, ModelId, SwitchyardError};

    use crate::backend::ProfileBackend;
    use crate::RequestMetadata;

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct ObservedCall {
        backend: &'static str,
        body: Value,
    }

    struct TestBackend {
        name: &'static str,
        fail: bool,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
    }

    #[async_trait]
    impl ProfileBackend for TestBackend {
        async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.calls.lock().push(ObservedCall {
                backend: self.name,
                body: request.body().clone(),
            });
            if self.fail {
                return Err(SwitchyardError::Backend(format!("{} failed", self.name)));
            }
            Ok(ChatResponse::openai_completion(json!({
                "served_by": self.name,
                "model": request.model(),
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            })))
        }
    }

    struct FixedScorer(f64);

    #[async_trait]
    impl ProbeScorer for FixedScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            Ok(self.0)
        }
    }

    struct ErrorScorer;

    #[async_trait]
    impl ProbeScorer for ErrorScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            Err(SwitchyardError::Other("probe unavailable".to_string()))
        }
    }

    struct CountingScorer {
        score: f64,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ProbeScorer for CountingScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.score)
        }
    }

    struct FlakyScorer {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ProbeScorer for FlakyScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                Err(SwitchyardError::Other(
                    "transient probe failure".to_string(),
                ))
            } else {
                Ok(1.0)
            }
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    fn profile(
        scorer: Arc<dyn ProbeScorer>,
        strong_fails: bool,
        weak_fails: bool,
    ) -> Result<(PrefillProbeProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let strong = target("strong", "frontier/model")?;
        let weak = target("weak", "cheap/model")?;
        Ok((
            PrefillProbeProfile {
                strong_backend: TargetBackend::new(
                    strong,
                    Arc::new(TestBackend {
                        name: "strong-backend",
                        fail: strong_fails,
                        calls: calls.clone(),
                    }),
                ),
                weak_backend: TargetBackend::new(
                    weak,
                    Arc::new(TestBackend {
                        name: "weak-backend",
                        fail: weak_fails,
                        calls: calls.clone(),
                    }),
                ),
                score_threshold: LEARNED_SCORE_THRESHOLD,
                scorer,
                stats: StatsAccumulator::new(),
                decision_cache: Arc::new(Mutex::new(HashMap::new())),
            },
            calls,
        ))
    }

    fn input(instruction: &str) -> ProfileInput {
        ProfileInput {
            request: ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [{"role": "user", "content": instruction}],
            })),
            metadata: RequestMetadata::default(),
        }
    }

    fn observed(calls: &Arc<Mutex<Vec<ObservedCall>>>) -> Vec<ObservedCall> {
        calls.lock().clone()
    }

    #[tokio::test]
    async fn binary_score_direction_and_model_rewrite_are_fixed() -> Result<()> {
        let (weak_profile, weak_calls) = profile(Arc::new(FixedScorer(1.0)), false, false)?;
        let weak = weak_profile.process(input("weak task")).await?;
        assert_eq!(weak_profile.score_threshold, 0.5);
        assert_eq!(weak.decision.tier, TIER_WEAK);
        assert_eq!(weak.decision.score, 1.0);
        assert_eq!(weak.profile_input.request.model(), Some("cheap/model"));
        assert!(observed(&weak_calls).is_empty());

        let (strong_profile, strong_calls) = profile(Arc::new(FixedScorer(0.0)), false, false)?;
        let strong = strong_profile.process(input("strong task")).await?;
        assert_eq!(strong.decision.tier, TIER_STRONG);
        assert_eq!(strong.decision.score, 0.0);
        assert_eq!(strong.profile_input.request.model(), Some("frontier/model"));
        assert!(observed(&strong_calls).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn successful_decision_is_cached_by_first_string_user_instruction() -> Result<()> {
        let scorer_calls = Arc::new(AtomicUsize::new(0));
        let (profile, _calls) = profile(
            Arc::new(CountingScorer {
                score: 1.0,
                calls: scorer_calls.clone(),
            }),
            false,
            false,
        )?;

        let first = profile.process(input("same instruction")).await?;
        let second = profile.process(input("same instruction")).await?;

        assert_eq!(first.decision.tier, TIER_WEAK);
        assert_eq!(second.decision.tier, TIER_WEAK);
        assert_eq!(scorer_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn probe_failure_forces_uncached_strong_then_retries_and_caches_success() -> Result<()> {
        let scorer_calls = Arc::new(AtomicUsize::new(0));
        let (profile, calls) = profile(
            Arc::new(FlakyScorer {
                calls: scorer_calls.clone(),
            }),
            false,
            false,
        )?;

        let fallback = profile.process(input("retry task")).await?;
        let retry = profile.process(input("retry task")).await?;
        let cached = profile.process(input("retry task")).await?;

        assert_eq!(fallback.decision.tier, TIER_STRONG);
        assert_eq!(fallback.decision.score, 0.0);
        assert_eq!(fallback.decision.selected_model, "frontier/model");
        assert_eq!(retry.decision.tier, TIER_WEAK);
        assert_eq!(cached.decision.tier, TIER_WEAK);
        assert_eq!(scorer_calls.load(Ordering::SeqCst), 2);
        assert!(observed(&calls).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn every_probe_failure_is_retried() -> Result<()> {
        let (profile, _calls) = profile(Arc::new(ErrorScorer), false, false)?;

        let first = profile.process(input("unavailable probe")).await?;
        let second = profile.process(input("unavailable probe")).await?;

        assert_eq!(first.decision.tier, TIER_STRONG);
        assert_eq!(second.decision.tier, TIER_STRONG);
        assert_eq!(profile.decision_cache.lock().len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn run_calls_selected_backend_and_returns_routing_metadata() -> Result<()> {
        let (weak_profile, weak_calls) = profile(Arc::new(FixedScorer(1.0)), false, false)?;

        let response = weak_profile.run(input("route weak")).await?;

        let weak_observed = observed(&weak_calls);
        assert_eq!(weak_observed.len(), 1);
        assert_eq!(weak_observed[0].backend, "weak-backend");
        assert_eq!(weak_observed[0].body["model"], "cheap/model");
        let metadata = response.routing_metadata.ok_or_else(|| {
            SwitchyardError::Other("routing metadata should be present".to_string())
        })?;
        assert_eq!(metadata.selected_model.as_deref(), Some("cheap/model"));
        assert_eq!(metadata.selected_tier.as_deref(), Some(TIER_WEAK));
        assert_eq!(metadata.confidence, None);
        assert_eq!(metadata.router_version.as_deref(), Some("prefill-probe:v1"));
        assert_eq!(metadata.tolerance, Some(0.5));
        assert!(metadata
            .rationale
            .as_deref()
            .is_some_and(|rationale| rationale.contains("score 1 selected weak")));

        let (strong_profile, strong_calls) = profile(Arc::new(FixedScorer(0.0)), false, false)?;
        let response = strong_profile.run(input("route strong")).await?;
        let strong_observed = observed(&strong_calls);
        assert_eq!(strong_observed.len(), 1);
        assert_eq!(strong_observed[0].backend, "strong-backend");
        assert_eq!(strong_observed[0].body["model"], "frontier/model");
        let metadata = response.routing_metadata.ok_or_else(|| {
            SwitchyardError::Other("strong routing metadata should be present".to_string())
        })?;
        assert_eq!(metadata.selected_model.as_deref(), Some("frontier/model"));
        assert_eq!(metadata.selected_tier.as_deref(), Some(TIER_STRONG));
        Ok(())
    }

    #[tokio::test]
    async fn run_records_usage_for_selected_tier() -> Result<()> {
        let (profile, _calls) = profile(Arc::new(FixedScorer(1.0)), false, false)?;

        let _response = profile.run(input("record weak")).await?;

        let snapshot = profile.stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_tokens.prompt, 5);
        assert_eq!(snapshot.total_tokens.completion, 3);
        let tier = snapshot.tiers.get(TIER_WEAK).ok_or_else(|| {
            SwitchyardError::Other("weak tier stats should be present".to_string())
        })?;
        assert_eq!(tier.calls, 1);
        assert_eq!(tier.model, "cheap/model");
        Ok(())
    }

    #[tokio::test]
    async fn backend_failure_is_attributed_to_selected_model_and_tier() -> Result<()> {
        let (profile, calls) = profile(Arc::new(FixedScorer(1.0)), false, true)?;

        let error = profile
            .run(input("weak backend fails"))
            .await
            .err()
            .ok_or_else(|| {
                SwitchyardError::Other("backend failure should be returned".to_string())
            })?;

        assert!(format!("{error}").contains("weak-backend failed"));
        assert_eq!(observed(&calls).len(), 1);
        let snapshot = profile.stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_errors, 1);
        let model = snapshot.models.get("cheap/model").ok_or_else(|| {
            SwitchyardError::Other("weak model error stats should be present".to_string())
        })?;
        assert_eq!(model.errors, 1);
        assert_eq!(model.tier.as_deref(), Some(TIER_WEAK));
        Ok(())
    }
}

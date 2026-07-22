// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random-routing profile implemented as a single profile-owned runtime.

use std::time::Instant;

use async_trait::async_trait;
use switchyard_components::request_processors::{
    RandomRoutingDecision, RandomRoutingEngine, RandomRoutingProcessorConfig, RandomRoutingTier,
};
use switchyard_components::StatsAccumulator;
use switchyard_core::{ChatResponse, LlmTarget, LlmTargetId, Result, SwitchyardError};

use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::stats_recording::record_usage_or_wrap_stream;
use crate::{
    profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse,
    RoutingMetadata,
};

/// Config for the flatter random-routing profile.
#[profile_config("random-routing")]
pub struct RandomRoutingProfileConfig {
    /// Strong target served by this profile.
    #[profile_target]
    pub strong: LlmTarget,
    /// Weak target served by this profile.
    #[profile_target]
    pub weak: LlmTarget,
    /// Probability of selecting the strong target.
    #[serde(default = "default_strong_probability")]
    pub strong_probability: f64,
    /// Optional deterministic RNG seed for reproducible routing.
    #[serde(default)]
    pub rng_seed: Option<u64>,
    /// Target used for one retry after context-window overflow; defaults to the strong target.
    #[serde(default)]
    pub fallback_target_on_evict: Option<LlmTargetId>,
}

impl ProfileConfig for RandomRoutingProfileConfig {
    type Runtime = RandomRoutingProfile;

    /// Builds the runtime profile using existing native backend construction.
    fn build(&self) -> Result<Self::Runtime> {
        // Resolve the evict fallback to the strong target when unset, then validate it
        // names one of this profile's two configured targets.
        let fallback_target_on_evict = self
            .fallback_target_on_evict
            .clone()
            .unwrap_or_else(|| self.strong.id.clone());
        if fallback_target_on_evict != self.strong.id && fallback_target_on_evict != self.weak.id {
            return Err(SwitchyardError::InvalidConfig(format!(
                "fallback_target_on_evict={} must match one of [{}, {}]",
                fallback_target_on_evict, self.weak.id, self.strong.id
            )));
        }
        let router_config =
            RandomRoutingProcessorConfig::new(self.strong.clone(), self.weak.clone())
                .with_strong_probability(self.strong_probability)?
                .with_rng_seed(self.rng_seed);
        Ok(RandomRoutingProfile {
            router: RandomRoutingEngine::new(router_config)?,
            strong_backend: native_target_backend(self.strong.clone())?,
            weak_backend: native_target_backend(self.weak.clone())?,
            fallback_target_on_evict,
            stats: profile_stats_accumulator(),
        })
    }
}

/// Random-routing profile in the flatter design.
pub struct RandomRoutingProfile {
    router: RandomRoutingEngine,
    strong_backend: TargetBackend,
    weak_backend: TargetBackend,
    fallback_target_on_evict: LlmTargetId,
    stats: StatsAccumulator,
}

/// Processed random-routing request with the profile-owned routing decision.
pub struct RandomRoutingProcessedRequest {
    /// Routed input prepared for the selected backend.
    pub profile_input: ProfileInput,
    /// Selected routing decision for this request.
    pub decision: RandomRoutingDecision,
}

impl RandomRoutingProfile {
    // Selects the target and rewrites the request model without side-channel state.
    fn route_request(&self, mut input: ProfileInput) -> Result<RandomRoutingProcessedRequest> {
        let decision = self
            .router
            .select(input.request.model().map(std::borrow::ToOwned::to_owned))?;
        input.request.set_model(decision.selected_model.as_str());
        Ok(RandomRoutingProcessedRequest {
            profile_input: input,
            decision,
        })
    }

    // Finds the routed backend by the target ID emitted by the routing engine.
    fn backend_for_target(&self, target_id: &LlmTargetId) -> Result<&TargetBackend> {
        if *target_id == self.strong_backend.target().id {
            Ok(&self.strong_backend)
        } else if *target_id == self.weak_backend.target().id {
            Ok(&self.weak_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "random-routing selected target {target_id} that is not configured for this profile"
            )))
        }
    }

    // Maps a configured target ID back to its random-routing tier.
    fn tier_for_target(&self, target_id: &LlmTargetId) -> Result<RandomRoutingTier> {
        if *target_id == self.strong_backend.target().id {
            Ok(RandomRoutingTier::Strong)
        } else if *target_id == self.weak_backend.target().id {
            Ok(RandomRoutingTier::Weak)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "random-routing target {target_id} is not configured for this profile"
            )))
        }
    }

    // Rewrites the processed request to dispatch against the evict fallback target.
    fn fallback_processed_request(
        &self,
        processed: &RandomRoutingProcessedRequest,
    ) -> Result<RandomRoutingProcessedRequest> {
        let backend = self.backend_for_target(&self.fallback_target_on_evict)?;
        let target = backend.target();
        let mut profile_input = processed.profile_input.clone();
        profile_input.request.set_model(target.model.as_str());
        let mut decision = processed.decision.clone();
        decision.tier = self.tier_for_target(&target.id)?;
        decision.selected_target = target.id.clone();
        decision.selected_model = target.model.clone();
        Ok(RandomRoutingProcessedRequest {
            profile_input,
            decision,
        })
    }

    // Dispatches the routed request to its backend and reports the backend latency.
    async fn call_selected(
        &self,
        processed: &RandomRoutingProcessedRequest,
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

    // Records success stats for the selected tier and finalizes the response,
    // wrapping streams so usage is recorded when the stream ends.
    fn record_success(
        &self,
        decision: &RandomRoutingDecision,
        response: ChatResponse,
        profile_started_at: Instant,
        backend_latency_ms: f64,
    ) -> Result<ChatResponse> {
        self.stats.record_success(
            decision.selected_model.as_str(),
            Some(backend_latency_ms),
            Some(decision.tier.as_str()),
        )?;
        record_usage_or_wrap_stream(
            &self.stats,
            decision.selected_model.as_str(),
            Some(decision.tier.as_str()),
            profile_started_at,
            backend_latency_ms,
            response,
        )
    }

    // Records a failed call against the selected model and tier stats.
    fn record_error(&self, decision: &RandomRoutingDecision) -> Result<()> {
        self.stats.record_error(
            decision.selected_model.as_str(),
            Some(decision.tier.as_str()),
        )
    }

    fn routing_metadata(&self, decision: &RandomRoutingDecision) -> RoutingMetadata {
        let comparison = if decision.tier == RandomRoutingTier::Strong {
            "<"
        } else {
            ">="
        };
        RoutingMetadata {
            selected_model: Some(decision.selected_model.to_string()),
            selected_tier: Some(decision.tier.as_str().to_string()),
            confidence: None,
            router_version: Some("random-routing:v1".to_string()),
            tolerance: Some(decision.strong_probability),
            rationale: Some(format!(
                "random draw {} {comparison} strong_probability {}; selected {}",
                decision.draw,
                decision.strong_probability,
                decision.tier.as_str()
            )),
        }
    }
}

#[async_trait]
impl ProfileHooks for RandomRoutingProfile {
    type ProcessedRequest = RandomRoutingProcessedRequest;

    /// Performs a standalone routing rewrite for hook-level inspection.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        self.route_request(input)
    }

    /// Leaves the backend response unchanged after random routing completes.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for RandomRoutingProfile {
    /// Executes random routing with one context-window fallback retry.
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
                Ok(ProfileResponse::with_routing_metadata(
                    response,
                    self.routing_metadata(&processed.decision),
                ))
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
                        // The random-draw rationale describes the original tier, which
                        // is now wrong after the fallback switched targets. Replace it
                        // with the overflow-and-retry explanation.
                        let mut routing_metadata = self.routing_metadata(&retry.decision);
                        routing_metadata.rationale = Some(format!(
                            "selected target {} exceeded its context window; retried fallback target {}",
                            processed.decision.selected_target, retry.decision.selected_target
                        ));
                        Ok(ProfileResponse::with_routing_metadata(
                            response,
                            routing_metadata,
                        ))
                    }
                    Err(SwitchyardError::ContextWindowExceeded { target_id, .. }) => {
                        self.record_error(&retry.decision)?;
                        Err(SwitchyardError::ContextPoolExhausted {
                            last_target_id: target_id,
                            reason: "all attempted targets returned context-window overflow"
                                .to_string(),
                        })
                    }
                    Err(error) => {
                        self.record_error(&retry.decision)?;
                        Err(error)
                    }
                }
            }
            Err(error) => {
                self.record_error(&processed.decision)?;
                Err(error)
            }
        }
    }
}

/// Default probability matches the existing random-routing config.
fn default_strong_probability() -> f64 {
    0.5
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures_util::StreamExt;
    use serde_json::{json, Value};
    use switchyard_core::{
        BackendFormat, ChatRequest, LlmTargetId, ModelId, StreamEvent, SwitchyardError,
    };

    use crate::backend::{ProfileBackend, TargetBackend};
    use crate::RequestMetadata;

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct ObservedCall {
        backend: &'static str,
        body: Value,
    }

    struct TestBackend {
        name: &'static str,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
    }

    struct StreamingUsageBackend;

    // Records the call, then reports a context-window overflow for evict/retry tests.
    struct ContextOverflowBackend {
        name: &'static str,
        target_id: String,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
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
            Ok(ChatResponse::openai_completion(json!({
                "served_by": self.name,
                "model": request.model(),
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                },
            })))
        }
    }

    #[async_trait]
    impl ProfileBackend for StreamingUsageBackend {
        async fn call(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse::OpenAiStream(Box::pin(
                futures_util::stream::iter([
                    Ok(StreamEvent::Json(json!({
                        "choices": [{"delta": {"content": "ok"}}],
                    }))),
                    Ok(StreamEvent::Json(json!({
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 4,
                            "total_tokens": 7,
                        },
                    }))),
                ]),
            )))
        }
    }

    #[async_trait]
    impl ProfileBackend for ContextOverflowBackend {
        async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.calls
                .lock()
                .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))?
                .push(ObservedCall {
                    backend: self.name,
                    body: request.body().clone(),
                });
            Err(SwitchyardError::ContextWindowExceeded {
                target_id: self.target_id.clone(),
                model: request.model().unwrap_or("").to_string(),
                message: "prompt is too long".to_string(),
            })
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    fn config(strong: LlmTarget, weak: LlmTarget, probability: f64) -> RandomRoutingProfileConfig {
        RandomRoutingProfileConfig {
            strong,
            weak,
            strong_probability: probability,
            rng_seed: Some(7),
            fallback_target_on_evict: None,
        }
    }

    fn backend(
        strong: &LlmTarget,
        weak: &LlmTarget,
        calls: Arc<Mutex<Vec<ObservedCall>>>,
    ) -> (TargetBackend, TargetBackend) {
        (
            TargetBackend::new(
                strong.clone(),
                Arc::new(TestBackend {
                    name: "strong-backend",
                    calls: calls.clone(),
                }),
            ),
            TargetBackend::new(
                weak.clone(),
                Arc::new(TestBackend {
                    name: "weak-backend",
                    calls,
                }),
            ),
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
        strong: LlmTarget,
        weak: LlmTarget,
        probability: f64,
    ) -> Result<(RandomRoutingProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = config(strong.clone(), weak.clone(), probability);
        let router_config =
            RandomRoutingProcessorConfig::new(config.strong.clone(), config.weak.clone())
                .with_strong_probability(config.strong_probability)?
                .with_rng_seed(config.rng_seed);
        let (strong_backend, weak_backend) = backend(&strong, &weak, calls.clone());
        let profile = RandomRoutingProfile {
            router: RandomRoutingEngine::new(router_config)?,
            strong_backend,
            weak_backend,
            fallback_target_on_evict: strong.id.clone(),
            stats: StatsAccumulator::new(),
        };
        Ok((profile, calls))
    }

    #[tokio::test]
    async fn random_routing_profile_routes_with_request_only_handoff() -> Result<()> {
        let (profile, calls) = profile(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            1.0,
        )?;

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [{"role": "user", "content": "hi"}],
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
        assert_eq!(routing_metadata.selected_tier.as_deref(), Some("strong"));
        assert_eq!(
            routing_metadata.router_version.as_deref(),
            Some("random-routing:v1")
        );
        assert_eq!(routing_metadata.tolerance, Some(1.0));
        let response = response.response;
        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].backend, "strong-backend");
        assert_eq!(calls[0].body["model"], "frontier/model");
        match response {
            ChatResponse::OpenAiCompletion(body) => {
                assert_eq!(body.body()["served_by"], "strong-backend");
                assert_eq!(body.body()["model"], "frontier/model");
            }
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_records_stats_with_selected_random_tier() -> Result<()> {
        let (profile, _calls) = profile(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            1.0,
        )?;

        let _response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            }))))
            .await?;

        let snapshot = profile.stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_tokens.prompt, 11);
        assert_eq!(snapshot.total_tokens.completion, 7);
        let model = snapshot.models.get("frontier/model").ok_or_else(|| {
            SwitchyardError::Other("frontier model stats should be present".into())
        })?;
        assert_eq!(model.calls, 1);
        assert_eq!(model.tier.as_deref(), Some("strong"));
        let tier = snapshot
            .tiers
            .get("strong")
            .ok_or_else(|| SwitchyardError::Other("strong tier stats should be present".into()))?;
        assert_eq!(tier.calls, 1);
        assert_eq!(tier.model, "frontier/model");
        Ok(())
    }

    #[tokio::test]
    async fn run_records_streaming_usage_with_selected_random_tier() -> Result<()> {
        let strong = target("strong", "frontier/model")?;
        let weak = target("weak", "cheap/model")?;
        let router_config = RandomRoutingProcessorConfig::new(strong.clone(), weak.clone())
            .with_strong_probability(0.0)?
            .with_rng_seed(Some(7));
        let strong_id = strong.id.clone();
        let profile = RandomRoutingProfile {
            router: RandomRoutingEngine::new(router_config)?,
            strong_backend: TargetBackend::new(
                strong,
                Arc::new(TestBackend {
                    name: "strong-backend",
                    calls: Arc::new(Mutex::new(Vec::new())),
                }),
            ),
            weak_backend: TargetBackend::new(weak, Arc::new(StreamingUsageBackend)),
            fallback_target_on_evict: strong_id,
            stats: StatsAccumulator::new(),
        };

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
                "stream": true,
            }))))
            .await?;
        let ChatResponse::OpenAiStream(mut stream) = response.response else {
            return Err(SwitchyardError::Other("expected streaming response".into()));
        };
        while let Some(event) = stream.next().await {
            event?;
        }

        let snapshot = profile.stats.snapshot()?;
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.total_tokens.prompt, 3);
        assert_eq!(snapshot.total_tokens.completion, 4);
        assert_eq!(snapshot.total_tokens.total, 7);
        let model = snapshot
            .models
            .get("cheap/model")
            .ok_or_else(|| SwitchyardError::Other("weak model stats should be present".into()))?;
        assert_eq!(model.calls, 1);
        assert_eq!(model.tier.as_deref(), Some("weak"));
        assert_eq!(model.total_tokens, 7);
        let tier = snapshot
            .tiers
            .get("weak")
            .ok_or_else(|| SwitchyardError::Other("weak tier stats should be present".into()))?;
        assert_eq!(tier.total_tokens, 7);
        Ok(())
    }

    #[tokio::test]
    async fn run_disambiguates_duplicate_target_models_without_context_state() -> Result<()> {
        let (profile, calls) = profile(
            target("strong-endpoint", "shared/model")?,
            target("weak-endpoint", "shared/model")?,
            0.0,
        )?;

        let _response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            }))))
            .await?;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].backend, "weak-backend");
        assert_eq!(calls[0].body["model"], "shared/model");
        Ok(())
    }

    #[tokio::test]
    async fn malformed_request_body_is_recovered_without_context_state() -> Result<()> {
        let (profile, calls) = profile(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            1.0,
        )?;

        let _response = profile
            .run(profile_input(ChatRequest::openai_chat(json!("bad-body"))))
            .await?;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].backend, "strong-backend");
        assert_eq!(calls[0].body, json!({"model": "frontier/model"}));
        Ok(())
    }

    #[tokio::test]
    async fn process_only_prepares_request_and_does_not_call_backend() -> Result<()> {
        let (profile, calls) = profile(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            1.0,
        )?;

        let request = profile
            .process(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            }))))
            .await?;

        assert_eq!(
            request.profile_input.request.model(),
            Some("frontier/model")
        );
        assert_eq!(request.decision.selected_model.as_str(), "frontier/model");
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rprocess_only_handles_response() -> Result<()> {
        let (profile, calls) = profile(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            1.0,
        )?;

        let processed = profile
            .process(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            }))))
            .await?;
        let response = profile
            .rprocess(
                &processed,
                ChatResponse::openai_completion(json!({"ok": true})),
            )
            .await?;

        match response {
            ChatResponse::OpenAiCompletion(body) => assert_eq!(body.body()["ok"], true),
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn run_retries_fallback_target_after_context_overflow() -> Result<()> {
        let strong = target("strong", "frontier/model")?;
        let weak = target("weak", "cheap/model")?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        // strong_probability 0.0 selects the weak target first; the default fallback is strong.
        let router_config = RandomRoutingProcessorConfig::new(strong.clone(), weak.clone())
            .with_strong_probability(0.0)?
            .with_rng_seed(Some(7));
        let profile = RandomRoutingProfile {
            router: RandomRoutingEngine::new(router_config)?,
            strong_backend: TargetBackend::new(
                strong.clone(),
                Arc::new(TestBackend {
                    name: "strong-backend",
                    calls: calls.clone(),
                }),
            ),
            weak_backend: TargetBackend::new(
                weak.clone(),
                Arc::new(ContextOverflowBackend {
                    name: "weak-backend",
                    target_id: weak.id.to_string(),
                    calls: calls.clone(),
                }),
            ),
            fallback_target_on_evict: strong.id.clone(),
            stats: StatsAccumulator::new(),
        };

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
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
        assert_eq!(routing_metadata.selected_tier.as_deref(), Some("strong"));
        // After the fallback switches tiers, the rationale describes the
        // overflow-and-retry, not the original random draw.
        assert_eq!(
            routing_metadata.rationale.as_deref(),
            Some("selected target weak exceeded its context window; retried fallback target strong")
        );
        let response = response.response;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].backend, "weak-backend");
        assert_eq!(calls[0].body["model"], "cheap/model");
        assert_eq!(calls[1].backend, "strong-backend");
        assert_eq!(calls[1].body["model"], "frontier/model");
        match response {
            ChatResponse::OpenAiCompletion(body) => {
                assert_eq!(body.body()["served_by"], "strong-backend");
                assert_eq!(body.body()["model"], "frontier/model");
            }
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }
        Ok(())
    }

    #[test]
    fn build_rejects_fallback_target_not_matching_strong_or_weak() -> Result<()> {
        let mut config = config(
            target("strong", "frontier/model")?,
            target("weak", "cheap/model")?,
            0.5,
        );
        config.fallback_target_on_evict = Some(LlmTargetId::new("ghost")?);

        match config.build() {
            Err(SwitchyardError::InvalidConfig(message)) => {
                assert!(message.contains("fallback_target_on_evict"));
            }
            Ok(_) => {
                return Err(SwitchyardError::Other(
                    "unknown fallback target should reject profile construction".into(),
                ));
            }
            Err(other) => {
                return Err(SwitchyardError::Other(format!(
                    "expected InvalidConfig, got {other}"
                )));
            }
        }
        Ok(())
    }
}

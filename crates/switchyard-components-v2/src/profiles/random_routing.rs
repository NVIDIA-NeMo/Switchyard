// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random-routing profile implemented as a single profile-owned runtime.

use std::time::Instant;

use async_trait::async_trait;
use switchyard_components::request_processors::{
    RandomRoutingDecision, RandomRoutingEngine, RandomRoutingProcessorConfig, RandomRoutingTier,
};
use switchyard_components::StatsAccumulator;
use switchyard_core::{ChatResponse, LlmTarget, Result, SwitchyardError};

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
}

impl ProfileConfig for RandomRoutingProfileConfig {
    type Runtime = RandomRoutingProfile;

    /// Builds the runtime profile using existing native backend construction.
    fn build(&self) -> Result<Self::Runtime> {
        let router_config =
            RandomRoutingProcessorConfig::new(self.strong.clone(), self.weak.clone())
                .with_strong_probability(self.strong_probability)?
                .with_rng_seed(self.rng_seed);
        Ok(RandomRoutingProfile {
            router: RandomRoutingEngine::new(router_config)?,
            strong_backend: native_target_backend(self.strong.clone())?,
            weak_backend: native_target_backend(self.weak.clone())?,
            stats: profile_stats_accumulator(),
        })
    }
}

/// Random-routing profile in the flatter design.
pub struct RandomRoutingProfile {
    router: RandomRoutingEngine,
    strong_backend: TargetBackend,
    weak_backend: TargetBackend,
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
    fn selected_backend(&self, decision: &RandomRoutingDecision) -> Result<&TargetBackend> {
        if decision.selected_target == self.strong_backend.target().id {
            Ok(&self.strong_backend)
        } else if decision.selected_target == self.weak_backend.target().id {
            Ok(&self.weak_backend)
        } else {
            Err(SwitchyardError::InvalidConfig(format!(
                "router selected target {} that is not configured for this profile",
                decision.selected_target
            )))
        }
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
    /// Executes random routing while keeping selected-target state local to this call.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile_started_at = Instant::now();
        let processed = self.process(input).await?;
        let decision = &processed.decision;
        let selected_backend = self.selected_backend(decision)?;
        let backend_started_at = Instant::now();
        let response = match selected_backend
            .call(&processed.profile_input.request)
            .await
        {
            Ok(response) => response,
            Err(error) => {
                self.stats.record_error(
                    decision.selected_model.as_str(),
                    Some(decision.tier.as_str()),
                )?;
                return Err(error);
            }
        };
        let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
        self.stats.record_success(
            decision.selected_model.as_str(),
            Some(backend_latency_ms),
            Some(decision.tier.as_str()),
        )?;
        let response = record_usage_or_wrap_stream(
            &self.stats,
            decision.selected_model.as_str(),
            Some(decision.tier.as_str()),
            profile_started_at,
            backend_latency_ms,
            response,
        )?;
        let response = self.rprocess(&processed, response).await?;
        Ok(ProfileResponse::with_routing_metadata(
            response,
            self.routing_metadata(decision),
        ))
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
}

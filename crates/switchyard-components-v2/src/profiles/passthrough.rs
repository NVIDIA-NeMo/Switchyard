// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Passthrough profile implemented as a single profile-owned runtime.

use std::time::Instant;

use async_trait::async_trait;
use switchyard_components::otel_metrics;
use switchyard_components::stats::usage_from_body;
use switchyard_core::{ChatResponse, LlmTarget, Result};

use crate::backend::{native_target_backend, TargetBackend};
use crate::{profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse};

/// Config for the flatter passthrough profile.
#[profile_config("passthrough")]
pub struct PassthroughProfileConfig {
    /// Target served by this profile without routing.
    #[profile_target]
    pub target: LlmTarget,
}

impl ProfileConfig for PassthroughProfileConfig {
    type Runtime = PassthroughProfile;

    /// Builds the runtime profile using the existing native backend stack.
    fn build(&self) -> Result<Self::Runtime> {
        Ok(PassthroughProfile {
            backend: native_target_backend(self.target.clone())?,
        })
    }
}

/// Passthrough profile in the flatter design.
pub struct PassthroughProfile {
    backend: TargetBackend,
}

#[async_trait]
impl ProfileHooks for PassthroughProfile {
    type ProcessedRequest = ProfileInput;

    /// Rewrites the request model to the single configured target.
    async fn process(&self, mut input: ProfileInput) -> Result<Self::ProcessedRequest> {
        input
            .request
            .set_model(self.backend.target().model.as_str());
        Ok(input)
    }

    /// Leaves the response unchanged after the backend call.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for PassthroughProfile {
    /// Executes passthrough by composing the hook methods around one backend call.
    ///
    /// Passthrough has no per-call routing decision to preserve, so `run()` can use the
    /// straightforward `process -> backend -> rprocess` shape described by the
    /// [`Profile`] trait. Profiles that select targets dynamically may use a different
    /// internal flow while still exposing the same public lifecycle methods.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile_started_at = Instant::now();
        let processed = self.process(input).await?;
        let target_model = self.backend.target().model.clone();
        let backend_started_at = Instant::now();
        let response = match self.backend.call(&processed.request).await {
            Ok(response) => response,
            Err(error) => {
                otel_metrics::record_error(target_model.as_str(), None, None);
                return Err(error);
            }
        };
        let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
        otel_metrics::record_request(target_model.as_str(), None, None);
        let total_latency_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;
        let routing_overhead_ms = (total_latency_ms - backend_latency_ms).max(0.0);
        otel_metrics::record_latencies(
            target_model.as_str(),
            None,
            None,
            Some(backend_latency_ms),
            Some(total_latency_ms),
            Some(routing_overhead_ms),
        );
        let usage = response.body().map(usage_from_body).unwrap_or_default();
        otel_metrics::record_usage_and_cost(target_model.as_str(), None, "routed", usage);
        let response = self.rprocess(&processed, response).await?;
        Ok(ProfileResponse::from(response))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::{json, Value};
    use switchyard_core::{BackendFormat, ChatRequest, LlmTargetId, ModelId, SwitchyardError};

    use crate::backend::{ProfileBackend, TargetBackend};
    use crate::RequestMetadata;

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct ObservedCall {
        body: Value,
    }

    struct TestBackend {
        calls: Arc<Mutex<Vec<ObservedCall>>>,
    }

    #[async_trait]
    impl ProfileBackend for TestBackend {
        async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.calls
                .lock()
                .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))?
                .push(ObservedCall {
                    body: request.body().clone(),
                });
            Ok(ChatResponse::openai_completion(json!({
                "model": request.model(),
                "served_model": request.model(),
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                },
            })))
        }
    }

    fn target(id: &str, model: &str) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = BackendFormat::OpenAi;
        Ok(target)
    }

    fn backend(target: &LlmTarget, calls: Arc<Mutex<Vec<ObservedCall>>>) -> TargetBackend {
        TargetBackend::new(target.clone(), Arc::new(TestBackend { calls }))
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

    fn profile(target: LlmTarget) -> Result<(PassthroughProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let profile = PassthroughProfile {
            backend: backend(&target, calls.clone()),
        };
        Ok((profile, calls))
    }

    #[tokio::test]
    async fn passthrough_profile_calls_single_target_backend() -> Result<()> {
        let (profile, calls) = profile(target("direct", "provider/model")?)?;

        let response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [{"role": "user", "content": "hi"}],
            }))))
            .await?;

        let response = response.response;
        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].body["model"], "provider/model");
        match response {
            ChatResponse::OpenAiCompletion(body) => {
                assert_eq!(body.body()["served_model"], "provider/model");
            }
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_emits_otel_metrics_for_single_target() -> Result<()> {
        let (profile, _calls) = profile(target("direct", "metrics-passthrough/model")?)?;

        let _response = profile
            .run(profile_input(ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            }))))
            .await?;

        // The process meter is a singleton, so assert the metric surface mentions
        // this target rather than exact counts another test may have moved.
        let metrics = switchyard_components::otel_metrics::render_prometheus()
            .map_err(SwitchyardError::Other)?;
        assert!(
            metrics.contains("metrics-passthrough/model"),
            "expected request metrics for the target model in:\n{metrics}"
        );
        assert!(metrics.contains("switchyard_requests_total"));
        Ok(())
    }

    #[tokio::test]
    async fn process_only_prepares_target_request_and_does_not_call_backend() -> Result<()> {
        let (profile, calls) = profile(target("direct", "provider/model")?)?;
        let request = ChatRequest::openai_chat(json!({
            "model": "client/model",
            "messages": [],
        }));

        let processed = profile.process(profile_input(request.clone())).await?;

        assert_eq!(processed.request.model(), Some("provider/model"));
        assert_eq!(
            processed.request.body()["messages"],
            request.body()["messages"]
        );
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rprocess_only_returns_response_unchanged() -> Result<()> {
        let (profile, calls) = profile(target("direct", "provider/model")?)?;
        let response = ChatResponse::openai_completion(json!({"ok": true}));

        let request = profile_input(ChatRequest::openai_chat(
            json!({"model": "client/model", "messages": []}),
        ));
        let processed = profile.rprocess(&request, response).await?;

        match processed {
            ChatResponse::OpenAiCompletion(body) => assert_eq!(body.body()["ok"], true),
            _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
        }
        assert!(observed(&calls)?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn malformed_body_still_gets_target_model_for_single_target_backend() -> Result<()> {
        let (profile, calls) = profile(target("direct", "provider/model")?)?;

        let _response = profile
            .run(profile_input(ChatRequest::openai_chat(json!("bad-body"))))
            .await?;

        let calls = observed(&calls)?;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].body, json!({"model": "provider/model"}));
        Ok(())
    }
}

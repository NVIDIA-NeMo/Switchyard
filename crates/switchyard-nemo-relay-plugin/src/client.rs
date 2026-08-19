// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard-owned HTTP clients bound to one semantic routing target.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::Value as Json;
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_protocol::{
    LlmClientError, ModelId, Request, Response, RoutedLlmClient, WireFormat,
};
use switchyard_translation::TranslationEngine;

use crate::translation;

/// A provider client bound to one configured Switchyard target.
///
/// libsy routes with a stable semantic name (for example `fast`). The provider
/// still expects its own model id (for example `meta/llama-3.1-8b-instruct`).
/// Keeping that mapping here prevents an algorithm's semantic labels from
/// leaking into provider requests.
pub(crate) struct TargetClient {
    provider_model: ModelId,
    target_format: WireFormat,
    drop_caller_extra_body: bool,
    inner: TranslatingLlmClient,
    translation: TranslationEngine,
}

impl TargetClient {
    pub(crate) fn new(
        provider_model: String,
        target_format: WireFormat,
        dispatch_url: String,
        headers: BTreeMap<String, String>,
        extra_body: BTreeMap<String, Json>,
        drop_caller_extra_body: bool,
    ) -> Result<Self, LlmClientError> {
        let backend_config = HttpBackendConfig {
            // `dispatch_url` is already resolved by configuration. Backend URL
            // joining accepts a complete canonical endpoint as well as a base
            // URL/prefix.
            base_url: dispatch_url,
            api_key: None,
            forward_auth: false,
            extra_headers: headers,
            extra_body,
            // Routing retries belong to the plugin: every retry must start a
            // fresh libsy run and obtain a fresh decision.
            max_retries: 0,
        };
        let backend = match target_format {
            WireFormat::OpenAiChat => Backend::OpenAiChat(backend_config),
            WireFormat::OpenAiResponses => Backend::OpenAiResponses(backend_config),
            WireFormat::AnthropicMessages => Backend::Anthropic(backend_config),
        };
        let model = ModelConfig::new(provider_model.clone(), backend, None);
        let inner = TranslatingLlmClient::new(&[model])?;
        Ok(Self {
            provider_model: ModelId::from(provider_model),
            target_format,
            drop_caller_extra_body,
            inner,
            translation: TranslationEngine::default(),
        })
    }

    /// Retargets only the provider-facing transport metadata.
    ///
    /// Correlation and agent identity remain available to libsy, while inbound
    /// HTTP headers are deliberately removed. Provider credentials come solely
    /// from this target's `header_env` configuration.
    fn prepare_request(&self, mut request: Request) -> Request {
        let metadata = request.metadata.get_or_insert_default();
        metadata.wire_format = Some(self.target_format);
        metadata.http_headers = None;
        if self.drop_caller_extra_body {
            request.llm_request.extensions.fields.remove("extra_body");
            for preserved in request.llm_request.preservation.requests.values_mut() {
                if let Some(body) = preserved.as_object_mut() {
                    body.remove("extra_body");
                }
            }
        }
        request
    }
}

#[async_trait]
impl RoutedLlmClient for TargetClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let request = self.prepare_request(request);
        translation::validate_target_request(
            &self.translation,
            self.target_format,
            &request.llm_request,
        )
        .map_err(LlmClientError::RequestEncoding)?;
        self.inner
            .call_rewrite_model(request, Some(&self.provider_model))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::{LlmRequest, Metadata, PreservationMetadata, ProviderExtensions};

    fn client(format: WireFormat) -> TargetClient {
        TargetClient::new(
            "provider/model".into(),
            format,
            match format {
                WireFormat::OpenAiChat => "https://provider.example/v1/chat/completions".into(),
                WireFormat::OpenAiResponses => "https://provider.example/v1/responses".into(),
                WireFormat::AnthropicMessages => "https://provider.example/v1/messages".into(),
            },
            BTreeMap::new(),
            BTreeMap::new(),
            false,
        )
        .unwrap()
    }

    #[test]
    fn target_preparation_forces_format_and_removes_inbound_headers() {
        let client = client(WireFormat::AnthropicMessages);
        let request = Request {
            metadata: Some(Metadata {
                correlation_id: Some("request-123".into()),
                wire_format: Some(WireFormat::OpenAiChat),
                http_headers: Some(http::HeaderMap::from_iter([
                    (
                        http::HeaderName::from_static("authorization"),
                        http::HeaderValue::from_static("Bearer caller-secret"),
                    ),
                    (
                        http::HeaderName::from_static("x-caller-only"),
                        http::HeaderValue::from_static("must-not-forward"),
                    ),
                ])),
                ..Metadata::default()
            }),
            ..Request::default()
        };

        let prepared = client.prepare_request(request);
        let metadata = prepared.metadata.unwrap();
        assert_eq!(metadata.wire_format, Some(WireFormat::AnthropicMessages));
        assert_eq!(metadata.correlation_id.as_deref(), Some("request-123"));
        assert!(metadata.http_headers.is_none());
    }

    #[test]
    fn missing_metadata_is_created_for_the_target_format() {
        let client = client(WireFormat::OpenAiResponses);
        let prepared = client.prepare_request(Request::default());
        assert_eq!(
            prepared.metadata.and_then(|metadata| metadata.wire_format),
            Some(WireFormat::OpenAiResponses)
        );
    }

    #[test]
    fn configured_target_drops_intercepted_caller_extra_body() {
        let client = TargetClient::new(
            "provider/model".into(),
            WireFormat::OpenAiChat,
            "https://provider.example/v1/chat/completions".into(),
            BTreeMap::new(),
            BTreeMap::new(),
            true,
        )
        .unwrap();
        let request = Request {
            llm_request: LlmRequest {
                extensions: ProviderExtensions {
                    fields: serde_json::Map::from_iter([(
                        "extra_body".into(),
                        json!({"reasoning": {"effort": "medium"}}),
                    )]),
                },
                preservation: PreservationMetadata {
                    requests: BTreeMap::from([(
                        WireFormat::OpenAiChat.into(),
                        json!({
                            "model": "route",
                            "messages": [{"role": "user", "content": "hello"}],
                            "extra_body": {
                                "reasoning": {"effort": "medium"},
                                "session_id": "hermes-session"
                            }
                        }),
                    )]),
                    ..PreservationMetadata::default()
                },
                ..LlmRequest::default()
            },
            ..Request::default()
        };

        let prepared = client.prepare_request(request);
        assert!(
            !prepared
                .llm_request
                .extensions
                .fields
                .contains_key("extra_body")
        );
        assert!(
            prepared
                .llm_request
                .preservation
                .requests
                .values()
                .all(|body| body.get("extra_body").is_none())
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered provider JSON translation using the default translation policy.

use napi_derive::napi;
use serde_json::{Value, json};
use switchyard_translation::{
    AggLlmResponse, LlmRequest, ProviderExtensions, TranslationEngine, TranslationError,
    TranslationOutput, TranslationPolicy,
};

use crate::{error, parse};

#[napi]
#[derive(Default)]
pub struct NativeTranslationEngine {
    inner: TranslationEngine,
    policy: TranslationPolicy,
}

#[napi]
impl NativeTranslationEngine {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[napi]
    pub fn decode_request(&self, source: String, body_json: String) -> napi::Result<String> {
        let body: Value = parse(&body_json)?;
        let output = self
            .inner
            .decode_request(source.as_str(), &body, &self.policy)
            .map_err(translation_error)?;
        Ok(json!({"request": output.request, "diagnostics": output.diagnostics}).to_string())
    }

    #[napi]
    pub fn encode_request(&self, target: String, request_json: String) -> napi::Result<String> {
        let request: LlmRequest = parse(&request_json)?;
        translation_output(
            self.inner
                .encode_request(target.as_str(), &request, &self.policy),
        )
    }

    #[napi]
    pub fn decode_response(&self, source: String, body_json: String) -> napi::Result<String> {
        let body: Value = parse(&body_json)?;
        let output = self
            .inner
            .decode_response(source.as_str(), &body, &self.policy)
            .map_err(translation_error)?;
        Ok(json!({"response": output.response, "diagnostics": output.diagnostics}).to_string())
    }

    #[napi]
    pub fn encode_response(
        &self,
        target: String,
        response_json: String,
        request_extensions_json: Option<String>,
    ) -> napi::Result<String> {
        let response: AggLlmResponse = parse(&response_json)?;
        let extensions: ProviderExtensions = request_extensions_json
            .as_deref()
            .map(parse)
            .transpose()?
            .unwrap_or_default();
        translation_output(self.inner.encode_response_with_extensions(
            target.as_str(),
            &response,
            &extensions,
            &self.policy,
        ))
    }

    #[napi]
    pub fn translate_request(
        &self,
        source: String,
        target: String,
        body_json: String,
    ) -> napi::Result<String> {
        let body: Value = parse(&body_json)?;
        translation_output(self.inner.translate_request(
            source.as_str(),
            target.as_str(),
            &body,
            &self.policy,
        ))
    }

    #[napi]
    pub fn translate_response(
        &self,
        source: String,
        target: String,
        body_json: String,
    ) -> napi::Result<String> {
        let body: Value = parse(&body_json)?;
        translation_output(self.inner.translate_response(
            source.as_str(),
            target.as_str(),
            &body,
            &self.policy,
        ))
    }
}

fn translation_output(result: Result<TranslationOutput, TranslationError>) -> napi::Result<String> {
    let output = result.map_err(translation_error)?;
    Ok(json!({"body": output.body, "diagnostics": output.diagnostics}).to_string())
}

fn translation_error(cause: TranslationError) -> napi::Error {
    let details = match &cause {
        TranslationError::InvalidJson(error) => json!({
            "message": error.to_string(),
            "line": error.line(),
            "column": error.column(),
        }),
        TranslationError::InvalidType { path, expected } => {
            json!({"path": path, "expected": expected})
        }
        TranslationError::UnsupportedTranslation { from, to } => json!({"from": from, "to": to}),
        TranslationError::LossyConversion(message) | TranslationError::Other(message) => {
            json!({"message": message})
        }
        TranslationError::UnknownField { path } => json!({"path": path}),
        TranslationError::InvalidValue { path, message } => {
            json!({"path": path, "message": message})
        }
        TranslationError::UpstreamFailure { error } => json!({"error": error}),
    };
    error(
        "ERR_TRANSLATION",
        json!({"message": cause.to_string(), "kind": cause.kind(), "details": details}),
    )
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Removes configured provider credentials from client-facing responses.

use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE};
use axum::middleware::Next;
use axum::response::Response;

use crate::{DEFAULT_MAX_REQUEST_BODY_BYTES, ServerState};

#[derive(Default)]
pub(crate) struct Redactor {
    raw: Vec<String>,
    json: Vec<String>,
}

impl Redactor {
    pub(crate) fn new(keys: &[String]) -> Self {
        let mut raw: Vec<String> = keys.iter().filter(|key| !key.is_empty()).cloned().collect();
        raw.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        raw.dedup();
        let mut json = Vec::with_capacity(raw.len());
        for key in &raw {
            let Ok(encoded) = serde_json::to_string(key) else {
                unreachable!("serde_json::to_string on a String cannot fail");
            };
            json.push(encoded[1..encoded.len() - 1].to_string());
        }
        json.sort_by_key(|key| std::cmp::Reverse(key.len()));
        Self { raw, json }
    }

    pub(crate) fn json(&self, value: String) -> String {
        replace(value, &self.json)
    }

    pub(crate) fn text(&self, value: String) -> String {
        replace(value, &self.raw)
    }
}

fn replace(mut value: String, secrets: &[String]) -> String {
    for secret in secrets {
        if value.contains(secret) {
            value = value.replace(secret, "[REDACTED]");
        }
    }
    value
}

pub(crate) async fn redact_response(
    State(state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let redactor = &state.redactor;
    if redactor.raw.is_empty() {
        return response;
    }
    let is_json = response.headers().get(CONTENT_TYPE).is_some_and(|value| {
        value.to_str().is_ok_and(|value| {
            value.split(';').next().is_some_and(|mime| {
                mime.trim() == "application/json" || mime.trim().ends_with("+json")
            })
        })
    });
    let is_encoded = response
        .headers()
        .get_all(CONTENT_ENCODING)
        .iter()
        .any(|value| {
            value.to_str().map_or(true, |value| {
                value
                    .split(',')
                    .any(|encoding| !encoding.trim().eq_ignore_ascii_case("identity"))
            })
        });
    for value in response.headers_mut().values_mut() {
        if let Ok(text) = value.to_str() {
            let sanitized = redactor.text(text.to_string());
            if sanitized != text {
                // Replacement contains only visible ASCII and cannot invalidate a header.
                if let Ok(header) = sanitized.parse() {
                    *value = header;
                }
            }
        }
    }
    // SSE bodies are redacted per event before framing, without buffering the stream.
    // The fallback proxy passes compressed bodies through without decoding them.
    if is_json && !is_encoded {
        let (mut parts, body) = response.into_parts();
        let body = match to_bytes(body, DEFAULT_MAX_REQUEST_BODY_BYTES).await {
            Ok(bytes) => match String::from_utf8(bytes.to_vec()) {
                Ok(json) => Body::from(redactor.json(json)),
                Err(_) => return crate::server_error("Invalid upstream JSON"),
            },
            Err(_) => return crate::server_error("Unable to read response body"),
        };
        parts.headers.remove(CONTENT_LENGTH);
        response = Response::from_parts(parts, body);
    }
    response
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::Json;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use serde_json::json;
    use switchyard_runner::Runner;
    use switchyard_translation::{LlmStreamError, WireFormat};
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn responses_redact_provider_keys_without_changing_errors()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "synthetic-\"provider\\key";
        let error = json!({"error": {
            "message": "upstream failed",
            "type": "provider_error",
            "debug": format!("Bearer {key}"),
            key: [key, "ordinary diagnostics"]
        }});
        let runner = Runner::new(Vec::new()).with_provider_api_keys(vec![key.to_string()]);
        let state = ServerState::from_runner(runner)?;
        let buffered = error.clone();
        let streamed = error.clone();
        let header = axum::http::HeaderValue::from_str(key)?;
        // gzip-compressed {"ok":true}.
        const GZIP_JSON: &[u8] = &[
            31, 139, 8, 0, 0, 0, 0, 0, 2, 3, 171, 86, 202, 207, 86, 178, 42, 41, 42, 77, 173, 5, 0,
            144, 95, 212, 167, 11, 0, 0, 0,
        ];
        let router = axum::Router::new()
            .route(
                "/compressed",
                get(|| async {
                    (
                        [
                            (CONTENT_TYPE, "application/json"),
                            (CONTENT_ENCODING, "gzip"),
                            (CONTENT_LENGTH, "31"),
                        ],
                        GZIP_JSON,
                    )
                }),
            )
            .route("/buffered", get(move || async move { Json(buffered) }))
            .route(
                "/stream",
                get(move |State(state): State<ServerState>| async move {
                    let events = futures_util::stream::iter([
                        Ok(json!({"choices": [], "model": "ordinary-model"})),
                        Err(LlmStreamError::Upstream(streamed)),
                    ]);
                    let mut response = crate::sse::frame_stream(
                        Box::pin(events),
                        WireFormat::OpenAiChat,
                        Arc::clone(&state.redactor),
                    )
                    .into_response();
                    response.headers_mut().insert("x-upstream-debug", header);
                    response
                }),
            );
        let app = crate::finish_router(router, state);
        let compressed = app
            .clone()
            .oneshot(Request::builder().uri("/compressed").body(Body::empty())?)
            .await?;
        assert_eq!(compressed.status(), axum::http::StatusCode::OK);
        assert_eq!(compressed.headers()[CONTENT_ENCODING], "gzip");
        assert_eq!(compressed.headers()[CONTENT_LENGTH], "31");
        assert_eq!(
            to_bytes(compressed.into_body(), usize::MAX).await?.as_ref(),
            GZIP_JSON
        );
        for path in ["/buffered", "/stream"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty())?)
                .await?;
            if path == "/stream" {
                assert_eq!(response.headers()["x-upstream-debug"], "[REDACTED]");
            }
            let body =
                String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;
            let error: serde_json::Value = if path == "/stream" {
                assert!(body.contains("ordinary-model"));
                assert!(!body.contains("[DONE]"));
                let data = body
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .next_back()
                    .ok_or("missing SSE error")?;
                serde_json::from_str(data)?
            } else {
                serde_json::from_str(&body)?
            };
            assert_eq!(error["error"]["message"], "upstream failed");
            assert_eq!(error["error"]["type"], "provider_error");
            assert_eq!(error["error"]["debug"], "Bearer [REDACTED]");
            assert_eq!(
                error["error"]["[REDACTED]"],
                json!(["[REDACTED]", "ordinary diagnostics"])
            );
            assert!(!body.contains("synthetic-"));
        }
        Ok(())
    }
}

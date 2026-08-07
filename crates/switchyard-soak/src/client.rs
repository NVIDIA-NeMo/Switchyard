// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP against the public Switchyard APIs: preflight, one request, and server-state reads.

use std::collections::BTreeMap;

use clap::ValueEnum;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use reqwest::Client;
use serde_json::{json, Value};

use crate::stats::{SERVER_ERRORS_METRIC, SERVER_REQUESTS_METRIC};

/// One public Switchyard API the soak test exercises.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Endpoint {
    Chat,
    Messages,
    Responses,
}

impl Endpoint {
    pub const ALL: [Endpoint; 3] = [Endpoint::Chat, Endpoint::Messages, Endpoint::Responses];

    pub fn path(self) -> &'static str {
        match self {
            Endpoint::Chat => "/v1/chat/completions",
            Endpoint::Messages => "/v1/messages",
            Endpoint::Responses => "/v1/responses",
        }
    }

    /// Field a successful response for this endpoint must contain.
    fn required_field(self) -> &'static str {
        match self {
            Endpoint::Chat => "choices",
            Endpoint::Messages => "content",
            Endpoint::Responses => "output",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Endpoint::Chat => "chat",
            Endpoint::Messages => "messages",
            Endpoint::Responses => "responses",
        }
    }
}

/// Keep at most *limit* characters, on a char boundary, for a logged detail string.
fn truncate(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

/// Map a transport-level reqwest error to a short, stable error kind.
fn transport_error_kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_decode() {
        // A body decode failure mirrors httpx's DecodingError, which reported "request_error".
        "request_error"
    } else if error.is_connect() || error.is_request() || error.is_body() {
        // Connect, send, and mid-stream read failures are httpx TransportError -> "transport".
        "transport"
    } else {
        "request_error"
    }
}

/// Build one request body for a public Switchyard API.
pub fn request_body(
    endpoint: Endpoint,
    model: &str,
    prompt: &str,
    max_output_tokens: u32,
    stream: bool,
) -> Value {
    match endpoint {
        // Chat Completions and Anthropic Messages take the same model/messages/max_tokens body.
        Endpoint::Chat | Endpoint::Messages => json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "stream": stream,
        }),
        Endpoint::Responses => json!({
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "stream": stream,
        }),
    }
}

/// Read the process-wide Switchyard counters from Prometheus text.
pub fn parse_metrics(text: &str) -> BTreeMap<String, f64> {
    let wanted = [SERVER_REQUESTS_METRIC, SERVER_ERRORS_METRIC];
    let mut parsed = BTreeMap::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name_part, rest)) = line.split_once(' ') else {
            continue;
        };
        let name = name_part.split('{').next().unwrap_or(name_part);
        if !wanted.contains(&name) {
            continue;
        }
        if let Some(token) = rest.split_whitespace().next() {
            if let Ok(value) = token.parse::<f64>() {
                parsed.insert(name.to_string(), value);
            }
        }
    }
    parsed
}

/// Send one request and return an error kind and detail, if any.
pub async fn send_request(
    client: &Client,
    base_url: &str,
    endpoint: Endpoint,
    body: &Value,
) -> (Option<String>, Option<String>) {
    let url = format!("{base_url}{}", endpoint.path());
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let response = match client.post(&url).json(body).send().await {
        Ok(response) => response,
        Err(error) => {
            return (
                Some(transport_error_kind(&error).to_string()),
                Some(truncate(&error.to_string(), 500)),
            )
        }
    };

    if stream {
        let status = response.status();
        if !status.is_success() {
            let content = response.text().await.unwrap_or_default();
            return (
                Some(format!("http_{}", status.as_u16())),
                Some(truncate(&content, 500)),
            );
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !content_type.contains("text/event-stream") {
            let content = response.text().await.unwrap_or_default();
            return (
                Some("invalid_stream".to_string()),
                Some(format!(
                    "expected text/event-stream, received {content_type:?}: {}",
                    truncate(&content, 300)
                )),
            );
        }
        let mut bytes = response.bytes_stream();
        let mut received_data = false;
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(chunk) => {
                    received_data =
                        received_data || chunk.iter().any(|byte| !byte.is_ascii_whitespace());
                }
                Err(error) => {
                    return (
                        Some(transport_error_kind(&error).to_string()),
                        Some(truncate(&error.to_string(), 500)),
                    )
                }
            }
        }
        if !received_data {
            return (
                Some("empty_stream".to_string()),
                Some("successful streaming response contained no data".to_string()),
            );
        }
        return (None, None);
    }

    let status = response.status();
    if !status.is_success() {
        let content = response.text().await.unwrap_or_default();
        return (
            Some(format!("http_{}", status.as_u16())),
            Some(truncate(&content, 500)),
        );
    }
    let text = match response.text().await {
        Ok(text) => text,
        Err(error) => {
            return (
                Some(transport_error_kind(&error).to_string()),
                Some(truncate(&error.to_string(), 500)),
            )
        }
    };
    let payload: Value = match serde_json::from_str(&text) {
        Ok(payload) => payload,
        Err(error) => return (Some("invalid_json".to_string()), Some(error.to_string())),
    };
    if !payload.is_object() || payload.get("error").is_some() {
        return (
            Some("invalid_response".to_string()),
            Some(truncate(&text, 500)),
        );
    }
    let field = endpoint.required_field();
    if payload.get(field).is_none() {
        return (
            Some("invalid_response".to_string()),
            Some(format!(
                "successful {} response did not contain {field:?}: {}",
                endpoint.as_str(),
                truncate(&text, 300)
            )),
        );
    }
    (None, None)
}

/// Check liveness and model discovery, then return the selected model.
pub async fn preflight(
    client: &Client,
    base_url: &str,
    requested_model: Option<&str>,
) -> Result<String, String> {
    let health = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let health_text = health.text().await.map_err(|error| error.to_string())?;
    let health_body: Value = serde_json::from_str(&health_text).map_err(|_| {
        format!(
            "GET /health did not return JSON: {}",
            truncate(&health_text, 300)
        )
    })?;
    if health_body.get("status").and_then(Value::as_str) != Some("ok") {
        return Err(format!(
            "GET /health returned an unexpected body: {}",
            truncate(&health_text, 300)
        ));
    }

    let response = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?;
    let text = response.text().await.map_err(|error| error.to_string())?;
    let body: Value = serde_json::from_str(&text).map_err(|_| {
        format!(
            "GET /v1/models did not return JSON: {}",
            truncate(&text, 300)
        )
    })?;
    let entries = body.get("data").and_then(Value::as_array).ok_or_else(|| {
        format!(
            "GET /v1/models returned an unexpected body: {}",
            truncate(&text, 300)
        )
    })?;
    let model_ids: Vec<String> = entries
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    let default_model = body
        .get("default_model")
        .and_then(Value::as_str)
        .filter(|id| model_ids.iter().any(|known| known == id));

    let model = requested_model
        .map(str::to_string)
        .or_else(|| default_model.map(str::to_string))
        .or_else(|| model_ids.first().cloned())
        .ok_or_else(|| "GET /v1/models returned no model to test".to_string())?;
    if let Some(requested) = requested_model {
        if !model_ids.iter().any(|known| known == requested) {
            return Err(format!(
                "model {requested:?} is not listed by GET /v1/models"
            ));
        }
    }
    Ok(model)
}

/// Read liveness and cumulative server metrics; never errors, so one bad read is one failed sample.
pub async fn read_server_state(client: &Client, base_url: &str) -> (bool, BTreeMap<String, f64>) {
    let healthy = match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status() == reqwest::StatusCode::OK => response
            .text()
            .await
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|body| {
                body.get("status")
                    .and_then(Value::as_str)
                    .map(|s| s == "ok")
            })
            .unwrap_or(false),
        _ => false,
    };
    let metrics = match client.get(format!("{base_url}/metrics")).send().await {
        Ok(response) if response.status().is_success() => response
            .text()
            .await
            .map(|text| parse_metrics(&text))
            .unwrap_or_default(),
        _ => BTreeMap::new(),
    };
    (healthy, metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bodies_match_public_endpoints() {
        let chat = request_body(Endpoint::Chat, "route", "hello", 8, true);
        let messages = request_body(Endpoint::Messages, "route", "hello", 8, true);
        let responses = request_body(Endpoint::Responses, "route", "hello", 8, true);

        assert_eq!(chat["messages"][0]["content"], json!("hello"));
        assert_eq!(chat["max_tokens"], json!(8));
        assert_eq!(messages["max_tokens"], json!(8));
        assert_eq!(responses["input"], json!("hello"));
        assert_eq!(responses["max_output_tokens"], json!(8));
        for body in [&chat, &messages, &responses] {
            assert_eq!(body["stream"], json!(true));
        }
    }

    #[test]
    fn parse_metrics_reads_counter_names_and_ignores_labelled_series() {
        let metrics = parse_metrics(
            "# TYPE switchyard_total_requests gauge\n\
             switchyard_total_requests 42\n\
             switchyard_total_errors{} 3\n\
             switchyard_requests_total{model=\"route\"} 10\n",
        );

        assert_eq!(metrics.get("switchyard_total_requests"), Some(&42.0));
        assert_eq!(metrics.get("switchyard_total_errors"), Some(&3.0));
        assert_eq!(metrics.get("switchyard_requests_total"), None);
    }
}

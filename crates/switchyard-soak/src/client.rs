// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP against the public Switchyard APIs: preflight, one request, and server-state reads.

use futures_util::StreamExt;
use reqwest::Client;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Value, json};

const SERVER_REQUESTS_METRIC: &str = "switchyard_total_requests";
const SERVER_ERRORS_METRIC: &str = "switchyard_total_errors";

/// One public Switchyard API the soak test exercises.
#[derive(Clone, Copy, PartialEq, Eq)]
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

fn transport_error_kind(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else if error.is_decode() {
        "request_error"
    } else if error.is_connect() || error.is_request() || error.is_body() {
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

#[derive(Default)]
struct Metrics {
    requests: Option<f64>,
    errors: Option<f64>,
}

fn parse_metrics(text: &str) -> Metrics {
    let mut metrics = Metrics::default();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name_part, rest)) = line.split_once(' ') else {
            continue;
        };
        let name = name_part.split('{').next().unwrap_or(name_part);
        if let Some(token) = rest.split_whitespace().next()
            && let Ok(value) = token.parse::<f64>()
        {
            match name {
                SERVER_REQUESTS_METRIC => metrics.requests = Some(value),
                SERVER_ERRORS_METRIC => metrics.errors = Some(value),
                _ => {}
            }
        }
    }
    metrics
}

#[derive(Debug)]
pub struct RequestError {
    pub kind: String,
    pub detail: String,
}

impl RequestError {
    fn new(kind: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            detail: detail.into(),
        }
    }

    fn transport(error: &reqwest::Error) -> Self {
        Self::new(
            transport_error_kind(error),
            truncate(&error.to_string(), 500),
        )
    }
}

struct StreamValidator {
    endpoint: Endpoint,
    event_name: Option<String>,
    saw_json: bool,
    saw_terminal: bool,
}

impl StreamValidator {
    fn new(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            event_name: None,
            saw_json: false,
            saw_terminal: false,
        }
    }

    fn read_line(&mut self, line: &[u8]) -> Result<(), RequestError> {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            self.event_name = None;
            return Ok(());
        }
        let line = std::str::from_utf8(line)
            .map_err(|error| RequestError::new("invalid_stream", error.to_string()))?;
        if let Some(event_name) = line.strip_prefix("event:") {
            self.event_name = Some(event_name.trim_start().to_string());
            return Ok(());
        }
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(());
        };
        let data = data.strip_prefix(' ').unwrap_or(data);
        if self.saw_terminal {
            return Err(RequestError::new(
                "invalid_stream",
                "stream contained data after its terminal event",
            ));
        }
        if data == "[DONE]" {
            if self.endpoint != Endpoint::Chat {
                return Err(RequestError::new(
                    "invalid_stream",
                    format!(
                        "{} stream contained an OpenAI [DONE] marker",
                        self.endpoint.as_str()
                    ),
                ));
            }
            self.saw_terminal = true;
            return Ok(());
        }

        let payload: Value = serde_json::from_str(data).map_err(|error| {
            RequestError::new("invalid_stream", format!("invalid SSE JSON: {error}"))
        })?;
        let event_type = payload.get("type").and_then(Value::as_str);
        if self.event_name.as_deref() == Some("error")
            || event_type == Some("error")
            || payload.get("error").is_some()
        {
            let detail = payload
                .pointer("/error/message")
                .or_else(|| payload.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("stream returned an error event");
            return Err(RequestError::new("stream_error", truncate(detail, 500)));
        }

        match self.endpoint {
            Endpoint::Chat => {}
            Endpoint::Messages | Endpoint::Responses => {
                let event_type = event_type.ok_or_else(|| {
                    RequestError::new("invalid_stream", "SSE data did not contain an event type")
                })?;
                if self.event_name.as_deref() != Some(event_type) {
                    return Err(RequestError::new(
                        "invalid_stream",
                        format!(
                            "SSE event name {:?} did not match data type {event_type:?}",
                            self.event_name
                        ),
                    ));
                }
                self.saw_terminal = matches!(
                    (self.endpoint, event_type),
                    (Endpoint::Messages, "message_stop")
                        | (
                            Endpoint::Responses,
                            "response.completed" | "response.incomplete"
                        )
                );
            }
        }
        self.saw_json = true;
        Ok(())
    }

    fn finish(self) -> Result<(), RequestError> {
        if !self.saw_json {
            return Err(RequestError::new(
                "empty_stream",
                "successful streaming response contained no JSON events",
            ));
        }
        if !self.saw_terminal {
            return Err(RequestError::new(
                "invalid_stream",
                format!(
                    "{} stream ended without its terminal event",
                    self.endpoint.as_str()
                ),
            ));
        }
        Ok(())
    }
}

/// Send one request and validate its response body or event stream.
pub async fn send_request(
    client: &Client,
    base_url: &str,
    endpoint: Endpoint,
    body: &Value,
) -> Result<(), RequestError> {
    let url = format!("{base_url}{}", endpoint.path());
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let response = match client.post(&url).json(body).send().await {
        Ok(response) => response,
        Err(error) => return Err(RequestError::transport(&error)),
    };

    if stream {
        let status = response.status();
        if !status.is_success() {
            let content = response.text().await.unwrap_or_default();
            return Err(RequestError::new(
                format!("http_{}", status.as_u16()),
                truncate(&content, 500),
            ));
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        if !content_type.contains("text/event-stream") {
            let content = response.text().await.unwrap_or_default();
            return Err(RequestError::new(
                "invalid_stream",
                format!(
                    "expected text/event-stream, received {content_type:?}: {}",
                    truncate(&content, 300)
                ),
            ));
        }
        let mut bytes = response.bytes_stream();
        let mut pending = Vec::new();
        let mut validator = StreamValidator::new(endpoint);
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(chunk) => {
                    pending.extend_from_slice(&chunk);
                    while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                        let mut line = pending.drain(..=newline).collect::<Vec<_>>();
                        line.pop();
                        validator.read_line(&line)?;
                    }
                }
                Err(error) => {
                    return Err(RequestError::transport(&error));
                }
            }
        }
        if !pending.is_empty() {
            validator.read_line(&pending)?;
        }
        return validator.finish();
    }

    let status = response.status();
    if !status.is_success() {
        let content = response.text().await.unwrap_or_default();
        return Err(RequestError::new(
            format!("http_{}", status.as_u16()),
            truncate(&content, 500),
        ));
    }
    let text = match response.text().await {
        Ok(text) => text,
        Err(error) => return Err(RequestError::transport(&error)),
    };
    let payload: Value = match serde_json::from_str(&text) {
        Ok(payload) => payload,
        Err(error) => return Err(RequestError::new("invalid_json", error.to_string())),
    };
    if !payload.is_object() || payload.get("error").is_some() {
        return Err(RequestError::new("invalid_response", truncate(&text, 500)));
    }
    let field = endpoint.required_field();
    if payload.get(field).is_none() {
        return Err(RequestError::new(
            "invalid_response",
            format!(
                "successful {} response did not contain {field:?}: {}",
                endpoint.as_str(),
                truncate(&text, 300)
            ),
        ));
    }
    Ok(())
}

/// Check liveness and verify that the requested model is advertised.
pub async fn preflight(client: &Client, base_url: &str, model: &str) -> Result<(), String> {
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
    if !entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(model))
    {
        return Err(format!("model {model:?} is not listed by GET /v1/models"));
    }
    Ok(())
}

pub struct ServerState {
    pub healthy: bool,
    pub requests: Option<f64>,
    pub errors: Option<f64>,
}

/// Read liveness and cumulative metrics; one bad read becomes one failed sample.
pub async fn read_server_state(client: &Client, base_url: &str) -> ServerState {
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
        _ => Metrics::default(),
    };
    ServerState {
        healthy,
        requests: metrics.requests,
        errors: metrics.errors,
    }
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
    fn parse_metrics_reads_only_required_counters() {
        let metrics = parse_metrics(
            "# TYPE switchyard_total_requests gauge\n\
             switchyard_total_requests 42\n\
             switchyard_total_errors{} 3\n\
             switchyard_requests_total{model=\"route\"} 10\n",
        );

        assert_eq!(metrics.requests, Some(42.0));
        assert_eq!(metrics.errors, Some(3.0));
    }

    #[test]
    fn stream_validator_accepts_each_public_terminal_event() -> Result<(), RequestError> {
        for (endpoint, stream) in [
            (Endpoint::Chat, "data: {\"choices\":[]}\n\ndata: [DONE]\n\n"),
            (
                Endpoint::Messages,
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            ),
            (
                Endpoint::Responses,
                "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
            ),
        ] {
            let mut validator = StreamValidator::new(endpoint);
            for line in stream.split('\n') {
                validator.read_line(line.as_bytes())?;
            }
            validator.finish()?;
        }
        Ok(())
    }

    #[test]
    fn stream_validator_rejects_errors_and_missing_terminal_events() {
        let mut error = StreamValidator::new(Endpoint::Chat);
        let result = error.read_line(b"data: {\"error\":{\"message\":\"boom\"}}");
        assert_eq!(result.unwrap_err().kind, "stream_error");

        let mut incomplete = StreamValidator::new(Endpoint::Responses);
        incomplete
            .read_line(b"event: response.output_text.delta")
            .expect("event name should parse");
        incomplete
            .read_line(b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}")
            .expect("data should parse");
        assert_eq!(incomplete.finish().unwrap_err().kind, "invalid_stream");
    }
}

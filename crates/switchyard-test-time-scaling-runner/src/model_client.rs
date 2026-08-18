// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small OpenAI-compatible chat client.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio::sync::Semaphore;
use tokio::time::sleep;

use crate::config::ModelConfig;

/// One lossless chat response and its text content.
pub struct ModelReply {
    /// Message content used by the method.
    pub content: String,
    /// Complete provider response.
    pub raw_response: String,
}

/// Bounded client shared by summary and comparison roles.
pub struct ModelClient {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model_id: String,
    attempts: usize,
    send_seed: bool,
    limit: Arc<Semaphore>,
    call_log_path: PathBuf,
    call_log_lock: Arc<Mutex<()>>,
}

impl ModelClient {
    /// Builds a client without logging the API key.
    pub fn new(
        model_id: String,
        config: &ModelConfig,
        call_log_path: PathBuf,
    ) -> Result<Self, String> {
        let api_key = std::env::var(&config.api_key_env)
            .map_err(|_| format!("{} is not set", config.api_key_env))?;
        if api_key.trim().is_empty() {
            return Err(format!("{} is empty", config.api_key_env));
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .build()
            .map_err(|error| format!("could not build model client: {error}"))?;
        Ok(Self {
            client,
            endpoint: format!("{}/chat/completions", config.base_url.trim_end_matches('/')),
            api_key,
            model_id,
            attempts: config.http_attempts,
            send_seed: config.send_seed,
            limit: Arc::new(Semaphore::new(config.max_concurrency)),
            call_log_path,
            call_log_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Sends one user message.
    pub async fn complete(
        &self,
        role: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<ModelReply, String> {
        let _permit = self
            .limit
            .acquire()
            .await
            .map_err(|_| "model concurrency limit closed".to_string())?;
        let mut body = json!({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        if self.send_seed {
            body["seed"] = json!(seed);
        }

        for attempt in 1..=self.attempts {
            let response = self
                .client
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;
            match response {
                Ok(response) => {
                    let status = response.status();
                    let raw = response
                        .text()
                        .await
                        .map_err(|error| format!("could not read model response: {error}"))?;
                    self.record_attempt(role, attempt, &body, Some(status), Some(&raw), None)
                        .await?;
                    if status.is_success() {
                        return parse_reply(raw, &self.model_id);
                    }
                    if !retryable_status(status) || attempt == self.attempts {
                        return Err(format!("model returned {status}: {}", bounded(&raw, 2_000)));
                    }
                }
                Err(error) => {
                    self.record_attempt(role, attempt, &body, None, None, Some(&error.to_string()))
                        .await?;
                    if attempt == self.attempts {
                        return Err(format!("model request failed: {error}"));
                    }
                }
            }
            sleep(Duration::from_secs(attempt.min(5) as u64)).await;
        }
        Err("model request exhausted attempts".to_string())
    }

    async fn record_attempt(
        &self,
        role: &str,
        attempt: usize,
        request: &Value,
        status: Option<StatusCode>,
        response: Option<&str>,
        error: Option<&str>,
    ) -> Result<(), String> {
        let record = json!({
            "role": role,
            "attempt": attempt,
            "request": request,
            "status": status.map(|value| value.as_u16()),
            "response": response,
            "error": error,
        });
        let mut bytes = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        let _guard = self.call_log_lock.lock().await;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.call_log_path)
            .await
            .map_err(|error| format!("could not open model call log: {error}"))?;
        file.write_all(&bytes)
            .await
            .map_err(|error| format!("could not write model call log: {error}"))
    }
}

fn retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn parse_reply(raw_response: String, expected_model_id: &str) -> Result<ModelReply, String> {
    let value: Value = serde_json::from_str(&raw_response)
        .map_err(|error| format!("model returned invalid JSON: {error}"))?;
    let response_model_id = value
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "model response has no model ID".to_string())?;
    if response_model_id != expected_model_id {
        return Err(format!(
            "model response used {response_model_id}; expected {expected_model_id}"
        ));
    }
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| "model response has no text content".to_string())?
        .to_string();
    Ok(ModelReply {
        content,
        raw_response,
    })
}

fn bounded(text: &str, limit: usize) -> &str {
    &text[..text.floor_char_boundary(limit.min(text.len()))]
}

#[cfg(test)]
mod tests {
    use super::{bounded, parse_reply};

    #[test]
    fn parses_chat_content_and_bounds_utf8() {
        let reply = parse_reply(
            r#"{"model":"model-1","choices":[{"message":{"content":"done"}}]}"#.to_string(),
            "model-1",
        )
        .expect("chat response");
        assert_eq!(reply.content, "done");
        assert_eq!(bounded("aéz", 2), "a");
        assert!(
            parse_reply(
                r#"{"model":"model-2","choices":[{"message":{"content":"done"}}]}"#.to_string(),
                "model-1",
            )
            .is_err()
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake payload sinks.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;
use switchyard_core::{Result, SwitchyardError};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::intake::config::{IntakeQueueFullPolicy, IntakeSinkConfig};

/// Async sink contract for intake payload delivery.
#[async_trait]
pub trait IntakeSink: Send + Sync {
    /// Enqueues or delivers one intake payload.
    async fn enqueue(&self, payload: Value) -> Result<()>;

    /// Flushes background resources before shutdown.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// HTTP-backed intake sink with an internal async worker queue.
#[derive(Clone)]
pub struct HttpIntakeSink {
    inner: Arc<HttpIntakeSinkInner>,
}

/// Shared sink state so cloned processors send into the same queue.
struct HttpIntakeSinkInner {
    config: IntakeSinkConfig,
    client: reqwest::Client,
    sender: mpsc::Sender<WorkerMessage>,
    receiver: Mutex<Option<mpsc::Receiver<WorkerMessage>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
    closed: AtomicBool,
    dropped: AtomicU64,
}

/// Messages consumed by the intake background worker.
enum WorkerMessage {
    /// Payload to POST to the intake service.
    Payload(Value),
    /// Graceful shutdown request.
    Shutdown,
}

impl HttpIntakeSink {
    /// Creates an HTTP intake sink and validates queue/client configuration.
    pub fn new(config: IntakeSinkConfig) -> Result<Self> {
        validate_config(&config)?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(config.request_timeout_s))
            .build()
            .map_err(|error| {
                SwitchyardError::InvalidConfig(format!(
                    "failed to build intake HTTP client: {error}"
                ))
            })?;
        let (sender, receiver) = mpsc::channel(config.max_queue_size);
        Ok(Self {
            inner: Arc::new(HttpIntakeSinkInner {
                config,
                client,
                sender,
                receiver: Mutex::new(Some(receiver)),
                handle: Mutex::new(None),
                closed: AtomicBool::new(false),
                dropped: AtomicU64::new(0),
            }),
        })
    }

    /// Returns the sink configuration.
    pub fn config(&self) -> &IntakeSinkConfig {
        &self.inner.config
    }

    /// Starts the worker lazily so unused intake configs do not spawn tasks.
    fn ensure_worker(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(SwitchyardError::InvalidConfig(
                "HttpIntakeSink is closed".to_string(),
            ));
        }

        let mut handle = self.inner.handle.lock();
        if handle.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return Ok(());
        }
        if handle.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(SwitchyardError::Other(
                "intake worker stopped unexpectedly".to_string(),
            ));
        }

        let receiver = self.inner.receiver.lock().take().ok_or_else(|| {
            SwitchyardError::Other("intake worker receiver is unavailable".to_string())
        })?;
        let config = self.inner.config.clone();
        let client = self.inner.client.clone();
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            SwitchyardError::InvalidConfig(format!(
                "HttpIntakeSink enqueue requires a Tokio runtime: {error}"
            ))
        })?;
        *handle = Some(runtime.spawn(worker_loop(config, client, receiver)));
        Ok(())
    }
}

impl fmt::Debug for HttpIntakeSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpIntakeSink")
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl IntakeSink for HttpIntakeSink {
    async fn enqueue(&self, payload: Value) -> Result<()> {
        self.ensure_worker()?;
        match self.inner.config.on_queue_full {
            IntakeQueueFullPolicy::Block => self
                .inner
                .sender
                .send(WorkerMessage::Payload(payload))
                .await
                .map_err(|_| SwitchyardError::Other("intake worker is closed".to_string())),
            IntakeQueueFullPolicy::Drop => {
                match self.inner.sender.try_send(WorkerMessage::Payload(payload)) {
                    Ok(()) => Ok(()),
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        let dropped = self.inner.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                        tracing::warn!(
                            max_queue_size = self.inner.config.max_queue_size,
                            dropped,
                            "intake queue full; dropping payload"
                        );
                        Ok(())
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => Err(SwitchyardError::Other(
                        "intake worker is closed".to_string(),
                    )),
                }
            }
        }
    }

    async fn shutdown(&self) -> Result<()> {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        let handle = self.inner.handle.lock().take();
        if handle.is_none() {
            self.inner.receiver.lock().take();
            return Ok(());
        }

        let _ignored = self.inner.sender.send(WorkerMessage::Shutdown).await;
        if let Some(handle) = handle {
            handle.await.map_err(|error| {
                SwitchyardError::Other(format!("intake worker join failed: {error}"))
            })?;
        }
        Ok(())
    }
}

async fn worker_loop(
    config: IntakeSinkConfig,
    client: reqwest::Client,
    mut receiver: mpsc::Receiver<WorkerMessage>,
) {
    // The worker is fail-open: individual POST failures are logged and dropped
    // so intake availability never blocks the LLM response path.
    while let Some(message) = receiver.recv().await {
        match message {
            WorkerMessage::Payload(payload) => {
                if let Err(error) = post_with_retries(&config, &client, &payload).await {
                    tracing::warn!(
                        error = %error,
                        "intake worker failed while posting payload; dropping"
                    );
                }
            }
            WorkerMessage::Shutdown => return,
        }
    }
}

/// Posts a payload, retrying transient failures according to config.
async fn post_with_retries(
    config: &IntakeSinkConfig,
    client: &reqwest::Client,
    payload: &Value,
) -> Result<()> {
    let mut attempt = 0;
    loop {
        match post_once(config, client, payload).await {
            Ok(()) => return Ok(()),
            Err(error) if attempt < config.max_retries => {
                attempt += 1;
                tracing::debug!(
                    error = %error,
                    attempt,
                    max_retries = config.max_retries,
                    "retrying intake payload POST"
                );
            }
            Err(error) => return Err(error),
        }
    }
}

/// Posts one payload to the intake API.
async fn post_once(
    config: &IntakeSinkConfig,
    client: &reqwest::Client,
    payload: &Value,
) -> Result<()> {
    // A configured target posts to its own URL with its own auth mode; otherwise
    // fall back to the authenticated nemo-platform chat-completions ingest.
    let (url, authenticated) = match &config.target {
        Some(target) => (target.url.clone(), target.authenticated),
        None => (chat_completions_ingest_url(config), true),
    };
    let mut request = client.post(url).json(payload);
    if authenticated {
        if let Some(api_key) = config.api_key.as_deref() {
            request = request.bearer_auth(api_key);
        }
    }
    let response = request.send().await.map_err(|error| {
        SwitchyardError::Upstream(format!("intake payload POST failed: {error}"))
    })?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let body = match response.text().await {
        Ok(body) => body,
        Err(error) => format!("<failed to read intake error body: {error}>"),
    };
    Err(SwitchyardError::Upstream(format!(
        "intake payload POST returned HTTP {status}: {body}"
    )))
}

/// Validates sink configuration before any background worker is started.
fn validate_config(config: &IntakeSinkConfig) -> Result<()> {
    // A configured target carries its own URL; only the nemo-platform fallback
    // needs intake_base_url to build the ingest URL.
    if config.target.is_none()
        && !matches!(config.intake_base_url.as_deref(), Some(base_url) if !base_url.is_empty())
    {
        return Err(SwitchyardError::InvalidConfig(
            "intake_base_url is required for HttpIntakeSink".to_string(),
        ));
    }
    if config.max_queue_size == 0 {
        return Err(SwitchyardError::InvalidConfig(
            "intake max_queue_size must be positive".to_string(),
        ));
    }
    if !config.request_timeout_s.is_finite() || config.request_timeout_s <= 0.0 {
        return Err(SwitchyardError::InvalidConfig(format!(
            "intake request_timeout_s must be finite and positive, got {:?}",
            config.request_timeout_s
        )));
    }
    Ok(())
}

/// Builds the chat-completions ingest URL from base URL and workspace.
fn chat_completions_ingest_url(config: &IntakeSinkConfig) -> String {
    let base_url = config
        .intake_base_url
        .as_deref()
        .unwrap_or_default()
        .trim_end_matches('/');
    format!(
        "{base_url}/apis/intake/v2/workspaces/{}/ingest/chat-completions",
        percent_encode_path_segment(config.workspace_or_default()),
    )
}

/// Percent-encodes one URL path segment without pulling in another dependency.
fn percent_encode_path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(nibble_to_hex(byte >> 4));
                encoded.push(nibble_to_hex(byte & 0x0f));
            }
        }
    }
    encoded
}

/// Converts a 4-bit value into uppercase hex.
fn nibble_to_hex(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        10..=15 => char::from(b'A' + (value - 10)),
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use switchyard_core::Result;

    use super::*;

    // Workspace names are path segments, so reserved characters must be encoded.
    #[test]
    fn percent_encodes_workspace_path_segment() -> Result<()> {
        assert_eq!(percent_encode_path_segment("team space"), "team%20space");
        assert_eq!(percent_encode_path_segment("a/b"), "a%2Fb");
        Ok(())
    }
}

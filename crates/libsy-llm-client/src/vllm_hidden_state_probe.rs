// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM hidden-state extraction for learned prefill routing.
//!
//! The probe calls an OpenAI-compatible chat endpoint configured with vLLM's
//! `ExampleHiddenStatesConnector`, waits for the returned safetensors artifact,
//! reduces `[prompt_tokens, layers, hidden_size]` to one token-mean vector per
//! layer, and removes the consumed artifact.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use thiserror::Error;

const ARTIFACT_WAIT_INTERVAL: Duration = Duration::from_millis(50);
const ARTIFACT_WAIT_TIMEOUT: Duration = Duration::from_secs(1);
const STALE_ARTIFACT_RETENTION: Duration = Duration::from_secs(5 * 60);

/// Token-mean hidden-state features returned by a vLLM probe.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenStateFeatures {
    /// Number of independently extracted hidden-state layers.
    pub layer_count: usize,
    /// Hidden width of each extracted layer.
    pub hidden_size: usize,
    /// Layer-major token-mean features.
    pub values: Vec<f32>,
}

/// Construction inputs for [`VllmHiddenStateProbe`].
#[derive(Clone, Debug)]
pub struct VllmHiddenStateProbeConfig {
    /// OpenAI-compatible vLLM base URL, normally ending in `/v1`.
    pub base_url: String,
    /// Probe model served by the vLLM endpoint.
    pub model: String,
    /// Dedicated shared directory used by `ExampleHiddenStatesConnector`.
    pub hidden_states_dir: PathBuf,
    /// Maximum duration of the HTTP request, including response-body decoding.
    pub request_timeout: Duration,
}

/// Failures returned while requesting or decoding vLLM hidden states.
#[derive(Debug, Error)]
pub enum VllmHiddenStateProbeError {
    /// The probe cannot be constructed from its configuration.
    #[error("invalid vLLM hidden-state probe configuration: {message}")]
    Configuration {
        /// Invalid field or invariant.
        message: String,
    },

    /// The vLLM HTTP request exceeded its configured timeout.
    #[error("vLLM hidden-state probe request timed out: {source}")]
    Timeout {
        /// Underlying HTTP timeout.
        #[source]
        source: reqwest::Error,
    },

    /// The vLLM HTTP request could not be completed.
    #[error("vLLM hidden-state probe transport failed: {source}")]
    Transport {
        /// Underlying HTTP transport failure.
        #[source]
        source: reqwest::Error,
    },

    /// The vLLM endpoint returned a non-success status.
    #[error("vLLM hidden-state probe returned HTTP {status}")]
    Http {
        /// Upstream HTTP status code.
        status: u16,
    },

    /// The vLLM response did not contain the expected connector payload.
    #[error("invalid vLLM hidden-state probe response: {message}")]
    Response {
        /// Response decoding or validation failure.
        message: String,
    },

    /// The returned hidden-state artifact was unsafe or malformed.
    #[error("vLLM hidden-state artifact error: {message}")]
    Artifact {
        /// Filesystem, safetensors, or feature validation failure.
        message: String,
    },

    /// Tokio could not complete a blocking filesystem task.
    #[error("vLLM hidden-state artifact task failed: {source}")]
    BlockingTask {
        /// Blocking task join failure.
        #[source]
        source: tokio::task::JoinError,
    },
}

/// Result type for vLLM hidden-state probe operations.
pub type VllmHiddenStateProbeResult<T> = Result<T, VllmHiddenStateProbeError>;

/// HTTP and filesystem client for vLLM prompt hidden-state extraction.
///
/// The configured hidden-state directory must be dedicated to probe artifacts.
/// Each extraction performs a bounded stale-file sweep, and a returned artifact
/// is removed after reading even when tensor parsing fails.
pub struct VllmHiddenStateProbe {
    completions_url: String,
    model: String,
    hidden_states_dir: PathBuf,
    client: reqwest::Client,
    artifact_wait_timeout: Duration,
    stale_artifact_retention: Duration,
}

impl VllmHiddenStateProbe {
    /// Validates the probe configuration and constructs its bounded HTTP client.
    pub fn new(config: VllmHiddenStateProbeConfig) -> VllmHiddenStateProbeResult<Self> {
        let base_url = config.base_url.trim();
        if base_url.is_empty() {
            return Err(configuration_error("base_url must not be empty"));
        }
        let model = config.model.trim();
        if model.is_empty() {
            return Err(configuration_error("model must not be empty"));
        }
        if config.request_timeout.is_zero() {
            return Err(configuration_error("request_timeout must be positive"));
        }

        let hidden_states_dir = config.hidden_states_dir.canonicalize().map_err(|error| {
            configuration_error(format!(
                "hidden_states_dir {} is not accessible: {error}",
                config.hidden_states_dir.display()
            ))
        })?;
        if !hidden_states_dir.is_dir() {
            return Err(configuration_error(format!(
                "hidden_states_dir {} is not a directory",
                hidden_states_dir.display()
            )));
        }
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .build()
            .map_err(map_reqwest_error)?;

        Ok(Self {
            completions_url: completions_url(base_url),
            model: model.to_string(),
            hidden_states_dir,
            client,
            artifact_wait_timeout: ARTIFACT_WAIT_TIMEOUT,
            stale_artifact_retention: STALE_ARTIFACT_RETENTION,
        })
    }

    /// Requests and token-mean pools hidden states for one task instruction.
    ///
    /// The task is sent only as the single user message. Filesystem parsing and
    /// cleanup run on Tokio's blocking pool. No task content is logged.
    pub async fn extract(&self, task: &str) -> VllmHiddenStateProbeResult<HiddenStateFeatures> {
        self.reap_stale_artifacts().await?;

        let response = self
            .client
            .post(&self.completions_url)
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": task}],
                "max_tokens": 1,
                "kv_transfer_params": {
                    "include_output_tokens": false,
                },
            }))
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = response.status();
        if !status.is_success() {
            return Err(VllmHiddenStateProbeError::Http {
                status: status.as_u16(),
            });
        }
        let response: CompletionResponse =
            response
                .json()
                .await
                .map_err(|error| match map_reqwest_error(error) {
                    VllmHiddenStateProbeError::Timeout { source } => {
                        VllmHiddenStateProbeError::Timeout { source }
                    }
                    other => VllmHiddenStateProbeError::Response {
                        message: other.to_string(),
                    },
                })?;
        let reported_path = response
            .kv_transfer_params
            .ok_or_else(|| response_error("missing kv_transfer_params"))?
            .hidden_states_path;
        let artifact_path = resolve_reported_path(&self.hidden_states_dir, &reported_path)?;

        let root = self.hidden_states_dir.clone();
        let artifact_wait_timeout = self.artifact_wait_timeout;
        tokio::task::spawn_blocking(move || {
            read_and_cleanup_hidden_states(&root, &artifact_path, artifact_wait_timeout)
        })
        .await
        .map_err(|source| VllmHiddenStateProbeError::BlockingTask { source })?
    }

    async fn reap_stale_artifacts(&self) -> VllmHiddenStateProbeResult<()> {
        let root = self.hidden_states_dir.clone();
        let retention = self.stale_artifact_retention;
        tokio::task::spawn_blocking(move || cleanup_stale_artifacts(&root, retention))
            .await
            .map_err(|source| VllmHiddenStateProbeError::BlockingTask { source })?
    }
}

impl fmt::Debug for VllmHiddenStateProbe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VllmHiddenStateProbe")
            .field("completions_url", &self.completions_url)
            .field("model", &self.model)
            .field("hidden_states_dir", &self.hidden_states_dir)
            .field("artifact_wait_timeout", &self.artifact_wait_timeout)
            .field("stale_artifact_retention", &self.stale_artifact_retention)
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct CompletionResponse {
    kv_transfer_params: Option<KvTransferParams>,
}

#[derive(Deserialize)]
struct KvTransferParams {
    hidden_states_path: String,
}

fn completions_url(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{base_url}/chat/completions")
    }
}

fn map_reqwest_error(error: reqwest::Error) -> VllmHiddenStateProbeError {
    if error.is_timeout() {
        VllmHiddenStateProbeError::Timeout { source: error }
    } else {
        VllmHiddenStateProbeError::Transport { source: error }
    }
}

fn configuration_error(message: impl Into<String>) -> VllmHiddenStateProbeError {
    VllmHiddenStateProbeError::Configuration {
        message: message.into(),
    }
}

fn response_error(message: impl Into<String>) -> VllmHiddenStateProbeError {
    VllmHiddenStateProbeError::Response {
        message: message.into(),
    }
}

fn artifact_error(message: impl Into<String>) -> VllmHiddenStateProbeError {
    VllmHiddenStateProbeError::Artifact {
        message: message.into(),
    }
}

fn resolve_reported_path(root: &Path, reported: &str) -> VllmHiddenStateProbeResult<PathBuf> {
    if reported.trim().is_empty() {
        return Err(response_error("hidden_states_path must not be empty"));
    }
    let reported = Path::new(reported);
    if !has_safetensors_extension(reported) {
        return Err(artifact_error(format!(
            "hidden-state artifact must be a .safetensors file: {}",
            reported.display()
        )));
    }
    let candidate = if reported.is_absolute() {
        reported.to_path_buf()
    } else {
        root.join(reported)
    };
    let parent = candidate
        .parent()
        .ok_or_else(|| artifact_error("hidden-state artifact has no parent directory"))?;
    let canonical_parent = parent.canonicalize().map_err(|error| {
        artifact_error(format!(
            "hidden-state artifact parent {} is not accessible: {error}",
            parent.display()
        ))
    })?;
    if !canonical_parent.starts_with(root) {
        return Err(artifact_error(format!(
            "hidden-state artifact parent {} is outside configured directory {}",
            canonical_parent.display(),
            root.display()
        )));
    }
    let file_name = candidate
        .file_name()
        .ok_or_else(|| artifact_error("hidden-state artifact has no file name"))?;
    Ok(canonical_parent.join(file_name))
}

fn validate_hidden_states_path(root: &Path, path: &Path) -> VllmHiddenStateProbeResult<PathBuf> {
    if !has_safetensors_extension(path) {
        return Err(artifact_error(format!(
            "hidden-state artifact must be a .safetensors file: {}",
            path.display()
        )));
    }
    let actual = path.canonicalize().map_err(|error| {
        artifact_error(format!(
            "hidden-state artifact {} is not accessible: {error}",
            path.display()
        ))
    })?;
    if !actual.starts_with(root) {
        return Err(artifact_error(format!(
            "hidden-state artifact {} is outside configured directory {}",
            actual.display(),
            root.display()
        )));
    }
    if !has_safetensors_extension(&actual) {
        return Err(artifact_error(format!(
            "canonical hidden-state artifact must be a .safetensors file: {}",
            actual.display()
        )));
    }
    let metadata = actual.metadata().map_err(|error| {
        artifact_error(format!(
            "hidden-state artifact metadata error for {}: {error}",
            actual.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(artifact_error(format!(
            "hidden-state artifact is not a regular file: {}",
            actual.display()
        )));
    }
    Ok(actual)
}

fn has_safetensors_extension(path: &Path) -> bool {
    path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
}

fn companion_lock_path(path: &Path) -> PathBuf {
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

fn open_synchronized_artifact(
    path: &Path,
    timeout: Duration,
) -> VllmHiddenStateProbeResult<(File, File)> {
    let lock_path = companion_lock_path(path);
    let deadline = Instant::now() + timeout;
    loop {
        match OpenOptions::new().read(true).open(&lock_path) {
            Ok(lock_file) => {
                let metadata = lock_file.metadata().map_err(|error| {
                    artifact_error(format!(
                        "hidden-state synchronization lock metadata error for {}: {error}",
                        lock_path.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(artifact_error(format!(
                        "hidden-state synchronization lock is not a regular file: {}",
                        lock_path.display()
                    )));
                }
                match lock_file.try_lock_shared() {
                    Ok(()) => {
                        let artifact = File::open(path).map_err(|error| {
                            artifact_error(format!(
                                "hidden-state artifact open error for {} after writer completed: \
                                 {error}",
                                path.display()
                            ))
                        })?;
                        return Ok((artifact, lock_file));
                    }
                    Err(std::fs::TryLockError::WouldBlock) => {}
                    Err(error) => {
                        return Err(artifact_error(format!(
                            "hidden-state synchronization lock error for {}: {error}",
                            lock_path.display()
                        )));
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(artifact_error(format!(
                    "hidden-state synchronization lock open error for {}: {error}",
                    lock_path.display()
                )));
            }
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(artifact_error(format!(
                "hidden-state artifact {} did not become readable within {} ms; ensure vLLM \
                 use_synchronization_lock is enabled",
                path.display(),
                timeout.as_millis()
            )));
        }
        std::thread::sleep(ARTIFACT_WAIT_INTERVAL.min(deadline - now));
    }
}

fn read_and_cleanup_hidden_states(
    root: &Path,
    path: &Path,
    timeout: Duration,
) -> VllmHiddenStateProbeResult<HiddenStateFeatures> {
    let (mut artifact, synchronization_lock) = open_synchronized_artifact(path, timeout)?;
    let artifact_path = validate_hidden_states_path(root, path)?;
    let mut bytes = Vec::new();
    let features = artifact
        .read_to_end(&mut bytes)
        .map_err(|error| {
            artifact_error(format!(
                "hidden-state artifact read error for {}: {error}",
                artifact_path.display()
            ))
        })
        .and_then(|_| parse_hidden_state_features(&bytes));
    drop(artifact);
    drop(synchronization_lock);
    let cleanup = cleanup_artifact_files(&artifact_path);
    match (features, cleanup) {
        (Ok(features), Ok(())) => Ok(features),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(feature_error), Ok(())) => Err(feature_error),
        (Err(feature_error), Err(cleanup_error)) => Err(artifact_error(format!(
            "{feature_error}; cleanup also failed: {cleanup_error}"
        ))),
    }
}

fn cleanup_artifact_files(path: &Path) -> VllmHiddenStateProbeResult<()> {
    let lock_path = companion_lock_path(path);
    std::fs::remove_file(path).map_err(|error| {
        artifact_error(format!(
            "hidden-state artifact cleanup error for {}: {error}",
            path.display()
        ))
    })?;
    match std::fs::remove_file(&lock_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(artifact_error(format!(
            "hidden-state synchronization lock cleanup error for {}: {error}",
            lock_path.display()
        ))),
    }
}

fn cleanup_stale_artifacts(root: &Path, retention: Duration) -> VllmHiddenStateProbeResult<()> {
    let entries = std::fs::read_dir(root).map_err(|error| {
        artifact_error(format!(
            "failed to scan hidden-state directory {}: {error}",
            root.display()
        ))
    })?;
    let now = SystemTime::now();
    for entry in entries {
        let entry = entry.map_err(|error| {
            artifact_error(format!(
                "failed to inspect hidden-state directory {}: {error}",
                root.display()
            ))
        })?;
        let path = entry.path();
        if !has_safetensors_extension(&path) {
            continue;
        }
        let file_type = entry.file_type().map_err(|error| {
            artifact_error(format!(
                "failed to inspect hidden-state artifact {}: {error}",
                path.display()
            ))
        })?;
        if !file_type.is_file() {
            continue;
        }
        let metadata = entry.metadata().map_err(|error| {
            artifact_error(format!(
                "failed to inspect hidden-state artifact {}: {error}",
                path.display()
            ))
        })?;
        let modified = metadata.modified().map_err(|error| {
            artifact_error(format!(
                "failed to read hidden-state artifact timestamp {}: {error}",
                path.display()
            ))
        })?;
        if now.duration_since(modified).unwrap_or_default() < retention {
            continue;
        }
        let lock_path = companion_lock_path(&path);
        let cleanup_lock = match OpenOptions::new().read(true).open(&lock_path) {
            Ok(lock_file) => match lock_file.try_lock_shared() {
                Ok(()) => lock_file,
                Err(std::fs::TryLockError::WouldBlock) => continue,
                Err(error) => {
                    return Err(artifact_error(format!(
                        "failed to lock stale hidden-state synchronization file {}: {error}",
                        lock_path.display()
                    )));
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let artifact = match OpenOptions::new().read(true).write(true).open(&path) {
                    Ok(file) => file,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => {
                        return Err(artifact_error(format!(
                            "failed to open stale hidden-state artifact {}: {error}",
                            path.display()
                        )));
                    }
                };
                match artifact.try_lock() {
                    Ok(()) => artifact,
                    Err(std::fs::TryLockError::WouldBlock) => continue,
                    Err(error) => {
                        return Err(artifact_error(format!(
                            "failed to lock stale hidden-state artifact {}: {error}",
                            path.display()
                        )));
                    }
                }
            }
            Err(error) => {
                return Err(artifact_error(format!(
                    "failed to open stale hidden-state synchronization file {}: {error}",
                    lock_path.display()
                )));
            }
        };
        std::fs::remove_file(&path).map_err(|error| {
            artifact_error(format!(
                "failed to remove stale hidden-state artifact {}: {error}",
                path.display()
            ))
        })?;
        drop(cleanup_lock);
        match std::fs::remove_file(&lock_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(artifact_error(format!(
                    "failed to remove stale hidden-state synchronization file {}: {error}",
                    lock_path.display()
                )));
            }
        }
    }
    Ok(())
}

fn parse_hidden_state_features(bytes: &[u8]) -> VllmHiddenStateProbeResult<HiddenStateFeatures> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| artifact_error(format!("safetensors parse error: {error}")))?;
    let hidden_states = tensors
        .tensor("hidden_states")
        .map_err(|error| artifact_error(format!("hidden_states tensor not found: {error}")))?;
    let prompt_tokens = match hidden_states.shape() {
        [prompt_tokens, _, _] => *prompt_tokens,
        _ => {
            return Err(artifact_error(
                "expected hidden_states shape [prompt_tokens, layers, hidden_size]",
            ));
        }
    };
    validate_token_ids(&tensors, prompt_tokens)?;
    token_mean_per_layer(
        hidden_states.data(),
        hidden_states.dtype(),
        hidden_states.shape(),
    )
}

fn token_mean_per_layer(
    data: &[u8],
    dtype: Dtype,
    shape: &[usize],
) -> VllmHiddenStateProbeResult<HiddenStateFeatures> {
    if shape.len() != 3 {
        return Err(artifact_error(
            "expected hidden_states shape [prompt_tokens, layers, hidden_size]",
        ));
    }
    let (prompt_tokens, layer_count, hidden_size) = (shape[0], shape[1], shape[2]);
    if prompt_tokens == 0 {
        return Err(artifact_error(
            "hidden_states token dimension must be non-zero",
        ));
    }
    if layer_count == 0 || hidden_size == 0 {
        return Err(artifact_error(
            "hidden_states layer and hidden dimensions must be non-zero",
        ));
    }

    let (element_size, decode): (usize, fn(&[u8]) -> f32) = match dtype {
        Dtype::F32 => (size_of::<f32>(), decode_f32),
        Dtype::BF16 => (size_of::<u16>(), decode_bf16),
        other => {
            return Err(artifact_error(format!(
                "unsupported hidden_states dtype: {other:?}"
            )));
        }
    };
    let features_per_token = layer_count
        .checked_mul(hidden_size)
        .ok_or_else(|| artifact_error("hidden_states shape is too large"))?;
    let bytes_per_token = features_per_token
        .checked_mul(element_size)
        .ok_or_else(|| artifact_error("hidden_states byte length is too large"))?;
    let expected_bytes = prompt_tokens
        .checked_mul(bytes_per_token)
        .ok_or_else(|| artifact_error("hidden_states byte length is too large"))?;
    if data.len() != expected_bytes {
        return Err(artifact_error(format!(
            "hidden_states byte length {} does not match shape byte length {expected_bytes}",
            data.len()
        )));
    }

    let mut pooled = vec![0.0f32; features_per_token];
    for token in data.chunks_exact(bytes_per_token) {
        for (index, bytes) in token.chunks_exact(element_size).enumerate() {
            accumulate(&mut pooled[index], decode(bytes))?;
        }
    }
    let token_count = prompt_tokens as f32;
    for value in &mut pooled {
        *value /= token_count;
        if !value.is_finite() {
            return Err(artifact_error(
                "hidden-state token mean produced a non-finite value",
            ));
        }
    }
    Ok(HiddenStateFeatures {
        layer_count,
        hidden_size,
        values: pooled,
    })
}

fn decode_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn decode_bf16(bytes: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
}

fn accumulate(sum: &mut f32, value: f32) -> VllmHiddenStateProbeResult<()> {
    if !value.is_finite() {
        return Err(artifact_error("hidden_states contains non-finite values"));
    }
    *sum += value;
    if !sum.is_finite() {
        return Err(artifact_error(
            "hidden-state token accumulation produced a non-finite value",
        ));
    }
    Ok(())
}

fn validate_token_ids(
    tensors: &SafeTensors<'_>,
    prompt_tokens: usize,
) -> VllmHiddenStateProbeResult<()> {
    if !tensors
        .names()
        .iter()
        .any(|name| name.as_str() == "token_ids")
    {
        return Ok(());
    }
    let token_ids = tensors
        .tensor("token_ids")
        .map_err(|error| artifact_error(format!("token_ids tensor error: {error}")))?;
    if token_ids.dtype() != Dtype::I64 {
        return Err(artifact_error(format!(
            "token_ids must use I64; got {:?}",
            token_ids.dtype()
        )));
    }
    if token_ids.shape() != [prompt_tokens] {
        return Err(artifact_error(format!(
            "token_ids shape {:?} does not match hidden_states token count {prompt_tokens}",
            token_ids.shape()
        )));
    }
    for bytes in token_ids.data().chunks_exact(size_of::<i64>()) {
        let token_id = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        if token_id < 0 {
            return Err(artifact_error("token_ids contains a negative token ID"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::error::Error;

    use safetensors::tensor::{serialize, TensorView};
    use serde_json::json;
    use tempfile::TempDir;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn serialize_hidden_states(
        dtype: Dtype,
        shape: Vec<usize>,
        hidden_data: &[u8],
        token_ids: Option<(Dtype, Vec<usize>, Vec<u8>)>,
    ) -> TestResult<Vec<u8>> {
        let hidden_view = TensorView::new(dtype, shape, hidden_data)?;
        let serialized = if let Some((dtype, shape, data)) = token_ids.as_ref() {
            let token_view = TensorView::new(*dtype, shape.clone(), data)?;
            serialize(
                BTreeMap::from([("hidden_states", hidden_view), ("token_ids", token_view)]),
                &None,
            )
        } else {
            serialize(BTreeMap::from([("hidden_states", hidden_view)]), &None)
        };
        Ok(serialized?)
    }

    fn valid_artifact_bytes() -> TestResult<Vec<u8>> {
        serialize_hidden_states(
            Dtype::F32,
            vec![2, 2, 2],
            &f32_bytes(&[
                1.0, 3.0, 5.0, 7.0, // token 0, layers 0 and 1
                2.0, 4.0, 6.0, 8.0, // token 1, layers 0 and 1
            ]),
            None,
        )
    }

    fn config(server: &MockServer, directory: &TempDir) -> VllmHiddenStateProbeConfig {
        VllmHiddenStateProbeConfig {
            base_url: format!("{}/v1", server.uri()),
            model: "probe/model".into(),
            hidden_states_dir: directory.path().to_path_buf(),
            request_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn token_mean_pooling_supports_f32_and_bf16() -> TestResult {
        let values = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let f32_features = token_mean_per_layer(&f32_bytes(&values), Dtype::F32, &[2, 2, 2])?;
        let bf16_features = token_mean_per_layer(&bf16_bytes(&values), Dtype::BF16, &[2, 2, 2])?;

        assert_eq!(f32_features.layer_count, 2);
        assert_eq!(f32_features.hidden_size, 2);
        assert_eq!(f32_features.values, vec![1.5, 3.5, 5.5, 7.5]);
        assert_eq!(bf16_features, f32_features);
        Ok(())
    }

    #[test]
    fn token_mean_pooling_rejects_invalid_layout_dtype_and_values() -> TestResult {
        let shape_error = token_mean_per_layer(&[], Dtype::F32, &[1, 2])
            .err()
            .ok_or_else(|| artifact_error("invalid shape should fail"))?;
        assert!(shape_error
            .to_string()
            .contains("expected hidden_states shape"));

        let dtype_error = token_mean_per_layer(&i64_bytes(&[1]), Dtype::I64, &[1, 1, 1])
            .err()
            .ok_or_else(|| artifact_error("invalid dtype should fail"))?;
        assert!(dtype_error.to_string().contains("unsupported"));

        let value_error = token_mean_per_layer(&f32_bytes(&[f32::NAN]), Dtype::F32, &[1, 1, 1])
            .err()
            .ok_or_else(|| artifact_error("non-finite value should fail"))?;
        assert!(value_error.to_string().contains("non-finite"));
        Ok(())
    }

    #[test]
    fn optional_token_ids_are_validated() -> TestResult {
        let valid = serialize_hidden_states(
            Dtype::F32,
            vec![2, 1, 2],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            Some((Dtype::I64, vec![2], i64_bytes(&[101, 102]))),
        )?;
        assert_eq!(parse_hidden_state_features(&valid)?.values, vec![2.0, 3.0]);

        let negative = serialize_hidden_states(
            Dtype::F32,
            vec![2, 1, 2],
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            Some((Dtype::I64, vec![2], i64_bytes(&[101, -1]))),
        )?;
        let error = parse_hidden_state_features(&negative)
            .err()
            .ok_or_else(|| artifact_error("negative token ID should fail"))?;
        assert!(error.to_string().contains("negative token ID"));
        Ok(())
    }

    #[tokio::test]
    async fn extract_sends_exact_contract_and_removes_artifact() -> TestResult {
        let server = MockServer::start().await;
        let directory = TempDir::new()?;
        let artifact_path = directory.path().join("hidden.safetensors");
        std::fs::write(&artifact_path, valid_artifact_bytes()?)?;
        std::fs::write(companion_lock_path(&artifact_path), b"")?;
        let expected_body = json!({
            "model": "probe/model",
            "messages": [{"role": "user", "content": "Explain the failure."}],
            "max_tokens": 1,
            "kv_transfer_params": {
                "include_output_tokens": false,
            },
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_json(expected_body))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "kv_transfer_params": {
                    "hidden_states_path": artifact_path,
                },
            })))
            .mount(&server)
            .await;
        let probe = VllmHiddenStateProbe::new(config(&server, &directory))?;

        let features = probe.extract("Explain the failure.").await?;

        assert_eq!(features.values, vec![1.5, 3.5, 5.5, 7.5]);
        assert!(!artifact_path.exists());
        assert!(!companion_lock_path(&artifact_path).exists());
        Ok(())
    }

    #[tokio::test]
    async fn malformed_artifact_is_removed() -> TestResult {
        let server = MockServer::start().await;
        let directory = TempDir::new()?;
        let artifact_path = directory.path().join("malformed.safetensors");
        std::fs::write(&artifact_path, b"not safetensors")?;
        std::fs::write(companion_lock_path(&artifact_path), b"")?;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "kv_transfer_params": {
                    "hidden_states_path": artifact_path,
                },
            })))
            .mount(&server)
            .await;
        let probe = VllmHiddenStateProbe::new(config(&server, &directory))?;

        let error = probe
            .extract("task")
            .await
            .err()
            .ok_or_else(|| artifact_error("malformed artifact should fail"))?;

        assert!(error.to_string().contains("safetensors parse error"));
        assert!(!artifact_path.exists());
        assert!(!companion_lock_path(&artifact_path).exists());
        Ok(())
    }

    #[tokio::test]
    async fn path_outside_configured_directory_is_rejected_without_cleanup() -> TestResult {
        let server = MockServer::start().await;
        let directory = TempDir::new()?;
        let outside = TempDir::new()?;
        let artifact_path = outside.path().join("outside.safetensors");
        std::fs::write(&artifact_path, valid_artifact_bytes()?)?;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "kv_transfer_params": {
                    "hidden_states_path": artifact_path,
                },
            })))
            .mount(&server)
            .await;
        let probe = VllmHiddenStateProbe::new(config(&server, &directory))?;

        let error = probe
            .extract("task")
            .await
            .err()
            .ok_or_else(|| artifact_error("outside path should fail"))?;

        assert!(error.to_string().contains("outside configured directory"));
        assert!(artifact_path.exists());
        Ok(())
    }

    #[tokio::test]
    async fn response_and_http_failures_are_typed() -> TestResult {
        let missing_server = MockServer::start().await;
        let directory = TempDir::new()?;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&missing_server)
            .await;
        let probe = VllmHiddenStateProbe::new(config(&missing_server, &directory))?;
        assert!(matches!(
            probe.extract("task").await,
            Err(VllmHiddenStateProbeError::Response { .. })
        ));

        let unavailable_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&unavailable_server)
            .await;
        let probe = VllmHiddenStateProbe::new(config(&unavailable_server, &directory))?;
        assert!(matches!(
            probe.extract("task").await,
            Err(VllmHiddenStateProbeError::Http { status: 503 })
        ));
        Ok(())
    }

    #[tokio::test]
    async fn request_timeout_is_enforced() -> TestResult {
        let server = MockServer::start().await;
        let directory = TempDir::new()?;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(100)))
            .mount(&server)
            .await;
        let mut config = config(&server, &directory);
        config.request_timeout = Duration::from_millis(10);
        let probe = VllmHiddenStateProbe::new(config)?;

        assert!(matches!(
            probe.extract("task").await,
            Err(VllmHiddenStateProbeError::Timeout { .. })
        ));
        Ok(())
    }

    #[test]
    fn stale_reaper_removes_only_unlocked_safetensors_files() -> TestResult {
        let directory = TempDir::new()?;
        let stale = directory.path().join("stale.safetensors");
        let locked = directory.path().join("locked.safetensors");
        let locked_path = companion_lock_path(&locked);
        let unrelated = directory.path().join("keep.txt");
        std::fs::write(&stale, b"stale")?;
        std::fs::write(&locked, b"locked")?;
        std::fs::write(&locked_path, b"")?;
        std::fs::write(&unrelated, b"keep")?;
        let locked_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&locked_path)?;
        locked_file.lock()?;

        cleanup_stale_artifacts(directory.path(), Duration::ZERO)?;

        assert!(!stale.exists());
        assert!(locked.exists());
        assert!(locked_path.exists());
        assert!(unrelated.exists());
        drop(locked_file);
        cleanup_stale_artifacts(directory.path(), Duration::ZERO)?;
        assert!(!locked.exists());
        assert!(!locked_path.exists());
        Ok(())
    }

    #[test]
    fn artifact_reader_waits_for_connector_lock_and_cleans_both_files() -> TestResult {
        let directory = TempDir::new()?;
        let root = directory.path().canonicalize()?;
        let artifact_path = directory.path().join("hidden.safetensors");
        let lock_path = companion_lock_path(&artifact_path);
        std::fs::write(&artifact_path, valid_artifact_bytes()?)?;
        std::fs::write(&lock_path, b"")?;
        let writer_lock = OpenOptions::new().read(true).write(true).open(&lock_path)?;
        writer_lock.lock()?;
        let reader_path = artifact_path.clone();
        let reader = std::thread::spawn(move || {
            read_and_cleanup_hidden_states(&root, &reader_path, Duration::from_secs(1))
        });

        std::thread::sleep(Duration::from_millis(20));
        assert!(!reader.is_finished());
        drop(writer_lock);
        let features = reader
            .join()
            .map_err(|_| artifact_error("artifact reader test thread panicked"))??;

        assert_eq!(features.values, vec![1.5, 3.5, 5.5, 7.5]);
        assert!(!artifact_path.exists());
        assert!(!lock_path.exists());
        Ok(())
    }

    #[test]
    fn configuration_rejects_invalid_values() -> TestResult {
        let directory = TempDir::new()?;
        let valid = VllmHiddenStateProbeConfig {
            base_url: "https://example.test/v1".into(),
            model: "probe/model".into(),
            hidden_states_dir: directory.path().to_path_buf(),
            request_timeout: Duration::from_secs(1),
        };
        let mut invalid = valid.clone();
        invalid.model = " ".into();
        assert!(matches!(
            VllmHiddenStateProbe::new(invalid),
            Err(VllmHiddenStateProbeError::Configuration { .. })
        ));

        let mut invalid = valid;
        invalid.request_timeout = Duration::ZERO;
        assert!(matches!(
            VllmHiddenStateProbe::new(invalid),
            Err(VllmHiddenStateProbeError::Configuration { .. })
        ));
        Ok(())
    }
}

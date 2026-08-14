// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal vLLM-Omni Cosmos image client for model-as-a-tool demos.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use switchyard_protocol::{
    LlmClientError, LlmResponse, Request, Response, RoutedLlmClient, prompt_text, text_response,
};

use crate::Result;

const IMAGE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Connection and artifact settings for [`CosmosMediaClient`].
#[derive(Clone, Debug)]
pub struct CosmosMediaConfig {
    /// vLLM-Omni base URL, normally `http://127.0.0.1:8000/v1`.
    pub base_url: String,
    /// Optional bearer token.
    pub api_key: Option<String>,
    /// Additional static request headers.
    pub extra_headers: BTreeMap<String, String>,
    /// Directory where generated PNG artifacts are written.
    pub output_dir: PathBuf,
}

/// Adapts the Cosmos image endpoint to the routed model-client contract.
///
/// Input is a one-message text request. Output is a text response listing the generated local
/// paths, keeping the endpoint-specific image protocol outside libsy algorithms.
pub struct CosmosMediaClient {
    base_url: String,
    api_key: Option<String>,
    extra_headers: HeaderMap,
    output_dir: PathBuf,
    client: reqwest::Client,
    sequence: AtomicU64,
}

impl CosmosMediaClient {
    /// Builds a client without contacting the server or creating the artifact directory.
    pub fn new(config: CosmosMediaConfig) -> Result<Self> {
        let base_url = config.base_url.trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(LlmClientError::Configuration {
                message: "Cosmos media base_url must not be empty".to_string(),
            });
        }
        if config.output_dir.as_os_str().is_empty() {
            return Err(LlmClientError::Configuration {
                message: "Cosmos media output_dir must not be empty".to_string(),
            });
        }
        let extra_headers = header_map(&config.extra_headers)?;
        let client = reqwest::Client::builder()
            .timeout(IMAGE_REQUEST_TIMEOUT)
            .build()
            .map_err(transport_error)?;
        Ok(Self {
            base_url,
            api_key: config.api_key,
            extra_headers,
            output_dir: config.output_dir,
            client,
            sequence: AtomicU64::new(0),
        })
    }

    async fn generate(&self, request: Request) -> Result<Response> {
        let model =
            request
                .llm_request
                .model
                .clone()
                .ok_or_else(|| LlmClientError::InvalidRequest {
                    message: "Cosmos media request has no model".to_string(),
                })?;
        let prompt = prompt_text(&request.llm_request);
        if prompt.trim().is_empty() {
            return Err(LlmClientError::InvalidRequest {
                message: "Cosmos media request requires a non-empty user prompt".to_string(),
            });
        }
        tokio::fs::create_dir_all(&self.output_dir)
            .await
            .map_err(|error| artifact_error(&self.output_dir, error))?;

        let image = self.generate_image(&model, &prompt).await?;
        if image.is_empty() {
            return Err(LlmClientError::ResponseTranslation(
                "Cosmos returned an empty image artifact".to_string(),
            ));
        }

        let stem = self.next_stem();
        let image_path = self.output_dir.join(format!("{stem}.png"));
        write_new(&image_path, &image).await?;

        let completion = format!(
            "Generated an image with {model}:\n- Image: `{}`",
            image_path.display()
        );
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(Some(model), completion)),
            metadata: request.metadata,
        })
    }

    async fn generate_image(&self, model: &str, prompt: &str) -> Result<Vec<u8>> {
        let response = self
            .authorized(
                self.client
                    .post(format!("{}/images/generations", self.base_url)),
            )
            .json(&json!({
                "model": model,
                "prompt": prompt,
                "negative_prompt": "blurry, distorted, low quality",
                "size": "1024x1024",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "seed": 42
            }))
            .send()
            .await
            .map_err(transport_error)?;
        let body = successful_body(response).await?;
        let payload: Value =
            serde_json::from_slice(&body).map_err(|error| LlmClientError::InvalidResponse {
                source: Box::new(error),
            })?;
        let encoded = payload
            .get("data")
            .and_then(Value::as_array)
            .and_then(|data| data.first())
            .and_then(|item| item.get("b64_json"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                LlmClientError::ResponseTranslation(
                    "Cosmos image response has no data[0].b64_json".to_string(),
                )
            })?;
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| LlmClientError::InvalidResponse {
                source: Box::new(error),
            })
    }

    fn authorized(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }
        builder.headers(self.extra_headers.clone())
    }

    fn next_stem(&self) -> String {
        let epoch_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        format!("cosmos-{epoch_millis}-{}-{sequence}", std::process::id())
    }
}

#[async_trait]
impl RoutedLlmClient for CosmosMediaClient {
    async fn call(&self, request: Request) -> Result<Response> {
        self.generate(request).await
    }
}

fn header_map(headers: &BTreeMap<String, String>) -> Result<HeaderMap> {
    let mut result = HeaderMap::new();
    for (name, value) in headers {
        if ["authorization", "content-length", "content-type", "host"]
            .iter()
            .any(|reserved| name.eq_ignore_ascii_case(reserved))
        {
            return Err(LlmClientError::Configuration {
                message: format!("Cosmos media extra_headers cannot set reserved header {name:?}"),
            });
        }
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            LlmClientError::Configuration {
                message: format!("invalid Cosmos media header name {name:?}: {error}"),
            }
        })?;
        let value =
            HeaderValue::from_str(value).map_err(|error| LlmClientError::Configuration {
                message: format!("invalid Cosmos media header value: {error}"),
            })?;
        result.append(name, value);
    }
    Ok(result)
}

async fn successful_body(response: reqwest::Response) -> Result<Vec<u8>> {
    let status = response.status();
    let body = response.bytes().await.map_err(transport_error)?.to_vec();
    if status.is_success() {
        return Ok(body);
    }
    Err(LlmClientError::UpstreamHttp {
        status: status.as_u16(),
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn transport_error(error: reqwest::Error) -> LlmClientError {
    if error.is_timeout() {
        LlmClientError::Timeout {
            source: Box::new(error),
        }
    } else {
        LlmClientError::Transport {
            source: Box::new(error),
        }
    }
}

fn artifact_error(path: &std::path::Path, error: std::io::Error) -> LlmClientError {
    LlmClientError::General(format!(
        "failed to write generated media artifact {}: {error}",
        path.display()
    ))
}

async fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await
        .map_err(|error| artifact_error(path, error))?;
    file.write_all(bytes)
        .await
        .map_err(|error| artifact_error(path, error))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use switchyard_protocol::{
        LlmResponse, Request, RoutedLlmClient, completion_text, text_request,
    };
    use tempfile::tempdir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::{CosmosMediaClient, CosmosMediaConfig};

    #[tokio::test]
    async fn generates_image_artifact_and_returns_path()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"b64_json": "cG5n"}]
            })))
            .mount(&server)
            .await;
        let output = tempdir()?;
        let client = CosmosMediaClient::new(CosmosMediaConfig {
            base_url: format!("{}/v1/", server.uri()),
            api_key: None,
            extra_headers: BTreeMap::new(),
            output_dir: output.path().join("media"),
        })?;

        let response = client
            .call(Request {
                llm_request: text_request(Some("nvidia/Cosmos3-Nano".to_string()), "A tiny robot"),
                raw_request: None,
                metadata: None,
            })
            .await?;
        let LlmResponse::Agg(response) = response.llm_response else {
            panic!("media response must be buffered");
        };
        let completion = completion_text(&response);
        assert!(completion.contains(".png"));
        let mut files = std::fs::read_dir(output.path().join("media"))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        files.sort_by_key(std::fs::DirEntry::file_name);
        assert_eq!(files.len(), 1);
        assert_eq!(std::fs::read(files[0].path())?, b"png");
        Ok(())
    }
}

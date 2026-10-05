// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product telemetry payloads, logging, and background HTTP delivery.

use std::{
    str::FromStr,
    sync::{Arc, LazyLock, OnceLock},
    time::{Duration, SystemTime},
};

use reqwest::Client;
use serde_json::Value;
use switchyard_protocol::ModelId;

use crate::OutcomeMetadata;

const DEFAULT_TELEMETRY_ENDPOINT: &str =
    "https://events.telemetry.data.nvidia.com/v1.1/events/json";
const LOG_ENDPOINT_DENTINEL: &str = "log";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_REDIRECTS: usize = 10;

static TELEMETRY: LazyLock<Telemetry> =
    LazyLock::new(|| Telemetry::from_env(|name| std::env::var(name).ok()));

enum Destination {
    Disabled,
    Log,
    Http(String),
}

struct Telemetry {
    destination: Destination,
    timeout: Duration,
    max_redirects: usize,
    session_id: uuid::Uuid,
    client: Arc<OnceLock<reqwest::Result<Client>>>,
}

impl Telemetry {
    fn from_env(read: impl Fn(&str) -> Option<String>) -> Self {
        let disabled = read("SWITCHYARD_TELEMETRY_ENABLED").is_some_and(|value| {
            ["0", "false", "no", "off"]
                .iter()
                .any(|disabled| value.trim().eq_ignore_ascii_case(disabled))
        });
        let destination = if disabled {
            Destination::Disabled
        } else {
            match read("SWITCHYARD_TELEMETRY_ENDPOINT") {
                Some(endpoint) if endpoint == LOG_ENDPOINT_DENTINEL => Destination::Log,
                endpoint => Destination::Http(
                    endpoint.unwrap_or_else(|| DEFAULT_TELEMETRY_ENDPOINT.to_string()),
                ),
            }
        };
        Self {
            destination,
            timeout: Duration::from_secs(parse_setting(
                &read,
                "SWITCHYARD_TELEMETRY_TIMEOUT_SECS",
                DEFAULT_TIMEOUT_SECS,
                |value| *value > 0,
            )),
            max_redirects: parse_setting(
                &read,
                "SWITCHYARD_TELEMETRY_MAX_REDIRECTS",
                DEFAULT_MAX_REDIRECTS,
                |_| true,
            ),
            // One session per process; each routing decision has its own outcome ID.
            session_id: uuid::Uuid::now_v7(),
            client: Arc::new(OnceLock::new()),
        }
    }

    fn emit(&self, metadata: &OutcomeMetadata, selected_models: &[ModelId]) {
        let endpoint = match &self.destination {
            Destination::Disabled => return,
            Destination::Log => {
                let payload = expand_metadata(metadata, selected_models, self.session_id);
                tracing::info!(payload = %payload, "Switchyard product telemetry");
                return;
            }
            Destination::Http(endpoint) => endpoint.clone(),
        };
        let runtime = match tokio::runtime::Handle::try_current() {
            Ok(runtime) => runtime,
            Err(error) => {
                tracing::debug!(%error, "could not spawn product telemetry POST");
                return;
            }
        };
        let metadata = metadata.clone();
        let selected_models = selected_models.to_vec();
        let session_id = self.session_id;
        let client = Arc::clone(&self.client);
        let timeout = self.timeout;
        let max_redirects = self.max_redirects;
        runtime.spawn(async move {
            let client = match client.get_or_init(|| {
                Client::builder()
                    .timeout(timeout)
                    .redirect(reqwest::redirect::Policy::limited(max_redirects))
                    .retry(reqwest::retry::never())
                    .build()
            }) {
                Ok(client) => client,
                Err(error) => {
                    tracing::debug!(%error, "could not build product telemetry HTTP client");
                    return;
                }
            };
            let payload = expand_metadata(&metadata, &selected_models, session_id);
            if let Err(error) = client
                .post(endpoint)
                .json(&payload)
                .send()
                .await
                .and_then(|response| response.error_for_status())
            {
                tracing::debug!(%error, is_timeout = error.is_timeout(), "product telemetry POST failed");
            }
        });
    }
}

fn parse_setting<T: FromStr>(
    read: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: T,
    valid: impl Fn(&T) -> bool,
) -> T {
    let Some(value) = read(name) else {
        return default;
    };
    match value.trim().parse().ok().filter(valid) {
        Some(value) => value,
        None => {
            tracing::debug!(name, "invalid product telemetry setting; using default");
            default
        }
    }
}

/// Logs immediately or schedules an HTTP POST without waiting for delivery.
pub(crate) fn emit(metadata: &OutcomeMetadata, selected_models: &[ModelId]) {
    TELEMETRY.emit(metadata, selected_models);
}

fn expand_metadata(
    metadata: &OutcomeMetadata,
    selected_models: &[ModelId],
    session_id: uuid::Uuid,
) -> Value {
    let timestamp = humantime::format_rfc3339_millis(SystemTime::now()).to_string();
    let mut parameters = serde_json::json!(metadata);
    parameters["nemoSource"] = serde_json::json!("switchyard");
    parameters["selected_model_id"] = serde_json::json!(selected_models.first());
    parameters["fallback_plan_model_ids"] =
        serde_json::json!(selected_models.get(1..).unwrap_or_default());

    serde_json::json!({
        "browserType": "undefined",
        "clientId": "184482118588404",  // NeMo Telemetry client ID
        "clientType": "Native",
        "clientVariant": "Release",
        "clientVer": env!("CARGO_PKG_VERSION"),
        "cpuArchitecture": std::env::consts::ARCH,
        "deviceGdprBehOptIn": "None",
        "deviceGdprFuncOptIn": "None",
        "deviceGdprTechOptIn": "None",
        "deviceId": "undefined",
        "deviceMake": "undefined",
        "deviceModel": "undefined",
        "deviceOS": std::env::consts::OS,
        "deviceOSVersion": "undefined",
        "deviceType": "undefined",
        "eventProtocol": "1.6",
        "eventSchemaVer": "1.12",
        "eventSysVer": "switchyard-telemetry/1.0",
        "externalUserId": "undefined",
        "gdprBehOptIn": "None",
        "gdprFuncOptIn": "None",
        "gdprTechOptIn": "None",
        "idpId": "undefined",
        "integrationId": "undefined",
        "productName": "undefined",
        "productVersion": "undefined",
        "sentTs": timestamp,
        "sessionId": session_id.to_string(),
        "userId": "undefined",
        "events": [{
            "ts": timestamp,
            "parameters": parameters,
            "name": "switchyard_outcome",
        }],
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use parking_lot::Mutex;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::oneshot,
    };

    use super::*;

    fn configured(values: &[(&str, &str)]) -> Telemetry {
        Telemetry::from_env(|name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    #[test]
    fn settings_use_defaults_and_accept_overrides() {
        let telemetry = configured(&[]);
        assert!(
            matches!(telemetry.destination, Destination::Http(ref url) if url == DEFAULT_TELEMETRY_ENDPOINT)
        );
        assert_eq!(telemetry.timeout, Duration::from_secs(30));
        assert_eq!(telemetry.max_redirects, 10);

        let telemetry = configured(&[
            ("SWITCHYARD_TELEMETRY_ENDPOINT", "http://localhost/events"),
            ("SWITCHYARD_TELEMETRY_TIMEOUT_SECS", " 2 "),
            ("SWITCHYARD_TELEMETRY_MAX_REDIRECTS", "0"),
        ]);
        assert!(
            matches!(telemetry.destination, Destination::Http(ref url) if url == "http://localhost/events")
        );
        assert_eq!(telemetry.timeout, Duration::from_secs(2));
        assert_eq!(telemetry.max_redirects, 0);
        assert!(matches!(
            configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", "log")]).destination,
            Destination::Log
        ));
    }

    #[test]
    fn invalid_settings_fall_back_to_defaults() {
        for invalid in ["", "-1", "bad", "1.5", "18446744073709551616"] {
            let telemetry = configured(&[
                ("SWITCHYARD_TELEMETRY_TIMEOUT_SECS", invalid),
                ("SWITCHYARD_TELEMETRY_MAX_REDIRECTS", invalid),
            ]);
            assert_eq!(telemetry.timeout, Duration::from_secs(30));
            assert_eq!(telemetry.max_redirects, 10);
        }
        assert_eq!(
            configured(&[("SWITCHYARD_TELEMETRY_TIMEOUT_SECS", "0")]).timeout,
            Duration::from_secs(30)
        );
    }

    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl LogBuffer {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().clone()).unwrap()
        }

        fn subscriber(&self) -> impl tracing::Subscriber + use<> {
            let buffer = self.clone();
            tracing_subscriber::fmt()
                .without_time()
                .with_ansi(false)
                .with_max_level(tracing::Level::DEBUG)
                .with_writer(move || buffer.clone())
                .finish()
        }
    }

    #[test]
    fn opt_out_does_no_logging_or_client_initialization() {
        for disabled in ["0", "false", "no", "off", " FALSE ", "No", "OFF"] {
            let telemetry = configured(&[
                ("SWITCHYARD_TELEMETRY_ENABLED", disabled),
                ("SWITCHYARD_TELEMETRY_ENDPOINT", "log"),
            ]);
            let logs = LogBuffer::default();
            tracing::subscriber::with_default(logs.subscriber(), || {
                telemetry.emit(&OutcomeMetadata::new("test".into(), None), &[]);
            });
            assert!(matches!(telemetry.destination, Destination::Disabled));
            assert!(telemetry.client.get().is_none());
            assert!(logs.text().is_empty());
        }
        for enabled in ["1", "true", "yes", "on", ""] {
            assert!(matches!(
                configured(&[("SWITCHYARD_TELEMETRY_ENABLED", enabled)]).destination,
                Destination::Http(_)
            ));
        }
    }

    #[test]
    fn log_endpoint_emits_complete_payload_at_info_level() {
        let telemetry = configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", "log")]);
        let metadata = OutcomeMetadata::new("test".into(), None);
        let selected = [ModelId::from("selected"), ModelId::from("fallback")];
        let logs = LogBuffer::default();
        tracing::subscriber::with_default(logs.subscriber(), || {
            telemetry.emit(&metadata, &selected);
        });
        let text = logs.text();
        assert!(text.contains("INFO"));
        let (_, payload) = text.split_once("payload=").unwrap();
        let record: Value = serde_json::from_str(payload.trim()).unwrap();
        assert_eq!(record["sessionId"], telemetry.session_id.to_string());
        assert_eq!(
            record["events"][0]["parameters"]["outcome_id"],
            metadata.outcome_id()
        );
        assert_eq!(
            record["events"][0]["parameters"]["fallback_plan_model_ids"],
            serde_json::json!(["fallback"])
        );
        assert!(telemetry.client.get().is_none());
    }

    #[test]
    fn http_without_runtime_is_skipped_with_debug_log() {
        let telemetry = configured(&[]);
        let logs = LogBuffer::default();
        tracing::subscriber::with_default(logs.subscriber(), || {
            telemetry.emit(&OutcomeMetadata::new("test".into(), None), &[]);
        });
        assert!(logs.text().contains("DEBUG"));
        assert!(logs.text().contains("could not spawn"));
        assert!(telemetry.client.get().is_none());
    }

    async fn read_request(stream: &mut TcpStream) -> (String, Value) {
        let mut bytes = Vec::new();
        let header_end = loop {
            let mut buffer = [0; 1024];
            let read = stream.read(&mut buffer).await.unwrap();
            assert_ne!(read, 0, "request ended before headers");
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
        let length: usize = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().unwrap())
            })
            .unwrap();
        while bytes.len() < header_end + length {
            let mut buffer = [0; 1024];
            let read = stream.read(&mut buffer).await.unwrap();
            assert_ne!(read, 0, "request ended before body");
            bytes.extend_from_slice(&buffer[..read]);
        }
        let body = serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
        (headers, body)
    }

    async fn wait_for_failure(logs: &LogBuffer) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !logs.text().contains("product telemetry POST failed") {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(logs.text().contains("DEBUG"));
    }

    #[tokio::test]
    async fn background_post_does_not_wait_for_response_and_reuses_client() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
        let telemetry = configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]);
        let first = OutcomeMetadata::new("test".into(), None);
        let second = OutcomeMetadata::new("test".into(), None);
        let (received_tx, received_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            received_tx.send(request).unwrap();
            release_rx.await.unwrap();
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            request
        });
        telemetry.emit(&first, &[ModelId::from("selected")]);
        let (headers, body) = tokio::time::timeout(Duration::from_secs(2), received_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(headers.starts_with("POST /events HTTP/1.1"));
        assert!(
            headers
                .to_lowercase()
                .contains("content-type: application/json")
        );
        assert_eq!(
            body["events"][0]["parameters"]["outcome_id"],
            first.outcome_id()
        );
        let client = telemetry.client.get().unwrap() as *const _;
        release_tx.send(()).unwrap();
        telemetry.emit(&second, &[]);
        let (_, other) = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(telemetry.client.get().unwrap() as *const _, client);
        assert_eq!(body["sessionId"], other["sessionId"]);
        assert_ne!(
            body["events"][0]["parameters"]["outcome_id"],
            other["events"][0]["parameters"]["outcome_id"]
        );
    }

    #[tokio::test]
    async fn redirect_limit_is_enforced_and_307_and_308_preserve_body() {
        for limit in [0, 1, 2] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
            let telemetry = configured(&[
                ("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint),
                ("SWITCHYARD_TELEMETRY_MAX_REDIRECTS", &limit.to_string()),
            ]);
            let logs = LogBuffer::default();
            let _subscriber = tracing::subscriber::set_default(logs.subscriber());
            let server = tokio::spawn(async move {
                let mut requests = Vec::new();
                for response in [
                    "HTTP/1.1 307 Temporary Redirect\r\nLocation: /second\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    "HTTP/1.1 308 Permanent Redirect\r\nLocation: /third\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n",
                ].into_iter().take(limit + 1) {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    requests.push(read_request(&mut stream).await);
                    stream.write_all(response.as_bytes()).await.unwrap();
                }
                assert!(
                    tokio::time::timeout(Duration::from_millis(100), listener.accept())
                        .await
                        .is_err()
                );
                requests
            });
            telemetry.emit(&OutcomeMetadata::new("test".into(), None), &[]);
            let requests = tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(requests.len(), limit + 1);
            for (headers, body) in &requests {
                assert!(headers.starts_with("POST "));
                assert_eq!(body, &requests[0].1);
            }
            if limit < 2 {
                wait_for_failure(&logs).await;
            }
        }
    }

    #[tokio::test]
    async fn http_errors_and_transport_failures_are_debug_logs_without_retries() {
        for response in [
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            "",
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
            let telemetry = configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]);
            let logs = LogBuffer::default();
            let _subscriber = tracing::subscriber::set_default(logs.subscriber());
            let server = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                read_request(&mut stream).await;
                stream.write_all(response.as_bytes()).await.unwrap();
                drop(stream);
                assert!(
                    tokio::time::timeout(Duration::from_millis(100), listener.accept())
                        .await
                        .is_err()
                );
            });
            telemetry.emit(&OutcomeMetadata::new("test".into(), None), &[]);
            wait_for_failure(&logs).await;
            tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test]
    async fn request_timeout_is_applied() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
        let mut telemetry = configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]);
        // Keep the test short while exercising the same client timeout setting.
        telemetry.timeout = Duration::from_millis(50);
        let logs = LogBuffer::default();
        let _subscriber = tracing::subscriber::set_default(logs.subscriber());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_request(&mut stream).await;
            let mut byte = [0];
            assert_eq!(stream.read(&mut byte).await.unwrap(), 0);
        });
        telemetry.emit(&OutcomeMetadata::new("test".into(), None), &[]);
        wait_for_failure(&logs).await;
        assert!(logs.text().contains("is_timeout=true"));
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }
    #[test]
    fn expanded_metadata_wraps_routing_fields_and_reuses_process_session() {
        let evidence = serde_json::json!({
            "source": "fail_open", "reason_code": "transport", "score": 0.5,
        });
        let metadata = OutcomeMetadata::new("test".to_string(), Some(evidence.clone()));
        let session_id = uuid::Uuid::now_v7();
        let selected_models = ["selected", "fallback-b", "fallback-a"].map(ModelId::from);
        let record = expand_metadata(&metadata, &selected_models, session_id);
        let parameters = &record["events"][0]["parameters"];
        assert_eq!(parameters["outcome_id"], metadata.outcome_id());
        assert_eq!(parameters["evidence"], evidence);
        assert_eq!(parameters["nemoSource"], "switchyard");
        assert_eq!(parameters["selected_model_id"], "selected");
        assert_eq!(
            parameters["fallback_plan_model_ids"],
            serde_json::json!(["fallback-b", "fallback-a"])
        );
        assert_eq!(record["clientVer"], env!("CARGO_PKG_VERSION"));
        assert_eq!(record["deviceOS"], std::env::consts::OS);
        assert_eq!(record["cpuArchitecture"], std::env::consts::ARCH);
        assert_eq!(record["eventSysVer"], "switchyard-telemetry/1.0");
        assert_eq!(record["events"][0]["name"], "switchyard_outcome");
        assert_eq!(record["sentTs"], record["events"][0]["ts"]);
        humantime::parse_rfc3339(record["sentTs"].as_str().unwrap()).unwrap();
        let session_id = uuid::Uuid::parse_str(record["sessionId"].as_str().unwrap()).unwrap();
        assert_eq!(session_id.get_version_num(), 7);
        let other = expand_metadata(
            &OutcomeMetadata::new("other".to_string(), None),
            &[],
            session_id,
        );
        assert_eq!(record["sessionId"], other["sessionId"]);
        for field in [
            "decision_source",
            "reason_codes",
            "selection_score",
            "fail_open",
        ] {
            assert!(
                parameters.get(field).is_none(),
                "duplicated evidence: {field}"
            );
        }
        let mut parameters = parameters.clone();
        for field in ["nemoSource", "selected_model_id", "fallback_plan_model_ids"] {
            parameters.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(parameters, serde_json::json!(metadata));
    }
}

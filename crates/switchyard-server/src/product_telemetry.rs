// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product telemetry payloads, logging, and background HTTP delivery.

use std::{
    str::FromStr,
    sync::{Arc, OnceLock},
    time::{Duration, Instant, SystemTime},
};

use parking_lot::{Condvar, Mutex};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::ModelId;
use tokio::sync::mpsc;
use tracing::instrument::WithSubscriber as _;
use tracing::{
    Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

#[cfg(test)]
use libsy::OutcomeMetadata;

const DEFAULT_TELEMETRY_ENDPOINT: &str =
    "https://events.telemetry.data.nvidia.com/v1.1/events/json";
const LOG_ENDPOINT_SENTINEL: &str = "log";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_REDIRECTS: usize = 10;

const QUEUE_CAPACITY: usize = 128;

static TELEMETRY: OnceLock<Arc<Telemetry>> = OnceLock::new();

pub(crate) fn layer() -> impl Layer<tracing_subscriber::Registry> {
    ProductTelemetryLayer {
        telemetry: TELEMETRY
            .get_or_init(|| Arc::new(Telemetry::from_env(|name| std::env::var(name).ok())))
            .clone(),
    }
    .filtered()
}

pub(crate) fn flush() {
    if let Some(telemetry) = TELEMETRY.get() {
        telemetry.flush();
    }
}

struct ProductTelemetryLayer {
    telemetry: Arc<Telemetry>,
}

impl ProductTelemetryLayer {
    fn filtered(self) -> impl Layer<tracing_subscriber::Registry> {
        let is_enabled = !matches!(self.telemetry.destination, Destination::Disabled);
        self.with_filter(tracing_subscriber::filter::filter_fn(move |metadata| {
            is_enabled
                && metadata.is_span()
                && metadata.target() == "libsy"
                && metadata.name() == "libsy.run"
        }))
    }
}

#[derive(Deserialize)]
struct OutcomeRecord {
    metadata: serde_json::Map<String, Value>,
    selected_model_ids: Vec<ModelId>,
}

#[derive(Default)]
struct RunOutcome(Option<OutcomeRecord>);

impl Visit for RunOutcome {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "switchyard.outcome" {
            match serde_json::from_str(value) {
                Ok(outcome) => self.0 = Some(outcome),
                Err(error) => tracing::debug!(%error, "invalid product telemetry outcome"),
            }
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

impl<S> Layer<S> for ProductTelemetryLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id) {
            let mut outcome = RunOutcome::default();
            attributes.record(&mut outcome);
            span.extensions_mut().insert(outcome);
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, context: Context<'_, S>) {
        if let Some(span) = context.span(id)
            && let Some(outcome) = span.extensions_mut().get_mut::<RunOutcome>()
        {
            values.record(outcome);
        }
    }

    fn on_close(&self, id: Id, context: Context<'_, S>) {
        let outcome = context
            .span(&id)
            .and_then(|span| span.extensions_mut().remove::<RunOutcome>())
            .and_then(|outcome| outcome.0);
        if let Some(outcome) = outcome {
            self.telemetry.emit(&outcome);
        }
    }
}

#[derive(Default)]
struct Pending {
    count: Mutex<usize>,
    idle: Condvar,
}

struct Worker {
    sender: mpsc::Sender<Value>,
    pending: Arc<Pending>,
}

impl Worker {
    fn new(
        endpoint: String,
        timeout: Duration,
        max_redirects: usize,
        client: Arc<OnceLock<reqwest::Result<Client>>>,
    ) -> Result<Self, String> {
        let (sender, mut receiver) = mpsc::channel::<Value>(QUEUE_CAPACITY);
        let pending = Arc::new(Pending::default());
        let worker_pending = pending.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        let dispatch = tracing::dispatcher::get_default(|dispatch| dispatch.downgrade());
        let delivery = async move {
            while let Some(payload) = receiver.recv().await {
                let dispatch = dispatch.upgrade().unwrap_or_default();
                let send = async {
                    let client = match client.get_or_init(|| {
                        Client::builder()
                            .timeout(timeout)
                            .redirect(reqwest::redirect::Policy::limited(max_redirects))
                            // Product telemetry is best effort; errors and timeouts are not retried.
                            .retry(reqwest::retry::never())
                            .build()
                    }) {
                        Ok(client) => client,
                        Err(error) => {
                            tracing::debug!(%error, "could not build product telemetry HTTP client");
                            return;
                        }
                    };
                    if let Err(error) = client
                        .post(&endpoint)
                        .json(&payload)
                        .send()
                        .await
                        .and_then(|response| response.error_for_status())
                    {
                        tracing::debug!(%error, is_timeout = error.is_timeout(), "product telemetry POST failed");
                    }
                };
                send.with_subscriber(dispatch).await;
                let mut count = worker_pending.count.lock();
                *count -= 1;
                if *count == 0 {
                    worker_pending.idle.notify_all();
                }
            }
        };
        std::thread::Builder::new()
            .name("switchyard-product-telemetry".into())
            .spawn(move || runtime.block_on(delivery))
            .map_err(|error| error.to_string())?;
        Ok(Self { sender, pending })
    }

    fn submit(&self, payload: Value) {
        let mut count = self.pending.count.lock();
        *count += 1;
        if let Err(error) = self.sender.try_send(payload) {
            *count -= 1;
            tracing::debug!(%error, "product telemetry queue rejected event");
        }
    }

    fn flush(&self, timeout: Duration) {
        let started = Instant::now();
        let mut count = self.pending.count.lock();
        while *count > 0 {
            if self
                .pending
                .idle
                .wait_for(&mut count, timeout.saturating_sub(started.elapsed()))
                .timed_out()
            {
                tracing::debug!(pending = *count, "product telemetry flush timed out");
                break;
            }
        }
    }
}

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
    worker: OnceLock<Result<Worker, String>>,
}

impl Telemetry {
    fn from_env(read: impl Fn(&str) -> Option<String>) -> Self {
        let disabled = read("SWITCHYARD_PRODUCT_TELEMETRY").is_some_and(|value| {
            ["0", "false", "no", "off"]
                .iter()
                .any(|disabled| value.trim().eq_ignore_ascii_case(disabled))
        });
        let destination = if disabled {
            Destination::Disabled
        } else {
            match read("SWITCHYARD_TELEMETRY_ENDPOINT") {
                Some(endpoint) if endpoint == LOG_ENDPOINT_SENTINEL => Destination::Log,
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
            worker: OnceLock::new(),
        }
    }

    fn emit(&self, outcome: &OutcomeRecord) {
        let endpoint = match &self.destination {
            Destination::Disabled => return,
            Destination::Log => {
                let payload = build_payload(outcome, self.session_id);
                tracing::info!(payload = %format_args!("{payload:#}"), "Switchyard product telemetry");
                return;
            }
            Destination::Http(endpoint) => endpoint,
        };
        let worker = match self.worker.get_or_init(|| {
            Worker::new(
                endpoint.clone(),
                self.timeout,
                self.max_redirects,
                self.client.clone(),
            )
        }) {
            Ok(worker) => worker,
            Err(error) => {
                tracing::debug!(%error, "could not start product telemetry worker");
                return;
            }
        };
        worker.submit(build_payload(outcome, self.session_id));
    }

    fn flush(&self) {
        if let Some(Ok(worker)) = self.worker.get() {
            worker.flush(self.timeout);
        }
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

fn build_payload(outcome: &OutcomeRecord, session_id: uuid::Uuid) -> Value {
    let timestamp = humantime::format_rfc3339_millis(SystemTime::now()).to_string();
    let mut parameters = Value::Object(outcome.metadata.clone());

    // TODO: Determine if we need to report routing failures and error codes.
    // If we do, this should be populated by libsy
    // if we do not, then we don't need these fields at all.
    // Either way these three lines should be removed prior to merging into main.
    parameters["routing_status"] = serde_json::json!("success");
    parameters["no_eligible_target"] = serde_json::json!(false);
    parameters["routing_error_code"] = Value::Null;
    let selected_models = &outcome.selected_model_ids;
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

    use futures_util::StreamExt;
    use libsy::{Algorithm, Driver, Noop, Passthrough, RoutingOutcome, RuntimeModels, Step};
    use parking_lot::Mutex;
    use switchyard_protocol::Request;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::oneshot,
    };
    use tracing_subscriber::layer::SubscriberExt as _;

    use super::*;

    fn outcome_record(metadata: &OutcomeMetadata, selected_models: &[ModelId]) -> OutcomeRecord {
        serde_json::from_value(serde_json::json!({
            "metadata": metadata, "selected_model_ids": selected_models,
        }))
        .unwrap()
    }

    impl Telemetry {
        fn emit_metadata(&self, metadata: &OutcomeMetadata, selected_models: &[ModelId]) {
            self.emit(&outcome_record(metadata, selected_models));
        }
    }

    fn configured(values: &[(&str, &str)]) -> Telemetry {
        Telemetry::from_env(|name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    struct NestedRouter;

    #[async_trait::async_trait]
    impl Algorithm for NestedRouter {
        fn name(&self) -> &str {
            "nested"
        }

        async fn route(
            self: Arc<Self>,
            _driver: Driver,
            request: Request,
        ) -> libsy::Result<RoutingOutcome> {
            let mut steps =
                Arc::new(Noop {}).run_stream(request, Arc::new(RuntimeModels::default()));
            while let Some(step) = steps.next().await {
                if let Step::Done(mut outcome) = step? {
                    let mut metadata = OutcomeMetadata::new(
                        self.name().into(),
                        Some(serde_json::json!({
                            "source": "test",
                            "custom": "patient name is Jane Doe",
                            "confidence": "wrong type",
                        })),
                    );
                    metadata.algorithm_version = Some("1".into());
                    metadata.feature_flags = Some([("test".into(), true)].into());
                    metadata.exclusion_reason_codes = Some(vec!["test_reason".into()]);
                    outcome.metadata = Some(metadata);
                    return Ok(*outcome);
                }
            }
            panic!("nested run produced no outcome");
        }
    }

    struct PendingRouter;

    #[async_trait::async_trait]
    impl Algorithm for PendingRouter {
        fn name(&self) -> &str {
            "pending"
        }

        async fn route(
            self: Arc<Self>,
            _driver: Driver,
            _request: Request,
        ) -> libsy::Result<RoutingOutcome> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn layer_collects_concurrent_nested_runs_without_otlp_or_logs() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
        let telemetry = Arc::new(configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]));
        let subscriber = tracing_subscriber::registry()
            .with(
                ProductTelemetryLayer {
                    telemetry: telemetry.clone(),
                }
                .filtered(),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_filter(tracing_subscriber::EnvFilter::new("off")),
            );
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let server = tokio::spawn(async move {
            let mut payloads = Vec::new();
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                payloads.push(read_request(&mut stream).await.1);
                stream
                    .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(100), listener.accept())
                    .await
                    .is_err()
            );
            payloads
        });
        let run = || async {
            let mut steps = Arc::new(NestedRouter)
                .run_stream(Request::default(), Arc::new(RuntimeModels::default()));
            match steps.next().await.unwrap().unwrap() {
                Step::Done(outcome) => outcome.metadata.unwrap(),
                Step::CallModel(_) => panic!("unexpected model call"),
                Step::CallDecision(_) => panic!("unexpected decision call"),
            }
        };
        let (first, second) = tokio::join!(run(), run());
        let mut failed = Arc::new(Passthrough)
            .run_stream(Request::default(), Arc::new(RuntimeModels::default()));
        assert!(failed.next().await.unwrap().is_err());
        let cancelled = Arc::new(PendingRouter)
            .run_stream(Request::default(), Arc::new(RuntimeModels::default()));
        tokio::task::yield_now().await;
        drop(cancelled);
        let payloads = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            payloads
                .iter()
                .filter(|payload| payload["events"][0]["parameters"]["algorithm"] == "noop")
                .count(),
            2
        );
        for metadata in [first, second] {
            let record = payloads
                .iter()
                .find(|payload| {
                    payload["events"][0]["parameters"]["outcome_id"] == metadata.outcome_id()
                })
                .unwrap();
            let mut parameters = record["events"][0]["parameters"]
                .as_object()
                .unwrap()
                .clone();
            assert_eq!(parameters.remove("nemoSource").unwrap(), "switchyard");
            assert_eq!(
                parameters.remove("selected_model_id").unwrap(),
                "switchyard/noop"
            );
            assert_eq!(
                parameters.remove("fallback_plan_model_ids").unwrap(),
                serde_json::json!([])
            );
            let mut expected = serde_json::json!(metadata);
            expected["evidence"] = serde_json::json!({"source": "test"});
            expected["routing_status"] = serde_json::json!("success");
            expected["no_eligible_target"] = serde_json::json!(false);
            expected["routing_error_code"] = Value::Null;
            assert_eq!(Value::Object(parameters), expected);
            assert!(!record.to_string().contains("patient name is Jane Doe"));
        }
    }

    #[test]
    fn full_queue_drops_new_events_and_flush_has_a_deadline() {
        let (sender, mut receiver) = mpsc::channel(1);
        let worker = Worker {
            sender,
            pending: Arc::new(Pending::default()),
        };
        worker.submit(serde_json::json!("first"));
        worker.submit(serde_json::json!("dropped"));
        assert_eq!(*worker.pending.count.lock(), 1);
        worker.flush(Duration::from_millis(1));
        assert_eq!(receiver.try_recv().unwrap(), serde_json::json!("first"));
        assert!(receiver.try_recv().is_err());
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
                ("SWITCHYARD_PRODUCT_TELEMETRY", disabled),
                ("SWITCHYARD_TELEMETRY_ENDPOINT", "log"),
            ]);
            let logs = LogBuffer::default();
            tracing::subscriber::with_default(logs.subscriber(), || {
                telemetry.emit_metadata(&OutcomeMetadata::new("test".into(), None), &[]);
            });
            assert!(matches!(telemetry.destination, Destination::Disabled));
            assert!(telemetry.client.get().is_none());
            assert!(logs.text().is_empty());
        }
        for enabled in ["1", "true", "yes", "on", ""] {
            assert!(matches!(
                configured(&[("SWITCHYARD_PRODUCT_TELEMETRY", enabled)]).destination,
                Destination::Http(_)
            ));
        }
    }

    #[test]
    fn log_endpoint_emits_complete_payload_at_info_level() {
        let telemetry = Arc::new(configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", "log")]));
        let metadata = OutcomeMetadata::new("test".into(), None);
        let selected = [ModelId::from("selected"), ModelId::from("fallback")];
        let logs = LogBuffer::default();
        let buffer = logs.clone();
        let subscriber = tracing_subscriber::registry()
            .with(
                ProductTelemetryLayer {
                    telemetry: telemetry.clone(),
                }
                .filtered(),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .without_time()
                    .with_ansi(false)
                    .with_writer(move || buffer.clone()),
            );
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(target: "libsy", "libsy.run", switchyard.outcome = tracing::field::Empty);
            let outcome = serde_json::json!({"metadata": metadata, "selected_model_ids": selected});
            span.record("switchyard.outcome", outcome.to_string().as_str());
            let retained = span.clone();
            drop(span);
            assert!(logs.text().is_empty());
            drop(retained);
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
    fn http_delivery_works_without_a_host_runtime() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let listener = runtime.block_on(TcpListener::bind("127.0.0.1:0")).unwrap();
        let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
        let telemetry = configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]);
        let server = runtime.spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            stream
                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            request
        });
        assert!(tokio::runtime::Handle::try_current().is_err());
        telemetry.emit_metadata(&OutcomeMetadata::new("test".into(), None), &[]);
        let (_, payload) = runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap()
        });
        telemetry.flush();
        assert_eq!(payload["events"][0]["parameters"]["algorithm"], "test");
        assert_eq!(
            *telemetry
                .worker
                .get()
                .unwrap()
                .as_ref()
                .unwrap()
                .pending
                .count
                .lock(),
            0
        );
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
        let telemetry = Arc::new(configured(&[("SWITCHYARD_TELEMETRY_ENDPOINT", &endpoint)]));
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
        telemetry.emit_metadata(&first, &[ModelId::from("selected")]);
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
        telemetry.emit_metadata(&second, &[]);
        let flushing = {
            let telemetry = telemetry.clone();
            tokio::task::spawn_blocking(move || telemetry.flush())
        };
        assert!(!flushing.is_finished());
        release_tx.send(()).unwrap();
        let (_, other) = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), flushing)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            *telemetry
                .worker
                .get()
                .unwrap()
                .as_ref()
                .unwrap()
                .pending
                .count
                .lock(),
            0
        );
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
            telemetry.emit_metadata(&OutcomeMetadata::new("test".into(), None), &[]);
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
            telemetry.emit_metadata(&OutcomeMetadata::new("test".into(), None), &[]);
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
        telemetry.emit_metadata(&OutcomeMetadata::new("test".into(), None), &[]);
        wait_for_failure(&logs).await;
        assert!(logs.text().contains("is_timeout=true"));
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }
    #[test]
    fn payload_wraps_routing_fields_and_reuses_process_session() {
        let evidence = serde_json::json!({
            "source": "fail_open", "reason_code": "transport", "score": 0.5,
        });
        let metadata = OutcomeMetadata::new("test".to_string(), Some(evidence.clone()));
        let session_id = uuid::Uuid::now_v7();
        let selected_models = ["selected", "fallback-b", "fallback-a"].map(ModelId::from);
        let record = build_payload(&outcome_record(&metadata, &selected_models), session_id);
        let parameters = &record["events"][0]["parameters"];
        assert_eq!(parameters["outcome_id"], metadata.outcome_id());
        assert_eq!(parameters["evidence"], evidence);
        assert_eq!(parameters["nemoSource"], "switchyard");
        assert_eq!(parameters["routing_status"], "success");
        assert_eq!(parameters["no_eligible_target"], false);
        assert_eq!(parameters["routing_error_code"], Value::Null);
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
        let other = build_payload(
            &outcome_record(&OutcomeMetadata::new("other".to_string(), None), &[]),
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
        let mut expected = serde_json::json!(metadata);
        expected["routing_status"] = serde_json::json!("success");
        expected["no_eligible_target"] = serde_json::json!(false);
        expected["routing_error_code"] = Value::Null;
        assert_eq!(parameters, expected);
    }
}

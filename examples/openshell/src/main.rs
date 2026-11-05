// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered inference middleware embedding Runner without forwarding the original call.

use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_llm_client::{RunObservation, RunObserver};
use switchyard_protocol::{Category, Metadata, ModelId, Request, WireFormat};
use switchyard_runner::{Runner, require_target_labels};
use switchyard_translation::{decode_request, encode_aggregated_response_with_extensions};
use tokio::io::AsyncWriteExt;
use tonic::{Response, Status, transport::Server};

#[allow(clippy::large_enum_variant, clippy::enum_variant_names)]
mod openshell {
    pub mod extension {
        pub mod v1 {
            tonic::include_proto!("openshell.extension.v1");
        }
    }
    pub mod middleware {
        pub mod v1 {
            tonic::include_proto!("openshell.middleware.v1");
        }
    }
}

use openshell::extension::v1::{PeerMetadata, ProtocolVersion};
use openshell::middleware::v1::supervisor_middleware_server::{
    SupervisorMiddleware, SupervisorMiddlewareServer,
};
use openshell::middleware::v1::*;

const CAPABILITY: &str = "openshell.supervisor-middleware.contract";
const RESPONSE_CAPABILITY: &str = "openshell.supervisor-middleware.buffered-response";

fn completed(status: u32, body: serde_json::Value) -> Result<HttpRequestResult> {
    let body = serde_json::to_vec(&body)?;
    ensure!(body.len() <= 262144, "response exceeds middleware limit");
    Ok(HttpRequestResult {
        decision: Decision::Respond as i32,
        response: Some(BufferedHttpResponse {
            status_code: status,
            headers: vec![HttpHeader {
                name: "content-type".into(),
                value: "application/json".into(),
            }],
            body,
        }),
        ..Default::default()
    })
}

#[derive(Deserialize)]
struct Config {
    listen: String,
    routes: PathBuf,
    route_id: String,
    journal: PathBuf,
    public_key: PathBuf,
    issuer: String,
    audience: String,
    tls: Option<Tls>,
    events_path: Option<PathBuf>,
}

/// Optional metadata-only observations for the local demo.
#[derive(Clone, Default)]
struct DemoEvents(Option<tokio::sync::mpsc::Sender<serde_json::Value>>);

impl DemoEvents {
    fn emit(&self, sandbox: &str, kind: &str, data: impl FnOnce() -> serde_json::Value) {
        let Some(sender) = &self.0 else { return };
        let event = serde_json::json!({
            "time_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
            "type": kind, "sandbox_id": sandbox, "data": data(),
        });
        if sender.try_send(event).is_err() {
            tracing::warn!("demo event unavailable");
        }
    }
}

/// Correlates metadata for one inference request without copying its contents.
struct RequestEvents<'a> {
    events: &'a DemoEvents,
    sandbox: &'a str,
    request_id: u64,
}

impl RequestEvents<'_> {
    fn emit(&self, kind: &str, data: impl FnOnce() -> serde_json::Value) {
        self.events.emit(self.sandbox, kind, || {
            let mut data = data();
            if let serde_json::Value::Object(fields) = &mut data {
                fields.insert("request_id".into(), self.request_id.into());
            }
            data
        });
    }

    fn observer(&self) -> Option<RunObserver> {
        self.events.0.as_ref()?;
        let events = self.events.clone();
        let sandbox = self.sandbox.to_owned();
        let request_id = self.request_id;
        Some(Arc::new(move |observation| {
            let events = RequestEvents {
                events: &events,
                sandbox: &sandbox,
                request_id,
            };
            let (role, call) = match observation {
                RunObservation::Outcome(metadata) => {
                    events.emit("algorithm_decision", || {
                        serde_json::json!({
                            "algorithm": metadata.algorithm, "evidence": metadata.evidence,
                        })
                    });
                    return;
                }
                RunObservation::RoutingOverhead(duration) => {
                    events.emit("routing_overhead", || {
                        serde_json::json!({
                            "elapsed_ms": duration.as_secs_f64() * 1000.0,
                        })
                    });
                    return;
                }
                RunObservation::AnswerCall(call) => ("answer", call),
                RunObservation::LlmCall(call) => ("routing", call),
                RunObservation::DecisionCall(call) => ("decision", call),
            };
            events.emit("model_call_completed", || {
                serde_json::json!({
                    "model": call.selected_model, "role": role, "is_success": call.is_success,
                    "elapsed_ms": call.duration.as_secs_f64() * 1000.0,
                })
            });
        }))
    }
}

/// Instruction selected by an administrator-owned OpenShell policy entry.
enum PolicyAction {
    RequireLabels(BTreeSet<String>),
    Execute,
}

impl PolicyAction {
    fn from_config(config: Option<&prost_types::Struct>) -> Result<Self> {
        let fields = &config.context("missing policy configuration")?.fields;
        let Some(prost_types::value::Kind::StringValue(action)) =
            fields.get("action").and_then(|value| value.kind.as_ref())
        else {
            anyhow::bail!("policy action must be a string");
        };
        match action.as_str() {
            "require_labels" => {
                ensure!(
                    fields.len() == 2,
                    "policy requires only action and required_labels"
                );
                let Some(prost_types::value::Kind::ListValue(values)) = fields
                    .get("required_labels")
                    .and_then(|value| value.kind.as_ref())
                else {
                    anyhow::bail!("required_labels must be an array of strings");
                };
                let labels = values
                    .values
                    .iter()
                    .map(|value| {
                        let Some(prost_types::value::Kind::StringValue(label)) = &value.kind else {
                            anyhow::bail!("required_labels must be an array of strings");
                        };
                        Ok(label.clone())
                    })
                    .collect::<Result<BTreeSet<_>>>()?;
                validate_labels(&labels)?;
                Ok(Self::RequireLabels(labels))
            }
            "mark_restricted" => {
                ensure!(
                    fields.len() == 1,
                    "policy configuration requires only action"
                );
                Ok(Self::RequireLabels(BTreeSet::from(["private".into()])))
            }
            "execute" => {
                ensure!(
                    fields.len() == 1,
                    "policy configuration requires only action"
                );
                Ok(Self::Execute)
            }
            _ => anyhow::bail!("unknown policy action"),
        }
    }
}

fn validate_labels(labels: &BTreeSet<String>) -> Result<()> {
    ensure!(!labels.is_empty(), "source requirements must not be empty");
    require_target_labels(&mut Request::default(), labels)?;
    Ok(())
}

fn validate_sandbox(sandbox: &str) -> Result<()> {
    ensure!(
        !sandbox.is_empty() && sandbox.len() <= 512 && sandbox.trim() == sandbox,
        "invalid sandbox identity"
    );
    Ok(())
}

#[derive(Deserialize)]
struct Tls {
    cert: PathBuf,
    key: PathBuf,
}

#[derive(Clone, Deserialize)]
struct Claims {
    sub: String,
    caller_kind: String,
    sandbox_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RequirementRecord {
    sandbox_id: String,
    required_labels: BTreeSet<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JournalEntry {
    Requirements(RequirementRecord),
    Legacy(String),
}

struct Retention {
    tasks: HashMap<String, BTreeSet<String>>,
    journal: File,
    healthy: bool,
}

impl Retention {
    const MAX_TASKS: usize = 4096;

    fn load(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        ensure!(
            contents.is_empty() || contents.ends_with('\n'),
            "truncated restriction journal"
        );
        let journal = OpenOptions::new().append(true).mode(0o600).open(path)?;
        let mut tasks: HashMap<String, BTreeSet<String>> = HashMap::new();
        for line in contents.lines() {
            let record = match serde_json::from_str::<JournalEntry>(line)? {
                JournalEntry::Requirements(record) => record,
                JournalEntry::Legacy(sandbox_id) => RequirementRecord {
                    sandbox_id,
                    required_labels: BTreeSet::from(["private".into()]),
                },
            };
            validate_sandbox(&record.sandbox_id)?;
            validate_labels(&record.required_labels)?;
            let labels = tasks.entry(record.sandbox_id).or_default();
            labels.extend(record.required_labels);
            validate_labels(labels)?;
            ensure!(
                tasks.len() <= Self::MAX_TASKS,
                "requirement capacity exhausted"
            );
        }
        Ok(Self {
            tasks,
            journal,
            healthy: true,
        })
    }

    fn requirements(&self, sandbox: &str) -> Result<BTreeSet<String>> {
        ensure!(self.healthy, "restriction journal unavailable");
        Ok(self.tasks.get(sandbox).cloned().unwrap_or_default())
    }

    // Persist the union before allowing protected data access.
    fn require(&mut self, sandbox: &str, labels: BTreeSet<String>) -> Result<bool> {
        ensure!(self.healthy, "restriction journal unavailable");
        validate_sandbox(sandbox)?;
        let retained = self.tasks.get(sandbox);
        if retained.is_some_and(|retained| labels.is_subset(retained)) {
            return Ok(false);
        }
        ensure!(
            retained.is_some() || self.tasks.len() < Self::MAX_TASKS,
            "requirement capacity exhausted"
        );
        let mut required_labels = retained.cloned().unwrap_or_default();
        required_labels.extend(labels);
        validate_labels(&required_labels)?;
        let record = RequirementRecord {
            sandbox_id: sandbox.into(),
            required_labels,
        };
        self.healthy = false;
        serde_json::to_writer(&mut self.journal, &record)?;
        self.journal.write_all(b"\n")?;
        self.journal.sync_data()?;
        self.tasks.insert(record.sandbox_id, record.required_labels);
        self.healthy = true;
        Ok(true)
    }
}

struct Middleware {
    runner: Runner,
    route_id: ModelId,
    retention: Mutex<Retention>,
    key: DecodingKey,
    validation: Validation,
    audience: String,
    issuer: String,
    events: DemoEvents,
    request_ids: AtomicU64,
}

impl Middleware {
    fn authenticate<T>(&self, call: &tonic::Request<T>) -> Result<Claims, Status> {
        let token = call
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| Status::unauthenticated("missing supervisor identity"))?;
        let header = decode_header(token).map_err(|_| Status::unauthenticated("invalid token"))?;
        if header.typ.as_deref() != Some("openshell-ext+jwt") {
            return Err(Status::unauthenticated("invalid token type"));
        }
        let claims = decode::<Claims>(token, &self.key, &self.validation)
            .map_err(|_| Status::unauthenticated("invalid supervisor identity"))?
            .claims;
        let valid = match (claims.caller_kind.as_str(), claims.sandbox_id.as_deref()) {
            ("gateway", None) => claims.sub == self.issuer,
            ("supervisor", Some(id)) => {
                validate_sandbox(id).is_ok()
                    && claims.sub == format!("spiffe://openshell/sandbox/{id}")
            }
            _ => false,
        };
        if !valid {
            return Err(Status::unauthenticated("invalid caller identity"));
        }
        Ok(claims)
    }

    async fn evaluate(
        &self,
        input: HttpRequestEvaluation,
        sandbox: String,
    ) -> Result<HttpRequestResult> {
        ensure!(
            input.phase == SupervisorMiddlewarePhase::PreCredentials as i32
                && input.body.len() <= 262144,
            "unsupported evaluation"
        );
        let action = PolicyAction::from_config(input.config.as_ref())?;
        let target = input.target.context("missing destination")?;
        if let PolicyAction::RequireLabels(labels) = action {
            let mut retention = self.retention.lock();
            if retention.require(&sandbox, labels)? && self.events.0.is_some() {
                let required_labels = retention.requirements(&sandbox)?;
                self.events.emit(&sandbox, "requirements_changed", || {
                    serde_json::json!({
                        "required_labels": required_labels,
                        "host": target.host, "path": target.path,
                    })
                });
            }
            return Ok(HttpRequestResult {
                decision: Decision::Allow as i32,
                ..Default::default()
            });
        }
        ensure!(
            target.method == "POST"
                && target.path == "/v1/chat/completions"
                && target.query.is_empty(),
            "unsupported inference target"
        );
        let mut body: serde_json::Value = serde_json::from_slice(&input.body)?;
        // Match serving ingress: caller JSON cannot supply internal preservation state.
        if let Some(metadata) = body
            .get_mut("metadata")
            .and_then(serde_json::Value::as_object_mut)
        {
            metadata.remove("_switchyard_translation");
        }
        let mut request = Request {
            llm_request: decode_request(WireFormat::OpenAiChat, &body)?,
            metadata: Some(Metadata {
                task_id: Some(sandbox.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        ensure!(!request.llm_request.stream, "buffered inference only");
        let extensions = request.llm_request.extensions.clone();
        let required_labels = self.retention.lock().requirements(&sandbox)?;
        require_target_labels(&mut request, &required_labels)?;
        let route = self
            .runner
            .route(self.route_id.as_str())
            .context("unknown route")?;
        let events = RequestEvents {
            events: &self.events,
            sandbox: &sandbox,
            request_id: self.request_ids.fetch_add(1, Ordering::Relaxed),
        };
        let started = Instant::now();
        events.emit("routing_started", || {
            serde_json::json!({
                "algorithm": route.algorithm_name(), "required_labels": required_labels,
            })
        });
        let output = match route.execute(request, events.observer()).await {
            Ok(output) => output,
            Err(error) => {
                let summary = error.execution_error_summary();
                tracing::debug!(kind = ?summary.kind, "model execution failed");
                events.emit("error", || {
                    serde_json::json!({
                        "message": "Approved models unavailable",
                    })
                });
                return completed(
                    503,
                    serde_json::json!({
                        "error": { "type": "upstream_unavailable", "message": "Approved models unavailable" }
                    }),
                );
            }
        };
        let served_target = output
            .response
            .served_model()
            .cloned()
            .unwrap_or_else(|| output.selected_model.clone());
        let selected = route.provider_model(&output.selected_model);
        let served = route.provider_model(&served_target);
        let response = output.response.llm_response.into_agg().await?;
        let body = encode_aggregated_response_with_extensions(
            &response,
            WireFormat::OpenAiChat,
            Some(served.as_str()),
            &extensions,
        )?;
        let mut result = completed(200, body)?;
        result.metadata = [
            ("selected_model".into(), selected.to_string()),
            ("served_model".into(), served.to_string()),
            (
                "lane".into(),
                if required_labels.is_empty() {
                    "standard"
                } else {
                    "restricted"
                }
                .into(),
            ),
            (
                "required_labels".into(),
                serde_json::to_string(&required_labels)?,
            ),
        ]
        .into();
        events.emit("request_completed", || {
            serde_json::json!({
                "algorithm": route.algorithm_name(), "required_labels": required_labels,
                "selected_model": selected, "served_model": served,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            })
        });
        Ok(result)
    }
}

#[tonic::async_trait]
impl SupervisorMiddleware for Middleware {
    async fn describe(
        &self,
        call: tonic::Request<MiddlewareDescribeRequest>,
    ) -> Result<Response<MiddlewareManifest>, Status> {
        self.authenticate(&call)?;
        let peer = call
            .into_inner()
            .gateway
            .ok_or_else(|| Status::failed_precondition("missing protocol"))?;
        if peer.protocol_version.as_ref().is_none_or(|v| v.major != 1)
            || !peer.supported_capabilities.iter().any(|c| c == CAPABILITY)
            || !peer
                .supported_capabilities
                .iter()
                .any(|c| c == RESPONSE_CAPABILITY)
            || peer
                .required_capabilities
                .iter()
                .any(|c| c != CAPABILITY && c != RESPONSE_CAPABILITY)
        {
            return Err(Status::failed_precondition("unsupported protocol"));
        }
        tracing::debug!("middleware protocol accepted");
        Ok(Response::new(MiddlewareManifest {
            name: "switchyard-openshell".into(),
            expected_audience: self.audience.clone(),
            bindings: vec![MiddlewareBinding {
                operation: SupervisorMiddlewareOperation::HttpRequest as i32,
                phase: SupervisorMiddlewarePhase::PreCredentials as i32,
                max_payload_bytes: 262144,
                ..Default::default()
            }],
            extension: Some(PeerMetadata {
                protocol_version: Some(ProtocolVersion { major: 1, minor: 0 }),
                implementation_name: "switchyard-openshell".into(),
                implementation_version: "0.0.0".into(),
                supported_capabilities: vec![CAPABILITY.into(), RESPONSE_CAPABILITY.into()],
                required_capabilities: vec![RESPONSE_CAPABILITY.into()],
            }),
            ..Default::default()
        }))
    }

    async fn validate_config(
        &self,
        call: tonic::Request<ValidateConfigRequest>,
    ) -> Result<Response<ValidateConfigResponse>, Status> {
        self.authenticate(&call)?;
        let valid = PolicyAction::from_config(call.into_inner().config.as_ref()).is_ok();
        Ok(Response::new(ValidateConfigResponse {
            valid,
            ..Default::default()
        }))
    }

    async fn evaluate_http_request(
        &self,
        call: tonic::Request<HttpRequestEvaluation>,
    ) -> Result<Response<HttpRequestResult>, Status> {
        let claims = self.authenticate(&call)?;
        let input = call.into_inner();
        let sandbox = claims
            .sandbox_id
            .filter(|id| input.context.as_ref().is_some_and(|c| c.sandbox_id == *id))
            .ok_or_else(|| Status::permission_denied("sandbox identity mismatch"))?;
        self.evaluate(input, sandbox)
            .await
            .map(Response::new)
            .map_err(|_| Status::permission_denied("request admission failed"))
    }

    type EvaluateWebSocketSessionStream =
        tokio_stream::Empty<Result<WebSocketSessionEventResult, Status>>;

    async fn evaluate_web_socket_session(
        &self,
        _: tonic::Request<tonic::Streaming<WebSocketSessionEvent>>,
    ) -> Result<Response<Self::EvaluateWebSocketSessionStream>, Status> {
        Err(Status::unimplemented("request-only spike"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("info")
        .init();
    let config_path = std::env::args()
        .nth(1)
        .context("usage: middleware config.json")?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let runner = Runner::load(&config.routes)?;
    let route = runner
        .route(&config.route_id)
        .context("unknown configured route")?;
    ensure!(
        matches!(route.algorithm_name(), "stage_router" | "passthrough")
            && route.models().models_for(&Category::Judge).is_empty(),
        "spike requires judge-free Stage or passthrough"
    );
    let mut validation = Validation::new(Algorithm::EdDSA);
    validation.set_audience(&[&config.audience]);
    validation.set_issuer(&[&config.issuer]);
    validation.validate_nbf = true;
    validation.leeway = 0;
    // Keep callbacks nonblocking and drain their bounded metadata queue on shutdown.
    let (events, writer) = if let Some(path) = config.events_path {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
            .await?;
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<serde_json::Value>(256);
        let writer = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                let mut line = serde_json::to_vec(&event)?;
                line.push(b'\n');
                file.write_all(&line).await?;
            }
            file.flush().await?;
            anyhow::Ok(())
        });
        (DemoEvents(Some(sender)), Some(writer))
    } else {
        (DemoEvents::default(), None)
    };
    let service = Middleware {
        key: DecodingKey::from_ed_pem(&std::fs::read(&config.public_key)?)?,
        validation,
        route_id: ModelId::from(config.route_id),
        retention: Mutex::new(Retention::load(&config.journal)?),
        runner,
        audience: config.audience,
        issuer: config.issuer,
        events: events.clone(),
        request_ids: AtomicU64::new(1),
    };
    let mut server = Server::builder();
    if let Some(tls) = config.tls {
        server = server.tls_config(tonic::transport::ServerTlsConfig::new().identity(
            tonic::transport::Identity::from_pem(std::fs::read(tls.cert)?, std::fs::read(tls.key)?),
        ))?;
    }
    tracing::info!(listen = %config.listen, "middleware ready");
    let result = server
        .add_service(SupervisorMiddlewareServer::new(service))
        .serve_with_shutdown(config.listen.parse()?, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;
    drop(events);
    if let Some(writer) = writer {
        writer.await??;
    }
    result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_observer_keeps_failed_attempts_and_disabled_events_are_lazy() {
        let disabled = DemoEvents::default();
        RequestEvents {
            events: &disabled,
            sandbox: "sandbox",
            request_id: 1,
        }
        .emit("ignored", || panic!("disabled sink"));
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let sink = DemoEvents(Some(sender));
        let observer = RequestEvents {
            events: &sink,
            sandbox: "sandbox",
            request_id: 7,
        }
        .observer()
        .unwrap();
        observer(RunObservation::AnswerCall(
            switchyard_llm_client::ModelCallObservation {
                selected_model: ModelId::from("approved/model"),
                is_success: false,
                duration: std::time::Duration::from_millis(12),
                usage: None,
            },
        ));
        let event = receiver.try_recv().unwrap();
        assert_eq!(event["type"], "model_call_completed");
        assert_eq!(event["sandbox_id"], "sandbox");
        assert_eq!(
            event["data"],
            serde_json::json!({
                "model": "approved/model", "role": "answer", "is_success": false, "elapsed_ms": 12.0, "request_id": 7,
            })
        );
        RequestEvents {
            events: &sink,
            sandbox: "sandbox",
            request_id: 8,
        }
        .emit(
            "routing_started",
            || serde_json::json!({"algorithm": "stage_router"}),
        );
        assert_eq!(receiver.try_recv().unwrap()["data"]["request_id"], 8);
    }
}

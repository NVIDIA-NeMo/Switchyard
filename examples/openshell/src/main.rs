// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered inference middleware embedding Runner without forwarding the original call.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use parking_lot::Mutex;
use serde::Deserialize;
use switchyard_protocol::{Category, Metadata, ModelId, Request, WireFormat};
use switchyard_runner::{Runner, require_restricted_targets};
use switchyard_translation::{decode_request, encode_aggregated_response_with_extensions};
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
}

/// Instruction selected by an administrator-owned OpenShell policy entry.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum PolicyAction {
    MarkRestricted,
    Execute,
}

impl PolicyAction {
    fn from_config(config: Option<&prost_types::Struct>) -> Result<Self> {
        let fields = &config.context("missing policy configuration")?.fields;
        ensure!(
            fields.len() == 1,
            "policy configuration requires only action"
        );
        let Some(prost_types::value::Kind::StringValue(action)) =
            fields.get("action").and_then(|value| value.kind.as_ref())
        else {
            anyhow::bail!("policy action must be a string");
        };
        Self::deserialize(serde::de::value::StrDeserializer::<serde::de::value::Error>::new(action))
            .context("unknown policy action")
    }
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

struct Retention {
    tasks: HashSet<String>,
    journal: File,
    healthy: bool,
}

impl Retention {
    fn load(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        ensure!(
            contents.is_empty() || contents.ends_with('\n'),
            "truncated restriction journal"
        );
        let journal = OpenOptions::new().append(true).mode(0o600).open(path)?;
        let tasks = contents
            .lines()
            .map(|line| Ok(serde_json::from_str::<String>(line)?))
            .collect::<Result<HashSet<_>>>()?;
        Ok(Self {
            tasks,
            journal,
            healthy: true,
        })
    }

    fn is_restricted(&self, sandbox: &str) -> Result<bool> {
        ensure!(self.healthy, "restriction journal unavailable");
        Ok(self.tasks.contains(sandbox))
    }

    // Record the restriction durably before allowing protected data access.
    fn restrict(&mut self, sandbox: &str) -> Result<()> {
        ensure!(self.healthy, "restriction journal unavailable");
        if !self.tasks.contains(sandbox) {
            ensure!(self.tasks.len() < 4096, "restriction capacity exhausted");
            self.healthy = false;
            serde_json::to_writer(&mut self.journal, sandbox)?;
            self.journal.write_all(b"\n")?;
            self.journal.sync_data()?;
            self.tasks.insert(sandbox.to_owned());
            self.healthy = true;
        }
        Ok(())
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
                !id.is_empty()
                    && id.len() <= 512
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
        if let PolicyAction::MarkRestricted = action {
            self.retention.lock().restrict(&sandbox)?;
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
        let restricted = self.retention.lock().is_restricted(&sandbox)?;
        require_restricted_targets(&mut request, restricted)?;
        let route = self
            .runner
            .route(self.route_id.as_str())
            .context("unknown route")?;
        let output = match route.execute(request, None).await {
            Ok(output) => output,
            Err(error) => {
                let summary = error.execution_error_summary();
                tracing::debug!(kind = ?summary.kind, "model execution failed");
                return completed(
                    503,
                    serde_json::json!({
                        "error": { "type": "upstream_unavailable", "message": "Approved models unavailable" }
                    }),
                );
            }
        };
        let served = output
            .response
            .served_model()
            .cloned()
            .unwrap_or_else(|| output.selected_model.clone());
        let response = output.response.llm_response.into_agg().await?;
        let body = encode_aggregated_response_with_extensions(
            &response,
            WireFormat::OpenAiChat,
            Some(served.as_str()),
            &extensions,
        )?;
        let mut result = completed(200, body)?;
        result.metadata = [
            ("selected_model".into(), output.selected_model.to_string()),
            ("served_model".into(), served.to_string()),
            (
                "lane".into(),
                if restricted { "restricted" } else { "standard" }.into(),
            ),
        ]
        .into();
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
    let service = Middleware {
        key: DecodingKey::from_ed_pem(&std::fs::read(&config.public_key)?)?,
        validation,
        route_id: ModelId::from(config.route_id),
        retention: Mutex::new(Retention::load(&config.journal)?),
        runner,
        audience: config.audience,
        issuer: config.issuer,
    };
    let mut server = Server::builder();
    if let Some(tls) = config.tls {
        server = server.tls_config(tonic::transport::ServerTlsConfig::new().identity(
            tonic::transport::Identity::from_pem(std::fs::read(tls.cert)?, std::fs::read(tls.key)?),
        ))?;
    }
    tracing::info!(listen = %config.listen, "middleware ready");
    server
        .add_service(SupervisorMiddlewareServer::new(service))
        .serve_with_shutdown(config.listen.parse()?, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

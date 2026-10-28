// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the worker's authenticated gRPC boundary and cross-provider execution.

use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use aws_lc_rs::{
    encoding::AsDer,
    rand::SystemRandom,
    signature::{Ed25519KeyPair, KeyPair},
};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tonic::transport::Channel;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
use openshell::middleware::v1::{supervisor_middleware_client::SupervisorMiddlewareClient, *};

const CAPABILITY: &str = "openshell.supervisor-middleware.contract";
const RESPONSE_CAPABILITY: &str = "openshell.supervisor-middleware.buffered-response";

fn identity() -> (EncodingKey, String) {
    let private = Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).unwrap();
    let pair = Ed25519KeyPair::from_pkcs8(private.as_ref()).unwrap();
    let public = pair.public_key().as_der().unwrap();
    (
        EncodingKey::from_ed_der(private.as_ref()),
        pem::encode(&pem::Pem::new("PUBLIC KEY", public.as_ref())),
    )
}

fn policy(action: &str) -> prost_types::Struct {
    prost_types::Struct {
        fields: [("action".into(), action.into())].into(),
    }
}

fn request() -> Value {
    json!({"model": "switchyard/demo", "stream": false, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Prepare a renewal meeting brief."}]})
}

struct Fixture {
    directory: TempDir,
    process: Option<Child>,
    client: SupervisorMiddlewareClient<Channel>,
    address: String,
    key: EncodingKey,
    standard: MockServer,
    private: MockServer,
    private_available: Arc<AtomicBool>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Fixture {
    async fn new(stage: bool) -> Result<Self> {
        let standard = MockServer::start().await;
        let private = MockServer::start().await;
        let private_available = Arc::new(AtomicBool::new(true));
        for (server, endpoint) in [
            (&standard, "/v1/chat/completions"),
            (&private, "/v1/messages"),
        ] {
            let available = Arc::clone(&private_available);
            Mock::given(method("POST")).and(path(endpoint))
                .respond_with(move |request: &wiremock::Request| {
                    let body: Value = serde_json::from_slice(&request.body).unwrap();
                    let model = body["model"].as_str().unwrap();
                    if endpoint == "/v1/messages" {
                        assert!(["private/efficient", "private/capable"].contains(&model));
                    } else {
                        assert_eq!(model, "cloud/efficient");
                    }
                    if (stage && model == "private/efficient") || (model != "cloud/efficient" && !available.load(Ordering::Relaxed)) {
                        return ResponseTemplate::new(503).set_body_json(json!({
                            "error": {"type": "overloaded_error", "message": "Unavailable"}
                        }));
                    }
                    ResponseTemplate::new(200).set_body_json(if model == "cloud/efficient" {
                        json!({"id": "chat_standard", "object": "chat.completion", "model": model,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Public brief"}, "finish_reason": "stop"}]})
                    } else {
                        json!({"id": "msg_private", "type": "message", "role": "assistant", "model": model,
                            "content": [{"type": "text", "text": "Private brief"},
                                {"type": "tool_use", "id": "toolu_next", "name": "read_account", "input": {"id": "next"}}],
                            "stop_reason": "tool_use", "usage": {"input_tokens": 11, "output_tokens": 7}})
                    })
                }).mount(server).await;
        }
        let directory = tempfile::tempdir()?;
        let algorithm = if stage {
            "type = \"stage_router\"\nefficient_target = \"fast\"\ncapable_target = \"strong\"\npicker = \"efficient_first\"\nconfidence_threshold = 0.5"
        } else {
            "type = \"passthrough\"\ntarget = \"fast\""
        };
        let overrides = if stage {
            "fast = \"private_fast\", strong = \"private_strong\""
        } else {
            "fast = \"private_fast\""
        };
        std::fs::write(
            directory.path().join("routes.toml"),
            format!(
                r#"
schema_version = 1
[llm_clients]
standard = {{ format = "openai_chat", base_url = "{}/v1", max_retries = 0, timeout_ms = 2000 }}
private = {{ format = "anthropic_messages", base_url = "{}/v1", max_retries = 0, failure_cooldown_ms = 0, timeout_ms = 2000 }}
[targets]
fast = {{ id = "cloud/efficient", llm_client = "standard" }}
strong = {{ id = "cloud/capable", llm_client = "standard" }}
private_fast = {{ id = "private/efficient", llm_client = "private", restricted = true, system_prompt = "Approved efficient" }}
private_strong = {{ id = "private/capable", llm_client = "private", restricted = true, system_prompt = "Approved capable" }}
[routes.demo]
id = "switchyard/demo"
{algorithm}
[routes.demo.restricted]
require_explicit_signal = true
scope = "task"
target_overrides = {{ {overrides} }}
"#,
                standard.uri(),
                private.uri()
            ),
        )?;
        let (key, public) = identity();
        std::fs::write(directory.path().join("public.pem"), public)?;
        std::fs::write(directory.path().join("journal.jsonl"), "")?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&json!({
                "listen": address.to_string(), "routes": directory.path().join("routes.toml"), "route_id": "switchyard/demo",
                "journal": directory.path().join("journal.jsonl"), "public_key": directory.path().join("public.pem"),
                "issuer": "openshell-gateway:test", "audience": "urn:openshell:extension:middleware:test"
            }))?,
        )?;
        let address = format!("http://{address}");
        let mut fixture = Self {
            directory,
            process: None,
            client: SupervisorMiddlewareClient::new(
                Channel::from_shared(address.clone())?.connect_lazy(),
            ),
            address,
            key,
            standard,
            private,
            private_available,
        };
        drop(listener);
        fixture.start().await?;
        Ok(fixture)
    }

    async fn start(&mut self) -> Result<()> {
        self.process = Some(
            Command::new(env!("CARGO_BIN_EXE_switchyard-openshell-middleware"))
                .arg(self.directory.path().join("config.json"))
                .stdout(Stdio::null())
                .stderr(std::fs::File::create(
                    self.directory.path().join("service.log"),
                )?)
                .spawn()?,
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Ok(client) = SupervisorMiddlewareClient::connect(self.address.clone()).await
                {
                    self.client = client;
                    return Ok(());
                }
                if let Some(status) = self.process.as_mut().unwrap().try_wait()? {
                    anyhow::bail!(
                        "worker exited {status}: {}",
                        std::fs::read_to_string(self.directory.path().join("service.log"))?
                    );
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .context("worker did not start")?
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn token(&self, sandbox: &str, key: &EncodingKey, expiration: u64) -> String {
        let mut header = Header::new(Algorithm::EdDSA);
        header.typ = Some("openshell-ext+jwt".into());
        encode(
            &header,
            &json!({
                "iss": "openshell-gateway:test", "aud": "urn:openshell:extension:middleware:test",
                "sub": format!("spiffe://openshell/sandbox/{sandbox}"), "caller_kind": "supervisor",
                "sandbox_id": sandbox, "exp": expiration,
            }),
            key,
        )
        .unwrap()
    }

    fn authenticated<T>(&self, sandbox: &str, input: T) -> tonic::Request<T> {
        let mut call = tonic::Request::new(input);
        let expiration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 120;
        let token = self.token(sandbox, &self.key, expiration);
        call.metadata_mut()
            .insert("authorization", format!("Bearer {token}").parse().unwrap());
        call
    }

    fn evaluation(
        &self,
        sandbox: &str,
        body: &Value,
        config: Option<prost_types::Struct>,
    ) -> HttpRequestEvaluation {
        HttpRequestEvaluation {
            phase: SupervisorMiddlewarePhase::PreCredentials as i32,
            context: Some(RequestContext {
                sandbox_id: sandbox.into(),
                ..Default::default()
            }),
            target: Some(HttpRequestTarget {
                method: "POST".into(),
                scheme: "https".into(),
                host: "models.bank.example".into(),
                port: 443,
                path: "/v1/chat/completions".into(),
                ..Default::default()
            }),
            body: serde_json::to_vec(body).unwrap(),
            config,
            ..Default::default()
        }
    }

    async fn evaluate(
        &self,
        sandbox: &str,
        input: HttpRequestEvaluation,
    ) -> Result<HttpRequestResult, tonic::Status> {
        Ok(self
            .client
            .clone()
            .evaluate_http_request(self.authenticated(sandbox, input))
            .await?
            .into_inner())
    }

    async fn infer(&self, sandbox: &str, body: &Value) -> Result<(HttpRequestResult, Value)> {
        let input = self.evaluation(sandbox, body, Some(policy("execute")));
        let result = self.evaluate(sandbox, input).await?;
        assert_eq!(result.decision, Decision::Respond as i32);
        assert!(!result.has_body);
        let response = serde_json::from_slice(&result.response.as_ref().unwrap().body)?;
        Ok((result, response))
    }

    async fn restrict(&self, sandbox: &str) -> Result<()> {
        let mut input = self.evaluation(sandbox, &json!({}), Some(policy("mark_restricted")));
        input.target.as_mut().unwrap().host = "records.bank.example".into();
        input.target.as_mut().unwrap().path = "/account/123".into();
        assert_eq!(
            self.evaluate(sandbox, input).await?.decision,
            Decision::Allow as i32
        );
        // ALLOW must not arrive before the durable restriction.
        assert!(
            self.journal()
                .contains(&format!("{}\n", serde_json::to_string(sandbox)?))
        );
        Ok(())
    }

    fn journal(&self) -> String {
        std::fs::read_to_string(self.directory.path().join("journal.jsonl")).unwrap()
    }

    async fn private_models(&self) -> Vec<Value> {
        self.private
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|call| serde_json::from_slice::<Value>(&call.body).unwrap()["model"].clone())
            .collect()
    }

    async fn valid_config(&self, config: Option<prost_types::Struct>) -> Result<bool> {
        let input = ValidateConfigRequest {
            config,
            ..Default::default()
        };
        let call = self.authenticated("task-a", input);
        Ok(self
            .client
            .clone()
            .validate_config(call)
            .await?
            .into_inner()
            .valid)
    }
}

#[tokio::test]
async fn contract_and_restriction_survive_worker_restart() -> Result<()> {
    let mut f = Fixture::new(false).await?;
    let mut client = f.client.clone();
    let mut describe = MiddlewareDescribeRequest {
        gateway: Some(PeerMetadata {
            protocol_version: Some(ProtocolVersion { major: 1, minor: 0 }),
            supported_capabilities: vec![CAPABILITY.into(), RESPONSE_CAPABILITY.into()],
            ..Default::default()
        }),
    };
    let call = f.authenticated("task-a", describe.clone());
    let manifest = client.describe(call).await?.into_inner();
    assert_eq!(
        manifest.expected_audience,
        "urn:openshell:extension:middleware:test"
    );
    assert_eq!(
        manifest.extension.unwrap().required_capabilities,
        [RESPONSE_CAPABILITY]
    );
    assert_eq!(
        manifest.bindings[0].operation,
        SupervisorMiddlewareOperation::HttpRequest as i32
    );
    describe
        .gateway
        .as_mut()
        .unwrap()
        .supported_capabilities
        .pop();
    let call = f.authenticated("task-a", describe);
    assert_eq!(
        client.describe(call).await.unwrap_err().code(),
        tonic::Code::FailedPrecondition
    );
    let body = request();
    assert_eq!(
        f.infer("task-a", &body).await?.1["model"],
        "cloud/efficient"
    );
    f.restrict("task-a").await?;
    assert_eq!(
        f.infer("task-a", &body).await?.1["model"],
        "private/efficient"
    );
    f.stop();
    f.start().await?;
    assert_eq!(
        f.infer("task-a", &body).await?.1["model"],
        "private/efficient"
    );
    assert_eq!(
        f.infer("task-b", &body).await?.1["model"],
        "cloud/efficient"
    );
    Ok(())
}

#[tokio::test]
async fn policy_actions_and_verified_identity_control_admission() -> Result<()> {
    let f = Fixture::new(false).await?;
    let mut client = f.client.clone();
    for action in ["mark_restricted", "execute"] {
        assert!(f.valid_config(Some(policy(action))).await?);
    }
    let mut extra = policy("execute");
    extra.fields.insert("extra".into(), Default::default());
    let non_string = prost_types::Struct {
        fields: [("action".into(), 1.0.into())].into(),
    };
    for config in [
        None,
        Some(Default::default()),
        Some(policy("unknown")),
        Some(extra),
        Some(non_string),
    ] {
        assert!(!f.valid_config(config.clone()).await?);
        let input = f.evaluation("task-a", &request(), config);
        assert_eq!(
            f.evaluate("task-a", input).await.unwrap_err().code(),
            tonic::Code::PermissionDenied
        );
    }
    let input = f.evaluation("task-b", &request(), Some(policy("mark_restricted")));
    let error = f.evaluate("task-a", input.clone()).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::PermissionDenied);
    let (forged_key, _) = identity();
    for (key, expiration) in [(&forged_key, u64::MAX / 2), (&f.key, 1)] {
        let mut call = f.authenticated("task-b", input.clone());
        call.metadata_mut().insert(
            "authorization",
            format!("Bearer {}", f.token("task-b", key, expiration)).parse()?,
        );
        let error = client.evaluate_http_request(call).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unauthenticated);
    }
    let error = client.evaluate_http_request(input).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
    assert!(f.standard.received_requests().await.unwrap().is_empty());
    assert!(f.private_models().await.is_empty());
    assert!(f.journal().is_empty());
    // The policy action, not the destination or body shape, selects the behavior.
    let mut execute_source = f.evaluation("public", &request(), Some(policy("execute")));
    execute_source.target.as_mut().unwrap().host = "records.bank.example".into();
    assert_eq!(
        f.evaluate("public", execute_source).await?.decision,
        Decision::Respond as i32
    );
    assert!(f.journal().is_empty());
    let mark = f.evaluation("private", &request(), Some(policy("mark_restricted")));
    assert_eq!(
        f.evaluate("private", mark).await?.decision,
        Decision::Allow as i32
    );
    assert_eq!(f.standard.received_requests().await.unwrap().len(), 1);
    let infer = |sandbox, value: &str| {
        let mut input = f.evaluation(sandbox, &request(), Some(policy("execute")));
        input.headers.push(HttpHeader {
            name: "x-switchyard-restricted".into(),
            value: value.into(),
        });
        f.evaluate(sandbox, input)
    };
    let (private, public) = tokio::join!(infer("private", "false"), infer("public", "true"));
    for (result, expected) in [
        (private?, "private/efficient"),
        (public?, "cloud/efficient"),
    ] {
        let response: Value = serde_json::from_slice(&result.response.unwrap().body)?;
        assert_eq!(response["model"], expected);
    }
    Ok(())
}

#[tokio::test]
async fn cross_provider_tools_and_fallbacks_stay_private() -> Result<()> {
    let f = Fixture::new(true).await?;
    f.infer("task-a", &request()).await?;
    f.restrict("task-a").await?;
    let mut body = request();
    body["tools"] = json!([{"type": "function", "function": {"name": "read_account",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]);
    body["messages"] = json!([
        {"role": "system", "content": "Prepare a brief"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_account", "type": "function",
            "function": {"name": "read_account", "arguments": "{\"id\":\"synthetic\"}"}}]},
        {"role": "tool", "tool_call_id": "call_account", "content": "SYNTHETIC-ACCOUNT"},
        {"role": "user", "content": "Summarize"}
    ]);
    let (result, response) = f.infer("task-a", &body).await?;
    assert_eq!(result.response.unwrap().status_code, 200);
    assert_eq!(result.metadata["selected_model"], "private/efficient");
    assert_eq!(
        f.private_models().await,
        ["private/efficient", "private/capable"]
    );
    let calls = f.private.received_requests().await.unwrap();
    for (call, prompt) in calls.iter().zip(["Approved efficient", "Approved capable"]) {
        assert_eq!(call.headers["anthropic-version"], "2023-06-01");
        let sent: Value = serde_json::from_slice(&call.body)?;
        assert_eq!(sent["system"], format!("{prompt}\n\nPrepare a brief"));
        assert_eq!(sent["tools"][0]["name"], "read_account");
        assert_eq!(
            sent["tools"][0]["input_schema"],
            body["tools"][0]["function"]["parameters"]
        );
        assert_eq!(
            sent["messages"][0]["content"][0],
            json!({"type": "tool_use",
            "id": "call_account", "name": "read_account", "input": {"id": "synthetic"}})
        );
        assert_eq!(
            sent["messages"][1]["content"][0],
            json!({"type": "tool_result",
            "tool_use_id": "call_account", "content": "SYNTHETIC-ACCOUNT"})
        );
        assert_eq!(sent["max_tokens"], 64);
    }
    assert_eq!(response["model"], "private/capable");
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    let message = &response["choices"][0]["message"];
    assert_eq!(message["content"], "Private brief");
    let function = &message["tool_calls"][0]["function"];
    assert_eq!(function["name"], "read_account");
    assert_eq!(
        serde_json::from_str::<Value>(function["arguments"].as_str().unwrap())?,
        json!({"id": "next"})
    );
    assert_eq!(
        response["usage"],
        json!({"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})
    );
    f.private_available.store(false, Ordering::Relaxed);
    let (failed, error) = f.infer("task-a", &body).await?;
    assert_eq!(failed.response.unwrap().status_code, 503);
    assert_eq!(error["error"]["type"], "upstream_unavailable");
    assert_eq!(
        f.private_models().await,
        [
            "private/efficient",
            "private/capable",
            "private/efficient",
            "private/capable"
        ]
    );
    body["stream"] = json!(true);
    let input = f.evaluation("task-a", &body, Some(policy("execute")));
    assert_eq!(
        f.evaluate("task-a", input).await.unwrap_err().code(),
        tonic::Code::PermissionDenied
    );
    assert_eq!(f.private_models().await.len(), 4);
    assert_eq!(f.standard.received_requests().await.unwrap().len(), 1);
    Ok(())
}

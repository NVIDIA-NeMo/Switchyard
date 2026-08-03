# switchyard-libsy

Provider-neutral orchestration for multi-LLM optimization. A libsy
[`Algorithm`] decides which model targets to call, in what order, and how to
combine their results. It can use target-owned clients or hand each call back to
the host, allowing it to embed in proxies, gateways, and agent runtimes without
owning an HTTP stack.

## Setup

```toml
[dependencies]
async-trait = "0.1"
futures = "0.3"
switchyard-libsy = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
tokio = { version = "1", features = ["macros", "rt"] }
```

## Quick start

This complete `src/main.rs` routes a request with [`LlmTaskClassifier`]. The
first [`Algorithm::run`] lets libsy call each target's client. The second drives
[`Algorithm::run_stream`] so the host can perform or override each model call
itself.

```rust
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, LibsyError, LlmTarget, LlmTaskClassifier, Step, TaskClassifierConfig,
};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, Context, Decision, LlmClientError, LlmRequest,
    LlmResponse, Message, Request, Response, ResponseOutput, Role, RoutedLlmClient,
};

struct DemoClient;

#[async_trait]
impl RoutedLlmClient for DemoClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, LlmClientError> {
        let model = decision.selected_model();
        let text = if model == "judge" {
            r#"{
                "recommended_route": "efficient",
                "p_solve": 0.9,
                "confidence": 0.9,
                "abstain": false,
                "capability_boundary": "supported",
                "primary_rule": "SUP-1",
                "crux": "bounded task"
            }"#
            .to_string()
        } else {
            format!("answer from {model}")
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                model: Some(model.to_string()),
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text { text }],
                    stop_reason: None,
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> switchyard_libsy::Result<()> {
    let client = Arc::new(DemoClient);
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone()),
    };

    let router: Arc<dyn Algorithm> = Arc::new(LlmTaskClassifier::new(
        target("judge"),
        target("efficient"),
        target("capable"),
        TaskClassifierConfig {
            base_threshold: 0.5,
            ..TaskClassifierConfig::default()
        },
    )?);
    let request = Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Explain tail latency".to_string(),
                }],
            }],
            ..LlmRequest::default()
        },
        ..Request::default()
    };

    // `run` serves the judge and routed calls through their target clients.
    let (_, response) = router
        .clone()
        .run(Context::default(), request.clone())
        .await?;
    println!("run returned {}", response.selected_model().unwrap_or("unknown"));

    // `run_stream` exposes those calls so the host controls their transport.
    let stream = router.run_stream(Context::default(), request, None);
    tokio::pin!(stream);
    while let Some(step) = stream.next().await {
        match step? {
            Step::CallLlm(call) => {
                let routed = call.get_routed().clone();
                let target = routed.decision.selected_model().to_string();
                let result = client
                    .call(routed.ctx, routed.request, routed.decision)
                    .await
                    .map_err(|source| LibsyError::client_call(target, source));
                call.respond(result)?;
            }
            Step::Decision(decision) => {
                println!("run_stream chose {}", decision.selected_model());
            }
            Step::ReturnToAgent(response) => {
                println!(
                    "run_stream returned {}",
                    response.selected_model().unwrap_or("unknown")
                );
            }
        }
    }
    Ok(())
}
```

## Built-in algorithms

| Type | Purpose |
|---|---|
| [`Passthrough`] | Always call one configured target. |
| [`Random`] | Select among any number of targets using uniform or weighted routing. |
| [`LlmTaskClassifier`] | Ask a judge model to choose an efficient or capable target. |
| [`StageRouter`] | Route coding-agent turns from tool and progress signals, with an optional judge fallback. |

[`Noop`] is a test helper, not a production routing algorithm.

## How it fits together

[`LlmTarget`] pairs a semantic routing name with an optional
[`RoutedLlmClient`](switchyard_protocol::RoutedLlmClient). An [`Algorithm`]
selects targets and records [`Decision`](switchyard_protocol::Decision)s. Use
[`Algorithm::run`] with target-owned clients, or [`Algorithm::run_stream`] when
the host owns model transport. The provider-neutral [`Request`], [`Response`],
[`Usage`], and [`LlmResponse`] contracts come from `switchyard-protocol`.

[`Request`]: switchyard_protocol::Request
[`Response`]: switchyard_protocol::Response
[`Usage`]: switchyard_protocol::Usage
[`LlmResponse`]: switchyard_protocol::LlmResponse

## Examples

- [Client-backed execution](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent.rs)
- [Host-owned model calls](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/research_agent_core.rs)
- [Streaming responses](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/streaming_agent.rs)
- [Custom algorithm](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/crates/libsy/examples/ensemble.rs)

## License

Licensed under the Apache License, Version 2.0.

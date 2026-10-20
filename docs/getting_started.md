# Getting Started with Switchyard

Add routing to your existing Rust gateway or agent harness with **libsy**, Switchyard's core library. Your application keeps its model clients, credentials, and service.

Start with a local routing decision, then send the selected request through a model client. Python bindings are covered [separately](reference/host_integration.md#python-bindings).

## Install

You need Rust 1.96.1+ with Cargo and a native build toolchain. See [installation requirements](../INSTALLATION.md#requirements). These examples use the development API, pinned to one Git revision. The released `0.3.0` crates do not include `Step::CallDecision` and use a different `respond` API.

```bash
cargo new switchyard-demo
cd switchyard-demo
cargo add switchyard-libsy switchyard-protocol --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
cargo add futures@0.3
cargo add tokio@1 --features macros,rt
```

## Route a request

Replace `src/main.rs` with this complete example:

```rust
use std::{collections::HashMap, sync::Arc};
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_protocol::{Category, Request, text_request};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec!["Efficient".into()]),
        (Category::Capable, vec!["Capable".into()]),
        (Category::Any, vec!["Efficient".into(), "Capable".into()]),
    ])));
    let request = Request {
        llm_request: text_request(None, "Hello"),
        ..Default::default()
    };
    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::CallModel(_) => {
                return Err("An LLM client is required for this algorithm.".into());
            }
            Step::CallDecision(_) => {
                return Err("A decision client is required for this algorithm.".into());
            }
            Step::Done(outcome) => {
                println!("Selected: {}", outcome.selected_model_id()?);
            }
        }
    }
    Ok(())
}
```

```bash
cargo run
```

Expected output, after Cargo's build messages:

```text
Selected: Efficient
```

`"Efficient"` and `"Capable"` are model IDs supplied by your application. `Category::Efficient` and `Category::Capable` assign their routing roles; `Category::Any` lists the full model pool. This example makes no network calls and needs no API key.

The Stage router reads conversation and tool-result signals. With `EfficientFirst` and no tool history, it selects the efficient model. The `0.5` value is a routing confidence threshold, not a measured probability of task success. This is the same preset the TOML runner calls **Auto**.

The loop lists every `Step` variant. This local example has no clients, so it reports an error if an algorithm requests a call. The next example serves those calls through clients.

## Use libsy with your client

In an existing gateway, insert libsy before the model call:

```text
request -> Algorithm::run_stream -> selected model + returned request -> your client
```

Keep your existing client behind the `RoutedLlmClient` trait, or use Switchyard's optional HTTP client. The complete example below uses that client against OpenRouter. It converts the provider-neutral request to Chat Completions and converts the reply back.

Add the client crate and create the directory for this second example:

```bash
cargo add switchyard-llm-client --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
mkdir -p examples
```

Set `OPENROUTER_API_KEY` with access to `z-ai/glm-5.2` and `anthropic/claude-opus-4.8`, or replace both model IDs with models you can use. These models also appear in our [Terminal-Bench 2.1 results](benchmarks.md#terminal-bench-21).

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
```

Save as `examples/gateway.rs`:

```rust
use std::{collections::HashMap, sync::Arc, time::Duration};
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, LibsyError, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, SystemOneClient, TranslatingLlmClient,
};
use switchyard_protocol::{
    Category, LlmResponse, Request, RoutedDecisionClient, RoutedLlmClient,
    completion_text, text_request,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let efficient = "z-ai/glm-5.2";
    let capable = "anthropic/claude-opus-4.8";
    let backend = Backend::OpenAiChat(HttpBackendConfig {
        base_url: "https://openrouter.ai/api/v1".into(),
        api_key: Some(std::env::var("OPENROUTER_API_KEY")?),
        forward_auth: false,
        extra_headers: Default::default(),
        extra_body: Default::default(),
        omit_body_fields: Default::default(),
        reasoning_effort: None,
        max_retries: 2,
        failure_cooldown: Default::default(),
        timeout: None,
    });
    let client = TranslatingLlmClient::new(&[
        ModelConfig::new(efficient, backend.clone(), None),
        ModelConfig::new(capable, backend, None),
    ])?;
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec![efficient.into()]),
        (Category::Capable, vec![capable.into()]),
        (Category::Any, vec![efficient.into(), capable.into()]),
    ])));
    let request = Request {
        llm_request: text_request(None, "Say hello in one sentence."),
        ..Default::default()
    };
    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::CallModel(call) => {
                let request = call.request.clone();
                let model = call.models[0].clone();
                call.respond(async {
                    client.call(request).await
                        .map_err(|error| LibsyError::client_call(model, error))
                }).await?;
            }
            Step::CallDecision(call) => {
                let decision_client = SystemOneClient::new(
                    std::env::var("SYSTEM_ONE_ENDPOINT")?.parse()?,
                    std::env::var("SYSTEM_ONE_API_KEY")?,
                    Duration::from_secs(30),
                )?;
                let request = call.request.clone();
                let model = call.model.clone();
                call.respond(async {
                    decision_client.call(request).await
                        .map_err(|error| LibsyError::client_call(model, error))
                }).await?;
            }
            Step::Done(outcome) => {
                println!("Selected: {}", outcome.selected_model_id()?);
                let response = match outcome.response {
                    Some(response) => response,
                    None => client.call(outcome.request).await?,
                };
                if let LlmResponse::Agg(answer) = response.llm_response {
                    println!("Answer: {}", completion_text(&answer));
                }
            }
        }
    }
    Ok(())
}
```

This command sends a billable provider request:

```bash
cargo run --example gateway
```

The first line is `Selected: z-ai/glm-5.2`. The second starts with `Answer:` and contains the model's reply. A successful response verifies provider access as well as routing.

The host sends **`outcome.request`**, including routing-time rewrites, to the client. This example uses the first selected model; your gateway can apply its own fallback policy to later entries in `selected_model_ids`. The [HTTP client crate](../crates/libsy-llm-client/README.md) also provides a `run` helper that drives routing and handles terminal calls, retries, and candidate fallback.

The handlers cover all three steps:

| Step | Host action |
| --- | --- |
| `CallModel` | Send the routing-time LLM request through your client and pass its future to `respond`. |
| `CallDecision` | Send the typed decision request through a decision client and pass its future to `respond`. |
| `Done` | Reuse the returned response, or send the returned request to the selected model. |

The Stage preset above yields only `Done`. An LLM classifier can also yield `CallModel`; a decision-model classifier can yield `CallDecision`. The decision handler uses the optional System One client. For that handler, set `SYSTEM_ONE_ENDPOINT` to a full URL ending in `/v1/systemone` and `SYSTEM_ONE_API_KEY` to its key. Stage routing needs neither variable. Your application can instead supply its own `RoutedDecisionClient`.

This example serves calls one at a time. For algorithms that issue concurrent calls, use `libsy::drive` to poll handlers together and cancel pending work when routing ends. See the [host integration guide](reference/host_integration.md#concurrent-calls-and-cancellation).

## See routing change

Routing can change between agent turns. In `src/main.rs`, add these imports:

```rust
use switchyard_protocol::{ContentBlock, Message, Role, ToolCall, ToolResult};
```

Replace the `let request = ...;` statement with this conversation, which includes a failed tool call. Keep the router, model pool, and step loop unchanged:

```rust
let mut request = Request {
    llm_request: text_request(None, "Run the tests."),
    ..Default::default()
};
request.llm_request.messages.extend([
    Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolCall(ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"cargo test"}"#.parse()?,
        })],
    },
    Message {
        role: Role::User,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: "c1".into(),
            is_error: Some(true),
            content: vec![ContentBlock::Text {
                text: "MemoryError: cannot allocate memory\nKilled".into(),
            }],
        })],
    },
]);
```

Run `cargo run` again. This local check needs no credentials:

```text
Selected: Capable
```

The critical tool error forces the capable model. This checks model selection; it does not measure answer quality or savings. Evaluate your agent and model pool against a single-model baseline before relying on savings.

## Drive the algorithm

`Algorithm::run_stream` is the integration boundary. It yields routing steps, ending with `Step::Done` and a `RoutingOutcome`:

| Value | Host action |
| --- | --- |
| `selected_model_ids` | Use the first model; later entries are ordered fallbacks. |
| `request` | Use this request after routing-time rewrites. |
| `response` | Reuse this answer when the algorithm already produced one. Otherwise make the answer call. |

For streaming, concurrent work, cancellation, and metadata, see the [host integration guide](reference/host_integration.md). Both guides use the development API, including the `State` argument and asynchronous `respond`. Keep all Switchyard dependencies on the same Git revision.

## Next steps

- **Explore the Rust API:** [libsy crate](../crates/libsy/README.md) and [generated API reference](reference/rust_api.md).
- **Use a gateway integration:** [LiteLLM](../examples/litellm/README.md) or [NeMo Relay](integrations/nemo_relay.md). Check each guide's supported algorithms and versions.
- **Choose another algorithm:** [routing overview](routing_algorithms/overview.md).
- **Try the standalone proxy:** [local evaluation guide](operations/local_evaluation.md). The server is for demos, evaluation, and personal use, not a supported production path.
- **Use Python:** [Python bindings](reference/host_integration.md#python-bindings).
- **Use an agent directly:** [Claude Code example](../examples/claude/README.md), [pi](integrations/pi.md), or [Oh My Pi](integrations/oh_my_pi.md).

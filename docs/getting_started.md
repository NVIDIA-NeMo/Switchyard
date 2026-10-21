# Getting Started with Switchyard

Add routing to your existing Rust gateway or agent harness with **libsy**, Switchyard's core library. Your application keeps its model clients, credentials, and service.

Start with a local routing decision. Then add tool history or send the selected request through your own model client. Python bindings are covered [separately](reference/host_integration.md#python-bindings).

## Install

You need Rust 1.96.1+ with Cargo and a native build toolchain. See [installation requirements](../INSTALLATION.md#requirements). These examples use the development API, pinned to one Git revision. The released `0.3.0` crates do not include `Step::CallDecision` and use a different `respond` API.

```bash
cargo new switchyard-demo
cd switchyard-demo
# Keep libsy and its protocol types on the same API revision.
cargo add switchyard-libsy switchyard-protocol --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
# futures reads the step stream; Tokio runs the asynchronous routing tasks.
cargo add futures@0.3
cargo add tokio@1 --features macros,rt
```

## Route a request

Replace `src/main.rs` with this complete example. It selects a model locally, without an API key:

```rust
use std::{collections::HashMap, sync::Arc};
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_protocol::{Category, Request, text_request};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start with Efficient when tool history gives no reason to change models.
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    let efficient = "Efficient";
    let capable = "Capable";
    // Assign model roles; Any lists the full pool. These are local demo IDs.
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec![efficient.into()]),
        (Category::Capable, vec![capable.into()]),
        (Category::Any, vec![efficient.into(), capable.into()]),
    ])));
    // Leave the model unset so Stage can select it.
    let request = Request {
        llm_request: text_request(None, "Run the tests."),
        ..Default::default()
    };

    // State holds routing state; next() reads each result from the step stream.
    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::CallModel(_) => {
                // Other presets may ask your LLM client to serve a routing-time call.
                return Err("An LLM client is required for this algorithm.".into());
            }
            Step::CallDecision(_) => {
                // Typed decision questions need a decision-model client.
                return Err("A decision client is required for this algorithm.".into());
            }
            Step::Done(outcome) => {
                println!("Selected: {}", outcome.selected_model_id()?);
                // Stage selects the model; your client or gateway can send
                // outcome.request for the final answer, now or later.
            }
        }
    }
    Ok(())
}
```

```bash
# Model selection is local; this makes no provider requests.
cargo run
```

Expected output, after Cargo's build messages:

```text
Selected: Efficient
```

This Stage preset yields only `Done`, with `outcome.response` set to `None`. Your application owns the final model call. The loop also shows where other algorithms can ask for intermediate model calls.

`0.5` is a signal-confidence threshold, not a measured probability of task success. This preset is **Auto** in TOML. See the [Stage router guide](routing_algorithms/stage_router_routing.md) for signal rules, and evaluate your workload before relying on savings.

<details markdown="1">
<summary>Try the same prompt after a tool failure</summary>

Keep the same program. Insert this block immediately before `let mut steps = ...`, then run `cargo run` again:

```rust
use switchyard_protocol::{ContentBlock, Message, Role, ToolCall, ToolResult};

let mut request = request;
// Describe the agent's past activity; this example does not execute bash.
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
            // The result refers to the tool call above by its ID.
            tool_call_id: "c1".into(),
            is_error: Some(true),
            content: vec![ContentBlock::Text {
                text: "MemoryError: cannot allocate memory\nKilled".into(),
            }],
        })],
    },
]);
```

The critical failure makes Stage select **Capable** for the same user prompt:

```text
Selected: Capable
```

Both runs are local and need no API key. Remove the inserted block to return to the fresh request.

</details>

## Use libsy with your client

**You do not have to use Switchyard's HTTP client.** Keep your existing SDK, or create your own client. The same program can send Stage's selected request through [OpenRouter](https://openrouter.ai/docs/quickstart).

Add the dependencies:

```bash
# Translation converts neutral protocol types to and from provider JSON.
cargo add switchyard-translation --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
# reqwest handles HTTPS and JSON; serde_json represents the provider's reply.
cargo add reqwest@0.13 --no-default-features --features json,rustls
cargo add serde_json@1
```

For the complete program, replace `src/main.rs` with the [OpenRouter example](assets/quickstart/openrouter.rs). It includes the router above and the adapter below. To extend your existing file instead, apply the edits shown here.

Create an [OpenRouter key](https://openrouter.ai/settings/keys) with access to your chosen models:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
```

In `main`, replace the two demo model-ID assignments with:

```rust
// Use provider model IDs for the same Efficient and Capable roles.
let efficient = "z-ai/glm-5.2";
let capable = "anthropic/claude-opus-4.8";
// Reuse one HTTP client, with a 60-second timeout for each call.
let client = OpenRouterClient {
    http: reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60)).build()?,
    api_key: std::env::var("OPENROUTER_API_KEY")?,
};
```

Inside `match step?`, replace the `CallModel` and `Done` arms with these. Keep `CallDecision` unchanged; it needs a separate [decision client](reference/host_integration.md#serve-routing-time-calls).

```rust
Step::CallModel(call) => {
    // Pass the future to respond so libsy can cancel pending routing calls.
    let request = call.request.clone();
    call.respond(client.call(request)).await?;
}
Step::Done(outcome) => {
    println!("Selected: {}", outcome.selected_model_id()?);
    // Stage has selected the model. Your application now makes the final call.
    let response = client.call(outcome.request).await?;
    if let LlmResponse::Agg(answer) = response.llm_response {
        println!("Answer: {}", completion_text(&answer));
    }
}
```

Append the adapter to the same file:

<details markdown="1">
<summary>OpenRouter adapter: append to src/main.rs</summary>

This reuses `Request` from the imports above. It uses `reqwest` for HTTP and `switchyard-translation` for provider JSON. It returns a buffered response and requires no client trait implementation.

```rust
use switchyard_libsy::LibsyError;
use switchyard_protocol::{LlmResponse, Response, completion_text};
use switchyard_translation::{
    WireFormat, decode_aggregated_response, encode_request,
};

struct OpenRouterClient {
    // Reuse one HTTP client for connection pooling across model calls.
    http: reqwest::Client,
    // The application loads this from the environment instead of source code.
    api_key: String,
}

impl OpenRouterClient {
    async fn call(&self, request: Request) -> switchyard_libsy::Result<Response> {
        // Encode the neutral request as OpenRouter Chat Completions JSON.
        // libsy has already set its model to the selected target.
        let body = encode_request(&request.llm_request, WireFormat::OpenAiChat)
            .map_err(|error| LibsyError::external("encoding OpenRouter request", error))?;
        // Your client performs the provider call and supplies the API key.
        let response = self.http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send().await
            // Return HTTP failures before trying to decode an answer.
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| LibsyError::external("OpenRouter request", error))?;
        // Save headers before consuming the body, then read the full JSON reply.
        let upstream_headers = response.headers().clone();
        let body = response.json::<serde_json::Value>().await
            .map_err(|error| LibsyError::external("reading OpenRouter response", error))?;
        // Convert the provider's reply back to libsy's neutral response type.
        let answer = decode_aggregated_response(&body, WireFormat::OpenAiChat)
            .map_err(|error| LibsyError::external("decoding OpenRouter response", error))?;
        Ok(Response {
            // Agg holds a complete answer; a streaming client would return Stream.
            llm_response: LlmResponse::Agg(answer),
            // Keep any correlation metadata attached to the routed request.
            metadata: request.metadata,
            upstream_headers,
        })
    }
}
```

</details>

Run the program again. **This makes one billable model call**:

```bash
cargo run
```

For the fresh request, expect this shape; the answer text depends on the model:

```text
Selected: z-ai/glm-5.2
Answer: <model reply>
```

With the optional tool-failure history, the selected model is `anthropic/claude-opus-4.8`. For routing only, keep the original `Done` arm. Other algorithms may supply an answer in `outcome.response`; the [general integration pattern](../README.md#start-with-libsy) shows how to reuse it.

Your client returns `switchyard_protocol::Response` and adapts errors to `libsy::Result`. The [optional Switchyard HTTP client](../crates/libsy-llm-client/README.md) provides translation, streaming, retries, and fallback. Use [`libsy::drive`](reference/host_integration.md#concurrent-calls-and-cancellation) for concurrent algorithms; this loop serves calls sequentially.

## Next steps

- **Explore the Rust API:** [libsy crate](../crates/libsy/README.md) and [generated API reference](reference/rust_api.md).
- **Use a gateway integration:** [LiteLLM](../examples/litellm/README.md) or [NeMo Relay](integrations/nemo_relay.md). Check each guide's supported algorithms and versions.
- **Choose another algorithm:** [routing overview](routing_algorithms/overview.md).
- **Try the standalone proxy:** [local evaluation guide](operations/local_evaluation.md). The server is for demos, evaluation, and personal use, not a supported production path.
- **Use Python:** [Python bindings](reference/host_integration.md#python-bindings).
- **Use an agent directly:** [Claude Code example](../examples/claude/README.md), [pi](integrations/pi.md), or [Oh My Pi](integrations/oh_my_pi.md).

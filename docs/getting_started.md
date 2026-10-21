# Getting Started with Switchyard

Add routing to your existing Rust gateway or agent harness with **libsy**, Switchyard's core library. Your application keeps its model clients, credentials, and service.

Start with a local routing decision, then send the selected request through a model client. Python bindings are covered [separately](reference/host_integration.md#python-bindings).

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

Replace `src/main.rs` with this complete example. It routes two requests through the same Stage router: a fresh request and a recovery request with a failed tool call.

```rust
// Arc lets both requests share the router and model pool.
use std::{collections::HashMap, sync::Arc};
// StreamExt provides next() for reading the algorithm's steps.
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_protocol::{
    Category, ContentBlock, Message, Request, Role, ToolCall, ToolResult, text_request,
};

// run_stream starts asynchronous tasks, so this example needs a Tokio runtime.
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // EfficientFirst is the default when tool signals do not decide the tier.
    // 0.5 is the signal-confidence threshold. No LLM judge is configured here.
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    // Replace these demo IDs with provider model IDs when adding a client.
    let efficient = "Efficient";
    let capable = "Capable";
    // Efficient/Capable assign roles; Any lists the full model pool.
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec![efficient.into()]),
        (Category::Capable, vec![capable.into()]),
        (Category::Any, vec![efficient.into(), capable.into()]),
    ])));
    // None leaves the model unset so the router can select it.
    // This request has only a user prompt, with no previous tool activity.
    let fresh = Request {
        llm_request: text_request(None, "Run the tests."),
        ..Default::default()
    };

    // Copy the prompt and add the history an agent would send after a tool failure.
    // These messages describe past activity; this example does not execute bash.
    let mut recovery = fresh.clone();
    recovery.llm_request.messages.extend([
        // The assistant previously asked the bash tool to run the tests.
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"cargo test"}"#.parse()?,
            })],
        },
        // Match the result to that call with the same ID, c1.
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "c1".into(),
                is_error: Some(true),
                // This critical error makes Stage select Capable.
                content: vec![ContentBlock::Text {
                    text: "MemoryError: cannot allocate memory\nKilled".into(),
                }],
            })],
        },
    ]);

    for (label, request) in [("Fresh request", fresh), ("Tool failure", recovery)] {
        // These are independent examples, each with fresh routing state.
        let mut steps = router.clone().run_stream(request, models.clone(), State::default());
        while let Some(step) = steps.next().await {
            // Each item is Result<Step>; ? returns a routing error to the caller.
            match step? {
                Step::CallModel(_) => {
                    // A routing-time LLM call would need a client. This preset makes none.
                    return Err("An LLM client is required for this algorithm.".into());
                }
                Step::CallDecision(_) => {
                    // Typed decision questions would need a separate decision client.
                    return Err("A decision client is required for this algorithm.".into());
                }
                Step::Done(outcome) => {
                    // Stage returns a selected model and request, with response: None.
                    println!("{label} -> Selected: {}", outcome.selected_model_id()?);
                    // Stop here for routing only, or send outcome.request through your
                    // client for an answer. Your gateway can also make that call later.
                }
            }
        }
    }
    Ok(())
}
```

```bash
# Both model selections are local; this makes no provider requests.
cargo run
```

Expected output, after Cargo's build messages:

```text
Fresh request -> Selected: Efficient
Tool failure -> Selected: Capable
```

The demo IDs `"Efficient"` and `"Capable"` stand in for your models. The fresh request follows `EfficientFirst`; the same prompt with a critical tool failure selects **Capable**. Both decisions are local and need no API key. See the [Stage router guide](routing_algorithms/stage_router_routing.md) for its signal rules.

The `0.5` value is a routing confidence threshold, not a measured probability of task success. This preset is called **Auto** in TOML. Evaluate your agent and model pool against a single-model baseline before relying on savings.

The loop shows every `Step`. This Stage preset yields only `Done`; the other branches report missing clients. **Stage selects the model; it does not produce the answer.** Its `outcome.response` is `None`. Your application decides whether to call the selected model or pass the request back to its gateway. `outcome.request` includes the selected model and any routing-time rewrites; `selected_model_ids` lists ordered fallbacks.

Next, add a client to this same program.

## Use libsy with your client

**You do not have to use Switchyard's HTTP client.** Keep your existing SDK, or create your own client. Continue in `src/main.rs`, keeping the router, model pool, requests, and loop above.

For a working [OpenRouter](https://openrouter.ai/docs/quickstart) client, add these dependencies and append the adapter below to the same file:

```bash
# Translation converts neutral protocol types to and from provider JSON.
cargo add switchyard-translation --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
# reqwest handles HTTPS and JSON; serde_json represents the provider's reply.
cargo add reqwest@0.13 --no-default-features --features json,rustls
cargo add serde_json@1
```

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

Create a key in your [OpenRouter account](https://openrouter.ai/settings/keys) with access to these models, or substitute models you can use. They also appear in our [Terminal-Bench 2.1 results](benchmarks.md#terminal-bench-21).

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
```

In `main`, replace the two demo ID assignments with this block. The existing model pool uses the new IDs automatically:

```rust
// Use provider IDs for the same Efficient and Capable roles.
let efficient = "z-ai/glm-5.2";
let capable = "anthropic/claude-opus-4.8";
// Reuse one HTTP client; each provider call has a 60-second timeout.
let client = OpenRouterClient {
    http: reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60)).build()?,
    api_key: std::env::var("OPENROUTER_API_KEY")?,
};
```

Replace the existing `Step::CallModel` branch with:

```rust
Step::CallModel(call) => {
    // Pass the future to respond so libsy can cancel pending routing calls.
    let request = call.request.clone();
    call.respond(client.call(request)).await?;
}
```

Replace the existing `Step::Done` branch with this host-side final call:

```rust
Step::Done(outcome) => {
    println!("{label} -> Selected: {}", outcome.selected_model_id()?);
    // Stage has finished selecting. This call is made by your application.
    // Use the returned request, including its selected model and rewrites.
    let response = client.call(outcome.request).await?;
    if let LlmResponse::Agg(answer) = response.llm_response {
        println!("{label} -> Answer: {}", completion_text(&answer));
    }
}
```

For routing-only use, leave the original `Done` branch in place. Other algorithms may return an answer in `outcome.response`; see the [general integration pattern](../README.md#start-with-libsy) for response reuse. Keep `Step::CallDecision` as-is unless you supply a separate [decision client](reference/host_integration.md#serve-routing-time-calls).

Run the same file again. **This now makes two billable model calls**, one for each request:

```bash
cargo run
```

Expected shape; the answer text depends on the model:

```text
Fresh request -> Selected: z-ai/glm-5.2
Fresh request -> Answer: <model reply>
Tool failure -> Selected: anthropic/claude-opus-4.8
Tool failure -> Answer: <model reply>
```

Your own client can fill the same call sites by returning `switchyard_protocol::Response` and adapting errors to `libsy::Result`. The [optional Switchyard HTTP client](../crates/libsy-llm-client/README.md) provides translation, streaming, retries, and fallback. This loop serves calls sequentially; use [`libsy::drive`](reference/host_integration.md#concurrent-calls-and-cancellation) for concurrent algorithms.

## Next steps

- **Explore the Rust API:** [libsy crate](../crates/libsy/README.md) and [generated API reference](reference/rust_api.md).
- **Use a gateway integration:** [LiteLLM](../examples/litellm/README.md) or [NeMo Relay](integrations/nemo_relay.md). Check each guide's supported algorithms and versions.
- **Choose another algorithm:** [routing overview](routing_algorithms/overview.md).
- **Try the standalone proxy:** [local evaluation guide](operations/local_evaluation.md). The server is for demos, evaluation, and personal use, not a supported production path.
- **Use Python:** [Python bindings](reference/host_integration.md#python-bindings).
- **Use an agent directly:** [Claude Code example](../examples/claude/README.md), [pi](integrations/pi.md), or [Oh My Pi](integrations/oh_my_pi.md).

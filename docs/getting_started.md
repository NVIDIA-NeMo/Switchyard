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
    // These are local demo IDs. Efficient/Capable assign roles; Any lists the pool.
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec!["Efficient".into()]),
        (Category::Capable, vec!["Capable".into()]),
        (Category::Any, vec!["Efficient".into(), "Capable".into()]),
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
                    // Routing is complete; this local example prints the selected model.
                    println!("{label} -> Selected: {}", outcome.selected_model_id()?);
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

`"Efficient"` and `"Capable"` are model IDs supplied by your application. `Category::Efficient` and `Category::Capable` assign their routing roles; `Category::Any` lists the full model pool. This example makes no network calls and needs no API key.

The fresh request has no tool history, so `EfficientFirst` selects **Efficient**. The recovery request includes a critical tool error, so Stage selects **Capable**. The user prompt is the same in both cases; the tool history changes the routing decision. See the [Stage router guide](routing_algorithms/stage_router_routing.md) for its signal rules.

The `0.5` value is a routing confidence threshold, not a measured probability of task success. This is the same preset the TOML runner calls **Auto**. Evaluate your agent and model pool against a single-model baseline before relying on savings.

The loop lists every `Step` variant. This local example has no clients, so it reports an error if an algorithm requests a call. The next example supplies an LLM client and shows where to connect a decision client.

## Use libsy with your client

In an existing gateway, insert libsy before the model call:

```text
request -> Algorithm::run_stream -> selected model + returned request -> your client
```

**You do not have to use Switchyard's HTTP client.** Keep your existing SDK, or create your own client. `run_stream` hands model calls back to your application; you decide how to send them and return their responses.

This complete example creates an `OpenRouterClient` with `reqwest`. It sends a real Chat Completions request to [OpenRouter](https://openrouter.ai/docs/quickstart) and prints the answer. It uses `switchyard-translation` to convert between libsy's types and OpenRouter's JSON format. It does not depend on `switchyard-llm-client` or require a client trait implementation.

Add the dependencies and create the directory for this second example:

```bash
# Translation converts neutral protocol types to and from provider JSON.
cargo add switchyard-translation --git https://github.com/NVIDIA-NeMo/Switchyard.git --rev 5415db205c1f4db7e079731e7bf33a6609c67c8e
# reqwest handles HTTPS and JSON; serde_json represents the provider's reply.
cargo add reqwest@0.13 --no-default-features --features json,rustls
cargo add serde_json@1
mkdir -p examples
```

Create an API key in your [OpenRouter account](https://openrouter.ai/settings/keys) and make sure the account can pay for requests. Set `OPENROUTER_API_KEY` with access to `z-ai/glm-5.2` and `anthropic/claude-opus-4.8`, or replace both model IDs with models you can use. These models also appear in our [Terminal-Bench 2.1 results](benchmarks.md#terminal-bench-21).

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
use switchyard_protocol::{
    Category, LlmResponse, Request, Response, completion_text, text_request,
};
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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Replace these with OpenRouter model IDs available to your account.
    let efficient = "z-ai/glm-5.2";
    let capable = "anthropic/claude-opus-4.8";
    // One client serves either model. The timeout bounds each provider call.
    let client = OpenRouterClient {
        http: reqwest::Client::builder().timeout(Duration::from_secs(60)).build()?,
        api_key: std::env::var("OPENROUTER_API_KEY")?,
    };
    // Use the same Stage preset as the local example above.
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    // Assign the real provider IDs to the two routing roles.
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec![efficient.into()]),
        (Category::Capable, vec![capable.into()]),
        (Category::Any, vec![efficient.into(), capable.into()]),
    ])));
    // With no tool history, this preset selects the Efficient model.
    let request = Request {
        llm_request: text_request(None, "Say hello in one sentence."),
        ..Default::default()
    };
    // Routing produces work for the host; only client.call makes HTTP requests.
    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::CallModel(call) => {
                // A classifier or another algorithm may request a routing-time LLM call.
                // Give respond the future so libsy can cancel pending work.
                let request = call.request.clone();
                call.respond(client.call(request)).await?;
            }
            Step::CallDecision(_) => {
                // Typed decision questions require their own client.
                return Err("Configure a decision client before using a decision-model algorithm.".into());
            }
            Step::Done(outcome) => {
                println!("Selected: {}", outcome.selected_model_id()?);
                // Some algorithms already produced an answer: reuse it to avoid a second call.
                // Otherwise use outcome.request, which includes the model and any rewrites.
                let response = match outcome.response {
                    Some(response) => response,
                    None => client.call(outcome.request).await?,
                };
                if let LlmResponse::Agg(answer) = response.llm_response {
                    // Extract the text from the buffered protocol response.
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
# This calls OpenRouter and may incur a charge.
cargo run --example gateway
```

The first line is `Selected: z-ai/glm-5.2`. The second starts with `Answer:` and contains the model's reply. A successful response verifies provider access as well as routing.

The host sends **`outcome.request`**, including routing-time rewrites, to the custom client. The selected model is already set on that request. This client handles buffered responses. Your gateway can add streaming, retries, and fallback to later entries in `selected_model_ids`.

The loop lists all three steps:

| Step | Host action |
| --- | --- |
| `CallModel` | Send the routing-time LLM request through your client and pass its future to `respond`. |
| `CallDecision` | Requires a separate decision client. This example reports that missing setup explicitly. |
| `Done` | Reuse the returned response, or send the returned request to the selected model. |

The Stage preset above yields only `Done`. An LLM classifier can also yield `CallModel`, which the custom client handles. A decision-model algorithm yields `CallDecision`, whose typed request is different from Chat Completions. To use one, supply your own decision client and replace that branch with the [decision-call handler](reference/host_integration.md#serve-routing-time-calls).

You can replace `OpenRouterClient` with your existing client or SDK. Return `switchyard_protocol::Response` and adapt errors to `libsy::Result` when replying to a call. The [optional Switchyard HTTP client](../crates/libsy-llm-client/README.md) provides translation, streaming, and a `run` helper for terminal calls, retries, and fallback. Implement `RoutedLlmClient` if you want to use your own client with that helper; `run_stream` itself does not require it.

This example serves calls one at a time. For algorithms that issue concurrent calls, use `libsy::drive` to poll handlers together and cancel pending work when routing ends. See the [host integration guide](reference/host_integration.md#concurrent-calls-and-cancellation).

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

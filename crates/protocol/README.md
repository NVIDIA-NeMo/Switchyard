# switchyard-protocol

`switchyard-protocol` defines the provider-neutral Rust contract shared by
Switchyard routing, translation, HTTP clients, and host integrations.

It contains data and interoperability traits. It does not perform translation,
routing, or network calls.

## Add the crate

```toml
[dependencies]
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
serde_json = "1"
```

Applications using libsy should depend on both `switchyard-libsy` and
`switchyard-protocol`. `switchyard-translation` re-exports the conversation and
format modules plus common stream types, but not every protocol module.

## Main types

| Area | Types |
|---|---|
| Conversation | `LlmRequest`, `Message`, `InstructionBlock`, `ContentBlock` |
| Tools | `ToolDefinition`, `ToolChoice`, `ToolCall`, `ToolResult` |
| Response | `AggLlmResponse`, `ResponseOutput`, `Usage`, `StopReason` |
| Streaming | `LlmResponse`, `LlmResponseChunk`, `LlmResponseStream` |
| Envelope | `Context`, `Request`, `Response`, `Metadata` |
| Routing I/O | `Decision`, `RoutedLlmClient`, `LlmClientError` |
| Wire identity | `WireFormat`, `FormatId` |

Types are also available through their owning modules, such as
`switchyard_protocol::llm::LlmRequest`; the crate root re-exports them for
concise imports.

## Simple request

```rust
use switchyard_protocol::{ContentBlock, LlmRequest, Message, Role};

let request = LlmRequest {
    model: Some("provider/model".into()),
    messages: vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Explain tail latency".into(),
        }],
    }],
    ..LlmRequest::default()
};

assert_eq!(request.model.as_deref(), Some("provider/model"));
assert_eq!(request.messages.len(), 1);
```

## Detailed request

Construct the normalized request directly when routing needs instructions,
tools, generation controls, and correlation metadata:

```rust
use serde_json::json;
use switchyard_protocol::{
    ContentBlock, InstructionBlock, LlmRequest, Message, Metadata, OutputParams,
    Request, Role, SamplingParams, ToolChoice, ToolDefinition,
};

let request = Request {
    llm_request: LlmRequest {
        model: Some("provider/model".into()),
        instructions: vec![InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "Answer with concise operational guidance.".into(),
            }],
        }],
        messages: vec![Message::text(Role::User, "Why is p99 latency rising?")],
        tools: vec![ToolDefinition {
            name: "lookup_metric".into(),
            description: Some("Read one service metric".into()),
            parameters: json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
            strict: Some(true),
        }],
        tool_choice: Some(ToolChoice::Auto),
        sampling: SamplingParams {
            temperature: Some(0.2),
            ..SamplingParams::default()
        },
        output: OutputParams {
            max_output_tokens: Some(512),
            ..OutputParams::default()
        },
        stream: true,
        ..LlmRequest::default()
    },
    metadata: Some(Metadata {
        session_id: Some("session-42".into()),
        correlation_id: Some("request-7".into()),
        ..Metadata::default()
    }),
    ..Request::default()
};

assert_eq!(request.llm_request.tools[0].name, "lookup_metric");
```

## Buffered and streaming responses

`LlmResponse::Agg` contains a completed `AggLlmResponse`.
`LlmResponse::Stream` owns a single-consumption stream of `LlmResponseChunk`
events. `LlmResponse::into_agg` consumes a stream and surfaces decoding or
upstream errors.

`AggLlmResponse::into_stream` is a synthetic, lossy conversion. It emits text,
reasoning, and tool-call events but cannot represent every aggregate content
block. `ResponseAccumulator` similarly combines text and reasoning into one
assistant output and should not be used when multiple output indices must be
preserved.

## Extensions and preservation

`ProviderExtensions` stores provider fields that lack normalized equivalents.
Codecs use these values when translating to another format.

`PreservationMetadata` stores exact request and response bodies by `FormatId`.
With translation's default preservation policy, encoding back to a stored source
format returns that exact body instead of rebuilding it from normalized fields.
Code that mutates the IR must clear the corresponding preserved body or use a
translation policy with preservation disabled when those edits must be encoded.

`Request::raw_request` is separate host envelope data. The protocol does not
reconcile it with `LlmRequest::preservation`; hosts that populate both must
choose which source they treat as authoritative.

## Usage normalization

`Usage::input_tokens` contains non-cached input tokens. Cache-read and
cache-creation counts live in `InputCacheUsage`. Provider codecs may normalize
an aggregate provider input count by subtracting cache detail. `total_tokens`
is the provider-reported or codec-computed total and can therefore include
non-cached input, cache detail, and output tokens.

## Metadata

`Metadata` carries session, agent, task, trace, forwarding-header, and source
wire-format information. `Metadata::from_headers` normalizes the coding-agent,
NeMo Relay, and Dynamo headers recognized by Switchyard. Explicit
`x-switchyard-*` values take precedence.

## Reference

- [Generated Rust API reference](../../docs/reference/rust_api.md)
- [Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard)

## License

Licensed under the Apache License, Version 2.0.

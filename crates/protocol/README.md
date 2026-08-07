# switchyard-protocol

Provider-neutral request, response, streaming, routing, and metadata types
shared by Switchyard algorithms, clients, and translation codecs. This crate
defines contracts; it does not route, translate, or perform network calls.

## Setup

```toml
[dependencies]
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
serde_json = "1"
```

## Main types

| Area | Types |
|---|---|
| Conversation | [`LlmRequest`], [`Message`], [`InstructionBlock`], [`ContentBlock`] |
| Tools | [`ToolDefinition`], [`ToolChoice`], [`ToolCall`], [`ToolResult`] |
| Response | [`AggLlmResponse`], [`ResponseOutput`], [`Usage`], [`StopReason`] |
| Streaming | [`LlmResponse`], [`LlmResponseStream`], [`LlmResponseStreamEvent`], [`LlmResponseChunk`], [`ProviderStreamEvent`] |
| Envelope | [`Context`], [`Request`], [`Response`], [`Metadata`] |
| Routing I/O | [`Decision`], [`RoutedLlmClient`], [`LlmClientError`] |
| Wire identity | [`WireFormat`], [`FormatId`] |

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

Construct the normalized [`Request`] directly when routing needs instructions,
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

## Response forms

[`LlmResponse`] contains either a completed [`AggLlmResponse`] or a
single-consumption [`LlmResponseStream`] of [`LlmResponseStreamEvent`] values.
Each event carries provider-neutral [`LlmResponseChunk`] values and may retain
one opaque [`ProviderStreamEvent`] for same-format parsed-JSON-value replay. See
[`LlmResponse::into_agg`] for aggregation and [`PreservationMetadata`],
[`Usage`], and [`Metadata`] for the data retained around a response.

## License

Licensed under the Apache License, Version 2.0.

# switchyard-protocol

Provider-neutral request, response, streaming, routing, and metadata types
shared by Switchyard algorithms, clients, and translation codecs. This crate
defines contracts; it does not route, translate, or perform network calls.

## Setup

```toml
[dependencies]
switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git", tag = "v0.2.0" }
serde_json = "1"
```

## Main types

| Area | Types |
|---|---|
| Conversation | [`LlmRequest`], [`Message`], [`InstructionBlock`], [`ContentBlock`] |
| Tools | [`Tool`], [`ToolDefinition`], [`McpToolset`], [`McpToolConfig`], [`McpContent`], [`ToolChoice`], [`ToolCall`], [`ToolResult`] |
| Response | [`AggLlmResponse`], [`ResponseOutput`], [`Usage`], [`StopReason`] |
| Streaming | [`LlmResponse`], [`LlmResponseStream`], [`LlmResponseStreamEvent`], [`LlmResponseChunk`], [`ProviderStreamEvent`] |
| Envelope | [`Request`], [`Response`], [`Metadata`] |
| Routing I/O | [`RoutedLlmClient`], [`LlmClientError`] |
| Wire identity | [`WireFormat`], [`FormatId`] |

## Simple request

```rust
use switchyard_protocol::{LlmRequest, Message, Role};

let request = LlmRequest {
    model: Some("provider/model".into()),
    messages: vec![Message::text(Role::User, "Explain tail latency")],
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
            content: vec![ContentBlock::text("Answer with concise operational guidance.")],
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
        }.into()],
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

assert_eq!(request.llm_request.tools[0].function().map(|tool| tool.name.as_str()), Some("lookup_metric"));
```

## Responses request history

Decoded Responses requests retain native input items as ordered messages,
including inline system and developer messages. Top-level `instructions` remain
separate. `Message.responses` retains item fields such as `id` and assistant
`phase`, including whether `phase` was missing, null, or supplied with a value.
The item `id` stays separate from the `call_id` that pairs a tool call with its
output. Text blocks retain their own metadata, so moving or editing a block
keeps that metadata with the block.
This metadata survives serialization and reconstruction after saved request
bodies are cleared. Reconstruction uses the current role and content for edited
fields. Edited reasoning does not reuse the original encrypted content or item ID.

The translation crate provides `conversation(&LlmRequest)` for routing and APIs
that need grouped turns. The request owns the native history and its metadata.

Use `Message::text` and `ContentBlock::text` to create plain text without native
metadata. Rust struct literals for `Message` and `ContentBlock::Text` must now
include `responses`; set it to `None` for content created without that metadata.

## Response forms

[`LlmResponse`] contains either a completed [`AggLlmResponse`] or a
single-consumption [`LlmResponseStream`] of [`LlmResponseStreamEvent`] values.
Each event carries provider-neutral [`LlmResponseChunk`] values and may retain
one opaque [`ProviderStreamEvent`] for same-format parsed-JSON-value replay. See
[`LlmResponse::into_agg`] for aggregation and [`PreservationMetadata`],
[`Usage`], and [`Metadata`] for the data retained around a response.

## License

Licensed under the Apache License, Version 2.0.

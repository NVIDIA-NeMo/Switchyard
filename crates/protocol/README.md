# switchyard-protocol

`switchyard-protocol` defines the provider-neutral Rust contract shared by
Switchyard routing, translation, HTTP clients, and host integrations.

It contains data and interoperability traits. It does not perform translation,
routing, or network calls.

## Add the crate

```toml
[dependencies]
switchyard-protocol = "0.2"
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

## Construct a request

```rust
use switchyard_protocol::{LlmRequest, Message, Role};

let request = LlmRequest {
    model: Some("provider/model".into()),
    messages: vec![Message::text(Role::User, "Explain tail latency")],
    ..LlmRequest::default()
};
```

`text_request` and `text_response` construct common single-turn text shapes.
`prompt_text` and `completion_text` return lossy text views: they intentionally
omit tools, reasoning, media, instructions, and additional outputs.

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

- [Protocol API documentation](https://docs.rs/switchyard-protocol)
- [libsy API documentation](https://docs.rs/switchyard-libsy)
- [Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard)

## License

Licensed under the Apache License, Version 2.0.

# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, Anthropic Messages,
and Amazon Bedrock Converse request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## Bedrock Converse

Use `WireFormat::BedrockConverse` (serialized as `bedrock_converse`) with `TranslationEngine`
for buffered bodies. Text, system instructions, function tools and tool results, common inference
settings, inline images, stop reasons, and cache-token usage have neutral mappings. JSON tool results
become serialized JSON text when normalized; same-format preservation retains their original JSON.
Native controls such as guardrails and additional model fields survive same-format request encoding.
Cross-format projection diagnoses their loss and rejects it under strict loss policy.
Normalized encoding merges consecutive messages with the same Bedrock role in content order.
Tool history requires tool definitions. Disabling tools with history retains the required config
and reports a lossy conversion; strict loss policy rejects it. Without history, disabling tools
omits the config. Same-format preservation keeps the original request body.
Context-window exhaustion maps to the neutral token-limit stop reason so other formats mark the
response incomplete. Same-format preservation retains the original Bedrock stop reason.

ConverseStream codecs operate on the JSON union events after the host removes and validates AWS
EventStream framing. Feed those events to `decode_event_stream`, then use `encode_stream` for target
JSON events. `decode_stream` is for SSE bytes and rejects Bedrock. A Bedrock stream must include
`messageStart`, closed content blocks, `messageStop`, and terminal `metadata` with reported usage;
truncation, malformed events, or provider exceptions fail the stream. Encoding requires reported
input and output tokens; an absent total is derived from those counts and cache details.

Same-format preserved events replay unchanged, including native reasoning signatures. Aggregated
Bedrock reasoning preserves signature and redacted-content fragments. Translation to Anthropic
streams omits Bedrock signature fragments while retaining visible reasoning and answer text;
Bedrock redacted reasoning cannot be mapped and fails the stream. Encoding Bedrock rejects
foreign opaque reasoning stream details, including Anthropic signatures. Tool arguments
may precede their ID and name; parallel calls are serialized into Bedrock blocks. A tool block
cannot resume after other content closes it. Encoding Bedrock rejects unsupported foreign media and provider-specific built-in tool
history.

The host owns the model ID in the request URL, AWS credentials, SigV4 signing, regions, retries,
and binary EventStream framing. This crate adds no Bedrock HTTP client or server endpoint. Adding
the public `WireFormat` member requires downstream exhaustive matches to handle the new variant.

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.

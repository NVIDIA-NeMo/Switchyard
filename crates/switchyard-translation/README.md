# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## MCP support

Buffered translation supports MCP server definitions, tool listings, calls, and results between
Responses and Anthropic. Definitions with only an extra `server_description` can become Anthropic
connections: permissive policy omits the description with a diagnostic, while strict policy rejects
that loss. Unsupported authentication, approval settings, and tunnel connections stay provider-specific.

When translating listings to Anthropic, permissive policy drops listing IDs and tool annotations
but retains the tool names, descriptions, and schemas. Strict policy rejects these losses. Discovery
errors still cause the listing to be dropped under permissive policy, since Anthropic cannot represent
a failed discovery. On the return trip, Responses requires a listing ID. The default ID policy generates
a unique replacement; `DeterministicIdPolicy::Preserve` instead rejects or drops a listing without an ID.

Responses accepts one output string for an MCP call. Multiple Anthropic text blocks are joined with
blank lines and a diagnostic. Mixed results retain text and report omitted non-text content; results
with only unsupported content are dropped. Strict policy rejects these losses.

Failed text results become Responses `mcp_call` items with `mcp_tool_execution_error` errors and Anthropic results with
`is_error: true`. Responses protocol and HTTP error details, and structured execution-error content,
are retained as JSON text in Anthropic results under permissive policy, with a diagnostic for the
changed representation. Strict policy rejects that change. Calls with approval links or unfinished
status and approval request/response items remain provider-specific. Native Responses error items
retain their original JSON for same-format replay.

The convenience encoders used by the client and server log loss diagnostics because their return
values contain only the body.

Streaming translation supports MCP calls and results between Responses and Anthropic, including
argument deltas, completed-item snapshots, and conversion through a buffered aggregate. Anthropic
tool listings can also become Responses listings. Same-format streams retain their raw events.

Conversion waits for complete MCP arguments and, when targeting Responses, the matching result.
Later content waits behind that item to preserve order. Completed items are emitted before the
response ends. Malformed or truncated MCP blocks produce one terminal error.
The queue has no fixed memory limit. If a call's result never arrives, later content stays queued
until the stream ends; memory use grows with that content. At termination, the loss policy decides
whether to drop a completed call without a result or reject the conversion.

Streaming uses the same conversion rules and loss policies as buffered translation. Approval items
remain provider-specific, and Chat Completions cannot represent MCP content.
`StreamTranslationState::with_policy` selects rejection or conversion with diagnostics; the default
allows losses. Diagnostics remain available on the state, and convenience encoders log them.

Aggregation uses `ResponseAccumulator::try_finish()` to reject unfinished MCP blocks. The unchecked
`finish()` method is deprecated. Synthetic streams assign text and reasoning indexes per content
block across all outputs, so distinct blocks retain their identity. Responses `response.incomplete`
events also recover snapshot content and usage while retaining the token-limit stop reason.

The `encode_stream` helpers return error events, including Responses `response.failed`, as
`Err(LlmStreamError::Upstream(original_event))` and stop reading the source. The server forwards
that event as SSE, with credential redaction, and then closes the stream.

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.

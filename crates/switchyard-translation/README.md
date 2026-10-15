# switchyard-translation

Pure Rust translation between OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
request, response, and streaming formats.

The crate translates through provider-neutral LLM types from `switchyard-protocol` and does not
depend on provider SDKs, HTTP servers, Python, or FFI bindings.

## MCP support

Buffered translation supports MCP server definitions, tool listings, and successful text calls
between Responses and Anthropic. Responses calls with errors, approval links, or unfinished status
keep their provider JSON for same-format replay. Approval request and response items also stay
provider-specific.

When translating to Responses, an Anthropic MCP call with a failed or non-text result is rejected
under `LossyConversionPolicy::Reject`. With `AllowWithDiagnostics`, the call and result are dropped
and a diagnostic is returned. Other conversation content is kept. The convenience encoders used
by the client and server log these diagnostics because their return values contain only the body.

Streaming translation supports MCP calls and results between Responses and Anthropic, including
argument deltas, completed-item snapshots, and conversion through a buffered aggregate. Anthropic
tool listings can also become Responses listings. Same-format streams retain their raw events.

Conversion waits for complete MCP arguments and, when targeting Responses, the matching result.
Later content waits behind that item to preserve order. Completed items are emitted before the
response ends. Malformed or truncated MCP blocks produce one terminal error.
The queue has no fixed memory limit. If a call's result never arrives, later content stays queued
until the stream ends; memory use grows with that content. At termination, the loss policy decides
whether to drop a completed call without a result or reject the conversion.

Streaming uses the same conversion limits as buffered translation. Anthropic cannot retain
Responses listing IDs, errors, or tool annotations. Responses cannot retain Anthropic failed or
non-text results. Approval items remain provider-specific, and Chat Completions cannot represent
MCP content. `StreamTranslationState::with_policy` selects rejection or dropping unsupported
content with diagnostics; the default allows dropping. Diagnostics remain available on the state,
and convenience encoders log them.

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

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

Typed MCP streaming is not supported. The stream decode/encode API preserves raw MCP events for
same-format replay. Cross-format MCP stream translation ends with an unsupported-translation
error frame.

## License

Licensed under the Apache License, Version 2.0. See the
[Switchyard repository](https://github.com/NVIDIA-NeMo/Switchyard) for details.

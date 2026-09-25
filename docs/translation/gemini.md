# Buffered Gemini translation

`TranslationEngine::default()` registers `gemini_generate_content` for native
Gemini `generateContent` JSON bodies. It maps requests and responses through the
same neutral IR as OpenAI Chat, OpenAI Responses, and Anthropic Messages.

The codec covers conversation text, system instructions, function declarations,
calls and results, inline images/audio/video, audio/video file URIs, thinking
text, common sampling and JSON output settings, stop reasons, and token usage.
Gemini cache-read input tokens are separated from uncached input. Thought tokens
remain separate from visible output tokens.

The model belongs in the Gemini request URL. The codec does not add a `model`
property to native request JSON. Callers own URL selection. This does not add
HTTP endpoints, server configuration, streaming, or Relay wiring.

The default preservation policy retains the complete source body for exact
same-format replay. Clear preservation after changing the IR, or select
`PreservationPolicy::Disabled`, to encode the changed fields. Provider-only
controls and unsupported parts produce diagnostics during projection; strict
loss policy rejects them. Opaque thought signatures require exact replay and
are not copied from other providers. Native file parts without a portable MIME
representation remain provider-tagged unknown blocks. This codec does not fetch
media or execute tools.

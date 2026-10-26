# LLM-Related Protocols

An LLM request describes a conversation, the tools available to the model, and
generation settings. Switchyard represents these in
<a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>.
The same type works whether the next call goes to Anthropic Messages, OpenAI
Responses, or Chat Completions.

!!! info "Fields, API mappings, and conversion rules"

    - **[Request reference](../reference/protocols/llm_requests.md):** messages, tools, generation settings, and their provider API fields.
    - **[Response and stream reference](../reference/protocols/llm_responses.md):** output content, token usage, event types, and stream lifetime.
    - **[Translation reference](../reference/protocols/translation.md):** engine methods, preservation policies, diagnostics, and edits before encoding.

## Read a request

<a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>
separates instructions from conversation turns. A
<a href="../../rust/switchyard_protocol/llm/struct.Message.html"><code>Message</code></a>
contains a role and ordered
<a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html"><code>ContentBlock</code></a>
values. Content can include text, media, reasoning, tool calls, and tool results.

This example constructs the neutral request passed to an algorithm. It
uses the [quickstart's dependencies](../getting_started.md#install).

```rust
use switchyard_protocol::{ContentBlock, InstructionBlock, LlmRequest, Message, Request, Role};

let request = Request {
    llm_request: LlmRequest {
        instructions: vec![InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text { text: "Answer briefly.".into() }],
        }],
        messages: vec![Message::text(Role::User, "Explain tail latency.")],
        ..LlmRequest::default()
    },
    ..Request::default()
};
```

<a href="../../rust/switchyard_protocol/envelope/struct.Request.html"><code>Request</code></a> is the host-facing envelope. It adds optional correlation metadata
and <a href="../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.raw_request"><code>raw_request</code></a>, a host-owned JSON body. **Codecs use
<a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.preservation"><code>LlmRequest::preservation</code></a> for replay**, not <a href="../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.raw_request"><code>Request::raw_request</code></a>.

## Read a response

The host returns a <a href="../../rust/switchyard_protocol/envelope/struct.Response.html"><code>Response</code></a> envelope containing
<a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>.
<a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Agg"><code>LlmResponse::Agg</code></a> holds a completed <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html"><code>AggLlmResponse</code></a>. <a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Stream"><code>LlmResponse::Stream</code></a>
holds a single-consumption stream of
<a href="../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a>
values.
Each stream event exposes neutral chunks through <a href="../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html#method.normalized"><code>normalized()</code></a>. Chunks describe
message starts, text deltas, tool arguments, usage, and completion. An event may
also retain parsed provider JSON for same-format replay.

<a href="../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.input_tokens"><code>Usage::input_tokens</code></a> counts **non-cached input**. The codecs separate cache-read
and cache-write tokens from OpenAI's aggregate input counts. Missing counts stay
`None`. Call <a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html#method.into_agg"><code>LlmResponse::into_agg().await</code></a> when you need a buffered result; this
consumes the stream and aggregates its normalized chunks.

## Understand lossless conversion

**The protocol retains source data for lossless same-format round trips.** With
the default preservation policy, decoding a supported API body and encoding it
back to that API reuses its preserved JSON. The
[preservation reference](../reference/protocols/translation.md#preservation-and-edits)
explains exactly what is retained and when replay takes precedence.

**Conversion between different provider APIs can be lossy.** For example,
Anthropic has no separate developer role, and Chat Completions has no native
equivalent for every Responses built-in tool. Keeping the original data in the
protocol does not give the destination API those capabilities.

Translation reports known losses in <a href="../../rust/switchyard_translation/engine/struct.TranslationOutput.html#structfield.diagnostics"><code>diagnostics</code></a>.
The [translation policy reference](../reference/protocols/translation.md#policies)
explains how to allow or reject these conversions.

!!! warning "Encoding an edited request"

    A preserved same-format body takes precedence over normalized fields. If you
    change the protocol fields and want those edits encoded, remove that format's
    entry from <a href="../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html#structfield.requests"><code>preservation.requests</code></a>, or encode with <a href="../../rust/switchyard_translation/policy/enum.PreservationPolicy.html#variant.Disabled"><code>PreservationPolicy::Disabled</code></a>.
    The same rule applies to <a href="../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html#structfield.responses"><code>preservation.responses</code></a> for buffered responses.

See [stream preservation and aggregation](../reference/protocols/llm_responses.md#stream-lifetime-and-preservation)
for how replay and edits work on streamed responses.

## Use the translation crate

[`switchyard-translation`](../../crates/switchyard-translation/README.md) supplies
the LLM codecs without making network calls. Add it at the same API revision as
libsy and <a href="../../rust/switchyard_protocol/"><code>switchyard-protocol</code></a>, following the
[quickstart's translation setup](../getting_started.md#use-libsy-with-your-client).

This excerpt decodes an inbound Chat Completions body and encodes it as an
OpenAI Responses request. It can run inside your adapter; <a href="../../rust/switchyard_translation/engine/struct.TranslationOutput.html#structfield.body"><code>outbound.body</code></a> is the
JSON your client sends.

```rust
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

let engine = TranslationEngine::default();
let policy = TranslationPolicy::default();
let inbound = json!({
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain tail latency."}]
});

let decoded = engine.decode_request(WireFormat::OpenAiChat, &inbound, &policy)?;
let outbound = engine.encode_request(
    WireFormat::OpenAiResponses,
    &decoded.request,
    &policy,
)?;
// Check diagnostics from both operations before sending the encoded body.
```

During a libsy run, place <a href="../../rust/switchyard_translation/engine/struct.RequestIrOutput.html#structfield.request"><code>decoded.request</code></a> in <a href="../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.llm_request"><code>Request::llm_request</code></a>. Encode
the request from <a href="../../rust/switchyard_libsy/enum.Step.html#variant.CallModel"><code>Step::CallModel</code></a> for the selected backend. Decode its reply
with <a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_response"><code>decode_response</code></a>, or use <a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.decode_stream_event"><code>decode_stream_event</code></a> and <a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.encode_stream_event"><code>encode_stream_event</code></a>
with a separate <a href="../../rust/switchyard_translation/codecs/stream/struct.StreamTranslationState.html"><code>StreamTranslationState</code></a> per response stream. Return the neutral
<a href="../../rust/switchyard_protocol/envelope/struct.Response.html"><code>Response</code></a> through the call object, as shown in
[Mechanics of Libsy](../mechanics_of_libsy.md#stepcallmodel).

The engine also exposes <a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.translate_request"><code>translate_request</code></a> and <a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html#method.translate_response"><code>translate_response</code></a> for direct
API-to-API conversion. See the
<a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html">translation engine reference</a>
for signatures and results.

## Keep your own client or conversions

You can construct these protocol types from your existing SDK and convert them
back yourself. When handling <a href="../../rust/switchyard_libsy/enum.Step.html#variant.CallModel"><code>Step::CallModel</code></a>, pass <a href="../../rust/switchyard_libsy/struct.CallModel.html#method.respond"><code>call.respond</code></a> a future that
uses your client and returns <a href="../../rust/switchyard_libsy/type.Result.html"><code>libsy::Result&lt;Response&gt;</code></a>. Keep content order, tool
IDs, provider-only data, and streaming lifetime intact where your integration
needs them.

<a href="../../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>
is an optional shared interface used by Switchyard's clients and execution
helpers. Direct <a href="../../rust/switchyard_libsy/trait.Algorithm.html#method.run_stream"><code>run_stream</code></a> handling works with your own functions or clients;
it does not require implementing that trait.

If you want Switchyard to handle HTTP and translation, use
<a href="../../rust/switchyard_llm_client/client/struct.TranslatingLlmClient.html"><code>TranslatingLlmClient</code></a>
from <a href="../../rust/switchyard_llm_client/">the HTTP client crate</a>.
The [quickstart](../getting_started.md#use-libsy-with-your-client) shows how to
keep your own HTTP client. Typed decisions use the separate contract described
in [Decision Model-Related Protocols](decisions.md).

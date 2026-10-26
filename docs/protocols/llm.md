# LLM-Related Protocols

An LLM request describes a conversation, the tools available to the model, and
generation settings. Switchyard represents these in
<a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>.
The same type works whether the next call goes to Anthropic Messages, OpenAI
Responses, or Chat Completions.

## Protocol and API mappings

`LlmRequest` separates instructions from conversation turns. A turn is a
`Message` with a `Role` and an ordered list of `ContentBlock` values. That list
can mix text, reasoning, media, tool calls, and tool results. Responses use
`LlmResponse` for a buffered answer or a stream of events.

The table combines each field's meaning with the built-in codecs' API mappings.
Select a field or type to open its crate reference for the full definition.
See the provider references for
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
for their complete wire contracts.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.model"><code>LlmRequest.model: Option&lt;String&gt;</code></a> | Model addressed by the request. | `model` | `model` | `model` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.instructions"><code>LlmRequest.instructions</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.InstructionBlock.html"><code>Vec&lt;InstructionBlock&gt;</code></a> | System and developer instructions. | Top-level `system`; no separate developer role | `instructions` and system/developer messages in `input` | System/developer messages in `messages` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.messages"><code>LlmRequest.messages</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.Message.html"><code>Vec&lt;Message&gt;</code></a> | Ordered turns containing text, media, or tool activity. | `messages[].content`, including `tool_use` and `tool_result` | Messages, `function_call`, and `function_call_output` in `input` | `messages[]`, assistant `tool_calls`, and tool messages |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tools"><code>LlmRequest.tools</code></a>: <a href="../../rust/switchyard_protocol/llm/enum.Tool.html"><code>Vec&lt;Tool&gt;</code></a> | Tools available to the model; function definitions contain an argument schema. | `tools[]`, with `input_schema` for functions | `tools[]`, with `parameters` for functions | `tools[].function`, with `parameters` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tool_choice"><code>LlmRequest.tool_choice</code></a>: <a href="../../rust/switchyard_protocol/llm/enum.ToolChoice.html"><code>Option&lt;ToolChoice&gt;</code></a> | Whether the model may, must, or must not use tools. | `tool_choice` | `tool_choice` | `tool_choice` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.sampling"><code>LlmRequest.sampling</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.SamplingParams.html"><code>SamplingParams</code></a> | Sampling controls, where supported. | `temperature`, `top_p`, `top_k` | `temperature`, `top_p`; no standard `top_k` | `temperature`, `top_p`; no standard `top_k` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.output"><code>LlmRequest.output</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.OutputParams.html"><code>OutputParams</code></a> | Output budget, format, and schema enforcement. | `max_tokens`, `output_config.format` | `max_output_tokens`, `text.format` | `max_completion_tokens` or `max_tokens`, `response_format` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.reasoning"><code>LlmRequest.reasoning</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.ReasoningParams.html"><code>ReasoningParams</code></a> | Reasoning effort and provider-specific controls. | `thinking`, `output_config.effort` | `reasoning` | `reasoning_effort`; other controls are provider-specific |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.stream"><code>LlmRequest.stream: bool</code></a>; <a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Stream"><code>LlmResponse::Stream</code></a> | Request and receive incremental output. | `stream`; `message_*` and `content_block_*` events | `stream`; `response.*` events | `stream`; `chat.completion.chunk` objects |
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.outputs"><code>AggLlmResponse.outputs</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.ResponseOutput.html"><code>Vec&lt;ResponseOutput&gt;</code></a> | Completed output content and stop reasons. | Assistant `content` and `stop_reason` | `output`, `status`, and `incomplete_details` | `choices[].message` and `finish_reason` |
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.usage"><code>AggLlmResponse.usage</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a> | Available token counts, including cache and reasoning detail. | Input/output and separate cache counts | Input/output totals and token details | Prompt/completion totals and token details |
| <a href="../../rust/switchyard_protocol/llm/struct.ProviderExtensions.html"><code>extensions: ProviderExtensions</code></a>; <a href="../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html"><code>preservation: PreservationMetadata</code></a> | Provider-only fields and original JSON, retained on requests and buffered responses. | Source fields and original body | Source fields and original body | Source fields and original body |

`Message.content` holds
<a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html"><code>ContentBlock</code></a>
values. A
<a href="../../rust/switchyard_protocol/llm/struct.ToolCall.html"><code>ToolCall</code></a>
and its
<a href="../../rust/switchyard_protocol/llm/struct.ToolResult.html"><code>ToolResult</code></a>
stay paired by ID. Tool arguments are JSON values in the protocol; the codecs
handle the JSON-encoded strings used by the OpenAI formats.

Support varies by API and model. For example, Anthropic has no separate developer
role, while the OpenAI formats have no dedicated `ToolResult.is_error` field.
Preserving a field does not give another API an equivalent feature.

## Read a request

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

`Request` is the host-facing envelope. It adds optional correlation metadata
and `raw_request`, a host-owned JSON body. **Codecs use
`LlmRequest::preservation` for replay**, not `Request::raw_request`.

## Read a response

The host returns a `Response` envelope containing
<a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>.
`LlmResponse::Agg` holds a completed `AggLlmResponse`. `LlmResponse::Stream`
holds a single-consumption stream of
<a href="../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a>
values.
Each stream event exposes neutral chunks through `normalized()`. Chunks describe
message starts, text deltas, tool arguments, usage, and completion. An event may
also retain parsed provider JSON for same-format replay.

`Usage::input_tokens` counts **non-cached input**. The codecs separate cache-read
and cache-write tokens from OpenAI's aggregate input counts. Missing counts stay
`None`. Call `LlmResponse::into_agg().await` when you need a buffered result; this
consumes the stream and aggregates its normalized chunks.

## Understand lossless conversion

**The protocol retains source data for lossless same-format round trips.** With the
default preservation policy, decoding a supported API body and encoding it back
to that API reuses its preserved JSON. Fields without a neutral representation
can remain in provider extensions, unknown content blocks, or the preserved body.
This preserves parsed JSON values, not the original HTTP bytes or whitespace.
Format-specific input cleanup can happen before capture, such as removing Codex
internal compaction markers.

**Conversion between different provider APIs can be lossy.** For example,
Anthropic has no separate developer role, and Chat Completions has no native
equivalent for every Responses built-in tool. Keeping the original data in the
protocol does not give the destination API those capabilities.

Translation reports known losses in `diagnostics`. Its default policy allows
them with warnings. Set `LossyConversionPolicy::Reject` to reject known lossy
conversions. This policy checks supported codec rules; the destination model must
still support the requested features.

!!! warning "Encoding an edited request"

    A preserved same-format body takes precedence over normalized fields. If you
    change the protocol fields and want those edits encoded, remove that format's
    entry from `preservation.requests`, or encode with `PreservationPolicy::Disabled`.
    The same rule applies to `preservation.responses` for buffered responses.

For streams, pass the original events through when replay matters.
`replace_normalized()` clears stale provider replay data after an edit.
Aggregating a stream retains its normalized content, not its original event
sequence or all provider-only data.

## Use the translation crate

[`switchyard-translation`](../../crates/switchyard-translation/README.md) supplies
the LLM codecs without making network calls. Add it at the same API revision as
libsy and `switchyard-protocol`, following the
[quickstart's translation setup](../getting_started.md#use-libsy-with-your-client).

This excerpt decodes an inbound Chat Completions body and encodes it as an
OpenAI Responses request. It can run inside your adapter; `outbound.body` is the
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

During a libsy run, place `decoded.request` in `Request::llm_request`. Encode
the request from `Step::CallModel` for the selected backend. Decode its reply
with `decode_response`, or use `decode_stream_event` and `encode_stream_event`
with a separate `StreamTranslationState` per response stream. Return the neutral
`Response` through the call object, as shown in
[Mechanics of Libsy](../mechanics_of_libsy.md#stepcallmodel).

The engine also exposes `translate_request` and `translate_response` for direct
API-to-API conversion. See the
<a href="../../rust/switchyard_translation/engine/struct.TranslationEngine.html">translation engine reference</a>
for signatures and results.

## Keep your own client or conversions

You can construct these protocol types from your existing SDK and convert them
back yourself. When handling `Step::CallModel`, pass `call.respond` a future that
uses your client and returns `libsy::Result<Response>`. Keep content order, tool
IDs, provider-only data, and streaming lifetime intact where your integration
needs them.

<a href="../../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>
is an optional shared interface used by Switchyard's clients and execution
helpers. Direct `run_stream` handling works with your own functions or clients;
it does not require implementing that trait.

If you want Switchyard to handle HTTP and translation, use
[`TranslatingLlmClient` in `switchyard-llm-client`](../../crates/libsy-llm-client/README.md).
The [quickstart](../getting_started.md#use-libsy-with-your-client) shows how to
keep your own HTTP client. Typed decisions use the separate contract described
in [Decision Model-Related Protocols](decisions.md).

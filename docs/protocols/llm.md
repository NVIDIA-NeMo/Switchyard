# LLM-Related Protocols

An LLM request describes a conversation, the tools available to the model, and
generation settings. Switchyard represents these in
<a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html"><code>LlmRequest</code></a>.
The same type works whether the next call goes to Anthropic Messages, OpenAI
Responses, or Chat Completions.

## Read a request

`LlmRequest` separates instructions from conversation turns. A turn is a
`Message` with a `Role` and an ordered list of `ContentBlock` values. That list
can mix text, reasoning, media, tool calls, and tool results.
Select a protocol field or type in the tables to open its crate reference.

| Field | Rust type | Meaning |
|---|---|---|
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.model"><code>LlmRequest.model</code></a> | `Option<String>` | The addressed model. A host can replace it with the selected target. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.instructions"><code>LlmRequest.instructions</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.InstructionBlock.html"><code>Vec&lt;InstructionBlock&gt;</code></a> | System and developer instructions, each with a role and content blocks. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.messages"><code>LlmRequest.messages</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.Message.html"><code>Vec&lt;Message&gt;</code></a> | Ordered conversation turns. |
| <a href="../../rust/switchyard_protocol/llm/struct.Message.html#structfield.role"><code>Message.role</code></a> | <a href="../../rust/switchyard_protocol/llm/enum.Role.html"><code>Role</code></a> | Who produced the message. |
| <a href="../../rust/switchyard_protocol/llm/struct.Message.html#structfield.content"><code>Message.content</code></a> | <a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html"><code>Vec&lt;ContentBlock&gt;</code></a> | Ordered text, reasoning, media, tool calls, tool results, MCP activity, refusals, or unknown provider blocks. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tools"><code>LlmRequest.tools</code></a> | <a href="../../rust/switchyard_protocol/llm/enum.Tool.html"><code>Vec&lt;Tool&gt;</code></a> | Available function tools, MCP servers, or provider tools. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tool_choice"><code>LlmRequest.tool_choice</code></a> | <a href="../../rust/switchyard_protocol/llm/enum.ToolChoice.html"><code>Option&lt;ToolChoice&gt;</code></a> | How the model may select tools. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.sampling"><code>LlmRequest.sampling</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.SamplingParams.html"><code>SamplingParams</code></a> | Sampling settings such as temperature. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.output"><code>LlmRequest.output</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.OutputParams.html"><code>OutputParams</code></a> | Output budget, format, and schema enforcement. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.reasoning"><code>LlmRequest.reasoning</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.ReasoningParams.html"><code>ReasoningParams</code></a> | Reasoning effort and provider-specific controls. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.stream"><code>LlmRequest.stream</code></a> | `bool` | Whether a streamed response is requested. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.extensions"><code>LlmRequest.extensions</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.ProviderExtensions.html"><code>ProviderExtensions</code></a> | Fields without neutral equivalents. |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.preservation"><code>LlmRequest.preservation</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.PreservationMetadata.html"><code>PreservationMetadata</code></a> | Original payloads retained for round trips. |

For example, this constructs the neutral request passed to an algorithm. It
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

## Map the request to a provider API

These are the main mappings in the built-in codecs. Each column names fields
or items in that API; it does not imply every provider or model supports every
feature. Fields without a type prefix below belong to `LlmRequest`.
For the full wire contracts, see
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

### Conversation and tools

| Switchyard field and type | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.model"><code>model: Option&lt;String&gt;</code></a> | `model` | `model` | `model` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.instructions"><code>instructions: Vec&lt;InstructionBlock&gt;</code></a> | Top-level `system`; no separate developer role | Top-level `instructions` and system/developer messages in `input` | System/developer messages in `messages` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.messages"><code>messages: Vec&lt;Message&gt;</code></a> | `messages[]`, with text in `content` blocks | Messages in `input`, with text strings or `input_text` / `output_text` content | `messages[]`, with text or content parts in `content` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tools"><code>tools: Vec&lt;Tool&gt;</code></a> (function tools) | `tools[]`: `name`, `description`, `input_schema` | Function entries in `tools[]`: `name`, `description`, `parameters` | `tools[].function`: `name`, `description`, `parameters` |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tool_choice"><code>tool_choice: Option&lt;ToolChoice&gt;</code></a> | `auto`, `any`, `none`, or a named `tool` | `auto`, `required`, `none`, or a named `function` | `auto`, `required`, `none`, or a named `function` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.id"><code>ToolCall.id: String</code></a> | `tool_use.id` | `function_call.call_id` | `tool_calls[].id` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.name"><code>ToolCall.name: String</code></a> | `tool_use.name` | `function_call.name` | `tool_calls[].function.name` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.arguments"><code>ToolCall.arguments: Value</code></a> | `tool_use.input` | `function_call.arguments` | `tool_calls[].function.arguments` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.tool_call_id"><code>ToolResult.tool_call_id: String</code></a> | `tool_result.tool_use_id` | `function_call_output.call_id` | Tool message `tool_call_id` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.content"><code>ToolResult.content: Vec&lt;ContentBlock&gt;</code></a> | `tool_result.content` | `function_call_output.output` | Tool message `content` |
| <a href="../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.is_error"><code>ToolResult.is_error: Option&lt;bool&gt;</code></a> | `tool_result.is_error` | No dedicated error flag | No standard error flag |
| <a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html#variant.Image"><code>ContentBlock::Image { source }</code></a>, <a href="../../rust/switchyard_protocol/llm/enum.ImageSource.html"><code>source: ImageSource</code></a> | `image.source` | `input_image` | `image_url` content part |
| <a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html#variant.File"><code>ContentBlock::File { source }</code></a>, <a href="../../rust/switchyard_protocol/llm/enum.FileSource.html"><code>source: FileSource</code></a> | `document.source` | `input_file` | `file` content part |

Tool arguments are JSON values in the protocol. The OpenAI formats commonly carry
function arguments as JSON-encoded strings; the codecs handle that encoding.
Tool calls and their results stay paired by ID.

### Generation settings

| Switchyard field and type | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|
| <a href="../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.temperature"><code>sampling.temperature: Option&lt;f64&gt;</code></a> | `temperature` | `temperature` | `temperature` |
| <a href="../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.top_p"><code>sampling.top_p: Option&lt;f64&gt;</code></a> | `top_p` | `top_p` | `top_p` |
| <a href="../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.top_k"><code>sampling.top_k: Option&lt;i64&gt;</code></a> | `top_k` | No standard field | No standard field |
| <a href="../../rust/switchyard_protocol/llm/struct.OutputParams.html#structfield.max_output_tokens"><code>output.max_output_tokens: Option&lt;u64&gt;</code></a> | `max_tokens` | `max_output_tokens` | `max_completion_tokens`, or legacy `max_tokens` |
| <a href="../../rust/switchyard_protocol/llm/struct.OutputParams.html#structfield.response_format"><code>output.response_format: Option&lt;Value&gt;</code></a> | `output_config.format` | `text.format` | `response_format` |
| <a href="../../rust/switchyard_protocol/llm/struct.ReasoningParams.html#structfield.effort"><code>reasoning.effort: Option&lt;String&gt;</code></a> | `output_config.effort` | <a href="../../rust/switchyard_protocol/llm/struct.ReasoningParams.html#structfield.effort"><code>reasoning.effort</code></a> | `reasoning_effort` |
| <a href="../../rust/switchyard_protocol/llm/struct.ReasoningParams.html#structfield.raw"><code>reasoning.raw: Option&lt;Value&gt;</code></a> | `thinking` controls | Provider `reasoning` object | No standard reasoning-control object |
| <a href="../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.stream"><code>stream: bool</code></a> | `stream` | `stream` | `stream` |

**A missing feature is not just a different field name.** For example,
`ToolResult.is_error` has no dedicated field in the OpenAI formats, and their
standard sampling settings have no `top_k`. Anthropic's `system` field cannot
express separate system and developer authority. A provider-compatible server
may offer extra fields, but those extensions are specific to that server.

## Read a response

The host returns a `Response` envelope containing
<a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>.
`LlmResponse::Agg` holds a completed `AggLlmResponse`. `LlmResponse::Stream`
holds a single-consumption stream of `LlmResponseStreamEvent` values.

| Switchyard field and type | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.id"><code>AggLlmResponse.id: Option&lt;String&gt;</code></a> | `id` | `id` | `id` |
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.model"><code>AggLlmResponse.model: Option&lt;String&gt;</code></a> | `model` | `model` | `model` |
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.outputs"><code>AggLlmResponse.outputs: Vec&lt;ResponseOutput&gt;</code></a> | Assistant `content` blocks | Items in `output` | `choices[].message` |
| <a href="../../rust/switchyard_protocol/llm/struct.ResponseOutput.html#structfield.content"><code>ResponseOutput.content: Vec&lt;ContentBlock&gt;</code></a> | `text`, `tool_use`, and other blocks | `output_text` content, `function_call` items, and other output | Message `content` and `tool_calls` |
| <a href="../../rust/switchyard_protocol/llm/enum.ContentBlock.html#variant.Reasoning"><code>ContentBlock::Reasoning { text, signature, details }</code></a>: `String`, `Option<String>`, `Vec<Value>` | `thinking`, signature, and opaque reasoning data | Reasoning summaries and encrypted content | Provider-specific reasoning fields; no standard equivalent for all reasoning data |
| <a href="../../rust/switchyard_protocol/llm/struct.ResponseOutput.html#structfield.stop_reason"><code>ResponseOutput.stop_reason: Option&lt;StopReason&gt;</code></a> | `stop_reason` | Response `status` and `incomplete_details` | `choices[].finish_reason` |
| <a href="../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.usage"><code>AggLlmResponse.usage: Usage</code></a> | Input/output counts and separate cache counts | Input/output totals and token details | Prompt/completion totals and token details |
| <a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Stream"><code>LlmResponse::Stream(LlmResponseStream)</code></a>; items are <a href="../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a> | `message_*` and `content_block_*` events | `response.*` events | `chat.completion.chunk` objects |

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
    change the protocol fields and want those edits encoded, remove that format's entry from
    `preservation.requests`, or encode with `PreservationPolicy::Disabled`.
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

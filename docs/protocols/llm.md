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

| Component | Rust shape | Meaning |
|---|---|---|
| Model | `model: Option<String>` | The addressed model. A host can replace it with the selected target. |
| Instructions | `Vec<InstructionBlock>` | System and developer instructions, each with a role and content blocks. |
| Conversation | `Vec<Message>` | Ordered turns. Each message carries its role and content. |
| Content | `ContentBlock` | Text, reasoning, image, audio, video, file, tool call, tool result, MCP activity, refusal, or an unknown provider block. |
| Tool catalog | `Vec<Tool>` and `Option<ToolChoice>` | Available function tools, MCP servers, or provider tools, and how the model may select them. |
| Generation settings | `SamplingParams`, `OutputParams`, `ReasoningParams` | Sampling, output budget and format, schema enforcement, and reasoning controls. |
| Streaming | `stream: bool` | Whether a streamed response is requested. |
| Provider data | `ProviderExtensions`, `PreservationMetadata` | Fields without neutral equivalents and original payloads retained for round trips. |

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
feature. For the full wire contracts, see
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

| Switchyard IR | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|
| `model` | `model` | `model` | `model` |
| `instructions` | Top-level `system`; developer authority has no separate equivalent | Top-level `instructions` and system/developer messages in `input` | System/developer messages in `messages` |
| `messages` and text blocks | `messages[].content` text blocks | Messages in `input`, with `input_text` or `output_text` | `messages[].content`, as text or content parts |
| `ToolDefinition` | `tools[]`: `name`, `description`, `input_schema` | Function entries in `tools[]`: `name`, `description`, `parameters` | `tools[].function`: `name`, `description`, `parameters` |
| `ToolChoice` | `auto`, `any`, `none`, or a named `tool` | `auto`, `required`, `none`, or a named `function` | `auto`, `required`, `none`, or a named `function` |
| `ToolCall { id, name, arguments }` | Assistant `tool_use`: `id`, `name`, `input` | `function_call`: `call_id`, `name`, `arguments` | Assistant `tool_calls[]`: `id`, `function.name`, `function.arguments` |
| `ToolResult { tool_call_id, content, is_error }` | User `tool_result`: `tool_use_id`, `content`, `is_error` | `function_call_output`: `call_id`, `output`; no direct error flag | Tool message: `tool_call_id`, `content`; no standard error flag |
| Image and file blocks | `image` and `document` blocks with `source` | `input_image` and `input_file` | `image_url` and `file` content parts |
| `sampling` | `temperature`, `top_p`, `top_k` | `temperature`, `top_p` | `temperature`, `top_p` |
| `output.max_output_tokens` | `max_tokens` | `max_output_tokens` | `max_completion_tokens`, or legacy `max_tokens` |
| `output.response_format` | `output_config.format` | `text.format` | `response_format` |
| `reasoning` | `thinking` and `output_config.effort` | `reasoning` | `reasoning_effort` and provider-specific extensions |
| `stream` | `stream` | `stream` | `stream` |

Tool arguments are JSON values in the IR. The OpenAI formats commonly carry
function arguments as JSON-encoded strings; the codecs handle that encoding.
Tool calls and their results stay paired by ID.

## Read a response

The host returns a `Response` envelope containing
<a href="../../rust/switchyard_protocol/stream/enum.LlmResponse.html"><code>LlmResponse</code></a>.
`LlmResponse::Agg` holds a completed `AggLlmResponse`. `LlmResponse::Stream`
holds a single-consumption stream of `LlmResponseStreamEvent` values.

| Switchyard IR | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|
| `AggLlmResponse.id`, `model` | `id`, `model` | `id`, `model` | `id`, `model` |
| `outputs: Vec<ResponseOutput>` | Assistant `content` blocks | Items in `output` | `choices[].message` |
| Text and tool content | `text` and `tool_use` blocks | `output_text` content and `function_call` items | Message `content` and `tool_calls` |
| `ContentBlock::Reasoning` | `thinking`, signature, and opaque reasoning data | Reasoning summaries and encrypted content | Provider-specific reasoning fields |
| `ResponseOutput.stop_reason` | `stop_reason` | Response `status` and `incomplete_details` | `choices[].finish_reason` |
| `Usage` | Input/output counts and separate cache counts | Input/output totals and token details | Prompt/completion totals and token details |
| Stream event | `message_*` and `content_block_*` events | `response.*` events | `chat.completion.chunk` objects |

Each stream event exposes neutral chunks through `normalized()`. Chunks describe
message starts, text deltas, tool arguments, usage, and completion. An event may
also retain parsed provider JSON for same-format replay.

`Usage::input_tokens` counts **non-cached input**. The codecs separate cache-read
and cache-write tokens from OpenAI's aggregate input counts. Missing counts stay
`None`. Call `LlmResponse::into_agg().await` when you need a buffered result; this
consumes the stream and aggregates its normalized chunks.

## Understand lossless conversion

**The IR retains source data for lossless same-format round trips.** With the
default preservation policy, decoding a supported API body and encoding it back
to that API reuses its preserved JSON. Fields without a neutral representation
can remain in provider extensions, unknown content blocks, or the preserved body.
This preserves parsed JSON values, not the original HTTP bytes or whitespace.
Format-specific input cleanup can happen before capture, such as removing Codex
internal compaction markers.

**Conversion between different provider APIs can be lossy.** For example,
Anthropic has no separate developer role, and Chat Completions has no native
equivalent for every Responses built-in tool. Keeping the original data in the
IR does not give the destination API those capabilities.

Translation reports known losses in `diagnostics`. Its default policy allows
them with warnings. Set `LossyConversionPolicy::Reject` to reject known lossy
conversions. This policy checks supported codec rules; the destination model must
still support the requested features.

!!! warning "Encoding an edited request"

    A preserved same-format body takes precedence over normalized fields. If you
    change the IR and want those edits encoded, remove that format's entry from
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
back yourself. Keep content order, tool IDs, provider-only data, and streaming
lifetime intact where your integration needs them. Implement
<a href="../../rust/switchyard_protocol/client/trait.RoutedLlmClient.html"><code>RoutedLlmClient</code></a>
to expose a reusable client to libsy, or serve calls directly from `run_stream`.

If you want Switchyard to handle HTTP and translation, use
[`TranslatingLlmClient` in `switchyard-llm-client`](../../crates/libsy-llm-client/README.md).
The [quickstart](../getting_started.md#use-libsy-with-your-client) shows how to
keep your own HTTP client. Typed decisions use the separate contract described
in [Decision Model-Related Protocols](decisions.md).

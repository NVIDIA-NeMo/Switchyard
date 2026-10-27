# LLM Responses and Streams

For the integration flow, read **[LLM-Related Protocols](../../protocols/llm.md)**.
For the input side, use **[LLM Requests](llm_requests.md)**.

## Response forms

<a href="../../../rust/switchyard_protocol/envelope/struct.Response.html"><code>Response</code></a> is the host-facing envelope. Its `llm_response` is either
<a href="../../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Agg"><code>LlmResponse::Agg</code></a>, containing an <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html"><code>AggLlmResponse</code></a>, or <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponse.html#variant.Stream"><code>LlmResponse::Stream</code></a>,
containing a live <a href="../../../rust/switchyard_protocol/stream/type.LlmResponseStream.html"><code>LlmResponseStream</code></a>. The stream has one consumer.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.id"><code>AggLlmResponse.id: Option&lt;String&gt;</code></a> | Provider response identifier, if available. | `id` | `id` | `id` |
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.model"><code>AggLlmResponse.model: Option&lt;String&gt;</code></a> | Provider-reported model name. | `model` | `model` | `model` |
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.outputs"><code>AggLlmResponse.outputs: Vec&lt;ResponseOutput&gt;</code></a> | Completed output items. | Assistant `content` blocks | Items in `output` | `choices[].message` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ResponseOutput.html#structfield.content"><code>ResponseOutput.content: Vec&lt;ContentBlock&gt;</code></a> | Ordered text, reasoning, tool, and other output content. | `text`, `thinking`, `tool_use`, and other blocks | Message content, reasoning, function calls, and other items | Message `content`, `tool_calls`, and provider-specific reasoning fields |
| <a href="../../../rust/switchyard_protocol/llm/struct.ResponseOutput.html#structfield.stop_reason"><code>ResponseOutput.stop_reason: Option&lt;StopReason&gt;</code></a> | Why this output stopped. | `stop_reason` | Response `status` and `incomplete_details` | `choices[].finish_reason` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ResponseOutput.html#structfield.url_citations"><code>ResponseOutput.url_citations: Vec&lt;UrlCitation&gt;</code></a> | Cited URLs with text offsets. | Not normalized by the current codec | `output_text.annotations` | Message `annotations` when supplied |
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.usage"><code>AggLlmResponse.usage: Usage</code></a> | Normalized usage counts and any provider-reported cost. | `usage` | `usage` | `usage` |
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.extensions"><code>AggLlmResponse.extensions: ProviderExtensions</code></a> | Provider-only response fields. | Source fields retained by the codec | Source fields retained by the codec | Source fields retained by the codec |
| <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html#structfield.preservation"><code>AggLlmResponse.preservation: PreservationMetadata</code></a> | Original JSON for same-format buffered replay. | Original parsed body | Original parsed body | Original parsed body |

The envelope also carries <a href="../../../rust/switchyard_protocol/envelope/struct.Response.html#structfield.metadata"><code>Response.metadata</code></a> and <a href="../../../rust/switchyard_protocol/envelope/struct.Response.html#structfield.upstream_headers"><code>Response.upstream_headers</code></a>.
<a href="../../../rust/switchyard_protocol/envelope/struct.Response.html#method.served_model"><code>Response.served_model()</code></a> reads the Switchyard target recorded in metadata.
<a href="../../../rust/switchyard_protocol/envelope/struct.Response.html#method.selected_model"><code>Response.selected_model()</code></a> reads the provider model from a buffered result and
returns no model for a live stream, since inspecting its events would consume it.

## Usage

<a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.input_tokens"><code>Usage.input_tokens</code></a> counts **non-cached input**. When an OpenAI-style provider
reports aggregate input counts, the codecs subtract reported cache-read and
cache-write counts. The cache details remain separate.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.input_tokens"><code>Usage.input_tokens: Option&lt;u64&gt;</code></a> | Non-cached input tokens. | `input_tokens` | `input_tokens` minus reported cache detail | `prompt_tokens` minus reported cache detail |
| <a href="../../../rust/switchyard_protocol/llm/struct.InputCacheUsage.html#structfield.cached_input_tokens"><code>InputCacheUsage.cached_input_tokens: Option&lt;u64&gt;</code></a> | Input read from cache. | `cache_read_input_tokens` | Input token details, when reported | Prompt token details, when reported |
| <a href="../../../rust/switchyard_protocol/llm/struct.InputCacheUsage.html#structfield.cache_creation_input_tokens"><code>InputCacheUsage.cache_creation_input_tokens: Option&lt;u64&gt;</code></a> | Input written to cache. | `cache_creation_input_tokens` | Input token details, when reported | Prompt token details, when reported |
| <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.output_tokens"><code>Usage.output_tokens: Option&lt;u64&gt;</code></a> | Output tokens, including reasoning when counted by the provider. | `output_tokens` | `output_tokens` | `completion_tokens` |
| <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.reasoning_tokens"><code>Usage.reasoning_tokens: Option&lt;u64&gt;</code></a> | Reasoning output reported separately. | Provider token details, when reported | `output_tokens_details.reasoning_tokens` | Completion token details, when reported |
| <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.total_tokens"><code>Usage.total_tokens: Option&lt;u64&gt;</code></a> | Provider-reported or codec-computed total. | Computed when needed | `total_tokens` or computed total | `total_tokens` or computed total |
| <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.provider_cost"><code>Usage.provider_cost: Option&lt;f64&gt;</code></a> | Provider-reported monetary charge in USD. | Only when reported | Only when reported | Only when reported |

Missing counts remain unknown. Do not turn an absent count into a measured zero
when adapting a response. <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html#structfield.cache"><code>Usage.cache</code></a> holds optional <a href="../../../rust/switchyard_protocol/llm/struct.InputCacheUsage.html"><code>InputCacheUsage</code></a> details.

## Stream events

Each item of <a href="../../../rust/switchyard_protocol/stream/type.LlmResponseStream.html"><code>LlmResponseStream</code></a> is a result containing <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a>.
An event contains zero or more neutral <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html"><code>LlmResponseChunk</code></a> values and may retain
one parsed provider event for same-format replay.

| Protocol variant | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.MessageStart"><code>LlmResponseChunk::MessageStart</code></a> | Establish or update response identity. | `message_start` | Response creation/start events | Initial model and response fields |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.TextDelta"><code>LlmResponseChunk::TextDelta</code></a> | Append text at an output index. | Text `content_block_delta` | Output text delta events | `choices[].delta.content` |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.ReasoningDelta"><code>LlmResponseChunk::ReasoningDelta</code></a> | Append reasoning text. | Thinking deltas | Reasoning summary deltas | Provider-specific reasoning deltas |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.ReasoningDetailsDelta"><code>LlmResponseChunk::ReasoningDetailsDelta</code></a> | Retain structured or encrypted reasoning detail. | Reasoning signatures and related detail | Encrypted reasoning items | Provider-specific structured reasoning fields |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.ToolCallDelta"><code>LlmResponseChunk::ToolCallDelta</code></a> | Build a tool call's ID, name, and argument string. | Tool start and input JSON deltas | Function-call item and argument events | Tool-call deltas |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.Usage"><code>LlmResponseChunk::Usage</code></a> | Update available token counts. | Message usage and deltas | Usage on response events | Usage chunks |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.MessageStop"><code>LlmResponseChunk::MessageStop</code></a> | End the message. | Message stop events | Response completion events | Finish reasons and terminal framing |
| <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.DecodeError"><code>LlmResponseChunk::DecodeError</code></a> / <a href="../../../rust/switchyard_protocol/stream/enum.LlmResponseChunk.html#variant.StreamError"><code>LlmResponseChunk::StreamError</code></a> | Report decoding or in-band upstream failure. | Codec/upstream error events | Codec/upstream error events | Codec/upstream error events |

These are semantic mappings. One provider event can produce several chunks,
and some events have no normalized chunks. Raw replay can still retain those
events. Algorithms read neutral chunks through <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html#method.normalized"><code>normalized()</code></a>.

## Stream lifetime and preservation

Pass original <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html"><code>LlmResponseStreamEvent</code></a> values through when exact same-format
parsed-JSON replay matters. <a href="../../../rust/switchyard_protocol/stream/struct.LlmResponseStreamEvent.html#method.replace_normalized"><code>replace_normalized()</code></a> returns an event with the new
chunks and clears its stale raw replay data.

<a href="../../../rust/switchyard_protocol/stream/enum.LlmResponse.html#method.into_agg"><code>LlmResponse::into_agg().await</code></a> consumes the stream and folds its normalized
chunks into an <a href="../../../rust/switchyard_protocol/llm/struct.AggLlmResponse.html"><code>AggLlmResponse</code></a>. Stream item failures, decode failures, and
in-band upstream errors become errors. Aggregation does not preserve the original
provider event sequence, MCP activity, unknown blocks, or all response extensions.
The current accumulator also folds output indices into one assistant output.

Dropping a stream stops local consumption. It cannot guarantee that the provider
stops generation or billing. See [concurrency and cancellation](../host_integration.md#concurrent-calls-and-cancellation)
for host-owned work, and **[Translation](translation.md#stream-translation)** for
decoding, encoding, and finishing provider streams.

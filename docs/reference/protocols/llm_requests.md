# LLM Requests

This reference covers request fields and the built-in provider mappings. For the
integration flow and a construction example, read **[LLM-Related Protocols](../../protocols/llm.md)**.
For the reply, continue to **[Responses and Streams](llm_responses.md)**.

## Request envelope

An algorithm receives <a href="../../../rust/switchyard_protocol/envelope/struct.Request.html"><code>Request</code></a>. Its `llm_request` contains the conversation;
the remaining envelope data belongs to the application.

| Field and type | Meaning |
|---|---|
| <a href="../../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.llm_request"><code>Request.llm_request: LlmRequest</code></a> | The conversation and settings used by the algorithm and model client. |
| <a href="../../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.metadata"><code>Request.metadata: Option&lt;Metadata&gt;</code></a> | Optional session, correlation, and routing metadata. |
| <a href="../../../rust/switchyard_protocol/envelope/struct.Request.html#structfield.raw_request"><code>Request.raw_request: Option&lt;Value&gt;</code></a> | An optional original body retained by the host. Translation codecs do not read this field. |

Codec replay uses <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.preservation"><code>LlmRequest.preservation</code></a>, separately from the envelope's raw
body. See **[preservation and edits](translation.md#preservation-and-edits)**
before changing a decoded request.

## Conversation fields

These mappings describe the built-in codecs. Supported features also depend on
the destination model. Select a Rust field or type for its full crate definition.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.model"><code>LlmRequest.model: Option&lt;String&gt;</code></a> | Addressed model; the host can replace it with the selected target. | `model` | `model` | `model` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.instructions"><code>LlmRequest.instructions: Vec&lt;InstructionBlock&gt;</code></a> | System and developer instructions. | Top-level `system`; no distinct developer authority | `instructions` and system/developer messages in `input` | System/developer messages in `messages` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.messages"><code>LlmRequest.messages: Vec&lt;Message&gt;</code></a> | Ordered turns, each with a role and content blocks. | `messages[]` | Messages and tool items in `input` | `messages[]` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tools"><code>LlmRequest.tools: Vec&lt;Tool&gt;</code></a> | Available functions, MCP toolsets, or retained provider definitions. | `tools[]` | `tools[]` | `tools[]`; function definitions are nested in `function` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.tool_choice"><code>LlmRequest.tool_choice: Option&lt;ToolChoice&gt;</code></a> | Let the model choose, require a call, forbid calls, or select a named tool. | `auto`, `any`, `none`, or `tool` | `auto`, `required`, `none`, or `function` | `auto`, `required`, `none`, or `function` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.stream"><code>LlmRequest.stream: bool</code></a> | Request incremental output. | `stream` | `stream` | `stream` |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.extensions"><code>LlmRequest.extensions: ProviderExtensions</code></a> | Provider fields without a first-class neutral field. | Source fields retained by the codec | Source fields retained by the codec | Source fields retained by the codec |
| <a href="../../../rust/switchyard_protocol/llm/struct.LlmRequest.html#structfield.preservation"><code>LlmRequest.preservation: PreservationMetadata</code></a> | Original JSON used for same-format replay. | Original parsed body | Original parsed body | Original parsed body |

### Messages and tool content

<a href="../../../rust/switchyard_protocol/llm/struct.Message.html#structfield.role"><code>Message.role</code></a> is a <a href="../../../rust/switchyard_protocol/llm/enum.Role.html"><code>Role</code></a>; <a href="../../../rust/switchyard_protocol/llm/struct.Message.html#structfield.content"><code>Message.content</code></a> is an ordered <a href="../../../rust/switchyard_protocol/llm/enum.ContentBlock.html"><code>Vec&lt;ContentBlock&gt;</code></a>.
The content variants cover text, reasoning, media, tools, MCP activity, refusals,
and unknown provider blocks. An unknown block retains its source format and raw
JSON. Its retention does not make it usable in a different API.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolDefinition.html#structfield.parameters"><code>ToolDefinition.parameters: Value</code></a> | JSON Schema for function arguments. | `input_schema` | Function `parameters` | `tools[].function.parameters` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.id"><code>ToolCall.id: String</code></a> | Identifier that pairs a call with its result. | `tool_use.id` | `function_call.call_id` | `tool_calls[].id` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.name"><code>ToolCall.name: String</code></a> | Function name. | `tool_use.name` | `function_call.name` | `tool_calls[].function.name` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolCall.html#structfield.arguments"><code>ToolCall.arguments: Value</code></a> | Parsed argument value. | JSON-valued `input` | `arguments`, commonly a JSON-encoded string | `function.arguments`, commonly a JSON-encoded string |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.tool_call_id"><code>ToolResult.tool_call_id: String</code></a> | Identifier of the call being answered. | `tool_result.tool_use_id` | `function_call_output.call_id` | Tool message `tool_call_id` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.content"><code>ToolResult.content: Vec&lt;ContentBlock&gt;</code></a> | Ordered tool output. | `tool_result.content` | `function_call_output.output` | Tool message `content` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ToolResult.html#structfield.is_error"><code>ToolResult.is_error: Option&lt;bool&gt;</code></a> | Whether tool execution failed, if reported. | `tool_result.is_error` | No dedicated error flag | No standard error flag |

Tool definitions describe available work. A <a href="../../../rust/switchyard_protocol/llm/struct.ToolCall.html"><code>ToolCall</code></a> records a model's request
to execute that work; it does not execute the tool. Keep call IDs and content
order intact when writing an adapter.

<a href="../../../rust/switchyard_protocol/llm/enum.Tool.html#variant.Provider"><code>Tool::Provider</code></a> retains an unsupported tool definition for its source format.
Built-in provider tools and reasoning signatures may have no valid counterpart
in another API. Consult the [translation limits](translation.md#conversion-limits)
instead of assuming a successful parse implies cross-provider support.

## Generation settings

The request groups these fields in <a href="../../../rust/switchyard_protocol/llm/struct.SamplingParams.html"><code>SamplingParams</code></a>, <a href="../../../rust/switchyard_protocol/llm/struct.OutputParams.html"><code>OutputParams</code></a>, and
<a href="../../../rust/switchyard_protocol/llm/struct.ReasoningParams.html"><code>ReasoningParams</code></a>. Optional values remain unspecified when absent; individual
providers apply their own defaults and accepted ranges.

| Protocol field and type | Meaning | Anthropic Messages | OpenAI Responses | Chat Completions |
|---|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.temperature"><code>SamplingParams.temperature: Option&lt;f64&gt;</code></a> | Sampling temperature. | `temperature` | `temperature` | `temperature` |
| <a href="../../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.top_p"><code>SamplingParams.top_p: Option&lt;f64&gt;</code></a> | Nucleus-sampling probability mass. | `top_p` | `top_p` | `top_p` |
| <a href="../../../rust/switchyard_protocol/llm/struct.SamplingParams.html#structfield.top_k"><code>SamplingParams.top_k: Option&lt;i64&gt;</code></a> | Maximum candidate-token count. | `top_k` | No standard field | No standard field |
| <a href="../../../rust/switchyard_protocol/llm/struct.OutputParams.html#structfield.max_output_tokens"><code>OutputParams.max_output_tokens: Option&lt;u64&gt;</code></a> | Output token budget. | `max_tokens` | `max_output_tokens` | `max_completion_tokens`, or legacy `max_tokens` |
| <a href="../../../rust/switchyard_protocol/llm/struct.OutputParams.html#structfield.response_format"><code>OutputParams.response_format: Option&lt;Value&gt;</code></a> | Requested structured-output shape. | `output_config.format` | `text.format` | `response_format` |
| <a href="../../../rust/switchyard_protocol/llm/struct.OutputParams.html#structfield.is_schema_enforced"><code>OutputParams.is_schema_enforced: Option&lt;bool&gt;</code></a> | Override whether the provider must enforce the schema; absence retains native behavior. | Schema-backed output configuration | Schema strictness in `text.format` | Schema strictness in `response_format` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ReasoningParams.html#structfield.effort"><code>ReasoningParams.effort: Option&lt;String&gt;</code></a> | Requested reasoning effort. | `output_config.effort` | `reasoning.effort` | `reasoning_effort` |
| <a href="../../../rust/switchyard_protocol/llm/struct.ReasoningParams.html#structfield.raw"><code>ReasoningParams.raw: Option&lt;Value&gt;</code></a> | Provider-specific reasoning configuration. | `thinking` controls | `reasoning` object | No standard equivalent for the whole object |

Preserved same-format bodies take precedence over edits to these settings,
including schema enforcement. See **[encoding an edited request](translation.md#preservation-and-edits)**.

## Provider contracts

For complete provider schemas, see [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
For response fields and event ownership, continue to **[Responses and Streams](llm_responses.md)**.

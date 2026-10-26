# Switchyard Protocols

[Mechanics of Libsy](../mechanics_of_libsy.md) follows the work exchanged through
`run_stream`. This section explains the data inside that work: an LLM conversation
or a set of typed questions for a decision model.

Both use **provider-neutral Rust types** from the
<a href="../rust/switchyard_protocol/"><code>switchyard-protocol</code> crate</a>. These types
define Switchyard's protocols. Your application converts its inputs into these
types, passes them to libsy, and converts model calls and responses at its
provider boundary.

## Why use a neutral protocol?

Provider APIs describe the same interactions in different ways. An Anthropic
tool call is a `tool_use` content block. OpenAI Responses uses a `function_call`
item. Chat Completions puts it in an assistant message's `tool_calls` array.

An algorithm needs to know that the model called a tool, its name, and its
arguments. A shared `ToolCall` gives it those facts without tying the algorithm
to one provider's JSON. Your gateway or agent harness keeps control of the client,
credentials, and execution.

```text
Application or provider data
          |
       convert
          v
Switchyard protocol types <--> libsy algorithm
          |
       convert
          v
Your model client and provider API
```

The protocol crate defines the data exchanged with libsy. Conversion and network
calls happen in your application or in optional Switchyard crates. When handling
`run_stream`, your application can use any client and return the protocol response
through `call.respond`. The crate also provides optional client traits that help
Switchyard's supplied clients and execution helpers work together.

## Two kinds of model interaction

| Interaction | Protocol fields and types | Used when | Read next |
|---|---|---|---|
| LLM conversation | <a href="../rust/switchyard_protocol/envelope/struct.Request.html#structfield.llm_request"><code>Request.llm_request: LlmRequest</code></a>; <a href="../rust/switchyard_protocol/envelope/struct.Response.html#structfield.llm_response"><code>Response.llm_response: LlmResponse</code></a> | Your application starts a run or serves `Step::CallModel`. Content can include text, tools, reasoning, and media. | [LLM-Related Protocols](llm.md) |
| Typed decision | <a href="../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>CallDecision.request: DecisionRequest</code></a>; the work future returns <a href="../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>libsy::Result&lt;DecisionResponse&gt;</code></a> | Your application serves `Step::CallDecision`. Named questions ask for a Boolean estimate, a choice, or a rubric score. | [Decision Model-Related Protocols](decisions.md) |

The LLM envelopes also carry `metadata: Option<Metadata>`. That application
metadata is separate from the provider's conversation content. Decision calls
use their own request and response types directly.

## Choose who owns conversion

| Integration choice | Use it for |
|---|---|
| Your own adapters | Convert an existing SDK's types or an in-process model's data to and from `switchyard-protocol`. Keep your existing client and transport. |
| `switchyard-translation` | Decode and encode supported LLM JSON and streaming events while keeping your own transport. |
| `switchyard-llm-client` | Use Switchyard's HTTP clients, including LLM translation and typed decision adapters. |

The two pages below explain these boundaries and show the field mappings. For
complete type and method definitions, open the
<a href="../rust/switchyard_protocol/">protocol crate reference</a>.

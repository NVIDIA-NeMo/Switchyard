# API and Model Compatibility

Check three things separately: the protocol format, the model's capabilities,
and the integration executing the algorithm. Supporting an API format does not
guarantee that a model supports every feature represented by that format.

## Supported API formats

These are the supplied adapter paths. Applications embedding libsy may own their
conversions instead. See **[Switchyard Protocols](../protocols/index.md)** for the
neutral types and integration flow.

| API | Supplied support | Read the exact contract |
|---|---|---|
| OpenAI Chat Completions | LLM request, buffered response, and streaming conversion; optional HTTP client. | [Requests](protocols/llm_requests.md), [responses](protocols/llm_responses.md) |
| OpenAI Responses | LLM request, buffered response, and streaming conversion; optional HTTP client. | [Requests](protocols/llm_requests.md), [responses](protocols/llm_responses.md) |
| Anthropic Messages | LLM request, buffered response, and streaming conversion; optional HTTP client. | [Requests](protocols/llm_requests.md), [responses](protocols/llm_responses.md) |
| System One | Typed Boolean, choice, and score questions through the supplied decision client. Used by open decision models, including Jev. | [Questions](protocols/decision_questions.md), [answers](protocols/decision_answers.md) |
| OpenAI Decisions | Typed decision questions through the supplied decision client. | [Questions](protocols/decision_questions.md), [answers](protocols/decision_answers.md) |

Provider fields have different equivalents and limitations across APIs.
Preservation supports lossless same-format replay. Cross-format conversion may
lose information. See **[translation policies and preservation](protocols/translation.md)**
before relying on a particular feature.

## Model requirements

Libsy algorithms select from the model IDs supplied by the host. There is no
fixed model-name allowlist in the routing core. Your endpoint and credentials
must serve the IDs you configure.

| Model role | Check before use | Guidance |
|---|---|---|
| Answer model | Context length, tool calling, input modalities, and requested output format for the actual workload. | [Model checks](../routing_algorithms/model_selection.md#check-the-efficient-model-before-you-route) |
| LLM judge | Structured output matching the algorithm's verdict schema and sufficient output budget. | [Judge compatibility](../routing_algorithms/llm_classifier_routing.md#judge-model-compatibility) |
| Decision model | Served question kinds and response fields matching the configured adapter. | [Decision protocol guide](../protocols/decisions.md) |
| Local or trained router | The trained assets and backend expected by that algorithm. | [Prefill routing](../routing_algorithms/prefill_routing.md) |

The **[measured model pairs](../routing_algorithms/model_selection.md#measured-pairs)**
and **[benchmark configurations](../benchmarks.md)** are examples with recorded
workloads and settings. They are not an exhaustive supported-model catalog.
Recheck capability and availability on the endpoint you will actually use.

## Integration restrictions

| Integration | Configuration owner | Important boundary |
|---|---|---|
| [Hosted OpenRouter router](https://openrouter.ai/docs/guides/routing/routers/switchyard-router) | OpenRouter request controls. | Only provider-exposed algorithms and settings apply; runner TOML is not its API. |
| [NeMo Relay](../integrations/nemo_relay.md) | Switchyard runner configuration inside Relay. | Check Relay version and plugin credential rules. |
| [LiteLLM](../../examples/litellm/README.md#supported-algorithm-semantics) | LiteLLM inventory and plugin configuration. | The experimental decision-only integration cannot serve intermediate model calls. |
| [Direct libsy](../reference/host_integration.md) | Your application. | Your host must fulfill every requested call and handle the final outcome. |
| [Local reference server](../operations/local_evaluation.md) | Runner TOML and local server settings. | Demo and personal use, not a supported production deployment. |

For dependency revisions, follow the **[quickstart](../getting_started.md#install)**
or the selected integration's instructions. Development callback examples use a
different API from the released `0.3.0` bindings and crates.

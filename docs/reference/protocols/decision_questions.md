# Decision Questions

For the integration flow and a request example, read **[Decision Model-Related Protocols](../../protocols/decisions.md)**.
For result interpretation and provider response mappings, continue to
**[Decision Answers](decision_answers.md)**.

System One is a common API for open decision models, including Jev.
<a href="../../../rust/switchyard_llm_client/struct.SystemOneClient.html"><code>SystemOneClient</code></a> uses `/v1/systemone`; <a href="../../../rust/switchyard_llm_client/struct.OpenAiDecisionClient.html"><code>OpenAiDecisionClient</code></a> uses
[`/v1/decisions`](https://developers.openai.com/api/reference/resources/decisions/methods/create).
Both accept <a href="../../../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>.

## Request fields

| Protocol field and type | Meaning | System One | OpenAI Decisions |
|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.model"><code>DecisionRequest.model: Option&lt;ModelId&gt;</code></a> | Selected decision model. The HTTP adapters require it before sending. | `model` | `model` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.context"><code>DecisionRequest.context: Value</code></a> | Shared conversation, application state, or other material to evaluate. | JSON-valued `state` | Text-valued `input`; structured values are serialized as JSON text |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.questions"><code>DecisionRequest.questions: BTreeMap&lt;String, DecisionQuestion&gt;</code></a> | Independent questions keyed by stable IDs. | `questions` object, keyed by ID | `questions[]`, with each ID in `name` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.instructions"><code>DecisionQuestion.instructions: Value</code></a> | Instructions for one question. | JSON-valued `instructions` | Text-valued `instructions`; structured values become JSON text |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.kind"><code>DecisionQuestion.kind: DecisionKind</code></a> | Expected Boolean, choice, or rubric-score answer. | Question `type` and `criteria` | Question `type`, with `choices` or `levels` where applicable |

Question IDs identify the answers. Preserve them through conversion; an array
position alone is not a replacement for an ID. Context is shared across the
request, while each question has its own instructions.

## Question kinds

| Kind and fields | Contract | System One | OpenAI Decisions |
|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Boolean"><code>DecisionKind::Boolean</code></a> with `true_description: Option<Value>` and `false_description: Option<Value>` | Ask for a Boolean judgment or probability of true. Descriptions define the two answers when needed. | `type: "noul"`; descriptions in `criteria.true` and `criteria.false` | `type: "predicate"`; descriptions are included in instruction text because there are no separate fields |
| <a href="../../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Choice"><code>DecisionKind::Choice</code></a> with <a href="../../../rust/switchyard_protocol/decision/struct.ChoiceOption.html"><code>options: Vec&lt;ChoiceOption&gt;</code></a> | Declare a nonempty list of options with unique IDs, preserving their order in the protocol. | `type: "choice"`; `criteria` maps option IDs to descriptions | `type: "choice"`; `choices[]` carries `value` and optional `description` |
| <a href="../../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Score"><code>DecisionKind::Score</code></a> with `levels: Vec<Value>` | Declare at least two levels, ordered low to high and indexed from zero. | `type: "score"`; `criteria` is the level array | `type: "score"`; `levels[].label` carries each level's text |
| <a href="../../../rust/switchyard_protocol/decision/struct.ChoiceOption.html#structfield.id"><code>ChoiceOption.id: String</code></a> | Stable ID used by the selected answer and any distribution. | Key in `criteria` | `choices[].value` |
| <a href="../../../rust/switchyard_protocol/decision/struct.ChoiceOption.html#structfield.description"><code>ChoiceOption.description: Option&lt;Value&gt;</code></a> | Optional meaning when the ID alone is insufficient. | JSON-valued description; absent description becomes null | Text description when supplied; structured values become JSON text |

Choice questions can contain more than two options. A score is a position on
the declared rubric, including fractional positions; it is not an arbitrary
numeric measurement. Keep the question's levels when interpreting the answer.

## Construction and validation

The fields are public. Constructing or deserializing a <a href="../../../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a> does
not enforce all question contracts. Supply a selected model for HTTP calls,
valid option IDs, and a correctly ordered rubric. The adapters apply their
format-specific checks; they are not a general validation layer for every
possible backend.

Enums in the protocol's own serialized form use snake-case `type` tags with a
`data` payload. Provider payloads use the mappings above. Serializing the protocol
value directly is therefore not a substitute for a provider adapter.

## Provider adapters

The [System One implementation](../../../crates/libsy-llm-client/src/system_one.rs)
and [OpenAI Decisions implementation](../../../crates/libsy-llm-client/src/openai_decisions.rs)
are reference examples of request encoding and response conversion. Both supplied
clients make one attempt per call, with a timeout covering the request and reading
the body.

You can use either supplied client, or convert an existing client's data yourself.
<a href="../../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a> is an optional shared client interface. The
**[guide's direct call handling](../../protocols/decisions.md#own-the-conversion)**
shows the application boundary; **[Decision Answers](decision_answers.md)** covers
the response contract.

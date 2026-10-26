# Decision Model-Related Protocols

A decision model evaluates a context against named questions and returns typed
answers. An algorithm can use those answers to choose a model or decide what
work to request next.

Switchyard represents this exchange with
<a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>
and
<a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>.
These are the types exchanged through `Step::CallDecision`.

## Describe the questions

One request carries shared context and a map of independent questions. The
question IDs let the algorithm match each answer to its question.
Select a protocol field or type in the tables to open its crate reference.

| Field | Rust type | Meaning |
|---|---|---|
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.model"><code>DecisionRequest.model</code></a> | <a href="../../rust/switchyard_protocol/model_id/struct.ModelId.html"><code>Option&lt;ModelId&gt;</code></a> | Optional while constructing the request. The HTTP adapters require a selected model before sending it. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.context"><code>DecisionRequest.context</code></a> | `serde_json::Value` | Conversation, application state, or other material to evaluate. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.questions"><code>DecisionRequest.questions</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html"><code>BTreeMap&lt;String, DecisionQuestion&gt;</code></a> | Independent questions keyed by stable IDs. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.instructions"><code>DecisionQuestion.instructions</code></a> | `serde_json::Value` | Instructions for evaluating one question. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.kind"><code>DecisionQuestion.kind</code></a> | <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html"><code>DecisionKind</code></a> | The expected Boolean, choice, or score answer. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Boolean"><code>DecisionKind::Boolean { true_description, false_description }</code></a> | Each field is `Option<Value>` | Optional descriptions of the true and false answers. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Choice"><code>DecisionKind::Choice { options }</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.ChoiceOption.html"><code>Vec&lt;ChoiceOption&gt;</code></a> | Nonempty options with unique `id: String` values and optional `description: Option<Value>` values. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Score"><code>DecisionKind::Score { levels }</code></a> | `Vec<Value>` | At least two rubric levels, ordered low to high and indexed from zero. |

Here is a request asking whether a task contains private information. This is
request construction only; it does not call a provider.

```rust
use std::collections::BTreeMap;
use serde_json::json;
use switchyard_protocol::{DecisionKind, DecisionQuestion, DecisionRequest};

let request = DecisionRequest {
    model: Some("privacy-judge".into()),
    context: json!({"task": "Summarize this customer's account history."}),
    questions: BTreeMap::from([(
        "contains_private_data".into(),
        DecisionQuestion {
            instructions: json!("Does the task contain private customer data?"),
            kind: DecisionKind::Boolean {
                true_description: Some(json!("Contains private customer data")),
                false_description: Some(json!("Contains only public information")),
            },
        },
    )]),
};
```

The protocol types have public fields. Their documentation states the contracts;
constructing a value does not validate every constraint. Your adapter must accept
valid requests and reject responses that do not match their questions.

## Interpret the answers

`DecisionResponse::answers` maps the same question IDs to `DecisionAnswer` values.
Each answer separates its typed `value` from optional `provider_confidence`.

| Field or variant | Rust type | How to interpret it |
|---|---|---|
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse.answers</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html"><code>BTreeMap&lt;String, DecisionAnswer&gt;</code></a> | Answers keyed by the request's question IDs. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.value"><code>DecisionAnswer.value</code></a> | <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html"><code>DecisionValue</code></a> | The typed answer matching the question's kind. |
| <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.Value"><code>BooleanEstimate::Value</code></a> | `bool` | A Boolean result without an implied probability. |
| <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.ProbabilityTrue"><code>BooleanEstimate::ProbabilityTrue</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.Probability.html"><code>Probability</code></a> | Probability of true in `[0, 1]`. The algorithm chooses any threshold. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Choice.field.selected"><code>DecisionValue::Choice { selected, .. }</code></a> | `String` | The selected option's declared ID. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Choice.field.probabilities"><code>DecisionValue::Choice { probabilities, .. }</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.Probability.html"><code>Option&lt;BTreeMap&lt;String, Probability&gt;&gt;</code></a> | A complete distribution keyed by option ID, when supplied. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Score.field.value"><code>DecisionValue::Score { value, .. }</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.ScoreValue.html"><code>ScoreValue</code></a> | A possibly fractional position in `0..=N-1`. Retain the question's rubric to interpret it. |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Score.field.probabilities"><code>DecisionValue::Score { probabilities, .. }</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.Probability.html"><code>Option&lt;Vec&lt;Probability&gt;&gt;</code></a> | A distribution in the request's rubric order, when supplied. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.provider_confidence"><code>DecisionAnswer.provider_confidence</code></a> | <a href="../../rust/switchyard_protocol/decision/struct.ProviderConfidence.html"><code>Option&lt;ProviderConfidence&gt;</code></a> | Provider-defined confidence, with its own scale and meaning. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.id"><code>DecisionResponse.id</code></a> | `Option<String>` | Provider-reported response ID, when available. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.model"><code>DecisionResponse.model</code></a> | <a href="../../rust/switchyard_protocol/model_id/struct.ModelId.html"><code>Option&lt;ModelId&gt;</code></a> | Provider-reported model ID, when available. |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.usage"><code>DecisionResponse.usage</code></a> | <a href="../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a> | Available token counts. Missing counts remain unknown. |

For the privacy question, `ProbabilityTrue(Probability(0.8))` means an estimated
0.8 probability that the answer is true. It is not a routing choice until the
algorithm applies its policy. Choice questions can name any supported set of
candidates; their shape is not limited to two models.

## Map to System One and OpenAI Decisions

The following tables describe Switchyard's current
[System One adapter](../../crates/libsy-llm-client/src/system_one.rs) and
[OpenAI Decisions adapter](../../crates/libsy-llm-client/src/openai_decisions.rs).
System One serves Jev through `/v1/systemone`; OpenAI uses
[`/v1/decisions`](https://developers.openai.com/api/reference/resources/decisions/methods/create).

### Request fields

Fields without a type prefix in this table belong to `DecisionRequest`.

| Switchyard field and type | System One | OpenAI Decisions |
|---|---|---|
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.model"><code>model: Option&lt;ModelId&gt;</code></a> | `model` | `model` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.context"><code>context: Value</code></a> | JSON-valued `state` | Text-valued `input`; no structured JSON input field, so the adapter serializes structured context as JSON text |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.questions"><code>questions: BTreeMap&lt;String, DecisionQuestion&gt;</code></a> | `questions` object; each key is the question ID | `questions[]` array; each `name` is the question ID |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.instructions"><code>DecisionQuestion.instructions: Value</code></a> | `questions[id].instructions`, as JSON | `questions[].instructions`, as text; structured instructions become JSON text |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Boolean"><code>DecisionKind::Boolean { true_description, false_description }</code></a>; each is `Option<Value>` | `type: "noul"`; `criteria.true` and `criteria.false` hold the descriptions | `type: "predicate"`; no separate description fields, so descriptions are included in instruction text |
| <a href="../../rust/switchyard_protocol/decision/struct.ChoiceOption.html#structfield.id"><code>ChoiceOption.id: String</code></a> | `type: "choice"`; option IDs are keys in `criteria` | `type: "choice"`; option IDs are `choices[].value` |
| <a href="../../rust/switchyard_protocol/decision/struct.ChoiceOption.html#structfield.description"><code>ChoiceOption.description: Option&lt;Value&gt;</code></a> | `criteria[option_id]`, as JSON | `choices[].description`, as text |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html#variant.Score"><code>DecisionKind::Score { levels }</code></a>; `levels: Vec<Value>` | `type: "score"`; `criteria` is the ordered level array | `type: "score"`; `levels[].label` carries each level's text |

### Response fields

| Switchyard field and type | System One | OpenAI Decisions |
|---|---|---|
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse.answers: BTreeMap&lt;String, DecisionAnswer&gt;</code></a> | `answers` object; each key is the question ID | `answers[]` array; each `name` is the question ID |
| <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.ProbabilityTrue"><code>BooleanEstimate::ProbabilityTrue(Probability)</code></a> | `type: "noul"`, `noul` | `type: "predicate"`, `probability` |
| <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.Value"><code>BooleanEstimate::Value(bool)</code></a> | No Boolean-only field; the adapter returns a probability | No Boolean-only field; the adapter returns a probability |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Choice.field.selected"><code>DecisionValue::Choice { selected, .. }</code></a>; `selected: String` | `choice` | `choice` |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Choice.field.probabilities"><code>DecisionValue::Choice { probabilities, .. }</code></a>; <a href="../../rust/switchyard_protocol/decision/struct.Probability.html"><code>probabilities: Option&lt;BTreeMap&lt;String, Probability&gt;&gt;</code></a> | Optional `probabilities` object keyed by option ID | `probabilities[]` with `value` and `probability`, converted to an ID-keyed map |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Score.field.value"><code>DecisionValue::Score { value, .. }</code></a>; <a href="../../rust/switchyard_protocol/decision/struct.ScoreValue.html"><code>value: ScoreValue</code></a> | `score` | `score` |
| <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Score.field.probabilities"><code>DecisionValue::Score { probabilities, .. }</code></a>; <a href="../../rust/switchyard_protocol/decision/struct.Probability.html"><code>probabilities: Option&lt;Vec&lt;Probability&gt;&gt;</code></a> | Optional `probabilities` object with numeric string keys | `probabilities[]` with numeric `value` and `probability`; reordered into rubric order |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.provider_confidence"><code>DecisionAnswer.provider_confidence: Option&lt;ProviderConfidence&gt;</code></a> | `confidence` | `confidence` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.id"><code>DecisionResponse.id: Option&lt;String&gt;</code></a> | Optional `id` | No response ID is read by the adapter; returns `None` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.model"><code>DecisionResponse.model: Option&lt;ModelId&gt;</code></a> | Optional `model` | `model` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.usage"><code>DecisionResponse.usage: Usage</code></a> | Deserialized into <a href="../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a> | Token counts normalized into <a href="../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a>, separating cache-read/write counts from total input |

**For supported questions and answers, conversion is a direct, lossless mapping
of decision data.** IDs, answer probabilities, rubric positions, and confidence
retain their meanings. Structured values become JSON text where OpenAI requires
strings. This retains their content, rather than making the two wire formats
identical or providing an arbitrary JSON round-trip guarantee.

Both adapters return probabilities for Boolean answers. A custom backend that
returns only true or false can use `BooleanEstimate::Value` without inventing
certainty. Optional distributions remain `None` when absent. The OpenAI adapter
reports refusals as errors because `DecisionValue` has no refusal variant.

## Own the conversion

When handling `Step::CallDecision`, use your existing client or a function to
convert `call.request` into the backend's format, send it, and convert the answer
into `DecisionResponse`. Pass that work to `call.respond` as a future returning
`libsy::Result<DecisionResponse>`.

Match question IDs and kinds, and retain available distributions and confidence
separately. See the two adapters above for reference implementations of request
encoding and response conversion. Decision conversions live in
`switchyard-llm-client`; `switchyard-translation` handles LLM formats.

<a href="../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>
is an optional interface that lets Switchyard's clients and execution helpers
work together. Direct `run_stream` handling does not require implementing it.
See [Mechanics of Libsy](../mechanics_of_libsy.md#stepcalldecision) for the
call-handling loop and cancellation behavior.

## Use the supplied clients

You can instead use the public `SystemOneClient` or `OpenAiDecisionClient` from
[`switchyard-llm-client`](../../crates/libsy-llm-client/README.md). Use dependencies at
the same API revision as libsy and the protocol crate.

This async excerpt sends the earlier `request` to System One. Set its `model` to
an ID served by your endpoint. `SYSTEM_ONE_ENDPOINT` must include `/v1/systemone`.
Running it makes a provider call and may incur a charge.
The trait import makes the supplied client's `call` method available; you do not
implement a client in this example.

```rust
use std::time::Duration;
use switchyard_llm_client::SystemOneClient;
use switchyard_protocol::RoutedDecisionClient;

let client = SystemOneClient::new(
    std::env::var("SYSTEM_ONE_ENDPOINT")?.parse()?,
    std::env::var("SYSTEM_ONE_API_KEY")?,
    Duration::from_secs(30),
)?;
let response = client.call(request).await?;
```

`OpenAiDecisionClient::new` takes the same arguments, with an endpoint ending in
`/v1/decisions` and that provider's key. Each adapter makes one attempt per call;
its timeout covers the request and reading the response body.

For the built-in execution path, register decision clients by model ID with
`ClientRouter::single_with_decision_clients` or `new_with_decision_clients`, then
use `switchyard_llm_client::run`. The algorithm still receives the same typed
`DecisionResponse` whichever client performs the call.

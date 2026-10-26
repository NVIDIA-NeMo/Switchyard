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

| Component | Rust shape | Meaning |
|---|---|---|
| Selected model | `Option<ModelId>` | Optional while constructing the request. The HTTP adapters require a selected model before sending it. |
| Context | `serde_json::Value` | Conversation, application state, or other material to evaluate. |
| Questions | `BTreeMap<String, DecisionQuestion>` | Questions keyed by stable IDs. Each has JSON-valued `instructions` and a `DecisionKind`. |
| Boolean question | `DecisionKind::Boolean` | A true/false judgment, with optional descriptions of each answer. |
| Choice question | `DecisionKind::Choice` | One of a nonempty list of `ChoiceOption` values. IDs must be unique; descriptions are optional. |
| Score question | `DecisionKind::Score` | A position on an ordered rubric with at least two levels, indexed from zero. |

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

| Answer | Rust representation | How to interpret it |
|---|---|---|
| Boolean judgment | `BooleanEstimate::Value(bool)` | A Boolean result without an implied probability. |
| Boolean probability | `BooleanEstimate::ProbabilityTrue(Probability)` | Probability of true in `[0, 1]`. The algorithm chooses any threshold. |
| Choice | `DecisionValue::Choice { selected, probabilities }` | `selected` names a declared option. An optional distribution maps every option ID to its probability. |
| Rubric score | `DecisionValue::Score { value, probabilities }` | A possibly fractional position in `0..=N-1`. An optional distribution follows the request's level order. Retain the request to interpret it. |
| Provider confidence | `Option<ProviderConfidence>` | A provider-defined confidence measure. Its scale is separate from answer probability. |
| Response details | `id`, `model`, `Usage` | Provider-reported identity and available usage counts. Missing information stays unknown. |

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

| Switchyard IR | System One | OpenAI Decisions |
|---|---|---|
| `model` | `model` | `model` |
| `context` | JSON-valued `state` | Text-valued `input`; structured context is serialized as JSON text |
| Question map key | Key in the `questions` object | `questions[].name` |
| `DecisionQuestion.instructions` | JSON-valued `instructions` | Text-valued `instructions`; structured instructions become JSON text |
| Boolean kind | `type: "noul"`; descriptions in `criteria.true` and `criteria.false` | `type: "predicate"`; optional true/false descriptions are included in instruction text |
| Choice kind | `type: "choice"`; `criteria` maps option IDs to descriptions | `type: "choice"`; `choices[]` contains `value` and optional `description` |
| Score kind | `type: "score"`; `criteria` is the ordered level array | `type: "score"`; `levels[].label` carries each level's text |

### Response fields

| Switchyard IR | System One | OpenAI Decisions |
|---|---|---|
| Answer map key | Key in the `answers` object | `answers[].name` |
| Boolean probability | `type: "noul"`, `noul` | `type: "predicate"`, `probability` |
| Selected choice | `choice` | `choice` |
| Choice distribution | `probabilities` object keyed by option ID | `probabilities[]` with `value` and `probability`, converted to an ID-keyed map |
| Score | `score` | `score` |
| Score distribution | `probabilities` object with numeric string keys | `probabilities[]` with numeric `value` and `probability`; reordered into rubric order |
| Provider confidence | `confidence` | `confidence` |
| Identity | Optional `id`, `model` | `model`; the adapter returns `id: None` |
| Usage | Deserialized into `Usage` | Token counts normalized into `Usage`, separating cache-read/write counts from total input |

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

To keep an existing client, implement
<a href="../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>.
Its `call` method accepts a `DecisionRequest` and returns
`Result<DecisionResponse, LlmClientError>` asynchronously.

Your implementation converts the request to the backend's format, sends it,
and converts the result back. Match question IDs and kinds, retain available
distributions and confidence separately, and report transport or decoding errors
through `LlmClientError`. See the two adapters above for reference implementations
of request encoding and response conversion. Decision conversions live in
`switchyard-llm-client`; `switchyard-translation` handles LLM formats.

When driving `run_stream` directly, pass that client future to the decision call's
`respond` method. See [Mechanics of Libsy](../mechanics_of_libsy.md#stepcalldecision)
for the call-handling loop and cancellation behavior.

## Use the supplied clients

You can instead use the public `SystemOneClient` or `OpenAiDecisionClient` from
[`switchyard-llm-client`](../../crates/libsy-llm-client/README.md). Use dependencies at
the same API revision as libsy and the protocol crate.

This async excerpt sends the earlier `request` to System One. Set its `model` to
an ID served by your endpoint. `SYSTEM_ONE_ENDPOINT` must include `/v1/systemone`.
Running it makes a provider call and may incur a charge.

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

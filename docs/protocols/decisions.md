# Decision Model-Related Protocols

A decision model evaluates a context against named questions and returns typed
answers. An algorithm can use those answers to choose a model or decide what
work to request next.

Switchyard represents this exchange with
<a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>
and
<a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>.
These are the types exchanged through `Step::CallDecision`.

## Protocol and API mappings

One request carries shared context and a map of independent questions. The
question IDs let the algorithm match each answer to its question. Each answer
contains a typed value and optional provider confidence.

This table combines their meanings with the mappings used by the current
[System One adapter](../../crates/libsy-llm-client/src/system_one.rs) and
[OpenAI Decisions adapter](../../crates/libsy-llm-client/src/openai_decisions.rs).
Select a field or type to open its crate reference for details.

| Protocol field and type | Meaning | System One | OpenAI Decisions |
|---|---|---|---|
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.model"><code>DecisionRequest.model</code></a>: <a href="../../rust/switchyard_protocol/model_id/struct.ModelId.html"><code>Option&lt;ModelId&gt;</code></a> | Selected decision model; required by the HTTP adapters before sending. | `model` | `model` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.context"><code>DecisionRequest.context: Value</code></a> | Shared material to evaluate, such as a conversation or application state. | JSON-valued `state` | Text-valued `input`; structured values become JSON text |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html#structfield.questions"><code>DecisionRequest.questions</code></a>: <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html"><code>BTreeMap&lt;String, DecisionQuestion&gt;</code></a> | Independent questions keyed by stable IDs. | `questions` object, keyed by ID | `questions[]`, identified by `name` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.instructions"><code>DecisionQuestion.instructions: Value</code></a> | Instructions for evaluating one question. | JSON-valued `instructions` | Text-valued `instructions`; structured values become JSON text |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html#structfield.kind"><code>DecisionQuestion.kind</code></a>: <a href="../../rust/switchyard_protocol/decision/enum.DecisionKind.html"><code>DecisionKind</code></a> | Boolean judgment, choice among options, or position on an ordered rubric. | `type`: `noul`, `choice`, or `score`; descriptions/options/levels in `criteria` | `type`: `predicate`, `choice`, or `score`; `choices` or `levels` where applicable |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse.answers</code></a>: <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html"><code>BTreeMap&lt;String, DecisionAnswer&gt;</code></a> | Answers matched to the request's question IDs. | `answers` object, keyed by ID | `answers[]`, identified by `name` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.value"><code>DecisionAnswer.value</code></a>: <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html"><code>DecisionValue</code></a> | Boolean estimate, selected option, or rubric score; optional distributions where available. | `noul`, `choice`, or `score`; optional `probabilities` for choices/scores | `probability`, `choice`, or `score`; `probabilities` for choices/scores |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.provider_confidence"><code>DecisionAnswer.provider_confidence</code></a>: <a href="../../rust/switchyard_protocol/decision/struct.ProviderConfidence.html"><code>Option&lt;ProviderConfidence&gt;</code></a> | Provider-defined confidence, separate from answer probability. | `confidence` | `confidence` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.id"><code>DecisionResponse.id: Option&lt;String&gt;</code></a> | Provider-reported response ID, when available. | Optional `id` | No response ID is read by the adapter; returns `None` |
| <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.usage"><code>DecisionResponse.usage</code></a>: <a href="../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a> | Available token counts; missing counts stay unknown. | Deserialized into `Usage` | Token counts normalized into `Usage`, with separate cache counts |

System One serves Jev through `/v1/systemone`; OpenAI uses
[`/v1/decisions`](https://developers.openai.com/api/reference/resources/decisions/methods/create).
OpenAI expects text for context and instructions, so the adapter serializes
structured values as JSON text. It also puts Boolean answer descriptions in
instruction text because there are no separate fields for them.

## Describe the questions

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
For the privacy question, `ProbabilityTrue(Probability(0.8))` means an estimated
0.8 probability that the answer is true. It is not a routing choice until the
algorithm applies its policy. Choice questions can name any supported set of
candidates; their shape is not limited to two models. A rubric score is a
possibly fractional position in `0..=N-1` for the question's ordered levels.
Retain the question to interpret its score and distribution.

## Understand conversion

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

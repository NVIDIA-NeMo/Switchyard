# Decision Model-Related Protocols

A decision model evaluates an input against named questions and returns typed
answers. An algorithm can use those answers to choose a model or decide what
work to request next.

Switchyard represents this exchange with
<a href="../../rust/switchyard_protocol/decision/struct.DecisionRequest.html"><code>DecisionRequest</code></a>
and
<a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>.
These are the types exchanged through <a href="../../rust/switchyard_libsy/enum.Step.html#variant.CallDecision"><code>Step::CallDecision</code></a>.

System One is a common API for open decision models, including Jev.
Switchyard supports it alongside OpenAI Decisions through the same protocol.

!!! info "Question fields, answer types, and API mappings"

    - **[Question reference](../reference/protocols/decision_questions.md):** shared input, question IDs, Boolean/choice/score questions, and provider request mappings.
    - **[Answer reference](../reference/protocols/decision_answers.md):** probabilities, rubric scores, confidence, usage, provider response mappings, and errors.

## Describe the questions

One request carries shared input and independent questions keyed by ID.
Each <a href="../../rust/switchyard_protocol/decision/struct.DecisionQuestion.html"><code>DecisionQuestion</code></a>
defines what to evaluate and which answer kind to return. The response uses the
same question IDs, so the algorithm can match each answer to its question.

Here is a request asking whether a task contains private information. This is
request construction only; it does not call a provider.

```rust
use std::collections::BTreeMap;
use serde_json::json;
use switchyard_protocol::{DecisionKind, DecisionQuestion, DecisionRequest};

let request = DecisionRequest {
    model: Some("privacy-judge".into()),
    input: json!({"task": "Summarize this customer's account history."}),
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

For required option IDs, rubric levels, and validation responsibilities, see
**[question construction and validation](../reference/protocols/decision_questions.md#construction-and-validation)**.

## Interpret the answers

<a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse::answers</code></a> maps the same question IDs to <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html"><code>DecisionAnswer</code></a> values.
Each answer separates its typed <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.value"><code>value</code></a> from optional <a href="../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.provider_confidence"><code>provider_confidence</code></a>.
For the privacy question, <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.ProbabilityTrue"><code>ProbabilityTrue(Probability(0.8))</code></a> means an estimated
0.8 probability that the answer is true. It is not a routing choice until the
algorithm applies its policy. Choice questions can name any supported set of
candidates; their shape is not limited to two models.

The **[answer value reference](../reference/protocols/decision_answers.md#answer-values)**
explains rubric scores, distributions, and the difference between a Boolean
judgment and an answer probability.

## Understand conversion

**For supported questions and answers, conversion is a direct, lossless mapping
of decision data.** IDs, answer probabilities, rubric positions, and confidence
retain their meanings. Structured values become JSON text where OpenAI requires
strings. This retains their content, rather than making the two wire formats
identical or providing an arbitrary JSON round-trip guarantee.

Both adapters return probabilities for Boolean answers. A custom backend that
returns only true or false can use <a href="../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.Value"><code>BooleanEstimate::Value</code></a> without inventing
certainty. Optional distributions remain `None` when absent. The OpenAI adapter
reports refusals as errors because <a href="../../rust/switchyard_protocol/decision/enum.DecisionValue.html"><code>DecisionValue</code></a> has no refusal variant.

## Own the conversion

When handling <a href="../../rust/switchyard_libsy/enum.Step.html#variant.CallDecision"><code>Step::CallDecision</code></a>, use your existing client or a function to
convert <a href="../../rust/switchyard_libsy/struct.CallDecision.html#structfield.request"><code>call.request</code></a> into the backend's format, send it, and convert the answer
into <a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a>. Pass that work to <a href="../../rust/switchyard_libsy/struct.CallDecision.html#method.respond"><code>call.respond</code></a> as a future returning
<a href="../../rust/switchyard_libsy/type.Result.html"><code>libsy::Result&lt;DecisionResponse&gt;</code></a>.

Match question IDs and kinds, and retain available distributions and confidence
separately. The
[question reference](../reference/protocols/decision_questions.md#provider-adapters)
links the System One and OpenAI adapter implementations. Decision conversions live in
<a href="../../rust/switchyard_llm_client/"><code>switchyard-llm-client</code></a>; <a href="../../rust/switchyard_translation/"><code>switchyard-translation</code></a> handles LLM formats.

<a href="../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html"><code>RoutedDecisionClient</code></a>
is an optional interface that lets Switchyard's clients and execution helpers
work together. Direct <a href="../../rust/switchyard_libsy/trait.Algorithm.html#method.run_stream"><code>run_stream</code></a> handling does not require implementing it.
See [Mechanics of Libsy](../mechanics_of_libsy.md#stepcalldecision) for the
call-handling loop and cancellation behavior.

## Use the supplied clients

You can instead use the public <a href="../../rust/switchyard_llm_client/struct.SystemOneClient.html"><code>SystemOneClient</code></a> or <a href="../../rust/switchyard_llm_client/struct.OpenAiDecisionClient.html"><code>OpenAiDecisionClient</code></a> from
[`switchyard-llm-client`](../../crates/libsy-llm-client/README.md). Use dependencies at
the same API revision as libsy and the protocol crate.

This async excerpt sends the earlier `request` to System One. Set its `model` to
an ID served by your endpoint. `SYSTEM_ONE_ENDPOINT` must include `/v1/systemone`.
Running it makes a provider call and may incur a charge.
The trait import makes the supplied client's <a href="../../rust/switchyard_protocol/client/trait.RoutedDecisionClient.html#tymethod.call"><code>call</code></a> method available; you do not
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

<a href="../../rust/switchyard_llm_client/struct.OpenAiDecisionClient.html#method.new"><code>OpenAiDecisionClient::new</code></a> takes the same arguments, with an endpoint ending in
`/v1/decisions` and that provider's key. Each adapter makes one attempt per call;
its timeout covers the request and reading the response body.

## Generate task or turn metadata

The [standalone assessment example](../../crates/libsy-llm-client/examples/assess.rs)
sends all questions in one decision-model call and prints JSON containing
`task_id`, optional `turn_id`, and the typed `assessment` response. The response
retains available model identity, usage, answer probabilities, and confidence.
It calls the decision client directly, without a routing algorithm or
completion model. The IDs only label the output.

Each call takes two files. The questions file holds the rubric. The context file
holds the evidence for this call. There is one questions file for each kind of
assessment:

- [Task questions](../../crates/libsy-llm-client/examples/task_questions.json)
  describe the whole task from its opening request. Use them once per task.
  There are eight: `task`, `work_scope`, `code_deliverable`, `difficulty`,
  `precision`, `output_length`, `codebase_integration`, and `visual_quality`.
- [Turn questions](../../crates/libsy-llm-client/examples/turn_questions.json)
  describe the state of the work at one turn boundary. Use them on each turn you
  want to assess. There are eleven: `difficulty` (of the remaining work), `task`,
  `new_goal`, `repeating_failure`, `false_success`, `user_rejection`, `off_task`,
  `qa_recovered`, `gave_up`, `next_action`, and `ready_to_finish`.

A questions file has a `preamble` and a `questions` map. The example puts the
preamble in front of every question's instructions. Edit the questions file to
change the rubric. These are draft rubrics. Their wording, token-size ranges, and
score labels are our own definitions and need evaluation on your workload.

The context files are examples: [task context](../../crates/libsy-llm-client/examples/task_context.json)
and [turn context](../../crates/libsy-llm-client/examples/turn_context.json).
Replace them with your evidence. The turn context takes a summary of the task
assessment in `task_anchor`.

Set `DECISION_ENDPOINT` to the full endpoint URL, `DECISION_API_KEY` to its key,
and `DECISION_MODEL` to the decision model it serves. `DECISION_FORMAT` accepts
`system_one` (the default) or `openai_decisions`. The call times out after 30
seconds. Running the example makes one provider call and may incur a charge:

```sh
cargo run -p switchyard-llm-client --example assess -- \
  crates/libsy-llm-client/examples/task_questions.json \
  crates/libsy-llm-client/examples/task_context.json \
  my-task
```

For a turn, pass the turn questions, your context, the task ID, and a turn ID as
the last argument. The caller owns triggering, history selection, and storing the
output. Each invocation makes one attempt; errors are returned to you.

## Use decisions during routing

For the built-in execution path, register decision clients by model ID with
<a href="../../rust/switchyard_llm_client/run/struct.ClientRouter.html#method.single_with_decision_clients"><code>ClientRouter::single_with_decision_clients</code></a> or <a href="../../rust/switchyard_llm_client/run/struct.ClientRouter.html#method.new_with_decision_clients"><code>new_with_decision_clients</code></a>, then
use <a href="../../rust/switchyard_llm_client/run/fn.run.html"><code>switchyard_llm_client::run</code></a>. The algorithm still receives the same typed
<a href="../../rust/switchyard_protocol/decision/struct.DecisionResponse.html"><code>DecisionResponse</code></a> whichever client performs the call.

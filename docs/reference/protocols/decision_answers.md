# Decision Answers

For the integration flow, read **[Decision Model-Related Protocols](../../protocols/decisions.md)**.
For question IDs, options, and rubric definitions, use **[Decision Questions](decision_questions.md)**.

## Response fields

<a href="../../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse.answers</code></a> maps the request's question IDs to <a href="../../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html"><code>DecisionAnswer</code></a>
values. Each answer separates its value from optional provider confidence.

| Protocol field and type | Meaning | System One | OpenAI Decisions |
|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.id"><code>DecisionResponse.id: Option&lt;String&gt;</code></a> | Provider response identifier, when available. | Optional `id` | The current adapter reads no response ID and returns none |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.model"><code>DecisionResponse.model: Option&lt;ModelId&gt;</code></a> | Provider-reported model identifier. | Optional `model` | `model` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.answers"><code>DecisionResponse.answers: BTreeMap&lt;String, DecisionAnswer&gt;</code></a> | Answers matched by question ID. | `answers` object | `answers[]`, identified by `name` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.value"><code>DecisionAnswer.value: DecisionValue</code></a> | Typed estimate matching the question's kind. | `noul`, `choice`, or `score` | `probability`, `choice`, or `score` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionAnswer.html#structfield.provider_confidence"><code>DecisionAnswer.provider_confidence: Option&lt;ProviderConfidence&gt;</code></a> | Provider-specific confidence with its own scale and meaning. | Optional `confidence` | Optional `confidence` |
| <a href="../../../rust/switchyard_protocol/decision/struct.DecisionResponse.html#structfield.usage"><code>DecisionResponse.usage: Usage</code></a> | Available token counts; absent counts remain unknown. | Deserialized into <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a> | Counts normalized into <a href="../../../rust/switchyard_protocol/llm/struct.Usage.html"><code>Usage</code></a>, separating cache reads and writes from aggregate input |

## Answer values

| Variant and fields | Contract | System One | OpenAI Decisions |
|---|---|---|---|
| <a href="../../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.Value"><code>BooleanEstimate::Value(bool)</code></a> | A Boolean-only judgment with no implied certainty. | The supplied adapter returns a probability instead | The supplied adapter returns a probability instead |
| <a href="../../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html#variant.ProbabilityTrue"><code>BooleanEstimate::ProbabilityTrue(Probability)</code></a> | Probability of true on a `[0, 1]` scale. The algorithm chooses any threshold. | `type: "noul"`, field `noul` | `type: "predicate"`, field `probability` |
| <a href="../../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Choice"><code>DecisionValue::Choice</code></a> with `selected: String` and <a href="../../../rust/switchyard_protocol/decision/struct.Probability.html"><code>probabilities: Option&lt;BTreeMap&lt;String, Probability&gt;&gt;</code></a> | Select a declared ID. When supplied, the distribution covers every declared option. | `choice`; optional ID-keyed `probabilities` | `choice`; `probabilities[]` with option `value` and `probability`, converted to a map |
| <a href="../../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Score"><code>DecisionValue::Score</code></a> with <a href="../../../rust/switchyard_protocol/decision/struct.ScoreValue.html"><code>value: ScoreValue</code></a> and <a href="../../../rust/switchyard_protocol/decision/struct.Probability.html"><code>probabilities: Option&lt;Vec&lt;Probability&gt;&gt;</code></a> | Position in `0..=N-1`; the optional distribution follows the question's level order. | `score`; optional `probabilities` keyed by numeric strings, reordered by level | `score`; numeric `value` / `probability` entries, reordered by level |

<a href="../../../rust/switchyard_protocol/decision/enum.DecisionValue.html#variant.Boolean"><code>DecisionValue::Boolean</code></a> wraps a <a href="../../../rust/switchyard_protocol/decision/enum.BooleanEstimate.html"><code>BooleanEstimate</code></a>. An absent distribution stays
unknown; it does not become a one-hot distribution. Likewise, a Boolean-only
answer does not imply probability zero or one.

<a href="../../../rust/switchyard_protocol/decision/struct.Probability.html"><code>Probability</code></a>, <a href="../../../rust/switchyard_protocol/decision/struct.ScoreValue.html"><code>ScoreValue</code></a>, and <a href="../../../rust/switchyard_protocol/decision/struct.ProviderConfidence.html"><code>ProviderConfidence</code></a> are public numeric wrappers.
Their stated ranges and meanings are contracts, not validation performed by
their constructors. Keep answer probability separate from provider confidence.

## Matching and errors

Retain the request when converting a response. Question IDs identify the expected
kind; choice IDs identify valid selections; rubric order gives scores meaning.
Both clients match responses against the request and perform format-specific
checks. An unexpected ID, kind, or malformed supported response can produce
<a href="../../../rust/switchyard_protocol/client/enum.LlmClientError.html#variant.ResponseTranslation"><code>LlmClientError::ResponseTranslation</code></a>.

The OpenAI adapter treats a refusal as an error because <a href="../../../rust/switchyard_protocol/decision/enum.DecisionValue.html"><code>DecisionValue</code></a> has no
refusal variant. Transport failures and upstream HTTP errors also surface through
<a href="../../../rust/switchyard_protocol/client/enum.LlmClientError.html"><code>LlmClientError</code></a>. A host converts its work result to <a href="../../../rust/switchyard_libsy/type.Result.html"><code>libsy::Result&lt;DecisionResponse&gt;</code></a>
when replying through <a href="../../../rust/switchyard_libsy/struct.CallDecision.html#method.respond"><code>CallDecision::respond</code></a>.

These clients do not retry automatically. Their constructors accept a timeout
covering the HTTP request and reading the body. See the
**[question reference's adapter links](decision_questions.md#provider-adapters)**
for the exact conversion and error paths.

## Conversion guarantees

Supported decision fields map directly while retaining question IDs, answer
probabilities, rubric positions, and confidence. OpenAI text-only fields carry
structured protocol values as JSON text. This preserves content, rather than
making the wire formats identical or guaranteeing arbitrary JSON round trips.

The provider must actually report a field for it to be available. Keep missing
IDs, confidence, distributions, and usage unknown. For cache and reasoning token
semantics, see **[LLM usage](llm_responses.md#usage)**.

Return to **[the decision protocol guide](../../protocols/decisions.md#own-the-conversion)**
to connect the response to your application's execution loop.

# GLiNER routing evaluation for Switchyard

## Executive summary

This evaluation tested whether GLiNER can act as the classifier behind
Switchyard's existing custom multi-target router. The prototype works end to
end: Switchyard sends its structured classifier request to a small sidecar,
the sidecar returns a schema-compatible target and confidence, and Switchyard
validates the verdict before dispatching to the selected completion target.

The principal tradeoff is clear. An untuned GLiNER checkpoint provides very
low local latency, but TypeSafe Jev is materially more accurate and more robust
zero-shot on this small routing suite. GLiNER is therefore promising when
deployment-local latency, offline operation, or task-specific fine-tuning
matters; it is not a drop-in quality replacement for Jev on the evidence here.

## Router design

The example uses Switchyard's existing `llm_classifier` custom mode instead of
adding a new routing algorithm or a Python dependency to the Rust server.

1. Switchyard constructs the judge conversation and strict JSON Schema.
2. The GLiNER sidecar extracts user-authored text and verifies that the schema
   allows every configured route target.
3. GLiNER classifies the request into one of four internal semantic labels.
4. The sidecar maps the internal label to a Switchyard model group and returns
   an OpenAI Chat Completions response containing the structured verdict.
5. Switchyard validates the verdict, applies its configured fallback order and
   session affinity, then calls the chosen model target.

The four target classes are:

| Target | Intended work |
| --- | --- |
| `deterministic_tool` | Exact lookup, calculation, retrieval, or deterministic workflow |
| `small_model` | Bounded, low-risk language transformation or extraction |
| `reasoning_model` | Complex analysis, diagnosis, planning, or multi-step reasoning |
| `human_review` | High-stakes decisions, irreversible actions, or human authority |

Internal GLiNER labels do not reuse these public target names. This limits the
effect of simple prompts that attempt to force a route by spelling its name.
The sidecar also avoids logging prompt content and returns only a generic error
if model inference fails.

## Evaluation method

The committed benchmark contains 36 cases:

- 24 ordinary scored cases, balanced across the four targets;
- 8 adversarial scored cases that contain route-name or instruction-injection
  attempts; and
- 4 intentionally ambiguous, unscored cases used to inspect behavior.

Both classifiers received the same request text, the same four semantic class
definitions, and the same expected labels. The CPU GLiNER and Jev comparisons
ran every case three times after warmup; the GPU run used ten repeats per case.
Accuracy uses the modal prediction. Latency is end-to-end request latency
measured by the benchmark client.

The GLiNER test used the public `fastino/gliner2.5-base-v1` checkpoint and the
GLiNER2 classification API. The TypeSafe test used a single `Choice` question
with Jev. Credentials and service locations are deliberately absent from the
benchmark artifacts and this report.

The two systems express guidance differently. GLiNER received four internal
label names plus their descriptions. Jev received those same labels and
descriptions plus a Choice instruction to classify the actual requested work
and ignore route-selection text inside the request. That is the natural API for
each system, but it means this is an application-level comparison rather than a
controlled comparison of identical model inputs.

## Classification results

| Classifier | Overall | Ordinary | Adversarial | Stable across repeats |
| --- | ---: | ---: | ---: | ---: |
| TypeSafe Jev | 31/32 (96.9%) | 24/24 (100%) | 7/8 (87.5%) | 35/36 (97.2%) |
| GLiNER, no confidence fallback | 25/32 (78.1%) | 21/24 (87.5%) | 4/8 (50.0%) | 36/36 (100%) |
| GLiNER, confidence below 0.55 sent to review | 23/32 (71.9%) | 19/24 (79.2%) | 4/8 (50.0%) | 36/36 (100%) |

Jev missed one adversarial case in the three-repeat run: a complex architecture
request prefixed with an instruction to choose the deterministic route. A quoted
route-injection case varied across Jev repeats but the modal prediction was
correct.

GLiNER's seven raw misses were:

| Case | Expected | Predicted | Confidence |
| --- | --- | --- | ---: |
| Date extraction | `small_model` | `deterministic_tool` | 1.000 |
| Sentiment labeling | `small_model` | `deterministic_tool` | 0.598 |
| Architecture tradeoff | `reasoning_model` | `small_model` | 0.848 |
| Route-name injection on rewrite | `small_model` | `human_review` | 0.722 |
| Internal-label injection on lookup | `deterministic_tool` | `reasoning_model` | 0.999 |
| Internal-label injection on authority | `human_review` | `small_model` | 0.998 |
| Quoted injection to summarize | `small_model` | `human_review` | 0.954 |

Several GLiNER errors had very high reported confidence. A global confidence
threshold therefore did not improve this suite: it converted two correct,
low-confidence answers into `human_review` without catching the confident
errors. The example leaves the threshold configurable but disabled. A real
deployment should calibrate per-route policy on representative data and should
never treat classifier confidence as authorization for a consequential action.

## Latency

| Deployment | Mean | Median | p95 | p99 | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLiNER local CPU, loopback HTTP | 54.8 ms | 54.6 ms | 56.5 ms | 59.2 ms | 108 |
| GLiNER single A100 80 GB GPU, loopback HTTP | 21.7 ms | 20.1 ms | 21.0 ms | 21.8 ms | 360 |
| TypeSafe Jev, remote API | 276.4 ms | 266.3 ms | 365.0 ms | 542.8 ms | 108 |

These latency rows are intentionally labeled by topology. The GLiNER numbers
measure a deployment-local sidecar over loopback. The Jev numbers include the
network path to a managed service. They answer the practical router-overhead
question, but they do not isolate model execution time and should not be read as
a hardware-normalized model comparison.

At the median, the GPU deployment was 2.7 times faster than local CPU GLiNER
and 13.3 times faster than the measured Jev remote round trip. One GPU request
took 366.7 ms despite warmup, which raised the mean while leaving p99 at
21.8 ms. The full-sample result is retained rather than discarding that outlier.
The 108 measured Jev requests consumed 43,242 input tokens and 4,860 output
tokens, excluding warmup calls.

## Switchyard integration test

The example configuration loaded successfully on the current upstream
Switchyard server. With the real GLiNER sidecar and a mock OpenAI-compatible
completion backend, end-to-end requests selected and served all four configured
targets:

| Request kind | Selected target |
| --- | --- |
| Exact calculation | `deterministic-tool` |
| Polite rewrite | `small-model` |
| Multi-step distributed-systems diagnosis | `reasoning-model` |
| Consequential financial approval | `human-review` |

This verifies the full path—Switchyard classifier call, structured response,
policy selection, and downstream dispatch—not only direct GLiNER inference.

## Operational findings

- The sidecar keeps GLiNER and PyTorch out of Switchyard's core dependency
  graph and can be scaled or replaced independently.
- A process-level lock serializes inference because the example prioritizes a
  simple, safe contract. Production throughput work should evaluate batching,
  multiple workers, and concurrent GPU execution.
- `classify_trigger = "user_turn"` avoids reclassifying tool continuations when
  callers provide a session identifier.
- The model checkpoint is loaded once at startup. Cold-start download and model
  initialization are excluded from warm latency.
- Switchyard remains the authority for schema validation and target fallback.
  The classifier never directly invokes a model or a consequential action.

## Limitations

- Thirty-two scored synthetic cases are useful for a smoke test, not a
  production quality claim.
- The class definitions were written for this experiment and were not tuned on
  private production traffic.
- The test measures single-request latency, not saturated throughput or
  multi-tenant tail latency.
- TypeSafe and GLiNER confidence values have different definitions and are not
  numerically interchangeable.
- No GLiNER fine-tuning was performed. Supervised route data may materially
  change both quality and calibration.

## Recommendation

Keep GLiNER as an optional sidecar example rather than a mandatory Switchyard
dependency. It is a credible low-latency router for teams willing to evaluate
and tune it on their domain. For zero-shot routing on the present suite, Jev is
the stronger default: it gained 18.8 percentage points overall and 37.5 points
on adversarial cases. Before production use, expand the dataset, define the
cost of each confusion pair, tune or fine-tune GLiNER, calibrate per-route
fallback policy, and add concurrency and soak benchmarks.

## Reproduction

The example directory contains:

- `server.py`: the OpenAI-compatible GLiNER sidecar;
- `routes.json`: semantic labels, Switchyard target mappings, and optional
  confidence fallback;
- `switchyard.toml`: a four-target custom-router configuration;
- `benchmark.py`: the local GLiNER suite and latency harness;
- `benchmark_typesafe.py`: the same suite through TypeSafe Jev; and
- `test_server.py`: hermetic sidecar contract tests using a fake classifier.

The benchmark scripts write JSON when passed `--output`, making every per-case
decision and latency sample available for inspection without storing secrets.

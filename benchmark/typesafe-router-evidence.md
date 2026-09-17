# TypeSafe router exploratory evidence

Date: 2026-09-16

This is a small live smoke comparison, not a statistically meaningful benchmark.
It checks whether a TypeSafe System One `Choice` can separate clear efficient-tier
and capable-tier requests, and compares routing-decision latency with a conventional
generative LLM classifier.

## Setup

- TypeSafe model: `jev-latest`
- TypeSafe interface: System One HTTP API
- Generative classifier: `azure/openai/gpt-5.6-sol`
- Generative endpoint: NVIDIA Inference API, OpenAI Responses format
- Samples: 10 total; 5 labeled `efficient`, 5 labeled `capable`
- Repetitions: one live request per classifier and sample
- Timing: local wall time for the routing decision only
- TypeSafe confidence policy: choices below `0.60` fall back to `capable`
- Region, backend load, connection reuse, and cold starts were not controlled

Both classifiers used the same tier definitions:

- `efficient`: routine or well-specified work where speed and efficiency matter,
  including direct questions, summaries, straightforward edits, and ordinary coding tasks;
- `capable`: complex, ambiguous, high-stakes, or deeply multi-step work requiring
  stronger planning, architecture, debugging, or judgment.

TypeSafe received the request as structured state and answered one typed `Choice`.
The generative classifier received the definitions and request as text and was
instructed to emit exactly `efficient` or `capable`. The final target model call was
excluded from both measurements.

## Results

| Router | Correct | Mean | Median (p50) | Observed p95 |
|---|---:|---:|---:|---:|
| TypeSafe `jev-latest` | 10/10 | 281 ms | 278.5 ms | 375 ms |
| Generative `gpt-5.6-sol` | 10/10 | 1,653 ms | 1,575 ms | 2,576 ms |

On this run, TypeSafe was 5.9x faster by arithmetic mean, 5.7x faster at the
median, and 6.9x faster at the observed p95. “Observed p95” is the slowest sample
in a set of ten and should not be interpreted as a stable production percentile.

### Per-case results

| Expected tier | Request | TypeSafe choice | Confidence | TypeSafe | SOL choice | SOL |
|---|---|---|---:|---:|---|---:|
| efficient | Summarize this paragraph in three bullet points. | efficient | 1.00 | 375 ms | efficient | 1,410 ms |
| efficient | Rename `userID` to `user_id` in this Python function. | efficient | 1.00 | 252 ms | efficient | 2,576 ms |
| efficient | Translate “good morning” into Spanish. | efficient | 1.00 | 249 ms | efficient | 1,388 ms |
| efficient | Explain what an HTTP 404 status means. | efficient | 1.00 | 266 ms | efficient | 1,411 ms |
| efficient | Add a null check before reading `account.name`. | efficient | 1.00 | 295 ms | efficient | 1,681 ms |
| capable | Design a zero-downtime migration from a monolith to multi-region services while preserving consistency. | capable | 1.00 | 277 ms | capable | 1,567 ms |
| capable | Diagnose an intermittent distributed deadlock using partial traces and propose instrumentation to prove the cause. | capable | 1.00 | 284 ms | capable | 1,583 ms |
| capable | Audit an authentication architecture for privilege-escalation paths and redesign the trust boundaries. | capable | 1.00 | 280 ms | capable | 1,828 ms |
| capable | Prove whether a lock-free queue remains linearizable under relaxed memory ordering. | capable | 0.99 | 285 ms | capable | 1,599 ms |
| capable | Find nondeterministic CUDA memory corruption across eight GPUs and design a minimal reproducer. | capable | 1.00 | 249 ms | capable | 1,491 ms |

No TypeSafe answer fell below the `0.60` confidence threshold, so the conservative
fallback did not change any result in this sample.

## Interpretation

The router worked correctly on all ten deliberately clear cases and added roughly
0.28 seconds of routing latency in this environment. It was materially faster than
using a general-purpose generative model as the classifier, which added roughly
1.65 seconds on average. This supports TypeSafe as a promising low-latency routing
primitive for Switchyard.

It does **not** establish production routing quality. The labels were authored for
this smoke test, the cases were intentionally separable, the sample is small, and
each request ran only once. The high TypeSafe confidence values are useful output,
not proof that every individual prediction is correct. TypeSafe documents confidence
as a statistic derived from the returned probability distribution and recommends
tuning thresholds against the target domain and consequences.

## Recommended next evaluation

Before enabling the router for real traffic:

1. Sample at least several hundred representative Switchyard requests, including
   ambiguous, underspecified, adversarial, multilingual, tool-heavy, and long-context cases.
2. Label the minimum model tier that produces an acceptable answer using a blinded rubric;
   routing labels based only on perceived prompt difficulty can be misleading.
3. Run repeated, randomized comparisons and report bootstrap confidence intervals for
   acceptable-answer rate, over-routing, under-routing, latency, and end-to-end cost.
4. Tune the TypeSafe criteria and confidence threshold on a development split, then
   report final results once on a held-out split.
5. Use a proposed fail-open policy for a future TypeSafe integration: low-confidence
   decisions and service failures should route to the capable tier. Switchyard's existing
   [`StageClassifier`](../crates/libsy/src/algorithms/util/stage.rs) falls back to the tier
   selected by its configured `PickerMode`; TypeSafe-specific failure handling is not
   implemented here.

Relevant TypeSafe guidance:

- [System One](https://docs.typesafe.ai/concepts/system-one)
- [Intent routing](https://docs.typesafe.ai/patterns/intent-routing)
- [Confidence](https://docs.typesafe.ai/confidence)

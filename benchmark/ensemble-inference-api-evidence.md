# Ensemble exploratory evidence: SOL + Opus versus Astra

Date: 2026-09-09

This is a small live smoke comparison, not a statistically meaningful benchmark.
It verifies that response-level ensemble routing works against heterogeneous
production endpoints and records the behavior that informed the initial defaults.

## Setup

- API: OpenAI Responses through an authenticated model gateway (endpoint omitted)
- Baseline: `azure/openai/gpt-6-astra`
- Ensemble candidates: `azure/openai/gpt-5.6-sol` and
  `azure/anthropic/claude-opus-5`
- Ensemble synthesizer: `azure/openai/gpt-5.6-sol`
- Candidate cap: 2,048 output tokens
- Required usable candidates: 2
- Final cap: 1,024 output tokens
- Reproduction config:
  [`server-configs/inference-api-ensemble-sol-opus-vs-astra.toml`](server-configs/inference-api-ensemble-sol-opus-vs-astra.toml)

Each comparison sent the same prompt to the `fusion` route first and the `astra`
route second. Network load, backend load, model sampling, and cold starts were not
controlled. Wall time was measured at the local HTTP client.

## Initial smoke results

| Case | Route | Status | Wall time | Visible words | Terminal response tokens |
|---|---|:---:|---:|---:|---:|
| First-success design | SOL + Opus → SOL | complete | 29.89 s | 209 | 612 |
| First-success design | Astra | complete | 56.15 s | 214 | 945 |
| Broken fan-out review | SOL + Opus → SOL | complete | 46.29 s | 201 | 740 |
| Broken fan-out review | Astra | complete | 185.28 s | 209 | 925 |

Both ensemble requests completed with `minimum_successful_candidates = 2`, so
SOL and Opus each contributed usable output before synthesis. This prevents the
comparison from silently degrading into a single-model answer. In these two
uncontrolled observations, fusion was about 1.9x and 4.0x faster than Astra.

### Quality assessment

Both routes passed the same manual checklist: complete response, requested word
limit, correct completion-order concurrency, loser cancellation without detached
work, useful all-failed error handling, and implementation-level pseudocode. The
fusion review explicitly caught insertion-order waiting, detached `JoinHandle`s,
discarded backend and join errors, and showed `JoinSet::shutdown().await` to abort
and join losers. Astra was also technically strong. These two samples support
comparable quality with lower observed latency, not a claim that fusion is more
capable than Astra. A quality-superiority claim needs a larger blinded rubric or
task benchmark.

## Expanded blinded quality comparison

A follow-up comparison used six checkable cases spanning Rust concurrency,
Python debugging, Bayesian arithmetic, Boolean logic, strict JSON formatting,
and transactional-outbox design. For each case, the `fusion` and `astra` requests
started together. Their answers were assigned alternating anonymous A/B labels
and judged by `azure/openai/gpt-5.6-terra`, which was neither the baseline nor an
ensemble candidate. The judge scored correctness, completeness, instruction
following, and clarity from 1–5 against a case-specific rubric. Deterministic
checks such as word counts and exact JSON took precedence over the model judge.

| Case | Fusion time | Astra time | Fusion score | Astra score | Blind result |
|---|---:|---:|---:|---:|:---:|
| Tokio first-success review | 33.27 s | 125.30 s | 20/20 | 20/20 | tie |
| Python LRU debugging | 12.46 s | 16.49 s | 20/20 | 20/20 | tie |
| Two-test Bayesian posterior | 10.66 s | 10.27 s | 20/20 | 20/20 | tie |
| Boolean truth-value derivation | 10.33 s | 19.85 s | 20/20 | 20/20 | tie |
| Exact JSON sorting | 6.10 s | 3.72 s | 20/20 | 20/20 | tie |
| Transactional outbox | 28.13 s | 46.92 s | 20/20 | 20/20 | tie |
| **Arithmetic mean** | **16.83 s** | **37.09 s** | **20/20** | **20/20** | **6 ties** |
| **Median** | **11.56 s** | **18.17 s** | — | — | — |

Fusion was faster in four of six cases. Astra was slightly faster on the short
Bayes calculation and the trivial exact-JSON transform, where three ensemble
calls offer little benefit. All twelve final answers completed successfully.

The quality pass exposed a defect that the judge missed: the first fusion outbox
answer contained 224 whitespace-delimited words against a 220-word maximum.
The default synthesis instruction was hardened to leave a safety margin below
hard limits. Its rerun contained 219 words, preserved every rubric requirement,
and again received a blind 20/20 tie. This corrected rerun is the row reported
above.

This remains an exploratory six-case comparison with one automated judge, not a
statistically powered capability benchmark. It supports comparable quality on
these cases and shows where ensemble overhead is wasteful; it does not establish
general superiority over Astra. A release claim should use a larger randomized
task set, multiple independent judges, repeated samples, and end-to-end cost
accounting.

The first-success answers covered concurrent polling, first observed success,
loser cancellation, typed all-failed errors, structured task lifetime, replayable
bodies, timeout policy, and simultaneous completions. The code-review answers
both identified insertion-order waiting and detached Tokio tasks. The ensemble
answer additionally showed `JoinSet::abort_all` followed by draining every task;
the Astra answer used owned futures and explained that dropping local work cannot
cancel remote side effects.

`Terminal response tokens` is the usage returned for the final visible response.
It excludes the two candidate calls, so it must not be used as an ensemble cost
comparison. The ensemble makes three calls per request and should be assumed more
expensive than the one-call Astra baseline until end-to-end candidate usage is
captured.

## Prompt 1: first-success design

> In at most 220 words, explain how a Rust/Tokio gateway should send the same
> request to three backends concurrently and return the first successful
> response. Include concise implementation-level pseudocode. It must cancel
> losing calls, preserve a useful typed error if every backend fails, and never
> leave detached tasks. End with a short list of the key concurrency invariants
> and edge cases. The answer must be complete and stay under 220 words.

## Prompt 2: broken fan-out review

> In at most 220 words, review this Rust/Tokio pseudocode against the requirement
> “return the first successful backend response and cancel all losers without
> detached work.” Identify the important correctness and lifecycle bugs, then
> show a corrected pattern. The answer must be complete.

The reviewed function spawned one task per backend, stored the handles in a
vector, awaited them in insertion order, returned on the first successful await,
and otherwise returned an untyped `AllFailed` error.

## Hardening evidence

An earlier configuration used Opus 5 as synthesizer and forwarded uncapped
candidate outputs including reasoning. On the first-success prompt it repeatedly
exhausted the 1,024-token cap and returned an incomplete or empty visible answer.
Using SOL as synthesizer worked, and the implementation was then hardened to:

- buffer internal candidate calls;
- budget candidate output independently from the final response;
- remove reasoning blocks before synthesis;
- optionally require more than one usable candidate before synthesis;
- leave margin below caller-specified hard length limits;
- require concise, complete synthesis by default while allowing prompt override;
- preserve the caller's final streaming and output settings.

The relevant unit tests verify concurrent fan-out, partial candidate failure,
all-candidate failure, candidate caps, reasoning removal, exact-replay
invalidation, and configuration validation. A server integration test sends an
ensemble request through the public OpenAI-compatible endpoint and verifies both
candidate calls, independent candidate budgets, both drafts in the synthesis
request, the final caller budget, and the synthesized response.

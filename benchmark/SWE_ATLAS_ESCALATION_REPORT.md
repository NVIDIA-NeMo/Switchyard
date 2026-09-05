# SWE-Atlas escalation-router findings

## Summary

The escalation router could interpret duplicate command serialization in a NeMo Gym Terminus
trajectory as repeated failed work. A single assistant turn may contain the same command both as
raw JSON text and as a structured tool call. The judge prompt previously treated a command shown
two or more times as loop evidence without requiring those attempts to occur in distinct turns.

This change makes escalation depend on fresh, consistent failure evidence across turns. It keeps
the existing one-way switch and latch: a session can move from the efficient model to the capable
model at most once and remains there afterward.

## Changes

- Normalize only the transcript sent to the escalation judge. When a raw Terminus command batch
  has a one-to-one match with structured tool calls in the same assistant turn, the redundant raw
  representation is omitted. The worker model still receives the original history.
- Require the judge to classify positive evidence as `repetition`, `false_progress`, `drift`,
  `desperation`, or `capability_gap`, and report whether the evidence is new.
- Advance the confirmation streak only for fresh positive evidence in the same category. A
  category change starts a new streak; a decline or stale evidence resets it. An unavailable judge
  preserves the prior streak rather than converting an infrastructure failure into a routing
  decision.
- Clarify in the judge prompt that duplicate serialization within one turn and an immediate,
  adaptive change of terminal strategy are not repeated failed attempts.
- Preserve the existing maximum-one-switch and post-switch latch behavior.

## Live SWE-Atlas observations

The live checks used GLM 5.2 as the efficient model and Opus 4.8 as the capable model. The reward
column is the task's binary verifier result. "GLM only" means that the patched router did not
switch models during the trajectory.

| Dataset | Task | Patched behavior | Reward | Observation |
| --- | --- | --- | ---: | --- |
| RF | `697e7458be1623d850a88838` | 28 GLM calls, then 36 Opus calls; one switch | 1 | Two fresh `repetition` findings confirmed the escalation path and latch. |
| RF | `69391d8d1ce51c407be1e531` | GLM only; 93 logged episodes | 0 | Findings changed from `repetition` to `false_progress`, so they did not form one confirmation streak. The earlier router switched without improving the reward. |
| RF | `694b4b99829f00e24fd11889` | GLM only; 34 logged episodes | 0 | No positive escalation finding. |
| TW | `6902ef3ab97fe23e2ad271f3` | GLM only; 58 logged episodes | 0 | One isolated `repetition` finding did not trigger a switch. The earlier router switched without improving the reward. |
| RF | `696719205599a51110d4b45f` | Three GLM-only runs: 72, 93, and 70 logged episodes | 0, 0, 0 | One isolated positive finding across the replicas; no latch. Direct GLM and direct Opus also scored 0. |
| RF | `696719205599a51110d4b455` | Three GLM-only runs: 51, 60, and 48 logged episodes | 0, 0, 0 | No transition in any replica. Direct GLM and direct Opus also scored 0. |

A separate passing run of `697e7458be1623d850a88838` produced a positive finding followed by a
decline, stayed on GLM, and scored 1. This checks that non-consecutive findings do not accumulate.

The positive switched task also scored 1 in a direct-GLM run. It therefore demonstrates correct
switching, confirmation, and latching, but does not establish a causal accuracy improvement.
Similarly, the unsuccessful controls show improved escalation precision and avoided Opus calls;
they do not show that the efficient model could solve those tasks.

## Validation evidence

The implementation was validated with Rust 1.96.1 on a Slurm compute node using:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
uv run ruff check .
uv run mypy switchyard
uv run maturin develop --uv
uv run pytest tests/ -v
```

The validation completed successfully: the Rust workspace passed, including 289 `libsy` tests;
ruff and mypy passed; the native wheel built and installed; and pytest reported 117 passed tests.
Focused regression coverage includes exact and multi-command transcript normalization, partial and
multiplicity mismatches, category changes, stale evidence, unavailable-judge behavior, and the
one-way latch.

No live provider calls are made by the repository validation commands. The live SWE-Atlas checks
described above were separate, explicitly configured benchmark runs.

## Conclusion

The evidence supports a narrower conclusion than an aggregate benchmark improvement: the router
now distinguishes repeated cross-turn failure from within-turn serialization noise, avoids several
observed unnecessary escalations, and still performs a confirmed one-time escalation when fresh
same-category evidence persists. A larger paired run is required to estimate accuracy and cost
effects with confidence.

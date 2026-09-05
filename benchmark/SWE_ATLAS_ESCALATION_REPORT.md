# SWE-Atlas escalation-router findings

## Summary

The escalation router could interpret duplicate command serialization in a NeMo Gym Terminus
trajectory as repeated failed work. A single assistant turn may contain the same command both as
raw JSON text and as a structured tool call. The judge prompt previously treated a command shown
two or more times as loop evidence without requiring those attempts to occur in distinct turns.

This change makes escalation depend on fresh, consistent failure evidence across turns. It keeps
the existing one-way switch and latch: a session can move from the efficient model to the capable
model at most once and remains there afterward.

On a deterministic 40-task paired sample (20 SWE-Atlas RF and 20 SWE-Atlas TW), the patched router
matched the two single-model arms at 15/40 correct and improved over the original router's 10/40.
It switched 5 times instead of 9. Under the synthetic token prices defined below, it cost
$0.889/task versus $1.171/task for the original router. GLM-only also scored 15/40 and remained the
least expensive arm at $0.501/task, so the result supports the patch over the original router but
does not show that routing beats the best fixed model on this sample.

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

## Focused SWE-Atlas reproducer observations

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

## Paired benchmark

### Design

The paired benchmark used the following fixed setup:

- 20 tasks from `scale-ai/swe-atlas-rf@1` and 20 from `scale-ai/swe-atlas-tw@1`.
  Tasks were selected deterministically by ranking the hash of
  `20260904:<dataset>:<task>`; they were not selected by outcome.
- Four arms per task: direct GLM, direct Opus, the original escalation router, and the patched
  escalation router. This produced 160 task-arm runs.
- GLM was `nvidia/zai-org/glm-5.2`; Opus was
  `aws/anthropic/bedrock-claude-opus-4-8` with medium effort. GLM also served as the escalation
  judge. The two escalation arms used two confirmations, maximum one switch, and a post-switch
  latch.
- The original router was built from upstream commit
  `9a743e89223a0d5b14011f1226d5b068f730a3b8`; the patched router was built from
  `507376b1d851b825378cf147d09a8b5594773298`.
- There was no agent-turn limit. Worker requests allowed up to 128,000 output tokens and a
  900-second model-server timeout. Five initial runs reached Harbor's separate 3,600-second agent
  timeout. Only those five task-arm runs were repeated with a 3x agent-timeout multiplier; all five
  retries completed and replaced the invalid attempts. The final effective matrix is 160/160 valid.
- Each table entry is one trial per task and arm. Binary verifier reward is reported as correct.
  Because model sampling is stochastic, differences between unswitched GLM router runs and direct
  GLM runs are not necessarily routing effects.

### Accuracy and routing

| Dataset | Arm | Correct | Accuracy | Switches | Correct after switch |
| --- | --- | ---: | ---: | ---: | ---: |
| RF (20) | Direct GLM | 7 | 0.350 | -- | -- |
| RF (20) | Always Opus | 9 | 0.450 | -- | -- |
| RF (20) | Original escalation | 6 | 0.300 | 7 | 2 |
| RF (20) | **Patched escalation** | **9** | **0.450** | **4** | **2** |
| TW (20) | Direct GLM | 8 | 0.400 | -- | -- |
| TW (20) | Always Opus | 6 | 0.300 | -- | -- |
| TW (20) | Original escalation | 4 | 0.200 | 2 | 0 |
| TW (20) | **Patched escalation** | **6** | **0.300** | **1** | **0** |
| Combined (40) | Direct GLM | 15 | 0.375 | -- | -- |
| Combined (40) | Always Opus | 15 | 0.375 | -- | -- |
| Combined (40) | Original escalation | 10 | 0.250 | 9 | 2 |
| Combined (40) | **Patched escalation** | **15** | **0.375** | **5** | **2** |

Against the original router on the same 40 tasks, the patched router won 7 outcomes, lost 2, and
tied 31. Against direct GLM it won 5 and lost 5; against direct Opus it won 6 and lost 6. All 14
switched trajectories across the two router arms had exactly one GLM-to-Opus transition and no
hand-back, confirming the switch and latch invariants in live execution.

The patch traded earlier sensitivity for precision. Original-router switches occurred after a
median of 23 GLM calls; patched-router switches occurred after a median of 51. One patched RF retry
did not switch until 141 GLM calls, then made 57 latched Opus calls and still scored 0. Direct Opus
and the original router also scored 0 on that task. This avoided a premature switch but exposed a
cost problem: judging every turn can be expensive when the destination model is unlikely to help.

### Synthetic cost calculation

The inference service did not provide billable cost. The values below are synthetic estimates for
comparing routing behavior, not NVIDIA prices or charges. The assumed rates are:

| Model | Uncached input / 1M tokens | Cached input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: | ---: |
| GLM 5.2 | $0.50 | $0.05 | $2.00 |
| Opus 4.8 | $5.00 | $0.50 | $25.00 |

For each model in each trajectory:

```text
uncached_input_tokens = max(prompt_tokens - cached_tokens, 0)

model_cost = (
    uncached_input_tokens * uncached_input_rate
    + cached_tokens * cached_input_rate
    + output_tokens * output_rate
) / 1,000,000

trajectory_cost = sum(worker_model_costs) + sum(escalation_judge_costs)
```

Cache-creation tokens are part of the non-cached prompt-token remainder and therefore receive the
uncached-input rate. The calculation includes GLM judge calls for the two router arms. It excludes
the Harbor verifier, cluster resources, and other infrastructure because those calls are outside
Switchyard's per-trajectory statistics.

| Dataset | Arm | Synthetic cost | Cost/task | Cost/correct |
| --- | --- | ---: | ---: | ---: |
| RF (20) | Direct GLM | $12.953 | $0.648 | $1.850 |
| RF (20) | Always Opus | $32.196 | $1.610 | $3.577 |
| RF (20) | Original escalation | $35.169 | $1.758 | $5.862 |
| RF (20) | **Patched escalation** | **$27.200** | **$1.360** | **$3.022** |
| TW (20) | Direct GLM | $7.093 | $0.355 | $0.887 |
| TW (20) | Always Opus | $14.829 | $0.741 | $2.471 |
| TW (20) | Original escalation | $11.657 | $0.583 | $2.914 |
| TW (20) | **Patched escalation** | **$8.356** | **$0.418** | **$1.393** |
| Combined (40) | Direct GLM | $20.045 | $0.501 | $1.336 |
| Combined (40) | Always Opus | $47.025 | $1.176 | $3.135 |
| Combined (40) | Original escalation | $46.826 | $1.171 | $4.683 |
| Combined (40) | **Patched escalation** | **$35.556** | **$0.889** | **$2.370** |

The patched router was 24.1% cheaper than the original router and produced five additional correct
outcomes. It was 24.4% cheaper than always-Opus at the same aggregate accuracy. It was 77.4% more
expensive than direct GLM at the same aggregate accuracy. Of the patched router's $35.556 synthetic
total, $3.698 (10.4%) came from judge calls; this motivates reducing judge cadence after repeated
negative verdicts.

### TW model-order reversal

TW demonstrates a limitation that this patch does not solve: the escalation route assumes its
configured Opus target has positive expected gain over GLM. On this sample, direct GLM scored 8/20
while direct Opus scored 6/20, so that assumption is not valid at the workload level.

The patched router switched on only one TW task. Direct GLM, direct Opus, and both router arms all
scored 0 on that task, so the switch added cost without changing its outcome. The patched router's
two-correct deficit versus direct GLM was not caused by switching: it lost three independently
sampled GLM-only outcomes and gained one different GLM-only outcome. The original router provides
a clearer harmful-switch example on task `6902ef3ab97fe23e2ad2727e`: direct GLM scored 1, direct Opus
scored 0, and the original router switched and scored 0. The patch avoided that switch, although its
independent GLM-only run also scored 0.

The next routing change should therefore be separate from the trajectory-evidence fix: gate
escalation on a workload- or task-conditioned estimate of expected accuracy gain and cost. When
recent paired evidence says the nominal strong model is worse, the route should disable or reverse
that transition rather than asking only whether the current trajectory looks stuck. This requires
held-out calibration or online exploration; the router cannot infer relative model quality from one
model's failing trajectory alone.

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
ruff and mypy passed; the native wheel built and installed; and pytest reported 115 passed,
2 deselected, and 2 subtests passed.
Focused regression coverage includes exact and multi-command transcript normalization, partial and
multiplicity mismatches, category changes, stale evidence, unavailable-judge behavior, and the
one-way latch.

No live provider calls are made by the repository validation commands. The live SWE-Atlas checks
described above were separate, explicitly configured benchmark runs.

## Conclusion

The patch fixes the concrete duplicate-serialization false positive and materially improves the
tested router: it matches the single-model aggregate accuracy, gains 5 correct outcomes over the
original router, cuts switches from 9 to 5, and reduces synthetic cost by 24.1%. It does not make
the router the best cost/accuracy arm; direct GLM matches its aggregate accuracy at substantially
lower cost. The remaining work is not to loosen the new evidence rule globally. It is to add
model-order calibration for workloads such as TW and reduce judge overhead or stop judging when
escalation has low expected value. Replicated trials on held-out tasks are needed before treating
the point estimates as stable production gains.

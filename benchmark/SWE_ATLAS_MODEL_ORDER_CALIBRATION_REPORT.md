# SWE-Atlas model-order calibration findings

## Summary

The escalation router can decide that a trajectory needs rescue, but that signal does not prove
that the configured capable target is the better rescue model. The paired sample reported in
[PR #637](https://github.com/NVIDIA-NeMo/Switchyard/pull/637) made the distinction concrete: Opus
outscored GLM on RF, while GLM outscored Opus on TW. A trajectory-only router always interprets
rescue as GLM-to-Opus and cannot represent the second ordering.

This experiment adds an optional deployment calibration gate named `expected_capable_gain`. A
positive value enables the existing trajectory judge. Zero or a negative value routes directly to
the efficient target, avoids judge calls, and prevents a judge-driven transition to a target whose
expected utility is not positive. Omitting the value preserves existing behavior.

The live experiment verifies the mechanism, not the quality of the calibration estimate. With a
TW-derived value of `-0.10`, all 20 calibrated held-out trajectories stayed on GLM: 800 worker
calls, zero judge calls, and zero Opus calls. They scored 8/20. However, independently sampled
GLM-only arms ranged from 4/20 to 9/20, and held-out direct Opus beat direct GLM 7/20 to 4/20—the
opposite ordering from the calibration sample. A static gate is therefore a useful safety
constraint when its input is reliable, but these 20-task samples are too noisy to establish a
stable TW model order.

This is a narrow vertical slice of the unified-routing proposal in
[issue #601](https://github.com/NVIDIA-NeMo/Switchyard/issues/601), not the proposed `auto` route.
A complete solution needs pool-owned, task-conditioned, uncertainty-aware utility estimates.

## Problem statement

The escalation judge observes one model's trajectory. It can detect repeated failures, false
progress, drift, desperation, or a capability gap. It cannot observe the counterfactual result of
calling a different model. In particular:

- a failing GLM trajectory does not imply that Opus will solve the task;
- the labels `efficient` and `capable` do not establish workload-specific model ordering;
- switching may lower accuracy as well as increase cost when that ordering is reversed; and
- judge calls add cost even when deployment evidence says that switching has no expected value.

The two signals should remain distinct: trajectory evidence answers whether rescue is needed;
calibration answers whether the proposed rescue target has positive expected utility.

## Change

`EscalationJudgeConfig` accepts an optional finite `expected_capable_gain`:

```toml
[routes.calibrated_escalation]
id = "switchyard/calibrated-escalation"
type = "llm_classifier"
mode = "escalation"
classifier_target = "glm"
strong_target = "opus"
weak_target = "glm"
escalation = { confirmations = 2, expected_capable_gain = -0.10 }
```

| Calibrated gain | Behavior |
| --- | --- |
| Unset | Preserve the existing trajectory-judge behavior. |
| Greater than zero | Run the trajectory judge and allow its normal confirmed, one-way switch. |
| Zero or less | Serve the efficient target directly; do not call the judge or proactively select the capable target. |

The setting is available in native TOML and the Python binding. Non-finite values are rejected
during route construction. Existing configurations remain backward compatible. The gate controls
judge-driven escalation; normal transport and context-window fallback behavior remains available.

## Calibration rule

For this experiment, expected capable gain is the signed difference in deployment utility:

```text
expected_capable_gain =
    (p_capable - p_efficient)
    - lambda * (cost_capable - cost_efficient)
```

`lambda` expresses how the deployment trades one unit of task success against one synthetic
dollar. With `lambda = 0`, this reduces to the observed accuracy difference. On the original TW
sample in PR #637, direct GLM scored 8/20 and direct Opus scored 6/20, so the correctness-only
estimate was `0.30 - 0.40 = -0.10`. The configured gate therefore kept GLM and skipped the judge.

That original sample is calibration data, not evidence that the estimate generalizes. The
held-out experiment below tests the estimate on a disjoint task sample.

## Held-out experiment

### Design

- 20 tasks from `scale-ai/swe-atlas-tw@1`, disjoint from PR #637's 20-task TW sample.
- Tasks were selected deterministically by ranking the hash of
  `20260905:tw-selector:<task>`; they were not selected by outcome.
- One independent trial per task and arm. Comparison arms were direct GLM, direct Opus, patched
  escalation from PR #637, and the existing capability selector. A fifth arm used calibrated
  escalation with `expected_capable_gain = -0.10`.
- The live calibrated binary combined PR #637 with the directional gate so all five arms could use
  one server build. The nonpositive gate returned before the escalation judge, so PR #637's judge
  changes were not exercised by any calibrated trajectory. This PR's clean branch and validation
  are independently based on `main`.
- GLM was `nvidia/zai-org/glm-5.2`; Opus was
  `aws/anthropic/bedrock-claude-opus-4-8` with medium effort. GLM was also the classifier/judge.
- There was no agent-turn limit. Worker requests allowed up to 128,000 output tokens, the model
  server timeout was 900 seconds, and the Harbor agent-timeout multiplier was 3.
- Infrastructure failures before a valid graded trajectory were excluded and rerun. A final cell
  was accepted only when it had a verifier reward and at least one model request. Server-startup
  and container-registry failures never entered the result matrix; all 100 final cells were valid.

The exact task manifest, Switchyard configuration, and Slurm runner are checked into
`benchmark/tw-model-order/`.

### Accuracy and routing

| Arm | Correct | Accuracy | Judge/classifier calls | Opus worker trajectories |
| --- | ---: | ---: | ---: | ---: |
| Direct GLM | 4/20 | 0.200 | 0 | 0 |
| Always Opus | 7/20 | 0.350 | 0 | 20 |
| Patched escalation | 7/20 | 0.350 | 674 | 2 |
| Capability selector | 9/20 | 0.450 | 20 | 0 |
| **Calibrated escalation** | **8/20** | **0.400** | **0** | **0** |

The calibrated arm enforced the intended routing invariant in every trajectory: all 800 recorded
model calls were GLM worker calls. The capability selector also selected GLM on every task, but it
spent one GLM classifier call per trajectory. The patched router switched on two tasks. Each had
one GLM-to-Opus transition followed by a latch with no hand-back.

The accuracy results must not be read as a causal win for the gate. Direct GLM, calibrated
escalation, and the capability selector all used GLM for worker calls, yet scored 4/20, 8/20, and
9/20 respectively. The calibrated arm beat direct GLM on four tasks and lost none, despite having
the same worker model configuration. That gap is sampling and long-agent trajectory variance, not
a routing effect. Likewise, direct Opus's 7/20 versus direct GLM's 4/20 reverses the ordering that
produced the `-0.10` calibration value.

The paired direct baselines tied on 17 tasks; Opus alone solved three and GLM alone solved none.
That directional result is still based on only three discordant task outcomes and should not be
treated as a stable model-ranking estimate.

### Synthetic cost

The inference service did not provide billable cost. These values are synthetic comparisons, not
NVIDIA prices or charges. The assumed rates match PR #637:

| Model | Uncached input / 1M tokens | Cached input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: | ---: |
| GLM 5.2 | $0.50 | $0.05 | $2.00 |
| Opus 4.8 | $5.00 | $0.50 | $25.00 |

For every worker, classifier, or judge call:

```text
uncached_input_tokens = max(prompt_tokens - cached_tokens, 0)

model_cost = (
    uncached_input_tokens * uncached_input_rate
    + cached_tokens * cached_input_rate
    + output_tokens * output_rate
) / 1,000,000

trajectory_cost = sum(worker costs) + sum classifier/judge costs
```

Cache-creation tokens are included in the uncached prompt remainder. The calculation includes
classifier and judge calls and excludes the Harbor verifier and cluster infrastructure.

| Arm | Synthetic total | Cost/task | Cost/correct |
| --- | ---: | ---: | ---: |
| Direct GLM | $6.422 | $0.321 | $1.606 |
| Always Opus | $10.838 | $0.542 | $1.548 |
| Patched escalation | $12.479 | $0.624 | $1.783 |
| Capability selector | $5.036 | $0.252 | $0.560 |
| **Calibrated escalation** | **$7.520** | **$0.376** | **$0.940** |

Skipping 20 classifier calls does not guarantee a lower per-task total in independently sampled
agent runs. The calibrated trajectories happened to make 800 worker calls, versus 573 worker calls
plus 20 classifier calls for the capability selector, so their synthetic total was higher. A
paired replay or multiple replicas would be needed to isolate judge-cost savings from worker-call
variance.

The calibrated arm's synthetic total was 39.7% lower than patched escalation, but that comparison
also includes independently sampled worker trajectories. The patched arm spent $1.414 on 674 judge
calls. Its largest outlier, `task-6902ef3ab97fe23e2ad27253`, made 146 judge calls and 144 GLM
worker calls before two consecutive fresh `false_progress` verdicts triggered a late switch. It
then made 30 latched Opus calls, scored 0, and cost $7.392 by itself—59.2% of the patched arm's
total. The gate would remove judge and Opus spending under a negative utility estimate, but it
cannot guarantee a shorter independently sampled GLM trajectory.

## Relationship to unified routing

Issue #601 proposes that algorithms emit needs and confidence while a pool optimizer owns the
model catalog, cost, budget, and preferences. This change does not add the proposed model pool,
`auto` route, capability discovery, or learned/online policy. It establishes one decision boundary:

```text
trajectory rescue evidence
            +
externally calibrated model utility
            |
            v
allow or reject the proposed transition
```

A complete implementation should replace the scalar deployment constant with a pool-owned,
task-conditioned estimate, account for uncertainty in that estimate, and let the optimizer choose
any eligible model. It should preserve the current maximum-one-switch and latch behavior for a
coding-agent session unless a later design explicitly introduces safe hand-back.

## Limitations

- One trial per task has high variance for long stochastic coding-agent trajectories.
- Twenty paired tasks were insufficient to establish a stable TW model ordering: the held-out
  direct-model result reversed the calibration sample's direction.
- A workload average does not estimate Opus's conditional value specifically on trajectories the
  judge would escalate.
- A static calibration can become stale and does not include confidence intervals or exploration.
- The gate prevents a negative-value transition; it does not automatically reverse the transition,
  learn model capabilities, or solve model selection by itself.
- The client requested up to 128,000 output tokens, but the longest GLM trajectory repeatedly
  received length-terminated 32,768-token responses from the upstream service. No agent-turn cap
  was imposed; this provider behavior contributed to the long-tail trajectory described above.

## Validation

The implementation was tested with Rust 1.96.1 on a Slurm compute node using:

```text
uv sync --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
uv run ruff check .
uv run mypy switchyard
uv run maturin develop --uv
uv run pytest tests/ -v -m "not integration"
make -C docs publish
```

All commands passed on the clean branch. Pytest reported 116 passed, 2 deselected, and 2 subtests
passed; the strict MkDocs build also completed successfully.

Focused tests verify that a nonpositive calibrated gain consumes neither a judge response nor a
capable-model response, a positive gain preserves confirmed escalation, non-finite values are
rejected, and the TOML and Python surfaces carry the setting.

## Conclusion

The change solves the mechanical failure mode: Switchyard can now refuse an escalation when an
external utility estimate says the destination is not better, while preserving the existing route
when calibration is absent or positive. The held-out run verifies that the gate eliminates both
judge and Opus calls when disabled.

It does not establish that `-0.10` is the right TW policy. The direct baselines reversed ordering
on the held-out tasks, and nominally identical GLM worker arms differed by five correct outcomes.
The practical next step toward issue #601 is replicated calibration with uncertainty bounds,
followed by a pool-owned task-conditioned utility estimate—not more trajectory-judge prompt
tuning.

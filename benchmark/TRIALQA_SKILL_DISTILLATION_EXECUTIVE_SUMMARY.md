# TrialQA Skill-Distillation Executive Summary

As of 2026-07-08, we have not yet reproduced Sergei's TrialQA result with
Nemotron Ultra. We have turned the work into a local, Switchyard-only
experiment with a small paid canary as the next step.

The current blocker is explicit approval to run the first live generation
canary. That canary is 4 questions x 1 repeat x 2 arms, so it buys 8 model
calls and 0 judge calls. No command with `--yes-spend` has been run for the
current packet.

The current handoff file is:

```text
.experiments/trialqa-local/prospective/current-packet-ctgov-prospective-v1-q0-q3-r1.json
```

It selects compact packet v25 with bundle SHA-256:

```text
sha256:5f23c79d5a75792e91b35f8fab7067566c1da07f2f89ad8de7d4628d1948e358
```

## What changed

We stopped treating this as a long benchmark to babysit. The workflow now uses
small gates that prove one thing at a time before buying the next stage:

1. Run a zero-spend readiness and spend-guard check.
2. Buy only an 8-call generation canary after explicit approval.
3. Inspect the operational gate before any judge calls.
4. Kill if skill ON does not look cheaper.
5. Score only if the operational gate says `promote_to_score`.
6. Expand only through 8 x 1, 8 x 3, and 8 x 5 before considering a larger run.

The detailed operator runbook is
`benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md`.

## Reference result we are trying to match

The reference TrialQA result is not an Ultra/TrialQA result. It uses
Nemotron-3-Super/Nano on TrialQA. The relevant TrialQA Super result was:

| Condition | Accuracy | Tokens / trial | Op. calls / trial |
| --- | ---: | ---: | ---: |
| Super Base | 0.610 | 549K | 15.5 |
| Super R1 | 0.738 | 385K | 8.6 |

That is a quality gain with about 30% fewer tokens and 45% fewer operational
calls.

The defensible local claim is narrower: use Switchyard with Nemotron Ultra to
test whether the TrialQA skill-distillation method transfers to Ultra. We
should not claim that a published Ultra/TrialQA number exists.

## Benchmark results so far

We have three benchmark outcomes and one dataset audit. None of them is the
final demo result yet.

### 1. The first small pilot looked promising

We froze one early skill candidate and tested it on 8 TrialQA questions. Each
question ran twice: once with skill distillation OFF and once with skill
distillation ON.

Skill ON matched baseline quality and used much less work:

| Metric | Skill OFF | Skill ON | What changed |
| --- | ---: | ---: | ---: |
| Judge score | 0.625 | 0.625 | no quality loss |
| Total tokens | 11,774,093 | 3,394,608 | 71.2% fewer tokens |
| Tool calls | 135 | 66 | 51.1% fewer tool calls |

This is the strongest positive signal we have. It is still only a pilot. Eight
questions are enough to justify the next run, not enough to claim reproduction.

What we can share:

- "In an 8-question paired pilot, skill ON matched the baseline judge score and
  reduced tokens by 71.2% and tool calls by 51.1%."
- "This was a pilot result, not a final benchmark."

### 2. The next larger run found a runtime failure mode

We expanded the same candidate from 8 questions to 16 questions. That run did
not produce a valid quality result.

The reason was not that skill ON answered worse. Ten of the 32 total runs ended
with an empty final answer:

- 5 empty final answers happened with skill OFF.
- 5 empty final answers happened with skill ON.

The model stream ended normally from the provider's point of view, but Codex did
not receive a final answer it could score. Because those failures affected both
arms evenly, they did not show that skill distillation was worse. They did make
the run unusable as a benchmark result.

The run still showed cheaper skill-ON behavior where the tasks completed:

| Metric | Result |
| --- | ---: |
| Token reduction | 68.9% fewer tokens |
| Tool-call reduction | 44.7% fewer tool calls |
| Skill ON cheaper on token pairs | 10/16 |
| Skill ON cheaper on tool-call pairs | 10/16 |

What we can share:

- "The 16-question expansion exposed a balanced empty-final-answer failure in
  the runtime path."
- "The completed tasks still showed a cost reduction, but the run cannot be
  used as quality evidence."

What we should not share:

- We should not present the 16-question run as a successful benchmark.
- We should not average around the empty answers and call it a score.

### 3. The later official TrialQA suffix failed the efficiency test

We then used the remaining official local TrialQA held-out questions for the
same candidate. This was a stricter run with a cleaner completion policy.

This time the candidate failed for a simple reason: skill ON cost more.

| Scope | Decision | What happened |
| --- | --- | --- |
| First 4 questions | Continue to scoring | Small checkpoint only |
| Full 8-question suffix | Kill | Skill ON used more work |

At the full 8-question suffix:

| Metric | Skill OFF | Skill ON | What changed |
| --- | ---: | ---: | ---: |
| Total tokens | 7,412,360 | 14,226,358 | 91.9% more tokens |
| Tool calls | 147 | 239 | 62.6% more tool calls |

This is a clear negative result for that candidate. It tells us not to expand or
rescore it.

What we can share:

- "The first frozen candidate failed the later official TrialQA efficiency
  gate."
- "Skill ON used 91.9% more tokens and 62.6% more tool calls on that suffix, so
  we killed the candidate."

What we should not share:

- We should not treat this candidate as the final Switchyard demo candidate.
- We should not spend more on this candidate.

### 4. The official local TrialQA population is exhausted

The exposure audit found:

| Item | Result |
| --- | ---: |
| Local TrialQA rows | 120 |
| Sergei-style train rows | 24 |
| Sergei-style held-out rows | 96 |
| Rows with local generation artifacts | 120 |
| Official unseen TrialQA split available | false |

The saved upstream metadata says the official `trialqa` config has one `train`
split and one parquet file. That means a new prospective performance claim
cannot come from the existing 120 local rows.

## Current recovery path

We selected a new compact candidate using only retrospective evidence:

```text
trialqa-03de7576dd996510ef449b01c06f7d3c
```

Its skill SHA-256 is:

```text
e80f30e6431d1aafe748b3c1028f68b2fa1e20e9967ef156885b936967edd827
```

The candidate previously showed:

| Evidence | Result |
| --- | ---: |
| Mean score delta | +0.0417 |
| Token reduction | 64.8% |
| Operational-call reduction | 52.2% |
| Treatment cheaper on token pairs | 21/24 |
| Treatment cheaper on call pairs | 18/24 |

Those numbers only justify candidate selection. They do not count as new
performance evidence.

We also built a new TrialQA-compatible prospective population from
ClinicalTrials.gov metadata:

```text
.experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet
```

It has 8 rows, 8 unique NCT IDs, and no overlap with the 102 NCT IDs found in
the official TrialQA parquet. It must be labeled `official_labbench2: false`.

This keeps the run local and container-free. It uses Switchyard, Codex,
ToolUniverse MCP, the prospective parquet, and Nemotron Ultra through the
Switchyard routing profile. It does not require Docker, Harbor, BiomniBench, or
a second Hugging Face repository.

## Current packet state

The current packet says:

| Field | Value |
| --- | --- |
| Status | `ready_for_user_spend_decision` |
| Goal status | `ready_for_generation_spend_decision` |
| Spend authorized | `false` |
| Next stage | generation |
| Scope | q0-q3, 1 repeat, 2 arms |
| Expected model calls | 8 |
| Expected judge calls | 0 |
| Requires `--yes-spend` | `true` |

The current capture directory is:

```text
.experiments/trialqa-local/trialqa-full-16abb801ed9bdc82a010
```

It has no `ledger.jsonl` and no `batch.lock`, so no current-generation tasks
have started.

The packet still needs two pieces of live evidence:

1. a live generation operational gate that passes; and
2. a later quality-parity and efficiency gate that passes after scoring.

## Safety work completed

The local workflow now has spend guards and promotion rules that prevent stale
or accidental expansion:

- The current-packet helper selects the newest matching compact packet and
  reruns the no-spend spend guard.
- `--no-verify-guard` intentionally returns no guarded spend command.
- The generation canary refuses to run paid generation without `--yes-spend`.
- The post-generation gate can return `kill` or `promote_to_score`.
- The status logic refuses to mark the final prospective scope complete unless
  the promotion gate marks the confirmatory scope complete and performance
  eligible.
- The ladder rehearsal covers kill and promote states through q8 x 5.

The focused local checks passed:

```text
559 focused TrialQA tests passed
ruff check passed
mypy passed
git diff --check passed
```

## Recommended next action

Approve the 8-call generation canary only if you are ready to spend model
calls. After approval, run the current packet's pre-spend guard and then the
exact `.commands.guarded_spend` command from:

```text
.experiments/trialqa-local/prospective/current-packet-ctgov-prospective-v1-q0-q3-r1.json
```

After generation completes, run the packet's post-spend gate inspection.

Stop immediately if the gate returns `kill`. Continue to judge scoring only if
the gate returns `promote_to_score`, and ask for a separate spend approval
before buying any judge calls.

## Things not to do

- Do not update PR #16 for this work.
- Do not rerun, resume, or rescore the killed v11 captures.
- Do not claim the prospective ClinicalTrials.gov parquet is official LABBench2
  TrialQA.
- Do not run a full benchmark before the 8-call canary passes.
- Do not add `--yes-spend` by hand to a stale command block.

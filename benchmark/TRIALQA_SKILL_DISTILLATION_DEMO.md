# Nemotron Ultra LABBench2 TrialQA Skill-Distillation Demo

This is a container-free Switchyard transfer study. It runs the TrialQA protocol
locally with Nemotron Ultra behind Switchyard and compares the same agent with
skill distillation OFF and ON. The current fast path does not require Docker,
Harbor, BiomniBench, or a second Hugging Face repository. The official
`EdisonScientific/labbench2` (`trialqa` config) dataset is needed only for the
saved exposure/upstream-metadata audit; because that official local population
is already exhausted, the live canary below uses the generated
ClinicalTrials.gov TrialQA-compatible prospective parquet and labels it
`official_labbench2: false`.

The only remote runtime operations are model requests routed by Switchyard and
ToolUniverse calls to public scientific APIs. Orchestration, Codex,
ToolUniverse MCP, evidence storage, scoring records, and reports run locally.
Both arms expose identical read-only ToolUniverse schemas; only treatment sees
the immutable distilled skill.

## Current operator packet selector

As of 2026-07-08, the current no-spend handoff is
`.experiments/trialqa-local/prospective/current-packet-ctgov-prospective-v1-q0-q3-r1.json`.
Use its `.selected.path`, `.selected.version`, and `.selected.bundle_sha256`
fields as the authoritative packet identity; do not hardcode an older compact
version or bundle hash from prose.

The next live boundary is intentionally tiny: 4 questions, 1 repeat, 2 arms,
8 generation/model calls, and 0 judge calls. The packet does not authorize
spend. It is ready only for a user spend decision, and the guarded generation
command must be copied from `.commands.guarded_spend` only after the fresh
spend guard has passed and the user has explicitly approved `--yes-spend`.

The safe sequence is:

1. review the current packet and spend-review artifact;
2. run `.commands.pre_spend_guard_check` with no `--yes-spend`;
3. if and only if spend is explicitly approved, run `.commands.guarded_spend`;
4. monitor with `.commands.progress_monitor` rather than starting any larger
   benchmark;
5. inspect the operational gate with `.commands.post_spend_gate_inspection`;
6. stop on `kill`, or continue only on `promote_to_score`; and
7. run `.commands.post_spend_checkpoint` before any judge spend.

Without a fresh guard, `trialqa_local_current_packet --no-verify-guard` must
return `commands.guarded_spend: null`; this is the expected safety behavior,
not a missing command.

## Concrete goal

Recover a performance-eligible path to the demo after the v11 candidate failed
and the local TrialQA population was exhausted. The target result is still a
prospective, complete, paired intent-to-treat comparison of Nemotron Ultra with
skill distillation OFF versus ON. It must answer two bounded questions:

1. Does skill ON preserve Ultra answer quality within a predeclared
   5-percentage-point noninferiority margin?
2. Does skill ON reduce Ultra work by at least 15% in total tokens and 20% in
   TrialQA operational calls, with the predeclared paired robustness checks?

The previous prospective primary over positions 88-95 is now consumed and
failed the frozen efficiency gate at repeat 1. A passing demo now requires a
new prospective population, or else it must be explicitly labeled exploratory /
non-performance. A complete result that misses the frozen gates is an honest
negative result, not a reason to retry selected draws, revise the skill, or
search for a friendlier subset.

The fastest safe route is:

1. stop spending on v11 and preserve the kill as the current negative result;
2. use historical/exposed data only for candidate selection, with
   `trialqa-03de7576dd996510ef449b01c06f7d3c` as the current best candidate
   because it previously showed strong retrospective efficiency and no quality
   harm;
3. verify whether an unseen official TrialQA population exists; if not, create
   a new TrialQA-like held-out population before making performance claims;
4. run a 4-question x 1-repeat paired operational canary on that new population;
5. promote only after operational benefit is visible, then buy scoring and
   expand to 8 questions x 1 repeat, 8 x 3, and 8 x 5; and
6. run any full 96-question-style benchmark only after the small ladder passes.

## What is and is not being replicated

This is an Ultra transfer experiment, not a literal rerun of a published
Ultra/TrialQA number. The reference material establishes the experimental
shape, but its published results have different model/dataset pairings:

- The TrialQA result uses Nemotron-3-Super/Nano. Across 96 held-out questions x
  5 repeats per arm, Super improves mean quality from 0.610 to 0.738 while
  reducing mean tokens by about 30% and operational calls by about 45%.
- The published Ultra result uses BiomniBench-DA, not TrialQA. Quality improves
  from 0.337 to 0.507, while tokens increase by about 4.5%.
- `skills-distillation/docs/exp1_ultra.md` contains placeholders rather than a
  committed Ultra/TrialQA result.

Accordingly, the defensible claim is: reproduce Sergei's paired
skill-distillation methodology with Switchyard, then measure whether the learned
TrialQA workflow transfers to Ultra. Do not claim to have replicated a
published Ultra/TrialQA score that does not exist.

The page-13 reference table from
`2026-05-21-Skill-Distillation-Demo-V2.pptx.pdf` is encoded in
`benchmark/fixtures/trialqa_reference_targets_v1.json` and visually checked via
Poppler rendering. The relevant TrialQA targets are:

| Condition | Accuracy | Worst-case | Tokens / trial | Op. calls / trial |
| --- | ---: | ---: | ---: | ---: |
| Super Base | 0.610 | 0.292 | 549K | 15.5 |
| Super R1 | 0.738 (+21%) | 0.500 (+71%) | 385K (-30%) | 8.6 (-45%) |
| Super R1b | 0.742 (+22%) | 0.479 (+64%) | 389K (-29%) | 9.0 (-42%) |
| Nano Base | 0.375 | 0.094 | 673K | 14.9 |
| Nano R1 | 0.531 (+42%) | 0.302 (+221%) | 436K (-35%) | 8.5 (-43%) |
| Nano R1b | 0.556 (+48%) | 0.281 (+199%) | 371K (-45%) | 8.4 (-44%) |

Local gates are intentionally weaker at early checkpoints because 4 or 8
questions cannot prove the 480-trial reference result. The first canary is an
operational safety screen. The complete 8-question prospective scope can
support only a directional local claim. A full reproduction claim requires a
96-heldout-question, 5-repeat, 480-trial comparable evaluation or a frozen
successor population agreed before generation.

An exposure audit adds a second limitation: the local TrialQA parquet contains
120 rows, and all 120 now have generation artifacts somewhere under the local
experiment tree. The original 24 train rows and all 96 held-out rows are
therefore exposed. The local parquet can still support debugging, candidate
selection, and retrospective/non-performance analysis, but it cannot support a
new prospective performance claim. A full confirmatory reproduction now
requires either a new unseen TrialQA population or an independently defensible
exposure policy established before looking at outcomes.

The upstream metadata check also closes the tempting "maybe another official
TrialQA split exists" branch: as of the saved Hugging Face metadata snapshot,
`trialqa` has one split (`train`) with 120 examples and one parquet file. The
paper may mention `futurehouse/labbench2`, but that URL resolves to the same
`EdisonScientific/labbench2` dataset record and SHA used locally.

The reference workflow also establishes important execution constraints:

- ToolUniverse runs in compact mode. The agent uses discovery, get-info, and
  execute, then invokes clinical-trial tools through `execute_tool`.
- A direct `ClinicalTrials_*` interface is not reference-aligned. Historical
  v1-v4 captures exposed curated `trialqa_*` wrappers and are therefore useful
  development falsifications, not reference reproductions.
- The reference budget is 1,800 seconds for the agent and 60 seconds per MCP
  call. The local protocol uses the same values.
- Sergei's Super experiment used 16 workers. That is an infrastructure detail,
  not a required treatment variable. The audited local v2 result supports a
  lower, manifest-bound generation cap of 4 for protocol v3.

## Audited v2 evidence

The immutable v11 candidate is
`trialqa-37f232aa7ed468008dc9c46243b3790d`, with skill SHA-256
`3d60edd60574f3bdcd9ae3788a75b4886bccef8437ca5e8c15540139d79ffd7a`.
It was frozen before the held-out pilot. The v2 capture is
`trialqa-full-f7704fd24ccaf4178d2b`.

### Promoted 8-pair pilot

The first unexposed v2 pilot covered held-out ordinals 8-15 at repeat 1. Its
immutable report is
`trialqa-full-f7704fd24ccaf4178d2b/promotion-q8-15-r1.json`.

| Metric | Skill OFF | Skill ON | Result |
| --- | ---: | ---: | ---: |
| Judge score | 0.625 | 0.625 | exact parity; all 8 pairs concordant |
| Total tokens | 11,774,093 | 3,394,608 | -71.2% |
| Operational calls | 135 | 66 | -51.1% |
| Model turns | 151 | 82 | -45.7% |

Treatment was cheaper on 6/8 pairs for both tokens and calls. The paired token
median was -23,344, the token trimmed mean was -1,047,435.625, and the
operational-call trimmed mean was -8.625. There were no timeouts or null-EOF
retries. One baseline outlier contributes heavily to the headline reduction,
but every predeclared robust-efficiency diagnostic passed. This remains a
promoted pilot, not a confirmatory result.

### Killed 16-pair expansion

The repeat-1 prefix was expanded to held-out ordinals 8-23, or 16 pairs / 32
tasks. Its immutable operational report is
`trialqa-full-f7704fd24ccaf4178d2b/operational-q8-23-r1.json`.

The expansion is killed. Five of 16 baseline draws and five of 16 treatment
draws ended with provider-terminal empty finals: 10/32 assigned tasks overall,
balanced at a 31.25% terminal rate in each arm. Quality was therefore
unavailable under the v2 completion rules, and v2's per-task evidence-tool and
treatment skill-load requirements failed.

The failure does not erase the efficiency signal. The 16-pair prefix still
showed robust operational benefit:

| Diagnostic | v2 16-pair result |
| --- | ---: |
| Total-token reduction | 68.9% |
| Operational-call reduction | 44.7% |
| Paired token median | -14,084.5 |
| Paired token trimmed mean | -60,808.5 |
| Paired operational-call median | -1 |
| Paired operational-call trimmed mean | -1.642857 |
| Treatment-cheaper token pairs | 10/16 |
| Treatment-cheaper call pairs | 10/16 |
| Null-EOF retries / timeouts | 0 / 0 |

All eight final-efficiency diagnostics in the report passed. This is useful
motivation for a clean prospective directional run, but it cannot promote a
capture whose quality population was not evaluated as assigned.

### Terminal-empty audit

The ten terminal failures were audited before designing protocol v3:

- every affected logical request had exactly one physical attempt, with no
  null-EOF retry and no transport error;
- each stream had provider terminal proof (`finish_reason: stop` or equivalent
  explicit completion), but no final assistant content, reasoning, or tool call
  in that final response and no final-turn usage;
- the harness surfaced six as `Codex final answer is empty` and four as
  `Codex completed without a TrialQA evidence-tool call`;
- the proxy returned HTTP 200, and earlier tool calls completed where present;
- the first balanced eight-task wave failed 8/8, while its complementary wave
  failed 2/8; two questions failed in both arms; and
- failures were exactly balanced at five baseline and five treatment.

This pattern is concurrency-correlated provider/model terminal-empty behavior.
It is not a null-EOF stream, a selective network retry opportunity, or evidence
that the harness discarded a semantic answer. The v2 capture correctly made
the completed draws terminal and did not replace them.

### Artifact disposition

The v2 capture and its reports are immutable historical evidence. Do not
resume, recover, retry, rescore under a new population rule, or pool them with
protocol v3.

All old `primary88` doctors/manifests, including
`full-manifest-compact-v11-contract-v2-primary88.json`, and capture
`trialqa-full-f7704fd24ccaf4178d2b` are stale for current execution. They bind
the old source, quarantine boundary 8, 88-question scope, concurrency behavior,
and v2 completion rules. They are not current v3 artifacts.

The broader exposure audit found completed historical draws through held-out
position 87, not merely through position 23. Positions 24-87 are consequently
retrospective/non-performance data even if a particular old manifest labeled
them primary. Before the v3 repeat-1 run, only positions 88-95 remained
eligible for a conservative fresh prospective primary; those positions are now
consumed by the failed v3 run below.

The current zero-spend v3 artifacts are frozen against the validated source:

- doctor `doctor-report-compact-v11-reference-itt-q88.json`, SHA-256
  `6e5feb9e3594f5c774a1141975955b10a5b06ae8db0808a6b30b0cd7451bb636`;
- descriptive manifest `full-manifest-compact-v11-reference-itt-q88-descriptive.json`,
  ID `trialqa-full-add32f24f7f17676eb20`, SHA-256
  `aeac130f3980d898748005ac83451ec31d35f864a6334a04fc044616c81bc734`;
  and
- prospective manifest `full-manifest-compact-v11-reference-itt-q88-primary8.json`,
  ID `trialqa-full-fb5dec4b9605f38afbe2`, SHA-256
  `a2f0d869a7b8e10a16c2df5a4f5f1e2a99d200cc2a3896de99d6d4be648fbcfc`.

The doctor passed with `model_calls: 0`. The primary manifest now has a
protocol-v3 repeat-1 capture at
`trialqa-full-fb5dec4b9605f38afbe2`; any subsequent execution-source change
makes the doctor and manifests stale.

### Current v3 repeat-1 result: killed

The current v11 candidate
`trialqa-37f232aa7ed468008dc9c46243b3790d` must not be expanded beyond
repeat 1. The 4-question canary over positions 88-91 promoted to scoring, but
the full repeat-1 suffix over positions 88-95 failed the frozen operational
gate:

| Scope | Report | Decision | Token delta | Operational-call delta |
| --- | --- | --- | ---: | ---: |
| q88-q91 r1 | `trialqa-full-fb5dec4b9605f38afbe2/operational-q88-91-r1-v3.json` | `promote_to_score` | +102.0% treatment tokens | +68.7% treatment calls |
| q88-q91 r1 | `trialqa-full-fb5dec4b9605f38afbe2/promotion-q88-91-r1-v3.json` | `promote_to_next_cohort` | +102.0% treatment tokens | +68.7% treatment calls |
| q88-q95 r1 | `trialqa-full-fb5dec4b9605f38afbe2/operational-q88-95-r1-v3.json` | `kill` | +91.9% treatment tokens | +62.6% treatment calls |

The q88-q95 operational gate had no timeouts, no null-EOF retries, no terminal
task-rate delta, and complete paired scope. The kill is purely the intended
efficiency gate doing its job: skill ON used 14,226,358 total tokens and 239
operational calls, while skill OFF used 7,412,360 total tokens and 147
operational calls. Four token pairs were cheaper with treatment and four were
cheaper with baseline, but the aggregate and robust paired checks all failed.

Do not score the q92-q95 expansion, run repeats 2-5 for this candidate, or
reinterpret this capture as performance evidence for the skill. The next valid
path is a new candidate selection plus a new prospective population, because
positions 88-95 have now been consumed by this failed candidate.

### Exposure audit after v11 kill

A post-kill local audit scanned the pinned TrialQA parquet, saved Hugging Face
dataset metadata, and benchmark run directories, pruning cache and
virtual-environment directories. Its durable report is
`.experiments/trialqa-local/population-audit-after-v11-kill.json`, SHA-256
`3740fd5052d6cc1d2ef6d983305029eec6734b748087cefab9a4f8c060ef227c`.
The saved upstream metadata is
`.experiments/trialqa-local/upstream/edisonscientific-labbench2-metadata-2026-07-08.json`,
SHA-256
`14304ad6f599683638bc6a54e53c19b81f7aa153e4d9a4637d96b2f14b406571`.

It found:

| Item | Result |
| --- | ---: |
| Local TrialQA rows | 120 |
| Sergei-style train rows | 24 |
| Sergei-style held-out rows | 96 |
| Official TrialQA split/files | one `train` split / one parquet |
| Rows planned in manifests | 120 |
| Rows with generation artifacts | 120 |
| Unexecuted train rows | 0 |
| Unexecuted held-out rows | 0 |
| Official unseen TrialQA split available | false |

The consumed q88-q95 held-out suffix maps to raw dataset row indices 112-119,
not raw row indices 88-95. That coordinate distinction matters because the
Sergei split is hash-based. The automated audit is partition-aware and confirms
there is no longer a fresh local suffix to promote. The next benchmark must
either use new data or be clearly marked exploratory.

Recreate the zero-spend audit with:

```bash
uv run python -m benchmark.trialqa_local_population_audit \
  --dataset .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet \
  --experiment-root .experiments/trialqa-local \
  --huggingface-metadata .experiments/trialqa-local/upstream/edisonscientific-labbench2-metadata-2026-07-08.json \
  --output .experiments/trialqa-local/population-audit-after-v11-kill.json \
  --pretty
```

### Fastest recovery plan

Use this plan before any further long benchmark:

1. Candidate choice, zero model spend:
   - reject v11 as the current candidate because q88-q95 repeat 1 had +91.9%
     treatment tokens and +62.6% treatment operational calls;
   - shortlist the compact candidate
     `trialqa-03de7576dd996510ef449b01c06f7d3c`, skill SHA-256
     `e80f30e6431d1aafe748b3c1028f68b2fa1e20e9967ef156885b936967edd827`;
   - rationale: its 24-pair exposed gate promoted with +0.0417 mean score
     delta, 64.8% token reduction, 52.2% operational-call reduction, treatment
     cheaper on 21/24 token pairs and 18/24 call pairs; and
   - treat those numbers only as candidate-selection evidence.
2. Population choice, zero or near-zero model spend:
   - the official LABBench2 TrialQA source has been checked and has no unseen
     split/config/revision to pin;
   - build or fetch a small new TrialQA-compatible held-out set from
     clinicaltrials.gov questions using the same ToolUniverse-only retrieval
     contract and frozen scorer;
   - current starter population:
     `.experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet`,
     SHA-256
     `7cfb6eef495572e0ec087a15ea9322f965c5b031e6287e677835b49008903161`,
     with report
     `.experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json`,
     SHA-256
     `96420e6f992947643d2974852b78b9a0d9e196da3864bed5d11c15ae312ef82f`;
   - the starter population has 8 rows, 8 unique NCT IDs, no overlap with the
     102 NCT IDs found in the official TrialQA parquet, and must not be reported
     as official LABBench2 TrialQA; and
   - freeze the population, candidate, route, tool contract, prompt, scoring
     policy, and exposure report before any model generation.
3. Spend ladder:
   - start with 4 questions x 1 repeat x 2 arms, generation only;
   - run the operational gate before scorer calls;
   - kill immediately if treatment is not directionally cheaper on total
     tokens, operational calls, and a majority of paired questions;
   - score only if the operational canary passes;
   - expand to 8 x 1, then 8 x 3, then 8 x 5; and
   - only after 8 x 5 passes should we run a large 96-question-style benchmark.

This is the shortest path that still protects the final claim. Anything run on
the existing 120 local rows is useful engineering signal but not performance
evidence.

Generate or regenerate a starter prospective population with:

```bash
uv run python -m benchmark.trialqa_local_prospective_population \
  --exclude-dataset .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet \
  --output .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --limit 8 --page-size 20 --max-pages 2
```

This generator performs no model calls. It fetches structured
ClinicalTrials.gov metadata, excludes all NCT IDs visible in official TrialQA,
writes a same-schema TrialQA-compatible parquet, and records exact provenance.
Because the rows are not official LABBench2 examples, use them as the first
prospective canary for Switchyard wiring and skill-efficiency direction. A
stronger final claim still needs either a larger independently curated
population or an agreed exposure policy before generation.

The prospective execution bridge is now implemented. The live planner, one-off
runner, and batch driver can load a hash-bound TrialQA-compatible population
when the manifest marks `official_labbench2: false`; the same code still keeps
official LABBench2 TrialQA pinned to the EdisonScientific parquet.

Current operator packet pattern, refreshed after the local-runtime audit
hardening:

- decision summary:
  `.experiments/trialqa-local/prospective/decision-summary-ctgov-prospective-v1-compact-vN-q0-q3-r1.json`
- spend review:
  `.experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-vN-q0-q3-r1.json`
- spend guard:
  `.experiments/trialqa-local/prospective/spend-guard-check-ctgov-prospective-v1-compact-vN-q0-q3-r1.json`
- next boundary:
  q0-q3, repeat 1, baseline/treatment = 8 generation model calls and 0 judge
  calls.

Use the newest `compact-vN` packet generated after the last source or runbook
edit; never spend from a packet generated before such an edit. The decision
summary is the copy/paste source of truth for the guarded command, progress
monitor, post-spend gate inspection, and no-spend checkpoint. It exposes
`proved_setup_evidence` showing the run is local Switchyard over the
TrialQA-compatible prospective parquet, not Docker or a second Hugging Face
runtime repository. Older packets are retained only as immutable audit history
after source-hash changes.

To avoid choosing among stale retained packets by hand, run the read-only
current-packet helper immediately before any spend decision. It selects the
highest matching `compact-vN` decision summary, re-runs the no-spend spend
guard, and writes a compact handoff. It does not run model or judge calls:

```bash
uv run python -m benchmark.trialqa_local_current_packet \
  --artifact-dir .experiments/trialqa-local/prospective \
  --stage generation \
  --scope q0-q3-r1 \
  --stem-contains ctgov-prospective-v1 \
  --repo-root . \
  --output .experiments/trialqa-local/prospective/current-packet-ctgov-prospective-v1-q0-q3-r1.json
```

Do not use `--no-verify-guard` for a spend decision. That mode is only a
read-only packet-selection probe; it intentionally returns
`status=selected_without_fresh_spend_guard`,
`fresh_spend_guard_required_before_spend=true`, and no `guarded_spend` command.
Rerun the helper without `--no-verify-guard` before considering any command that
ends in `--yes-spend`.

Prospective canary inputs:

- candidate wrapper:
  `.experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887`
- candidate ID: `trialqa-03de7576dd996510ef449b01c06f7d3c`
- skill SHA-256:
  `e80f30e6431d1aafe748b3c1028f68b2fa1e20e9967ef156885b936967edd827`
- current doctor report pattern:
  `.experiments/trialqa-local/doctor-report-compact-vN-prospective-v1.json`
- current prospective manifest pattern:
  `.experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-vN.json`
- manifest ID: `trialqa-full-16abb801ed9bdc82a010`
- scope: 8 prospective questions x 5 repeats x 2 arms = 80 tasks

The explicit command blocks below are historical/templates. If a filename in a
command block disagrees with the current decision summary, prefer the decision
summary and regenerate a new packet rather than editing `--yes-spend` by hand.

Recreate the zero-spend doctor and prospective manifest with:

```bash
uv run python benchmark/trialqa_local_demo.py doctor \
  --dataset .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet \
  --experiment-root .experiments/trialqa-local \
  --candidate-root .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard-bin .venv/bin/switchyard \
  --codex-bin /opt/homebrew/bin/codex \
  --tooluniverse-bin .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --routing-profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --output .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json

uv run python benchmark/trialqa_local_demo.py plan-prospective \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --dataset-sha256 7cfb6eef495572e0ec087a15ea9322f965c5b031e6287e677835b49008903161 \
  --dataset-row-count 8 \
  --dataset-revision clinicaltrials-gov-prospective-v1 \
  --experiment-root .experiments/trialqa-local \
  --candidate-root .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard-bin .venv/bin/switchyard \
  --codex-bin /opt/homebrew/bin/codex \
  --tooluniverse-bin .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --routing-profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --doctor-report .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --output .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json
```

The first paid canary should be generation-only for 4 questions x repeat 1 x 2
arms. Prefer the guarded driver because it reruns readiness first, refuses to
spend unless `--yes-spend` is present, runs generation, and then runs only the
operational gate. Without `--yes-spend`, the same command is a zero-spend dry
run that prints the exact child commands:

```bash
uv run python -m benchmark.trialqa_local_canary \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --question-start 0 --question-limit 4 --repeat-limit 1 \
  --workers 4 --max-generation-attempts 1 \
  --readiness-output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --gate-output .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --summary-output .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Do not manually tack `--yes-spend` onto that dry-run command. After the
spend-review packet is generated, use the guarded command from the compact
decision summary instead; it includes `--spend-review <packet>`, and the canary
refuses to spend unless that reviewed packet, hash-bound bundle, and selected
ledger/lock progress still match the current files. Kill the candidate if the
resulting operational gate is not directionally cheaper on total tokens,
operational calls, and a majority of paired questions.

As soon as the paid generation canary writes the operational gate, inspect that
gate against the previously generated decision summary before doing anything
larger. This command is read-only; it rejects the wrong gate path, wrong
manifest, wrong gate kind, stale embedded gate-inspection command, or any
unsafe post-spend inspection/checkpoint command containing `--yes-spend`. A
promoted result still does not authorize judge spend; it tells you to run the
no-spend generation checkpoint first.

```bash
uv run python -m benchmark.trialqa_local_gate_inspect \
  --gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --decision-summary .experiments/trialqa-local/prospective/decision-summary-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/gate-inspection-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

After the paid generation canary finishes, prefer the no-spend generation
checkpoint. It refreshes status with the operational gate, asks the next-step
planner what is safe, and then either records a terminal kill or prepares the
score preflight plus score-spend review packet. It does not run judge calls or
authorize spend. The spend-review packet includes structured
`post_spend_acceptance_criteria`: after generation, inspect the operational
gate artifact, continue only on `decision: promote_to_score`, and run this
checkpoint before buying any judge calls:

```bash
uv run python -m benchmark.trialqa_local_generation_checkpoint \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --question-start 0 --question-limit 4 --repeat-limit 1 \
  --workers 4 --max-generation-attempts 1 \
  --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
  --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
  --skills-distillation-repo skills-distillation \
  --artifact-dir .experiments/trialqa-local/prospective \
  --artifact-stem ctgov-prospective-v1-compact-v5 \
  --status-output .experiments/trialqa-local/prospective/status-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --next-step-output .experiments/trialqa-local/prospective/next-step-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --score-summary-output .experiments/trialqa-local/prospective/canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --promotion-gate-output .experiments/trialqa-local/prospective/gate-promotion-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --protocol-audit-output .experiments/trialqa-local/prospective/protocol-audit-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --reference-alignment-output .experiments/trialqa-local/prospective/reference-alignment-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --audit-bundle-output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --audit-bundle-verification-output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --score-preflight-output .experiments/trialqa-local/prospective/no-spend-score-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --score-progress-output .experiments/trialqa-local/prospective/progress-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --spend-review-output .experiments/trialqa-local/prospective/spend-review-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --ladder-rehearsal .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json \
  --goal-audit-output .experiments/trialqa-local/prospective/goal-audit-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --decision-summary-output .experiments/trialqa-local/prospective/decision-summary-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/generation-checkpoint-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Run the guarded score driver only if the checkpoint returns
`status: awaiting_score_spend_authorization` and its score-spend review packet
is acceptable. The score-spend review packet carries the guarded score command,
the read-only score-progress monitor command, current-progress verification for
the promoted scope, and the no-spend post-score checkpoint command to run after
paid scoring finishes. The checkpoint output also carries a no-spend
`pre_spend_guard_check`; run it immediately before approving judge spend to
recheck the reviewed score command, current hash-bound bundle, and selected
ledger/lock progress. The generated generation-checkpoint command includes
`--ladder-rehearsal`,
`--goal-audit-output`, and `--decision-summary-output`, so it also writes a
compact score-boundary operator summary directly. Without `--yes-spend`, the
direct score driver is only a dry run that refuses to proceed unless the
operational gate already promoted to score:

```bash
uv run python -m benchmark.trialqa_local_canary_score \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --question-start 0 --question-limit 4 --repeat-limit 1 \
  --workers 4 --max-generation-attempts 1 \
  --promotion-gate-output .experiments/trialqa-local/prospective/gate-promotion-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --summary-output .experiments/trialqa-local/prospective/canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Add `--yes-spend` only after inspecting the operational report and confirming
`decision: promote_to_score`.

After the paid score canary finishes, prefer the no-spend score checkpoint. It
refreshes status with both the operational and promotion gates, asks the
next-step planner what is safe, and then either records a terminal kill /
directional-complete decision or prepares the next generation-expansion
preflight plus spend-review packet. It does not run generation or authorize
spend. The score-spend packet carries the matching acceptance criteria: inspect
the promotion gate, continue only on `decision: promote_to_next_cohort`, and run
this checkpoint before buying any additional model calls:

```bash
uv run python -m benchmark.trialqa_local_score_checkpoint \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --current-readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --promotion-gate .experiments/trialqa-local/prospective/gate-promotion-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --workers 4 --max-generation-attempts 1 \
  --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
  --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
  --skills-distillation-repo skills-distillation \
  --artifact-dir .experiments/trialqa-local/prospective \
  --artifact-stem ctgov-prospective-v1-compact-v5 \
  --post-score-status-output .experiments/trialqa-local/prospective/status-after-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --next-step-output .experiments/trialqa-local/prospective/next-step-after-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --expansion-readiness-output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --expansion-operational-gate-output .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --generation-summary-output .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --expansion-status-output .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --protocol-audit-output .experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --reference-alignment-output .experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --audit-bundle-output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --audit-bundle-verification-output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --generation-preflight-output .experiments/trialqa-local/prospective/no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --spend-review-output .experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --ladder-rehearsal .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json \
  --goal-audit-output .experiments/trialqa-local/prospective/goal-audit-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --decision-summary-output .experiments/trialqa-local/prospective/decision-summary-ctgov-prospective-v1-compact-v5-q0-q7-r1.json \
  --output .experiments/trialqa-local/prospective/score-checkpoint-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Run the next guarded generation expansion only if the score checkpoint returns
`status: awaiting_generation_expansion_spend_authorization` and its expansion
spend-review packet is acceptable. The checkpoint output also carries a
no-spend `pre_spend_guard_check`; run it immediately before approving the
generation-expansion spend. The generated score-checkpoint command includes
`--ladder-rehearsal`, `--goal-audit-output`, and
`--decision-summary-output`, so it also writes a compact expansion-boundary
operator summary directly. The same pattern repeats for 8 x 1 -> 8 x 3 and
8 x 3 -> 8 x 5.

Lower-level no-spend path after an operational promotion: run the score
preflight directly. The generation checkpoint above calls this automatically
on the promote path. Use this command by hand only when debugging the score
boundary. It validates the operational gate, persists the score dry-run
summary, regenerates status and protocol audit, hash-binds the score summary
into a pre-score-spend bundle, and verifies that bundle. It does not run judge
calls:

```bash
uv run python -m benchmark.trialqa_local_score_preflight \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
  --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
  --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --question-start 0 --question-limit 4 --repeat-limit 1 \
  --workers 4 --max-generation-attempts 1 \
  --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
  --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
  --skills-distillation-repo skills-distillation \
  --readiness-output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --score-summary-output .experiments/trialqa-local/prospective/canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --promotion-gate-output .experiments/trialqa-local/prospective/gate-promotion-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --status-output .experiments/trialqa-local/prospective/status-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --protocol-audit-output .experiments/trialqa-local/prospective/protocol-audit-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --reference-alignment-output .experiments/trialqa-local/prospective/reference-alignment-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --audit-bundle-output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --audit-bundle-verification-output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/no-spend-score-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Manual equivalent: after the score dry-run summary exists, regenerate status
with the operational gate and then rerun the protocol audit with the score
summary. This makes the second spend boundary self-contained in the audit
artifact, just like the generation boundary:

```bash
uv run python -m benchmark.trialqa_local_status \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/status-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json

uv run python -m benchmark.trialqa_local_protocol_audit \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --status .experiments/trialqa-local/prospective/status-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --score-canary-summary .experiments/trialqa-local/prospective/canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --operational-gate .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/protocol-audit-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

Before judge spend, hash-bind and verify the score boundary too. The bundle
builder requires the relevant dry-run summary for the active boundary: a
generation summary before generation spend, or a score summary before score
spend.

```bash
uv run python -m benchmark.trialqa_local_audit_bundle \
  --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
  --readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --status .experiments/trialqa-local/prospective/status-after-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --protocol-audit .experiments/trialqa-local/prospective/protocol-audit-score-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
  --score-canary-summary .experiments/trialqa-local/prospective/canary-score-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
  --output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json

uv run python -m benchmark.trialqa_local_audit_bundle_verify \
  --bundle .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
  --output .experiments/trialqa-local/prospective/pre-score-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
```

## Archived protocol v3 for consumed q88-q95 run

### Fresh population and quarantine

- Dataset revision: `27d12d72af24e3f70db8a99df63e567366cbdb80`
- Parquet SHA-256:
  `b571c93dce7497f678e019c17b1d4bc230da7a4d180c3cb9f22343ecc2efcd42`
- Split seed: `trace2skill-trialqa`
- Original held-out population: 96 questions
- Conservative exposure quarantine: held-out positions 0-87
- Fresh prospective primary for this now-consumed run: held-out positions
  88-95, 8 questions
- Repeats: 5
- Arms: baseline skill OFF and treatment skill ON
- Prospective directional task count: 8 x 5 x 2 = 80

Positions 0-7 were consumed during candidate development. Positions 8-23 were
consumed by the promoted pilot and killed expansion, and their outcomes
informed protocol v3. Other historical full captures contain completed draws
across positions 24-87. Under the conservative ledger/output exposure rule, all
88 positions are quarantined. A retrospective report over positions 0-87 or a
descriptive all-question report may be produced separately, but it must be
marked `performance_eligible: false` and never enter the primary estimate.

The exposure audit reconstructed the pinned 96-question order and scanned 29
manifests plus 23 capture ledgers. Its 1,158 normal generation records match
1,158 `generation.json` artifacts one-for-one; another 120 terminal completed
draws have present, hash-matching failure artifacts. Before protocol-v3
spending, positions 88-95 had zero ledger event of any type, pair directory, or
generation output. They have now been consumed by the failed repeat-1 v3 run
described above and are no longer an untouched prospective population.

The 8-question suffix is large enough for a clean paired directional check and
repeat-level stability analysis, but too small to stand in for the published
96-question population. Passing it validates the local Switchyard
skill-distillation direction; it does not complete a full confirmatory
reproduction.

### Intent-to-treat scoring

Every assigned task in a frozen checkpoint remains in its assigned arm:

1. A non-empty assistant final is scored by the frozen semantic judge.
2. An explicitly provider-terminal empty assistant final receives quality score
   0. Its observed token and call usage is retained exactly as recorded.
3. Evidence-tool use and treatment skill loading are compliance diagnostics.
   Missing either does not exclude, replace, or automatically zero an otherwise
   scoreable final answer.
4. A completed draw is never regenerated because it is empty, malformed,
   tool-noncompliant, skill-noncompliant, low quality, or expensive.
5. Never-started tasks may be resumed. Hash-bound score work may be resumed.
   Neither operation changes the assigned model draw.
6. Run with `--max-generation-attempts 1`; do not use `--retry-failed` or
   whole-task recovery to obtain a replacement outcome.

The transport may still record its existing bounded physical recovery only for
a proven pre-semantic null EOF before a model draw completes. That is not a
second selectable benchmark draw, but it is fully accounted and makes the
capture non-promotable under the frozen zero-null-EOF performance rule. Do not
replace the task or hide the retry.

This intent-to-treat rule fixes the v2 estimand problem: terminal empties can
hurt quality, but cannot disappear from the quality population. The treatment
still has to win on the complete assigned population.

### Execution and isolation

- Executor: `nvidia/nvidia/nemotron-3-ultra` through `sd-executor`
- Distiller and semantic judge: `aws/anthropic/bedrock-claude-opus-4-8`
- Tool surface: compact discovery/get-info/execute, with clinical-trial calls
  made through `execute_tool`
- Agent/MCP deadlines: 1,800/60 seconds
- Maximum generation concurrency: 4
- Control: concurrent paired waves with deterministic arm-order crossover

The OFF and ON arms receive the same prompt and read-only tool schemas. The ON
arm differs only because Codex discovers the immutable
`tooluniverse-trialqa` skill in its isolated project skill directory. Pair-safe
waves never run both arms in the same workspace. Baseline and treatment
alternate which arm is in the first provider-time wave.

The manifest must bind `protocol.max_generation_concurrency: 4`. The batch
driver must reject generation or all-stage execution above 4. Score-only work
may use more CPU concurrency, but using 4 everywhere is simpler and keeps the
runbook unambiguous.

### Frozen promotion policy

The `ultra-efficiency-v3` policy retains the v2 numerical thresholds while
changing the completion/quality estimand to intent-to-treat. Promotion requires:

- at least 15% total-token reduction;
- at least 20% TrialQA operational-call reduction, excluding skill loading;
- negative paired median and 10%-trimmed-mean token deltas;
- negative 10%-trimmed-mean operational-call delta;
- more treatment-cheaper than baseline-cheaper pairs for both tokens and calls;
- zero null-EOF transport retries in a performance-eligible capture;
- no more than a 5-percentage-point treatment-minus-baseline terminal-rate
  increase;
- at interim checkpoints, the frozen one-sided paired-quality harm screen must
  not rule out the -5-point margin; and
- at the complete 8 x 5 directional scope, report the 95% one-sided
  question-clustered paired-quality lower bound and apply the frozen -0.05
  local success threshold.

All quality calculations include terminal-empty zeros. Tool/skill compliance is
reported beside quality and efficiency, not used as a hidden eligibility
filter. Passing these gates on eight question clusters is directional
validation, not a 96-question confirmatory reproduction.

## Zero-spend preflight

Do not make a model call until every item here passes:

1. Run the focused local suite:

   ```bash
   uv run pytest \
     tests/test_trialqa_local_dataset.py \
     tests/test_trialqa_local_prospective_population.py \
     tests/test_trialqa_local_demo.py \
     tests/test_trialqa_local_batch.py \
     tests/test_trialqa_local_gate.py \
     tests/test_trialqa_local_generation_checkpoint.py \
     tests/test_trialqa_local_goal_audit.py \
     tests/test_trialqa_local_ladder_rehearsal.py \
     tests/test_trialqa_local_readiness.py \
     tests/test_trialqa_local_canary.py \
     tests/test_trialqa_local_canary_score.py \
     tests/test_trialqa_local_status.py \
     tests/test_trialqa_local_protocol_audit.py \
     tests/test_trialqa_local_reference_alignment.py \
     tests/test_trialqa_local_audit_bundle.py \
     tests/test_trialqa_local_audit_bundle_verify.py \
     tests/test_trialqa_local_preflight.py \
     tests/test_trialqa_local_score_checkpoint.py \
     tests/test_trialqa_local_score_preflight.py \
     tests/test_trialqa_local_next_step.py \
     tests/test_trialqa_local_spend_review.py \
     tests/test_trialqa_reference_targets.py -q
   ```

2. Run lint and type checks over the TrialQA local harness:

   ```bash
   uv run ruff check \
     benchmark/trialqa_local_dataset.py \
     benchmark/trialqa_local_population_audit.py \
     benchmark/trialqa_local_prospective_population.py \
     benchmark/trialqa_local_demo.py \
     benchmark/trialqa_local_batch.py \
     benchmark/trialqa_local_gate.py \
     benchmark/trialqa_local_generation_checkpoint.py \
     benchmark/trialqa_local_goal_audit.py \
     benchmark/trialqa_local_ladder_rehearsal.py \
     benchmark/trialqa_local_readiness.py \
     benchmark/trialqa_local_canary.py \
     benchmark/trialqa_local_canary_score.py \
     benchmark/trialqa_local_status.py \
     benchmark/trialqa_local_protocol_audit.py \
     benchmark/trialqa_local_reference_alignment.py \
     benchmark/trialqa_local_audit_bundle.py \
     benchmark/trialqa_local_audit_bundle_verify.py \
     benchmark/trialqa_local_preflight.py \
     benchmark/trialqa_local_score_checkpoint.py \
     benchmark/trialqa_local_score_preflight.py \
     benchmark/trialqa_local_next_step.py \
     benchmark/trialqa_local_spend_review.py \
     tests/test_trialqa_local_dataset.py \
     tests/test_trialqa_local_population_audit.py \
     tests/test_trialqa_local_prospective_population.py \
     tests/test_trialqa_local_demo.py \
     tests/test_trialqa_local_batch.py \
     tests/test_trialqa_local_gate.py \
     tests/test_trialqa_local_generation_checkpoint.py \
     tests/test_trialqa_local_goal_audit.py \
     tests/test_trialqa_local_ladder_rehearsal.py \
     tests/test_trialqa_local_readiness.py \
     tests/test_trialqa_local_canary.py \
     tests/test_trialqa_local_canary_score.py \
     tests/test_trialqa_local_status.py \
     tests/test_trialqa_local_protocol_audit.py \
     tests/test_trialqa_local_reference_alignment.py \
     tests/test_trialqa_local_audit_bundle.py \
     tests/test_trialqa_local_audit_bundle_verify.py \
     tests/test_trialqa_local_preflight.py \
     tests/test_trialqa_local_score_checkpoint.py \
     tests/test_trialqa_local_score_preflight.py \
     tests/test_trialqa_local_next_step.py \
     tests/test_trialqa_local_spend_review.py \
     tests/test_trialqa_reference_targets.py

   uv run mypy --explicit-package-bases \
     benchmark/trialqa_local_dataset.py \
     benchmark/trialqa_local_population_audit.py \
     benchmark/trialqa_local_prospective_population.py \
     benchmark/trialqa_local_demo.py \
     benchmark/trialqa_local_batch.py \
     benchmark/trialqa_local_gate.py \
     benchmark/trialqa_local_generation_checkpoint.py \
     benchmark/trialqa_local_goal_audit.py \
     benchmark/trialqa_local_ladder_rehearsal.py \
     benchmark/trialqa_local_readiness.py \
     benchmark/trialqa_local_canary.py \
     benchmark/trialqa_local_canary_score.py \
     benchmark/trialqa_local_status.py \
     benchmark/trialqa_local_protocol_audit.py \
     benchmark/trialqa_local_reference_alignment.py \
     benchmark/trialqa_local_audit_bundle.py \
     benchmark/trialqa_local_audit_bundle_verify.py \
     benchmark/trialqa_local_preflight.py \
     benchmark/trialqa_local_score_checkpoint.py \
     benchmark/trialqa_local_score_preflight.py \
     benchmark/trialqa_local_next_step.py \
     benchmark/trialqa_local_spend_review.py \
     tests/test_trialqa_local_dataset.py \
     tests/test_trialqa_local_population_audit.py \
     tests/test_trialqa_local_prospective_population.py \
     tests/test_trialqa_local_demo.py \
     tests/test_trialqa_local_batch.py \
     tests/test_trialqa_local_gate.py \
     tests/test_trialqa_local_generation_checkpoint.py \
     tests/test_trialqa_local_goal_audit.py \
     tests/test_trialqa_local_ladder_rehearsal.py \
     tests/test_trialqa_local_readiness.py \
     tests/test_trialqa_local_canary.py \
     tests/test_trialqa_local_canary_score.py \
     tests/test_trialqa_local_status.py \
     tests/test_trialqa_local_protocol_audit.py \
     tests/test_trialqa_local_reference_alignment.py \
     tests/test_trialqa_local_audit_bundle.py \
     tests/test_trialqa_local_audit_bundle_verify.py \
     tests/test_trialqa_local_preflight.py \
     tests/test_trialqa_local_score_checkpoint.py \
     tests/test_trialqa_local_score_preflight.py \
     tests/test_trialqa_local_next_step.py \
     tests/test_trialqa_local_spend_review.py \
     tests/test_trialqa_reference_targets.py
   ```

3. Re-run the doctor after the last source change. Any later source, prompt,
   route, tool schema, candidate, or policy change invalidates the doctor and
   manifest before model spending.
4. Regenerate the prospective manifest and inspect the saved file, not just
   console output. It must bind `official_labbench2: false`, population report
   SHA-256, row count 8, train count 0, test count 8, primary start 0, primary
   count 8, task count 80, and `max_generation_concurrency: 4`.
5. Run a no-spend batch-helper sanity check: load the manifest, rebuild it from
   current inputs, and create one worker-private dataset copy. This catches
   stale path attestations such as resolving `/opt/homebrew/bin/codex` to its
   symlink target before execution.
6. Rehearse the full promotion ladder with synthetic readiness/gate artifacts.
   This catches state-machine drift for generation, score, expansion, kill, and
   directional-complete transitions without model or judge calls:

   ```bash
   uv run python -m benchmark.trialqa_local_ladder_rehearsal \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
     --experiment-root .experiments/trialqa-local \
     --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
     --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
     --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
     --switchyard .venv/bin/switchyard \
     --codex /opt/homebrew/bin/codex \
     --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
     --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
     --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
     --artifact-dir .experiments/trialqa-local/prospective \
     --artifact-stem ctgov-prospective-v1-compact-v5 \
     --rehearsal-dir .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5 \
     --workers 4 --max-generation-attempts 1 \
     --output .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json
   ```

7. Preferred path: regenerate the whole no-spend spend-boundary bundle with the
   single ordered preflight command. It runs readiness, the guarded generation
   dry-run, staged status, protocol audit, reference-alignment audit, audit
   bundle, and bundle verification in dependency order. It does not execute
   generation, scoring, or any model call; the emitted next command still
   requires `--yes-spend`. The reference-alignment artifact must say
   `canary_alignment_status: proved` and `claim_scope:
   prospective_transfer_canary`; `official_96_question_reproduction_bound`
   should remain missing for this local proxy.

   ```bash
   uv run python -m benchmark.trialqa_local_preflight \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
     --experiment-root .experiments/trialqa-local \
     --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
     --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
     --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
     --switchyard .venv/bin/switchyard \
     --codex /opt/homebrew/bin/codex \
     --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
     --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
     --question-start 0 --question-limit 4 --repeat-limit 1 \
     --workers 4 --max-generation-attempts 1 \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
     --skills-distillation-repo skills-distillation \
     --readiness-output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --gate-output .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --generation-summary-output .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --status-output .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --protocol-audit-output .experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --reference-alignment-output .experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --audit-bundle-output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --audit-bundle-verification-output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --ladder-rehearsal .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json \
     --output .experiments/trialqa-local/prospective/no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

8. If in doubt after any gate, ask the next-step planner for the safe next
   command. It reads the staged status artifact and emits one no-spend command:
   generation preflight, score preflight, expansion preflight, or no command
   when the candidate is killed or the prospective directional scope is done.
   Expansion preflights are cumulative: the selected scope may include an
   already-completed promoted prefix plus not-yet-started new tasks. The
   readiness check rejects partial/nonterminal selected states, so expansion
   cannot silently proceed over interrupted, failed, or half-scored work.

   ```bash
   uv run python -m benchmark.trialqa_local_next_step \
     --status .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
     --experiment-root .experiments/trialqa-local \
     --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
     --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
     --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
     --switchyard .venv/bin/switchyard \
     --codex /opt/homebrew/bin/codex \
     --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
     --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
     --skills-distillation-repo skills-distillation \
     --artifact-dir .experiments/trialqa-local/prospective \
     --artifact-stem ctgov-prospective-v1-compact-v5 \
     --ladder-rehearsal .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json \
     --workers 4 --max-generation-attempts 1 \
     --output .experiments/trialqa-local/prospective/next-step-ctgov-prospective-v1-compact-v5.json
   ```

   While a guarded generation or score canary is running, use the read-only
   progress inspector instead of waiting blindly for the final gate. It reads
   only the frozen manifest plus the append-only ledger, reports
   not-started/in-progress/done/failed counts for the exact scope, and gives a
   non-authorizing recommendation. It does not call the model, judge, Switchyard,
   Codex, or ToolUniverse. The inspector also probes the local `batch.lock`
   without writing to it. If selected tasks are `generation_started` or in a
   partial score state and the lock is still held, keep waiting/monitoring. If
   selected tasks are partial but no batch driver holds the lock, treat the run
   as interrupted instead of slow: rerun the same guarded canary wrapper with
   `--recover-interrupted` before `--yes-spend` after reviewing the spend packet.
   For score recovery the wrapper passes both `--recover-interrupted` and
   `--retry-failed` to the underlying batch command.

   ```bash
   uv run python -m benchmark.trialqa_local_progress \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --experiment-root .experiments/trialqa-local \
     --stage generation \
     --question-start 0 --question-limit 4 --repeat-limit 1 \
     --output .experiments/trialqa-local/prospective/progress-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

The remaining commands are the manual equivalent of the preflight wrapper and
are retained for auditing or debugging individual artifacts:

9. Regenerate the first-checkpoint readiness report with the reproducible
   command:

   ```bash
   uv run python -m benchmark.trialqa_local_readiness \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
     --experiment-root .experiments/trialqa-local \
     --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
     --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
     --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
     --switchyard .venv/bin/switchyard \
     --codex /opt/homebrew/bin/codex \
     --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
     --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
     --question-start 0 --question-limit 4 --repeat-limit 1 \
     --output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```
10. Regenerate the staged protocol status report:

   ```bash
   uv run python -m benchmark.trialqa_local_status \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --output .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

11. Regenerate the persisted dry-run summary and the protocol audit:

   ```bash
   uv run python -m benchmark.trialqa_local_canary \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --dataset .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1.parquet \
     --experiment-root .experiments/trialqa-local \
     --doctor .experiments/trialqa-local/doctor-report-compact-v5-prospective-v1.json \
     --population-report .experiments/trialqa-local/prospective/trialqa-ctgov-prospective-v1-report.json \
     --candidate .experiments/trialqa-local/prospective/candidates/trialqa-compact-6b62dbeba91b5279c7022579ab8fc887 \
     --switchyard .venv/bin/switchyard \
     --codex /opt/homebrew/bin/codex \
     --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
     --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
     --question-start 0 --question-limit 4 --repeat-limit 1 \
     --workers 4 --max-generation-attempts 1 \
     --readiness-output .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --gate-output .experiments/trialqa-local/prospective/gate-operational-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --summary-output .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json

   uv run python -m benchmark.trialqa_local_protocol_audit \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --status .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --generation-canary-summary .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json

   uv run python -m benchmark.trialqa_local_reference_alignment \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --output .experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

12. Regenerate the pre-spend audit bundle:

   ```bash
   uv run python -m benchmark.trialqa_local_audit_bundle \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --readiness .experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --status .experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --protocol-audit .experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --reference-alignment .experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --generation-canary-summary .experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --runbook benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md \
     --output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

13. Verify the saved pre-spend audit bundle immediately before any live canary:

   ```bash
   uv run python -m benchmark.trialqa_local_audit_bundle_verify \
     --bundle .experiments/trialqa-local/prospective/pre-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/pre-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

14. Generate a compact spend-review packet for the human authorization step.
   It verifies the passed preflight, bundle verification, source-file checks,
   next-step plan, and current read-only progress report. It exposes the
   guarded `--yes-spend` command, a read-only progress monitor command for the
   exact same scope, the guarded interrupted-run recovery command, the frozen
   promotion/kill policy, and the no-spend post-generation checkpoint command
   to run after paid generation finishes, but marks both audit and packet
   authorization as false:

   ```bash
   uv run python -m benchmark.trialqa_local_spend_review \
     --preflight .experiments/trialqa-local/prospective/no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --bundle-verification .experiments/trialqa-local/prospective/pre-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --next-step .experiments/trialqa-local/prospective/next-step-ctgov-prospective-v1-compact-v5.json \
     --progress .experiments/trialqa-local/prospective/progress-generation-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

15. Generate the goal-level audit. This keeps the completion claim honest: it
   should say the no-spend workflow is ready for the next guarded boundary, but
   `goal_complete: false` until both the operational generation gate and the
   scored promotion gate cover the declared final primary scope. An early
   promoted q0-q3-r1 or q0-q7-r1 gate is only evidence to continue the ladder,
   not completion evidence.

   ```bash
   uv run python -m benchmark.trialqa_local_goal_audit \
     --manifest .experiments/trialqa-local/prospective/full-manifest-ctgov-prospective-v1-compact-v5.json \
     --reference-targets benchmark/fixtures/trialqa_reference_targets_v1.json \
     --reference-alignment .experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --ladder-rehearsal .experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json \
     --preflight .experiments/trialqa-local/prospective/no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --protocol-audit .experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --spend-review .experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/goal-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

16. Generate the compact operator decision summary. This is the quickest file
   to read immediately before approving or declining the guarded canary. It
   restates the exact spend scope, the guarded command, the no-spend pre-spend
   guard check, the read-only monitor, the post-spend gate-inspection command,
   the post-spend checkpoint, and the post-spend promote/kill checklist. It
   also refuses stale progress, guarded commands missing `--spend-review`, or
   any supposedly safe command containing `--yes-spend`.

   ```bash
   uv run python -m benchmark.trialqa_local_decision_summary \
     --spend-review .experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --goal-audit .experiments/trialqa-local/prospective/goal-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/decision-summary-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

17. Immediately before approving spend, run the no-spend guard check from the
   decision summary. This re-verifies the reviewed guarded command, the
   hash-bound audit bundle, and selected ledger/lock progress against the
   current filesystem, but it does not run model or judge calls and does not
   authorize spend.

   ```bash
   uv run python -m benchmark.trialqa_local_spend_guard \
     --spend-review .experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q3-r1.json \
     --output .experiments/trialqa-local/prospective/spend-guard-check-ctgov-prospective-v1-compact-v5-q0-q3-r1.json
   ```

Current zero-spend status: the focused TrialQA suite passes 533 tests; Ruff
passes; mypy passes across 69 source files; the doctor report has
`model_calls: 0`; and the active prospective manifest
`trialqa-full-a21b050f6f6a33f9b2b6` contains 80 tasks. The first-checkpoint
readiness report is
`.experiments/trialqa-local/prospective/readiness-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`
and records status `ready_for_generation`, 4 pairs / 8 tasks, no ledger yet,
and all selected tasks `not_started`. The staged status report is
`.experiments/trialqa-local/prospective/status-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`
and records next action `run_guarded_generation_canary` and
`requires_yes_spend: true`. The persisted generation dry-run summary is
`.experiments/trialqa-local/prospective/canary-generation-dryrun-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`
and records status `awaiting_spend_authorization` and the exact generation and
operational-gate commands, plus the guarded top-level
`authorized_rerun_command` ending in `--yes-spend`. The protocol audit is
`.experiments/trialqa-local/prospective/protocol-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`
and records completion state `awaiting_generation_canary_spend_authorization`,
proves all no-spend prerequisites, verifies that the guarded rerun command
points back to the same manifest and dry-run summary, embeds the same guarded
`next_command` in both argv-list and shell-quoted forms, and marks live
generation, quality parity, and efficiency benefit evidence as still missing.
The same protocol-audit code now has tested score-stage support: once an
operational gate promotes to score, it requires a score dry-run summary,
verifies the score command points back to the same manifest and operational
gate, and embeds the guarded score `next_command` before any judge spend.
The pre-spend audit bundle and verifier also support this score boundary and
require the score dry-run summary to be hash-bound before judge spend.
The score-boundary preflight wraps those steps into one no-spend command after
an operational gate promotes to score.
The post-generation checkpoint wraps the first decision after paid generation:
it either records a terminal kill or prepares the score preflight plus
score-spend review packet, still without authorizing judge spend. It can also
write the score-boundary goal audit and decision summary when those optional
output paths are supplied.
The post-score checkpoint wraps the next decision after paid scoring: it either
records a terminal kill/directional-complete decision or prepares the next
generation-expansion preflight plus spend-review packet, still without
authorizing model spend. It can also write the expansion-boundary goal audit
and decision summary when those optional output paths are supplied.
The ladder rehearsal report is
`.experiments/trialqa-local/prospective/ladder-rehearsal-ctgov-prospective-v1-compact-v5.json`;
it uses synthetic gates against the real manifest, records `model_calls: 0`,
`judge_calls: 0`, and passes all 8 expected transition scenarios:
initial generation, post-generation promote, post-generation kill,
post-score q4 promote, post-score q8 r1 promote, post-score q8 r3 promote,
post-score q8 r5 complete, and post-score kill.
Its `ladder_budget` shows the first spend boundary is only 8 generation calls
and 0 judge calls. If every gate promotes through the prospective directional
scope, the upper bound before stopping is 80 generation calls plus 80 judge
calls, each bought behind a separate pre-spend packet.
The next-step planner emits the safe no-spend command for the current status
and has fixture coverage for generation, score, expansion, gate-context
preservation, and kill states.
The readiness/canary/status/protocol-audit path now also has coverage for
cumulative generation expansion: completed prefix tasks plus not-started suffix
tasks are allowed, while nonterminal selected states remain non-spendable.
Status tests cover the full intended promotion ladder:
4-question canary -> 8-question repeat-1 -> repeat-3 -> repeat-5 ->
prospective directional scope complete.
The reference-alignment artifact is
`.experiments/trialqa-local/prospective/reference-alignment-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it proves the current Switchyard/Nemotron Ultra ON/OFF canary matches the
reference comparison shape while marking the official 96-question LABBench2
reproduction as missing.
The pre-spend audit bundle is
`.experiments/trialqa-local/prospective/pre-spend-audit-bundle-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it hash-binds the manifest, readiness report, status report, protocol audit,
reference target fixture, source PDF named by the reference fixture,
reference-alignment report, generation dry-run summary, this runbook, and the
TrialQA guardrail/runtime source files, then copies the guarded next command
into one read-only snapshot. Because the bundle includes the runbook hash, do
not hard-code the bundle hash in this runbook; compute it after regenerating
the bundle if a release note needs it.
The bundle verification report is
`.experiments/trialqa-local/prospective/pre-spend-audit-bundle-verification-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it rehashes every bundle-bound artifact and source file, records
`status: passed`, and exposes the same guarded generation `next_command`.
The spend-review packet is
`.experiments/trialqa-local/prospective/spend-review-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it summarizes the preflight, bundle verification, next-step plan, safe
no-spend command, read-only progress monitor, post-spend no-spend checkpoint,
guarded spend command, and guarded spend scope without authorizing spend. For
the current first generation boundary, that scope is q0-q3, one repeat, two
arms, 8 generation tasks, and zero judge calls.
The goal audit is
`.experiments/trialqa-local/prospective/goal-audit-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it proves the manifest, reference targets, ladder rehearsal, preflight, and
spend-review requirements, plus the reference/proxy scope boundary, but records
`goal_complete: false` because final-primary-scope live generation and scored
promotion evidence are still missing.
The decision summary is
`.experiments/trialqa-local/prospective/decision-summary-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it is the compact handoff for the current boundary: run only the guarded
generation command if explicit spend is approved, monitor with the no-spend
progress command, inspect the operational gate with the embedded
`post_spend_gate_inspection` command, continue only on `promote_to_score`, and
run the generation checkpoint before judge spend.
The one-command preflight report is
`.experiments/trialqa-local/prospective/no-spend-preflight-ctgov-prospective-v1-compact-v5-q0-q3-r1.json`;
it is only a convenience summary over the hash-bound artifacts above and does
not authorize spend.

## Archived fast promotion ladder for consumed q88-q95 run

The commands in this section describe the already-consumed q88-q95 protocol-v3
run. They are retained for auditability and must not be used as the current
execution plan. All question counts below were cumulative prefixes of the
former primary suffix starting at held-out position 88. That suffix is no
longer fresh.

| Checkpoint | Cumulative assigned tasks | New task generations |
| --- | ---: | ---: |
| 4 questions x repeat 1 | 8 | 8 |
| 8 questions x repeat 1 | 16 | 8 |
| 8 questions x repeats 1-2 | 32 | 16 |
| 8 questions x repeats 1-3 | 48 | 16 |
| 8 questions x repeats 1-5 | 80 | 32 |

For each boundary:

1. Generate only the newly admitted paired tasks with at most 4 workers and one
   generation attempt per task.
2. Verify ledger integrity, paired completeness, model routing, source
   attestation, concurrency, terminal accounting, and absence of selectable
   retries. Terminal-empty assigned draws count as complete ITT observations.
3. Run the operational gate before buying judge calls. At small repeat-1
   checkpoints, require visible aggregate token and operational-call benefit,
   then apply the predeclared optimistic futility/reliability screen. Noisy
   failure to hit the final 15%/20% point targets is not by itself a kill.
4. If operationally eligible, score the same immutable generations. Assign
   terminal empties 0 locally and judge non-empty finals. Then apply the
   interim quality-harm screen.
5. Stop on a predeclared kill. Do not revise the candidate from primary data and
   continue. If primary observations inform a repair, quarantine every consumed
   question and define a new prospective population.

At 8 x 1, enforce every final efficiency and robustness criterion before
buying repeats 2-5. Reapply operational and intent-to-treat quality gates at
repeat boundaries 2, 3, and 5. The complete 8 x 5 scope supports only the
predeclared small-sample directional claim. It cannot support a full
confirmatory TrialQA reproduction.

The first generation checkpoint has this shape:

```bash
.venv/bin/python benchmark/trialqa_local_batch.py \
  --manifest .experiments/trialqa-local/full-manifest-compact-v11-reference-itt-q88-primary8.json \
  --dataset .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet \
  --experiment-root .experiments/trialqa-local \
  --doctor .experiments/trialqa-local/doctor-report-compact-v11-reference-itt-q88.json \
  --candidate .experiments/trialqa-local/trialqa-donor-5ee59836e25a2073803f/.switchyard/skill-distillation/tooluniverse-trialqa/candidates/trialqa-37f232aa7ed468008dc9c46243b3790d \
  --switchyard .venv/bin/switchyard \
  --codex /opt/homebrew/bin/codex \
  --tooluniverse .experiments/trialqa-local/tooluniverse-venv/bin/tooluniverse-smcp-stdio \
  --profile benchmark/routing-profiles/skill-distillation-nemotron-ultra.yaml \
  --stage generation \
  --question-start 88 --question-limit 4 --repeat-limit 1 \
  --workers 4 --max-generation-attempts 1
```

Then run the manifest-bound operational gate over exactly the same scope. Only
after it says to score, rerun the batch command with `--stage score`, followed
by the promotion gate. Expand `--question-limit` once from 4 to 8, then change
only `--repeat-limit` through 2, 3, and 5. Never use an ad-hoc batch script under
`.experiments`.

## Local setup

Install Switchyard and a separate pinned ToolUniverse environment in this
worktree:

```bash
uv sync
python3 -m venv .experiments/trialqa-local/tooluniverse-venv
.experiments/trialqa-local/tooluniverse-venv/bin/pip install tooluniverse==1.1.11
```

Place the sole benchmark parquet at:

```text
.experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet
```

One way to fetch the pinned file is:

```bash
mkdir -p .experiments/trialqa-local/source/trialqa
curl -fL \
  https://huggingface.co/datasets/EdisonScientific/labbench2/resolve/27d12d72af24e3f70db8a99df63e567366cbdb80/trialqa/train-00000-of-00001.parquet \
  -o .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet
shasum -a 256 .experiments/trialqa-local/source/trialqa/train-00000-of-00001.parquet
```

Clone and pin the reference repository outside model-visible task workspaces:

```bash
git clone https://gitlab-master.nvidia.com/skolchenko/skills-distillation \
  .experiments/skill-distillation-demo/reference
git -C .experiments/skill-distillation-demo/reference \
  checkout --detach 0618068ccef126e2e5623cd44a379217dca449d8
```

Set `NVIDIA_API_KEY` only in the shell that will make paid model calls.

## Donor and distillation stages

The existing immutable v11 skill must not be reused as the current prospective
evaluation candidate: it failed the consumed q88-q95 repeat-1 operational gate.
Historical candidates may be reused for candidate selection, but any final
performance claim needs a new prospective population. Re-running donor and
distillation calls against the same exposed 120 local rows would add cost
without creating performance evidence.

For a from-scratch end-to-end demo, preserve the original design:

- 24 train questions x 5 repeats = 120 unskilled Ultra donor runs;
- one content-addressed native evidence bundle per validated donor run;
- 120 trajectory-analyst calls, 24 repeat-group merges, and one final merge;
- no held-out TrialQA content in distillation; and
- a new immutable candidate before any prospective held-out task starts.

Start with one donor and one-evidence distillation pilot before paying for the
full train matrix. These are pipeline checks, never performance evidence. The
full distiller remains fail-closed unless it receives the exact 24 x 5 donor
matrix.

## Final reporting contract

The final report must make the estimand and evidence boundary obvious:

- identify this as an Ultra/TrialQA transfer result, not a published
  Ultra/TrialQA replication;
- label the exact prospective population size and show every paired
  intent-to-treat assignment;
- disclose that historical generation artifacts now expose all 120 local
  TrialQA rows, so the existing local parquet is retrospective/non-performance
  only;
- report empty-final counts and rates by arm, with each empty scored 0;
- report evidence-tool and skill-load compliance by arm as diagnostics;
- report mean, per-question macro mean, worst-case, oracle, paired quality
  delta, and the question-clustered one-sided bound;
- report total tokens, operational calls, model turns, paired medians, trimmed
  means, cheaper-pair counts, timeouts, physical attempts, and null-EOF retries;
- include the frozen candidate, skill, source, doctor, manifest, capture, and
  report hashes; and
- keep the v2 8-pair promotion and 16-pair kill in a clearly historical appendix.

Anything short of the complete 80-task primary is pilot or interim evidence.
Even a complete passing primary supports only the bounded small-sample
skill-ON-versus-OFF directional claim. A full confirmatory reproduction needs a
new unseen TrialQA population or a separately justified exposure policy that is
defined before outcomes are inspected.

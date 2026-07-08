# TrialQA Skill-ON Cost Investigation

As of 2026-07-08, the failed official TrialQA suffix did not fail because
Switchyard routed to the wrong model or because the provider retried requests.
It failed because the frozen v11 skill made Nemotron Ultra keep searching.

The failed run was:

```text
.experiments/trialqa-local/trialqa-full-fb5dec4b9605f38afbe2/operational-q88-95-r1-v3.json
```

The candidate was:

```text
trialqa-37f232aa7ed468008dc9c46243b3790d
```

Skill ON used 14,226,358 tokens and 239 operational calls. Skill OFF used
7,412,360 tokens and 147 operational calls. There were no generation timeouts,
no null-EOF retries, and no terminal-task imbalance.

## What Drove the Token Increase

Two paired questions explain almost all of the token increase.

| Pair | Token delta | Tool-call delta | What happened |
| --- | ---: | ---: | --- |
| STAR-T / Ozempic | +3,759,143 | +43 | Treatment repeated ClinicalTrials searches and still answered incorrectly. |
| DiP-in-CML | +3,137,513 | +46 | Both arms searched too much, but treatment searched even more. |

The largest STAR-T treatment session made 68 model turns and about 60
`ClinicalTrials_search_studies` calls for one question. The matching baseline
made 25 model turns and about 16 searches.

The largest DiP-in-CML treatment session made 116 model turns and about 107
`ClinicalTrials_search_studies` calls. Its baseline was already bad at 70 model
turns and about 63 searches, but skill ON made it worse.

The expensive sessions were prompt-token explosions. For example, the STAR-T
treatment session used 4,496,891 prompt tokens and only 22,936 completion
tokens. That means repeated prior context and tool outputs dominated the bill.

## Why the Skill Made This Worse

The v11 skill said to keep looking when a retrieved slice did not contain the
requested field:

```text
If the selected slice lacks the requested field, call another relevant getter
whose documented output can contain it; one empty wrapper does not prove the
datum absent.
```

It also said to stop after at most three semantically distinct searches, but it
did not set a hard total operational-call ceiling. Ultra did not reliably obey
the search bound. In the bad sessions it kept trying near-duplicate search
phrases such as `STAR-T Ozempic`, `STAR-T semaglutide`,
`STAR T semaglutide`, `STAR-T Ozempic semaglutide`, and many more.

The task prompt also reinforced the loop:

```text
Before answering, verify that retrieved evidence explicitly identifies the field
the question asks for. If it does not, inspect operation definitions and
retrieve another relevant read-only evidence slice; never guess a specific
value.
```

That is a good quality rule, but paired with a weak skill it becomes expensive.
For ambiguous questions, Ultra kept searching instead of making a bounded
evidence decision and stopping.

## Difference From Sergei's Workflow

Sergei's TrialQA result works when the skill reduces trial-registry lookup work.
The reference TrialQA table reports:

| Metric | Baseline | Distilled R1 | Change |
| --- | ---: | ---: | ---: |
| Mean tokens / trial | 549,406 | 384,654 | 30.0% fewer |
| Mean agent turns / trial | 19.0 | 12.9 | 32% fewer |
| Operational tool calls / trial | 15.5 | 8.6 | 45% fewer |

The reference metrics also separate local skill/read/write calls from real
operational tool calls. Skill loading can increase local I/O calls, but the
benefit comes from fewer trial-registry searches and getters.

Our failed v11 run did the opposite. It increased real TrialQA searches and
model turns on the hard questions. That is not a successful transfer of the
reference behavior.

The updated reference repo also says test comparisons must be strict paired
comparisons on a frozen grid. If a test result is used to pick or edit the next
candidate, that next comparison is no longer certifying on the same test source.
So this failure can guide the next candidate and workflow, but it cannot be
tuned against the consumed official suffix and then reported as a clean result.

## What We Did Wrong

The main mistake was trusting textual skill guidance as if it were an enforced
runtime budget.

The v11 skill had useful routing advice, but it did not make the agent stop. It
also mixed two incentives:

- "do not guess unless evidence directly supports the answer"; and
- "try another relevant evidence slice when the first one is empty."

For Ultra, that combination encouraged repeated search loops on ambiguous
questions.

We also let invalid or non-canonical tool arguments appear in the transcripts,
including `max_results` and `query` where the compact contract described
`page_size` and `query_term`. That did not cause the whole failure by itself,
but it shows the skill contract was not being enforced by the tool boundary.

## Current Fix Direction

The current compact candidate already addresses the biggest problem. Its skill
adds the missing budget:

```text
Use at most 3 semantically distinct searches ... Never exceed 5 operational
TrialQA calls total.
```

That is the right direction, but it is still a model instruction. Before buying
a larger run, the canary should prove that Ultra obeys it.

Recommended next checks:

1. Keep v11 killed. Do not expand or rescore it.
2. Run only the 8-call generation canary for the compact candidate.
3. Inspect the operational gate before any judge spend.
4. Add or keep a post-generation transcript check that flags:
   - more than 3 ClinicalTrials searches;
   - more than 5 operational TrialQA calls;
   - repeated equivalent search queries;
   - unsupported `execute_tool` argument names; and
   - treatment sessions with more model turns than baseline on most pairs.
5. If the compact candidate violates those rules, kill before scoring.

The clean interpretation is: the failed official suffix found a real candidate
failure, not a Switchyard failure. Switchyard successfully exposed that skill ON
made Ultra more expensive. The next run should test a candidate with explicit
stop rules and should kill immediately if those rules do not show up in the
transcripts.

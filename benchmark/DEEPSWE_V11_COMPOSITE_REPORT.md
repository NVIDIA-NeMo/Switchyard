# DeepSWE v1.1 Composite results

Three full 113-task repeats measured Composite with GPT-6 Astra at medium effort,
GPT-6 Luna at max effort, and Luna as the Decision model. The campaign ran from
October 31 to November 1, 2026, using Switchyard revision
`bdd7647b5d55be46b19764136128646edb08692e`, Codex CLI 0.154.0, and Harbor 0.13.2.
They used closed-book execution and a maximum of five concurrent tasks.
These historical results support the routing recipe. They do not measure the
current PR commit.

## Results

Costs use recorded tokens at fixed base rates. Each cost is for a full 113-task run.

| Configuration | Passes by repeat | Task completion | Mean cost per run |
| --- | --- | ---: | ---: |
| Composite CF0/EF0.4 | 79, 79, 82 | 240/339, 70.80% | $269.99 |
| Astra only | 83, 78, 82 | 243/339, 71.68% | $362.19 |

Composite used 25.45% less recorded-token cost, with task completion 0.88 percentage
points lower. All tasks remain in the denominator, including two verifier timeouts
in the second Composite repeat. Three repeats do not establish statistical
equivalence or generalization to other workloads.

The [JSON results](deepswe-v11-composite-results.json) include all six run IDs,
pass counts, token totals, pricing rates, and source hashes.

## Configuration

The [route configuration](routing-profiles/deepswe-v11-composite-astra-luna.toml)
expresses the measured settings using the current configuration schema. Merge it
with deployment bindings for `astra` at medium effort, `luna` at max effort, and
the `classifier` Decision target using Luna. Serving endpoints are deployment-specific.

- One profile selection per session, with `classify_trigger = "new_session"`.
- Decision cutoff `0.92`, with capable-first selected only above the cutoff.
- Efficient-first threshold `0.4`, capable hold `2`, efficient hold `0`.
- Capable-first threshold `0.0`, capable hold `0`, efficient hold `-1`.
- A window of three recent tool results and the capable-first planning prompt.
- No inner judge or handoff notes. The measured configuration used `fail_open = false`.

The current default [Decision instructions](https://github.com/NVIDIA-NeMo/Switchyard/blob/5bde8855556c242d4a994b5d985a2d66947f7665/crates/libsy/src/algorithms/llm_class/decision.rs)
and [Composite planning prompt](https://github.com/NVIDIA-NeMo/Switchyard/blob/5bde8855556c242d4a994b5d985a2d66947f7665/crates/switchyard-runner/src/algorithm.rs)
match the saved recipe, with choice IDs renamed from `capable_first` and
`efficient_first` to `capable` and `efficient`. The route file inherits those
prompts. Interactive `user_turn` selection was not measured in these runs.

## Cost accounting and coverage

The same rates apply to both configurations, in USD per million tokens:

| Model | Input | Cached input | Cache creation | Output |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 10.00 | 1.00 | 12.50 | 50.00 |
| GPT-6 Luna | 0.10 | 0.01 | 0.125 | 0.50 |

Prompt-token totals include cached input and cache creation. Subtract both before
pricing ordinary input. Recorded selector tokens are included. Infrastructure and
verifier costs are outside this model-token comparison.

Two interrupted Composite answer calls have missing usage. The Astra baseline has
priced usage for 109, 111, and 111 tasks. Its shared logs also contain unmatched
requests that cannot all be attributed to these runs. Per-request long-context
pricing is unverified. The 25.45% reduction compares incomplete recorded-token
estimates, not complete bills or a guaranteed savings bound.

Tuning and evaluation task overlap has not been independently verified. The live
results above should not be presented as held-out validation.

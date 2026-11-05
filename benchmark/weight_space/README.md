# Weight space router experiment

This experiment evaluates an explicit neural policy in Rust. The reference weights are a research candidate. Live task success and cost improvements have not been established.

## Model interface

The runner accepts `type = "weight_space"`, an ordered list of target aliases, inline `weights`, and `cost_weight`. The library exposes `WeightSpaceRouter` and `WeightSpaceClassifier` for in-process use. There is no external scoring service and no judge request.

The two-target reference policy has 18 inputs, 13 ReLU hidden units, and two output scores. It contains 277 numeric coefficients, including two relative cost penalties. Each hidden unit has a name in `circuits.json` so its contribution can be inspected.

The first 16 inputs are `new_user_turn`, `task_bytes`, `change_task`, `analysis_task`, `history_depth`, `recent_turns`, `severity`, `repeated_failure`, `recovery`, `mutation_share`, `observation_share`, `planning_share`, `tests_passed`, `unknown_streak`, `has_mutated`, and `truncated`. Previous-target indicators follow in configured target order. These refer to the last routing choice, which can differ from the model ultimately served after a downstream fallback.

Features use at most 32 recent messages and eight content blocks per message. Copied recent text has a 64 KiB aggregate limit and a 4 KiB limit per string. A cached task excerpt has an 8 KiB aggregate limit. Long text excerpts preserve their beginning and end. New user tasks clear old mutation evidence. The runtime already has the parsed request, so these bounds apply to feature extraction rather than HTTP parsing or the worker model's context.

State retains the last selection, whether the current task has mutated files, and the bounded task excerpt. It is scoped by the runtime session and agent identity. A cold scorer joining a conversation after the task has left the recent window has no cached task excerpt. Tool results and command names provide approximate signals. Truncation can omit relevant evidence.

The forward computation is:

```text
hidden = ReLU(input_weights * features + hidden_bias)
utilities = output_weights * hidden + output_bias - cost_weight * cost_bias
selected = argmax(utilities)
```

Ties select the first target. Scores express preferences, not calibrated success probabilities. `cost_weight` adjusts relative preference. It does not enforce a dollar budget, predict future token use, or guarantee monotonic total trajectory cost. Evidence includes the inputs, activations, utilities, and selected index.

## Running the experiment

Generate the literal reference matrices and a route fragment:

```sh
python benchmark/weight_space/write_reference_weights.py
python benchmark/weight_space/route_config.py benchmark/weight_space/reference_weights.json --targets efficient capable > route-fragment.toml
```

Append that fragment to a config with existing `efficient` and `capable` targets. Output order must match the checkpoint. Model names are configuration aliases, independent of provider model IDs.

Replay request snapshots without worker calls:

```sh
cargo build --release -p switchyard-libsy --example weight_space_replay
target/release/examples/weight_space_replay benchmark/weight_space/reference_weights.json 0.75 < snapshots.jsonl > traces.jsonl
```

Each input line contains `id`, `task`, a protocol `LlmRequest` under `request`, and optional `agent_id`. `task` is a diagnostic state key, not a network input. A snapshot must contain only information available before the decision. The example binds output zero to `efficient` and output one to `capable`. Reported time excludes JSON parsing and serialization.

## How weights are edited

[Weight space programming](https://rikiyat.github.io/blog/weight-space-programming/) uses an LLM to write neural weights directly. Here, the author examines development trajectories and changes literal neuron coefficients in `write_reference_weights.py`. No optimizer or parameter search is used. The first revision changes three escalation circuits to avoid treating isolated soft errors as hard failures.

Historical success identifies candidate behaviors to investigate. It does not establish that a specific switch caused the success. Compare successful stays, failed switches, and matched fixed-model runs. Then test the policy on fresh trajectories. Final-answer gates and advisor workflows can generate information this model selector does not observe, so their complete behavior cannot be reproduced by imitation of model choices alone.

## Evidence required for a superiority claim

Freeze task identities, aliases for duplicate tasks, environment, provider versions, reasoning settings, target set, pricing, policy weights, comparison arms, and cost operating points before opening test results. Group shared problem identifiers across catalog prefixes before splitting. Keep all repeats and variants of the same task in one split. Any development exposure disqualifies every alias from the holdout. Do not select test tasks because a historical router solved them.

Use fixed models, per-request random routing, task-level mixtures, classifier modes including escalation, stage, the auto preset, plan-execute, advisor, composite, VGR, and prefill routing where they support the same model pool and harness. Prefill routing needs a compatible checkpoint and training-data provenance. Historical variants outside the current runner need their original pinned implementation or a demonstrated equivalent. Record exclusions. A comparison against a subset cannot establish superiority over every existing router.

Tune each baseline and the neural cost setting on development data only. Preregister task-level random mixtures of baseline operating points to match expected cost. Include all worker, judge, advisor, retry, and failed-call costs under a common pricing snapshot. Unknown costs are missing data, not zero dollars. Keep agent failures and timeouts in the outcome population. Apply any infrastructure rerun rule symmetrically.

For per-call cache diagnostics, run the server with `--routing-log-file` and enable `RUST_LOG=info,switchyard.routing_usage=debug`. Captured server logs then contain the routing record for each logged call, including model, target, cache reads, cache writes, aggregate input, output, and the recorded cost when available. No prompt or response text is added by these events. The summarizer derives uncached input. Retain the final session stats and reconcile them with the captured calls:

```sh
python benchmark/weight_space/summarize_usage.py switchyard.log routing_stats_final.json SESSION_ID > call-usage.json
```

This diagnostic rejects mismatched call counts and token totals. It distinguishes judge calls from answer-target changes and identifies returns to a previously served target. A switch may reuse a prior cache. Do not assign a full-context cache miss to every switch or optimize switch count as a substitute for total cost. The reference network still uses static cost preferences. Prospective development runs must measure whether each escalation pays for its additional generation and cache costs. Reconciliation cannot recover unlogged billed attempts or cache detail omitted by the provider.

Run `python benchmark/weight_space/price_usage.py capture.json rates.json` to price every captured call under one frozen token-rate snapshot. The JSON rate file maps exact served model IDs to `input`, `output`, `cache_read`, and `cache_write` rates in USD per million tokens. Add `cache_write_1h` when those writes occur. Gateway model aliases need explicit entries. The helper rejects missing rates or token counts and reports provider charges separately. Choose the accounting definition before comparing policies. A reconciled capture does not by itself establish that all billed attempts were recorded.

`evaluate.py` checks complete paired results for at least 200 distinct canonical tasks and two repeats per arm. This is a minimum evidence gate, not a power calculation. For a rough two-arm comparison with 30 percent paired disagreement, detecting a 10 percentage point difference at 80 percent power needs about 235 tasks before correcting for multiple comparisons. A five point difference needs about 940. Use pilot variance to choose the actual sample size and preregister a stopping rule.

The script computes approximate paired task cluster bootstrap bounds. Repeats stay within their task. Bonferroni correction covers both success and cost bounds across the listed baselines. A comparison passes only when the success difference has a positive lower bound and the cost excess has a nonpositive upper bound. The plan can allow at most two percent cost tolerance. Report the tolerance explicitly.

The plan JSON contains `task_ids`, `development_task_ids`, `protocol_sha256`, `pricing_sha256`, `arm_sha256` mapping arm names to fingerprints, `candidate`, `replicates`, `cost_tolerance`, and `baselines`. Each baseline contains `name` and a `mixture` mapping arm names to fixed probabilities summing to one. Archive the plan before running the test. Hashes check identity consistency, they do not prove when the plan was written.

Each results JSONL row contains `task_id`, zero-based `replicate`, `arm`, `arm_sha256`, `protocol_sha256`, `pricing_sha256`, `split` equal to `test`, `terminal` equal to true, binary `reward`, `total_cost_usd`, and `cost_complete` equal to true.

```sh
uv run --script benchmark/weight_space/evaluate.py plan.json results.jsonl --draws 160000 > evidence.json
```

Use at least 4,000 bootstrap draws per declared comparison so the adjusted tail contains at least 100 samples. The example supports up to 40 comparisons.

Bootstrap intervals are approximate sampling evidence for the declared population. They are not a universal guarantee about future tasks. Feature ablations, neuron contributions, and historical choice agreement explain mechanisms but do not substitute for on-policy task results.

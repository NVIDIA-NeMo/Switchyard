# Evaluate task routing

`switchyard.sim` compares routing policies against completed agent runs. It routes
only the input available before a task starts, then scores the selected target
using that target's recorded outcomes. It accepts Harbor runs and ATIF trajectories.
Custom formats can be converted to ATIF.

The library uses `switchyard.runner` in the same process. The native runner loads
Switchyard TOML and handles judge calls, retries, and routing state. No local
server or Harbor installation is needed. From a source checkout containing
`switchyard/sim`, install in a Python 3.10+ virtual environment:

```sh
python -m pip install .
```

Building requires the pinned Rust toolchain and the
[installation requirements](../INSTALLATION.md#requirements).

## Start with paired recordings

Each target represents a recorded model and agent configuration. Compare the same
task versions, agent settings, reasoning budgets, and verifier settings across runs.
Replace the model IDs below with those in your recordings and save as `routes.toml`:

```toml
schema_version = 1

[llm_clients.unused]
format = "openai_chat"
base_url = "http://127.0.0.1:9/v1"

[targets.fast]
id = "fast-model"
llm_client = "unused"

[targets.strong]
id = "strong-model"
llm_client = "unused"

[routes.fixed]
id = "auto"
type = "passthrough"
target = "fast"
```

This route selects `fast` without provider calls. `auto` is the route ID;
`fast` and `strong` are target keys; their `id` fields are provider model IDs.
For live judge calls, use a [task classifier](routing_algorithms/llm_classifier_routing.md).
Its default input excludes system and developer messages; see the
[classifier input settings](routing_algorithms/llm_classifier_routing.md#tuning-options).

```python
import asyncio

from switchyard.runner import Runner
from switchyard.sim import Dataset, evaluate, load_harbor

dataset = Dataset.from_runs(
    {
        "fast": load_harbor("jobs/fast", target="fast"),
        "strong": load_harbor("jobs/strong", target="strong"),
    },
    input_target="fast",
)

async def main():
    report = await evaluate(
        dataset, Runner.load("routes.toml"), route="auto",
        concurrency=8, timeout=60,
        on_result=lambda row: print(row.task_id, row.target, row.error),
    )
    print(report.format_text())
    return report.to_dict()

summary = asyncio.run(main())
```

Use a fresh runner per experiment. Each task receives a distinct, final session
ID. Use `concurrency=1` with seeded random routing for reproducible task ordering.
Known recorded model IDs must match configured IDs. Pass
`model_aliases={"recorded-id": "configured-id"}` only when they identify the same
model. Missing identities remain visible as unverified coverage. Different target
keys may share a provider model ID.

## Recordings and coverage

`load_harbor` accepts a trial, a job, or a downloaded `jobs/` directory containing
`result.json` and `agent/trajectory.json`. ATIF 1.5 and 1.7 text inputs are supported.
For custom formats, use `Trajectory.from_dict(atif).to_trial(...)` with explicit
`task_id`, `trial_id`, `target`, outcome measurements, and whole-run model identity.
Collect trials in a `Run` and pass it to `Dataset.from_runs`.

- Only initial system/user messages before the first agent step become routing input.
  Supply `task_inputs={task_id: original_text}` for missing input or copied continuation
  context. Malformed trajectories and multimodal input are rejected.
- Task checksums must agree; otherwise initial user messages must match. `input_target`
  chooses the input from its first string-sorted trial ID. `dataset="benchmark-v1"`
  namespaces task IDs.
- Repeats are averaged within each task and target; tasks receive equal weight.
  A measurement is unknown if any repeat lacks it. Zero remains an observed value.
- Imports are strict. Use `on_error="record"` and `intersection=True` to evaluate the
  common valid subset, with exclusions in coverage. Preserve custom import failures
  as `LoadIssue` entries in `Run.issues`. Unidentified failures and job trial-count
  mismatches must be repaired before pairing.

## Results and costs

`report.to_dict()` includes coverage, selection counts, recorded reward/cost, and
routing measurements. `complete` requires a scored reward for every expected task
and no evaluation errors; cost coverage is separate. Unknown totals and means are
`null`; `observed_total` retains known measurements.

Reward and cost comparisons each use tasks where that measurement is known for
all targets. The reward oracle chooses the best recorded target mean per task.
Compare identical task IDs across policies and keep tuning and held-out tasks separate.

Recorded cost prefers `result.agent_result.cost_usd`, then ATIF
`final_metrics.total_cost_usd`; `cost_source="trajectory"` reverses the preference.
Coverage records the source. Duration covers agent execution only.

Live judge calls add routing cost. Supply `price_call(call) -> float | None` to
`evaluate`; missing usage or prices remain unknown. Without pricing, only zero-call
routes have known routing cost (zero). Cache tokens are separate from other input;
reasoning detail can overlap output tokens. Retries may incur unreported spend.

`fail_open = true` can select a fallback after a judge failure. Transport failures
remain in call observations; invalid verdicts can instead appear in outcome evidence.
Use `fail_open = false` to surface routing failures as evaluation errors.
Whole-decision timeouts leave call counts and cost unknown.

## Save progressive results

```sh
python -m switchyard.sim \
  --config routes.toml --route auto \
  --run fast=jobs/fast --run strong=jobs/strong \
  --input-target fast --output evaluation-output
```

The Harbor CLI requires a new output directory outside the recordings, including
resolved symlinks. It writes `manifest.json` with the loaded configuration's hash
and coverage, flushes rows to `results.jsonl`, then writes `report.json`.
Configuration source and credentials are not copied. Retain the referenced
recordings, configuration, version, and pricing rules for reproducibility.

Use `--skip-invalid --intersection` for partial input coverage and
`--model-alias RECORDED=CONFIGURED` for explicit model equivalence.
Exit codes: **0** complete, **1** incomplete, **2** invalid input/configuration or
file error, **130** interrupted. Interrupted runs retain completed rows but may
lack a final report. A failed write may leave an incomplete last line. There is
no automatic resume.

Python callers can save rows with a short, synchronous `on_result` callback;
async callbacks are not awaited. Callback failures and cancellation stop and
drain local workers. Already submitted provider calls may still finish or be billed.

## Use the Python decision API

Custom schedulers can use `Runner.decide(request)`, `score(task, decision)`, and
`Report.add`. Requests use normalized IR messages with typed content blocks.
`selected.target` and fallback targets are configuration keys; `selected.model`
and `call.model` are provider model IDs. Calls expose success, duration including
retries, and optional usage. `outcome` is the existing `RoutingOutcome` binding.

`DecisionError` exposes safe `kind`, `upstream_status`, `target`, `calls`, and
`duration_seconds` fields. Invalid inputs raise `ValueError` or `TypeError`.
Keep requests unchanged while awaiting them. Give independent tasks distinct
sessions, serialize turns within a session, and mark its last turn final.
Response-based routing requires `allow_response=True`; task simulation rejects it.

## Library boundaries

The scorer rejects routing-time answers and visible conversation rewrites. It
cannot verify transport-time overrides such as `extra_body`, `omit_body_fields`,
or `reasoning_effort`; ensure these match the recorded experiment. Recorded task
scores do not validate mid-trajectory model switches or changed agent behavior.

Treat nested trial and dataset data as read-only. Import retains initial input
and metadata; `Report` retains aggregates, not result rows. Concurrency bounds
active decisions, not dataset memory. Release imported runs when no longer needed.

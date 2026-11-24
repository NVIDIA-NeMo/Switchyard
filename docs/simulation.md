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

## Use ATIF or custom recordings

`Trajectory.from_dict` copies an ATIF document; `to_dict` returns an independent
copy. `to_trial` extracts initial input and accepts explicit outcome metadata:

```python
from switchyard.sim import Run, Trajectory

record = {"task": "sum-1", "attempt": "1", "prompt": "What is 2 + 2?", "reward": 1}
trajectory = Trajectory.from_dict({
    "schema_version": "ATIF-v1.7",
    "steps": [{"source": "user", "message": record["prompt"]}],
})
trial = trajectory.to_trial(
    task_id=record["task"], trial_id=record["attempt"], target="fast",
    reward=record["reward"], model="fast-model",
)
fast = Run((trial,))
```

Convert each target's recordings into a `Run`, then use `Dataset.from_runs` and
`evaluate` as above. Supply model identity for the complete recorded run;
per-step models may differ. Optional measurements include `cost_usd`,
`cost_source`, `duration_seconds`, and `usage`. Missing values stay unknown.

Projection reads initial system and user messages before the first agent step.
It checks the ATIF version and fields it consumes, rather than the full schema.
Supply `task_input` when the initial task text is missing. Copied continuation
context requires the original task text, which replaces that context. Unmarked
summaries cannot reliably be detected. Keep rejected records in `Run.issues` as
`LoadIssue(source, message, task_id)` so failed repeats remain visible.

## Import rules and coverage

`load_harbor` accepts a trial, a job, or a downloaded `jobs/` directory. It reads
`result.json` and `agent/trajectory.json`; ATIF 1.5 and 1.7 text recordings are
covered. Use `dataset="benchmark-v1"` to namespace task IDs.

- Task checksums must agree. Without checksums, initial user messages must match.
  `input_target` chooses the routing input, using the first string-sorted trial ID.
- Repeats are averaged within each task and target; tasks receive equal weight.
  A measurement is unknown if any repeat lacks it. Zero is an observed value.
- Imports are strict. To retain errors and evaluate the common valid subset, use
  `load_harbor(..., on_error="record")` and `Dataset.from_runs(..., intersection=True)`.
  Invalid or missing tasks are excluded across all targets and listed in coverage.
- Rejections without a task ID and job trial-count mismatches must be repaired
  before pairing. Missing or malformed summaries cannot prove archive completeness.
- `task_inputs={task_id: original_text}` supplies missing input or replaces copied
  continuation context. Malformed trajectories and multimodal input still fail.

## Interpret and compare results

`report.to_dict()` includes coverage, selection counts, recorded reward/cost,
routing latency, call counts, usage, and routing cost. `complete` requires a scored
reward for every expected task and no evaluation errors; cost coverage is separate.
Unknown totals and means are `null`; `observed_total` retains the known portion.
A missing reward is unscored. A failed decision or rejected selection is an error.

Reward comparisons use tasks with observed rewards for the routed choice and
all fixed targets. The recorded-outcome oracle takes the best target mean per
task; it does not establish future policy performance. Cost comparisons use their
own common cohort. Compare the same task IDs across policies, not just equal counts.
Use separate development and held-out task sets when tuning policies.

Recorded cost prefers `result.agent_result.cost_usd`, then ATIF
`final_metrics.total_cost_usd`; `cost_source="trajectory"` reverses the preference.
Coverage records the source. Costs are not repriced, and duration covers agent
execution only. Signed rewards are supported; unrepresentable aggregates raise
`ValueError`.

Judge calls are live work, separate from recorded task cost. Supply
`price_call(call) -> float | None` for routing USD estimates. Return `None` for
unknown usage or prices. Without pricing, zero-call routes cost zero and other
routing costs stay unknown. Native usage separates cache tokens from other input;
reasoning detail can overlap output tokens. Retries may incur unreported spend.

The runner preserves the configured failure policy. Default `fail_open = true`
may select the capable target after a judge failure; the failed call remains in
the observations. Use `fail_open = false` for evaluation error rows instead.
Failures expose safe kind/status/target fields. Whole-decision timeouts have
unknown call counts and cost, since the provider may already have done work.

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

Applications with their own scheduler can use `Runner.decide`,
`score(task, decision)`, and `Report.add`. Requests use normalized, typed content:

```python
import asyncio
from switchyard.runner import Runner

async def decide():
    runner = Runner.load("routes.toml")
    return await runner.decide({
        "model": "auto",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Fix the parser."}]}],
    }, headers={"x-switchyard-session-id": "task-1", "x-switchyard-session-final": "true"})

decision = asyncio.run(decide())
print(decision.selected.target, decision.selected.model, decision.calls)
```

`selected.target`, fallback targets, and outcome model IDs use target keys;
`selected.model` and `call.model` use provider model IDs. Completed calls include
success, duration (including retries), and optional usage. `outcome` is the existing
`RoutingOutcome` binding. Execution failures raise `DecisionError` with safe
`kind`, `upstream_status`, `target`, `calls`, and `duration_seconds` fields.
Invalid configuration or request values raise `ValueError` or `TypeError`.

Keep requests unchanged while awaiting them. Use distinct sessions for independent
tasks; process turns within a session in order and mark only its last turn final.
Response-based routing needs `allow_response=True`; task simulation rejects it.

## Library boundaries

The scorer rejects routing-time answers and visible conversation rewrites. It
cannot verify transport-time overrides such as `extra_body`, `omit_body_fields`,
or `reasoning_effort`; ensure these match the recorded experiment. Recorded task
scores do not validate mid-trajectory model switches or changed agent behavior.

Treat nested trial and dataset data as read-only. Import retains initial input
and metadata; `Report` retains aggregates, not result rows. Concurrency bounds
active decisions, not dataset memory. Release imported runs when no longer needed.

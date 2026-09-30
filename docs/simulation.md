# Evaluate task routing

`switchyard.sim` evaluates a routing policy against completed agent runs. It
accepts ATIF trajectories, custom recordings converted to ATIF, and Harbor runs.
It loads the input available before an agent starts, asks Switchyard to select a
target, and scores that choice using the target's recorded task outcomes.

The simulator is a Python library included in `nemo-switchyard`. It uses the
public `switchyard.runner` Python interface. Switchyard's native bindings load
the existing TOML configuration and execute routing calls. No local server or
Harbor installation is needed.

## Start with paired recordings

Each target names a complete recorded model and agent configuration. Its name
must match a target in your Switchyard deployment. Use comparable task versions,
agent settings, reasoning budgets, and verifier settings across runs. A matching
task name or model name alone does not establish that the experiments are comparable.

Start with this `routes.toml` to check the pipeline without provider calls. Replace
`fast-model` and `strong-model` with the model IDs in your recordings:

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

This route always selects `fast`; the unused client is never called. `auto` is the
route ID passed to `evaluate` or `--route`. `fast` and `strong` are target keys used
by the dataset and `--run`; their `id` fields name the actual models. For a live
classifier, use the [Task routing configuration](routing_algorithms/llm_classifier_routing.md).

```python
import asyncio

from switchyard.runner import Runner
from switchyard.sim import Dataset, evaluate, load_harbor

fast = load_harbor("jobs/fast-baseline", target="fast")
strong = load_harbor("jobs/strong-baseline", target="strong")
dataset = Dataset.from_runs(
    {"fast": fast, "strong": strong},
    input_target="fast",
)

async def main():
    report = await evaluate(
        dataset,
        Runner.load("routes.toml"),
        route="auto",
        concurrency=8,
        timeout=60,
        on_result=lambda result: print(result.task_id, result.target, result.error),
    )
    print(report.format_text())
    return report.to_dict()

summary = asyncio.run(main())
```

Create a fresh `Runner` for each independent experiment. It owns routing state,
including affinity and random-number generators. The evaluator creates a distinct
session ID for each task and marks its decision as the final turn so native
per-session state can be released. Use `concurrency=1` with a seeded random route when task
order must be reproducible. Concurrent scheduling can change which task receives
each random draw.

The configured route may select any recorded target, including a fixed-target
subset. Model validation covers that route's configured completion targets; their
known recorded model IDs must match the configured model IDs. Verify the model
and agent settings of other recorded targets before using their fixed baselines. Use the
explicit `model_aliases={"recorded/provider/model": "configured/model"}` argument
to `evaluate` or `score` when the two systems name the same model differently. Aliases are a caller assertion;
the library does not guess equivalence by trimming model names. Missing model
metadata remains visible as unverified coverage.

For task evaluation, distinct completion targets in one route must have distinct model IDs. Native
decisions identify models, so two target keys sharing an ID cannot be scored
separately; decision validation rejects that route before provider calls. Reuse one target key
for the same candidate, or configure distinct served model IDs. Separate routes
can still use different target keys for the same model. Identical aliases remain
valid for ordinary serving and do not prevent evaluating other routes in the deployment.

## Use ATIF or custom recordings

`Trajectory.from_dict(data)` takes an owned copy of an ATIF document.
`trajectory.to_dict()` returns an independent copy with all recorded steps,
metrics, and extension fields preserved. You can load JSON with
`Trajectory.from_dict(json.loads(path.read_text(encoding="utf-8")))`.
Construction checks the ATIF version and steps container. `initial_messages()`
validates the fields it reads to extract text input; neither method performs full
ATIF schema validation.

A custom converter is a plain function returning `Trajectory`. For example,
suppose each line of `fast.jsonl` or `strong.jsonl` contains a harness record:

```json
{"task_id": "sum-1", "trial_id": "attempt-1", "model": "fast-model", "prompt": "What is 2 + 2?", "agent_steps": ["4"], "reward": 1.0, "cost_usd": 0.02}
```

Convert each record, then project it into a trial with explicit outcome metadata:

```python
import json

from switchyard.sim import Dataset, Run, Trajectory

def custom_to_atif(record):
    steps = [
        {"source": "user", "message": record["prompt"]},
        *({"source": "agent", "message": text} for text in record["agent_steps"]),
    ]
    return Trajectory.from_dict({
        "schema_version": "ATIF-v1.7",
        "session_id": record["trial_id"],
        "agent": {"name": "custom-agent", "version": "1"},
        "steps": [dict(step_id=i, **step) for i, step in enumerate(steps, start=1)],
    })

def trials_from_jsonl(path, target):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            yield custom_to_atif(record).to_trial(
                task_id=record["task_id"],
                trial_id=record["trial_id"],
                target=target,
                reward=record.get("reward"),
                model=record.get("model"),
                cost_usd=record.get("cost_usd"),
                cost_source="harness" if record.get("cost_usd") is not None else None,
            )

runs = {
    target: Run(tuple(trials_from_jsonl(f"{target}.jsonl", target)))
    for target in ("fast", "strong")
}
dataset = Dataset.from_runs(runs, input_target="fast")
```

Use the same `evaluate` call shown above. Each file must contain its own recorded
outcomes and model identity for comparable tasks. The converter owns the source
format; `to_trial` owns the ATIF input projection. It does not infer task IDs,
rewards, costs, or the recorded model from ATIF. Per-step models can override an
ATIF agent default, so choose and verify that metadata explicitly. Omitted
measurements and model IDs remain unknown. You can also supply `task_checksum`,
`duration_seconds`, `usage`, `source`, and `error` to `to_trial`.
This example stops on invalid records. If your importer tolerates errors, retain
each rejection as a `LoadIssue(source, message, task_id)` in `Run.issues`; silently
dropping failed repeats can bias the reported score.

`to_trial` keeps the initial input and supplied metadata without retaining the
full history. The generator above therefore releases each full trajectory before
reading the next record; `Run` retains only the trials. `HarborRun` is a
compatibility alias for `Run`. Neither path requires Harbor to be installed.

When a recording has no nonblank initial user message, `task_input` appends the
original task text after any recorded initial system and user messages. Without
that fallback, projection fails.

Copied continuation context can contain earlier answers or progress summaries.
If an initial step has `is_copied_context=true`, projection rejects it unless you
provide the original task text with `to_trial(task_input=...)` or
`initial_messages(task_input=...)`. That text replaces the initial conversation.
Choose root task recordings and have converters supply the original input before
execution. The library cannot reliably recognize unmarked subagent logs or summaries.
The full ATIF document remains available through `to_dict()`. Storing full history
does not implement trajectory replay or validate outcomes after model switching.

## Import rules and coverage

`load_harbor` accepts a trial directory, a job directory, or a downloaded run's
`jobs/` layout. It reads per-trial `result.json` and `agent/trajectory.json`.
ATIF 1.5 and 1.7 Claude/Codex recordings are covered by tests. The importer reads
the fields it needs using the Python standard library.

Routing input contains every initial system and user message before the first
agent step. This matters for Codex, whose first user message can describe the
environment and whose second contains the task. Completed agent messages,
verifier rewards, model names, and trial IDs are not added to classifier input.
The recorded initial input itself may contain harness or environment details.
`input_target` explicitly chooses which run supplies that input. For repeated
trials, it uses the first trial after sorting `trial_id` as strings. Keep that
choice fixed when comparing policies; the recorded outcomes still average all repeats.

Task names are matched within the supplied runs. Use `dataset="benchmark-v1"`
to namespace them. Conflicting task checksums are rejected. Blank strings and
non-string checksums are rejected; use `None` when unavailable. When a checksum is
missing, all initial user messages must match. Full system scaffolding may differ
between agents. Repeated trials retain their identities; each target's reward,
cost, and duration are averaged across all repeats of a task. Tasks then have
equal weight regardless of their number of repeats.

Missing rewards, cost, duration, and token counts stay unknown. A task measurement
is unknown if any of its repeats lacks that measurement. Agent errors remain on
the imported trials. An observed verifier reward of zero is retained as zero.

Imports and pairing are strict by default. To inspect an imperfect recording:

```python
fast = load_harbor("jobs/fast", target="fast", on_error="record")
strong = load_harbor("jobs/strong", target="strong", on_error="record")
dataset = Dataset.from_runs(
    {"fast": fast, "strong": strong},
    input_target="fast",
    intersection=True,
)
print(dataset.coverage)
```

The intersection excludes a task from every target if any target is missing or
has an invalid trial for it. Issues and excluded task IDs remain in the report.
When a job summary supplies a valid `n_total_trials`, it must match the number of
discovered trial directories, including rejected trials. A mismatch becomes an
issue with no task identity, so even an entirely missing repeat remains visible.
Repair incomplete copies or wait for unfinished jobs before pairing. Missing or
malformed summaries cannot establish directory completeness; check archive integrity
separately. For an intentional subset, import its trial directories individually.
An issue with a missing or blank task ID must be repaired before pairing;
otherwise a failed repeat could silently disappear from a task's mean. Custom
importers should use `None` when a rejected record's task cannot be identified.
For a missing trajectory or copied continuation context, callers can supply
`task_inputs={task_id: original_instruction}` to `load_harbor`.
Invalid JSON, unsupported ATIF versions, and invalid fields read during input
projection still fail validation. Unused history fields are not fully validated.
Raw agent logs and multimodal ATIF inputs are outside this importer.

## Interpret the report

The report includes processed, scored, unscored, and error counts; target
selection counts; recorded reward, task cost, and agent execution duration;
routing latency, call counts, usage, and routing cost coverage. Unknown totals
and means are `null`; `observed_total` retains the known portion. `complete`
means every expected task has a scored reward and no evaluation error. Inspect
each cost and usage field's coverage separately.

Native routing failures retain `routing_error_kind`, `routing_error_status`, and
`routing_error_target` on each result row. The status is an upstream HTTP code
when available; the target identifies the failing model, such as the classifier.
These fields distinguish failures such as HTTP 401 and 503 without storing
provider response bodies. Other exception details remain suppressed.

Reward comparisons use one common cohort where the routed outcome and every
fixed-target reward are observed. The empirical recorded-outcome oracle selects
the largest target mean reward for each task. It is an upper bound on these
recordings, not a validated policy or evidence about future tasks. Cost comparisons
use their own common cohort with all fixed-target costs observed. Their task
counts can differ from reward comparisons.

Recorded cost prefers `result.agent_result.cost_usd`, falling back to ATIF's
`final_metrics.total_cost_usd`. Choose `cost_source="trajectory"` to reverse that
preference. These producers can use different prices. Imported trials retain the
chosen source, and coverage includes source counts. Costs are not repriced to a
shared catalog. Duration covers only agent execution, excluding environment setup
and verification.

Routing calls are live classifier work. They are distinct from the selected
recorded task cost. Native usage is returned unchanged; cache fields are separate
from non-cached input, and reasoning detail must not blindly be added to output.
No current price catalog is embedded. For USD estimates, supply
`price_call(call) -> float | None` to `evaluate` or `score`. The callback receives
the actual model and native usage. Return `None` for unknown rates or usage.
Without it, zero-call routes cost zero and other routing costs remain unknown.
Backend retries and failed calls may incur unreported spend.

## Compare and tune policies

Reuse one paired dataset to compare configurations. Keep the target keys, recorded
model identities, and route ID the same. For example, keep `routes.toml` as the
fixed baseline above and put a candidate policy in `candidate.toml`:

```python
import asyncio
import json

from switchyard.runner import Runner
from switchyard.sim import evaluate

async def compare(dataset):
    summaries = {}
    rows = {}
    for name, path in {"fixed": "routes.toml", "candidate": "candidate.toml"}.items():
        rows[name] = []
        report = await evaluate(
            dataset, Runner.load(path), route="auto", concurrency=1,
            on_result=rows[name].append,
        )
        summaries[name] = report.to_dict()
    return summaries, rows

summaries, rows = asyncio.run(compare(dataset))
for name, summary in summaries.items():
    print(json.dumps({
        "policy": name,
        "complete": summary["complete"],
        "counts": summary["counts"],
        "reward_comparison": summary["comparison"],
        "cost_comparison": summary["cost_comparison"],
    }, indent=2))
```

Each configuration gets a fresh runner. The reward comparison includes routed,
fixed-target, and empirical-oracle means. Before ranking policies, check that all
runs are complete and the reward comparison covers every task. For a cost ranking,
also require full cost-comparison coverage. Unknown routing cost stays `null`;
supply the same `price_call` function to each evaluation to include classifier
cost estimates. If coverage is partial, use `on_result` rows to compare the same
task IDs across policies; matching counts alone do not establish a common cohort.

The example retains each policy's `Result` objects in `rows[name]`. For a partial
reward comparison, keep successful rows whose selected reward and every fixed-target
reward are known. Intersect those task IDs across policies, then recompute every
policy, fixed-target, and oracle mean on that intersection. Record the included and
excluded task IDs. Repeat this process separately for cost, requiring known routing
cost when comparing totals that include classifier calls. The saved rows contain
the recorded outcomes needed for these comparisons; no new routing calls are needed.
For large evaluations, use a synchronous callback to stream policy-tagged rows to
storage instead of retaining them in memory.

Keep the original recordings or their immutable content identities, converter code
version, task splits, `input_target`, configuration snapshots, Switchyard revision,
evaluation options, and pricing rules alongside those rows. The report's `run_id`
identifies an evaluation; it does not fingerprint its inputs or policy. Python API
calls leave artifact storage to the caller. The Harbor CLI records configuration
hashes and input paths, so retain the referenced content as well.

Split recordings by task ID before tuning, keeping every repeat of a task in the
same split. Use the development split to choose prompts, thresholds, or routing
rules. Freeze that choice before evaluating a separate held-out dataset. Reusing
held-out rewards to select the next policy makes them tuning data.

## Save progressive results

```sh
python -m switchyard.sim \
  --config routes.toml --route auto \
  --run fast=jobs/fast --run strong=jobs/strong \
  --input-target fast --output evaluation-output
```

This CLI imports Harbor layouts. Use the Python API for custom formats.
The output directory must be new. The CLI writes a `manifest.json` with the
configuration hash, inputs, package version, and coverage, then flushes each
completed row to `results.jsonl`. It writes `report.json` after evaluation returns.
The hash covers the exact TOML bytes loaded. Invalid routes and recorded-model
mismatches fail before output creation, leaving the path available for a corrected run.
The deployment source and credentials are not copied.
Use `--skip-invalid --intersection` to retain import issues and explicitly
evaluate the common valid subset. Exit status is 0 for complete reward coverage,
1 for an incomplete evaluation, and 2 for invalid inputs, configuration, or file errors.
An interruption exits 130 and preserves the manifest and completed result rows
once those files exist; there may be no final report. Automatic resume is not implemented.
If an output write fails, the last JSONL line may be incomplete. Earlier complete
lines remain usable; check the exit status and final report before treating a run as complete.
Use `--model-alias RECORDED=CONFIGURED` for an explicit model-ID equivalence.

The Python `on_result` callback runs synchronously on the event loop after each
completed task. Supply a regular function; asynchronous callbacks are not awaited.
It can update a dashboard or write an application-owned result stream. Keep it
short; an exception from the callback stops evaluation and drains pending work.
If the callback queues background writes, wait for them and handle their failures
before treating the records as saved.
Each decision has a deadline.
Cancellation stops local workers, but cannot revoke requests already received
by a provider. Task concurrency bounds simultaneous decisions; an algorithm may
make multiple provider calls inside a decision.

## Use the Python decision API

You can call `switchyard.runner` directly when your application owns task
scheduling and scoring. Use the same `routes.toml` shown above. Requests use
Switchyard's normalized format: `model` is the route ID, and message content is
a list of typed blocks.

```python
import asyncio
from uuid import uuid4

from switchyard.runner import DecisionError, Runner

async def inspect_decision():
    runner = Runner.load("routes.toml")
    request = {
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Fix the parser's handling of empty input."}],
        }],
    }
    try:
        decision = await asyncio.wait_for(
            runner.decide(request, headers={
                "x-switchyard-session-id": uuid4().hex,
                "x-switchyard-session-final": "true",
            }),
            timeout=60,
        )
    except DecisionError as error:
        print(error.kind, error.upstream_status, error.target, error.duration_seconds)
        for call in error.calls:
            print(call.model, call.is_success, call.duration_seconds, call.usage)
        raise

    print(decision.selected.target, decision.selected.model)
    print([target.target for target in decision.fallbacks], decision.duration_seconds)
    for call in decision.calls:
        print(call.model, call.is_success, call.duration_seconds, call.usage)
    return decision

decision = asyncio.run(inspect_decision())
```

The fixed route selects target key `fast` and model ID `fast-model`, with no model
calls. Classifier routes may make judge calls. `calls` contains completed logical
calls; durations include backend retries. `usage` is a mapping or `None`, and
missing token fields remain unknown. `DecisionError.target` names the failing
model, while `decision.selected.target` is the configured target key.

`Runner.load` and `Runner.from_toml` use Switchyard's configuration parser.
Configuration and request validation failures raise `ValueError`. Execution
failures raise `DecisionError` with safe diagnostics and completed observations;
an application deadline can instead raise `asyncio.TimeoutError`. Cancellation
waits for local routing and its Python bridge to stop.

Give independent tasks distinct session IDs. For successive turns of one session,
reuse its ID, await each decision in order, and mark only the last turn as final.
Use a fresh runner for independent experiments. `decision.outcome` retains the
existing `RoutingOutcome`. Algorithms that generate an answer while routing need
explicit `allow_response=True`; task simulation rejects them.

## Library boundaries

`Trajectory` preserves ATIF documents. `Trial` and `Run` represent projected
task evidence. `Dataset.from_runs` validates and pairs it.
`score(task, decision)` is a pure scorer for a saved native decision.
`evaluate` owns bounded scheduling and calls that scorer. `Report.add` accumulates
results without retaining full trajectories or result rows. Import processes
one trajectory file at a time; paired task inputs and small outcome records stay
in memory.

Treat projected `Trial`, `Task`, and `Dataset` records as read-only. Their frozen
dataclasses still contain mutable nested mappings. If you change recorded input,
create new trials and rebuild the dataset so pairing is validated again.

The task scorer also rejects answers and material conversation rewrites, because
recorded outcomes cannot score those changes.

Future trajectory replay can send successive normalized histories through the
same public Runner interface and supply its own scorer. Task simulation makes no
claim about mid-trajectory switching, replay fidelity, or changed agent behavior.

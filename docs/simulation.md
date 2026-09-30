# Evaluate task routing

`switchyard.sim` evaluates a routing policy against completed Harbor runs. It
loads the input available before an agent starts, asks Switchyard to select a
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
session ID for each task. Use `concurrency=1` with a seeded random route when task
order must be reproducible. Concurrent scheduling can change which task receives
each random draw.

The configured route may select any recorded target, including a fixed-target
subset. Known recorded model IDs must match the configured model IDs. Use the
explicit `model_aliases={"recorded/provider/model": "configured/model"}` argument
to `evaluate` or `score` when the two systems name the same model differently. Aliases are a caller assertion;
the library does not guess equivalence by trimming model names. Missing model
metadata remains visible as unverified coverage.

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
`input_target` explicitly chooses which run supplies that input.

Task names are matched within the supplied runs. Use `dataset="benchmark-v1"`
to namespace them. Conflicting task checksums are rejected. When a checksum is
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
An invalid result with no recoverable task identity must be repaired before
pairing; otherwise a failed repeat could silently disappear from a task's mean.
For a missing trajectory, callers can supply `task_inputs={task_id: instruction}`.
Malformed existing trajectories still fail validation. Raw agent logs and
multimodal ATIF inputs are outside this initial importer.

## Interpret the report

The report includes processed, scored, unscored, and error counts; target
selection counts; recorded reward, task cost, and agent execution duration;
routing latency, call counts, usage, and routing cost coverage. Unknown totals
and means are `null`; `observed_total` retains the known portion. `complete`
means every expected task has a scored reward and no evaluation error. Inspect
each cost and usage field's coverage separately.

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

## Save progressive results

```sh
python -m switchyard.sim \
  --config routes.toml --route auto \
  --run fast=jobs/fast --run strong=jobs/strong \
  --input-target fast --output evaluation-output
```

The output directory must be new. It contains flushed `results.jsonl` rows, a
`report.json` summary, and a `manifest.json` with configuration hash, inputs,
versions, and coverage. The deployment source and credentials are not copied.
Use `--skip-invalid --intersection` to retain import issues and explicitly
evaluate the common valid subset. Exit status is 0 for complete reward coverage,
1 for an incomplete evaluation, and 2 for invalid inputs or configuration.
Completed rows survive interruption; automatic resume is not implemented.
Use `--model-alias RECORDED=CONFIGURED` for an explicit model-ID equivalence.

The Python `on_result` callback runs after each completed task. It can update a
dashboard or write an application-owned result stream. Keep it short; a callback
failure stops evaluation and drains pending work. Each decision has a deadline.
Cancellation stops local workers, but cannot revoke requests already received
by a provider. Task concurrency bounds simultaneous decisions; an algorithm may
make multiple provider calls inside a decision.

## Library boundaries

`Trial` and `HarborRun` represent imported evidence. `Dataset.from_runs` validates
and pairs it. `score(task, decision)` is a pure scorer for a saved native decision.
`evaluate` owns bounded scheduling and calls that scorer. `Report.add` accumulates
results without retaining full trajectories or result rows. Import holds only
one full trajectory at a time; paired task inputs and small outcome records stay
in memory.

Native `Runner.load` and `Runner.from_toml` use Switchyard's configuration parser.
`await runner.decide(normalized_request, headers=...)` returns configured selected
and fallback targets, the existing `RoutingOutcome`, completed-call observations,
and elapsed routing time. Provider failures raise `DecisionError` with completed
observations and safe error metadata. Algorithms that generate an answer while
routing require explicit `allow_response=True`; task simulation rejects them.
The task scorer also rejects answers and material conversation rewrites, because
recorded outcomes cannot score those changes.

Future trajectory replay can send successive normalized histories through the
same public Runner interface and supply its own scorer. Task simulation makes no
claim about mid-trajectory switching, replay fidelity, or changed agent behavior.

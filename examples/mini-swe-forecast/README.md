# mini-swe-agent forecasts through OpenTelemetry

Run mini-swe-agent through Switchyard and observe a LightGBM forecast before
each model call. The forecast uses the observed execution history and its IR
snapshot. It predicts next-call token bands, remaining model calls and, on the
first call, peak context length. It does not change the agent's actions.

The example uses Switchyard's existing W3C trace propagation. A collector such
as Langfuse can display the forecast beside the actual model request:

```text
agent.call
├── lightgbm.forecast       predictions + source IR metadata
└── switchyard.request
    └── model generation   actual token usage and provider cost
```

## Prerequisites

This is a research integration. The feature extractor and checkpoints live in
the access-controlled [agent-progress-research repository](https://gitlab-master.nvidia.com/aire/agents/agent-progress-research).
They are supplied separately from Switchyard:

- Research checkout at `be51b905a` with Git LFS files downloaded.
- B checkpoint: `v1/checkpoints/context-lightgbm-b-cache-write/B` in that checkout.
  Short and tuned B bundles also work; they omit the cache-write head.
  C requires requirement annotations and is rejected by this live adapter.
- [replay-gap](https://gitlab-master.nvidia.com/aire/agents/replay-gap) checkout at
  `f21165163390fa76c0d448daa384a45b7580331c` for its mini-swe-agent wrapper.
- A compatible `base.traj.json` reference with `messages`, `info.config`, and
  `replay_gap.instance_id`. Only initial messages and harness settings start the
  new run. Recorded future responses are not passed to the agent.
- Python 3.12, uv, a `switchyard-server` executable, Docker and the task image.
  The recorded SWE-bench images require `linux/amd64`; ARM hosts use emulation.
- An upstream serving the reference's model and an OTLP HTTP trace collector.

The adapter reuses the research feature builder: five calls of usage history,
three calls of execution history and three IR snapshots. Models and encoders
are verified against their manifest and loaded once per run. These B bundles
were trained on Opus traces; another model needs separate validation.

## Start Switchyard

Run these commands from this directory. Copy `routes.example.toml` outside the
checkout and set the upstream URL and model ID. Both `id` fields must match the
reference model name after removing the leading `openai/` provider prefix.

```sh
uv sync --frozen
export UPSTREAM_API_KEY='<upstream credential>'
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='http://localhost:3000/api/public/otel/v1/traces'
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL='http/protobuf'
export OTEL_EXPORTER_OTLP_TRACES_HEADERS='Authorization=Basic <collector credential>'
export OTEL_SERVICE_NAME=switchyard

switchyard-server --config /path/to/routes.toml --dry-run
switchyard-server --config /path/to/routes.toml --host 127.0.0.1 --port 4110
```

The trace export settings must be present when Switchyard starts. For Langfuse,
the Basic credential is the base64 encoding of `public_key:secret_key`.

## Run one task

In a second terminal, export the same three `OTEL_EXPORTER_OTLP_TRACES_*`
settings. The agent uses its own service name, `mini-swe-lightgbm`.
Choose a new output directory outside the checkout:

```sh
uv run --frozen python run.py \
  --research-root /path/to/agent-progress-research \
  --replay-repo /path/to/replay-gap \
  --reference-trace /path/to/base.traj.json \
  --forecast-checkpoint /path/to/agent-progress-research/v1/checkpoints/context-lightgbm-b-cache-write/B \
  --base-url http://127.0.0.1:4110/v1 \
  --output /path/to/new-run \
  --max-steps 30
```

This starts fresh model requests and a Docker task environment. A submitted
agent solution is not a verifier result. The launcher forwards the trace parent
on every model request and restores the model's headers afterward.

## Fields and outputs

| Field | Meaning |
| --- | --- |
| `forecast.next_call.*` | Predicted token class, bounds and all class probabilities |
| `forecast.remaining_task.model_calls.count_estimate` | Remaining calls, including the upcoming call; L1-trained estimate |
| `forecast.task_start.peak_context_tokens.*` | Peak-context band, emitted on the first call |
| `ir.source.*` | IR from completed calls used for the forecast |
| `ir.observed.*` | IR after the acting call and its observed tool results |
| `actual.*` | Observed usage on the parent span, for comparison |

- Bands are token intervals, not confidence bounds or point token estimates.
  The highest class has `upper_exclusive: null` in JSON and
  `upper_unbounded: true` in telemetry.
- Cached input means cache reads. Uncached input includes cache writes; do not
  add cache writes to uncached input again when calculating total usage.
- IR metadata preserves each tool result's categories and subtypes. Multiple
  categories are labeled `Mixed`. Initial state is `Initial`.
- Forecast spans use custom attributes, not billable generation usage or
  `langfuse.observation.cost_details`. Switchyard exports actual model usage.
- `features.jsonl`, `forecasts.jsonl`, `actuals.jsonl`, `live.traj.json` and
  `status.json` are saved locally. Features and trajectories contain task
  content. This observer exports predictions and selected IR metadata; it does
  not export the full raw prefix. Switchyard's own tracing policy still applies.

For chartable Langfuse scores, also set `LANGFUSE_BASE_URL`,
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the agent terminal. The observer
submits `prediction.*` numeric scores asynchronously and links them to the
forecast span. Check `score-export.jsonl` for submission errors. Stable IDs
allow retrying saved forecasts without duplicating scores:

```sh
uv run --frozen python scores.py --run-folder /path/to/completed-run
```

## Checks

```sh
uv run --frozen pytest -q
uv run --frozen ruff check --config ../../pyproject.toml .
```

The tests use synthetic observations and an in-memory trace exporter. They
require no private checkouts, Docker, inference credentials or collector.

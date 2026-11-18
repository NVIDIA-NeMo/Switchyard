# mini-swe-agent forecasts through OpenTelemetry

**[Quickstart: run the agent with forecasts and dashboards](QUICKSTART.md)**

Run mini-swe-agent through Switchyard and observe a LightGBM forecast before
each model call. It uses the observed execution history and IR snapshots to
predict next-call token bands, remaining calls and initial peak context length.
Forecasts do not change the agent's actions.

```text
agent.call
├── lightgbm.forecast       predictions + source IR metadata
└── switchyard.request
    └── model generation   actual usage
```

The adapter reuses the external research feature builder: five calls of usage
history, three calls of execution history and three IR snapshots. It verifies
the saved bundle's manifest and keeps models and encoders resident. The
quickstart pins the research checkout, replay-gap wrapper and live B checkpoint.
C requires live requirement annotations and is rejected by this launcher.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Start a fresh mini-swe-agent task through the local proxy |
| `backend.py` | Connect the external feature builder and LightGBM bundle |
| `telemetry.py` | Link forecasts and observed IR/usage to the acting request |
| `scores.py` | Submit numeric forecast scores; retry saved forecasts |
| `routes.example.toml` | Upstream and passthrough-route template |
| `observability/up.sh`, `down.sh` | Start/stop local Langfuse and Grafana |
| `observability/switchyard_otel_env.sh` | Export trace settings before process startup |
| `observability/forecast_cost_enricher.py` | Price token bands and emit dashboard scores |
| `observability/build_grafana_dashboard.py` | Generate the dashboard at startup |
| `observability/grafana/provisioning/` | Connect Grafana to ClickHouse and the generated dashboard |
| `observability/langfuse_dashboard/dashboard.sql` | Install Langfuse dashboard widgets |
| `test_telemetry.py` | Offline checks for forecast boundaries and trace/score semantics |

## Telemetry and local output

| Field | Meaning |
| --- | --- |
| `forecast.next_call.*` | Token class, interval bounds and all class probabilities |
| `forecast.remaining_task.model_calls.count_estimate` | Remaining calls, including the upcoming call; L1-trained estimate |
| `forecast.task_start.peak_context_tokens.*` | Peak-context band on the first call |
| `ir.source.*` | Completed-call IR used for the forecast |
| `ir.observed.*` | IR after the acting call and observed tool results |
| `actual.*` | Observed usage on the parent span for comparison |

- Bands are token intervals, not confidence bounds. The highest class has
  `upper_exclusive: null` in JSON and `upper_unbounded: true` in telemetry.
- IR retains per-result categories and subtypes. Multiple categories stay
  `Mixed`; the initial state is `Initial`.
- The observer writes custom attributes, not billed generation usage. The
  optional cost enricher creates separate `predicted:opus-4-8` generations;
  filter by the actual model when reading Langfuse's aggregate cost views.
- Local outputs are `features.jsonl`, `forecasts.jsonl`, `actuals.jsonl`,
  `live.traj.json` and `status.json`. Features and trajectories contain task
  content; keep outputs outside Git. The observer exports selected IR metadata,
  not the full raw prefix. Switchyard's own tracing policy still applies.
- Optional `prediction.*` scores link to forecast spans. Check
  `score-export.jsonl` for errors. Stable score IDs allow retrying saved
  forecasts with `uv run --frozen python scores.py --run-folder /path/to/run`.

## Dashboard maintenance

- `up.sh` downloads Langfuse's compose file at `LANGFUSE_VERSION` (default
  `v4.82.0`). Dashboard queries use internal ClickHouse/Postgres schemas;
  validate them before changing the version.
- Grafana refreshes every five seconds; its **Run** selector defaults to
  `latest (auto)`. To change panels, edit the generator and run
  `python3 observability/build_grafana_dashboard.py`. Grafana reloads the file.
- The enricher and `up.sh` contain configured Opus demo prices. They must be
  updated together for a different deployment. The [quickstart](QUICKSTART.md#interpret-the-demo-correctly)
  explains midpoint pricing, unbounded bands and the remaining-cost estimate.

## Checks

```sh
uv run --frozen pytest -q
uv run --frozen ruff check --config ../../pyproject.toml .
```

Tests use synthetic observations and an in-memory trace exporter. No private
checkouts, Docker, inference credentials or collector are needed for these tests.

# Predicted vs actual cost dashboards

Local Langfuse and Grafana dashboards for this example. They price each
forecast's token bands and show predicted beside actual cost, live, per model
call and per task.

```text
run.py (agent.call, lightgbm.forecast) ─┐
switchyard-server (gen_ai.* usage)      ─┼─ OTLP ─► Langfuse :3000 ◄─ forecast_cost_enricher.py
                                         │                │ ClickHouse
                                         │                ▼
                                         └──────────► Grafana :3001 (per model call)
```

- Switchyard exports actual token usage on each `chat <model>` span.
- `run.py` exports `lightgbm.forecast` and `agent.call` spans in the same trace.
- `forecast_cost_enricher.py` polls Langfuse's ClickHouse every 3 s. For each
  forecast it writes a `predicted:opus-4-8` generation, priced at the band
  midpoints, into the same trace. It also writes numeric scores:
  `pred_next_call_cost_{low,mid,high}`, `pred_remaining_calls`,
  `pred_remaining_task_cost`, `pred_total_task_cost`, `actual_call_cost`,
  `actual_cost_so_far` and `next_call_in_band`.
- Langfuse shows cost by model (actual vs predicted) and a dashboard named
  "Switchyard cost: actual vs predicted".
- Grafana reads ClickHouse directly for per-call charts, which Langfuse's chart
  builder can't draw.

## Run

Requires Docker. From this directory:

```sh
./up.sh                              # Langfuse :3000, Grafana :3001, Opus 4.8 price, Langfuse dashboard
source ./switchyard_otel_env.sh      # in the Switchyard terminal and the agent terminal
python3 forecast_cost_enricher.py    # third terminal; keep it running
```

Then start Switchyard and run tasks as described in [../README.md](../README.md).
To get `prediction.*` scores, also set `LANGFUSE_BASE_URL=http://localhost:3000`,
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from `.env` in the agent terminal.

Open Grafana at <http://localhost:3001>. The **Run** variable defaults to
`latest (auto)` and follows new runs. The dashboard refreshes every 5 s.
Langfuse credentials are in `.env.example`. They're for local use only: change
them in `.env` before exposing either service.

| Script | Purpose |
| --- | --- |
| `up.sh` / `down.sh [-v]` | Start or stop the stack. `-v` deletes all stored data. |
| `clear_runs.sh` | Delete traces and scores but keep prices and dashboards. Restart the enricher afterwards. |
| `build_grafana_dashboard.py` | Regenerate `grafana/dashboards/per-call.json`. Grafana reloads it within about 10 s. |
| `langfuse_dashboard/dashboard.sql` | Langfuse dashboard and widgets. `up.sh` applies it. |

`up.sh` downloads Langfuse's self-hosting compose file at `LANGFUSE_VERSION`
(default `v4.82.0`) and runs the matching images.

## Grafana panels

- **All runs:** cost per run (actual, sum of per-call predictions, predicted at
  task start, predicted at the last call), cost by model, and totals.
- **Selected run:** task, model, router and checkpoint from span attributes;
  next-call cost (actual vs predicted band); the next-call token split
  (uncached input, cache reads, cache writes, output); the predicted task bill
  vs actual; remaining calls (predicted vs actual); cost by IR phase; and a
  per-call table.

## Pricing

The enricher and `up.sh` use Anthropic list prices for Opus 4.8, in USD per 1M
tokens: input $5, output $25, cache read $0.50, cache write $6.25. Bedrock or
other upstreams may differ. To change them, edit `PRICE` in the enricher and
the model price in `up.sh`.

- Predicted uncached input includes predicted cache writes, as in
  `actual.uncached_input_tokens`, so only the cache-write premium is added.
- Checkpoints without a cache-write head count cache writes as 0.
- A band with no upper bound is priced at its lower edge.

## Limitations

- Grafana and `dashboard.sql` use Langfuse's internal ClickHouse and Postgres
  schemas, not a public API. Keep `LANGFUSE_VERSION` pinned.
- Langfuse v4 runs in `events_only` mode, which disables its legacy trace and
  observation read APIs.
- Predicted generations count toward Langfuse's total-cost views. Filter by
  model to see actual cost only.
- Langfuse time-series charts bucket by minute at best. Use Grafana for
  per-call views.

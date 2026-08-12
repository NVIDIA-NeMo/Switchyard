# Cost Savings Reporting

Switchyard can report, in real time, how much a routed deployment is saving
compared to sending every request to a single baseline model. The feature is
purely additive: it prices the token counters the server already records, and
it never influences routing decisions.

## Enabling

Add a `[pricing]` table to the deployment TOML. Keys are **model ids** (the
`id` of a target, not the TOML target name), and rates are USD per 1 million
tokens:

```toml
[pricing."anthropic/claude-opus-4.7"]
input = 15.00
output = 75.00
cached = 1.50        # optional: cache-read rate, defaults to input x 0.1
cache_write = 18.75  # optional: defaults to input (no cache-write premium)

[pricing."moonshotai/kimi-k2.7-code"]
input = 0.60
output = 2.50

[savings]
baseline_model = "anthropic/claude-opus-4.7"
```

The optional `[savings]` section selects the baseline. When omitted, the most
expensive priced model (by combined input and output rate) is used. A
`baseline_model` must have its own `[pricing]` entry, and a `[savings]`
section without any `[pricing]` table is rejected at startup.

When no `[pricing]` table is present, the endpoints below are not registered
and the server behaves exactly as before.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/savings` | JSON snapshot of actual spend, baseline spend, and savings |
| `GET /dashboard` | Self-contained live HTML dashboard (auto-refreshes) |

`GET /v1/savings` prices every model that served completions and compares it
against serving the same token traffic with the baseline model:

```json
{
  "total_requests": 11,
  "actual_cost": 0.2864,
  "baseline_cost": 0.4722,
  "classifier_cost": 0.0026,
  "saved": 0.1858,
  "saved_pct": 39.35,
  "baseline_model": "anthropic/claude-opus-4.7",
  "models": { "...": { "calls": 7, "cost": 0.1256, "baseline_cost": 0.3141 } },
  "unpriced_models": []
}
```

Classifier and judge traffic is priced separately as `classifier_cost` and
deducted from the savings, so routing overhead is charged against the result
rather than hidden. Models that served traffic without a pricing entry are
costed at zero and listed in `unpriced_models` so under-counting is visible.

Counters reset together with the existing stats via `POST /v1/stats/reset`.

## Semantics

- Prompt tokens are split into base input, cache reads, and cache writes, and
  each bucket is priced separately, matching the cost model of the
  `switchyard.cli.launchers` cost estimator.
- The baseline cost re-prices each model's token traffic at the baseline
  model's rates. It answers "what would this traffic have cost on the
  baseline model", not "what would the baseline model have generated".
- Savings measure cost only. Whether the cheaper model's answers were good
  enough is a quality question the report cannot answer.

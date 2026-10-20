# Record Switchyard session costs

Switchyard records each observed call's cost in the routing log, including calls
made by classifiers and judges. To estimate costs that the upstream does not
report, start the server with a pricing file:

```bash
switchyard-server --config routes.toml --routing-log-file routing.jsonl --pricing-file prices.json
```

The JSON file maps served model IDs to USD per million tokens. These rates are
illustrative; supply your provider's rates:

```json
{
  "upstream/model": {
    "input": 1, "output": 5, "cache_read": 0.1, "cache_write": 2,
    "cache_write_1h": 3
  }
}
```

The four base rates are required; `cache_write_1h` is optional. The server rejects
unknown fields, duplicate, empty or padded model IDs, and rates outside 0 to 10000.
Zero means free usage. Use distinct served IDs when gateways charge different
rates for the same model. Rates load once at startup; saved costs do not change
when the pricing file changes.

A valid charge in `usage.cost` takes precedence, including zero. Buffered answers
can also report a charge in `x-litellm-response-cost`. Otherwise, Switchyard prices
input, output, cache reads and cache writes separately. Reasoning tokens already
count toward output. Reported one-hour cache writes require `cache_write_1h`;
writes without a reported duration use `cache_write`. Missing usage or rates leave
cost unknown. The table does not apply service-tier or long-context surcharges.
Omit `--pricing-file` to record reported charges alone.

Send `x-switchyard-session-id: my-session` with model requests, then read:

```bash
curl 'http://localhost:4000/v1/routing/session-stats?session_id=my-session'
jq -c '{model, cost}' routing.jsonl
```

A priced log record contains `"cost": {"usd": 0.01, "source": "estimated"}` or
`"source": "provider_reported"`; an unpriced record contains `"cost": null`.
The report includes session and per-model `cost`, plus `answer_cost` and
`routing_cost`. Each total has `known_usd`, `total_usd`, `unknown_calls`,
`estimated_calls`, and `provider_reported_calls`. If any recorded call has unknown
cost, `total_usd` is null and `known_usd` remains the priced subtotal. Old records
without a cost count as unknown.

Totals cover recorded calls, including failed calls observed by the server.
Classifier and judge calls cannot use header charges. Internal retries, cancelled
requests and abandoned streams may leave charges unrecorded. These totals cannot
establish the provider's bill or replace a client's native budget enforcement.

When Switchyard translates a buffered response to Chat Completions, it writes
cache-write tokens as `prompt_tokens_details.cache_write_tokens`. Earlier
versions wrote `cache_creation_tokens` there. Streamed Chat output already used
`cache_write_tokens`. Both names are accepted on input, and same-format replay
returns the upstream JSON unchanged.

The log writer holds one rate table in memory. The log grows per call and has no
size limit or rotation from this option. Failed appends produce a warning and are
not recovered; separate replica logs give separate totals. The session endpoint
scans the log on each read.

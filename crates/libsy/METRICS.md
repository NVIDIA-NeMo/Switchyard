# libsy metrics API

Enable the `metrics` feature on `switchyard-libsy` to collect and query metrics
in Rust. Queries are synchronous and require no server or async runtime.

## Collection and queries

Import these APIs from `switchyard_libsy::metrics`:

| API | Purpose |
| --- | --- |
| `Metrics::install()` | Create a collector and install its global OpenTelemetry provider |
| `Metrics::new()` | Create a collector without installing it |
| `Metrics::with_cardinality_limit(n)` | Create a collector with a chosen series limit |
| `metrics.meter_provider()` | Access the provider for host-managed installation |
| `metrics.stats(filter)` | Return typed, serializable metric series |
| `metrics.prometheus(filter)` | Return Prometheus text |
| `metrics.gather(filter)` | Return raw Prometheus metric families |
| `snapshot.counter_total(name)` | Sum a named counter across matching series |

Install the provider before running requests. Keep the collector for the
application's lifetime. Snapshots are cumulative since collector creation.

```rust
use switchyard_libsy::metrics::{MetricFilter, Metrics};

let metrics = Metrics::install()?;
// Run Switchyard requests.

let turn = metrics.stats(MetricFilter {
    session_id: Some("session-1"),
    agent_id: Some("root"),
    agent_turn_idx: Some(3),
    ..Default::default()
});
let tokens = turn.counter_total("switchyard_llm_call_total_tokens_total");

let all = metrics.stats(MetricFilter::default());
let text = metrics.prometheus(MetricFilter::default())?;
```

## Filters and values

`MetricFilter` accepts optional `session_id`, `agent_id`, `agent_turn_idx`, and
`user_turn_idx` fields. Omitted fields match all values. Indices start at zero;
agent indices identify inference requests, while user indices group continuations.

`MetricsSnapshot.series` contains `MetricSeries` values with a name, labels, and
`MetricValue`: counter, gauge, or histogram. Histograms retain count, sum, and
cumulative buckets. Names use Prometheus spelling.

## Labels and host-owned usage

From `switchyard_libsy::metric_context`:

- `for_request(&request)` derives identity labels and automatically tracks turn indices.
- `for_turn(session_id, agent_id, agent_turn_idx, user_turn_idx)` uses host-owned indices.
- `attributes(&base)` appends current identity labels to custom metric attributes.

Poll request futures or streams in the returned OpenTelemetry context. Attach it
for synchronous recording. Automatic indices are process-local; a missing agent
ID defaults to `root` when a session is present.

`metrics::record_usage(algorithm, model, call_kind, &usage)` records provider-reported
input, cache read/write, output, reasoning, and total token counts. Call kinds are
`routing`, `answer`, or `decision`. The Rust client driver records usage automatically;
hosts executing calls themselves should record it once. Unknown counts stay absent.

## Limits

The default limit is 2,000 label combinations per instrument. Further combinations
enter an overflow series and lose their original session/turn labels. Choose a
larger limit when retaining many turns.

`stats()` preserves metric names and labels; it has a different schema from the
server's `/v1/stats` summary.

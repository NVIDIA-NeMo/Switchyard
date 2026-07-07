# Prometheus + Alertmanager deployable artifacts

Drop-in Prometheus configuration for a Switchyard deployment. Pair with
[`docs/METRICS_REFERENCE.md`](../../docs/METRICS_REFERENCE.md) for the
full metric inventory and label semantics.

## Files

| File | Purpose |
|---|---|
| `prometheus.yml` | A single `scrape_configs:` entry to merge into your existing Prometheus config. Targets Switchyard's `/metrics`. |
| `switchyard.rules.yaml` | Alert rule group implementing the operational and success-criterion alerts. Validates with `promtool check rules`. |

A Grafana dashboard is intentionally not included — dashboard authoring
belongs to whichever team owns observability conventions in your
deployment. The metric catalog in `docs/METRICS_REFERENCE.md` is the
input.

## Wire-up

### 1. Add the scrape job

Merge the `scrape_configs:` entry from `prometheus.yml` into your
Prometheus configuration (adjust the `targets:` list to point at your
Switchyard pods or hosts). The endpoint is plain `GET /metrics` over
HTTP, no auth.

If you're running Switchyard on Kubernetes via
[`examples/k8s/latency-aware-deployment.yaml`](../k8s/latency-aware-deployment.yaml),
the service-level DNS is
`switchyard-latency.llm-routing.svc.cluster.local`. The example scrape
job uses static targets; a `kubernetes_sd_configs` variant is a one-step
swap for cluster-native discovery.

### 2. Load the alert rules

```bash
cp switchyard.rules.yaml /etc/prometheus/rules/
promtool check rules /etc/prometheus/rules/switchyard.rules.yaml
```

Then reference the file in your Prometheus config's `rule_files:`
section and reload:

```yaml
rule_files:
  - /etc/prometheus/rules/switchyard.rules.yaml
```

```bash
curl -X POST http://<prometheus>:9090/-/reload
```

### 3. Validate

Once Prometheus has scraped a few cycles, sanity-check the active
series:

```promql
{__name__=~"switchyard_.*"}
```

You should see the families documented in `METRICS_REFERENCE.md` —
top-line gauges, per-endpoint counters/summaries, Latency Service state,
poll health, and outcome counters.

## Alert tuning notes

Default thresholds and `for:` windows in `switchyard.rules.yaml` are
conservative — tuned to avoid pages during routine scrape-config rolls
or single-poll blips. Review them against your SLOs before enabling
notifications:

* `LatencyServiceUnreachable` fires after 60s of `poll_ok=0`. Bring this
  down if your routing logic should react faster than one poll cycle.
* `RouterOverheadHigh` is set at the success-criterion threshold (p99 >
  1 ms for 5 m). Retry-heavy workloads inflate this — cross-check
  `switchyard_upstream_attempts_total{outcome="retryable_error"}` rate
  before treating it as a routing-decision regression.
* `RouterNotRescuing` requires upstream failures to be occurring before
  it fires, so it does not flap on a healthy chain.

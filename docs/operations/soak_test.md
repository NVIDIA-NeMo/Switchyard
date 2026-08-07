# Soak test a release candidate

A Switchyard soak test sends sustained traffic through a release-candidate
server long enough to expose failures that short tests miss. Run it for 48
hours before code freeze when a release changes libsy, the Rust server,
routing, streaming, translation, or server lifecycle behavior.

The test sends closed-loop traffic through:

- OpenAI Chat Completions (`/v1/chat/completions`)
- Anthropic Messages (`/v1/messages`)
- OpenAI Responses (`/v1/responses`)
- streaming and non-streaming responses
- four repeated prompt prefixes

The runner also checks `/health` and `/metrics` every minute. Every five
minutes, it sends an invalid Chat Completions request, expects HTTP 400, and
then confirms the server is still live.

## Prepare the server

Run the exact commit, build, route bundle, backend, and model planned for the
release. Do not use a development server in front of a different Switchyard
build.

For the Python server:

```bash
uv sync
uv run switchyard --routing-profiles release-routes.yaml -- serve --port 4000 \
  > switchyard-soak.log 2>&1 &
SOAK_SERVER_PID=$!
```

For the Rust server and its libsy algorithms:

```bash
cargo build --release -p switchyard-server
target/release/switchyard-server --config release-routes.toml \
  > switchyard-soak.log 2>&1 &
SOAK_SERVER_PID=$!
```

Wait for `GET http://127.0.0.1:4000/health` to return HTTP 200 with the JSON
body `{"status": "ok"}`; the runner requires both before it starts. Check
`GET http://127.0.0.1:4000/v1/models` and choose the model id that represents
the release workload.

Run the server and test from a dedicated host, job scheduler, or terminal
multiplexer that will stay alive for the full test. Confirm that the host will
not suspend or restart and has enough disk space for the server log.

## Build the soak tester

The soak tester is a Rust binary. Build it once from the same checkout as the
release, then run it directly:

```bash
cargo build --release -p switchyard-soak
```

## Rehearse against a mock backend

Rehearse the soak with no live-provider credentials or cost by running
[VidaiMock](https://github.com/vidaiUK/VidaiMock) — an Apache-2.0 mock LLM server that speaks the
OpenAI, Anthropic, and Responses wire formats with realistic streaming — as the backend behind
`switchyard-server`. `scripts/soak-rehearsal.sh` starts the mock, points a passthrough
`switchyard-server` at it, waits for both to report healthy, and runs the soak against the local
stack:

```bash
# Requires the vidaimock binary on PATH (github.com/vidaiUK/VidaiMock/releases) and release builds
# of switchyard-server and switchyard-soak.
scripts/soak-rehearsal.sh --duration 5m --concurrency 8 --invalid-canary-interval 0
```

Disable the invalid-request canary (`--invalid-canary-interval 0`) against a mock backend: the
canary checks that invalid input is rejected with HTTP 400, which a permissive mock does not do.

To rehearse the failure gates, degrade the backend and confirm the soak fails closed. For example,
raise the mock's latency past the request timeout and every request times out:

```bash
MOCK_LATENCY_MS=2500 scripts/soak-rehearsal.sh --duration 30s --request-timeout 1 \
  --invalid-canary-interval 0
# Soak FAIL: ... error_rate=100.0000% ...  (error_kinds: timeout), exit 1
```

## Raw throughput with vegeta

The soak test is closed-loop and validates every response body and stream. For a quick open-loop
capacity figure — a fixed request rate, HTTP status only — [vegeta](https://github.com/tsenart/vegeta)
is a useful companion. It does not validate response bodies or streaming, so it complements the soak
rather than replacing it.

```bash
printf 'POST http://127.0.0.1:4000/v1/chat/completions\nContent-Type: application/json\n@body.json\n' > targets.txt
printf '{"model":"RELEASE_MODEL_ID","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' > body.json
vegeta attack -targets=targets.txt -rate=100 -duration=30s | vegeta report
```

## Run the 48-hour test

Choose concurrency from the release capacity plan. Increase it in short
rehearsals until you find the highest expected steady load that remains below
the backend's rate limit. Use that load for the 48-hour run. An overload test
that spends most of its time throttled does not measure release stability.

```bash
./target/release/switchyard-soak \
  --base-url http://127.0.0.1:4000 \
  --model RELEASE_MODEL_ID \
  --duration 48h \
  --concurrency 16 \
  --server-pid "$SOAK_SERVER_PID" \
  --max-rss-growth-mib 512
```

The runner keeps 16 requests in flight until the test ends. This can generate
large usage charges against a metered backend. Use a dedicated test deployment,
estimate the request volume first with a short run, and get approval for any
paid-provider cost.

Use a five-minute rehearsal to confirm the route and result files:

```bash
./target/release/switchyard-soak \
  --base-url http://127.0.0.1:4000 \
  --model RELEASE_MODEL_ID \
  --duration 5m \
  --concurrency 4 \
  --report-interval 10
```

If the Switchyard endpoint requires a bearer token, pass the environment
variable name instead of putting the token on the command line:

```bash
export SWITCHYARD_SOAK_TOKEN="..."
./target/release/switchyard-soak \
  --api-key-env SWITCHYARD_SOAK_TOKEN \
  --model RELEASE_MODEL_ID
```

## Pass criteria

The command exits with status 0 only when:

- the requested duration completes;
- at least one inference request completes;
- the inference error rate stays at or below `--max-error-rate` (default `0`,
  which means no inference request may fail);
- every periodic liveness check passes;
- every `/metrics` read returns both Switchyard request counters;
- every requested process sample returns RSS and CPU data;
- every invalid-request recovery check passes;
- the server request counter never resets; and
- RSS growth stays within `--max-rss-growth-mib` when that limit is set.

If the release plan permits transient failures from a remote provider, set an
explicit error budget with `--max-error-rate`. Record the reason for that
exception in the release record.

The RSS limit is deployment-specific. Set it from an approved baseline for the
same model, concurrency, and worker count. Omit `--server-pid` and
`--max-rss-growth-mib` when the server runs on another host, then collect
memory and restart data from that host's monitoring system.

## Review the results

Each run creates a timestamped directory under `soak-results/`:

- `config.json` records the non-secret test inputs.
- `intervals.csv` records request rate, errors, latency percentiles, health,
  Switchyard counters, RSS, and CPU once per reporting interval. `cpu_percent`
  is the `ps` lifetime-average CPU for the process, not the interval's usage, so
  read it as a long-run average rather than a spike detector.
- `errors.jsonl` records up to 10,000 request and canary failures.
- `summary.json` records the final pass result and any failed gates.
- `status.json` is overwritten atomically each interval with a one-object live snapshot — progress
  toward the target duration, cumulative requests, cumulative error rate, health, detected restarts,
  and a `status` of `ok`, `degraded`, or `stalled`. Read it (or tail the run log, whose interval line
  carries the same fields and status token) to monitor a run in progress; on a remote host,
  `cat <results-dir>/status.json` gives an at-a-glance confidence check without parsing `intervals.csv`.

When the run ends the command also prints a latency percentile table (p50 through
p99.9) and an ASCII latency histogram to the terminal, for a quick read of the
distribution without opening the result files.

Before approving the release, check `intervals.csv` for late failures, falling
throughput, increasing p95 or p99 latency, and steady RSS growth. Compare the
first and last several hours, not only the run-wide averages. Attach
`summary.json`, the interval chart, the Switchyard log, the tested commit, and
the route bundle to the release record.

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

Run the exact commit, build, server config, backend, and model planned for the
release. Do not use a development server in front of a different Switchyard
build.

Start the standalone server and its libsy algorithms:

```bash
cargo build --release -p switchyard-server
target/release/switchyard-server --config release-routes.toml \
  > switchyard-soak.log 2>&1 &
SOAK_SERVER_PID=$!
```

Wait for `GET http://127.0.0.1:4000/health` to return HTTP 200 with the JSON
body `{"status": "ok"}`; the runner requires both before it starts. Check
`GET http://127.0.0.1:4000/v1/models` and choose the model id that represents
the release workload. Pass that exact id with `--model`.

Run the server and test from a dedicated host, job scheduler, or terminal
multiplexer that will stay alive for the full test. Confirm that the host will
not suspend or restart and has enough disk space for the server log.

## Build the soak tester

The soak tester is a Rust binary. Build it once from the same checkout as the
release, then run it directly:

```bash
cargo build --release -p switchyard-soak
```

## Check the local server and load tools

The local test embeds [VidaiMock](https://github.com/vidaiUK/VidaiMock) as a Rust library, so it does
not need a separate `vidaimock` command. Build the server, soak tester, and mock helper from the
commit under test:

```bash
cargo build --release -p switchyard-server -p switchyard-soak \
  --bins --example switchyard-soak-mock
```

Install [oha](https://github.com/hatoo/oha) and
[NVIDIA AIPerf](https://docs.nvidia.com/aiperf/reference/command-line-options):

```bash
cargo install oha
uv tool install aiperf
```

Then run the local test from the repository root:

```bash
python3.12 scripts/run_local_soak_test.py --duration 10s --concurrency 4
```

`--duration` controls how long the Rust soak tester runs. The route checks, oha, and AIPerf runs are
short and limited by request count. `--help` explains every flag. Set `OHA_BIN`, `AIPERF_BIN`,
`SWITCHYARD_SERVER_BIN`, `SWITCHYARD_SOAK_BIN`, or `SWITCHYARD_SOAK_MOCK_BIN` when a command is not
on `PATH` or not under `target/release`.

The script checks all five commands before starting a process. A missing command prints a warning,
the build or install command, and the matching environment variable. Cargo compiles the embedded
VidaiMock library, but it does not install the unrelated oha or AIPerf programs during a normal
build.

The script gives each tool one job:

| Tool | Job in the local test |
|---|---|
| Embedded VidaiMock | Supplies local OpenAI-compatible model responses with configurable latency and no provider cost. |
| oha | Sends raw concurrent HTTP requests through the random route and writes a status and latency distribution. |
| Python route checks | Sends one Chat Completions request through each route. The stage request includes a critical tool failure so the signal scorer selects the capable target. |
| AIPerf | Sends streaming Chat Completions through passthrough routing and records LLM, token, and response-time results. |
| `switchyard-soak` | Runs passthrough routing through Chat Completions, Messages, and Responses, with streaming on and off, while checking server health, metrics, process use, and required results. |

`scripts/local_soak_test.toml` exercises `noop`, `random`, `passthrough`, `llm_classifier`, and
`stage_router`. It uses the accepted maximum retry count (10), a zero-weight random target, and the
upper classifier and stage thresholds (1.0). The config is validated with
`switchyard-server --dry-run` before either service starts.

The runner does not add limits for target count or recent-turn history because the server does not
limit them. Rust tests cover the real limits: 4,096 saved route assignments, 100,000 response-time
samples, and 10,000 error records. Server tests cover invalid thresholds, bad classifier responses,
missing stage-router request IDs, target errors, and retry count 11, which is one above the maximum.

## Run the 48-hour test

Choose concurrency from the release capacity plan. Increase it in short
test runs until you find the highest expected steady load that remains below
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

Use a five-minute run to confirm the route and result files:

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

Tail the run log to monitor progress, cumulative error rate, health, RSS, and the
`OK`, `DEGRADED`, or `STALLED` interval status while the test runs.

Before approving the release, check `intervals.csv` for late failures, falling
throughput, increasing p95 or p99 latency, and steady RSS growth. Compare the
first and last several hours, not only the run-wide averages. Attach
`summary.json`, the interval chart, the Switchyard log, the tested commit, and
the server config to the release record.

# Soak test a release candidate

A Switchyard soak test sends sustained traffic through a release-candidate
server long enough to expose failures that short tests miss. Run it for 48
hours before code freeze when a release changes libsy, the Rust server,
routing, streaming, translation, or server lifecycle behavior.

The standard test sends closed-loop traffic through:

- OpenAI Chat Completions (`/v1/chat/completions`)
- Anthropic Messages (`/v1/messages`)
- OpenAI Responses (`/v1/responses`)
- streaming and non-streaming responses
- short and long inputs, long outputs, shared prefixes, and mixed request sizes
- growing conversations, large tool catalogs, tool-call bursts, and stage transitions
- deterministic easy/hard classifier mixes

The runner also checks `/health` and `/metrics` every minute. Every five
minutes, it sends an invalid Chat Completions request, expects HTTP 400, and
then confirms the server is still live.

The scenario catalog covers these distinct pressure angles:

| Scenario | Pressure angle | What to review |
|---|---|---|
| `short-interactive` | Short prompts under fixed load, a concurrency knee, or a 10x request-rate burst | HTTP ceiling, TTFT, routing overhead, and the saturation point |
| `long-context` | 8K, 32K, and near-window inputs | TTFT, memory, and context-sensitive route failures |
| `decode-heavy` | 512-token and 1,024-token output limits | ITL, output tokens/second, and stream stability |
| `prefix-reuse` | Matched shared and unique long prefixes | Cache-sensitive TTFT and token accounting |
| `mixed-traffic` | A 70/20/10 short, medium, and long mix | p99 latency and head-of-line effects |
| `growing-conversation` | Eight cumulative conversation turns in one session | Affinity, history cost, and per-turn latency growth |
| `large-tool-catalog` | 16-tool and 64-tool JSON-schema catalogs | Serialization/routing overhead and intact tool forwarding |
| `tool-call-burst` | Eight linked assistant call/tool-result turns | Session continuity and burst handling |
| `stage-transitions` | One growing history across exploration, critical failure, and productive work | Stage-router tier changes and scorer output |
| `classifier-mix` | Deterministic 80/20 then 50/50 easy/hard requests | Target share, classifier calls/errors, and classifier latency |
| `context-overflow` | One target rejects a near-window request | Fallback to another eligible target |
| `failure-pressure` | Bounded 429, 500, malformed verdict, and truncated stream | Retry recovery, explicit terminal errors, and connection health |
| `client-cancellation` | A client timeout during a delayed response | Teardown and recovery of later traffic |

`standard` includes the core and agentic rows. Run `resilience` separately because expected
failures should not be compared as throughput samples.

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

## Compare routing algorithm performance

Use `scripts/benchmark_routing_algorithms.py` to compare routing algorithms under the same load.
The command runs oha and AIPerf sequentially for every model id, then writes `report.md`,
`report.csv`, and `report.json` beside both tools' raw results:

```bash
python3.12 scripts/benchmark_routing_algorithms.py \
  --base-url http://127.0.0.1:4000 \
  --model noop=switchyard/noop \
  --model passthrough=switchyard/passthrough \
  --model random=switchyard/random \
  --model llm_classifier=switchyard/classifier \
  --model stage_router=switchyard/stage \
  --concurrency 100 \
  --request-count 1000 \
  --scenario-set standard \
  --load-profile fixed \
  --profile-runs 3 \
  --backend-label "release model deployment"
```

The Rust crate exports one AIPerf `inputs-json` file per scenario. oha reuses the first exported
`short-interactive` payload as a non-streaming fixed body and reports the raw HTTP request rate and
latency ceiling. AIPerf replays every selected streaming session and reports request latency, time
to first token (TTFT), inter-token latency (ITL), request throughput, output-token throughput, and
multi-run confidence intervals. The command saves `/v1/stats` before and after each AIPerf run so
the report also records selected-target calls and shares, classifier calls/errors and latency, and
mean routing overhead. It runs
jobs sequentially because simultaneous tools would compete for the same server capacity.

Use the load schedules after the fixed scenario comparison establishes a baseline:

```bash
python3.12 scripts/benchmark_routing_algorithms.py \
  --base-url http://127.0.0.1:4000 \
  --model random=switchyard/random \
  --model classifier=switchyard/classifier \
  --scenario short-interactive \
  --load-profile concurrency-knee \
  --load-profile traffic-burst \
  --concurrency 128 \
  --request-rate 20
```

The concurrency knee uses bounded steps up to `--concurrency`. The traffic burst holds the base
request rate, raises it to 10 times that rate for five seconds, then returns to the base. Pass
`--load-profile all` to run fixed, knee, and burst schedules. These schedules apply to the short
baseline; they are not separate request scenarios.

To isolate routing overhead, configure every route to use the same target deployment and keep the
scenario manifest, concurrency, request count, and profile-run count fixed. Set `--tokenizer` to
the real model tokenizer when exact token counts matter. AIPerf uses deterministic sessions and a
fixed seed so each route receives the same requests.

This comparison command does not start a backend or Switchyard. Point it at a running Switchyard
server backed by real models to measure end-to-end TTFT and token throughput with real tokenization
and generated tokens. Real-model runs cost tokens and include provider queuing and model variance,
so use a dedicated deployment and repeat the run before treating a small difference as an
algorithm effect. Use the local scenario backend first for deterministic routing correctness and
overhead, then rerun the same manifest against real models for capacity claims.

## Check the local server and load tools

The local test uses a request-aware Axum backend from the soak crate. Build the server, soak tester,
and scenario backend from the commit under test:

```bash
cargo build --release -p switchyard-server -p switchyard-soak \
  --bins --example switchyard-soak-mock
```

Install [oha](https://github.com/hatoo/oha) and
[NVIDIA AIPerf](https://docs.nvidia.com/aiperf/reference/command-line-options):

```bash
cargo install oha
uv tool install --python 3.12 aiperf
```

Then run the local test from the repository root:

```bash
python3.12 scripts/run_local_soak_test.py \
  --duration 10s \
  --concurrency 4 \
  --request-count 100
```

`--duration` controls how long the Rust soak tester runs. `--request-count` controls how many
measured requests each load tool sends to each algorithm. `--help` explains every flag. Set
`OHA_BIN`, `AIPERF_BIN`,
`SWITCHYARD_SERVER_BIN`, `SWITCHYARD_SOAK_BIN`, or `SWITCHYARD_SOAK_MOCK_BIN` when a command is not
on `PATH` or not under `target/release`.

The script checks all five commands before starting a process. A missing command prints a warning,
the build or install command, and the matching environment variable. Cargo builds the scenario
backend, but it does not install oha or AIPerf during a normal build.

The script gives each tool one job:

| Tool | Job in the local test |
|---|---|
| Scenario backend | Returns local OpenAI-compatible responses, valid easy/hard classifier verdicts, and bounded failures with no provider cost. |
| oha | Measures the non-streaming `short-interactive` HTTP baseline through every route. |
| Python route checks | Sends one ordinary Chat Completions request through each configured route before load starts. |
| AIPerf | Replays Rust-exported streaming sessions through every route and records LLM, token, response-time, and confidence results. |
| Combined report | Joins scenario, load, oha, AIPerf, and routing-counter metrics in Markdown, CSV, and JSON. It keeps resilience rows separate from throughput rows. |
| `switchyard-soak` | Runs the standard scenario set while checking public API variants, server health, metrics, process use, and required results. |

`scripts/local_soak_test.toml` exercises `noop`, `random`, `passthrough`, `llm_classifier`, and
`stage_router`. It uses the accepted maximum retry count (10), a zero-weight random target, and the
upper classifier and stage thresholds (1.0). Classifier affinity is disabled so every measured
request includes the classifier call. The scenario backend returns `p_solve=1.0` for easy markers
and `p_solve=0.1` for hard markers; with the configured threshold, those requests select the weak
and strong targets respectively. The config is validated with
`switchyard-server --dry-run` before either service starts.

Run resilience cases separately so expected transport failures do not contaminate throughput
comparisons:

```bash
python3.12 scripts/benchmark_routing_algorithms.py \
  --base-url http://127.0.0.1:4000 \
  --model classifier=switchyard/classifier \
  --scenario-set resilience \
  --load-profile fixed \
  --profile-runs 1
```

`context-overflow` checks target fallback, `failure-pressure` injects bounded 429, 500, malformed
classifier, and truncated-stream cases, and `client-cancellation` uses a one-second client timeout
against a delayed response. Their expected error-rate ranges appear in the report's Resilience
section, and the command fails after writing the report when any row misses its range. The local
runner calls the scenario backend's `/reset` endpoint before each AIPerf cell so every algorithm
receives the same transient-failure sequence. When you run the comparison command directly against
that backend, pass `--scenario-backend-reset-url http://127.0.0.1:8100/reset` to preserve the same
comparison.

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

# switchyard-soak

`switchyard-soak` keeps a fixed number of requests in flight against one route on a live
`switchyard-server`. Its scenario catalog covers short and long inputs, long outputs, shared
prefixes, mixed traffic, growing conversations, tool schemas, tool-call bursts, routing signals,
and bounded failure cases. The short baseline also exercises Chat Completions, Messages, and
Responses in streaming and non-streaming form. The command samples health, metrics, and an
optional local server process, then exits with status 1 when a release gate fails.

Build the server and soak tester from the commit being tested:

```bash
cargo build --release -p switchyard-server -p switchyard-soak \
  --bins --example switchyard-soak-mock
```

Run a five-minute check against a route advertised by `GET /v1/models`:

```bash
target/release/switchyard-soak \
  --base-url http://127.0.0.1:4000 \
  --model switchyard/general \
  --duration 5m \
  --concurrency 4 \
  --report-interval 10
```

Run the 48-hour release gate and measure the local server process:

```bash
target/release/switchyard-soak \
  --model switchyard/general \
  --duration 48h \
  --concurrency 16 \
  --server-pid "$SWITCHYARD_SERVER_PID" \
  --max-rss-growth-mib 512
```

If the endpoint needs a bearer token, put the secret in an environment variable and pass the
variable's name:

```bash
export SWITCHYARD_SOAK_TOKEN="..."
target/release/switchyard-soak \
  --model switchyard/general \
  --api-key-env SWITCHYARD_SOAK_TOKEN
```

## Flags

| Flag | Default | What it does |
|---|---:|---|
| `--base-url URL` | `http://127.0.0.1:4000` | Sends every check and inference request to this Switchyard server. |
| `--model ID` | required | Selects one exact route id returned by `GET /v1/models`. The test stops before sending load if the id is missing. |
| `--duration TIME` | `48h` | Keeps the load running for this long. Use seconds, minutes, or hours, such as `30s`, `5m`, or `48h`. |
| `--concurrency N` | `16` | Keeps this many requests in flight. Each worker waits for its response before sending another request. |
| `--max-output-tokens N` | `32` | Sets the default output limit. `decode-heavy` uses its bounded 512-token and 1,024-token limits. |
| `--prompt-bytes N` | `1024` | Sizes the repeated input used by short and prefix-reuse scenarios. Larger values put more pressure on request memory and prefix caching. |
| `--scenario-set NAME` | `standard` | Selects `core`, `agentic`, `resilience`, `standard`, or `all`. `standard` includes core and agentic scenarios without expected failures. |
| `--scenario ID` | none | Selects one exact scenario id. Repeat the flag to select more scenarios; explicit ids take precedence over `--scenario-set`. |
| `--context-window-tokens N` | `32768` | Bounds generated long-context and overflow inputs. The command rejects values above 1,000,000. |
| `--export-scenarios PATH` | none | Writes a manifest and AIPerf `inputs-json` files, then exits without contacting a server. The performance script uses this mode. |
| `--request-timeout SECONDS` | `120` | Limits connection time and idle time between response bytes. A healthy stream may run longer if it keeps producing data. |
| `--report-interval SECONDS` | `60` | Sets how often the test samples health, metrics, RSS, CPU, and interval latency. |
| `--invalid-canary-interval SECONDS` | `300` | Sends malformed input this often, expects HTTP 400, and then checks recovery. Set it to `0` only when a permissive mock cannot reject the canary. |
| `--max-error-rate FRACTION` | `0` | Allows at most this fraction of inference requests to fail. `0.01` means one percent. The value must be between 0 and 1. |
| `--server-pid PID` | none | Samples RSS and CPU from this local `switchyard-server` process. Omit it for a remote server. |
| `--max-rss-growth-mib MIB` | none | Fails when the last RSS sample exceeds the first by more than this amount. It requires `--server-pid`. |
| `--api-key-env NAME` | none | Reads a bearer token from this environment variable without putting the secret in the command or result files. |
| `--results-dir PATH` | `soak-results/<UTC time>` | Creates this new directory for the run. The command refuses to reuse an existing directory. |
| `--help` | n/a | Prints the command reference and examples. |
| `--version` | n/a | Prints the crate version. |

The scenario source lives in `src/scenarios/`. Each request shape has one file and a pure builder.
The exported manifest is the only scenario input read by the Python performance runner, so model
ids, sessions, tools, and message histories cannot drift between the Rust soak and AIPerf runs.
Concurrency knees and traffic bursts are load profiles attached to the short baseline, not
duplicate request scenarios.

The catalog contains:

| Group | Scenarios |
|---|---|
| Core | `short-interactive`, `long-context`, `decode-heavy`, `prefix-reuse`, `mixed-traffic` |
| Agentic | `growing-conversation`, `large-tool-catalog`, `tool-call-burst`, `stage-transitions`, `classifier-mix` |
| Resilience | `context-overflow`, `failure-pressure`, `client-cancellation` |

## Check the local server and load tools

`scripts/run_local_soak_test.py` starts the crate's request-aware scenario backend and
`switchyard-server`, sends one HTTP request through each configured route, then runs `oha` and
NVIDIA AIPerf sequentially for every routing algorithm. The backend returns valid classifier
verdicts and injects the context, retry, stream, and cancellation cases named by resilience
scenarios. It resets transient-failure counters before each AIPerf cell so every routing algorithm
receives the same attempts. The local test needs no provider key and incurs no inference cost.

Run `scripts/benchmark_routing_algorithms.py` by itself against an existing Switchyard server to
compare the same routes with real model output. oha runs only for the fixed `short-interactive`
baseline. AIPerf replays every selected session and reports TTFT, ITL, request throughput, output
token throughput, multi-run confidence, selected-target calls, classifier calls, and routing
overhead. The operations guide explains how to keep the comparison fair and when to use the local
backend or real models.

The script needs the built Rust programs plus installed `oha` and AIPerf commands.
`switchyard-soak` itself does not require those extra programs and can run by itself against a
release server. See the [operations guide](../../docs/operations/soak_test.md) for setup, the local
command, the 48-hour release run, and result review.

## Results

Each run creates `config.json`, `intervals.csv`, `errors.jsonl`, and `summary.json`. Error details
stop at 10,000 records, and run-wide latency uses a 100,000-sample reservoir, so failures and a
high request rate cannot grow memory or disk use without bound.

See `docs/operations/soak_test.md` for release preparation, pass criteria, and result review.

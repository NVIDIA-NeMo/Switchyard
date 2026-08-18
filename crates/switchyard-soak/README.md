# switchyard-soak

`switchyard-soak` keeps a fixed number of requests in flight against one route on a live
`switchyard-server`. It exercises Chat Completions, Messages, and Responses in both streaming and
non-streaming form. It also samples health, metrics, and an optional local server process. The
command writes evidence to a new results directory and exits with status 1 when a release gate
fails.

Build the server and soak tester from the commit being tested:

```bash
cargo build --release -p switchyard-server -p switchyard-soak
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
| `--max-output-tokens N` | `32` | Sends this output-token limit with every request. It must be at least 1. |
| `--prompt-bytes N` | `1024` | Adds this many bytes of repeated prefix to each of four prompts. Larger values put more pressure on request memory and prefix caching. |
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

The inference workers use a fixed six-case cycle: three public endpoint formats multiplied by
streaming on and off. A deterministic cycle makes missing endpoint or streaming coverage visible
in short runs.

## Check the local server and load tools

`scripts/run_local_soak_test.py` starts an embedded VidaiMock server and `switchyard-server`, sends
one HTTP request through each configured route, then runs `oha`, NVIDIA AIPerf, and
`switchyard-soak`. VidaiMock returns fixed local responses, so this local test needs no provider key
and incurs no inference cost.

The script needs the built Rust programs plus installed `oha` and AIPerf commands.
`switchyard-soak` itself does not require those extra programs and can run by itself against a
release server. See the [operations guide](../../docs/operations/soak_test.md) for setup, the local
command, the 48-hour release run, and result review.

## Results

Each run creates `config.json`, `intervals.csv`, `errors.jsonl`, and `summary.json`. Error details
stop at 10,000 records, and run-wide latency uses a 100,000-sample reservoir, so failures and a
high request rate cannot grow memory or disk use without bound.

See `docs/operations/soak_test.md` for release preparation, pass criteria, and result review.

# Quickstart: live agent forecasts and cost dashboards

Run mini-swe-agent on `django__django-14631` through Switchyard. LightGBM B
predicts token bands and remaining model calls before each request. Langfuse
and Grafana show those predictions alongside observed usage and cost.

Draft PR: [NVIDIA-NeMo/Switchyard#1342](https://github.com/NVIDIA-NeMo/Switchyard/pull/1342).

## 1. Get the code and checkpoint

You need Python 3.12, uv, Git LFS, a Rust toolchain, Docker Compose, and access
to the two internal repositories below. Start Docker. The task image uses
`linux/amd64`; an ARM laptop needs Docker emulation.

Run in a directory where you want the three checkouts:

```bash
git clone --branch sabhatinas/lightgbm-otel-demo \
  https://github.com/NVIDIA-NeMo/Switchyard.git

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://gitlab-master.nvidia.com/aire/agents/agent-progress-research.git
GIT_LFS_SKIP_SMUDGE=1 git -C agent-progress-research checkout be51b905a
git -C agent-progress-research lfs pull \
  --include="v1/checkpoints/context-lightgbm-b-cache-write/B/*.txt" --exclude=""

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://gitlab-master.nvidia.com/aire/agents/replay-gap.git
GIT_LFS_SKIP_SMUDGE=1 git -C replay-gap checkout \
  f21165163390fa76c0d448daa384a45b7580331c

cd Switchyard
cargo build --locked --release -p switchyard-server
cd examples/mini-swe-forecast
uv sync --frozen
uv run --frozen pytest -q
```

If Switchyard is already checked out, switch to the PR branch and run
`git pull --ff-only` instead of cloning it again. On macOS, LightGBM may need
`brew install libomp` if loading its shared library fails.

**Use `agent-progress-research/v1/checkpoints/context-lightgbm-b-cache-write/B`.**
This is the B bundle used in the live demo, with the added cache-write head.
The tuned B/C bundles are separate experiments. C needs live requirement
annotations and is not supported by this launcher.

Obtain the original `django__django-14631/base.traj.json` from the trace owner.
It is not bundled in the PR. It supplies the task's initial messages, model
settings and Docker image. Future recorded responses are not supplied to the
new agent. Keep the trace and new run outputs outside the checkouts.

## 2. Start the dashboards

All remaining commands start from `Switchyard/examples/mini-swe-forecast`.
Use **Bash** in each terminal: the environment helper uses `BASH_SOURCE`.

```bash
./observability/up.sh
```

This creates `observability/.env` if absent, downloads the pinned Langfuse
compose file, starts Langfuse and Grafana, and installs the Opus price entry
and dashboard. It needs free ports 3000, 3001 and the backing-service ports.
If another local stack owns those ports, use that stack or stop it separately.
Do not start a second stack over an existing demo.

- Langfuse: <http://localhost:3000/project/switchyard>
- Grafana: <http://localhost:3001>
- Langfuse login: `LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD`
  in `observability/.env`. Grafana has anonymous access in this demo config.

These settings are for a local demo. The compose configuration publishes
ports and uses demo credentials; do not expose it as a shared service unchanged.

## 3. Terminal A: start Switchyard

Copy `routes.example.toml` outside the checkout, for example to
`/path/to/demo/routes.toml`. Set the upstream URL to your OpenAI-compatible
endpoint. For the original Opus run, set **both** model IDs to
`aws/anthropic/bedrock-claude-opus-4-8`. Keep `format = "openai_chat"`.

```bash
source ./observability/switchyard_otel_env.sh
export UPSTREAM_API_KEY='<your upstream credential>'

../../target/release/switchyard-server \
  --config /path/to/demo/routes.toml --dry-run
../../target/release/switchyard-server \
  --config /path/to/demo/routes.toml --host 127.0.0.1 --port 4110
```

Leave it running. The helper must be sourced **before** starting Switchyard;
changing exports in another terminal cannot configure an already-running server.

## 4. Terminal B: start cost enrichment

From the same example directory, in Bash:

```bash
source ./observability/switchyard_otel_env.sh
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY="$LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
export LANGFUSE_SECRET_KEY="$LANGFUSE_INIT_PROJECT_SECRET_KEY"

uv run --frozen python observability/forecast_cost_enricher.py
```

Leave this running. It polls for forecast/actual spans and adds the cost
estimates used by the dashboards. It uses the local stack's default
ClickHouse credentials; custom installations can set `CH_URL` and
`CLICKHOUSE_AUTH` as described in the script.

## 5. Terminal C: run one task

From the example directory, in Bash, substitute absolute paths below. Choose
an output directory that does not exist yet:

```bash
source ./observability/switchyard_otel_env.sh
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY="$LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
export LANGFUSE_SECRET_KEY="$LANGFUSE_INIT_PROJECT_SECRET_KEY"

uv run --frozen python run.py \
  --research-root /path/to/agent-progress-research \
  --replay-repo /path/to/replay-gap \
  --reference-trace /path/to/django__django-14631/base.traj.json \
  --forecast-checkpoint /path/to/agent-progress-research/v1/checkpoints/context-lightgbm-b-cache-write/B \
  --base-url http://127.0.0.1:4110/v1 \
  --output /path/to/demo/runs/django-first-run \
  --max-steps 30
```

The agent starts a fresh task container and makes new model requests. It uses
only the local proxy credential; the real upstream credential stays with
Switchyard. The container can pull its task image on first use.

## 6. Check the results

- **Grafana:** leave **Run** on `latest (auto)`. The dashboard refreshes every
  five seconds. Inspect next-call token/cost bands, actual usage, remaining
  calls, task cost estimates and IR phases.
- **Langfuse:** open the `switchyard` project. Each model call has an
  `agent.call` parent with `lightgbm.forecast` and Switchyard request spans.
  The enricher also adds a `predicted:opus-4-8` generation for cost charts.
- **Local output:** `status.json` records progress; `forecasts.jsonl` and
  `actuals.jsonl` contain the matched predictions and usage. `features.jsonl`
  and `live.traj.json` contain task content. `score-export.jsonl` records
  numeric-score submission results.

If charts stay empty, check the agent's output, the enricher terminal, the
project keys and whether both processes started with OTLP exports. Allow time
for ingestion. An agent status of `Submitted` is not a SWE-bench verifier pass.

## Interpret the demo correctly

- Token outputs are **predicted classes with interval bounds**, not confidence
  intervals. The top band is unbounded.
- The dashboard converts finite bands to midpoint cost estimates. For an
  unbounded band it uses the lower edge; that is not a cost ceiling.
- Remaining-task cost is a dashboard approximation:
  `predicted remaining calls × predicted next-call midpoint cost`. It is not
  a separately trained dollar prediction.
- Uncached input includes cache writes. The enricher adds only the cache-write
  premium, avoiding counting those tokens twice.
- The enricher uses fixed Opus rates in `PRICE`; `up.sh` installs the model price
  entry. These are configured demo rates, not a guarantee of your endpoint's
  bill. Update both before using another deployment or model.
- Predicted generations contribute to Langfuse's aggregate cost views. Filter
  by the actual model to read actual cost alone. The original generation's
  billing fields remain unchanged.

To stop, interrupt the agent, enricher and Switchyard in their terminals, then
run `./observability/down.sh`. This preserves stored dashboard data; avoid
`-v` or `clear_runs.sh` unless you intend to delete it.

For field definitions see [README.md](README.md). For dashboard implementation
details see [observability/README.md](observability/README.md).

# Compare LightGBM and Jev forecasts in Pi

`switchyard/compare` runs both predictors before each acting-model call. One
fixed completion model executes the task, so both forecasts observe the same
pre-call conversation. The example configuration uses Opus 4.8 through
OpenRouter (`openrouter/anthropic/claude-opus-4.8`). The display has two rows
below the editor:

```text
LightGBM · rem ~$0.0200 · total ~$0.0500 · ETA ~20s · next ~8.0k tok (avg err +4.0%) | remaining ~30.0k tok (avg err −8.0%)
Jev · rem ~$0.0400 · total ~$0.0700 · ETA ~40s · remaining ~60.0k tok (avg err +12.0%)
```

These are illustrative values. `rem` and ETA describe remaining work; `total`
adds observed acting-model spending for the current task/turn. The two
rows replace the Switchyard footer. `medium`, `high`, and `max` are passed through to the
completion model, which must support the selected level.

## Setup

First follow the Node/Rust build instructions in [README.md](README.md). Pi
1.0.4 is the tested version. Provider credentials are owned by Pi; for the
example configuration, make `OPENROUTER_API_KEY` available in each terminal.
Jev requests are additional API calls and may incur fees.

The LightGBM worker needs three external inputs, which are not bundled:

- An `agent-progress-research` checkout with its `v1` scripts and grouped-IR
  checkpoints. The tested revision is `d87391d39`. Access to the research
  repository is required. Download the actual Git LFS checkpoint files rather
  than pointer files.
- [TokenCast](https://github.com/DEFENSE-SEU/TokenCast) checked out at
  `d2e2490cb794dc57dd5bb03f732bb33fd2ec5d19`.
- The research project's Python 3.12 environment, installed with
  `uv sync --project /path/to/agent-progress-research/v1 --frozen`.

The worker uses the first published checkpoint,
`v1/checkpoints/grouped-ir/seed-20260719-window-8/pipeline.pkl`, and verifies its
manifest SHA256 before loading. It reuses the fitted research pipeline; there
is no retraining or checkpoint selection in this example. External research
and TokenCast code remain in their own checkouts under their respective licenses.

Copy and edit the configuration with **absolute paths**:

```bash
mkdir -p "$HOME/.config/switchyard-pi-forecast"
cp examples/pi/switchyard-forecast.example.json \
  "$HOME/.config/switchyard-pi-forecast/switchyard-forecast.json"
```

The first slash in each model reference separates the provider from its model
ID. `decision_model` must be registered as a classifier in Pi. The Python
worker paths can alternatively be set with `SY_FORECAST_PYTHON`,
`SY_FORECAST_RESEARCH_ROOT`, and `SY_FORECAST_TOKENCAST_ROOT`.

Run from a task directory, using an absolute path to this checkout's launcher:

```bash
/path/to/Switchyard/examples/pi/run-forecast.sh --thinking max
```

In another task directory:

```bash
/path/to/Switchyard/examples/pi/run-forecast.sh --thinking medium
```

For a third comparison, run from another task directory:

```bash
/path/to/Switchyard/examples/pi/run-forecast.sh --thinking high
```

Use separate directories and the same prompt to compare reasoning settings
without editing the same task files. Each level has its own Pi settings and
sessions under `~/.config/switchyard-pi-forecast/{max,high,medium}`. Add `--continue`
from the same task directory to resume. `SY_FORECAST_AGENT_ROOT` overrides the
settings root; `SY_FORECAST_CONFIG` selects a shared configuration file. When
set, `XDG_CONFIG_HOME` replaces `~/.config` for the default locations.

For direct Pi usage, put `switchyard-forecast.json` in the selected Pi agent
directory, or set `SY_FORECAST_CONFIG`, then load `switchyard-compare.ts` with
`--model switchyard/compare --thinking medium`.

## Shared prediction target

Both remaining forecasts count **all provider input plus output**:

```text
input + cacheRead + cacheWrite + output
```

Input counts again on every call, including repeated cached context. Reasoning
is already part of output and is counted once. Predictor API usage is excluded
from these acting-model token targets and recorded separately.

LightGBM forecasts next-call and remaining-task tokens from its existing
grouped IR and causal features. The worker replays only the observed branch;
historical tools are parsed, never executed. Pi's standard tools have explicit
adapters. Unsupported tools make the forecast unavailable without blocking the
completion model. Requirement outputs remain null.

Jev sees Greg's bounded task-history representation, full per-call and
cumulative token usage, completion model, and selected reasoning level. It
scores the original 80 token-budget bands through 1.5M tokens plus `impossible`.
Its numeric total estimate is the probability-weighted mean of finite band
**midpoints**, conditional on finite-budget probability. This comparison uses
that point estimate rather than the original display's upper-bound mean.
Remaining is `max(0, estimated total - tokens already used)`. Jev has no
next-call forecast.

Both predictors run concurrently before each normal user, continuation, or
retry call. Direct helper calls skip forecasting. Cancellation propagates to
Pi. Failed forecasts are recorded independently and allow completion to proceed.
The displayed remaining values consume observed usage until the next forecast.
When Pi finishes a turn normally, both rows show remaining `0 tok`, `$0.0000`,
and `ETA 0s`; LightGBM shows `next —`. The original forecasts remain unchanged
for error scoring. This completed display restores on resume and resets at
the next forecast. Aborted or failed turns retain their forecast estimates.

## Cost, ETA, and errors

For a task/turn, let `C[t]` be the remaining-cost prediction before call `t`,
`c[t]` its observed cost, and `S[t]` the sum of observed acting-model costs so
far. The cost-to-go identity is `C[t] = c[t] + C[t+1]` for a consistent forecast;
the projected total is `S[t] + C[t]`. Remaining forecasts are not summed across
calls. Predictions may revise this total, so their differences are not treated
as observed charges.

Every saved forecast has a typed `cost` frame with USD fields for each
predictor: `remaining_usd`, `next_call_usd` (LightGBM), `total_usd`,
`session_total_usd`, `initial_total_usd`, the first-priced baseline, and forecast
revisions. The frame records observed task and session spend, plus forecasting
API spend and the current forecasting API call's cost separately. See
[`cost-forecast.ts`](cost-forecast.ts) for the API types. Charges are computed
from actual recorded usage and its saved per-call catalog rates; missing prices
or usage remain unknown, not zero. Retries and partially charged failed calls
are included when their usage and prices are available.

Each forecast's dollar value is frozen at its pre-call boundary. Observed
dollars consume that budget until a new prediction arrives; updated token mix
does not retroactively reprice an old forecast. `initial_total_usd` freezes the
first total-cost prediction (`C[0]` at zero task spend). At normal completion,
the settlement's `cost_outcome` reports observed acting-model cost minus that
original prediction, and the widget prints `C0 err` in dollars. Positive means
actual cost exceeded the predicted budget; this sign is the reverse of the
token error convention below. Remaining cost and ETA still become zero.

A completely fresh session initially has no observed token/cache mix, so its
first dollar forecast and `C0` are unknown. A later estimate is saved as
`first_priced_total_usd` and is scored/displayed as `first err`, not relabeled
as `C0`. Existing timed observations can price the first forecast of a new
turn. Baselines persist on resume and reset after each settled turn, since
future user requests are outside the current forecast horizon.

`revision_usd = total[t] - total[t-1]` records each forecast update. The API also
retains `C[t-1] - C[t]` as an implied preceding-call cost and the actual observed
cost separately. Including the final revision to actual spend makes revisions
telescope to `actual - C[0]` when all values are known. Predictor API charges
remain outside the acting-model forecast/error target.

Both rows use the same calibration from successful, timed acting-model calls
in their session. Let `F` be their observed full-token total and `R` the
remaining forecast:

```text
projected channel tokens = R × observed channel tokens / F
ETA = R / F × (observed answer duration + tool wall time)
```

The channels are uncached input, cache reads, cache writes, and output. Integer
allocation preserves the token budget. One-hour cache writes are a subset of
cache writes. Switchyard prices the projected channels using the selected
model's rates, with pricing tiers selected by mean observed prompt size.
Tool timing retains Greg's non-overlap accounting. Answer timing starts after
forecasting, so predictor latency is excluded from completion timing.

These estimates exclude future predictor fees and latency. First-call cost
and ETA are unavailable until there is a usable timed response. They assume
the observed channel mix, prompt sizes, and speed continue.

Next-call error updates after each valid completed call. Remaining errors for
both predictors become available when Pi produces a normal final response and
settles the user turn. Each saved pre-call forecast is scored against actual
full-token usage from its own cutoff to that boundary:

```text
signed percentage error = 100 × (predicted remaining - actual remaining)
                                   / actual remaining
```

`avg err` averages these signed percentages across eligible forecasts in the
session branch. Positive means overprediction; negative means underprediction.
Signs can cancel, so this measures bias rather than MAE or MAPE. Aborted/failed
turns are not scored for remaining usage. Missing usage and unavailable
forecasts are skipped; zero actual usage has no percentage error. A normal Pi
stop defines the measured horizon, not independent task correctness.

Predictions and scores persist as `switchyard.compare.forecast.v2` and
`switchyard.compare.remaining-errors.v2`. Resume deduplicates scores. Earlier
uncached Jev entries are excluded. Actual metrics keep their existing schema.

The checkpoint was trained on Opus mini-SWE-agent traces with a 30-call limit.
Pi and other model/reasoning settings are transfer settings. No Pi call cap is
invented. The checkpoint has no explicit reasoning-level or cache-split head;
passing reasoning metadata does not change its fitted features.

## Validation

The default Node suite uses a local JSON-lines worker fixture and needs no
checkpoint, Python environment, or API calls:

```bash
npm --prefix examples/pi run typecheck
npm --prefix examples/pi test
```

To exercise the real checkpoint and native-tool adapters as well:

```bash
export SY_FORECAST_RESEARCH_ROOT=/path/to/agent-progress-research/v1
export SY_FORECAST_TOKENCAST_ROOT=/path/to/TokenCast
/path/to/agent-progress-research/v1/.venv/bin/python \
  -m unittest discover -s examples/pi/forecast -p 'test_*.py'
```

For another compatible JSON-lines forecaster, `worker_command` may contain an
executable and argument array in the configuration. The worker receives one
JSON request per line and returns `{ "id": ..., "result": ... }` or
`{ "id": ..., "error": ... }`. See the Node fixture for the response contract.

# Pi virtual model using Switchyard composite routing

This extension registers `switchyard/auto`. Pi calls a Decision classifier on
user turns. Switchyard's composite algorithm uses that answer and recent tool
activity to select a completion model. Thinking stays `medium`.

The fixed policy selects the capable tier when the classifier's `advantage`
probability exceeds 0.4. Exactly 0.4 selects efficient. Stage routing then picks
the model on every request, using the classifier's tier as its fallback. Its
efficient-first and capable-first thresholds are 0.5 and 0.2. A tool failure can
escalate to the capable model, with a two-request hold during recovery.

## Configuration

Create `<agent-dir>/switchyard-router.json`. The agent directory defaults to
`~/.pi/agent`; `PI_CODING_AGENT_DIR` overrides it.

```json
{
  "decision_model": "openrouter/inception/mercury-decide:free",
  "capable_model": "openai-codex/gpt-6.1-sol",
  "efficient_model": "openai-codex/gpt-6-luna"
}
```

Use `provider/model` references from Pi's catalog. The first slash separates the
provider from the model ID, so `openrouter/vendor/model` keeps `vendor/model` as
the ID. The Decision model must be registered as a classifier in Pi.

The policy's evidence describes candidate a (capable) as a larger,
reasoning-oriented model for complex work. Candidate b (efficient) is a smaller,
fast model for simple questions, local reads and edits, routine tests, and
straightforward fixes. Pick completion models that fit these roles.

The extension reads the config on its first routing request and caches it.
Reload the extension after editing the config. Pi owns credentials and all model
calls, including classifier calls. Classifier and completion calls may cost money.

## Local setup

Requires Node.js 22.19 or newer, the repository's Rust toolchain, and Pi with the
virtual-model and classifier APIs. Development checks use Pi 1.0.4.

From the repository root:

```bash
npm --prefix bindings/typescript ci
npm --prefix bindings/typescript run build
npm --prefix examples/pi ci
npm --prefix examples/pi run typecheck
npm --prefix examples/pi test
```

Tests run the real composite algorithm with a local classifier mock, check
message conversion, and load the extension through Pi's public resource loader.

Sign in to the configured providers through Pi's `/login` and confirm the models
are in your catalog. Then:

```bash
pi -e ./examples/pi/switchyard-router.ts --model switchyard/auto
```

## Conversation handling

The extension passes visible text, tool-call arguments, paired tool results, and
failure flags to Switchyard. Pi keeps the original images and reasoning for the
chosen completion model. The classifier sees the opening user task and latest
user message. Its instructions come from libsy and are sent to Pi as a JSON
string.

Pi stores routing state on the session branch. It follows forks and `/tree`, and
survives compaction, resume, and extension reload. Tool continuations reuse the
stored tier and capable hold while stage signals select the model. State is
returned only when it changes.

Retries reuse stored state. Retrying a user-turn request calls the classifier
again. Direct requests, such as compaction summaries, start with fresh state;
Pi discards their returned state. They call the classifier when their last
message contains user text. Image-only user messages retain the previous tier
because routing uses text and tool activity.

Classifier failures fail the request, and cancellation propagates to Pi. A valid
answer without a usable `advantage` probability retains the previous tier.

For the binding API, see
[`bindings/typescript`](../../bindings/typescript/README.md).

## Status metrics

With `switchyard/auto` selected, the status bar shows estimated remaining task
tokens and time, cumulative session metrics, and the latest request's metrics:

```text
SY remaining ~247.9k tok ETA ~12m30s | session 2.1k/420ms/8.30s | last 180/350ms/1.20s
```

`session` and `last` list **tokens / TTFT / request latency**. Session TTFT is
an average; session latency is a sum. Narrow terminals may truncate the line.
Measured tokens include routing, prediction, and completion usage, including cached tokens. Timing
starts when routing begins. TTFT ends at the first text, reasoning, or tool-call
content. Request latency ends when Pi finishes the assistant message, before tool
execution. Missing measurements appear as `—`. Direct helper requests, such as
compaction summaries, are excluded.

The extension records Pi's measurements in the in-process Rust metrics collector
and reads its filtered snapshots for the status bar. No metrics server is needed.
Each observation carries `session_id`, `agent_id` (`root`), `agent_turn_idx`, and
`user_turn_idx`. Indices start at zero. Agent indices advance on each request;
continuations and retries retain the user index.

Completed measurements are saved in Pi's session entries. Resume and reload
restore them without counting them twice. Totals cover all branches in the same
session; a fork with a new session ID starts fresh. The collector retains up to
20,000 label combinations per instrument; further combinations enter an overflow
series and cannot be queried by session or turn.

### Task forecast

Before each agent request, including continuations and retries, the extension
makes an additional classifier call using `decision_model` and Pi's credentials.
It uses the Rust `switchyard-token-prediction` crate's task-completion question:
80 budget ranges through 1.5 million tokens, plus `impossible`. This call adds
latency and may cost money. Its usage is included in the session and request metrics.

The forecast estimates **total uncached input plus output until task completion**,
including agent tokens already used. The displayed remaining estimate uses the
probability-weighted mean of finite range upper bounds, minus agent usage so far,
with a minimum of zero. It updates after each completed response. Reasoning is
counted once as output.
Prediction and routing classifier usage is excluded from the forecast's prior
agent usage. Its token definition differs from measured session totals, which
include cache reads and classifier work. The estimate is conditional on a finite
budget. All-impossible results and errors show `remaining unavailable` and allow
completion routing to proceed.

The predictor reads the current session branch, retaining the initial task and
recent messages. History is limited to 4,000 characters per message and about
60,000 characters overall. Earlier completed-call usage remains included even
after compaction. Forecasts and their usage baseline are saved for reload and
resume; each agent request refreshes them for its current branch. Once Pi settles
the user turn and waits for input, the forecast is cleared, including its saved
display state. Tool continuations keep it visible. Direct helper requests skip
prediction.

### Time estimate

ETA uses successful agent responses from the current session that have known
input/output usage, TTFT, and positive time after the first token. The extension
splits remaining tokens using the **mean per-response uncached input / output mix**.
Each response contributes its output fraction, `output / (uncached input + output)`,
with equal weight. This keeps a single large prompt from dominating the split.
Cache writes count as uncached input; cache reads are excluded. Reasoning is
already part of output.

Input-side time uses mean TTFT per expected remaining response. TTFT includes
routing, prediction, network delay, and input processing. Output speed is total
output tokens divided by total time after the first token. Expected remaining
responses are estimated output tokens divided by mean output per response.
ETA adds that count times mean TTFT to the estimated output generation time.
This treats TTFT as a request delay and counts it once, including fully cached calls.

Tool execution time is measured between Pi's tool start and end events. Overlapping
and nested calls count their combined wall time once. Sequential execution
intervals add together. Failed tools still contribute their observed execution
time. ETA adds expected remaining responses times mean tool wall time per
response. Responses without tools count in this average too. Tool time is paired
with the same responses used for model timing.

The formula is `remaining responses * (mean TTFT + mean tool wall time) +
remaining output tokens / output tokens per second`. Tool intervals are saved in
session entries and restored once on reload/resume. Model request latency in
`session` and `last` continues to end before tool execution.

ETA assumes the session's token mix, speeds, and tool usage continue. User input
time is excluded. Before a usable response is observed, it shows `ETA —`. It
updates with completed responses and tool execution intervals, and clears with
the token forecast when the user turn settles. A new session starts with fresh
calibration.

# Pi virtual model using Switchyard composite routing

This extension registers `switchyard/auto`. Pi calls a decision classifier on
user turns. Switchyard's composite algorithm uses that answer and recent tool
activity to select a completion model. Thinking stays `medium`.

The fixed policy selects a capable-first profile when the classifier's `capable`
probability exceeds 0.6. Exactly 0.6 selects the efficient-first profile. Stage
routing then picks the model on every request from the last three tool results.

- Efficient-first uses a 0.4 threshold. A tool failure can escalate to the
  capable model, with a two-request hold during recovery.
- Capable-first starts on the capable model with a planning prompt. The prompt
  asks the model to plan with read-only tools, then make the first edit. After the
  switch to the efficient model, routing stays there until the next user turn.

The extension adds the planning prompt ahead of Pi's system prompt on each
request routed to the capable model in the capable-first profile.

## Configuration

Create `<agent-dir>/switchyard-router.json`. The agent directory defaults to
`~/.pi/agent`; `PI_CODING_AGENT_DIR` overrides it.

```json
{
  "dllm": {
    "library": "/absolute/path/to/dllm/zig-out/lib/libdllm.so",
    "tokenizer_library": "/absolute/path/to/dllm/zig-out/lib/libdllm_tokenizer.so",
    "weights": "/absolute/path/to/Nemotron-Deciffusion-8B",
    "tokenizer": "/absolute/path/to/Nemotron-Deciffusion-8B"
  },
  "capable_model": "openai-codex/gpt-6.1-sol",
  "efficient_model": "openai-codex/gpt-6-luna"
}
```

The `dllm` block loads the Zig inference engine and Deciffusion checkpoint in
Pi's process when the session starts. Startup waits for the model to load.
Decision calls use the checkpoint's masked-token candidate scores. CUDA calls
run on one native thread, and session shutdown releases the model. Cancelling
a request stops waiting for its decision; submitted GPU work finishes on that
thread before the next decision runs. One engine is supported per process.

Use a trained Deciffusion checkpoint and its matching tokenizer, rather than the
general text diffusion checkpoint. `weights` accepts a BF16 directory or an
existing dllm packed file. `tokenizer` always names a directory containing
`tokenizer.json`. Native libraries and CUDA dependencies must match your host.

Use `provider/model` references from Pi's catalog for completion models. The first slash separates the
provider from the model ID, so `openrouter/vendor/model` keeps `vendor/model` as
the ID.

To use a classifier registered in Pi instead, omit `dllm` and set
`"decision_model": "openrouter/inception/mercury-decide:free"`.

The classifier chooses between starting with the more capable model and starting
with the efficient model. Pick a capable model for complex work and an efficient
model for routine work.

The extension caches the config for the session. Restart Pi after editing it.
Pi owns completion credentials and calls. The embedded classifier runs locally.
Provider classifier and completion calls may cost money.

## Local setup

Requires Node.js 22.19 or newer, the repository's Rust toolchain, and Pi with the
virtual-model and classifier APIs. Development checks use Pi 1.0.4.

For embedded decisions, first build dllm's native engine and GPU libraries using
its build instructions. The full dllm build also installs `libdllm_tokenizer.so`.
To build only the CPU tokenizer library, run this from the dllm checkout with
its supported Zig version:

```bash
zig build tokenizer -Doptimize=ReleaseSafe
```

Switchyard loads both libraries directly. Python is not used at runtime.

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
To include local GPU inference, put the four `dllm` path fields in a JSON file and
set `SWITCHYARD_DLLM_OPTIONS=/path/to/options.json` when running tests. This also
enables the native engine lifecycle tests in `bindings/typescript`.

Sign in to the configured providers through Pi's `/login` and confirm the models
are in your catalog. Then:

```bash
pi -e ./examples/pi/switchyard-router.ts --model switchyard/auto
```

## Conversation handling

The extension passes visible text, tool-call arguments, paired tool results, and
failure flags to Switchyard. Pi keeps the original images and reasoning for the
chosen completion model. The classifier sees the opening user task and latest
user message. Its instructions come from libsy. The
classifier state is `{ "input": [...] }`, holding the routing messages.

Pi stores routing state on the session branch. It follows forks and `/tree`, and
survives compaction, resume, and extension reload. Tool continuations reuse the
stored tier and capable hold while stage signals select the model. State is
returned only when it changes. Sessions saved before stage profiles keep their
tier and drop any stage hold.

Retries reuse stored state. Retrying a user-turn request calls the classifier
again. Direct requests, such as compaction summaries, start with fresh state;
Pi discards their returned state. They call the classifier when their last
message contains user text. Image-only user messages retain the previous tier
because routing uses text and tool activity.

Classifier failures fail the request, and cancellation propagates to Pi. An
answer without a usable `capable` probability also fails the request.

For the binding API, see
[`bindings/typescript`](../../bindings/typescript/README.md).

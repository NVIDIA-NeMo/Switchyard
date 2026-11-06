# Pi virtual model using Switchyard composite routing

This extension registers `switchyard/auto`. Pi calls a Decision classifier on
user turns. Switchyard's composite algorithm uses that answer and recent tool
activity to select a completion model. Thinking stays `medium`.

The fixed policy selects the capable tier when the classifier's `capable`
probability exceeds 0.6. Exactly 0.6 selects efficient. Stage routing then picks
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

The classifier chooses between starting with the more capable model and starting
with the efficient model. Pick a capable model for complex work and an efficient
model for routine work.

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
user message. Its instructions come from libsy and are sent to Pi as text. The
classifier state is `{ "input": [...] }`, holding the routing messages.

Pi stores routing state on the session branch. It follows forks and `/tree`, and
survives compaction, resume, and extension reload. Tool continuations reuse the
stored tier and capable hold while stage signals select the model. State is
returned only when it changes.

Retries reuse stored state. Retrying a user-turn request calls the classifier
again. Direct requests, such as compaction summaries, start with fresh state;
Pi discards their returned state. They call the classifier when their last
message contains user text. Image-only user messages retain the previous tier
because routing uses text and tool activity.

Classifier failures fail the request, and cancellation propagates to Pi. An
answer without a usable `capable` probability also fails the request.

For the binding API, see
[`bindings/typescript`](../../bindings/typescript/README.md).

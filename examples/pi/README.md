# Pi virtual model using Switchyard composite routing

This extension registers `switchyard/auto`. In unrestricted mode, Pi calls a Decision classifier on
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
  "efficient_model": "openai-codex/gpt-6-luna",
  "local_model": "local/coder"
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

## Local only demo

`local_model` is optional. It names a physical model in Pi's catalog, with its
endpoint and credentials configured through Pi as usual. The operator must bind
that model to an approved local endpoint; this extension does not verify the
endpoint or classify request content.

Start with `--local-only` to use that model for the whole session:

```bash
pi -e ./examples/pi/switchyard-router.ts --model switchyard/auto --local-only
```

Or enter `/local` while Pi is idle. The status bar shows `local only`. The
restriction is saved as session state, not prompt text, and survives reload,
resume, compaction and tree navigation. Start a new blank session without
`--local-only` to return to cloud routing. The command cannot remove the
restriction from a conversation that may already contain private data. A
restriction cannot recall a request that was already dispatched.

Local-only requests select `local_model` before any capability-classifier lookup
or call. Pi sends the original conversation to that model; local mode does not
redact it. It also covers tool continuations, retries and direct requests such
as compaction summaries. Thinking is clamped to the local model's supported
levels. A missing or unavailable local model fails the request; there is no
cloud fallback.

This is a host-side routing demo in the maintained extension. It does not use
Runner or TOML eligibility enforcement. It requires no detector, customer
adapter script or policy service. It governs only requests using
`switchyard/auto`; manually choosing another provider or making a tool's own
network calls bypasses this path. The unrestricted composite router is
unchanged.

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

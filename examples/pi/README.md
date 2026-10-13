# Pi virtual model using Switchyard auto

This example registers `switchyard/auto` and asks Switchyard to select a model on
every request:

- **Efficient:** Luna (`openai-codex/gpt-6-luna`).
- **Capable:** Sol (`openai-codex/gpt-6.1-sol`).
- **Thinking:** always `medium`.

`auto` uses composite routing. A decision model classifies user turns, and
local tool signals handle escalation and recovery between them.

## Configuration

`switchyard.toml` contains the algorithm, classifier and decision-client settings.
The example uses OpenRouter's Mercury Decide endpoint. Set `OPENROUTER_API_KEY`
in the environment before launching Pi and replace the example comparison
evidence with evaluations for your tasks and models.

Set either target to a `provider/model` reference available in Pi's catalog, such
as `openai-codex/gpt-6-astra`. The first slash separates the provider from the model
ID. `openrouter/vendor/model` keeps `vendor/model` as the ID. Both providers and
models come from this file. Thinking remains `medium`.

The extension generates the runner's target entries, route ID, and credential-free
placeholder client. Pi uses the selected model's catalog limits. Set
`SWITCHYARD_CONFIG` to use another algorithm-settings file, and reload the
extension after changing its configuration.

## Local setup

Requires Node.js 22.19 or newer, the repository's Rust toolchain, and Pi with the
virtual-model API. Development checks use Pi 1.0.4.

From the repository root:

```bash
npm --prefix bindings/typescript ci
npm --prefix bindings/typescript run build
npm --prefix examples/pi ci
npm --prefix examples/pi run typecheck
npm --prefix examples/pi test
```

The tests exercise the actual auto algorithm locally, check message conversion,
and load the extension through Pi's public resource loader.

Sign in to the configured providers through Pi's `/login` and confirm the models
are in your catalog. Then:

```bash
pi -e ./examples/pi/switchyard-router.ts --model switchyard/auto
```

The classifier calls its configured endpoint on user turns. Pi owns completion
credentials and connections. Provider calls can incur charges.

## Conversation handling

The extension passes visible text, tool-call arguments, paired tool results, and
failure flags to Switchyard. Pi keeps the original images and reasoning for the
chosen completion model. Each decision uses the current branch's transcript,
including the history available after compaction or resume.

The classifier retains its tier by hashing the first user message. Conversations
or branches with identical opening text can share that tier. Recovery signals
come from recent tool history. Cross-request capable holds require session
identity and are not retained by this extension.

Routing errors and cancellation propagate to Pi. The extension checks target
identities and resolves the selected model through Pi's catalog.

For the binding's API and configuration restrictions, see
[`bindings/typescript`](../../bindings/typescript/README.md).

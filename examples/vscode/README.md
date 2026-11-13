# Switchyard for Copilot Chat in VS Code

This extension adds Switchyard models to the Copilot Chat model picker. Each
one pairs two models: a capable model for hard turns and an efficient model for
routine turns. Switchyard picks one of them for every request.

Routing uses Switchyard's [Auto](../../docs/routing_algorithms/overview.md#auto)
settings. A chat starts on the efficient model. Repeated or severe tool failures
can move it to the capable model, which then serves the next two requests in
that chat.

## Prerequisites

- VS Code 1.145 or newer, with GitHub Copilot Chat signed in.
- Node.js 22.19 or newer.
- The Rust toolchain. [rustup](https://rustup.rs) installs the version this
  repository uses.
- macOS or Linux. Windows is untested.

## Install

From the repository root:

```bash
npm --prefix bindings/typescript ci && npm --prefix bindings/typescript run build
npm --prefix examples/vscode ci && npm --prefix examples/vscode run package
code --install-extension examples/vscode/switchyard.vsix
```

The first command compiles Switchyard, so it takes a few minutes. The package
works only on the operating system and CPU type that built it.

## Use

1. Run **Switchyard: Create Routing Profile** from the Command Palette.
2. Pick a capable model, then an efficient model, then press Enter to accept the name.
3. Open Copilot Chat and pick the profile in the model picker.

The first request to each Copilot model asks you to allow access. Requests use
your Copilot quota.

Other commands:

- **Switchyard: Edit Routing Profile** changes the models or name, or deletes a
  profile. It also turns the Enterprise workaround on or off (see the note below).
- **Switchyard: View Routing Profiles** lists your profiles.
- **Switchyard: Add Custom Model** adds your own endpoint. It supports OpenAI
  Chat Completions, OpenAI Responses and Anthropic Messages. Custom models can
  be used in profiles next to Copilot models. Their replies do not stream and
  are limited to 8192 tokens.
- **Switchyard: Remove Custom Model** removes an endpoint and its API key.
- **Switchyard: Show Routing Log** shows which model served each request.

## Test

```bash
npm --prefix examples/vscode test
```

The tests run the router with the real Switchyard binding.

## Settings

Profiles are stored in your user `settings.json`. The create command fills in
the model IDs.

```json
"switchyard.profiles": [
  {
    "id": "a1b2c3d4",
    "name": "Switchyard: Claude Opus 4.5 / GPT-5 mini",
    "capable": "claude-opus-4.5",
    "efficient": "gpt-5-mini",
    "threshold": 0.5
  }
]
```

`threshold` is optional and defaults to 0.5. Raise it to use the efficient
model more often. Lower it to use the capable model more often.

> [!NOTE]
> **Profiles missing from the model picker?** Copilot Business and Enterprise
> can hide models from extensions. Ask your admin to allow them, or turn on the
> Enterprise workaround:
>
> 1. Run **Switchyard: Edit Routing Profile**.
> 2. Select **Enterprise workaround** at the bottom of the list.
>
> It works within a few seconds. Select it again to turn it off, then reload
> the window. You can also set `"switchyard.enterpriseWorkaround": true` in
> `settings.json`.
>
> The workaround overrides your organization's Copilot policy. This is known
> and is why it is off by default. It relies on Copilot internals, so it is
> unsupported and may stop working.

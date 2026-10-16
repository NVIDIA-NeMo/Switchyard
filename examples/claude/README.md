# Switchyard plugin for Claude Code

Install the plugin, reload Claude's plugins, and routing starts automatically.
Every new Claude session starts with routing on.
Switchyard's local `auto` stage router selects Sonnet 5.5 or Opus 5.5 before each
main-conversation request, including tool continuations. Claude Code keeps the
conversation, tools, reasoning effort, authentication, and completion connection.

The package contains the mod, a native Switchyard executable, and default
configuration. Prebuilt packages need no Rust, Node.js, routing API key, or running server.

## Install

Initial support: **Linux x86_64 with glibc 2.35 or newer**, Claude Code CLI
**2.1.301 or newer**, mods allowed by your organization, and a Claude login with
access to both default models. Desktop, VS Code, Windows, and macOS are not yet
validated. Locally built packages inherit the build host's glibc requirement.
Organization policy can prevent mods from loading at all; check
Claude's plugin errors if no Switchyard status appears.

Build the package from the repository root with Python 3 and the repository's
Rust toolchain:

```bash
python3 examples/claude/package.py
```

This creates `dist/switchyard-claude-linux-x86_64` and a `.tar.gz` archive for
sharing. Recipients can extract the archive without building it. Keep the built
or extracted marketplace directory: Claude uses a local marketplace in place.

In Claude Code, run these once, using the extracted directory's absolute path:

```text
/plugin marketplace add /absolute/path/switchyard-claude-linux-x86_64
/plugin install switchyard@switchyard-local
/reload-plugins
```

A new Claude session also loads the installed plugin. A hosted marketplace and
public release assets are a separate publication step; installing this source
directory alone does not include the executable.

## Controls and status

- `Switchyard Auto · waiting for a request`: routing is enabled.
- `Switchyard Auto · claude-sonnet-5-5` (or Opus): the latest selected model.
- `Switchyard unavailable · keeping …`: routing failed; Claude keeps its model.
- `/switchyard` shows details. `/switchyard off` and `/switchyard on` control the
  current session. Disable the plugin to turn it off across sessions.

One plugin-owned helper keeps one Switchyard Runner alive. Routing state such
as the capable-model hold stays in libsy's existing session storage.
`/switchyard off` pauses routing; `/switchyard on` reuses that Runner and its state.
Manual model overrides also pause routing without ending the helper.
`/clear` starts a new session with routing on. Resuming a conversation in the same
plugin runtime reuses its session ID and retained routing state, subject to
libsy's existing session expiry. Each session's on/off preference is saved
separately in Claude's plugin store.

Replacing the plugin on reload, Claude exit, or a helper crash ends the Runner. Routing state is
held in memory, with no checkpoints or restoration after restart. A replacement
helper reports that it is starting with fresh routing state. Saved on/off
preferences survive reload and restart.
Reloading an unchanged plugin leaves its Runner alive.
A decision that takes more than three seconds, or a helper that fails, leaves
Claude's model in place and ends that helper. The next request starts a new one
with fresh routing state.

Changing the session model, including through the model picker or a native
fallback, turns routing off until `/switchyard on`. An explicit `/model <name>`
command also turns routing off. Opening and dismissing the picker leaves it on.
Completion errors remain Claude's to handle.
Routing advances when the Runner makes a decision. Interrupting the subsequent
Claude completion does not roll back that decision. Late answers from before
a toggle, model change, or session switch cannot override the current selection.

`auto` starts with Sonnet and reads tool activity and failure signals. It can
escalate to Opus for recovery and return to Sonnet as work proceeds. This is the
existing stage router, not a separate model-based task classifier. Its recent
tool-error window is unchanged; plain chat does not necessarily age out a tool
failure. Switching
models can incur cache costs; this example makes no savings or quality claim.

Subagents retain Claude's native model selection. The mod's transcript API reads
the main conversation and cannot classify a subagent's private history. Routing
uses visible text, tool arguments, results, and error flags; Claude retains the
original images and thinking. Each decision reads the current transcript after
resume or compaction.

For optional customization, set `SWITCHYARD_CONFIG` to another complete runner
TOML file before starting Claude. Keep route ID `switchyard` and use full Claude
model IDs, not aliases such as `sonnet`. Only model-selection algorithms accepted
by `Runner::from_toml_for_model_selection` are supported. The bundled `auto`
configuration runs entirely locally; its placeholder client is never contacted.

To try the built plugin for one session:

```bash
claude --plugin-dir ./dist/switchyard-claude-linux-x86_64/switchyard
```

Live requests use your Claude account. Inspect Claude's reported model usage to
confirm routing; plugin status alone does not prove a completion used that model.

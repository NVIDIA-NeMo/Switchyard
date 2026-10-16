# Switchyard plugin for Claude Code

Install the plugin, reload Claude's plugins, and routing starts automatically.
Switchyard's local `auto` stage router selects Sonnet 5.5 or Opus 5.5 before each
main-conversation request, including tool continuations. Claude Code keeps the
conversation, tools, reasoning effort, authentication, and completion connection.

The package contains the mod, a native Switchyard executable, and default
configuration. Users need no Rust, Node.js, routing API key, or running server.

## Install a prebuilt package

Initial support: **Linux x86_64 with glibc 2.35 or newer**, Claude Code CLI
**2.1.301 or newer**, mods allowed by your organization, and a Claude login with
access to both default models. Desktop, VS Code, Windows, and macOS are not yet
validated. Locally built packages inherit the build host's glibc requirement.
Organization policy can prevent mods from loading at all; check
Claude's plugin errors if no Switchyard status appears.

Download the `switchyard-claude-linux-x86_64` artifact from this PR's **Claude
plugin** CI job. Unzip the artifact, then extract its `.tar.gz` file. The tar
archive preserves the native executable's permissions. Keep the extracted
marketplace directory: Claude uses a local marketplace in place.

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
  current session. Off pauses routing and preserves its checkpoint; on resumes it.
- `/switchyard reset` clears this session's checkpoint and enables routing again.
  It does not delete Claude's conversation. Disable the plugin to turn it off
  across sessions.

Changing the session model, including through the model picker or a native
fallback, turns routing off until `/switchyard on`. An explicit `/model <name>`
command also turns routing off. Opening and dismissing the picker leaves it on.
Completion errors remain Claude's to handle.

`auto` starts with Sonnet and reads tool activity and failure signals. It can
escalate to Opus for recovery and return to Sonnet as work proceeds. This is the
existing stage router, not a separate model-based task classifier. Switching
models can incur cache costs; this example makes no savings or quality claim.

Subagents retain Claude's native model selection. The mod's transcript API reads
the main conversation and cannot classify a subagent's private history. Routing
uses visible text, tool arguments, results, and error flags; Claude retains the
original images and thinking.

## Session lifecycle

The plugin stores a versioned libsy checkpoint under Claude's session ID. Each
short-lived native helper restores it, runs the same stage router used by the
server, and returns a model plus the updated checkpoint. Claude still sends the
completion itself. There is no proxy, listener, or background routing process.

A successful model step commits the checkpoint. A failed or abandoned stream,
or a decision superseded by off/on, reset, or a session transition, cannot commit
it. Replaying the same decision ID and input reuses the existing decision without
consuming another recovery step. Main steps are serialized within the mod.

- **New session / clear:** a new session ID starts enabled with fresh state.
- **Resume / plugin reload / process restart:** the same session ID restores its
  checkpoint and on/off setting from Claude's plugin store.
- **Compaction:** tool signals are recomputed from the current transcript while
  the router's persistent counters survive, as they do in a running server.
- **Configuration change:** the helper detects a changed TOML fingerprint and
  starts fresh routing state. An incompatible checkpoint fails visibly; use reset.

The store retains up to 128 recent session records (pruned at plugin startup).
Checkpoints contain routing counters and a fingerprint of the last request, not
conversation text or credentials. One active Claude process per session ID is
supported; concurrent processes resuming the same ID are not coordinated.

The stage policy is unchanged. Its recent window counts tool results, so plain
chat after an error can keep choosing Opus until newer tool results replace the
failure. This policy behavior also occurs with the server.

For optional customization, set `SWITCHYARD_CONFIG` to another complete runner
TOML file before starting Claude. Keep route ID `switchyard` and use full Claude
model IDs, not aliases such as `sonnet`. Only model-selection algorithms accepted
by `Runner::from_toml_for_model_selection` and implementing session checkpoints
are supported: auto, passthrough, and per-request capability classification.
Composite and retaining classifiers need their own complete checkpoint support
before this bridge can use them. The bundled `auto`
configuration runs entirely locally; its placeholder client is never contacted.

## Development

From the repository root, with Python 3 and the repository's Rust toolchain:

```bash
python3 examples/claude/package.py
claude plugin validate dist/switchyard-claude-linux-x86_64/switchyard
claude plugin test examples/claude
python3 examples/claude/test_package.py dist/switchyard-claude-linux-x86_64/switchyard
```

The helper source is
[`crates/switchyard-runner/examples/switchyard-claude.rs`](../../crates/switchyard-runner/examples/switchyard-claude.rs).
It invokes the existing runner once over stdin/stdout. A three-second deadline
bounds routing; failed decisions leave the original model request intact.
The helper tests compare decisions and checkpoints against a persistent Runner
through three escalation/recovery cycles and a replaced transcript. Mod tests
cover state restoration, clear/resume, off/on, cancellation, stale decisions,
subagent isolation, storage failure, and stream forwarding without replay.

To try the built plugin for one session:

```bash
claude --plugin-dir ./dist/switchyard-claude-linux-x86_64/switchyard
```

Live requests use your Claude account. Inspect Claude's reported model usage to
confirm routing; plugin status alone does not prove a completion used that model.

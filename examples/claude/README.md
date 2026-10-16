# Switchyard plugin for Claude Code

Install the plugin, reload Claude's plugins, and routing starts automatically.
Every new Claude session starts with routing on.
Switchyard's `composite` router uses Sonnet 5.5 at low effort to classify each new
user turn, then selects Sonnet 5.5 or Opus 5.5 before each main-conversation
request. The stage router handles tool continuations without another classifier call. Claude Code keeps the
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

- `Switchyard · waiting for a request`: routing is enabled.
- `Switchyard Composite · claude-sonnet-5-5` (or Opus): the latest selected model.
- `classifier unavailable; stage routing`: the judge failed; stage routing continues
  using the last retained tier, or Sonnet if the session has no verdict.
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
Helper startup has a three-second deadline. Classifier calls have a ten-second
deadline and use Claude's native client and existing login. The classifier sees
the opening task and six recent conversation messages and returns at most 64 tokens. A
failed or invalid verdict falls back to stage routing without restarting the
Runner. Retries within one user turn reuse its classifier result, including failure.
An unresponsive helper or a routing decision exceeding fifteen seconds leaves
Claude's model in place and ends that helper; the next request starts a fresh Runner.

Changing the session model, including through the model picker or a native
fallback, turns routing off until `/switchyard on`. An explicit `/model <name>`
command also turns routing off. Opening and dismissing the picker leaves it on.
Completion errors remain Claude's to handle.
Routing advances when the Runner makes a decision. Interrupting the subsequent
Claude completion does not roll back that decision. Late answers from before
a toggle, model change, or session switch cannot override the current selection.

The classifier sets the stage router's default tier. Tool activity can still
escalate to Opus for recovery and return to Sonnet within the same user turn.
Turning routing off or cancelling a turn cancels a pending classifier call and
keeps the Runner alive. The existing stage router's recent tool-error window is
unchanged; plain chat does not necessarily age out a tool failure.
Classifier calls add latency and account usage at user-turn boundaries, and
switching models can incur cache costs. This example makes no savings or quality claim.

The custom prompt and JSON schema live in `switchyard.toml`. The prompt uses
generic tier descriptions without model or harness names. The judge returns
`{"decision":{"target":"efficient"}}` or `{"decision":{"target":"capable"}}`.
The existing `target_selector` policy reads `/decision/target` and sets the
stage router's default tier. Invalid verdicts use the fallback described above.

Subagents retain Claude's native model selection. The mod's transcript API reads
the main conversation and cannot classify a subagent's private history. Routing
uses visible text, tool arguments, results, and error flags; Claude retains the
original images and thinking. Each decision reads the current transcript after
resume or compaction.

For optional customization, set `SWITCHYARD_CONFIG` to another complete runner
TOML file before starting Claude. Keep route ID `switchyard` and use full Claude
model IDs, not aliases such as `sonnet`. Only model-selection algorithms accepted
by `Runner::from_toml_for_model_selection` are supported. The helper uses
`Route::decide_with_client` to send routing-time LLM calls back to the mod. Claude
owns their transport, so configured HTTP endpoints, credentials, target overrides,
and separate decision clients are not used by this example. The placeholder
client endpoint is never contacted. Keep classifier settings in the route table.

For the previous local-only behavior, replace the route tables with:

```toml
[routes.claude]
id = "switchyard"
type = "auto"
efficient_target = "efficient"
capable_target = "capable"
```

Keep the client and target tables above it. Auto makes no classifier calls.

To try the built plugin for one session:

```bash
claude --plugin-dir ./dist/switchyard-claude-linux-x86_64/switchyard
```

Live requests use your Claude account. Inspect Claude's reported model usage to
confirm routing; plugin status alone does not prove a completion used that model.

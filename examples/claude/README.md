# Switchyard plugin for Claude Code

Automatically routes between Sonnet and Opus using your Claude login.

Requires Linux x86_64 (glibc 2.35+), Claude Code CLI 2.1.301+ with mods enabled,
and access to both models.

## Install

Build from the repository root with Python 3 and Rust:

```bash
python3 examples/claude/package.py
```

Then run in Claude Code, using the generated directory's absolute path:

```text
/plugin marketplace add /absolute/path/to/dist/switchyard-claude-linux-x86_64
/plugin install switchyard@switchyard-local
/reload-plugins
```

Keep the generated directory in place. Routing is enabled automatically after reload.

## Use

- `/switchyard`: show status.
- `/switchyard off`: pause routing.
- `/switchyard on`: enable routing for the next request.

The status line shows the selected model or a routing error.
Manual model selection pauses routing. Subagents keep Claude's model selection.
Routing state survives off/on and resets when Claude exits or the plugin restarts.

Settings: [switchyard.toml](switchyard.toml).

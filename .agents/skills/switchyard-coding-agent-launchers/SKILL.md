---
name: switchyard-coding-agent-launchers
description: Modify or debug Switchyard's Claude Code, Codex CLI, or OpenClaw launchers. Use for changes under switchyard/cli/launchers, launch_command.py, launcher configuration, temporary agent workspaces, model catalogs, or launcher smoke tests.
---

# Coding-Agent Launchers

Launchers start a local Switchyard server and configure an external agent process to use it. Read
the current launcher, its dispatch path, and its tests before editing; routing construction changes
often, while the process-specific contracts below are stable.

## Process Contracts

- Claude Code is configured through Anthropic environment variables.
- Codex receives a temporary provider and model catalog through CLI configuration.
- OpenClaw receives a temporary state directory and `openclaw.json`.
- Temporary files, environment changes, and child processes must be cleaned up on success, error,
  and interruption.
- Secrets may be passed to child processes but must not be logged, persisted in committed files, or
  rendered in status output.

## Workflow

1. Trace the command from `switchyard/cli/switchyard_cli.py` through
   `switchyard/cli/launch_command.py` into the selected launcher.
2. Read the launcher-specific tests and any shared server/configuration helpers it calls.
3. Keep launcher code focused on external-process configuration. Routing policies and LLM behavior
   belong in the shared server or library path, not launcher-only branches.
4. Preserve optional-dependency boundaries with lazy imports.
5. Run the focused launcher tests discovered for the changed command, then the CLI help smoke.

Find the current focused tests rather than maintaining a static test inventory:

```bash
rg --files tests | rg 'launch|openclaw|codex|claude|user_config|model_discovery|verify'
```

For OpenAI-compatible providers such as OpenRouter or NVIDIA endpoints, prefer existing base URL,
API key, and model configuration. Add provider-specific launcher behavior only for a real process,
authentication, or protocol difference.

## Anti-Patterns

- Adding one launcher flag per routing algorithm.
- Building a separate routing stack inside a launcher.
- Importing server or provider dependencies at module import time.
- Leaving temporary catalogs, config files, or modified environment variables behind.
- Assuming the three external agents accept the same configuration mechanism.

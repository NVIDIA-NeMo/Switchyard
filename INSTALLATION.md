# Installation Guide

Switchyard supports modular installation based on your use case. Install only the dependencies you need.

## System Requirements

- Linux x86_64 wheels require an x86-64-v3 / AVX2-class CPU (post 2013).
- Linux aarch64 wheels require a Neoverse N1-class CPU (post 2020).

## Core Installation (Library Only)

For applications that use Switchyard as a Python library for routing and recipe composition:

```bash
pip install nemo-switchyard
```

**Includes:**
- Core routing logic
- All recipe factories (Passthrough, RandomRouting, etc.)
- Format translation engine (Anthropic ↔ OpenAI)
- Request/response processors

**Does NOT include:**
- FastAPI / Uvicorn (server)
- prompt-toolkit (CLI launcher)

**Use case:** Library users, middleware plugins, embedded integrations.

## Optional Extras

### `[server]` — Run as a Proxy Server

Add FastAPI and Uvicorn to run Switchyard as a standalone HTTP proxy:

```bash
pip install nemo-switchyard[server]
```

**Adds:**
- FastAPI
- Uvicorn with standard extras
- sse-starlette (for SSE streaming)

**Use case:** Deploying Switchyard as a service, e2e proxy operations.

### `[cli]` — Claude Code Launcher

Add terminal UI support for the `switchyard launch claude` command:

```bash
pip install nemo-switchyard[cli]
```

**Adds:**
- prompt-toolkit (terminal UI framework for the interactive launch TUI)

**Use case:** Running Claude Code with Switchyard routing.

## Combined Extras

### Full Installation

All optional dependencies for complete feature set:

```bash
pip install nemo-switchyard[all]
```

Equivalent to: `switchyard[server,cli]`

### Common Combinations

**Middleware plugin (no server/CLI):**
```bash
pip install nemo-switchyard                  # Core only
```

**Proxy server without launcher:**
```bash
pip install nemo-switchyard[server]
```

**Full Claude Code integration:**
```bash
pip install nemo-switchyard[cli]             # Includes core
```

**Production deployment with all features:**
```bash
pip install nemo-switchyard[all]
```

## Dependency Structure

### Core (Always Installed)
```
- openai>=2.34.0,<3.0
- anthropic>=0.99.0,<1.0
- httpx>=0.28.1,<1.0
- pydantic>=2.13.3,<3.0
```

### Optional Dependencies
| Extra | Size | Purpose |
|-------|------|---------|
| `[server]` | ~50 MB | HTTP proxy (FastAPI + Uvicorn) |
| `[cli]` | ~5 MB | Terminal UI (prompt-toolkit) |

## Embedding in Your Own Application

### For Custom Applications

Embed Switchyard with minimal overhead:

```python
from switchyard import SwitchyardRecipes

# Core library only — no server/CLI dependencies
switchyard = SwitchyardRecipes.passthrough_recipe(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```

## Troubleshooting

### Import Error: "No module named 'fastapi'"

You're trying to run the HTTP server without the `[server]` extra:

```bash
# Install with server support
pip install nemo-switchyard[server]
```

```python
from switchyard import Switchyard                                       # OK (core)
from switchyard.server.switchyard_app import build_switchyard_app   # needs [server]
```

### Import Error: "No module named 'prompt_toolkit'"

You're trying to run the CLI launcher without the `[cli]` extra:

```bash
# Install CLI support
pip install nemo-switchyard[cli]
```

```python
from switchyard import Switchyard                                  # OK (core)
from switchyard.cli.launchers.claude_code_launcher import launch_claude  # needs [cli]
```

## Development

For development with all testing tools, use `uv` (recommended):

```bash
uv sync                  # core + dev tooling (dev is uv's default group)
uv sync --all-extras     # add every user-facing extra as well
```

Or with pip ≥ 25.1 from a checkout:

```bash
pip install -e ".[all]"
pip install --group dev .
```

This includes:
- Core + all optional extras
- pytest, ruff, mypy, respx for testing

> **Note:** dev tooling lives in a PEP 735 dependency group, not an extra,
> so `pip install nemo-switchyard[dev]` is **not supported** and dev tooling
> never appears in the published wheel's METADATA (it's invisible to
> downstream vulnerability scans).

## Version Compatibility

Switchyard requires Python 3.12 or later.

Supported versions:
- Python 3.12
- Python 3.13

# Switchyard — Development Guide

Setup, testing, project layout, and contribution conventions for hacking on
Switchyard itself. If you only want to **use** the package, see
[README](README.md).

For deeper architectural docs, see [Agents](AGENTS.md) and
[Architecture](docs/architecture.md).

## Setup

Switchyard uses [uv](https://github.com/astral-sh/uv) to manage the virtualenv
and dependencies. Install `uv` first if you don't have it
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard

uv sync                      # creates .venv, installs core + dev tooling
uvx pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
source .venv/bin/activate
```

`dev` is a PEP 735 dependency group (uv's default), so a bare `uv sync` already
installs pytest, ruff, mypy, and friends. Use `uv sync --group dev` if you want
to be explicit. Dev tooling is **not** part of the published wheel's METADATA,
so it never appears in downstream vulnerability scans.

To pull in optional user-facing features as well:

```bash
uv sync --extra cli      # add prompt-toolkit
uv sync --all-extras     # everything (dev group is still included by default)
```

(`server` is already pulled in by the dev group, so you don't need
`--extra server` for local development.)

## Project Structure

```
switchyard/                       # Python package
├── cli/                          # CLI and coding-agent launchers
└── libsy/                        # Typed wrappers for libsy algorithms
switchyard_rust/                  # Python facades over the PyO3 extension
crates/
├── libsy/                         # Routing algorithms and driver
├── libsy-llm-client/              # Translated HTTP model calls
├── protocol/                      # Provider-neutral protocol types
├── switchyard-py/                 # PyO3 bindings for libsy and the server
├── switchyard-server/             # Native HTTP server and TOML configuration
├── switchyard-skill-distillation/ # Skill-distillation contracts
└── switchyard-translation/        # Request, response, and stream translation
tests/                            # Python tests (no API keys required)
docs/                             # User and architecture documentation
pyproject.toml                    # Python package and development tooling
```

## Development Workflow

```bash
uv sync
source .venv/bin/activate

# Run unit tests
uv run pytest tests/ -v

# Lint and type check
uv run ruff check .
uv run mypy switchyard

# Rust formatting, linting, and tests
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

> **Local Git hooks:** install both `pre-commit` and `commit-msg` hooks with
> `uvx pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg`.
> The `commit-msg` hook runs commitlint against Conventional Commits.

> **Pre-commit gate:** `uv run ruff check .` must pass with zero errors before
> any commit or push. The CI lint job runs the same command.

## Integration Tests

The default unit suite runs without any API keys or network access. Live
end-to-end tests against real LLM backends are not part of the public CI
pipeline today; if you want to write one, set credentials and target your
backend directly:

```bash
export OPENAI_API_KEY="sk-..."
# or NVIDIA_API_KEY / ANTHROPIC_API_KEY for the matching backend

uv run pytest tests/your_e2e_test.py -v -x
```

`secrets/secrets.template.json` shows the structure expected by
`secrets/secrets.json` if you prefer a credential file over env vars.

## Human-AI Development Convention

This project uses a structured human-AI collaboration model. The table below
defines who leads each phase of work depending on the type of task.

| Task Type | What to do (architecture) | How to do (details, APIs) | Do it (coding) |
|-----------|---------------------------|---------------------------|----------------|
| **Core infra** (pipeline, abstract base classes) | Human Lead | Human Lead | AI Lead |
| **General classes** (e.g. logging processor) | Human Lead | Human <-> AI Co-lead | AI Lead |
| **Testing** (unit, integration) | Human <-> AI Co-lead | AI Lead | AI Lead |
| **General improvement** (grounded in usage cases) | Human Lead | Human <-> AI Co-lead | AI Lead |
| **Bug fixing** (critical logic bugs) | Human <-> AI Co-lead | AI Lead | AI Lead |

**Definitions:**

- **Human Lead** — Human makes the decision. AI may provide information, but the human owns the outcome.
- **Human <-> AI Co-lead** — Human prompts AI, reviews the response, and makes a quick judgment call. Both contribute, but human has final say.
- **AI Lead** — AI drives the work autonomously. Leverage agents to iterate directly on the repo; human reviews the result.

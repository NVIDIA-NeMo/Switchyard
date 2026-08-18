# AGENTS.md — Switchyard

Switchyard is a Python library for LLM traffic orchestration. It sits between client applications (Claude Code, OpenAI / Anthropic SDK clients, Codex CLI) and LLM backends, handling routing, format translation, logging, A/B testing, and health-aware multi-endpoint serving.

> **Note:** The public distribution is `nemo-switchyard`. All imports use `switchyard.*`,
> and the CLI command is `switchyard` (registered via `pyproject.toml` scripts).

## Engineering Guidelines

Working principles for any agent (or human) writing code in this repo. These are
about *how* to work; project-specific conventions and validation commands live
elsewhere in this `AGENTS.md`.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Comments Explain Code, Not Project Management

**Source comments are about the code. Tracking lives in the tracker.**

- No issue/PLAN/step references in code (`TODO(step-6)`, "lands in step 4",
  "tracked as ISSUE-001", links to `docs/issues/`). These rot the moment the
  plan changes and leak project-management state into source.
- A plain `// TODO:` describing a concrete code gap is fine; a `// TODO`
  pointing at a tracker step is not.
- Comment what isn't obvious from the code: why a thing is done this way,
  invariants, non-obvious edge cases. Don't narrate what the code already says.
- Module doc comments should state what the module is *for*, not its build
  schedule or its "empty for now" status.

### 6. Commit Discipline

**One step, one reviewed, one-line commit.**

- One focused commit per step; every changed line traces to that step.
- Single-line commit message in Conventional Commits form
  (`type(scope): summary`). No body, no `Co-Authored-By` trailer.
- Pull request titles use the same Conventional Commits form.
- Use `git commit -s` so every commit carries the required DCO sign-off.
- Never commit unprompted. Show the diff, get approval, then commit.

Before repairing DCO, inspect every affected commit:

```bash
git log origin/main..HEAD --format='%h %an <%ae> %s'
```

If every affected commit is yours, add the trailers and update the remote safely:

```bash
git rebase origin/main --signoff
git push --force-with-lease origin HEAD
```

For a mixed-author branch, use an interactive rebase and mark only your unsigned commits for
editing. At each stop, run `git commit --amend --no-edit --signoff`, then
`git rebase --continue`. Never add your sign-off to another contributor's commit.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### 7. Review Discipline

- Verify every finding against the current code path before reporting it.
- Draft findings before posting review comments unless the user explicitly asks you to post them.
- Resolve only your own review threads, and only after verifying the fix in code.

## Task-Specific Skills

The repository keeps a small set of optional runbooks under `.agents/skills/`. Read a skill only
when its description directly matches the task. Ordinary code exploration, implementation,
testing, and review do not require loading a skill.

| Skill | Use it for |
|---|---|
| `publish-python-release` | Python wheel artifacts, PyPI releases, and release workflow changes |
| `switchyard-coding-agent-launchers` | Claude Code, Codex, or OpenClaw launcher behavior |
| `switchyard-docs` | Published MkDocs pages, strict builds, previews, and docs CI |
| `switchyard-rust-review` | Focused review of Rust, PyO3, async, streaming, and crate boundaries |
| `switchyard-testing-ci` | Selecting non-obvious validation or diagnosing CI failures |

Skills should contain stable operational constraints, not mutable architecture inventories. Read the
current source and CI workflows for implementation details.

## Architecture

The supported serving path is native Rust:

```
HTTP request → switchyard-server → libsy Algorithm → RoutedLlmClient
             → switchyard-translation → upstream model
```

`switchyard-server` loads explicit TOML deployments and exposes the OpenAI Chat,
OpenAI Responses, and Anthropic Messages APIs. `switchyard-libsy` owns routing
algorithms, `switchyard-protocol` owns provider-neutral request and response
types, and `switchyard-llm-client` performs translated HTTP calls.

Python is an integration layer. `switchyard launch` hosts the native server for
coding agents, `switchyard.libsy` exposes selected algorithms, and
`switchyard_rust.server` exposes the native server lifecycle through PyO3.

## Project Structure

```
switchyard/
├── __init__.py                     # Package version
├── cli/                            # CLI (requires `nemo-switchyard[cli]`)
│   ├── switchyard_cli.py           # `switchyard` entry point
│   ├── launch_command.py           # `switchyard launch`
│   ├── defaults/                   # packaged OpenRouter TOML deployment
│   └── launchers/                  # Claude, Codex, and OpenClaw launchers
└── libsy/                          # typed Python wrappers for libsy algorithms

switchyard_rust/                    # Python facades over the PyO3 extension
crates/libsy/                       # routing algorithms and driver
crates/libsy-llm-client/            # translated HTTP LLM client
crates/protocol/                    # provider-neutral protocol types
crates/switchyard-server/           # native HTTP server and TOML config
crates/switchyard-translation/      # wire-format codecs
crates/switchyard-py/               # libsy and server PyO3 bindings

tests/                              # Unit tests (pytest)
```

## Tech Stack

- **Rust 1.96.1**, edition 2024, Tokio, Axum, and PyO3
- **Python 3.10+** for launchers and native bindings
- **prompt-toolkit** for interactive launcher sessions
- **uv** as the package manager (preferred over pip)
- **Cargo test + pytest** for testing
- **ruff** for linting, **mypy** (strict) for type checking

## Setup

```bash
uv sync               # Core + dev tooling (dev is uv's default group)
uv sync --group dev   # Explicit form, equivalent to the above
source .venv/bin/activate
```

`dev` lives in `[dependency-groups]` (PEP 735), not in `[project.optional-dependencies]`,
so it is **not** advertised in the published wheel's METADATA — pytest, ruff, mypy,
and their transitives never appear in downstream vulnerability scans.

## Commands

### Running Switchyard

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Launch against the packaged OpenRouter deployment.
switchyard launch claude --model switchyard
switchyard launch codex --model switchyard

# Or select a route from a custom native TOML deployment.
switchyard launch claude --model my-route --config routes.toml

# Run a standalone native server.
switchyard-server --config routes.toml --port 4000
```

### Testing

```bash
# Unit tests — no API keys needed
uv run pytest tests/ -v

# Single test file / function
uv run pytest tests/test_launchers.py -v

# Live end-to-end tests are not part of the public test suite; if you write
# one, set the provider key explicitly and run it directly, e.g.:
#   OPENAI_API_KEY=sk-... uv run pytest tests/your_e2e_test.py -v -x

# Lint / type check (run before every commit)
uv run ruff check .
uv run mypy switchyard
cargo test --workspace
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible backends |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API |
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude |
| `NVIDIA_API_KEY` | API key for NVIDIA NIM / Inference Hub |
| `OPENROUTER_API_KEY` | OpenRouter key used by the packaged launcher deployment |

## Code Style

- **Line length**: 100 chars (ruff, E501 ignored)
- **Target version**: Python 3.10 — use `X | Y` union syntax
- **Imports**: sorted by ruff (`I` rules). Use `TYPE_CHECKING` guards for circular imports.
- **File naming**: file name = `snake_case` of its primary class. One class per file when practical.
- **Type hints**: throughout. `py.typed` marker present; mypy runs strict.
- **Async**: async-only. If you need sync, use `asyncio.run()`.
- **Testing**: `respx` for HTTP mocking, `pytest-mock` for general mocking. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- **Rust**: Panicking calls such as `panic!()`, `unwrap()` and `.except()` are not allowed in production source code. Propagate errors with `?`. They are allowed in tests.
  return typed errors, or match explicitly in tests so failures stay intentional and visible.
- **Comments**: For Rust changes, add concise comments for module/file intent, public structs/enums,
  public methods, private helpers with non-obvious behavior, and tests that encode important
  behavior. Prefer one-line comments when enough. Add block comments before complex validation,
  routing, config-building, async, lifecycle, or concurrency logic.
- **Docstrings**: Add docstrings for public functions, classes, methods, and API entry points. In
  Rust, use `///` doc comments for public items; in Python, use concise triple-quoted docstrings.
  Public docs should state what the API does, important invariants, and error behavior when relevant.

## Boundaries

### Always do
- File name = snake_case of the primary class exported. Rename on touch.
- Run `uv run ruff check .` (zero errors) and `uv run pytest tests/` before pushing.
- Run `cargo test --workspace` for Rust behavior changes.
- Write focused unit tests for new behavior and bug fixes.
- Keep provider-neutral request and response types in `switchyard-protocol`.
- Map upstream context-window errors to `SwitchyardError::ContextWindowExceeded`.

### Ask first
- Modifying `pyproject.toml` dependencies.
- Adding new HTTP endpoints.
- Removing or renaming public Rust, PyO3, or Python APIs.

### Never do
- Commit API keys or secrets (`secrets/` is gitignored).
- Remove or rename public API exports without an explicit deprecation plan.

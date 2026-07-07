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
- Never commit unprompted. Show the diff, get approval, then commit.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Start here: discover skills before doing anything

This repo ships project-specific agent skills under `.agents/skills/`. **Before reading code, planning a change, debugging a failure, or running validation, list and consult these skills.** They encode the workflows the maintainers expect you to follow.

```bash
ls .agents/skills/
```

Two general-purpose entry points cover most tasks:

- `.agents/skills/switchyard-codebase-exploration/SKILL.md` — read-before-edit workflow; builds an impact map of the files, symbols, and tests touched by a change.
- `.agents/skills/switchyard-testing-ci/SKILL.md` — picks the smallest trustworthy local validation set and maps failures to fixes.
- `.agents/skills/switchyard-pr-reviewer/SKILL.md` — multi-mode PR-review workflow (correctness, tests, design-vs-ticket, simplify, docs, Rust-craft); adversarially verifies every finding and drafts comments before posting. Dispatches Rust to `rust-code-reviewer`. Use for any "review this PR / is this blocking? / post comments" request.

Task-specific skills (publish-package, run-pre-merge-checks, etc.) live alongside them — scan the directory and read the SKILL.md of any whose `description` matches the task. If no skill applies, say so explicitly and proceed; do not silently skip discovery.

### Keep skills in sync with the code they cover

A skill that points at stale `file:line` references, removed flags, or a renamed symbol is worse than no skill — it actively misleads the next agent. **If your change modifies code that a skill describes, update the skill in the same change.** Specifically:

- After editing any file referenced by a SKILL.md, re-grep the skill for the file path and verify each `file:line` and symbol still resolves.
- If you rename a symbol, move a file, change a CLI flag, add or remove a factory/recipe/processor/backend/translator, or change a public export, update every skill that mentions it.
- If you discover a workflow the skill does not cover but should, add a row to the skill's Quick Reference or Anti-Patterns table — do not let the gap survive the PR.
- Anti-patterns called out in `switchyard-lib-core` and `switchyard-coding-agent-launchers` are the contract: if you change the golden patterns those skills describe (random-routing factory + recipe shape, CLI-launcher recipe consumption), update the skill first and have the skill change reviewed alongside the code.

The skill files are part of the codebase. Treat them like tests or docs — drift is a real defect, caught only by the next person who consults the skill.

## Architecture: staged chain

Everything flows through a fixed-shape chain enforced at construction time:

```
request-side component* → LLMBackend → response-side component* → TranslationEngine
```

The chain executor is `Switchyard` (`switchyard/lib/switchyard.py`). `LLMBackend` remains the
shared role class re-exported from `switchyard/lib/roles.py`; request-side and response-side
processors are plain async components with `process(...)` methods.
Direct Rust bindings for migrated concrete processors/backends are exposed from
`switchyard_rust.components` and implemented under `crates/switchyard-py/src/component_bindings/`.

| Stage | Binding | Method | Purpose |
|------|-----|--------|---------|
| Request component | Plain Python/Rust object | `async process(ctx, request) -> ChatRequest` | Pre-process (routing, buffering, auth) |
| `LLMBackend` | `switchyard.lib.roles` | `async call(ctx, request) -> ChatResponse` | Make the LLM call. Exactly one per chain. |
| Response component | Plain Python/Rust object | `async process(ctx, response) -> ChatResponse` | Post-process (logging, stats) |
| `TranslationEngine` | `switchyard_rust.translation` | `async translate(ctx, request, response) -> Any` | Convert to client's wire format |

## Project Structure

```
switchyard/
├── __init__.py                     # Public API — all exports live here
├── lib/                            # Core library
│   ├── roles.py                    # Rust-owned LLMBackend re-export and translation aliases
│   ├── switchyard.py               # Switchyard — chain executor
│   ├── proxy_context.py            # ProxyContext — per-request state carrier
│   ├── profiles/                   # Profile configs/runtimes for pre-built routing behavior
│   ├── route_table.py              # RouteTable — model-id dispatch to runnable chains
│   ├── route_table_builders.py     # Shared profile-backed table builders
│   ├── llm_client.py               # OpenAILLMClient
│   ├── cost_estimator.py           # Token-cost bookkeeping
│   ├── live_stats_collector.py     # Live request stats
│   ├── stats_accumulator.py        # Stats accumulation helpers
│   ├── request_metadata.py         # RequestMetadata
│   ├── chat_response/              # Rust-backed response re-exports + stream adapters
│   │   ├── base.py                 # ChatResponse, ChatResponseType compatibility re-export
│   │   ├── openai_chat.py          # ResponseStream
│   │   ├── openai_responses.py     # ResponsesApiStream
│   │   ├── anthropic.py            # AnthropicResponseStream
│   │   └── gemini.py               # GeminiResponseStream
│   ├── backends/                   # LLMBackend implementations
│   │   ├── openai_llm_backend.py           # OpenAiPassthroughBackend
│   │   ├── openai_native_backend.py        # OpenAiNativeBackend
│   │   ├── anthropic_native_llm_backend.py # AnthropicNativeBackend
│   │   ├── gemini_native_llm_backend.py    # GeminiNativeBackend
│   │   ├── latency_service_llm_backend.py  # LatencyServiceLLMBackend
│   │   ├── llm_target.py                   # LlmTarget, BackendFormat
│   │   ├── multi_llm_backend.py            # MultiLlmBackend helpers
│   │   ├── stats_llm_backend.py            # StatsLlmBackend
│   │   ├── backend_format_resolver.py      # BackendFormatResolver
│   │   └── health_poller.py                # HealthPoller, EndpointHealthStatus
│   ├── processors/                 # Request-side / response-side component implementations
│   │   ├── format_translate.py
│   │   ├── routellm_request_processor.py
│   │   ├── random_routing_request_processor.py
│   │   ├── stats_request_processor.py
│   │   ├── stats_response_processor_accumulator.py
│   │   ├── stats_response_processor_live_collector.py
│   │   ├── intake_request_processor.py
│   │   ├── intake_response_processor.py
│   │   ├── intake_payload_builder.py
│   │   └── intake_client.py
│   ├── endpoints/                  # FastAPI endpoint wrappers (require `nemo-switchyard[server]`)
│   │   ├── openai_chat_endpoint.py         # OpenAIChatEndpoint
│   │   ├── anthropic_messages_endpoint.py  # AnthropicMessagesEndpoint
│   │   ├── gemini_endpoint.py              # GeminiEndpoint
│   │   ├── responses_endpoint.py           # ResponsesEndpoint
│   │   ├── stats_endpoint.py               # StatsEndpoint
│   │   ├── sse_helpers.py
│   │   └── base.py
│   └── config/
│       ├── intake_sink_config.py
│       └── latency_service_backend_config.py
├── cli/                            # CLI (requires `nemo-switchyard[cli]`)
│   ├── switchyard_cli.py           # `switchyard` entry point
│   ├── launch_command.py           # `switchyard launch claude/codex`
│   ├── configure_command.py        # `switchyard configure`
│   ├── status.py                   # render_status helper used by `configure --show`
│   ├── command_utils.py
│   ├── output.py
│   ├── launchers/                  # claude_code_launcher, codex_cli_launcher
│   ├── model_catalog/              # model_discovery
│   ├── routing/                    # route_builder
│   ├── tui/                        # Terminal UI widgets
│   └── config/
│       └── user_config.py          # Saved user defaults (~/.config/switchyard/)
└── server/                         # FastAPI app factory + server utilities
    ├── switchyard_app.py           # build_switchyard_app()
    ├── server_util.py              # Shared CLI / server plumbing
    ├── shell_tui.py                # Shell TUI session
    └── verify.py                   # e2e verification helpers

tests/                              # Unit tests (pytest)
```

## Tech Stack

- **Python 3.12+**, async-first (asyncio)
- **FastAPI + Uvicorn** for HTTP (`nemo-switchyard[server]`), **sse-starlette** for SSE streaming
- **OpenAI SDK** (`openai>=2.30`) — primary client; the translation engine converts all inbound formats to Chat Completions
- **Anthropic SDK** (`anthropic>=0.94`)
- **RouteLLM** for ML-based strong/weak routing (`nemo-switchyard[gpu]`)
- **httpx** for direct HTTP (health polling, Anthropic Messages)
- **uv** as the package manager (preferred over pip)
- **pytest + pytest-asyncio** for testing, **respx** for HTTP mocking
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

### Running the server

```bash
export OPENAI_API_KEY="sk-..."       # or NVIDIA_API_KEY / ANTHROPIC_API_KEY where supported
export OPENROUTER_API_KEY="sk-or-..." # pass with --api-key or save via configure

# Serve a profile config (passthrough, random-routing, llm-routing, stage_router).
# Endpoints, targets, and profiles live in the YAML; see docs/routing_algorithms/overview.md.
switchyard serve --config profiles.yaml --port 4000
switchyard serve --config profiles.yaml --inbound anthropic --port 4000

# One-command launchers — single-model passthrough via --model
switchyard launch claude --model openai/gpt-4o-mini \
    --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"
switchyard launch codex --model openai/gpt-4o-mini \
    --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"

# Launchers default to built-in LLM-classifier routing; tune the tiers:
switchyard launch claude \
    --weak-model openai/gpt-4o-mini --classifier-model openai/gpt-4o \
    --classifier-min-confidence 0.6

# Verification and saved config
switchyard verify --model openai/gpt-4o-mini
switchyard configure --show
```

### Testing

```bash
# Unit tests — no API keys needed
uv run pytest tests/ -v

# Single test file / function
uv run pytest tests/test_switchyard.py -v
uv run pytest tests/test_random_routing_llm_backend.py::test_server_config -v

# Live end-to-end tests are not part of the public test suite; if you write
# one, set the provider key explicitly and run it directly, e.g.:
#   OPENAI_API_KEY=sk-... uv run pytest tests/your_e2e_test.py -v -x

# Lint / type check (run before every commit)
uv run ruff check .
uv run mypy switchyard
```

### Adding a new component

1. Pick the right stage: request component (pre-call), response component (post-call), `LLMBackend` (rare), or Rust translation codec work.
2. Create a file with the explicit name (`snake_case` of the class name), one class per file.
3. Implement the async method for that stage (`process` for components, `call` for backends).
4. Wire into the owning profile config.
5. Add tests under `tests/`.
6. Export from the relevant `__init__.py` and from `switchyard/__init__.py`'s `__all__`.

### Example: custom request component

```python
from switchyard.lib.proxy_context import ProxyContext
from switchyard import ChatRequest


class MyRequestComponent:
    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.metadata["my_key"] = "my_value"
        return request
```

### Profiles and app factory

```python
from switchyard import PassthroughProfileConfig, ProfileSwitchyard, build_switchyard_app
import uvicorn

switchyard = ProfileSwitchyard(PassthroughProfileConfig(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
).build())
uvicorn.run(build_switchyard_app(switchyard), port=4000)
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible backends |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API |
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude |
| `GEMINI_API_KEY` | API key for Google Gemini generateContent backends |
| `NVIDIA_API_KEY` | API key for NVIDIA NIM / Inference Hub |
| `OPENROUTER_API_KEY` | OpenRouter key for examples and saved provider setup; pass via `--api-key` or `switchyard configure` |
| `SWITCHYARD_INTAKE_CAPTURE_CONTENT` | Set truthy to include prompt/response text in intake; default off (metadata-only) |

## Code Style

- **Line length**: 100 chars (ruff, E501 ignored)
- **Target version**: Python 3.12 — use `X | Y` union syntax
- **Imports**: sorted by ruff (`I` rules). Use `TYPE_CHECKING` guards for circular imports.
- **File naming**: file name = `snake_case` of its primary class. One class per file when practical.
- **Type hints**: throughout. `py.typed` marker present; mypy runs strict.
- **Async**: async-only. If you need sync, use `asyncio.run()`.
- **Testing**: `respx` for HTTP mocking, `pytest-mock` for general mocking. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- **Rust**: do not use `.expect()` in Rust source or tests. Propagate errors with `?`,
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
- Export new public classes from `switchyard/__init__.py` with `__all__`.
- Write unit tests for new roles and bug fixes.
- Use `ProxyContext.metadata` for cross-component state within a request.
- In a new `LLMBackend`, map upstream context-window 4xx to `SwitchyardError::ContextWindowExceeded` (Rust) — the chain executor uses it for evict-and-retry. See [Context-Window Handling](docs/context_window.md).

### Ask first
- Modifying `pyproject.toml` dependencies.
- Changes to the chain shape or Rust-owned role classes in `switchyard/lib/roles.py`.
- Adding new HTTP endpoints.
- Removing or renaming any public API currently in `switchyard/__init__.__all__`.

### Never do
- Commit API keys or secrets (`secrets/` is gitignored).
- Remove or rename public API exports without an explicit deprecation plan.

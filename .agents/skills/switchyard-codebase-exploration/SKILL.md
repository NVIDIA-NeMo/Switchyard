---
name: switchyard-codebase-exploration
description: Use when modifying, debugging, reviewing, refactoring, renaming, restructuring, or planning any Switchyard change — even if the user only names a symptom, test failure, CLI flag, endpoint, model route, or file path. Triggers on phrases like "fix this bug", "why is X failing", "where does Y live", "add a new profile", "rename X to Y", "refactor this", "explain how X works", or any request that will touch `switchyard/`, `tests/`, or `examples/`. Forces a fresh impact map before editing so agents do not edit from stale memory, and routes the agent to the matching workstream skill (`switchyard-lib-core`, `switchyard-coding-agent-launchers`, `switchyard-testing-ci`) before reading source.
---

# Switchyard Codebase Exploration

> Rust crates under `crates/` are active on the Rust core branch. Inspect
> `crates/switchyard-core`, `crates/switchyard-components`,
> `crates/switchyard-server`, `crates/switchyard-translation`,
> `crates/switchyard-py`, and `crates/*/tests`
> directly when a change touches Rust.
> Direct PyO3 bindings for concrete components live under
> `crates/switchyard-py/src/component_bindings/` with Python lazy exports in
> `switchyard_rust/components.py`.

## Overview

**Read the current checkout before you edit it.** Branch state, profile wiring, exports, tests, and optional-dependency boundaries drift faster than any memorized map. This skill produces a fresh impact map for the specific change, so the edit plan is grounded in what is actually there — not what was there last week.

**Load the matching workstream skill before diving into source.** Once you have classified the change (see [§ Decompose, step 1](#1-scope-packet)), open the workstream SKILL.md — [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md) for anything under `lib/`, [`switchyard-coding-agent-launchers`](../switchyard-coding-agent-launchers/SKILL.md) for `cli/launchers/` work. Skipping this step is how agents reinvent construction paths, stats collectors, and CLI flags.

If you are about to run validation (`pytest`, `ruff`, `mypy`, CI gates), pair this skill with [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md): explore here first, then pick the validation set there.

## Quick Reference

| You need to… | Command |
|---|---|
| Map current diff | `git status -sb && git diff --stat && git diff --name-only` |
| Map a specific file's importers and tests | `rg -n "module_stem|ClassName|function_name" switchyard tests docs AGENTS.md` |
| Map by symbol (class/function name) | `rg -n "ProfileSwitchyard|RandomRoutingPresets" switchyard tests docs AGENTS.md` |
| List source and tests by area | `find switchyard tests examples crates -maxdepth 3 -type f | sort` |
| Find class/function definitions across repo | `rg -n "class \|def \|async def " switchyard tests` |
| Trace a symbol through the codebase | `rg -n "<symbol-or-route-or-flag>" switchyard tests docs AGENTS.md` |

## Operating Rule

Read the current checkout before editing. Do not rely on a memorized architecture map: branch state,
profile wiring, exports, tests, and optional dependency boundaries move often.

Before making a code change, be able to name:

- the client ingress path: CLI, FastAPI endpoint, Python API, or route-table host
- the chain slot touched: request-side component, `LLMBackend`, response-side component, `TranslationEngine`, or wiring only
- the wire formats involved: OpenAI Chat, Anthropic Messages, OpenAI Responses, streaming/non-streaming
- the profile/route-table/export path that exposes the behavior
- the focused tests and the broad validation gate to run

## Manual Map First

From the repo root, build a fresh impact map every time the branch, diff, or task changes:

```bash
git status -sb
git diff --stat
git diff --name-only
```

If there is no diff yet, search likely owners explicitly:

```bash
rg -n "ProfileSwitchyard|RandomRoutingPresets" switchyard tests docs AGENTS.md
rg -n "class |def |async def " switchyard tests examples
find switchyard tests examples crates -maxdepth 3 -type f | sort
```

For a specific file, search by the module stem, primary class/function names, and public flags:

```bash
rg -n "random_routing|RandomRouting|strong_probability" switchyard tests docs AGENTS.md
rg -n "test_.*random|random_routing" tests
```

Ignore generated docs build artifacts (`docs/.venv-docs/`, `docs/_build/`, `site/_build/`) and
local virtualenvs when building the map; remove them if they pollute `git status`.

## Examples

Input: `fix the --preset help regression in launch claude`
Output: Map with `--symbol RandomRoutingPresets`, **load [`switchyard-coding-agent-launchers`](../switchyard-coding-agent-launchers/SKILL.md)** (ingress area), then inspect CLI parser, launcher wiring, slim-smoke command, and launch/verify tests before editing.

Input: `add a new routing preset opus_qwen3` (or any "add a preset" / "new shipping pair" request)
Output: profile/preset area → **load [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md)** for the profile-owned config pattern and the "no parallel stats collector" rule. Also load [`switchyard-coding-agent-launchers`](../switchyard-coding-agent-launchers/SKILL.md) when launchers expose the preset. Then map `--symbol RandomRoutingPresets` to confirm test/CLI/cost-estimator touch points before adding the new id.

Input: `this responses streaming test is failing`
Output: Map the touched test or `ResponsesApiStreamingChatResponse`, **load [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md)** (chain-core + wire-types area) and [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md), then read response types, translation engine, SSE helpers, endpoint code, and focused streaming tests before changing shapes.

Input: `rename random_routing to weighted_routing everywhere`
Output: **Load [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md) first** — it flags random routing as a reference profile and lists the anti-patterns that block thoughtless renames of public/profile identifiers. Then map every importer and CLI flag before proposing the plan.

Input: `add intake telemetry that records routing decisions`
Output: Observability/state area → **load [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md)** for the "don't fork the stats stack" rule and the existing intake processor wiring, then map `processors/intake_*`, `processors/stats_*`, and the profile that owns the existing routing-decision payload.

## Decompose the Exploration

Use these packets independently, then merge the findings into one edit plan.

### 1. Scope Packet

```bash
git status -sb
git diff --stat
git diff --name-only
```

Classify the task as one or more of:

- chain-core: `roles.py`, `switchyard.py`, profile runtimes, `ProxyContext`
- wire-types/translation: `chat_request/`, `chat_response/`, `translation/`, `translators/`
- backend/routing: `backends/`, profile routing helpers, health polling, tier config
- profiles/route-table: `profiles/`, `route_table.py`, `route_table_builders.py`
- ingress: `cli/`, `server/`, `lib/endpoints/`
- observability/state: stats, live stats, intake, metadata, cost estimator
- public/package: `switchyard/__init__.py`, `pyproject.toml`, optional extras, entry points
- tests/docs/skills: `tests/`, `.agents/skills/`, docs, CI-only validation

**Then load the matching workstream skill — before reading source.** Skipping this step is the single most common way agents reinvent existing infrastructure.

| If the area is… | Load this skill next |
|---|---|
| chain-core, wire-types/translation, backend/routing, profiles/route-table, observability/state | [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md) — profile pattern + anti-patterns (no new stats collectors, no factory reinvention, preserve chain shape) |
| ingress (CLI launchers, server, endpoints) | [`switchyard-coding-agent-launchers`](../switchyard-coding-agent-launchers/SKILL.md) — launchers build profile-backed apps or route tables |
| any planned edit followed by validation | [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — picks the smallest local validation set that is CI-equivalent |
| public/package, tests/docs/skills | Use the lib-core skill if the public API touches `lib/`; otherwise testing-ci's skill/docs gate |

### 2. Runtime Path Packet

Trace the request end-to-end for the affected surface. Prefer `rg` over guessing:

```bash
rg -n "<class-or-function>|<route>|<flag>|<config-field>" switchyard tests docs AGENTS.md
rg -n "ProfileSwitchyard|RouteTable|build_switchyard_app|app.state.switchyard" switchyard tests
```

Read the nearest owner files plus their call sites. For runtime behavior, include at least one
upstream entry point and one downstream test.

### 3. Role Contract Packet

When touching any chain component, reread:

- `switchyard/lib/roles.py`
- `switchyard/lib/switchyard.py`
- `switchyard/lib/proxy_context.py`

Keep the logical shape fixed: request-side work → `LLMBackend` → response-side work → `TranslationEngine`.
Processor components use async `process(...)`; backend and translation role methods are async. Use `ctx.metadata` for cross-component state.

### 4. Wiring Packet

For behavior exposed outside a unit test, inspect each applicable wiring layer:

- public API: `switchyard/__init__.py` and `__all__`
- profiles: `switchyard/lib/profiles/`
- route tables: `switchyard/lib/route_table.py`, `switchyard/lib/route_table_builders.py`
- server: `switchyard/server/switchyard_app.py`, `switchyard/server/server_util.py`
- endpoints: `switchyard/lib/endpoints/*.py`
- CLI: `switchyard/cli/switchyard_cli.py`, `switchyard/cli/launch_command.py`, launcher/config modules
- packaging: `pyproject.toml` extras and `project.scripts`

Do not add a class without checking whether it needs an import/export, profile config,
CLI flag, endpoint mount, or test fixture update.

### 5. Test Packet

Search by symbol and module stem to catch direct and indirect coverage:

```bash
rg -n "<symbol>|<module-stem>" tests switchyard
```

Run focused tests first. Use hermetic broad tests by default:

```bash
uv run pytest tests/test_<area>.py -v -o addopts=
uv run pytest tests/ -v -m "not integration"
```

Run live `tests/e2e/` only when the user explicitly wants provider calls and credentials are
intentionally available.

## Path-Specific Read Rings

Use these rings as starting points, not replacements for dynamic search.

- CLI launch/config: `switchyard/cli/switchyard_cli.py`, `launch_command.py`, `launchers/*`,
  `config/user_config.py`, `server/server_util.py`, plus `tests/test_launch_*`,
  `tests/test_user_config.py`, `tests/test_model_discovery.py`.
- FastAPI endpoints/server: `server/switchyard_app.py`, `lib/endpoints/*`, request/response
  types, translators, `tests/test_switchyard_app_factory.py`, `tests/test_endpoint_state_contract.py`,
  endpoint-specific tests.
- Translation: `lib/translation/request_engine.py`, `response_engine.py`, provider conversion
  modules, `lib/translators/default_response_translator.py`, translation chaos tests.
- Routing/backends: `llm_target.py`, `backend_format_resolver.py`, concrete backend,
  `multi_llm_backend.py`, stats wrapper, profile config for that router, routing tests.
- Profiles/route tables: profile config/runtime module, route-table builders, YAML tests,
  public exports.
- Stats/intake: request processor, response processor, accumulator/live collector/client, stats
  endpoint, intake processors, metadata tests.
- Skills/docs-only: parse frontmatter/YAML, run `git diff --check`, and remember `.agents` is
  gitignored so use `git add -f` when publishing.

## Common Traps

- `switchyard` is the package and CLI name; do not reintroduce `nemo_switchyard` imports or docs.
- Do not change Rust-owned backend role classes, chain shape, public API exports, HTTP endpoints, or dependencies without
  explicit user intent.
- Keep optional heavy/server/CLI/intake dependencies out of top-level imports and default install
  paths. Lazy imports in profile/backend helpers and `switchyard/__init__.__getattr__` are deliberate.
- Treat NVIDIA Inference Hub and OpenRouter as OpenAI-compatible upstreams unless the current code
  proves otherwise. Prefer provider/base-url/API-key config and existing OpenAI-compatible backends
  before adding provider-specific translation paths or backend roles.
- Backends should declare `supported_request_types` and normalize with the translation engine rather
  than assuming OpenAI Chat input.
- Endpoints read `request.app.state.switchyard`; it may be a single profile-backed runtime or a `RouteTable`.
- Random routing uses `strong_probability` (`rng.random() < strong_probability`); do not revive the
  old inverted threshold semantics.
- Stats paths often require a shared `StatsAccumulator` across request/response processors and
  `StatsLlmBackend`.
- Pytest has `addopts = "-x"`; use `-o addopts=` when triaging all failures.
- Generated docs artifacts and local virtualenvs are not CI inputs; remove them or keep them out of
  local validation.

## Edit Plan Output

Before editing, write a compact internal plan with:

```text
Change: <one sentence>
Ingress: <CLI/server/Python/profile/route-table>
Chain slots: <roles touched>
Files to read: <source + tests>
Files to edit: <minimal set>
Validation: <focused tests + broad gate>
Risk: <optional deps/live calls/public API/streaming/stats>
```

After editing, report exactly what ran and whether live provider calls were involved.

## Related Skills

- [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — after the impact map is in hand, use that skill to pick the smallest local validation set that is genuinely CI-equivalent for the files you touched.

---
name: switchyard-coding-agent-launchers
description: Use when editing `switchyard launch claude` / `switchyard launch codex` / `switchyard launch openclaw` or anything under `switchyard/cli/launchers/`, `switchyard/cli/launch_command.py`, `switchyard/cli/routing/`, `switchyard/cli/model_catalog/`, or `switchyard/cli/config/user_config.py`. Triggers on "modify the claude launcher", "wire codex to a new model", "add an openclaw launcher flag", "saved user defaults", "routing profiles", or "first-run configure".
---

# Switchyard Coding-Agent Launchers

## Overview

CLI launchers are thin shells over profile-backed `SwitchyardApp` values:

- Single-model launches build a passthrough profile with
  `build_tier_passthrough_switchyard(...)`.
- Multi-route launches load routing-profile YAML with
  `load_route_bundle_table(...)`, which returns a `RouteTable`.
- The FastAPI app receives either a single profile-backed runtime or a
  `RouteTable` and serves it through `build_switchyard_app(...)`.

Do not reintroduce `SwitchyardRecipes`, factory registries, middleware bundles,
or request/response pipeline wrappers in launcher code.

`--routing-profiles` is a global switchyard flag before the subcommand.
`--model` stays on the launcher subcommand. They are mutually exclusive for
the launcher entry points that need a single initial model.

Pair this skill with [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md)
before editing and [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md)
after.

## Quick Reference

| You need to… | Command / file pointer |
|---|---|
| Launch claude with a single model | `switchyard launch claude --model nvidia/moonshotai/kimi-k2.5` |
| Launch codex with a single model | `switchyard launch codex --model nvidia/moonshotai/kimi-k2.5` |
| Launch openclaw with a single model | `switchyard launch openclaw --model nvidia/moonshotai/kimi-k2.5` |
| Launch a one-off OpenRouter model | `switchyard launch codex --model openai/gpt-5.2 --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"` |
| Save OpenRouter provider defaults | `switchyard configure --provider openrouter --base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"` |
| Launch with a YAML route bundle | `switchyard --routing-profiles ~/.config/switchyard/profiles.yaml launch claude` |
| Standalone proxy server | `switchyard --routing-profiles routes.yaml serve --port 4000` |
| Build one launcher chain | `build_tier_passthrough_switchyard(...)` in `switchyard/lib/route_table_builders.py` |
| Build/merge YAML routes | `load_route_bundle_table(...)` / `build_route_bundle_table(...)` in `switchyard/cli/route_bundle.py` |
| Look up upstream model catalogs | `fetch_model_ids(...)` in `switchyard/cli/model_catalog/model_discovery.py` |
| Add a plan-execute route | Add `type: plan_execute` in route YAML; `_plan_execute_switchyard` in `route_bundle.py` maps it to `PlanExecuteProfileConfig`. |
| Add a deterministic route | Add `type: deterministic` in route YAML; `_deterministic_switchyard` maps it to `DeterministicRoutingProfileConfig`. |
| Add a preset | Put it beside the profile config under `switchyard/lib/profiles/`; presets return typed config objects, not runnable apps. |
| Inspect saved defaults | `switchyard configure --show` reads `switchyard/cli/config/user_config.py` and renders via `switchyard/cli/status.py`. |

## Launcher Pattern

1. Parse CLI args in `switchyard/cli/switchyard_cli.py` and dispatch through
   `switchyard/cli/launch_command.py`.
2. Resolve credentials, endpoint, saved defaults, and routing-profile path.
   Use OpenRouter through the OpenAI-compatible path: pass
   `--base-url https://openrouter.ai/api/v1 --api-key "$OPENROUTER_API_KEY"`
   for a one-off launch, or save provider defaults with `switchyard configure`.
   Do not assume launcher subcommands accept `--provider`; provider ids are a
   configure/defaults concern.
3. Build a `StatsAccumulator` once for the launcher process.
4. Build the single-target profile-backed app with
   `build_tier_passthrough_switchyard(...)`, or load route YAML with
   `load_route_bundle_table(...)`.
5. If both a launcher model and YAML table are present, wrap the single model
   in a `RouteTable` with `build_single_model_table(...)` and merge YAML entries.
6. Call `build_switchyard_app(...)`, start uvicorn in a daemon thread, wait for
   `/health`, then spawn the external CLI.

Claude, Codex, and OpenClaw differ only in how they configure the external
agent process:

- Claude uses environment variables such as `ANTHROPIC_BASE_URL` and
  `ANTHROPIC_MODEL`.
- Codex injects a transient provider with repeated `-c` flags and a temporary
  model catalog.
- OpenClaw writes a transient `openclaw.json` workspace and points the process
  at it with `OPENCLAW_STATE_DIR`, `OPENCLAW_HOME`, and
  `OPENCLAW_CONFIG_PATH`.

## RouteTable Behavior

`RouteTable` maps inbound request `model` values to profile-backed runtimes.
Route YAML and launchers share this model-dispatch path:

- `type: model` registers only the YAML key as an alias to one target.
- `type: passthrough` registers the YAML key and discovered direct models.
- `type: random_routing` registers the YAML key as the routing profile plus
  direct strong/weak passthrough entries.
- `type: deterministic` and `type: stage_router` register the route key plus direct
  strong/weak passthrough entries.
- `type: plan_execute` registers the route key plus the executor as a direct
  passthrough; the planner is internal routing logic.
- `type: routellm`, `type: latency_service`, and `type: noop` register the
  route key.

Runtime-only hooks such as intake are passed to route-table builders as kwargs;
route YAML does not declare arbitrary Python processors.

## Anti-Patterns

| Anti-pattern | What to do instead | Why |
|---|---|---|
| Adding a launcher flag for a routing policy. | Add a route field in `switchyard/cli/route_bundle.py`. | Routing policies are composable YAML/profile concerns; launcher flags should stay small. |
| Building `Switchyard(...)` manually in launcher code. | Use `build_tier_passthrough_switchyard(...)` or `load_route_bundle_table(...)`. | Launchers should not own chain internals or stats ordering. |
| Reintroducing recipes, factories, middleware bundles, or pipeline wrappers. | Put construction in profile configs and dispatch with `RouteTable`. | Those abstractions were removed to keep construction flat and profile-owned. |
| Writing a new stats collector in launcher code. | Pass one `StatsAccumulator` into the profile/route-table builder. | Live stats and `/v1/routing/stats` must read the same accumulator. |
| Implementing routing backends inside `switchyard/cli/launchers/`. | Add backend/profile logic under `switchyard/lib/` or Rust components, then expose it through route YAML. | CLI-only routing drifts from `switchyard serve`. |
| Importing heavy packages at launcher module top level. | Lazy-import inside the function that needs them. | Slim installs must still run `switchyard launch <agent> --help`. |

## Testing Launcher Changes

Run focused launcher tests plus route-bundle coverage:

```bash
uv run pytest tests/test_launch_claude.py tests/test_launch_codex.py \
  tests/test_launch_openclaw.py tests/test_launch_route_builder.py \
  tests/test_user_config.py tests/test_model_discovery.py \
  tests/test_verify.py tests/test_route_bundle.py -v -o addopts=
```

When a change touches optional dependencies or top-level CLI imports, also run
the slim-install smoke gate from `switchyard-testing-ci` and verify all three
launcher help commands.

## Related Skills

- [`switchyard-lib-core`](../switchyard-lib-core/SKILL.md) — profile-backed construction and lib-level contracts.
- [`switchyard-codebase-exploration`](../switchyard-codebase-exploration/SKILL.md) — map importers and tests before editing.
- [`switchyard-testing-ci`](../switchyard-testing-ci/SKILL.md) — choose validation gates after editing.

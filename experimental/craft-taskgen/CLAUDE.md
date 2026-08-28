# craft-taskgen

Automated CRAFT benchmark task generation pipeline.

## Code Style

This repository is intended to be made public. Follow common Python conventions:
- Prefer explicit named functions over callable parameters for branching behaviour
- Use `from __future__ import annotations` and type hints throughout
- Private helpers are `_`-prefixed but should still be readable to outside contributors

## Commands

```
uv sync                                      # Install deps (NEVER use pip)
uv run pytest tests/ -v                      # Run tests (387 tests)
uv run ruff check src/ tests/                # Lint
uv run ruff format src/ tests/               # Format
uv run ruff format --check src/ tests/       # Check format
```

All code must pass `ruff check` and `ruff format --check` before merge.

## Operational CLIs

Registered via `pyproject.toml` `[project.scripts]`:
- **`craft-taskgen`** — main pipeline (Tools dimension by default; `--dimension search` for the Search pipeline)
- **`craft-taskgen-mine`** — GitHub PR miner (needs `gh auth`)
- **`craft-taskgen-validate`** — task directory validator
- **`craft-taskgen-convert`** — Harbor task converter (see Adapters)
- **`craft-taskgen-preflight`** — env validator: repos/, disk, Harbor, Docker, auth
- **`craft-taskgen-status`** — one-screen state.json summary for monitoring
- **`craft-taskgen-split`** — balance candidate files into N shards (multi-machine runs)
- **`craft-taskgen-dashboard`** — interactive HTML dashboard from state.json (`--watch` regenerates every 10s)
- **`craft-taskgen-planning-score`** — planning-track scorer (see `src/craft_taskgen/planning/`)
- **`craft-taskgen-import`** — PR-reference importer for external benchmarks (see `src/craft_taskgen/importers/`)
- **`scripts/run-pipeline.sh`** — launcher: preflight → nohup → logged pipeline run
- **`scripts/run-baselines.sh`** — baseline launcher (claude-code/codex/opencode); see `docs/runbooks/baseline-reproducibility.md`

For end-to-end batch setup (fresh VM, slice repos, mine, launch, monitor, resume) see **`docs/runbooks/running-a-batch.md`**.

## Architecture

Pipeline: select candidates → evaluate → build task (instruction) → alignment judge → assemble artifacts (task.toml, solve.sh, Dockerfile, postmerge tests) → F2P/P2P classify → oracle check → agent smoke (Harbor; codex + GPT-5.5 by default) → triage (Opus per-test skip/keep + cross-family fairness review, severity-gated Build regen) → auto-skip loop → accept/reject (efficiency + reviewer-concern flags as soft signals for human review).

Per-step model/transport map (from the Apr 2026 refactor):

| Step | Transport | Model | Notes |
|---|---|---|---|
| evaluate | direct API (litellm) | Opus 4.6 | Binary accept/reject on PR candidates |
| build + alignment | direct API | Opus 4.6 + GPT-5.4 | Combined orchestrator — N parallel candidate loops (see below) |
| build_dockerfile | `claude -p` | Opus 4.6 | Needs Bash/Write/Edit against the repo |
| docker_classify / oracle | subprocess | — | No LLM |
| smoke | Harbor (subprocess) | codex + GPT-5.5 (default) | Agent trial produces reward + trial dir. Agent/model configurable via `smoke_agent`/`smoke_model`/`smoke_reasoning_effort` (defaults in `config.py`). codex is cross-agent from the Opus deep-dive judge. Internal label/state fields keep the historical `opus_` prefix. |
| deep dive (Opus) | direct API | Opus 4.6 | Per-test `skip` / `keep` verdict on failing reference tests. Deterministic auto-skip writes to `f2p_skip.txt` and re-scores. |
| fairness review | direct API | GPT-5.4 | Cross-family task-level severity verdict (`none`/`minor`/`major`). `major` + verbatim instruction quote + named failing test triggers one-shot Build regen. Else `reviewer_concern_flag` (soft signal). |
| task_summary | direct API | Opus 4.6 | One-line narrative from diagnostics |

**Judge models are calibrated — don't bump them casually.** The smoke *agent/model* is freely configurable (it's just the candidate being judged). The **judge** models are not: the triage severity-gating (`MAX_TRIAGE_REGENS`, the `major`+quote+test regen trigger) and the deep-dive/fairness prompts were tuned for **Opus 4.6 + GPT-5.4**. A June 2026 replay over 84 previously-fair v2b tasks (`scripts/triage-replay.py`) found GPT-5.5 returns **20–25 `major`** verdicts vs GPT-5.4's **2** on *identical* inputs (codex or Opus-4.6 trajectories alike) — a pure model-strictness shift, ~78% run-to-run stable. Opus-4.6 deep-dive proposes **0** new skips on codex trials (all `keep`). So: changing a judge model (or its prompt) requires re-running `triage-replay.py` to recalibrate thresholds/prompts before trusting the verdicts. See `docs/analyses/` for the full write-up.

**Build + alignment orchestrator detail.** Runs N=3 parallel build+align candidate loops per task (`build_n_candidates`, capped at 4). Each loop: build → alignment (`alignment_max_retries`=1 by default — single roll, no retention bias; configurable [1,5]) → on `leaked`/`narrow_tests` rebuild with cumulative feedback up to `max_build_regens_per_candidate`=2 times → re-align. Winner picked uniformly at random among passers; task rejects only if all candidates fail. Apr 25 2026 calibration on rerun-accepts-v2 cohort: default config (N=3, α=1, r=2) yielded 79/80 vs prior N=1's 73/80 at lower cost. Triage-induced Build regen is a separate single-shot path (see `steps.py::_run_triage_build_regen`).

Two-family design, separated concerns at triage: Opus returns per-test `skip`/`keep` verdicts, GPT-5.4 returns one task-level fairness-review severity. Reviewer's `severity=major` with both a verbatim instruction quote AND a named failing test triggers a one-shot Build regen; anything else sets `reviewer_concern_flag` as a soft signal (task still ships). A separate `reward==1.0 + easiness_flag` path also triggers Build regen (with prescriptive-instruction feedback asking for a more abstract rewrite); regen budget is shared with the reviewer path via `MAX_TRIAGE_REGENS=1`, and a second-pass easiness flag shelves the task as `NEEDS_FIX`. No merge logic at triage. No step has an LLM judging output from its own family.

Key modules in `src/craft_taskgen/`:
- **config.py** — PipelineProfile (TOML), PipelineContext, state models (Stage, TaskState, PipelineState). `PipelineState.run_info` has hostname + CLI args.
- **steps.py** — step implementations. Uses `_cfg.CONSTANT` for profile-controlled values.
- **pipeline.py** — entry point. Top-level `asyncio.gather(return_exceptions=True)`; a crashed task is NEEDS_FIX, the rest of the run continues.
- **prompts.py** — prompt templates + schemas (EVALUATE_SCHEMA, BUILD_SCHEMA, ALIGNMENT_SCHEMA, DEEP_DIVE_SCHEMA, REVIEWER_SCHEMA, SUMMARY_SCHEMA).
- **rubrics.py** — authoritative rubric text as string constants (H1–H6, V4 audit, anti-leakage, alignment categories). Interpolated into prompts.
- **llm_judge.py** — `async judge(prompt, schema, model)` via litellm; manual-parse + `jsonschema.validate`; one parse-retry with error feedback; 5× transient retry. Returns usage + latency for cost tracking.
- **gateway.py** — `build_gateway_env(model)` for both `claude -p` and `llm_judge.judge`. Gateway-only (no OAuth fallback).
- **claude_cli.py** — `run_claude_async` for the remaining `claude -p` call (build_dockerfile). Retry: 5s base, 30s cap, jittered.
- **runner.py** — Harbor smoke test runner
- **docker.py** — Docker build/verify helpers
- **prefilters.py** — pre-LLM candidate rejection (docs-only, CI-only, etc.)
- **task_format.py** — task directory validator
- **preflight.py / status.py / split.py** — operational CLIs
- **adapters/** / **mining/** / **search/** — subpackages
- **baselines/** — baseline-launcher helpers (output cap, reasoning-effort defaults, run-manifest schema). Driven by `scripts/run-baselines.sh`.
- **planning/** — planning-track scorer + harbor + synth + CLI (`craft-taskgen-planning-score`)
- **importers/** — external-benchmark PR-reference importer + CLI (`craft-taskgen-import`)

All tools-pipeline LLM steps are direct-API. The only remaining `claude -p` call is `build_dockerfile`. (The empty `launch/` and `v2/` directories are stale skeletons left over from earlier iterations and can be removed.)

Skills in `.claude/skills/`: `harbor-f2p-p2p-deep-dive` (interactive triage helper). Note: craft-bench's CLAUDE.md mentions `harbor-trial-deep-dive` and `task-hardness-checker` as living here — both are stale references.

External CLIs:
- **harbor-lab** — required by the triage step. `_fetch_deep_dive_context` in `steps.py` shells out to `harbor-lab errors / edits / tool-sequence / metrics` to pre-assemble context for the direct-API deep-dive judge. Resolver: `$HARBOR_LAB` → PATH → `~/Documents/vscode/harbor-lab/.venv/bin/harbor-lab` → `/data/projects/harbor-lab/.venv/bin/harbor-lab`. Preflight fails if none work.

Referenced docs (operational — load these when changing pipeline behavior):
- **docs/runbooks/running-a-batch.md** — canonical end-to-end runbook for a deterministic repo batch.
- **docs/runbooks/baseline-reproducibility.md** — reasoning-effort matrix + version-pin rationale for `scripts/run-baselines.sh`.
- **docs/reference/easiness-heuristics.md** — objective rules that flag accepted tasks as possibly "too easy" (efficiency signals from harbor-lab trajectory). Calibrated against the Apr 17 2026 run.
- **docs/reference/pipeline-state-machine.md** — stage transitions, `state.json` schema.
- **docs/runbooks/task-review-workflow.md** — manual-review procedure for `reviewer_concern_flag` / `easiness_flag` tasks.
- **docs/reference/n-parallel-calibration-apr25.md** — N-parallel build+alignment calibration that drove `build_n_candidates=3`.
- **scripts/calibrate-alignment.py**, **scripts/calibrate-deep-dive.py**, **scripts/analyze-easiness.py** — offline calibration tools for the direct-API judges and easiness flags.
- **scripts/smoke-probe.py** — fast smoke-step iteration harness. Runs only the Harbor agent trial (`runner._run_smoke_async`) against an already-built task dir, so a new smoke agent/model (e.g. codex + GPT-5.5) can be validated in minutes instead of a 2-hr full pipeline run. `--dry-run` prints the resolved harbor argv + env (catalog path, reasoning effort) with no Docker.

Research-artifact docs (don't load by default — these document one-off analyses):
- **docs/analyses/** — frozen-in-time analyses of completed runs; date-prefixed (e.g. `may02-v2b-regression-rigorous.md`). See `docs/analyses/README.md` for the index.

## When Changing Pipeline Logic

Any change to pipeline steps, stage ordering, or what each step produces must be reflected in:
1. **`README.md`** — Pipeline Steps section (step names, numbering, descriptions)
2. **`references/task-building-guide.md`** — Pipeline steps list, "Verifier debugging" step reference, and any section that describes what a step produces (e.g. "Selecting Reference Tests")
3. **`src/craft_taskgen/prompts.py`** — Check whether any prompt template references files, steps, or artifacts that changed (e.g. if a file is now generated by a different step, the prompt that previously created it needs updating)

## Key Conventions

- `from __future__ import annotations` in all files
- Profile-controlled constants accessed via `import craft_taskgen.config as _cfg` then `_cfg.CONSTANT` (NOT `from config import CONSTANT` — that creates copies that profile.apply() can't override)
- ruff, line-length 110, target py312
- HTML template strings in dashboard.py are exempt from E501
- **Gateway-only LLM**: `claude -p` is pinned to the NVIDIA gateway. `.env` must set `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL`; every profile must set `llm_step_model` to a gateway-shaped ID (e.g. `aws/anthropic/bedrock-claude-opus-4-6`). Never add an OAuth fallback path. Model is passed via `ANTHROPIC_MODEL` env, not `--model` on the CLI. Tests that shell out need the fake gateway env from `tests/conftest.py` (autouse) or must assert the `RuntimeError` guardrail explicitly.
- **Claude binary pinning**: `config._resolve_claude_cmd()` prefers `~/.local/share/claude/versions/{CC_VERSION}` over the system binary for consistent behavior across machines. Falls back to system `claude` if the pinned version isn't installed.

## Profiles

`PipelineProfile` loads from TOML, calls `.apply()` to update module-level constants. Stored in `PipelineState.profile_data` for run reproducibility.

- **`profiles/craft-tools-v4.toml`** — tools pipeline (default for `scripts/run-pipeline.sh`); tuned for 12-CPU xlarge with concurrency 10, `max_promising_per_repo=25`, output under `harbor-tasks/craft-tools-v4/`
- **`profiles/craft-search.toml`** — Search dimension

## Search Pipeline

`craft-taskgen --dimension search` routes to `src/craft_taskgen/search/` package.
Steps: extract → synthesize → validate → dedup → harbor → smoke-opus → smoke-codex → smoke-haiku → gold-review → filter → report.
Uses `SearchPipelineState` (not `PipelineState`). State file: `runs/<ts>/state.json`.
Extract/validate delegate to craft-bench via subprocess (needs `--craft-bench-dir`).
`--limit N` processes only first N input tasks (useful for testing).

Key search modules:
- **search/synthesize.py** — 3-model LLM synthesis + cross-judging via litellm
- **search/dedup.py** — Embedding cosine similarity dedup (threshold 0.65)
- **search/harbor.py** — Search task → Harbor directory conversion (bundles verifier templates)
- **search/gold_review.py** — Automated gold review via Harbor + /gold-review skill

## Harbor

Harbor pinned at `git+...@a56546f` (post-v0.13.1 `main`; includes #1826 native pi log-trim and #1840 wildcard network allowlist). Patches in `patches/harbor-agent-patches.diff`:
- base.py: raise `_truncate_output` cap 1000→8000 chars (diagnosable setup/exec logs)
- claude_code.py: version-aware skip-install (skip only when the installed agent matches the pinned version) + apt tolerance. The attribution-header / telemetry / `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` opt-outs moved out of the patch into the launchers (`scripts/run-baselines.sh` and `planning/run_plan_impl.sh`) via `--agent-env`
- codex.py: preserve full model path for NVIDIA gateway; full `[model_providers.nvidia_gateway]` stanza (`wire_api=responses`); custom model catalog (`CODEX_MODEL_CATALOG_JSON`) + `--disable unified_exec`; install hardening (version-aware skip-install, NVM retry, node16/ripgrep fallbacks)
- opencode.py: custom `nvidia`/`vllm` provider (chat-completions) via unquoted-heredoc config write, per-mode sampling env-vars (`OPENCODE_BUILD_*`/`OPENCODE_PLAN_*`), `permission=allow` for non-interactive containers, Node16-direct install (NVM bypass), version-aware skip-install
- pi.py: npm package rename `@mariozechner`→`@earendil-works`; custom `nvidia` provider via planted `~/.pi/agent/models.json`
- openhands_sdk.py: uv-bootstrap a parallel Python 3.12 when the container's `python3` is older

The host-gateway reachability for `--backend vllm` is no longer a compose patch — it's `patches/compose-overrides/host-gateway.yaml` passed via harbor's native `--extra-docker-compose` (wired into `scripts/run-baselines.sh`).

Reapply after `uv sync --reinstall-package harbor`. If patch context mismatches, apply manually (see patches/README.md).

Agent-specific job output dirs (`jobs/{agent}/`) prevent collisions when running agents in parallel.

## Adapters

`src/craft_taskgen/adapters/` is the home for Harbor task converters. Each adapter is a subpackage with a `converter.py` exposing `run_convert(...)`. Register new adapters in `adapters/cli.py::_ADAPTERS` — they become available as `craft-taskgen-convert --adapter <name>`.

Current adapters:
- **search_native** — native Search converter; builds fresh Dockerfile per task (clones repo, installs agent runtimes).

Shared search helpers (`_harbor_utils.py` in `search/`) define the canonical `task.toml`, instruction, verifier, and solve.sh layout so from-T2 and native converters stay in sync.

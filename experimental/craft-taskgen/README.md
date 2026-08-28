# craft-taskgen

> [!WARNING]
> Experimental software. Not for production use.

Generates task suites for the CRAFT benchmark along two dimensions:

- **Tools** — tool-orchestration tasks mined from merged GitHub PRs (the production track that fed the v2b cohort)
- **Search** — codebase-navigation tasks derived from implementation problems

Both pipelines evaluate with LLMs and validate with Docker. The Tools pipeline smoke-tests with a single Opus agent through Harbor; the Search pipeline uses a three-model Opus/Codex/Haiku battery for gold-review discrimination. The paper's headline metric, **Iterative Planning**, is constructed in craft-bench from Tools-track tasks — it is not produced here.

## Related repos

- **craft-bench** — evaluation harness; consumes the suites produced here as `harbor-tasks/craft-taskgen-v2b/`, `harbor-tasks/craft-search-v2c/`, etc.
- **craft-paper** — LaTeX source for the in-progress CRAFT paper.

## Where to start

Three entry paths cover almost all use:

- **Generate a few tasks for testing or demo** → [Quickstart](#quickstart)
- **Run a real bulk batch (overnight, multi-machine)** → [`docs/runbooks/running-a-batch.md`](docs/runbooks/running-a-batch.md) is the canonical runbook
- **Run baseline trials on an already-built suite** → [How to run the baselines](#how-to-run-the-baselines)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Claude CLI (`claude`) for LLM steps
- [gh](https://cli.github.com/) (GitHub CLI) for candidate mining — `brew install gh && gh auth login`
- Docker for task validation
- Harbor (optional) for oracle and agent smoke testing
- **harbor-lab** (required by triage) — Deep-dive triage shells out to `harbor-lab errors / edits / tool-sequence / metrics` on each Harbor trial. Install from the harbor-lab repo (clone + `uv sync`). Put `.venv/bin/harbor-lab` on PATH or export `HARBOR_LAB=<path>`. Verify with `craft-taskgen-preflight`.
- **`repos/{owner}/{repo}/`** — a local clone of each target repo must exist before running the pipeline. The `select` and `evaluate` steps read the repo to show the diff to the LLM, so the clone is required from the very first step regardless of which step you resume from. Clone with:
  ```bash
  git clone https://github.com/{owner}/{repo} repos/{repo}
  ```
- **craft-bench sibling checkout** — required by the Search pipeline (`--craft-bench-dir`). Clone alongside this repo (e.g. `~/projects/craft-bench/`).

## Gateway-only LLM routing

Every `claude -p` invocation is pinned to the NVIDIA LiteLLM gateway
(`ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` from `.env`). OAuth fallback
via `~/.claude` is disabled.

Enforcement (see `src/craft_taskgen/gateway.py::build_gateway_env`):

- Pipeline refuses to start if `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL`
  is missing (`RuntimeError`, not a silent fallback).
- `ANTHROPIC_MODEL` + every sub-agent alias (SONNET/OPUS/HAIKU/SUBAGENT)
  is pinned to the profile's `llm_step_model`.
- `CLAUDE_CODE_OAUTH_TOKEN` is stripped from the child process env;
  telemetry and attribution-header traffic are disabled.
- `craft-taskgen-preflight --check-endpoints` verifies the gateway
  responded by asserting the `modelUsage` key is gateway-shaped
  (e.g. `azure/anthropic/claude-opus-4-6`).

Always run `craft-taskgen-preflight --check-endpoints --profile <profile>`
before a bulk run.

### Verifying the policy end-to-end

Three layers of verification cover the gateway-only path:

1. **Preflight positive check** — `craft-taskgen-preflight --check-endpoints`
   makes a live `claude -p` call and asserts the response's `modelUsage`
   key is gateway-shaped (contains `/`). A bare alias like `opus` or
   `claude-opus-4-6` would indicate a non-gateway route and FAIL the check.

2. **Guardrail unit test** — `tests/test_pipeline.py::test_run_claude_raises_without_gateway_env`
   asserts that `run_claude` raises `RuntimeError` if `ANTHROPIC_API_KEY` /
   `ANTHROPIC_BASE_URL` are unset. Runs in CI on every change.

3. **Captive-server test (ad-hoc, definitive)** — bind a local TCP server,
   point `ANTHROPIC_BASE_URL` at it, and confirm every outbound connection
   from `claude -p` lands on the captive port with zero leaks elsewhere.
   This is the definitive proof that no fallback path exists:

   ```python
   import socket, subprocess, threading
   from craft_taskgen.gateway import build_gateway_env

   srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(10)
   port = srv.getsockname()[1]

   def loop():
       while True:
           conn, peer = srv.accept()
           print(f"hit: {peer} {conn.recv(200).split(b'\\n', 1)[0]!r}")
           conn.send(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
           conn.close()

   threading.Thread(target=loop, daemon=True).start()
   env = build_gateway_env("aws/anthropic/bedrock-claude-opus-4-6")
   env["ANTHROPIC_API_KEY"] = "sk-fake"
   env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
   subprocess.run(["claude", "-p", "hi"], env=env, timeout=15)
   ```

   Expected: every log line hits `127.0.0.1`. Zero hits to `api.anthropic.com`
   or any other host proves there is no fallback.

## Quickstart

```bash
cp .env.example .env       # Fill in API credentials
uv sync                    # Install dependencies (incl. dev tooling)

# (Optional) Patch Harbor agents (used in smoke + triage steps)
# Re-run any time the venv is recreated (uv sync rebuilds it on Python interpreter changes)
pushd .venv/lib/python*/site-packages && patch -p1 < ../../../../patches/harbor-agent-patches.diff && popd

# Run with a profile (pauses after evaluate for review)
craft-taskgen \
    --profile profiles/craft-tools-v4.toml \
    --candidates candidates/*.json

# Run fully unattended — skip the checkpoint after evaluate
craft-taskgen \
    --profile profiles/craft-tools-v4.toml \
    --candidates candidates/*.json --no-checkpoint

# Resume a run (--no-checkpoint skips the pause on resume too)
craft-taskgen --resume harbor-tasks/craft-tools-v4/runs/2026-04-09-120000/state.json \
    --from-step build --no-checkpoint

# Dashboard
craft-taskgen-dashboard harbor-tasks/craft-tools-v4/runs/2026-04-09-120000/state.json

# Validate a task directory
craft-taskgen-validate templates/t2v3-CE0266-celery-retry-unification/
```

## Mining candidates

The pipeline takes candidate JSONs as input. Generate them by mining merged GitHub PRs via the `gh` CLI:

```bash
# Mine all repos from repo_list.csv (primary workflow — 111 repos for overnight runs)
craft-taskgen-mine --repos-csv references/repo_list.csv \
    --repos-dir repos/ \
    --out candidates/ --top 50 --after 2025-09-01

# Or the smaller craft-repos.csv (48 repos, original set)
craft-taskgen-mine --repos-csv references/craft-repos.csv \
    --repos-dir repos/ \
    --out candidates/ --top 20 --after 2025-09-01

# Mine a single repo by path (github_repo derived from git remote)
craft-taskgen-mine /path/to/cloned/scrapy --top 20 --out candidates/scrapy.json

# Mine a single repo by owner/repo slug (cloned automatically)
craft-taskgen-mine scrapy/scrapy --top 20 --out candidates/scrapy.json
```

The `--repos-csv` mode reads the repo list, finds each repo in `--repos-dir`, and writes one candidate JSON per repo to the output directory. Repos not found locally are cloned automatically.

The miner fetches merged PRs from the GitHub API and scores them by structural heuristics (multi-file changes, test coverage, line counts, iteration clusters) — no LLM calls, stdlib only. Diffs are computed against the true merge-base so scores aren't inflated for PRs where the author rebased main mid-review.

`--after YYYY-MM-DD` limits to PRs merged on or after that date.

See `src/craft_taskgen/miner.py` docstring for the full output format and scoring breakdown.

### Import PR-Reference Maps

For curated PR lists (for example `{task_id: {...metadata...}}`), use the
importer to convert them into miner-compatible candidate files without
re-mining all GitHub PR history:

```json
{
  "craft-click-17b8cad6": {
    "pr_url": "https://github.com/pallets/click/pull/3152",
    "repo": "pallets/click",
    "task_type": "bug_fix",
    "matched_practice": "systematic-debugging"
  }
}
```

```bash
craft-taskgen-import \
    --input task_map.json \
    --repos-dir repos/ \
    --out-dir candidates/pr-refs/
```

Accepted input formats: `.jsonl`, `.json`, `.csv`, `.tsv`.

The importer extracts `(owner/repo, pr_number)` from fields like:
- `pull_request` / `pull_request_url` / `pr_url` (GitHub pull URL)
- `repo`/`github_repo` + `pr_number`/`pull_number`
- `instance_id` suffix fallback (e.g. `sympy__sympy-27690`) when repo is present

This importer preserves per-record metadata on each emitted candidate under:
- `source_task_id`
- `source_task_type`
- `source_matched_practice`
- `source_metadata`

Output files land in `candidates/pr-refs/*.json` and can be fed directly to:

```bash
craft-taskgen --candidates 'candidates/pr-refs/*.json' --profile profiles/craft-tools-v4.toml
```

### Import SWE-bench Pro JSONL

For a SWE-bench Pro-style dataset with fields like `repo`, `base_commit`,
`patch`, and `test_patch`, import it into the same candidate JSON format:

```bash
craft-taskgen-import \
    --format swebench-pro \
    --input swebench_pro.jsonl \
    --out-dir candidates/swebench-pro/
```

Then feed the generated candidates into the normal pipeline entrypoint:

```bash
craft-taskgen --candidates 'candidates/swebench-pro/*.json'
```

This importer emits miner-compatible candidate JSON for selection and
evaluation. Later git-dependent pipeline stages are not yet supported for
dataset-only inputs.


## Pipeline Steps

1. **Select** — Pick top candidates from mining output (score > 0, has tests)
2. **Evaluate** — Opus (direct API) classifies each PR: `accept` or `reject` (binary; no middle-ground `MAYBE` band)
3. **Build + Alignment (parallel candidates)** — N=3 independent build+alignment loops per task. Each loop: Opus drafts the task instruction (direct API; H1-H7 rules enforced at construction), then GPT-5.4 (cross-family) audits instruction ↔ reference-test alignment + leakage on a single roll (`alignment_max_retries=1` — no retention bias). On `leaked`/`narrow_tests` the loop rebuilds with cumulative feedback up to `max_build_regens_per_candidate=2` times. The orchestrator picks one passing candidate uniformly at random; tasks reject only if no candidate passes. Configurable via `build_n_candidates` (default 3, capped at 4), `alignment_max_retries` (default 1, capped [1,5]), and `max_build_regens_per_candidate` (default 2, capped [0,3]). Addresses build-step instructional non-determinism that drove run-to-run alignment churn (see `docs/reference/n-parallel-calibration-apr25.md`).
4. **Assemble artifacts** — Mechanically generates task.toml, solve.sh, and extracts postmerge test files
5. **Build Dockerfile** — Claude creates environment/Dockerfile
6. **F2P/P2P classify** — Two Docker pytest passes (overlay → oracle) produce per-test F2P/P2P lists
7. **Oracle check** — Apply solve.sh and run score.py; hard gate that blocks pipeline if task doesn't resolve
8. **Smoke** — Run a coding agent on the task via Harbor to produce the reward (primary quality gate). Agent/model are configurable (`smoke_agent`/`smoke_model`/`smoke_reasoning_effort`); the default is codex + GPT-5.5, cross-agent from the Opus deep-dive judge. Iterate on this step in isolation with `scripts/smoke-probe.py`.
9. **Triage** — Two parallel judges answering different questions:
    - **Opus deep-dive** (direct API) classifies each failing reference test as `skip` (exclude from scoring, auto-appended to `f2p_skip.txt`) or `keep` (counted as a genuine capability gap). Reward re-scored against the updated skip set; if reward=1.0 the task accepts.
    - **Fairness review** (direct API, GPT-5.4 — cross-family) returns one severity verdict (`none`/`minor`/`major`) at the task level. `severity=major` with both a verbatim instruction quote AND a named failing test triggers a one-shot Build regen (bounded by `MAX_TRIAGE_REGENS`). Anything else sets `reviewer_concern_flag` as a soft signal for human review; the task still proceeds on Opus's per-test verdict.
    No merge logic — the two judges answer different questions so there is nothing to reconcile.
10. **Accept / reject** — Triage advances. `reviewer_concern_flag` is a soft signal on accepted tasks that surfaces for human review without blocking. `easiness_flag` is gated: first occurrence triggers a Build regen with prescriptive-instruction feedback (shared regen budget with the reviewer path); a second-pass easiness flag shelves as `NEEDS_FIX` rather than accepting. See `docs/reference/easiness-heuristics.md`.

## Output layout

A successful run produces one task directory per accepted candidate plus a single state file:

```
harbor-tasks/craft-tools-v4/
├── runs/<YYYY-MM-DD-HHMMSS>/
│   └── state.json                  # incremental run state (resume target)
└── <task-id>/                      # one dir per accepted task
    ├── task.toml                   # name, instruction, agent config, memory_mb
    ├── solve.sh                    # reference solution (oracle replays this)
    ├── Dockerfile                  # built environment (deps + repo at base_sha)
    ├── tests/                      # F2P + P2P reference tests
    ├── f2p_skip.txt                # tests excluded by Opus triage (skip verdicts)
    └── diagnostics.json            # smoke-test reward, easiness flags, reviewer notes
```

This is the layout craft-bench's Harbor consumes. Suite names that ship to craft-bench drift from profile names (e.g. the `craft-tools-v4` profile produced the `craft-taskgen-v2b` cohort).

## Candidates Format

The `--candidates` argument takes one or more JSON files (one per repo), each produced by the miner. Each file has the following structure:

```json
{
  "repo": "{repo}",
  "github_repo": "{owner}/{repo}",
  "after": "2025-09-01",
  "n_prs_scanned": 207,
  "n_candidates": 1,
  "candidates": [
    {
      "sha": "a1b2c3d4...",
      "base_sha": "e5f6a7b8...",
      "subject": "Add support for streaming responses",
      "author": "username",
      "date": "2025-10-01T12:00:00Z",
      "source_files": ["src/client.py", "src/utils.py"],
      "test_files": ["tests/test_client.py"],
      "other_files": ["docs/api.rst"],
      "source_lines_changed": 47,
      "test_lines_changed": 23,
      "packages_touched": 1,
      "package_names": ["{repo}"],
      "has_test_patch": true,
      "is_multi_file": true,
      "is_multi_package": false,
      "is_nontrivial_source": true,
      "is_nontrivial_tests": true,
      "is_refactoring": false,
      "has_iteration_signal": false,
      "score": 4.5,
      "score_breakdown": {
        "multi_file_5plus": 3.0,
        "source_100plus_lines": 1.5
      }
    }
  ]
}
```

**Required fields per candidate:**
- `sha` — merge commit SHA (the commit that merged the PR into the base branch)
- `base_sha` — base branch HEAD SHA at the time the PR was merged (`pr["base"]["sha"]` from the GitHub PR API)
- `score` — relevance score from the miner (candidates with `score <= 0` are skipped)
- `has_test_patch` — must be `true` (candidates without test changes are dropped)

## Profiles

Pipeline parameters (models, thresholds, tuning) are configured via TOML profiles. The profile is recorded in the run's state.json for reproducibility.

```bash
craft-taskgen --profile profiles/craft-tools-v4.toml ...
```

Available profiles:

- `profiles/craft-tools-v4.toml` — Tools pipeline; tuned for a 12-CPU xlarge (concurrency 10, `max_promising_per_repo=25`, output under `harbor-tasks/craft-tools-v4/`). Note: the profile generation (`v4`) and the consumer cohort name (`craft-taskgen-v2b` in craft-bench) drift on purpose — the v2b cohort was *built* with the v4 profile. Don't conflate them.
- `profiles/craft-search.toml` — Search dimension

## Running a long, unattended pipeline

> **Canonical runbook**: [`docs/runbooks/running-a-batch.md`](docs/runbooks/running-a-batch.md) — deterministic seeded shuffle, multi-machine sharding, resume protocol, monitoring. Follow it for any real batch.
>
> The section below is the abbreviated 5-step version, kept here for quick reference.

Full unattended run across all repos in `references/repo_list.csv`. Expected output: a few dozen accepted tasks in `harbor-tasks/craft-tools-v4/`.

### 1. Pre-flight

`craft-taskgen-preflight` runs every check a bulk run needs — CLI auth, Docker daemon + memory, `.env` API keys, disk space, repo clones, optional repos CSV + GitHub API rate limit.

```bash
uv run craft-taskgen-preflight --repos-csv references/repo_list.csv
```

Fix any FAIL lines before proceeding. WARN lines (e.g. low Docker memory) are advisory.

### 2. Mine candidates

Fetches top-N merged PRs per repo and scores them by structural heuristics. Missing repos are auto-cloned. Expect 10–20 min for 100+ repos.

```bash
craft-taskgen-mine \
    --repos-csv references/repo_list.csv \
    --repos-dir repos/ \
    --out candidates/overnight/ \
    --top 50 \
    --after 2025-09-01
```

### 3. Launch

`scripts/run-pipeline.sh` runs preflight, then `nohup`s the pipeline. Default profile is `profiles/craft-tools-v4.toml` (concurrency 10, output under `harbor-tasks/craft-tools-v4/`). Log is `runs/<ts>-<host>.log`.

```bash
scripts/run-pipeline.sh 'candidates/overnight/*.json'
```

The main task gather uses `asyncio.gather(return_exceptions=True)` — one crashed task gets marked `NEEDS_FIX` and the rest of the run continues.

### Resuming a stopped run

`state.json` is written incrementally, so it's always safe to kill the pipeline. To resume:

```bash
# 1. Kill the process (PID printed at launch in runs/*.log)
kill <PID>

# 2. Optionally pull latest
git pull origin main

# 3. Resume — preflight still runs (auth, disk, docker) but skips candidate selection
SHARD_LABEL=<your-shard> \
RESUME=harbor-tasks/craft-tools-v4/runs/<ts>/state.json \
scripts/run-pipeline.sh
```

### 4. Monitor

```bash
# One-screen summary of stage counts, in-progress, NEEDS_FIX, accepted
uv run craft-taskgen-status <state.json>

# Interactive HTML dashboard (regenerate every 10s with --watch)
uv run craft-taskgen-dashboard <state.json> --watch &
```

The state file path is `<task_suite_dir>/runs/<YYYY-MM-DD-HHMMSS>/state.json` (for the default profile: `harbor-tasks/craft-tools-v4/runs/<ts>/state.json`). The path is printed at pipeline startup and visible in the log.

### 5. Resume

```bash
uv run craft-taskgen --resume <state.json> --no-checkpoint
# Or restart from a specific step:
uv run craft-taskgen --resume <state.json> --from-step build --no-checkpoint
```

Valid `--from-step` values for the tools dimension: `select`, `evaluate`, `build`, `alignment`, `assemble_artifacts`, `build_dockerfile`, `docker_classify`, `oracle`, `smoke`, `opus_triage`, `report`.

### Splitting across machines

For large runs across multiple machines, use `craft-taskgen-split` to partition candidate files into balanced shards (by total candidate count):

```bash
# On a coordinator, compute balanced shards:
uv run craft-taskgen-split 3 'candidates/overnight/*.json'
# shard 1 (142 cands, 6 files): candidates/overnight/fastapi.json candidates/overnight/scrapy.json ...
# shard 2 (138 cands, 7 files): candidates/overnight/pydantic.json candidates/overnight/starlette.json ...
# shard 3 (135 cands, 8 files): candidates/overnight/rich.json candidates/overnight/textual.json ...

# On each machine, with that machine's file list + a label for provenance:
SHARD_LABEL=east-1 scripts/run-pipeline.sh 'candidates/overnight/fastapi.json candidates/overnight/scrapy.json ...'
```

Each machine records its hostname, user, full CLI args, and candidate file list into `state.run_info` so you can tell which machine produced which accepted tasks. `craft-taskgen-status` surfaces this metadata.

## How to run the baselines

`scripts/run-baselines.sh` runs one agent + one model against a task suite via Harbor, and emits a reproducibility manifest alongside Harbor's `result.json`. One process per (agent, model) — to compare configurations, launch multiple invocations.

### Prerequisites (one-time)

1. `.env` populated (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`).
2. Docker daemon running with ≥8 GB available to containers (preflight enforces a `memory_mb >= 8192` floor on every `task.toml`).
3. A built task suite directory (e.g. `harbor-tasks/craft-taskgen-v2b/`).

### Quickstart

Gateway-routed baseline against a task suite:

```bash
scripts/run-baselines.sh \
    --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
    --agent opencode \
    --model aws/anthropic/bedrock-claude-haiku-4-5 \
    --n-tasks 3
```

The script runs preflight, picks the right `reasoning_effort` for the (agent, model) pair from `src/craft_taskgen/baselines/reasoning_defaults.py`, applies the agent-specific wiring needed to actually deliver that effort (different agents drop reasoning silently in different ways — see `docs/runbooks/baseline-reproducibility.md`), nohups Harbor, and prints `PID`, `log`, and `kill` hints. Output lands under `baselines/<ts>/`.

### Common variations

```bash
# Local vLLM endpoint (same launcher; --backend vllm + VLLM_BASE_URL in env)
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
    scripts/run-baselines.sh \
        --tasks-dir <suite> --agent opencode \
        --model qwen/Qwen3.5-397B-A17B-FP8 --backend vllm

# Sanity check — apply the reference solution and run the verifier (no LLM)
scripts/run-baselines.sh --tasks-dir <suite> --agent oracle

# Sanity check — do nothing, run the verifier (expected fail; floor)
scripts/run-baselines.sh --tasks-dir <suite> --agent nop

# Inspect the harbor invocation without running it
scripts/run-baselines.sh --tasks-dir <suite> --agent codex --model <id> --dry-run

# Scope to a glob; rebuild the Docker image (after Dockerfile edits)
scripts/run-baselines.sh --tasks-dir <suite> --agent claude-code --model <id> \
    --task-name 'craft-click-*' --force-build
```

Supported agents: `claude-code`, `codex`, `opencode`, `openhands-sdk`, `qwen-coder`, `pi` (LLM-driven) plus `oracle` and `nop` (sanity checks). Run `scripts/run-baselines.sh --help` for the full flag list. The `qwen-coder` and `pi` integrations are documented in [`docs/runbooks/baseline-reproducibility.md`](docs/runbooks/baseline-reproducibility.md) — both require harbor patches applied (`patches/harbor-agent-patches.diff`).

### Reading the output

Each run writes:

- **`baselines/<ts>/<job-name>.log`** — Harbor's stdout (`tail -f` to monitor).
- **`baselines/<ts>/<job-name>/run_manifest.json`** — pinned versions, effort, output-cap, sampling overrides, gateway URLs, `harbor_rc`. Written before Harbor launches and finalized after it exits.
- **`baselines/<ts>/<job-name>/result.json`** — Harbor's per-trial scores.

To summarize a finished run see `scripts/summarize_baseline.py` (tools track) or `scripts/summarize_search_baseline.py` (search track), both of which support `--csv` for per-trial output.

For the reasoning-effort matrix and the version-pin rationale (Claude Code 2.1.118), see [`docs/runbooks/baseline-reproducibility.md`](docs/runbooks/baseline-reproducibility.md).

### Published results

Live results are surfaced on the internal GitLab Pages site:

- **Leaderboard** — K=5 End-to-End numbers with CIs (claude-code, codex, opencode + a few model picks).
- **Planner × Implementer matrix** — K=1, n=92, lift in pp vs no-plan baseline.
- **Agent × Open-Model matrix** — K=1, n=92. Same 92 tasks across {opencode, openhands-sdk, qwen-coder, pi} × {Qwen-3.6-35B-A3B, GLM-5.1, Nemotron-3 Ultra (EA), Claude Haiku 4.5}. Rows are open models; columns are agents with pinned versions. Cells are Resolved%. Data: [`site/data/harness_x_model.json`](site/data/harness_x_model.json).

## Search Pipeline

Derives codebase-navigation tasks from implementation problems via 3-model LLM synthesis. Requires a sibling craft-bench checkout — pass via `--craft-bench-dir`.

```bash
craft-taskgen \
    --dimension search \
    --tasks-dir harbor-tasks/craft-tools-v4 \
    --repos-dir repos \
    --output-dir gold/craft-search \
    --profile profiles/craft-search.toml \
    --limit 2  # for testing

# Resume
craft-taskgen --resume gold/craft-search/runs/<ts>/state.json --from-step smoke-opus
```

### Search Pipeline Steps

1. **Extract** — Parse input tasks, mine repo maps at pre-change commits
2. **Synthesize** — 3-model LLM synthesis + cross-judging (Sonnet, Gemini, GPT-5.4)
3. **Validate** — Gold answer AST validation + alt_function expansion
4. **Dedup** — Embedding cosine similarity dedup (threshold 0.65)
5. **Harbor** — Convert to Harbor task directories with search verifier
6. **Smoke-All** — 3-tier agent evaluation (Opus + Codex + Haiku) in a single step
7. **Gold Review** — Automated gold review via Harbor + /gold-review skill
8. **Filter** — 3-model criteria: both_low, haiku_inversion, flat_easy, no_gold_functions
9. **Report** — Summary with 3-tier monotonicity check

## Adapters

`src/craft_taskgen/adapters/` holds Harbor task converters for each synthesis source. Use `craft-taskgen-convert --adapter <name>` to run a converter. Adapters share the task.toml, instruction, verifier, and solve.sh writers in `search/_harbor_utils.py` so output stays consistent.

### search-native

Builds Harbor task directories from native Search TaskCandidate JSON files (e.g. produced by craft-bench's `synthesize_tasks.py` → `judge_search_tasks.py` → `aggregate_and_dedup.py`). Generates a fresh Dockerfile per task that clones the target repo at a pinned commit and installs agent runtimes.

```bash
craft-taskgen-convert \
    --adapter search-native \
    --candidates-dir tasks/candidates/search/ \
    --manifest repos/manifest.json \
    --output-dir harbor-tasks/craft-search/ \
    --limit 10
```

- `--candidates-dir` — root dir with per-repo candidate subdirs: `{repo}/{uuid}.json`
- `--manifest` — `repos/manifest.json` mapping `{repo_name: {url, commit}}`
- `--limit N` — stop after N tasks (0 = all, useful for testing)

Stale task dirs (those not in the current candidate set) are automatically moved to `_stale_<timestamp>/` inside the output directory.

### Registering a new adapter

1. Create `src/craft_taskgen/adapters/<name>/converter.py` with a `run_convert(...)` function.
2. Add an `AdapterSpec` entry to `_ADAPTERS` in `adapters/cli.py` wiring up its CLI args and entry point.
3. Reuse the writers in `search/_harbor_utils.py` (`write_task_toml`, `write_instruction`, `write_search_verifier`, `write_gold_answer`, `write_solve_sh`, `write_registry`, `task_id`) so the Harbor output layout stays in sync with other adapters.

## Development

```bash
uv run pytest tests/ -v                    # Run tests
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

## Contributing

See ROADMAP.md for planned improvements and known issues.

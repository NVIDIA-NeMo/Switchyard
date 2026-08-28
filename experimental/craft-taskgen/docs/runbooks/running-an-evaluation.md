# Running an evaluation on CRAFT

How to take an existing CRAFT task suite (the 92-task `craft-taskgen-v2b`
release, or any other pre-built suite under `craft-bench/harbor-tasks/`) and
score one or more agent+model configurations against it. This is the
evaluation path. If you instead want to **generate new tasks**, read
[`running-a-batch.md`](running-a-batch.md).

If you only need the headline numbers and don't want to reproduce them
yourself, see
`craft-bench/planning/results.md`.

## Prerequisites

You need a Linux/macOS host with:

| | What | Why |
|---|---|---|
| **Docker** | Daemon running. 50 GB+ free on the Docker root, 16 GB+ RAM for the container fleet. ZFS-backed `/scratch/docker` recommended for repeated runs (image caching). | Each task runs in its own container; Harbor builds per-task images. |
| **Harbor + harbor-lab** | Both auto-installed by `uv sync` in step 01. Harbor is pinned (`46bb68c`); harbor-lab provides the post-run triage CLIs (`errors`, `tool-sequence`, etc.). | Harbor walks the dataset and runs the trials; harbor-lab is what you use to dig into failed trials. |
| **Inference gateway creds** | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` in `.env`. Base URLs point at your OpenAI-compatible inference gateway. Optional `VLLM_BASE_URL`/`VLLM_API_KEY` for local serving. | Every LLM call routes through the gateway — there is no OAuth fallback. |
| **`gh auth status` OK** | One-time `gh auth login` if you've never used the GitHub CLI on this host. | The mining step (only needed when you build tasks from scratch) uses it; preflight checks it regardless. |

## 01 · Clone and install

Two repos. `craft-taskgen` has the pipeline + launcher; `craft-bench` has
the task suites.

```bash
cd ~/projects   # or wherever you stage source trees
git clone <craft-taskgen-repo-url>
git clone <craft-bench-repo-url>
cd craft-taskgen
uv sync

# Reapply Harbor patches (uv sync wipes them out of .venv/.../site-packages/harbor/)
pushd .venv/lib/python*/site-packages
patch -p1 < ../../../../patches/harbor-agent-patches.diff
popd
```

If the glob doesn't match exactly one directory, fix the path manually
(`.venv/lib/python3.13/site-packages`). Silent glob misses are the most
common source of "Harbor smoke behaves weirdly" later.

## 02 · Add inference gateway creds and run preflight

```bash
# .env at the repo root
cat > .env <<'EOF'
ANTHROPIC_API_KEY=<your-key>
ANTHROPIC_BASE_URL=<your-gateway-base-url>
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<your-gateway-base-url>
EOF

uv run craft-taskgen-preflight \
  --profile profiles/craft-tools-v4.toml \
  --check-endpoints
```

Preflight catches the common failure modes — missing `.env`, wrong Docker
daemon, no `gh auth`, harbor-lab not on `PATH`, etc. `--check-endpoints`
makes a few cents of live gateway calls but is worth it before you commit
hours of unattended work.

## 03 · Smoke-test on a single task

Confirms your environment end-to-end before you commit GPU-hours. Should
finish in a few minutes; artifacts land in `baselines/<timestamp>/`.

```bash
scripts/run-baselines.sh \
  --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
  --agent claude-code \
  --model aws/anthropic/bedrock-claude-opus-4-7 \
  --backend gateway \
  --n-tasks 1
```

`run-baselines.sh` runs preflight, nohup's the harbor invocation so it
survives terminal exit, then prints the PID, log path, and kill hint. Tail
the log; if the trial completes and a `verifier.json` lands under
`baselines/<ts>/<task-id>/`, your env is good.

## 04 · Full 92-task baseline run

Same launcher, no `--n-tasks` cap. K=1 cell at `--n-concurrent 6` takes
roughly 6 hours of wall time on a `craftbench02`-class host. K=5 (the
canonical leaderboard setting) is 5×.

Pick your agent. The flags below match the canonical `craft-taskgen-v2b`
configurations in `results.md`.

### claude-code · Opus 4.7

```bash
scripts/run-baselines.sh \
  --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
  --agent claude-code \
  --model aws/anthropic/bedrock-claude-opus-4-7 \
  --backend gateway \
  --n-concurrent 6
```

Claude Code CLI pinned at **2.1.118** (clean of the April-23 postmortem
regressions). Reasoning effort `high` applied automatically by the
launcher — gateway caps `xhigh` on Opus 4.7 until harbor's `CliFlag`
validator lifts the limit. See
[`baseline-reproducibility.md`](baseline-reproducibility.md) for the full
effort table.

### codex · GPT-5.5

```bash
scripts/run-baselines.sh \
  --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
  --agent codex \
  --model openai/openai/gpt-5.5 \
  --backend gateway \
  --n-concurrent 6
```

Codex CLI pinned at **0.121.0**. Reasoning effort `high` applied by the
launcher (Harbor default is `high` for our smoke tasks).

### opencode · Haiku 4.5

```bash
scripts/run-baselines.sh \
  --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
  --agent opencode \
  --model aws/anthropic/claude-haiku-4-5-v1 \
  --backend gateway \
  --n-concurrent 6
```

opencode CLI pinned at **1.4.9**. The launcher prepends `nvidia/` to the
model slug automatically when `--agent opencode --backend gateway` — that
prefix selects the harbor-internal `nvidia` provider (which routes through
`@ai-sdk/openai-compatible`); the gateway only sees the canonical slug on
the wire.

### opencode · GLM-5.1

```bash
scripts/run-baselines.sh \
  --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
  --agent opencode \
  --model nvidia/zai-org/glm-5.1 \
  --backend gateway \
  --n-concurrent 6
```

GLM is not effort-capable; the launcher passes no `reasoning_effort` and
harbor uses the model's default sampling.

### opencode · Qwen-3.6 (local vLLM)

```bash
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY \
  scripts/run-baselines.sh \
    --tasks-dir ~/projects/craft-bench/harbor-tasks/craft-taskgen-v2b \
    --agent opencode \
    --model qwen/Qwen3.6-35B-A3B \
    --backend vllm \
    --n-concurrent 6
```

vLLM-served Qwen needs `--served-model-name qwen-3.6` (or
`OPENCODE_BUILD_TEMPERATURE` etc.) to trip the launcher's Qwen-sampling
branch — otherwise vLLM defaults (T=1.0, top_p=1.0) are used. See the
"Sampling caveat — qwen runs" section in `results.md`.

## 05 · Score and inspect

Each per-trial directory under `baselines/<ts>/<task-id>/` ships:

| File | What it contains |
|---|---|
| `verifier.json` | `resolved` (bool), F2P, P2P, per-test pass/fail. The authoritative score for the trial. |
| `agent.log` | Full trajectory output (Claude Code) or `codex.txt` / `opencode.txt` for the other agents. |
| `agent/sessions/*.jsonl` | Structured tool-call stream (Claude Code only). |
| `run_manifest.json` | Hostname, harbor SHA, agent kwargs, env, output cap, reasoning effort, dataset digest. The reproducibility record. |

Aggregate scores for a whole run:

```bash
uv run craft-taskgen-status \
  --state baselines/<timestamp>/state.json
```

Deep-dive a single failed trial:

```bash
harbor-lab errors        baselines/<timestamp>/<task-id>/
harbor-lab tool-sequence baselines/<timestamp>/<task-id>/
harbor-lab edits         baselines/<timestamp>/<task-id>/
harbor-lab metrics       baselines/<timestamp>/<task-id>/
```

For full triage workflows (including the `harbor-f2p-p2p-deep-dive` skill
and the per-test skip/keep judging path), see
`harbor-f2p-p2p-deep-dive` in this repo's
`.claude/skills/`.

## 06 · Iterating

- **Add a new model.** Add a row in
  `src/craft_taskgen/baselines/reasoning_defaults.py`, run a single-task
  smoke against it (step 03), then commit.
- **K > 1.** The launcher does **not** loop trials — Harbor runs K=1 per
  invocation. For K=5, run the same command 5× with distinct `--output-dir`
  values, then aggregate.
- **Subset of tasks.** Use `--task-name 'aiogram-*'` (include glob) or
  `--exclude-task-name 'scrapy-*'` (exclude glob).
- **Rebuild image cache.** Pass `--force-build` after editing the task's
  Dockerfile or agent-version pin.

## Common failure modes

- **`gateway 401 / 403`** — `.env` keys wrong or expired. Re-pull from
  another working VM or ask in `#craft-dev`.
- **`harbor-lab not on PATH`** — preflight will say this. Either set
  `HARBOR_LAB=…` or symlink the binary to a directory on `PATH`.
- **`Docker compose v5 rejects __ in image tags`** — your harbor patch
  didn't apply. Re-run the patch from step 01 and confirm
  `models/trial/config.py` has the single-dash trial-name fix.
- **Trial silently disabled reasoning** — the
  [`baseline-reproducibility.md`](baseline-reproducibility.md) document
  describes how this happens silently on each agent. Inspect the trial's
  `run_manifest.json` for `agent.kwargs.reasoning_effort` and
  `agent.env.OPENCODE_REASONING_EFFORT` to verify.

## Related docs

- [`baseline-reproducibility.md`](baseline-reproducibility.md) — per-agent
  reasoning-effort defaults and version pins. Read this before changing any
  model.
- [`running-a-batch.md`](running-a-batch.md) — how to **generate new
  tasks**, end to end. Use when you need a new task suite, not when you're
  evaluating an existing one.
- [`task-review-workflow.md`](task-review-workflow.md) — how to handle
  `reviewer_concern_flag` / `easiness_flag` tasks after generation.
- `craft-bench/planning/results.md`
  — canonical CRAFT scores. Source of truth for the leaderboard on this
  site.

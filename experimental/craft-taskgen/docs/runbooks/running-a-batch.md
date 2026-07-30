# Running a batch

End-to-end runbook for launching one deterministic repo batch through the
tools pipeline. Replaces / supersedes GitLab snippet 13878.

A "batch" is 50 repos drawn from
`~/projects/craft-bench/docs/plans/expanded-repos/candidates.csv` via
deterministic seeded shuffle (seed 42). Batches are disjoint by index — batch 0
is repos 1–50, batch 1 is 51–100, etc. Same seed + same candidates file on
every machine ⇒ shards never overlap.

## One-time setup on a fresh VM

1. **Clone and sync.**

   ```bash
   cd ~/projects   # or wherever you stage source trees
   git clone <craft-taskgen-repo-url>
   cd craft-taskgen
   uv sync
   ```

2. **Re-apply Harbor agent patches** (required — `uv sync` wipes
   `.venv/lib/.../site-packages/harbor/`):

   ```bash
   pushd .venv/lib/python*/site-packages
   patch -p1 < ../../../../patches/harbor-agent-patches.diff
   popd
   ```

   If the glob doesn't match exactly one directory, fix the path manually
   (e.g. `.venv/lib/python3.13/site-packages`). Silent glob misses are the
   most common source of "Harbor smoke behaves weirdly" later on.

3. **`.env`** must set the NVIDIA gateway keys — every LLM step is
   gateway-only, no OAuth fallback.

   ```bash
   # .env (at repo root)
   ANTHROPIC_API_KEY=<gateway key>
   ANTHROPIC_BASE_URL=<gateway base url>
   OPENAI_API_KEY=<gateway key>
   OPENAI_BASE_URL=<gateway base url>
   ```

   Copy from another working VM, or ask in #craft-dev.

4. **Run preflight** — catches missing `.env`, wrong Docker daemon, no `gh
   auth`, harbor-lab not on PATH, etc. `--check-endpoints` makes a few
   cents of live gateway calls but is worth it before committing to hours of
   unattended work.

   ```bash
   uv run craft-taskgen-preflight \
       --profile profiles/craft-tools-v4.toml \
       --check-endpoints
   ```

5. **Check Docker storage.** ZFS-backed hosts (craftbench02, craftbench03)
   have 300–400 GB on `/scratch/docker`; non-ZFS hosts may have as little as
   25 GB. Confirm before launching:

   ```bash
   docker info | grep -E "Storage Driver|Docker Root Dir"
   df -h $(docker info 2>/dev/null | awk '/Docker Root Dir/ {print $NF}')
   ```

   Under 50 GB free is a red flag — prune before starting (`docker system
   prune -af --volumes` if nothing else is running).

## Running batch N

1. **Pick the batch number.** Batches already run are tracked informally;
   ask or check `grep -lr "SHARD_LABEL" runs/` on the machine last running a
   bulk batch. Pick the next unused integer.

   ```bash
   BATCH_N=5    # <-- set me
   ```

2. **Slice the batch.**

   ```bash
   uv run python scripts/slice_repo_batch.py \
       ~/projects/craft-bench/docs/plans/expanded-repos/candidates.csv \
       --batch "$BATCH_N" \
       --batch-size 50 \
       --seed 42 \
       --out candidates/batch_${BATCH_N}.csv

   head -3 candidates/batch_${BATCH_N}.csv   # sanity-check
   ```

   `--seed 42` and `--batch-size 50` must match every prior run so slices stay
   disjoint.

3. **Mine PR candidates.** Clones any missing repos under `repos/` (~1–2
   GB/repo) and writes one `candidates/<repo>.json` per repo.

   ```bash
   uv run craft-taskgen-mine \
       --repos-csv candidates/batch_${BATCH_N}.csv \
       --out candidates/ \
       --top 50
   ```

4. **Launch the pipeline.**

   ```bash
   # Move any stale candidate JSONs out of candidates/ first, or the glob
   # below will pick them up. Only batch-N candidates should remain.
   ls candidates/*.json | head

   PROFILE=profiles/craft-tools-v4.toml \
       MAX_EVALUATE=2500 \
       TOP_PER_REPO=50 \
       SHARD_LABEL=batch-${BATCH_N} \
       nohup scripts/run-pipeline.sh 'candidates/*.json' \
       > /tmp/pipeline-batch-${BATCH_N}.log 2>&1 &
   ```

   `run-pipeline.sh` runs preflight first, `nohup`s the pipeline so it
   survives SSH disconnect, and prints the state.json path
   (`harbor-tasks/craft-tools-v4/runs/<ts>/state.json`) in its initial log
   output.

## Monitoring

```bash
# Nohup log tail
tail -f /tmp/pipeline-batch-${BATCH_N}.log

# One-screen summary
uv run craft-taskgen-status harbor-tasks/craft-tools-v4/runs/<ts>/state.json

# HTML dashboard (regenerates every 10s) + simple HTTP server
RUN_DIR=harbor-tasks/craft-tools-v4/runs/<ts>
nohup uv run craft-taskgen-dashboard "$RUN_DIR/state.json" --watch \
    --output /tmp/craft-dashboard/index.html \
    > /tmp/dashboard-watcher.log 2>&1 &
nohup python3 -m http.server 8765 --bind 127.0.0.1 \
    > /tmp/http-server.log 2>&1 &

# From your laptop:
ssh -NL 8765:127.0.0.1:8765 <host>
# then open http://127.0.0.1:8765/craft-dashboard/
```

## Resuming a killed or stalled run

`state.json` is append-safe. If the pipeline crashes or you `kill` it:

```bash
RESUME=harbor-tasks/craft-tools-v4/runs/<ts>/state.json \
    PROFILE=profiles/craft-tools-v4.toml \
    scripts/run-pipeline.sh
```

The script auto-detects RESUME and skips the candidate-selection/preflight
mining. Tasks pick up from whatever stage they were in — no work is lost.

## Rollback + resume infra-induced NEEDS_FIX (optional, post-run)

After a run completes, some tasks usually shelve with infra-flavored
reasons (Docker timed out, Docker build failed, smoke timed out,
classification oracle 0 tests) rather than quality reasons. These are
worth one more shot before declaring the run final — kill the orchestrator
first to avoid a write race, then dry-run, review, apply, and resume:

```bash
pgrep -fa "craft-taskgen --profile" | head
kill <PID>   # only if still running
sleep 5

uv run python scripts/rollback_wedged_tasks.py \
    harbor-tasks/craft-tools-v4/runs/<ts>/state.json          # dry-run

# Review the printed table. Quality-flavored NEEDS_FIX (Build failed,
# patch failed to apply, easiness=, judge parse error, no test files)
# are auto-skipped. If the eligible-rollback list looks right:
uv run python scripts/rollback_wedged_tasks.py \
    harbor-tasks/craft-tools-v4/runs/<ts>/state.json --apply

RESUME=harbor-tasks/craft-tools-v4/runs/<ts>/state.json \
    PROFILE=profiles/craft-tools-v4.toml scripts/run-pipeline.sh
```

When to do this: after every batch as a routine final step is fine, but
especially worth it if the run produced > ~5 NEEDS_FIX with reasons
matching the infra patterns above. The script writes a `.bak-<ts>` of
state.json before applying, so it's easy to undo by `mv` if needed.

## Killing a run

```bash
pgrep -fa "craft-taskgen\|run-pipeline" | head
kill <pid>   # the PID run-pipeline.sh printed at launch
# state.json is durable; resume with --resume whenever you're ready.
```

## Gotchas

- **`candidates/*.json` is greedy.** Any JSON left in `candidates/` from
  prior runs will be re-evaluated. Move or delete stale ones.
- **Harbor patches silently no-op** if the `.venv/lib/python*/...` glob
  doesn't match. If Harbor smoke behavior looks wrong, re-check step 2.
- **Disk pressure on ZFS hosts is misleading.** If `df -h` shows `/dockerroot
  95%` but `docker info` says `Docker Root Dir: /scratch/docker`, ignore the
  legacy mount — real storage is ZFS on `/scratch` (check with
  `zpool list`).
- **Timing.** 50 repos × up to 25 builds/repo (per
  `max_promising_per_repo=25` in the profile) → ~12–24 h wall clock
  depending on complexity and Opus smoke queue depth.
- **`MAX_FIX_ATTEMPTS=1`** (profile default as of 2026-04-24). Tasks whose
  Build step fails once get shelved as `NEEDS_FIX` immediately rather than
  retrying 4×. Empirically retries 2–4 yield 0% acceptance. The
  alignment-feedback regen loop is a separate knob
  (`alignment_regen_count`, capped at 1) and still runs.
- **Claude's throwaway verify-builds.** The `build_dockerfile` step invokes
  `claude -p` with Bash/Write/Edit, and Claude sometimes tests its
  Dockerfile by running `docker build -t test-<whatever>` inside the
  step. These tags are non-deterministic and the pipeline's per-task
  cleanup hook can't match them. On a long batch they accumulate
  (hundreds of 1-2 GB images) and push ZFS pool fill > 70%. Mop up every
  ~4-6 h of wall time, or between batches:
  ```bash
  docker system df                     # check the damage
  docker image prune -af               # wipes dangling + unused (won't touch running containers)
  docker builder prune -af             # BuildKit cache (separate pool)
  zpool list scratch                   # confirm ALLOC dropped
  ```
  Safe while the pipeline is running — prune skips images that are
  referenced by live containers.
- **Disk-pressure-induced `NEEDS_FIX` pile-up.** When a batch run
  flushes the disk past ~80% mid-run, you'll often see the back half
  shelve with infra reasons (Docker timed out, Docker build failed,
  smoke timed out). After cleaning disk, retry those selectively via the
  rollback procedure documented above (`Rollback + resume infra-induced
  NEEDS_FIX`).

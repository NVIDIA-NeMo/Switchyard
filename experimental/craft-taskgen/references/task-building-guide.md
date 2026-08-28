# Phase 2: Task Evaluation & Building Guide

Practical guide for evaluating candidates and building Harbor task packages for CRAFT Tool Orchestration. Updated as we learn lessons from each task.

**Canonical criteria:** `src/craft_taskgen/rubrics.py` (H1-H7, V1-V6, Y1-Y5 with dimension-specific additions). This guide adds practical lessons, not theory.

## How to Run the Pipeline

The pipeline is fully automated via `craft-taskgen`. It takes mined PR candidates, evaluates them with Claude, builds task packages, validates with Docker, smoke-tests with Harbor agents, and accepts tasks that discriminate between model tiers.

### Mine candidates

```bash
# Mine all repos from craft-repos.csv (primary workflow)
craft-taskgen-mine --repos-csv references/craft-repos.csv \
    --repos-dir /path/to/cloned/repos \
    --out candidates/ --top 20 --after 2025-09-01

# Mine a single repo
craft-taskgen-mine scrapy/scrapy --top 20 --out candidates/scrapy.json
```

The miner fetches merged PRs from the GitHub API and scores by structural heuristics — no LLM calls. Diffs are computed against `git merge-base` so scores aren't inflated for PRs where main was rebased mid-review. Requires local clones in `repos/{owner}/{repo}/`.

### Run the pipeline

```bash
# Run with a profile (pauses after evaluate for review)
craft-taskgen run \
    --profile profiles/craft-tools-v4.toml \
    --candidates candidates/*.json

# Run fully unattended (skip checkpoint after evaluate)
craft-taskgen run \
    --profile profiles/craft-tools-v4.toml \
    --candidates candidates/*.json --no-checkpoint

# Resume from a specific step
craft-taskgen --resume runs/2026-04-09/state.json --from-step build --no-checkpoint

# Dashboard
craft-taskgen-dashboard runs/2026-04-09/state.json
```

### Pipeline steps

1. **Select** — Pick top candidates from mining output (score > 0, has tests)
2. **Evaluate** — Opus (direct API) assesses each PR: `accept` or `reject`, using this guide's Quick Decision Framework and Design Principles
3. **Build + Alignment (parallel candidates)** — N=3 (configurable via `build_n_candidates`, capped at 4) independent build+alignment loops run concurrently per task. Each loop: Opus drafts `instruction.md` using the H1-H7, V1-V6 rubric → GPT-5.4 (cross-family) audits instruction ↔ reference tests + diff on a single roll (`alignment_max_retries=1` by default; no retention bias). On `leaked`/`narrow_tests` the loop rebuilds with cumulative feedback up to `max_build_regens_per_candidate=2` times before giving up. The orchestrator picks one passing candidate uniformly at random; the task rejects only if no candidate passes. Each candidate writes to a per-candidate subdir (`t2v3-{tid}-cand{i}`); the winner's dir is renamed to canonical (`t2v3-{tid}-{slug}`) and loser dirs are deleted. Loser summaries land in `task.build_align_losers` (audit-only). Triage-induced Build regen is a separate single-shot path that does NOT fan out. See `docs/reference/n-parallel-calibration-apr25.md` for the calibration data behind these defaults.
4. **Assemble artifacts** — Mechanical: generates `task.toml`, `solution/solve.sh`, and `solution/changes.patch` from the git diff; extracts postmerge test files from the commit.
5. **Build Dockerfile** — Claude creates `environment/Dockerfile`.
6. **F2P/P2P classify** — Two Docker pytest passes (overlay → oracle) produce per-test F2P/P2P lists. `fail_to_pass.txt`, `pass_to_pass.txt`, `score.py`, and `test.sh` are generated automatically.
7. **Oracle check** — Apply solve.sh, run score.py; hard gate that blocks the pipeline if the task doesn't fully resolve.
8. **Smoke** — Run a coding agent via Harbor to produce the reward (primary quality gate); auto-fix loop. Agent/model are configurable (`smoke_agent`/`smoke_model`/`smoke_reasoning_effort` in the profile / `config.py`); default is codex + GPT-5.5, cross-agent from the Opus deep-dive judge. Use `scripts/smoke-probe.py` to iterate on this step against a pre-built task dir without a full pipeline run.
9. **Triage** — Two parallel judges on different questions:
    - **Opus deep-dive** (direct API) returns per-test `skip` or `keep` verdicts on failing reference tests. Skip verdicts are auto-appended to `f2p_skip.txt` and the trial is re-scored; if reward reaches 1.0 the task accepts immediately. Keep verdicts are treated as genuine capability gaps.
    - **Fairness review** (direct API, GPT-5.4) returns one task-level severity (`none`/`minor`/`major`). `severity=major` with both a verbatim instruction quote AND a named failing test triggers a one-shot Build regen (bounded by `MAX_TRIAGE_REGENS`). Anything else sets `reviewer_concern_flag` for human batch review; the task still ships on Opus's verdict.
    On `reward == 1.0` trials both LLM calls are skipped entirely (no failures to classify); a deterministic easiness check (`_deterministic_easiness`) evaluates the trajectory. If the agent solved with ≤ `_EASINESS_GREP_READ_MAX` grep/read tool calls, `easiness_flag` fires and the pipeline routes to a Build regen with prescriptive-instruction feedback (asking Build to rewrite more abstractly — strip named files/classes/data-structures). Regen budget is shared with the reviewer path.
11. **Accept / reject** — Triage advances. `reviewer_concern_flag` is a soft signal on accepted tasks for human batch review (no blocking). `easiness_flag` gated: first occurrence triggers Build regen; a second-pass easiness flag shelves the task as `NEEDS_FIX` (structurally too easy, regen didn't help). The post-run `scripts/analyze-easiness.py` adds cohort-relative stats. See `docs/reference/easiness-heuristics.md`.

### Verifier debugging (step 9)

The critical question for each 0.0 score: **is this a genuine agent failure or a verifier issue?**

Common verifier issues:
- **Missing API symbols in instruction:** Agent builds valid alternative that doesn't match test imports. Fix: add symbols to instruction (specificity tradeoff).
- **Too-strict assertions:** Tests check exact error messages or log formatting. Fix: three-layer audit (V4) to identify.
- **Missing imports in initial Docker build:** Sometimes not all imports are available in the initial Docker build. Fix: the fix loop attempts resolution.

**Replay technique for fast iteration** (previously tied to the `harbor-trial-deep-dive` skill, now retired): reconstruct the agent's container state from `harbor-lab edits`, mount updated tests, rerun the verifier. Turns a 15-min agent re-run into a 30-second Docker replay.

### Expected rates

- 15-30% accept rate from PR candidates (evaluate step, binary verdict)
- Most tasks need 1-2 triage passes before verifiers are solid
- The agent smoke trial is the primary quality gate; rank-inversion detection via a Haiku comparison was dropped (0 fires in ~500 tasks, see Apr 2026 refactor)

## Quick Decision Framework

### Is the candidate worth building?

1. **Will models diverge in approach?** Bug fixes where every model follows find-line-fix-line are useless. Bug fixes requiring architectural reasoning can be discriminating. Feature implementations are generally better because they require integration decisions, but the label isn't what matters — strategy divergence is.
2. **Is the core problem genuinely hard?** Repo size, file count, and parameter-threading breadth don't create difficulty. If the underlying logic is trivial, the task is trivial. Tightening instructions on an easy problem doesn't make it hard.
3. **Reference tests exist?** The commit must include test files that exercise real behavior. No tests = no verifier = no task.
4. **Pure Python?** Rust/C extension work is untaskable in a Python-only agent harness.
5. **Low contamination risk?** Prefer post-Sept-2025 commits. Well-known library features with clear GitHub issues are risky.

### Reject patterns

| Pattern | Example | Why it fails |
|---------|---------|-------------|
| Bug fix with obvious strategy | SA1 (SQLAlchemy M2M join) | All models use identical pipeline |
| Injected bug in error-absorbing system | MY1 (mypy encoding) | Can't verify — errors silently swallowed |
| Constructed task | AL1 (Alembic constraints) | Too easy, solved in 1 min |
| Trivial core logic in complex repo | BT1 (beets genre migration) | Easy problem stays easy |

## Design Principles

- **Integration is the discriminating step, not component creation.** The gap between "I created a class" and "the system uses it" is where weaker models fail. Design tasks where wiring into an existing system is the hard part.
- **Preserving existing behavior while adding new behavior is a genuine trap.** Tasks requiring additive changes without regression force models to reason about the entire call graph.
- **Repository-level variance is signal, not noise.** Codebase-specific factors (docs quality, architecture, test harness) create authentic difficulty. Don't normalize across repos.
- **Failure mode signatures differ by model tier.** Stronger models fail on semantic understanding; weaker models fail on context overflow and navigation. Multi-trial discrimination captures this.

## Writing Instructions (H1-H7)

### The specificity tradeoff

**More specific instructions** enable reliable test verifiers. **Vaguer instructions** are more realistic but require fragile or expensive verification.

We accept the tradeoff: naming the API contract (class names, module paths, config keys) is ~20% given free; the remaining 80% (implementation, integration, preserving existing behavior) is genuinely hard. The agent can't see the tests, so it needs the contract.

**Rule of thumb:** Design choices in test FIXTURES (config creation, constructor signatures) must be specified in the instruction — otherwise agents design valid alternatives that don't match. Design choices in test ASSERTIONS (internal attributes, error messages) can be left for discovery.

### Checklist

- [ ] **H1: Outcome-oriented.** Describe what "done" looks like, not how to get there.
- [ ] **H2: No diagnosis.** State the symptom/goal, not the internal cause. OK to name API contract.
- [ ] **H3: Essential difficulty.** Failures should be reasoning errors, not format compliance.
- [ ] **H4: First approach fails.** Naive attempt should hit a wall.
- [ ] **H5: Non-trivial sequence.** Not solvable in <3 tool calls.
- [ ] **H6: Post-exploration hard.** Even knowing which files, the work is non-trivial.
- [ ] **H7: 50-100 words.** Written for a senior engineer.
- [ ] **T2-H1: Post-exploration still hard.** Assume agent knows the files. Is the rest a textbook exercise?

### Template

```markdown

[1-2 sentences: what the system currently does / what's missing]

[2-4 sentences: desired behavior, mentioning API contract elements the tests expect]

[1 sentence: constraint about preserving existing behavior]

## Environment

The project is at `/code/`. Write any output files to the `/code/output/` directory.
```

## Selecting Reference Tests (V1-V6)

The assemble artifacts step automatically discovers test files from the commit diff (via `git diff`) and places them in `tests/postmerge_tests/`. The F2P/P2P classify step then generates `fail_to_pass.txt`, `pass_to_pass.txt`, `score.py`, and `test.sh` from those files. The three-layer audit below is the discipline for evaluating whether the discovered tests are good verifiers — not for choosing which files to include.

### Reference tests must verify by executing, not inspecting

Tests should call the function, instantiate the class, run the endpoint, and check output — not grep source files for keywords. A test that imports the class and asserts on behavior is good. A test that opens a `.py` file and checks for a class name is gameable.

## Dockerfile conventions

- Clone repo at merge base SHA {merge_base_sha} (exact pre-change state, clean diff base)
- Install all deps + test deps (pytest, etc.)
- **Pre-download** anything tests need (models, data, etc.)
- Reset git: `rm -rf .git && git init && git config user.email "agent@test" && git config user.name "Agent" && git add -A && git commit -m 'initial commit'`
- Create output dir: `mkdir -p /code/output`

## Task file structure

```
<repo>_<pr_number>/
├── instruction.md
├── task.toml
├── environment/
│   └── Dockerfile
├── solution/
│   ├── solve.sh
│   └── changes.patch
└── tests/                        ← generated by F2P/P2P classify step
    ├── fail_to_pass.txt
    ├── pass_to_pass.txt
    ├── test.sh
    ├── score.py
    └── postmerge_tests/
        └── <repo>/<path>/test_*.py
```

## task.toml template

```toml
version = "1.0"

[metadata]
name = "t2v3-<ID>-<name>"
difficulty = "hard"

[verifier]
timeout_sec = 600

[agent]
timeout_sec = 3600

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
allow_internet = true
mcp_servers = []

[environment.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"
OPENAI_API_KEY = "${OPENAI_API_KEY}"
OPENAI_BASE_URL = "${OPENAI_BASE_URL}"

[solution.env]
```

## Lessons log

- **ES1 (espnet):** Verifier went through 6 iterations (24 → 22 → 10 → 5 tests) via three-layer audit. Integration is the discriminator — all models create components, only Opus partially wires them.
- **SA1 (sqlalchemy M2M join):** No strategy discrimination. Bug fixes don't test orchestration.
- **MY1 (mypy):** Abandoned. Error-absorbing system can't be verified.
- **AL1 (alembic constraints):** Too easy. Constructed task.
- **BT1 (beets genre migration):** Abandoned. Trivial core logic.
- **LO1 (locust pytest):** Rejected. Verifier gap — tests only check class structure, not runtime behavior.
- **After editing an instruction, re-verify that all previously-passing tests still pass for the right reason** (see challenge question 6 in `harbor-trial-deep-dive` skill).

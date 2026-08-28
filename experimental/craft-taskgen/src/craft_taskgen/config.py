# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline configuration constants, path resolution, and state model."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_CLAUDE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")


def _resolve_claude_cmd(version: str) -> str:
    """Return pinned Claude binary if installed, else fall back to system 'claude'.

    The version string can come from a TOML profile, so validate it as a bare
    version token (no path separators, no '..') before building a filesystem
    path, preventing path traversal out of the versions directory (CWE-22).
    """
    if not _CLAUDE_VERSION_RE.match(version) or ".." in version:
        raise ValueError(f"Invalid claude_code_version: {version!r}")
    pinned = os.path.expanduser(f"~/.local/share/claude/versions/{version}")
    if os.path.isfile(pinned) and os.access(pinned, os.X_OK):
        return pinned
    return "claude"


CC_VERSION = "2.1.118"
CLAUDE_CMD = _resolve_claude_cmd(CC_VERSION)
DEFAULT_PERMISSION_MODE = "auto"
DEFAULT_MAX_TURNS = 100  # effectively uncapped; timeout is the real safety net
DEFAULT_TIMEOUT = 1200  # 20 min per LLM step
TASK_SUITE_DIR = "harbor-tasks/craft-tools-v4"
HAIKU_MODEL = "azure/anthropic/claude-haiku-4-5"
OPUS_MODEL = "azure/anthropic/claude-opus-4-6"
LLM_STEP_MODEL = "azure/anthropic/claude-opus-4-6"  # azure avoids bedrock context_management rejection
# Smoke-test agent — the Harbor agent trial that actually *solves* the task and
# produces the reward (the primary quality gate). Decoupled from the direct-API
# judge models above: the agent that attempts the task can differ from the
# Opus/GPT judges that triage its trial. Default is codex + GPT-5.5 (cross-agent
# from the Opus deep-dive judge, per the no-self-family-judging design).
# Gateway naming: codex routes via the NVIDIA gateway, whose slugs double the
# vendor segment (e.g. `openai/openai/gpt-5.5`). Reasoning tokens require the
# codex model catalog (see runner._filtered_codex_catalog).
SMOKE_AGENT = "codex"
SMOKE_MODEL = "openai/openai/gpt-5.5"
SMOKE_REASONING_EFFORT = "high"  # "" → fall back to baselines.reasoning_defaults
# Cross-family judge for build-time alignment audit (Opus generates → GPT judges).
# Paper story: no LLM judges output from its own family (arXiv 2410.21819
# self-preference bias mitigation). Defaults to GPT-5.4 via the NVIDIA gateway;
# per-profile override via llm_alignment_model TOML field.
LLM_ALIGNMENT_MODEL = "openai/us/azure/openai/gpt-5.4"
MAX_FIX_ATTEMPTS = 2  # gates retries in build, docker-fix, and classify-fix paths
# Max times a single task may loop back through the pipeline from triage into
# a new Build regeneration. Persists across pipeline iterations so the task
# can't bounce between triage and Build indefinitely. One regen empirically
# captures the conversions; additional regens add cost without recovery.
MAX_TRIAGE_REGENS = 1
# Number of independent build+alignment candidate loops run per task. Each
# loop is build → alignment → (rebuild on leaked/narrow_tests with feedback,
# up to MAX_BUILD_REGENS_PER_CANDIDATE times). The orchestrator picks a
# passing candidate uniformly at random; REJECTED only if all candidates
# fail. N=3 was empirically the highest-yield + lowest-cost setting on the
# rerun-accepts-v2 calibration cohort (Apr 25). Capped at 4.
BUILD_N_CANDIDATES = 3
# Number of retention retries on the alignment judge per evaluation.
# These are re-polls of the SAME instruction (judge-flakiness mitigation,
# accept-on-first-ok). α=1 disables retention bias entirely (single roll,
# no shielding for borderline-leaky verdicts). On Apr 25 calibration, α=1
# combined with N=3 + r=2 produced the best yield/cost. Capped at [1,5].
ALIGNMENT_MAX_RETRIES = 1
# Maximum alignment-feedback-driven Build regens per candidate. After the
# initial build's alignment evaluation, if verdict is leaked/narrow_tests
# the candidate can rebuild with the assessor's evidence in the prompt.
# r=2 means up to 2 such rebuilds per candidate. r=0 disables build-retry
# entirely. On Apr 25 calibration, r=2 paired with N=3 + α=1 was optimal.
# Capped at [0,3].
MAX_BUILD_REGENS_PER_CANDIDATE = 2
MAX_SMOKE_RETRIES = 2  # retry infra failures (not task failures)
DEFAULT_CONCURRENCY = 4
LLM_CONCURRENCY = 4  # parallel claude -p calls
DOCKER_CONCURRENCY = 2  # parallel Docker build/classify/oracle ops
SMOKE_CONCURRENCY = 2  # parallel Harbor smoke tests
MAX_PROMISING_PER_REPO = 3  # cap per-repo builds to maintain diversity
# Pre-execution difficulty prediction (REJECT_MAYBE / REJECT_BAND_B) was removed
# when evaluate moved to direct-API with a binary accept/reject verdict.


def _load_env() -> None:
    """Load .env file into os.environ (required for Harbor API keys)."""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(env_path):
        print("WARNING: No .env file found — Harbor runs may fail (missing API keys)")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            # Expand bare env-var references: ${VAR} or $VAR as the entire value
            if value.startswith("${") and value.endswith("}"):
                value = os.environ.get(value[2:-1], value)
            elif value.startswith("$") and value[1:].isidentifier():
                value = os.environ.get(value[1:], value)
            if key and key not in os.environ:
                os.environ[key] = value
    print(f"Loaded .env ({env_path})")


# ---------------------------------------------------------------------------
# Path resolution — locates reference docs and templates relative to package
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PipelineContext:
    """Resolved paths for reference docs, templates, and output directories.

    Auto-resolves relative to the craft-taskgen repo root by default.
    All paths are absolute strings for embedding in prompts.
    """

    references_dir: str = ""
    templates_dir: str = ""
    task_suite_dir: str = ""
    craft_bench_dir: str = ""

    def __post_init__(self) -> None:
        if not self.references_dir:
            self.references_dir = str(_PACKAGE_ROOT / "references")
        if not self.templates_dir:
            self.templates_dir = str(_PACKAGE_ROOT / "templates")
        if not self.task_suite_dir:
            self.task_suite_dir = TASK_SUITE_DIR

    @property
    def task_building_guide(self) -> str:
        return os.path.join(self.references_dir, "task-building-guide.md")

    @property
    def craft_bench_skills(self) -> str:
        return os.path.join(self.craft_bench_dir, ".claude", "skills")

    @property
    def template_task_dir(self) -> str:
        return os.path.join(self.templates_dir, "t2v3-CE0266-celery-retry-unification")

    @property
    def instruction_template(self) -> str:
        return os.path.join(self.templates_dir, "structural_template_instruction.md")

    @property
    def instruction_preamble(self) -> str:
        """First line of the instruction structural template — the standard task preamble."""
        path = self.instruction_template
        try:
            with open(path) as f:
                line = f.readline().strip()
        except FileNotFoundError:
            raise RuntimeError(
                f"Instruction template not found: {path}\n"
                "Ensure templates/structural_template_instruction.md exists in the craft-taskgen repo."
            ) from None
        except OSError as e:
            raise RuntimeError(f"Failed to read instruction template {path}: {e}") from e
        if not line:
            raise RuntimeError(f"Instruction template first line is blank: {path}")
        return line


# ---------------------------------------------------------------------------
# Pipeline profile — configurable parameters, loadable from TOML
# ---------------------------------------------------------------------------


@dataclass
class PipelineProfile:
    """Tunable pipeline parameters. Defaults match the module-level constants.

    Load from TOML for named, reproducible configurations. When applied,
    updates the module-level constants so all existing code reads the right values.
    """

    opus_model: str = OPUS_MODEL
    haiku_model: str = HAIKU_MODEL
    llm_step_model: str = LLM_STEP_MODEL
    llm_alignment_model: str = LLM_ALIGNMENT_MODEL
    smoke_agent: str = SMOKE_AGENT
    smoke_model: str = SMOKE_MODEL
    smoke_reasoning_effort: str = SMOKE_REASONING_EFFORT
    claude_code_version: str = CC_VERSION
    max_fix_attempts: int = MAX_FIX_ATTEMPTS
    max_triage_regens: int = MAX_TRIAGE_REGENS
    build_n_candidates: int = BUILD_N_CANDIDATES
    alignment_max_retries: int = ALIGNMENT_MAX_RETRIES
    max_build_regens_per_candidate: int = MAX_BUILD_REGENS_PER_CANDIDATE
    max_smoke_retries: int = MAX_SMOKE_RETRIES
    max_promising_per_repo: int = MAX_PROMISING_PER_REPO
    default_timeout: int = DEFAULT_TIMEOUT
    default_concurrency: int = DEFAULT_CONCURRENCY
    llm_concurrency: int = LLM_CONCURRENCY
    docker_concurrency: int = DOCKER_CONCURRENCY
    smoke_concurrency: int = SMOKE_CONCURRENCY
    task_suite_dir: str = TASK_SUITE_DIR
    skip_smoke: bool = False
    # task.toml resource/timeout values — written mechanically by assemble_task_dir_artifacts
    task_verifier_timeout: int = 600
    task_agent_timeout: int = 3600
    task_build_timeout: int = 900
    task_cpus: int = 2
    task_memory_mb: int = 4096
    task_storage_mb: int = 10240
    task_gpus: int = 0
    task_allow_internet: bool = True

    @classmethod
    def from_toml(cls, path: str) -> PipelineProfile:
        """Load profile from a TOML file. Unknown keys are ignored."""
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        flat: dict = {}
        for section in data.values():
            if isinstance(section, dict):
                flat.update(section)
            else:
                continue
        # Map TOML keys to dataclass field names
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in known})

    def apply(self) -> None:
        """Update module-level constants from this profile. Call once at startup."""
        import craft_taskgen.config as cfg

        cfg.OPUS_MODEL = self.opus_model
        cfg.HAIKU_MODEL = self.haiku_model
        cfg.LLM_STEP_MODEL = self.llm_step_model
        cfg.LLM_ALIGNMENT_MODEL = self.llm_alignment_model
        cfg.SMOKE_AGENT = self.smoke_agent
        cfg.SMOKE_MODEL = self.smoke_model
        cfg.SMOKE_REASONING_EFFORT = self.smoke_reasoning_effort
        cfg.CC_VERSION = self.claude_code_version
        cfg.CLAUDE_CMD = _resolve_claude_cmd(self.claude_code_version)
        cfg.MAX_FIX_ATTEMPTS = self.max_fix_attempts
        cfg.MAX_TRIAGE_REGENS = self.max_triage_regens
        n = self.build_n_candidates
        if n < 1:
            print(f"WARNING: build_n_candidates={n} < 1, clamping to 1")
            n = 1
        elif n > 4:
            print(f"WARNING: build_n_candidates={n} > 4, clamping to 4 (cost guardrail)")
            n = 4
        cfg.BUILD_N_CANDIDATES = n
        a = self.alignment_max_retries
        if a < 1:
            print(f"WARNING: alignment_max_retries={a} < 1, clamping to 1")
            a = 1
        elif a > 5:
            print(f"WARNING: alignment_max_retries={a} > 5, clamping to 5 (cost guardrail)")
            a = 5
        cfg.ALIGNMENT_MAX_RETRIES = a
        r = self.max_build_regens_per_candidate
        if r < 0:
            print(f"WARNING: max_build_regens_per_candidate={r} < 0, clamping to 0")
            r = 0
        elif r > 3:
            print(f"WARNING: max_build_regens_per_candidate={r} > 3, clamping to 3 (cost guardrail)")
            r = 3
        cfg.MAX_BUILD_REGENS_PER_CANDIDATE = r
        cfg.MAX_SMOKE_RETRIES = self.max_smoke_retries
        cfg.MAX_PROMISING_PER_REPO = self.max_promising_per_repo
        cfg.DEFAULT_TIMEOUT = self.default_timeout
        cfg.DEFAULT_CONCURRENCY = self.default_concurrency
        cfg.LLM_CONCURRENCY = self.llm_concurrency
        cfg.DOCKER_CONCURRENCY = self.docker_concurrency
        cfg.SMOKE_CONCURRENCY = self.smoke_concurrency
        cfg.TASK_SUITE_DIR = self.task_suite_dir

    def to_dict(self) -> dict:
        """Serialize for embedding in state.json."""
        return asdict(self)


# Valid --from-step values. Only "select" (index 0) and "evaluate" (index 1)
# change pipeline behavior; everything from "build" onward is handled by the
# per-task stage machine in run_task_pipeline and all resume to the same place.
# The later entries are kept for backward-compat and operator ergonomics —
# someone typing `--from-step oracle` after a resume should still work.
STEP_ORDER = [
    "select",
    "evaluate",
    "build",
    "alignment",
    "assemble_artifacts",
    "build_dockerfile",
    "docker_classify",
    "oracle",
    "smoke",
    "opus_triage",
    "report",
]


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    PROMISING = "promising"
    BUILT = "built"
    ALIGNMENT_CHECKED = "alignment_checked"
    TESTS_DISCOVERED = "tests_discovered"
    DOCKERFILE_BUILT = "dockerfile_built"
    F2P_P2P_CLASSIFIED = "f2p_p2p_classified"

    ORACLE_CHECKED = "oracle_checked"
    OPUS_SMOKE_TESTED = "opus_smoke_tested"
    OPUS_TRIAGED = "opus_triaged"
    ACCEPTED = "accepted"
    NEEDS_FIX = "needs_fix"
    REJECTED = "rejected"


@dataclass
class TaskState:
    task_id: str
    repo: str
    commit_sha: str  # merge commit SHA
    description: str
    base_sha: str  # pre-merge base branch HEAD (pr.base.sha from GitHub PR API)
    merge_base_sha: str  # git merge-base base_sha sha — used for Docker clone and patch application
    stage: Stage = Stage.CANDIDATE

    eval_verdict: str = ""
    eval_reason: str = ""
    eval_instruction_sketch: str = ""
    eval_verifier_notes: str = ""  # removed from EVALUATE_SCHEMA; kept for state.json back-compat

    # Per-step LLM call accounting populated by direct-API judges. Key = step
    # name ("evaluate", "build", "alignment", "deep_dive_opus",
    # "fairness_review", "summary"), value = list of {tokens_in,
    # tokens_out, tokens_cached, model, latency_s} per call. Supports
    # cost/latency reporting in dashboard + status.
    llm_usage: dict[str, list[dict]] = field(default_factory=dict)

    task_dir: str = ""
    instruction_words: int = 0

    # Alignment-judge verdict: ok / vague / narrow_tests / leaked / misaligned.
    # Retention-biased 3× retry — attempts list stores each retry's raw verdict
    # so humans can audit disagreement. Task is accepted if any attempt said ok.
    alignment_verdict: str = ""
    alignment_reason: str = ""
    alignment_v4_audit: dict[str, bool] = field(default_factory=dict)
    alignment_attempts: list[dict] = field(default_factory=list)
    # When alignment rejects with `leaked` or `narrow_tests`, its specific
    # evidence quotes are concatenated into this field and fed back to Build
    # for a single regen attempt (loop bounded at 1 per task).
    alignment_feedback: str = ""
    alignment_regen_count: int = 0
    # Per-candidate loser summaries from the parallel build+alignment
    # orchestrator. One entry per non-winning candidate when N>1; empty
    # when N=1 or for tasks whose winner was the only passer. Each dict:
    # {cand_id, outcome, verdict, short_reason (≤500 chars),
    #  regen_count, instruction_words}. Audit-only — no programmatic
    # consumer downstream.
    build_align_losers: list[dict] = field(default_factory=list)

    # F2P/P2P classification results
    f2p_tests: list[str] = field(default_factory=list)
    p2p_tests: list[str] = field(default_factory=list)

    # Oracle check (hard gate — blocks pipeline if not resolved)
    oracle_resolved: bool = False
    oracle_f2p_score: float = 0.0
    oracle_p2p_score: float = 0.0
    oracle_flagged: bool = False
    oracle_flag_reason: str = ""

    opus_score: str = ""
    opus_trial_dir: str = ""

    issues: list[dict] = field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: str = ""

    # Triage bookkeeping (overwritten each triage pass). `dd_*` counts
    # come from the Opus per-test skip/keep judge; `reviewer_*` fields
    # come from the cross-family fairness-review step.
    # `reviewer_concern_flag` is advisory (task still accepts); Build
    # regen fires only when `reviewer_concern_severity=major` with both
    # evidence fields populated.
    dd_failure_count: int = 0
    dd_dropped_by_skip_filter: int = 0
    dd_dropped_by_reward_filter: int = 0

    reviewer_concern_flag: bool = False
    reviewer_concern_severity: str = ""  # "none", "minor", "major", or ""
    reviewer_concern_reason: str = ""
    reviewer_concern_evidence_quote: str = ""
    reviewer_concern_evidence_test: str = ""

    # Deterministic easiness check (no LLM) — see steps.py::_deterministic_easiness
    # and docs/reference/easiness-heuristics.md. Fires on reward==1.0 when the agent
    # made <= 5 Grep/Read calls on the full (un-tailed) tool sequence.
    # Routes to needs_human_review; never auto-rejects. `easiness_concern`
    # stays as a kept-for-state.json-compatibility boolean no longer
    # populated (was set by the now-removed skeptical reviewer).
    easiness_flag: bool = False
    easiness_reason: str = ""
    easiness_concern: bool = False

    summary: str = ""  # narrative summary generated after pipeline completes

    fix_attempts: int = 0
    fix_history: list[str] = field(default_factory=list)
    # Count of Build regenerations triggered from triage feedback. Persists
    # across pipeline iterations so a task can't bounce between triage and
    # Build indefinitely. Capped by _cfg.MAX_TRIAGE_REGENS.
    triage_regen_count: int = 0
    # Set by triage when a fix agent runs; consumed by run_task_pipeline to route
    # the retry without re-diffing files a second time. "" means no pending fix.
    pending_fix_type: str = ""

    # Per-iteration log: [{iteration, opus_score, issues, fix_applied, timestamp}]
    iteration_log: list[dict] = field(default_factory=list)

    # What step is currently running (empty = idle). Set at step entry, cleared on completion.
    in_progress_step: str = ""

    # Raw candidate dict from the miner (sha, score, source_files, etc.). Written to
    # candidate.json in the task dir during assemble. Empty for tasks loaded from old state.json.
    candidate_data: dict = field(default_factory=dict)


@dataclass
class PipelineState:
    created: str = ""
    last_updated: str = ""
    run_dir: str = ""  # isolated output directory for this pipeline run
    profile_data: dict = field(default_factory=dict)  # profile snapshot for reproducibility
    # Host/input metadata — lets us identify which machine processed which candidates
    # when a large run is split across multiple machines.
    run_info: dict = field(default_factory=dict)
    tasks: dict[str, TaskState] = field(default_factory=dict)

    def save(self, path: str) -> None:
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> PipelineState:
        with open(path) as f:
            data = json.load(f)
        state = cls(
            created=data["created"],
            last_updated=data["last_updated"],
            run_dir=data.get("run_dir", ""),
            profile_data=data.get("profile_data", {}),
            run_info=data.get("run_info", {}),
        )
        known_fields = {f.name for f in TaskState.__dataclass_fields__.values()}
        # Map removed stage values to their replacements for backward compat
        _stage_migration = {"docker_validated": "oracle_checked"}
        for tid, tdata in data.get("tasks", {}).items():
            raw_stage = tdata["stage"]
            tdata["stage"] = Stage(_stage_migration.get(raw_stage, raw_stage))
            # Filter out unknown fields for forward/backward compatibility
            filtered = {k: v for k, v in tdata.items() if k in known_fields}
            state.tasks[tid] = TaskState(**filtered)
        return state

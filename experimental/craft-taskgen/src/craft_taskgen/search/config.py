# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search pipeline configuration: step order, state model, and search-specific constants."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Search-specific constants (overridable via PipelineProfile)
# ---------------------------------------------------------------------------

CODEX_MODEL = "openai/openai/gpt-5.3-codex"
DEDUP_THRESHOLD = 0.65
DEDUP_EMBEDDING_MODEL = "openai/azure/openai/text-embedding-3-small"
REJECT_THRESHOLD = 0.3  # both Opus + Codex <= this → reject
HAIKU_INVERSION_MARGIN = 0.1  # haiku must beat opus by more than this to trigger
HAIKU_INVERSION_FLOOR = 0.5  # and haiku must actually perform (>= this) — not just non-zero noise
REPO_MAP_MAX_CHARS = 120_000  # character budget for build_repo_map output
SYNTHESIS_CONCURRENCY = 5

# LiteLLM model names for 3-model synthesis
SYNTHESIS_MODELS = [
    "openai/aws/anthropic/bedrock-claude-sonnet-4-6",
    "openai/gcp/google/gemini-3.1-pro-preview",
    "openai/us/azure/openai/gpt-5.4",
]
JUDGE_MODEL = "openai/aws/anthropic/bedrock-claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Step order
# ---------------------------------------------------------------------------

SEARCH_STEPS = [
    "extract",
    "synthesize",
    "validate",
    "dedup",
    "harbor",
    "smoke-all",
    "gold-review",
    "filter",
    "report",
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

APPROACHES = ["a", "b", "c"]


def load_approach_tasks(output_dir: str) -> list[dict]:
    """Load all search tasks across approach-{a,b,c}/search_tasks.json."""
    all_tasks: list[dict] = []
    for approach in APPROACHES:
        path = os.path.join(output_dir, f"approach-{approach}", "search_tasks.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            all_tasks.extend(json.load(f))
    return all_tasks


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


@dataclass
class SearchTaskStatus:
    """Per-task agent scores and filter result."""

    opus_reward: float | None = None
    opus_nav: float | None = None
    opus_assert: float | None = None
    opus_file_recall: float | None = None
    opus_func_recall: float | None = None

    codex_reward: float | None = None
    codex_nav: float | None = None
    codex_assert: float | None = None
    codex_file_recall: float | None = None
    codex_func_recall: float | None = None

    haiku_reward: float | None = None
    haiku_nav: float | None = None
    haiku_assert: float | None = None
    haiku_file_recall: float | None = None
    haiku_func_recall: float | None = None

    review_recommendation: str = ""
    review_flags: list[str] = field(default_factory=list)

    status: str = ""  # accepted / rejected / flagged
    flags: list[str] = field(default_factory=list)


@dataclass
class SearchPipelineState:
    """State for the search-from-T2 pipeline."""

    created: str = ""
    last_updated: str = ""
    run_dir: str = ""
    profile_data: dict = field(default_factory=dict)

    # Pipeline progress
    stages_completed: list[str] = field(default_factory=list)
    current_stage: str = ""

    # Config (populated at start, persisted for resume)
    tasks_dir: str = ""
    repos_dir: str = ""
    output_dir: str = ""
    harbor_dir: str = ""
    limit: int = 0  # limit number of input tasks to process (0 = all)
    concurrency: int = 4

    # Job directories from agent smoke runs
    job_dirs: dict[str, str] = field(default_factory=dict)

    # Per-task statuses (task_id -> scores + filter result)
    task_statuses: dict[str, SearchTaskStatus] = field(default_factory=dict)

    def save(self, path: str) -> None:
        self.last_updated = datetime.now().isoformat()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> SearchPipelineState:
        with open(path) as f:
            data = json.load(f)
        state = cls(
            created=data.get("created", ""),
            last_updated=data.get("last_updated", ""),
            run_dir=data.get("run_dir", ""),
            profile_data=data.get("profile_data", {}),
            stages_completed=data.get("stages_completed", []),
            current_stage=data.get("current_stage", ""),
            tasks_dir=data.get("tasks_dir", data.get("t2_tasks_dir", "")),
            repos_dir=data.get("repos_dir", ""),
            output_dir=data.get("output_dir", ""),
            harbor_dir=data.get("harbor_dir", ""),
            limit=data.get("limit", 0),
            concurrency=data.get("concurrency", 4),
            job_dirs=data.get("job_dirs", {}),
        )
        known_fields = {f.name for f in SearchTaskStatus.__dataclass_fields__.values()}
        for tid, tdata in data.get("task_statuses", {}).items():
            filtered = {k: v for k, v in tdata.items() if k in known_fields}
            state.task_statuses[tid] = SearchTaskStatus(**filtered)
        return state

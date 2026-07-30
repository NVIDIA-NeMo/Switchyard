# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for search pipeline: config, dedup, filter, state management."""

from __future__ import annotations

import json
import os
import tempfile

from craft_taskgen.search.config import SEARCH_STEPS, SearchPipelineState, SearchTaskStatus

# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


def test_search_state_round_trip():
    """SearchPipelineState serializes and deserializes correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        state = SearchPipelineState(
            created="2026-04-13T10:00:00",
            run_dir=tmp,
            tasks_dir="harbor-tasks/craft-tools-v4",
            repos_dir="repos",
            output_dir="gold/test",
            concurrency=4,
            stages_completed=["extract", "synthesize"],
            current_stage="validate",
        )
        state.task_statuses["craft-test-c-abc"] = SearchTaskStatus(
            opus_reward=0.75, codex_reward=0.60, haiku_reward=0.50, status="accepted"
        )
        path = os.path.join(tmp, "state.json")
        state.save(path)

        loaded = SearchPipelineState.load(path)
        assert loaded.tasks_dir == "harbor-tasks/craft-tools-v4"
        assert loaded.stages_completed == ["extract", "synthesize"]
        assert loaded.current_stage == "validate"
        assert loaded.task_statuses["craft-test-c-abc"].opus_reward == 0.75
        assert loaded.task_statuses["craft-test-c-abc"].status == "accepted"


def test_search_state_load_ignores_unknown_fields():
    """Forward-compatible: unknown fields in task_statuses are ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        data = {
            "created": "2026-04-13",
            "last_updated": "2026-04-13",
            "run_dir": tmp,
            "profile_data": {},
            "stages_completed": [],
            "current_stage": "",
            "t2_tasks_dir": "",
            "repos_dir": "",
            "output_dir": "",
            "harbor_dir": "",
            "craft_bench_dir": "",
            "concurrency": 4,
            "job_dirs": {},
            "task_statuses": {
                "task-1": {
                    "opus_reward": 0.5,
                    "future_field": "should be ignored",
                }
            },
        }
        path = os.path.join(tmp, "state.json")
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = SearchPipelineState.load(path)
        assert loaded.task_statuses["task-1"].opus_reward == 0.5


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_cosine_similarity():
    from craft_taskgen.search.dedup import _cosine_similarity

    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
    assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-10
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0  # zero norm


def test_cosine_similarity_similar_vectors():
    from craft_taskgen.search.dedup import _cosine_similarity

    a = [1.0, 0.5, 0.3]
    b = [0.9, 0.6, 0.2]
    sim = _cosine_similarity(a, b)
    assert 0.95 < sim < 1.0  # very similar


def test_gold_richness():
    from craft_taskgen.search.dedup import _gold_richness

    task = {"gold_answer": {"files": ["a.py", "b.py"], "functions": ["f1"], "assertions": ["a1", "a2"]}}
    assert _gold_richness(task) == (3, 2)  # 2 files + 1 func, 2 assertions


def test_merge_alt_gold():
    from craft_taskgen.search.dedup import _merge_alt_gold

    kept = {
        "gold_answer": {
            "files": ["a.py"],
            "functions": ["f1"],
            "alt_files": [],
            "alt_functions": [],
        }
    }
    removed = {
        "gold_answer": {
            "files": ["a.py", "b.py"],
            "functions": ["f2"],
        }
    }
    _merge_alt_gold(kept, removed)
    assert "b.py" in kept["gold_answer"]["alt_files"]
    assert "f2" in kept["gold_answer"]["alt_functions"]
    # a.py is already in kept's files, so not added to alt
    assert kept["gold_answer"]["alt_files"] == ["b.py"]


def test_deduplicate_removes_similar():
    from craft_taskgen.search.dedup import deduplicate

    tasks = [
        {"id": "task-a", "gold_answer": {"files": ["a.py", "b.py"], "functions": ["f1"], "assertions": []}},
        {"id": "task-b", "gold_answer": {"files": ["a.py"], "functions": ["f1"], "assertions": []}},
    ]
    # Identical embeddings → should dedup
    embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    kept, pairs = deduplicate(tasks, embeddings, threshold=0.65)
    assert len(kept) == 1
    assert kept[0]["id"] == "task-a"  # richer gold
    assert len(pairs) == 1
    assert pairs[0]["removed"] == "task-b"


def test_deduplicate_keeps_dissimilar():
    from craft_taskgen.search.dedup import deduplicate

    tasks = [
        {"id": "t1", "gold_answer": {"files": [], "functions": [], "assertions": []}},
        {"id": "t2", "gold_answer": {"files": [], "functions": [], "assertions": []}},
    ]
    # Orthogonal embeddings → keep both
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    kept, pairs = deduplicate(tasks, embeddings, threshold=0.65)
    assert len(kept) == 2
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_both_low_rejects():
    """Tasks where both Opus and Codex score <= 0.3 are rejected."""
    from craft_taskgen.search.steps import step_filter

    with tempfile.TemporaryDirectory() as tmp:
        # Create a minimal task file
        approach_dir = os.path.join(tmp, "approach-c")
        os.makedirs(approach_dir)
        tasks = [{"id": "task-1", "gold_answer": {"files": ["a.py"], "functions": ["f1"]}}]
        with open(os.path.join(approach_dir, "search_tasks.json"), "w") as f:
            json.dump(tasks, f)

        state = SearchPipelineState(output_dir=tmp)
        state.task_statuses["task-1"] = SearchTaskStatus(opus_reward=0.2, codex_reward=0.1, haiku_reward=0.1)
        step_filter(state)
        assert state.task_statuses["task-1"].status == "rejected"
        assert any("both_low" in f for f in state.task_statuses["task-1"].flags)


def test_filter_haiku_inversion_rejects():
    """Tasks where Haiku > Opus are rejected."""
    from craft_taskgen.search.steps import step_filter

    with tempfile.TemporaryDirectory() as tmp:
        approach_dir = os.path.join(tmp, "approach-c")
        os.makedirs(approach_dir)
        tasks = [{"id": "task-1", "gold_answer": {"files": ["a.py"], "functions": ["f1"]}}]
        with open(os.path.join(approach_dir, "search_tasks.json"), "w") as f:
            json.dump(tasks, f)

        state = SearchPipelineState(output_dir=tmp)
        state.task_statuses["task-1"] = SearchTaskStatus(opus_reward=0.5, codex_reward=0.6, haiku_reward=0.7)
        step_filter(state)
        assert state.task_statuses["task-1"].status == "rejected"
        assert any("haiku_inversion" in f for f in state.task_statuses["task-1"].flags)


def test_filter_flat_easy_rejects():
    """Tasks where all 3 models >= 0.9 are rejected."""
    from craft_taskgen.search.steps import step_filter

    with tempfile.TemporaryDirectory() as tmp:
        approach_dir = os.path.join(tmp, "approach-c")
        os.makedirs(approach_dir)
        tasks = [{"id": "task-1", "gold_answer": {"files": ["a.py"], "functions": ["f1"]}}]
        with open(os.path.join(approach_dir, "search_tasks.json"), "w") as f:
            json.dump(tasks, f)

        state = SearchPipelineState(output_dir=tmp)
        state.task_statuses["task-1"] = SearchTaskStatus(
            opus_reward=0.95, codex_reward=0.92, haiku_reward=0.91
        )
        step_filter(state)
        assert state.task_statuses["task-1"].status == "rejected"
        assert any("flat_easy" in f for f in state.task_statuses["task-1"].flags)


def test_filter_no_gold_functions_rejects():
    """Tasks with empty gold functions are rejected."""
    from craft_taskgen.search.steps import step_filter

    with tempfile.TemporaryDirectory() as tmp:
        approach_dir = os.path.join(tmp, "approach-c")
        os.makedirs(approach_dir)
        tasks = [{"id": "task-1", "gold_answer": {"files": ["a.py"], "functions": []}}]
        with open(os.path.join(approach_dir, "search_tasks.json"), "w") as f:
            json.dump(tasks, f)

        state = SearchPipelineState(output_dir=tmp)
        state.task_statuses["task-1"] = SearchTaskStatus(opus_reward=0.8, codex_reward=0.7, haiku_reward=0.5)
        step_filter(state)
        assert state.task_statuses["task-1"].status == "rejected"
        assert any("no_gold_functions" in f for f in state.task_statuses["task-1"].flags)


def test_filter_good_task_accepted():
    """Tasks with good scores and valid gold are accepted."""
    from craft_taskgen.search.steps import step_filter

    with tempfile.TemporaryDirectory() as tmp:
        approach_dir = os.path.join(tmp, "approach-c")
        os.makedirs(approach_dir)
        tasks = [{"id": "task-1", "gold_answer": {"files": ["a.py"], "functions": ["f1"]}}]
        with open(os.path.join(approach_dir, "search_tasks.json"), "w") as f:
            json.dump(tasks, f)

        state = SearchPipelineState(output_dir=tmp)
        state.task_statuses["task-1"] = SearchTaskStatus(opus_reward=0.8, codex_reward=0.6, haiku_reward=0.4)
        step_filter(state)
        assert state.task_statuses["task-1"].status == "accepted"
        assert state.task_statuses["task-1"].flags == []


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------


def test_search_steps_all_registered():
    """Every step in SEARCH_STEPS has a corresponding function in SEARCH_STEP_FUNCS."""
    from craft_taskgen.search.steps import SEARCH_STEP_FUNCS

    for step in SEARCH_STEPS:
        assert step in SEARCH_STEP_FUNCS, f"Step '{step}' not in SEARCH_STEP_FUNCS"


def test_search_steps_are_callable():
    """All step functions are callable."""
    from craft_taskgen.search.steps import SEARCH_STEP_FUNCS

    for name, func in SEARCH_STEP_FUNCS.items():
        assert callable(func), f"Step '{name}' is not callable"


# ---------------------------------------------------------------------------
# Extract module
# ---------------------------------------------------------------------------


def test_parse_solve_sh_extracts_commit():
    """parse_solve_sh extracts commit hash and upstream URL."""
    from craft_taskgen.search.extract import parse_solve_sh

    solve_sh = """#!/bin/bash
COMMIT=abc123def456
cd /repo
git remote add upstream https://github.com/test/repo.git
git fetch upstream $COMMIT
git checkout FETCH_HEAD -- src/module.py tests/test_module.py
echo "Applied changes"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(solve_sh)
        f.flush()
        info = parse_solve_sh(f.name)
    os.unlink(f.name)
    assert info.commit_hash == "abc123def456"
    assert info.upstream_url == "https://github.com/test/repo.git"


def test_extract_dataclasses_exist():
    """Core data structures are importable."""
    from craft_taskgen.search.extract import SolveShInfo

    info = SolveShInfo(commit_hash="abc123")
    assert info.commit_hash == "abc123"
    assert info.upstream_url == ""
    assert info.checkout_paths == []


# ---------------------------------------------------------------------------
# parse_instruction format branches
# ---------------------------------------------------------------------------


def test_parse_instruction_new_format(tmp_path):
    """New-format instructions (with preamble) have preamble stripped."""
    from craft_taskgen.config import PipelineContext
    from craft_taskgen.search.extract import parse_instruction

    preamble = PipelineContext().instruction_preamble
    path = tmp_path / "instruction.md"
    path.write_text(f"{preamble}\n\nImplement the feature in module.py.\n")
    result = parse_instruction(str(path))
    assert result == "Implement the feature in module.py."
    assert not result.startswith(preamble)


def test_parse_instruction_old_format(tmp_path):
    """Old-format instructions (with # header and ## Environment) are parsed correctly."""
    from craft_taskgen.search.extract import parse_instruction

    path = tmp_path / "instruction.md"
    path.write_text("# Tool Orchestration Task\nDo the thing.\n## Environment\nPython 3.12\n")
    result = parse_instruction(str(path))
    assert result == "Do the thing."

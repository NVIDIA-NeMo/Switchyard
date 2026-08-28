# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the planning-task scorer."""

from __future__ import annotations

import json

from craft_taskgen.planning import scorer as sc


def _default_cfg(**overrides) -> sc.ScorerConfig:
    cfg = sc.ScorerConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _write_trial(trials_dir, task_name, f2p_score, p2p_score=1.0):
    trial = trials_dir / f"{task_name}__abc"
    (trial / "verifier").mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    (trial / "verifier" / "results.json").write_text(
        json.dumps(
            {
                "reward": 1.0 if f2p_score == 1.0 and p2p_score == 1.0 else 0.0,
                "f2p": {
                    "score": f2p_score,
                    "total": 10,
                    "passed": int(f2p_score * 10),
                    "failed_tests": [],
                },
                "p2p": {
                    "score": p2p_score,
                    "total": 5,
                    "passed": int(p2p_score * 5),
                    "failed_tests": [],
                },
            }
        )
    )


def test_score_tasks_reads_both_trial_dirs(tmp_path):
    a_trials = tmp_path / "a"
    b_trials = tmp_path / "b"
    a_trials.mkdir()
    b_trials.mkdir()
    _write_trial(a_trials, "task-a", 0.95, 1.0)
    _write_trial(b_trials, "task-a", 0.60, 1.0)

    scores = sc.score_tasks(["task-a"], a_trials, b_trials)

    assert len(scores) == 1
    s = scores[0]
    assert s.planner_a_f2p == 0.95
    assert s.planner_b_f2p == 0.60
    assert round(s.delta, 2) == 0.35


def test_score_tasks_missing_trial_zeroes_score(tmp_path):
    a_trials = tmp_path / "a"
    b_trials = tmp_path / "b"
    a_trials.mkdir()
    b_trials.mkdir()
    _write_trial(a_trials, "task-a", 0.95)

    scores = sc.score_tasks(["task-a"], a_trials, b_trials)

    s = scores[0]
    assert s.planner_a_f2p == 0.95
    assert s.planner_b_f2p == 0.0
    assert round(s.delta, 2) == 0.95


def test_write_back_records_scores_without_planning_task_flag(tmp_path):
    cand = {
        "task_name": "task-a",
        "repo": "x/y",
        "pr": 1,
        "spec": "s",
        "src_files": [],
        "test_files": [],
        "test_command": "pytest",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "docker": {"install": "pip install -e ."},
    }
    path = tmp_path / "task-a.json"
    path.write_text(json.dumps(cand))

    s = sc.Score(
        task_name="task-a",
        planner_a_f2p=0.95,
        planner_b_f2p=0.6,
        planner_a_p2p=1.0,
        planner_b_p2p=1.0,
        delta=0.35,
    )
    sc.write_back(tmp_path, [s], _default_cfg())

    updated = json.loads(path.read_text())
    block = updated["planning_scores"]
    assert block["planner_a_f2p"] == 0.95
    assert block["planner_b_f2p"] == 0.6
    assert block["delta"] == 0.35
    assert "planning_task" not in updated
    assert "tentative_verdict" not in block


def test_write_back_uses_task_name_field_not_filename(tmp_path):
    cand = {
        "task_name": "owner__repo-123",
        "repo": "owner/repo",
        "pr": 123,
        "spec": "s",
        "src_files": [],
        "test_files": [],
        "test_command": "pytest",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "docker": {"install": "pip install -e ."},
    }
    path = tmp_path / "repo-123.json"
    path.write_text(json.dumps(cand))

    s = sc.Score(
        task_name="owner__repo-123",
        planner_a_f2p=0.8,
        planner_b_f2p=0.2,
        planner_a_p2p=1.0,
        planner_b_p2p=1.0,
        delta=0.6,
    )
    sc.write_back(tmp_path, [s], _default_cfg())

    updated = json.loads(path.read_text())
    assert updated["planning_scores"]["delta"] == 0.6


def test_resolve_harbor_output_dir_picks_newest_timestamp_subdir(tmp_path):
    import os
    import time

    from craft_taskgen.planning.harbor import resolve_harbor_output_dir

    trials = tmp_path / "trials"
    old_ts = trials / "2026-04-16__10-00-00"
    new_ts = trials / "2026-04-17__11-34-16"
    old_trial = old_ts / "task-a__xxx"
    new_trial = new_ts / "task-a__yyy"
    old_trial.mkdir(parents=True)
    new_trial.mkdir(parents=True)
    (old_trial / "result.json").write_text("{}")
    (new_trial / "result.json").write_text("{}")

    now = time.time()
    os.utime(old_ts, (now - 3600, now - 3600))
    os.utime(new_ts, (now, now))

    resolved = resolve_harbor_output_dir(trials)
    assert resolved == new_ts


def test_resolve_harbor_output_dir_passes_through_flat_layout(tmp_path):
    from craft_taskgen.planning.harbor import resolve_harbor_output_dir

    trials = tmp_path / "trials"
    trial = trials / "task-a__xxx"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}")

    resolved = resolve_harbor_output_dir(trials)
    assert resolved == trials


def test_collect_trial_dirs_handles_timestamped_layout(tmp_path):
    from craft_taskgen.planning.harbor import collect_trial_dirs

    trials = tmp_path / "trials"
    trial = trials / "2026-04-17__11-34-16" / "aiogram-1642__di7YAMg"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "results.json").write_text("{}")

    by_task = collect_trial_dirs(trials)
    assert "aiogram-1642" in by_task
    assert by_task["aiogram-1642"].name.startswith("aiogram-1642__")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native search Harbor adapter."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from craft_taskgen.adapters.search_native import converter

MANIFEST = {
    "aiohttp": {
        "url": "https://github.com/aio-libs/aiohttp.git",
        "commit": "abc123def456",
    },
}


def _make_candidate(
    uuid: str = "d1e5bec3abcdef01",
    repo: str = "aiohttp",
    tier: str = "medium",
    instruction: str = "Where does aiohttp decompress response bodies?",
    files: list[str] | None = None,
    functions: list[str] | None = None,
    explanation: str = "The decompression happens in compression_utils.py.",
) -> dict:
    return {
        "task": {
            "id": uuid,
            "repo": repo,
            "tier": tier,
            "instruction": instruction,
            "gold_answer": {
                "files": files if files is not None else ["aiohttp/compression_utils.py"],
                "functions": functions if functions is not None else ["aiohttp.compression_utils.decompress"],
                "explanation": explanation,
            },
        }
    }


def _make_manifest_file(tmp: str) -> str:
    path = os.path.join(tmp, "manifest.json")
    with open(path, "w") as f:
        json.dump(MANIFEST, f)
    return path


def _make_candidates_tree(tmp: str, candidates: list[tuple[str, dict]]) -> str:
    """Write candidates to `{candidates_dir}/{repo}/{uuid}.json`."""
    root = os.path.join(tmp, "candidates")
    for uuid, cand in candidates:
        repo = cand["task"]["repo"]
        repo_dir = os.path.join(root, repo)
        os.makedirs(repo_dir, exist_ok=True)
        with open(os.path.join(repo_dir, f"{uuid}.json"), "w") as f:
            json.dump(cand, f)
    return root


# ---------------------------------------------------------------------------
# convert_task
# ---------------------------------------------------------------------------


def test_convert_task_produces_expected_layout():
    with tempfile.TemporaryDirectory() as tmp:
        candidate = _make_candidate()
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        task_dir = converter.convert_task(candidate, MANIFEST, out)
        tid = os.path.basename(task_dir)
        assert tid == "craft-aiohttp-d1e5bec3"

        # Directory structure
        assert os.path.isfile(os.path.join(task_dir, "task.toml"))
        assert os.path.isfile(os.path.join(task_dir, "instruction.md"))
        assert os.path.isfile(os.path.join(task_dir, "environment", "Dockerfile"))
        assert os.path.isfile(os.path.join(task_dir, "tests", "test.sh"))
        assert os.path.isfile(os.path.join(task_dir, "tests", "test_runner.py"))
        assert os.path.isfile(os.path.join(task_dir, "tests", "gold_answer.json"))
        assert os.path.isfile(os.path.join(task_dir, "solution", "solve.sh"))


def test_dockerfile_substitutes_repo_url_and_commit():
    with tempfile.TemporaryDirectory() as tmp:
        candidate = _make_candidate()
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        task_dir = converter.convert_task(candidate, MANIFEST, out)
        with open(os.path.join(task_dir, "environment", "Dockerfile")) as f:
            dockerfile = f.read()

        assert "https://github.com/aio-libs/aiohttp.git" in dockerfile
        assert "abc123def456" in dockerfile
        assert "{repo_url}" not in dockerfile
        assert "{repo_commit}" not in dockerfile


def test_gold_answer_and_solve_sh_contain_gold():
    with tempfile.TemporaryDirectory() as tmp:
        candidate = _make_candidate(
            files=["a.py", "b.py"],
            functions=["mod.f1", "mod.f2"],
            explanation="hello",
        )
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        task_dir = converter.convert_task(candidate, MANIFEST, out)

        with open(os.path.join(task_dir, "tests", "gold_answer.json")) as f:
            gold = json.load(f)
        assert gold["files"] == ["a.py", "b.py"]
        assert gold["functions"] == ["mod.f1", "mod.f2"]

        with open(os.path.join(task_dir, "solution", "solve.sh")) as f:
            solve = f.read()
        assert '"files"' in solve
        assert "a.py" in solve
        assert "mod.f1" in solve


def test_task_toml_reflects_tier_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        task_dir = converter.convert_task(_make_candidate(tier="easy"), MANIFEST, out)
        with open(os.path.join(task_dir, "task.toml")) as f:
            toml = f.read()
        assert 'difficulty = "easy"' in toml
        assert "timeout_sec = 600" in toml


def test_missing_repo_in_manifest_raises():
    with tempfile.TemporaryDirectory() as tmp:
        candidate = _make_candidate(repo="not-in-manifest")
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        with pytest.raises(ValueError, match="not found in manifest"):
            converter.convert_task(candidate, MANIFEST, out)


def test_missing_required_fields_raise():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "harbor")
        os.makedirs(out, exist_ok=True)

        # Missing top-level 'task'
        with pytest.raises(ValueError, match="missing 'task' key"):
            converter.convert_task({"other": {}}, MANIFEST, out)

        # Missing gold_answer
        bad = _make_candidate()
        del bad["task"]["gold_answer"]
        with pytest.raises(ValueError, match="gold_answer"):
            converter.convert_task(bad, MANIFEST, out)


# ---------------------------------------------------------------------------
# run_convert
# ---------------------------------------------------------------------------


def test_run_convert_writes_registry_and_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        candidates_dir = _make_candidates_tree(
            tmp,
            [
                ("aaaaaaaabbbbb", _make_candidate(uuid="aaaaaaaabbbbb")),
                ("ccccccccddddd", _make_candidate(uuid="ccccccccddddd")),
            ],
        )
        manifest_path = _make_manifest_file(tmp)
        out = os.path.join(tmp, "harbor")

        result = converter.run_convert(candidates_dir, manifest_path, out)
        assert result["converted"] == 2
        assert set(result["task_ids"]) == {"craft-aiohttp-aaaaaaaa", "craft-aiohttp-cccccccc"}

        with open(os.path.join(out, "registry.json")) as f:
            registry = json.load(f)
        assert registry[0]["name"] == "craft-search"
        assert set(registry[0]["tasks"]) == set(result["task_ids"])


def test_run_convert_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        candidates_dir = _make_candidates_tree(
            tmp,
            [
                ("aaaaaaaa11111", _make_candidate(uuid="aaaaaaaa11111")),
                ("aaaaaaaa22222", _make_candidate(uuid="aaaaaaaa22222")),
                ("aaaaaaaa33333", _make_candidate(uuid="aaaaaaaa33333")),
            ],
        )
        manifest_path = _make_manifest_file(tmp)
        out = os.path.join(tmp, "harbor")

        result = converter.run_convert(candidates_dir, manifest_path, out, limit=2)
        assert result["converted"] == 2


def test_run_convert_archives_stale_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        # Seed output dir with a stale task from a previous run
        out = os.path.join(tmp, "harbor")
        stale_tid = "craft-aiohttp-oldstale"
        os.makedirs(os.path.join(out, stale_tid), exist_ok=True)
        Path(out, stale_tid, "marker.txt").touch()

        candidates_dir = _make_candidates_tree(tmp, [("newnewnewnew", _make_candidate(uuid="newnewnewnew"))])
        manifest_path = _make_manifest_file(tmp)

        result = converter.run_convert(candidates_dir, manifest_path, out)
        assert result["archived"] == [stale_tid]

        # Stale task moved to a `_stale_<ts>/` subdir
        assert not os.path.isdir(os.path.join(out, stale_tid))
        stale_subdirs = [d for d in os.listdir(out) if d.startswith("_stale_")]
        assert len(stale_subdirs) == 1
        assert os.path.isdir(os.path.join(out, stale_subdirs[0], stale_tid))

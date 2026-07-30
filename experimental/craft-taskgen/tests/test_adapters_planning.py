# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the planning-track Harbor adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from craft_taskgen.adapters.planning import converter


def _make_candidate(
    task_name: str = "hugapi__hug-651",
    repo: str = "hugapi/hug",
    spec: str = "Add a context factory API.",
) -> dict:
    return {
        "task_name": task_name,
        "abbrev": task_name,
        "repo": repo,
        "pr": 651,
        "category": "feature",
        "parent_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "merge_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "spec": spec,
        "src_files": ["hug/api.py", "hug/interface.py"],
        "test_files": ["tests/test_context_factory.py"],
        "gold_symbols": ["API.context_factory", "API.delete_context"],
        "test_command": "python3 -m pytest tests/test_context_factory.py -v --tb=short -o 'addopts='",
        "fail_to_pass": ["tests/test_context_factory.py::test_factory"],
        "pass_to_pass": [],
        "docker": {
            "python": "3.9",
            "pre_install": ["sed -i 's/foo/bar/' hug/_async.py"],
            "install": "pip install -e . && pip install pytest",
        },
        "removed_files": [],
        "pinned_requirements": "pytest==8.3.4\npytest-mock==3.14.0\nmarshmallow==3.26.0\n",
    }


def test_validate_candidate_accepts_complete_payload() -> None:
    converter._validate_candidate(_make_candidate())


def test_validate_candidate_rejects_missing_fields() -> None:
    bad = _make_candidate()
    del bad["spec"]
    with pytest.raises(ValueError, match="spec"):
        converter._validate_candidate(bad)


def test_validate_candidate_rejects_docker_without_install() -> None:
    bad = _make_candidate()
    bad["docker"] = {"python": "3.11"}
    with pytest.raises(ValueError, match="install"):
        converter._validate_candidate(bad)


def test_convert_single_rejects_candidate_without_pinned_requirements() -> None:
    """Every task must ship pinned. No opt-out. Adapter refuses to build."""
    import tempfile

    candidate = _make_candidate()
    del candidate["pinned_requirements"]

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        (cache / "hugapi__hug").mkdir(parents=True)
        templates = converter._load_all_templates()
        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            with pytest.raises(ValueError, match="pinned_requirements"):
                converter._convert_single(candidate, out, cache, templates)


def test_convert_single_emits_requirements_lock_into_environment_dir() -> None:
    """The lock file lands alongside the Dockerfile, ready for COPY at build time."""
    import tempfile

    candidate = _make_candidate()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        (cache / "hugapi__hug").mkdir(parents=True)
        templates = converter._load_all_templates()
        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            converter._convert_single(candidate, out, cache, templates)
        env_dir = out / "hugapi__hug-651" / "environment"
        assert (env_dir / "Dockerfile").exists()
        assert (env_dir / "requirements.lock").exists()
        assert "pytest==8.3.4" in (env_dir / "requirements.lock").read_text()


def test_convert_single_bakes_pinned_agents_into_dockerfile() -> None:
    """Planning tasks ship agent CLIs at pinned versions so trial-time agent
    behavior is reproducible. Opt-in via bake_agents=True on the spec."""
    import tempfile

    from craft_taskgen.adapters._docker import CLAUDE_CODE_VERSION, CODEX_VERSION, OPENCODE_VERSION

    candidate = _make_candidate()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        (cache / "hugapi__hug").mkdir(parents=True)
        templates = converter._load_all_templates()
        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            converter._convert_single(candidate, out, cache, templates)
        dockerfile = (out / "hugapi__hug-651" / "environment" / "Dockerfile").read_text()
        assert f"@anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}" in dockerfile
        assert f"@openai/codex@{CODEX_VERSION}" in dockerfile
        assert f"opencode-ai@{OPENCODE_VERSION}" in dockerfile


def test_build_f2p_command_strips_verbose_and_tb() -> None:
    candidate = _make_candidate()
    cmd = converter._build_f2p_command(candidate)
    assert "python3 -m pytest" in cmd
    assert "--tb=short" not in cmd
    # The trailing -v is added exactly once by the converter regardless of input verbosity.
    assert cmd.count("-v") == 1
    assert "-o 'addopts='" in cmd
    assert "-p no:cacheprovider" in cmd


def test_build_f2p_command_rewrites_code_path_to_repo() -> None:
    candidate = _make_candidate()
    candidate["test_command"] = "python3 -m pytest /code/tests/foo.py"
    cmd = converter._build_f2p_command(candidate)
    assert "/code/" not in cmd
    assert "/repo/tests/foo.py" in cmd


def test_build_dockerfile_includes_pre_install_and_install() -> None:
    from craft_taskgen.adapters._docker import build_dockerfile, spec_from_candidate

    candidate = _make_candidate()
    dockerfile = build_dockerfile(spec_from_candidate(candidate))
    assert "FROM python:3.9-slim-bookworm@sha256:" in dockerfile
    assert "pip install --no-cache-dir uv==0.7.12" in dockerfile
    assert "git clone https://github.com/hugapi/hug.git" in dockerfile
    assert "aaaaaaaa" in dockerfile  # parent_sha (first chars)
    assert "sed -i 's/foo/bar/' hug/_async.py" in dockerfile
    assert "pip install -e . && pip install pytest" in dockerfile
    assert "mkdir -p /repo/output" in dockerfile
    # Strict-by-default: Dockerfile installs pins with --no-deps.
    assert "uv pip install --system --no-deps -r /tmp/requirements.lock" in dockerfile


def test_main_package_name_normalizes() -> None:
    from craft_taskgen.adapters._docker import main_package_name

    assert main_package_name("hugapi/hug") == "hug"
    assert main_package_name("aio-libs/aiohttp") == "aiohttp"
    assert main_package_name("scrapy/scrapy") == "scrapy"
    # hyphens become underscores for Python package names
    assert main_package_name("python-poetry/poetry") == "poetry"


def test_convert_single_emits_expected_structure() -> None:
    """End-to-end on a single candidate with git calls mocked."""
    candidate = _make_candidate()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        cache.mkdir()
        # Pre-create the clone dir so _clone_or_reuse skips the git clone.
        (cache / "hugapi__hug").mkdir()

        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test file\n"),
        ):
            templates = converter._load_all_templates()
            converter._convert_single(candidate, out, cache, templates)

        task_dir = out / "hugapi__hug-651"
        assert task_dir.is_dir()
        assert (task_dir / "task.toml").exists()
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "environment" / "Dockerfile").exists()
        assert (task_dir / "solution" / "solve.sh").exists()
        assert (task_dir / "tests" / "test.sh").exists()
        assert (task_dir / "tests" / "score.py").exists()
        assert (task_dir / "tests" / "fail_to_pass.txt").exists()
        assert (task_dir / "tests" / "pass_to_pass.txt").exists()
        assert (task_dir / "tests" / "verify_hugapi__hug-651.sh").exists()

        # Spec ends up in instruction.md, not the planner template.
        instruction = (task_dir / "instruction.md").read_text()
        assert "Add a context factory API." in instruction
        assert "implementation plan" not in instruction  # no planner framing

        # Binary-reward gate is present in score.py.
        score_py = (task_dir / "tests" / "score.py").read_text()
        assert "binary_reward" in score_py
        assert "not f2p_failed and not p2p_failed" in score_py

        # fail_to_pass list is written verbatim.
        f2p = (task_dir / "tests" / "fail_to_pass.txt").read_text()
        assert "tests/test_context_factory.py::test_factory" in f2p


def test_run_convert_skips_invalid_candidate() -> None:
    """A bad candidate should be skipped without aborting the run."""
    good = _make_candidate(task_name="good__task-1")
    bad = _make_candidate(task_name="bad__task-2")
    del bad["spec"]

    with tempfile.TemporaryDirectory() as td:
        candidates = Path(td) / "cands"
        candidates.mkdir()
        (candidates / "good__task-1.json").write_text(json.dumps(good))
        (candidates / "bad__task-2.json").write_text(json.dumps(bad))

        out = Path(td) / "out"
        cache = Path(td) / "cache"
        (cache / "hugapi__hug").mkdir(parents=True)

        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            result = converter.run_convert(
                candidates_dir=str(candidates),
                output_dir=str(out),
                repo_cache=str(cache),
            )

        assert result["converted"] == 1
        assert result["task_ids"] == ["good__task-1"]
        assert result["skipped"] == ["bad__task-2"]
        assert (out / "good__task-1").is_dir()
        assert not (out / "bad__task-2").exists()
        registry = json.loads((out / "registry.json").read_text())
        assert registry[0]["tasks"] == [{"name": "good__task-1", "path": "good__task-1"}]


def test_convert_single_applies_candidate_resource_overrides() -> None:
    """Per-task memory/storage/build_timeout overrides flow from candidate to task.toml."""
    candidate = _make_candidate()
    candidate["memory_mb"] = 8192
    candidate["storage_mb"] = 20480
    candidate["build_timeout_sec"] = 1800.0

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        cache.mkdir()
        (cache / "hugapi__hug").mkdir()

        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            templates = converter._load_all_templates()
            converter._convert_single(candidate, out, cache, templates)

        toml = (out / "hugapi__hug-651" / "task.toml").read_text()
        assert "memory_mb = 8192" in toml
        assert "storage_mb = 20480" in toml
        assert "build_timeout_sec = 1800.0" in toml


def test_convert_single_uses_default_resources_when_not_declared() -> None:
    """Candidates without resource fields get the adapter's conservative defaults."""
    candidate = _make_candidate()
    # No memory_mb / storage_mb / build_timeout_sec on the candidate.

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        out.mkdir()
        cache.mkdir()
        (cache / "hugapi__hug").mkdir()

        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            templates = converter._load_all_templates()
            converter._convert_single(candidate, out, cache, templates)

        toml = (out / "hugapi__hug-651" / "task.toml").read_text()
        assert f"memory_mb = {converter._DEFAULT_MEMORY_MB}" in toml
        assert f"storage_mb = {converter._DEFAULT_STORAGE_MB}" in toml
        assert f"build_timeout_sec = {converter._DEFAULT_BUILD_TIMEOUT_SEC}" in toml


def test_run_convert_respects_limit() -> None:
    candidates = [_make_candidate(task_name=f"repo__task-{i}") for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        candidates_dir = Path(td) / "cands"
        candidates_dir.mkdir()
        for c in candidates:
            (candidates_dir / f"{c['task_name']}.json").write_text(json.dumps(c))

        out = Path(td) / "out"
        cache = Path(td) / "cache"
        (cache / "hugapi__hug").mkdir(parents=True)

        with (
            mock.patch.object(converter, "_git_fetch"),
            mock.patch.object(converter, "_git_show", return_value=b"# test\n"),
        ):
            result = converter.run_convert(
                candidates_dir=str(candidates_dir),
                output_dir=str(out),
                repo_cache=str(cache),
                limit=2,
            )
        assert result["converted"] == 2

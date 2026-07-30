# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dockerfile builder: tests for the reproducibility-focused install layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from craft_taskgen.adapters import _docker

_DEFAULT_PINS = "pytest==8.3.4\npytest-mock==3.14.0\n"


def _cand(with_pins: bool = True, **overrides) -> dict:
    base = {
        "task_name": "hugapi__hug-651",
        "repo": "hugapi/hug",
        "parent_sha": "a" * 40,
        "docker": {
            "python": "3.11",
            "install": "uv pip install --system -e .",
            "pre_install": ["sed -i 's/x/y/' pkg/_m.py"],
            "test_deps": "pytest pytest-mock",
        },
    }
    if with_pins:
        base["pinned_requirements"] = _DEFAULT_PINS
    base.update(overrides)
    return base


def test_build_dockerfile_uses_base_template_for_python_version():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    # Base image pinned by digest so the base layer is reproducible.
    assert "FROM python:3.11-slim-bookworm@sha256:" in df
    # uv itself is pinned so the installer is reproducible too.
    assert "pip install --no-cache-dir uv==0.7.12" in df


def test_build_dockerfile_clones_repo_at_parent_sha():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    assert "git clone https://github.com/hugapi/hug.git" in df
    assert f"git checkout {'a' * 40}" in df


def test_build_dockerfile_applies_pre_install_patches():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    assert "RUN sed -i 's/x/y/' pkg/_m.py" in df


def test_build_dockerfile_installs_pins_with_no_deps():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    assert "COPY requirements.lock /tmp/requirements.lock" in df
    assert "uv pip install --system --no-deps -r /tmp/requirements.lock" in df
    # Floating test_deps line is suppressed when pins are present.
    assert "uv pip install --system pytest pytest-mock" not in df


def test_write_environment_emits_dockerfile_and_lockfile():
    with tempfile.TemporaryDirectory() as td:
        env = Path(td) / "environment"
        _docker.write_environment(env, "FROM python:3.11\n", "pytest==8.3.4\n")
        assert (env / "Dockerfile").read_text() == "FROM python:3.11\n"
        assert (env / "requirements.lock").read_text() == "pytest==8.3.4\n"


def test_write_environment_appends_trailing_newline_to_lockfile():
    with tempfile.TemporaryDirectory() as td:
        env = Path(td) / "environment"
        _docker.write_environment(env, "FROM python:3.11\n", "pytest==8.3.4")
        assert (env / "requirements.lock").read_text().endswith("\n")


def test_spec_from_candidate_rejects_missing_install():
    candidate = _cand()
    del candidate["docker"]["install"]
    with pytest.raises(ValueError, match="docker.install"):
        _docker.spec_from_candidate(candidate)


def test_spec_from_candidate_refuses_unpinned():
    """Every task must ship with pins. No opt-out."""
    with pytest.raises(ValueError, match="pinned_requirements"):
        _docker.spec_from_candidate(_cand(with_pins=False))


def test_spec_from_candidate_accepts_pins_from_candidate_field():
    spec = _docker.spec_from_candidate(_cand())
    assert spec.pinned_requirements == _DEFAULT_PINS


def test_spec_from_candidate_accepts_pins_from_parameter():
    """Pins can be threaded in by the caller (e.g., a lock-deps tool) without
    requiring the candidate to be rewritten first."""
    spec = _docker.spec_from_candidate(
        _cand(with_pins=False),
        pinned_requirements=_DEFAULT_PINS,
    )
    assert spec.pinned_requirements == _DEFAULT_PINS


def test_spec_from_candidate_default_python_when_absent():
    candidate = _cand()
    del candidate["docker"]["python"]
    spec = _docker.spec_from_candidate(candidate)
    assert spec.python == "3.11"


def test_main_package_name_normalizes():
    assert _docker.main_package_name("hugapi/hug") == "hug"
    assert _docker.main_package_name("python-poetry/poetry") == "poetry"
    assert _docker.main_package_name("aio-libs/aiohttp") == "aiohttp"


def test_bake_agents_false_omits_agent_layer():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    assert "@anthropic-ai/claude-code" not in df
    assert "@openai/codex" not in df
    assert "opencode-ai" not in df
    assert "nodesource" not in df.lower()


def test_bake_agents_true_installs_all_three_agents_pinned():
    candidate = _cand()
    spec = _docker.spec_from_candidate(candidate)
    spec.bake_agents = True
    df = _docker.build_dockerfile(spec)
    assert f"@anthropic-ai/claude-code@{_docker.CLAUDE_CODE_VERSION}" in df
    assert f"@openai/codex@{_docker.CODEX_VERSION}" in df
    assert f"opencode-ai@{_docker.OPENCODE_VERSION}" in df
    assert "setup_22.x" in df  # NodeSource setup for the npm runtime
    # Agent layer sits AFTER the pinned install so the python env is stable
    # first.
    agent_idx = df.index("@anthropic-ai/claude-code")
    pinned_idx = df.index("uv pip install --system --no-deps")
    assert pinned_idx < agent_idx


def test_produce_manifest_captures_pinned_surface():
    spec = _docker.spec_from_candidate(_cand())
    spec.bake_agents = True
    df = _docker.build_dockerfile(spec)
    m = _docker.produce_manifest(spec, adapter="planning", dockerfile=df)
    assert m["schema_version"] == _docker.MANIFEST_SCHEMA_VERSION
    assert m["adapter"] == "planning"
    assert m["base_image"]["reference"].startswith("python:3.11-slim-bookworm")
    assert m["base_image"]["digest"].startswith("sha256:")
    assert m["installer"] == {"name": "uv", "version": "0.7.12"}
    assert m["python"] == "3.11"
    assert m["agents"]["claude_code"] == _docker.CLAUDE_CODE_VERSION
    assert m["agents"]["codex"] == _docker.CODEX_VERSION
    assert m["agents"]["opencode"] == _docker.OPENCODE_VERSION
    assert m["harbor_commit"] == _docker.HARBOR_COMMIT
    assert m["repo"] == "hugapi/hug"
    assert len(m["pinned_requirements_sha256"]) == 64  # hex sha256
    assert "produced_at" in m


def test_produce_manifest_omits_agents_when_not_baked():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    m = _docker.produce_manifest(spec, adapter="planning", dockerfile=df)
    assert m["agents"] == {}


def test_write_environment_emits_manifest_when_provided():
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    manifest = _docker.produce_manifest(spec, adapter="planning", dockerfile=df)
    with tempfile.TemporaryDirectory() as td:
        env = Path(td) / "environment"
        _docker.write_environment(env, df, spec.pinned_requirements, manifest=manifest)
        assert (env / "manifest.json").exists()
        import json as _json

        on_disk = _json.loads((env / "manifest.json").read_text())
        assert on_disk["adapter"] == "planning"
        assert on_disk["base_image"]["digest"].startswith("sha256:")


def test_dockerfile_order_is_stable():
    """Layer order matters for docker build cache. Lock all consumers by
    asserting the major RUN lines appear in the expected order."""
    spec = _docker.spec_from_candidate(_cand())
    df = _docker.build_dockerfile(spec)
    clone_idx = df.index("git clone")
    pre_idx = df.index("sed -i")
    install_idx = df.index("uv pip install --system -e .")
    lock_idx = df.index("COPY requirements.lock")
    pinned_idx = df.index("uv pip install --system --no-deps")
    assert clone_idx < pre_idx < install_idx < lock_idx < pinned_idx

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for cleanup_task_images against a real Docker daemon.

Marked @pytest.mark.docker so they're skipped by default via
`pytest -m "not docker"` (declared in pyproject.toml). Run locally with
`uv run pytest -m docker` when a daemon is available.

Each test tags a tiny hello-world image under patterns that cleanup_task_images
should (or should not) match, invokes cleanup, and checks which tags survive.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from craft_taskgen.docker import cleanup_task_images

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _list_tags() -> set[str]:
    r = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


@pytest.fixture
def tagged_images():
    """Tag hello-world under 5 names: 3 should be cleaned, 2 should survive."""
    if not _docker_ok():
        pytest.skip("docker daemon unavailable")
    _run(["docker", "pull", "--quiet", "hello-world"])

    # task_dir basename for the fake task. 41 chars total, so basename[:32]
    # = "t2v3-cleanuptest-sample-slug-ext" exercises the Harbor-style pattern.
    task_dir = "/tmp/t2v3-cleanuptest-sample-slug-extra-suffix"
    prefix32 = "t2v3-cleanuptest-sample-slug-ext"  # basename[:32]
    cleanable = [
        "craft-t2v3-cleanuptest-sample-slug-extra-suffix:latest",  # craft-* exact
        f"{prefix32}-abcd123-main:latest",  # Harbor: prefix[:32]+uuid+service
        f"{prefix32}-abcd123-sidecar:latest",  # extra compose service
    ]
    survivors = [
        "unrelated-fixture:latest",
        "test-cleanup-fixture:latest",  # Claude-style residue — not our scope
    ]
    for tag in cleanable + survivors:
        _run(["docker", "tag", "hello-world", tag])

    yield task_dir, cleanable, survivors

    # teardown: force-remove anything still around
    for tag in cleanable + survivors:
        _run(["docker", "image", "rm", "-f", tag])


def test_cleanup_removes_craft_and_harbor_patterns_only(tagged_images):
    task_dir, cleanable, survivors = tagged_images

    before = _list_tags()
    for tag in cleanable + survivors:
        assert tag in before, f"fixture setup failed: {tag} missing"

    asyncio.run(cleanup_task_images(task_dir))

    after = _list_tags()
    for tag in cleanable:
        assert tag not in after, f"expected {tag} removed, still present"
    for tag in survivors:
        assert tag in after, f"unexpectedly removed {tag}"


def test_cleanup_is_idempotent(tagged_images):
    """Running cleanup twice is safe (docker rm -f on nonexistent is a no-op)."""
    task_dir, _, _ = tagged_images
    asyncio.run(cleanup_task_images(task_dir))
    asyncio.run(cleanup_task_images(task_dir))  # must not raise

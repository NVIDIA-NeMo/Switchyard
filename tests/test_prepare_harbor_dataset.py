# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "benchmark" / "prepare_harbor_dataset.py"
HERMES_INSTALLER_FIXTURE = b"#!/usr/bin/env bash\nset -euo pipefail\n"
UV_IMAGE = (
    "ghcr.io/astral-sh/uv"
    "@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff"
)


def _load_generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchyard_prepare_harbor_dataset", GENERATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, name: str, task_toml: str, dockerfile: str | None = None) -> Path:
    task = root / name
    env = task / "environment"
    env.mkdir(parents=True)
    (task / "task.toml").write_text(task_toml)
    if dockerfile is not None:
        (env / "Dockerfile").write_text(dockerfile)
    return task


def _prepare(
    tmp_path: Path,
    source: Path,
    *,
    source_dataset: str = "openthoughts-tblite@2.0",
    prefer_source_dockerfiles: bool = False,
) -> Path:
    module = _load_generator_module()
    module._fetch_hermes_installer = lambda _pins: HERMES_INSTALLER_FIXTURE
    output = tmp_path / "prepared"
    return module.prepare_dataset(
        source_dataset=source_dataset,
        source_dir=source,
        output_dir=output,
        harbor_command="harbor",
        overwrite=False,
        prefer_source_dockerfiles=prefer_source_dockerfiles,
    )


def test_find_exported_dataset_root_uses_harbor_package_short_name(tmp_path: Path) -> None:
    module = _load_generator_module()
    download_root = tmp_path / "_downloads"
    _write_task(download_root / "terminal-bench-2", "example-task", "[environment]\n")

    found = module._find_exported_dataset_root(
        download_root,
        "terminal-bench/terminal-bench-2",
    )

    assert found == download_root / "terminal-bench-2"


def test_find_exported_dataset_root_keeps_legacy_dataset_name(tmp_path: Path) -> None:
    module = _load_generator_module()
    download_root = tmp_path / "_downloads"
    _write_task(download_root / "openthoughts-tblite", "example-task", "[environment]\n")

    found = module._find_exported_dataset_root(download_root, "openthoughts-tblite@2.0")

    assert found == download_root / "openthoughts-tblite"


def test_find_exported_dataset_root_ignores_other_exported_datasets(tmp_path: Path) -> None:
    module = _load_generator_module()
    download_root = tmp_path / "_downloads"
    _write_task(download_root / "openthoughts-tblite", "lite-task", "[environment]\n")
    _write_task(download_root / "terminal-bench-2", "tb2-task", "[environment]\n")

    found = module._find_exported_dataset_root(
        download_root,
        "terminal-bench/terminal-bench-2",
    )

    assert found == download_root / "terminal-bench-2"


def test_terminal_bench_2_dataset_adds_proxy_allowlist_hosts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "tb2-task", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(
        tmp_path,
        source,
        source_dataset="terminal-bench/terminal-bench-2",
    )
    allowlist = (output / "tb2-task" / "environment" / "proxy" / "allowlist-base.txt").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    for host in (
        "archive.ubuntu.com",
        "deb.debian.org",
        "pypi.org",
        "files.pythonhosted.org",
        "github.com",
        "huggingface.co",
        "download.pytorch.org",
        "download-r2.pytorch.org",
        "cloud.r-project.org",
        "www.cs.toronto.edu",
        "www.rcsb.org",
    ):
        assert host in allowlist
        assert host in manifest["closed_book"]["proxy_allowlist_hosts"]


def test_terminal_bench_2_1_dataset_reuses_terminal_bench_2_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "tb21-task", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(
        tmp_path,
        source,
        source_dataset="terminal-bench/terminal-bench-2-1",
    )
    allowlist = (output / "tb21-task" / "environment" / "proxy" / "allowlist-base.txt").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    for host in (
        "archive.ubuntu.com",
        "pypi.org",
        "github.com",
        "huggingface.co",
        "download.pytorch.org",
    ):
        assert host in allowlist
        assert host in manifest["closed_book"]["proxy_allowlist_hosts"]


def test_swe_bench_pro_dataset_keeps_agent_proxy_allowlist_empty(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "swe-task", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(
        tmp_path,
        source,
        source_dataset="cais/swebenchpro",
    )
    allowlist = (output / "swe-task" / "environment" / "proxy" / "allowlist-base.txt").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    for host in (
        "pypi.org",
        "github.com",
        "registry.npmjs.org",
    ):
        assert host not in allowlist
    assert manifest["closed_book"]["proxy_allowlist_hosts"] == []


def test_legacy_dataset_keeps_proxy_allowlist_empty(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "lite-task", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(tmp_path, source)
    allowlist = (output / "lite-task" / "environment" / "proxy" / "allowlist-base.txt").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    assert "pypi.org" not in allowlist
    assert "archive.ubuntu.com" not in allowlist
    assert manifest["closed_book"]["proxy_allowlist_hosts"] == []


def test_prebuilt_docker_image_task_becomes_derived_dockerfile(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "prebuilt-task",
        '[environment]\ndocker_image = "python:3.12-slim"\n',
    )

    output = _prepare(tmp_path, source)
    task = output / "prebuilt-task"

    assert "docker_image" not in (task / "task.toml").read_text()
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM --platform=$BUILDPLATFORM {UV_IMAGE} AS switchyard_uv_build\n")
    assert "\nFROM python:3.12-slim\nUSER root\n" in dockerfile
    assert "@anthropic-ai/claude-code@2.1.211" in dockerfile
    assert "@openai/codex@0.144.5" in dockerfile
    assert "opencode-ai@1.18.3" in dockerfile


def test_arm_policy_rebuilds_prebuilt_task_from_source_dockerfile(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "prebuilt-task",
        '[environment]\ndocker_image = "amd64-only/task:latest"\n',
        "FROM ubuntu:24.04\nRUN echo source-environment\n",
    )

    output = _prepare(tmp_path, source, prefer_source_dockerfiles=True)
    task = output / "prebuilt-task"
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    assert "docker_image" not in (task / "task.toml").read_text()
    assert "\nFROM ubuntu:24.04\nRUN echo source-environment\n" in dockerfile
    assert "FROM amd64-only/task:latest" not in dockerfile
    assert manifest["closed_book"]["prefer_source_dockerfiles"] is True
    assert manifest["tasks"][0]["docker_image_source"] == "amd64-only/task:latest"
    assert manifest["tasks"][0]["docker_image_removed"] is True
    assert manifest["tasks"][0]["task_image_build_source"] == "source-dockerfile"


def test_arm_policy_keeps_prebuilt_image_when_source_dockerfile_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "prebuilt-task",
        '[environment]\ndocker_image = "amd64-only/task:latest"\n',
    )

    output = _prepare(tmp_path, source, prefer_source_dockerfiles=True)
    task = output / "prebuilt-task"
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    assert "\nFROM amd64-only/task:latest\nUSER root\n" in dockerfile
    assert manifest["tasks"][0]["task_image_build_source"] == "prebuilt-image"


@pytest.mark.parametrize(
    ("architecture", "expected"),
    (("aarch64", True), ("arm64", True), ("x86_64", False), ("AMD64", False)),
)
def test_source_dockerfile_policy_defaults_on_arm(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    expected: bool,
) -> None:
    module = _load_generator_module()
    monkeypatch.setattr(module.platform, "machine", lambda: architecture)

    assert module._prefer_source_dockerfiles() is expected


def test_dockerfile_only_task_gets_prebake_layer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "dockerfile-task",
        "[environment]\n",
        "FROM ubuntu:22.04\nRUN echo task\n",
    )

    output = _prepare(tmp_path, source)
    dockerfile = (output / "dockerfile-task" / "environment" / "Dockerfile").read_text()

    assert dockerfile.startswith(f"FROM --platform=$BUILDPLATFORM {UV_IMAGE} AS switchyard_uv_build\n")
    assert "\nFROM ubuntu:22.04\nRUN echo task\n" in dockerfile
    assert "COPY switchyard-hermes-install.sh /tmp/switchyard-hermes-install.sh" in dockerfile
    assert "COPY --from=switchyard_uv_build /uv /root/.hermes/bin/uv" in dockerfile
    assert "raw.githubusercontent.com/NousResearch/hermes-agent" not in dockerfile
    assert "astral.sh/uv/install.sh" not in dockerfile
    assert r"grep -E '^uv 0\.12\.9($| )'" in dockerfile
    assert "Hermes prebake requires native task images" in dockerfile
    assert "SWITCHYARD_HERMES_PYTHON" not in dockerfile
    assert "SWITCHYARD_PREBAKED_AGENT_VERSIONS" in dockerfile
    assert "/usr/local/lib/node_modules/npm" in dockerfile
    assert "node-v20.11.1-linux-$node_arch.tar.gz" in dockerfile
    assert (
        output
        / "dockerfile-task"
        / "environment"
        / "switchyard-hermes-install.sh"
    ).read_bytes() == HERMES_INSTALLER_FIXTURE


def test_uv_stage_preserves_parser_directives_and_global_args(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "dockerfile-task",
        "[environment]\n",
        "# syntax=docker/dockerfile:1\nARG BASE=ubuntu:22.04\nFROM ${BASE}\n",
    )

    output = _prepare(tmp_path, source)
    dockerfile = (output / "dockerfile-task" / "environment" / "Dockerfile").read_text()

    assert dockerfile.startswith(
        "# syntax=docker/dockerfile:1\n"
        "ARG BASE=ubuntu:22.04\n"
        f"FROM --platform=$BUILDPLATFORM {UV_IMAGE} AS switchyard_uv_build\n"
        "FROM ${BASE}\n"
    )


def test_prepare_fetches_hermes_installer_once_for_all_tasks(tmp_path: Path) -> None:
    module = _load_generator_module()
    source = tmp_path / "source"
    _write_task(source, "task-a", "[environment]\n", "FROM ubuntu:22.04\n")
    _write_task(source, "task-b", "[environment]\n", "FROM ubuntu:22.04\n")
    fetched_versions: list[str] = []

    def fetch_once(pins: dict[str, str]) -> bytes:
        fetched_versions.append(pins["HERMES_VERSION"])
        return HERMES_INSTALLER_FIXTURE

    module._fetch_hermes_installer = fetch_once
    output = module.prepare_dataset(
        source_dataset="openthoughts-tblite@2.0",
        source_dir=source,
        output_dir=tmp_path / "prepared",
        harbor_command="harbor",
        overwrite=False,
    )

    assert fetched_versions == ["3c27eb6234bf91b8ceee9e9071591b31e9b148cb"]
    for task in ("task-a", "task-b"):
        assert (
            output / task / "environment" / "switchyard-hermes-install.sh"
        ).read_bytes() == HERMES_INSTALLER_FIXTURE


def test_generated_compose_contains_closed_book_proxy_topology(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(
        source,
        "compose-task",
        "[environment]\n",
        "FROM ubuntu:22.04\n",
    )

    output = _prepare(tmp_path, source)
    dockerfile = (output / "compose-task" / "environment" / "Dockerfile").read_text()
    entrypoint = output / "compose-task" / "environment" / "switchyard-agent-entrypoint.sh"

    assert entrypoint.is_file()
    entrypoint_text = entrypoint.read_text()
    assert 'PROXY_CA="/etc/proxy-ca/ca-cert.pem"' in entrypoint_text
    assert "test -f" in entrypoint_text
    assert "update-ca-certificates" in entrypoint_text
    assert "SWITCHYARD_PROXY_CA" not in entrypoint_text
    assert "update-ca-trust" not in entrypoint_text
    assert "skipping CA install" not in entrypoint_text
    assert "COPY switchyard-agent-entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/switchyard-agent-entrypoint.sh"]' in dockerfile

    compose = yaml.safe_load(
        (output / "compose-task" / "environment" / "docker-compose.yaml").read_text()
    )

    assert {"main", "proxy"} <= set(compose["services"])
    assert compose["services"]["main"]["networks"] == ["agent-internal"]
    assert "switchyard-egress" in compose["services"]["proxy"]["networks"]
    assert "agent-internal" in compose["services"]["proxy"]["networks"]
    assert "extra_hosts" not in compose["services"]["proxy"]
    assert compose["networks"]["agent-internal"]["internal"] is True
    assert compose["networks"]["switchyard-egress"] == {
        "external": True,
        "name": "${SWITCHYARD_DOCKER_NETWORK:?set SWITCHYARD_DOCKER_NETWORK}",
    }
    assert "proxy-ca-public" in compose["volumes"]
    main_env = "\n".join(compose["services"]["main"]["environment"])
    assert "NODE_EXTRA_CA_CERTS=/etc/proxy-ca/ca-cert.pem" in main_env
    assert "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in main_env
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in main_env
    assert "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in main_env
    assert "GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt" in main_env
    proxy_env = "\n".join(compose["services"]["proxy"]["environment"])
    assert "OPENAI_BASE_URL=" in proxy_env
    assert "ANTHROPIC_BASE_URL=" in proxy_env
    assert "ALLOWED_HOSTS=" in proxy_env
    assert "VERIFIER_PROXY_TOKEN=${SWITCHYARD_VERIFIER_PROXY_TOKEN:-}" in proxy_env
    assert "SWITCHYARD_HOST_SOCKET" not in proxy_env
    proxy_assets = output / "compose-task" / "environment" / "proxy"
    assert (proxy_assets / "Dockerfile").is_file()
    assert (proxy_assets / "entrypoint.sh").is_file()
    rewriter = proxy_assets / "rewriter.py"
    assert rewriter.is_file()
    assert 'SESSION_ID_HEADER = "x-switchyard-session-id"' in rewriter.read_text()
    assert not (proxy_assets / "verifier_proxy.py").exists()
    healthcheck = "\n".join(compose["services"]["proxy"]["healthcheck"]["test"])
    assert "3128" in healthcheck
    assert "3129" in healthcheck
    assert "/etc/proxy-public/ca-cert.pem" in healthcheck


def test_generated_proxy_assets_normalize_windows_line_endings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_generator_module()
    proxy_source = tmp_path / "proxy-source"
    proxy_source.mkdir()
    for name in ("Dockerfile", "allowlist-base.txt", "entrypoint.sh", "rewriter.py"):
        (proxy_source / name).write_bytes(b"first\r\nsecond\r\n")
    monkeypatch.setattr(module, "PROXY_ASSET_DIR", proxy_source)

    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    module._merge_compose(task, ())

    for name in ("Dockerfile", "allowlist-base.txt", "entrypoint.sh", "rewriter.py"):
        assert b"\r" not in (task / "environment" / "proxy" / name).read_bytes()


def test_generated_dataset_manifest_records_pins_tasks_and_digests(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "task-a", "[environment]\n", "FROM ubuntu:22.04\n")
    _write_task(source, "task-b", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(tmp_path, source)
    manifest = json.loads((output / "switchyard_dataset_manifest.json").read_text())

    assert manifest["source_dataset"] == "openthoughts-tblite@2.0"
    assert manifest["task_count"] == 2
    assert manifest["agent_versions"] == {
        "CLAUDE_CODE_VERSION": "2.1.211",
        "CODEX_VERSION": "0.144.5",
        "HERMES_INSTALLER_SHA256": (
            "45f589461248c7a6ec3aecd7522a69dd49c5c8dbf4798ba1296af5c0c5e7ccd3"
        ),
        "HERMES_VERSION": "3c27eb6234bf91b8ceee9e9071591b31e9b148cb",
        "NODE_VERSION": "20.11.1",
        "OPENCODE_VERSION": "1.18.3",
        "UV_IMAGE": UV_IMAGE,
        "UV_VERSION": "0.12.9",
    }
    assert manifest["closed_book"]["proxy_asset_digest"].startswith("sha256:")
    assert manifest["closed_book"]["verifier_egress"] == "open-via-authenticated-proxy"
    assert {task["name"] for task in manifest["tasks"]} == {"task-a", "task-b"}
    assert all(task["dockerfile_digest"].startswith("sha256:") for task in manifest["tasks"])
    assert all(task["compose_digest"].startswith("sha256:") for task in manifest["tasks"])
    assert manifest["closed_book"]["hermes_installer"] == {
        "source_url": (
            "https://raw.githubusercontent.com/NousResearch/hermes-agent/"
            "3c27eb6234bf91b8ceee9e9071591b31e9b148cb/scripts/install.sh"
        ),
        "digest": "sha256:" + hashlib.sha256(HERMES_INSTALLER_FIXTURE).hexdigest(),
    }
    assert manifest["closed_book"]["uv"] == {
        "image": UV_IMAGE,
        "version": "0.12.9",
    }


def test_generated_compose_bakes_task_id_into_proxy_env(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "task-id-check", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(tmp_path, source)
    compose = yaml.safe_load(
        (output / "task-id-check" / "environment" / "docker-compose.yaml").read_text()
    )

    proxy_env = "\n".join(compose["services"]["proxy"]["environment"])
    assert "SWITCHYARD_TASK_ID=task-id-check" in proxy_env
    assert "SWITCHYARD_TRIAL_DIR=${HOST_AGENT_LOGS_PATH:-}" in proxy_env

def test_a_hermes_ref_that_is_not_a_commit_sha_is_rejected() -> None:
    """Only a full commit SHA can be recorded as a pin.

    The dataset manifest presents HERMES_VERSION as a reproducibility guarantee. Two
    builds recording the same string while installing different Hermes code is worse
    than recording nothing, so the build fails rather than asserting a pin it does not
    have. Tags are rejected alongside branches: a tag can be deleted or repointed, so
    it reads as immutable without being so.
    """
    base = {
        "CLAUDE_CODE_VERSION": "1",
        "CODEX_VERSION": "2",
        "OPENCODE_VERSION": "3",
        "NODE_VERSION": "4",
        "UV_VERSION": "0.12.9",
        "UV_IMAGE": UV_IMAGE,
    }
    rejected = (
        "main",
        "master",
        "HEAD",
        "v2026.8.3",
        "release/2026.8",
        "3c27eb6",
        "3C27EB6234BF91B8CEEE9E9071591B31E9B148CB",
        "3c27eb6234bf91b8ceee9e9071591b31e9b148cbb",
    )
    for ref in rejected:
        with pytest.raises(SystemExit, match="commit SHA"):
            _load_generator_module()._install_layer({**base, "HERMES_VERSION": ref})


def test_the_hermes_install_layer_uses_the_vendored_installer() -> None:
    sha = "3c27eb6234bf91b8ceee9e9071591b31e9b148cb"
    pins = {
        "CLAUDE_CODE_VERSION": "1",
        "CODEX_VERSION": "2",
        "OPENCODE_VERSION": "3",
        "NODE_VERSION": "4",
        "HERMES_VERSION": sha,
        "UV_VERSION": "0.12.9",
        "UV_IMAGE": UV_IMAGE,
    }

    layer = _load_generator_module()._install_layer(pins)

    assert "bash /tmp/switchyard-hermes-install.sh" in layer
    assert "COPY --from=switchyard_uv_build /uv /root/.hermes/bin/uv" in layer
    assert "raw.githubusercontent.com/NousResearch/hermes-agent" not in layer


def test_the_hermes_installer_is_fetched_at_the_pinned_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generator_module()
    sha = "3c27eb6234bf91b8ceee9e9071591b31e9b148cb"
    digest = hashlib.sha256(HERMES_INSTALLER_FIXTURE).hexdigest()
    requested_urls: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        requested_urls.append(request.full_url)
        assert timeout == 60
        return io.BytesIO(HERMES_INSTALLER_FIXTURE)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    content = module._fetch_hermes_installer(
        {"HERMES_VERSION": sha, "HERMES_INSTALLER_SHA256": digest}
    )

    assert content == HERMES_INSTALLER_FIXTURE
    assert requested_urls == [
        (
            "https://raw.githubusercontent.com/NousResearch/hermes-agent/"
            f"{sha}/scripts/install.sh"
        )
    ]


def test_the_hermes_installer_fetch_retries_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generator_module()
    sha = "3c27eb6234bf91b8ceee9e9071591b31e9b148cb"
    digest = hashlib.sha256(HERMES_INSTALLER_FIXTURE).hexdigest()
    attempts = 0

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise module.HTTPError(request.full_url, 429, "rate limited", {}, None)
        return io.BytesIO(HERMES_INSTALLER_FIXTURE)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)

    content = module._fetch_hermes_installer(
        {"HERMES_VERSION": sha, "HERMES_INSTALLER_SHA256": digest}
    )

    assert content == HERMES_INSTALLER_FIXTURE
    assert attempts == 2


def test_the_hermes_installer_digest_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generator_module()

    def fake_urlopen(_request: object, timeout: int) -> io.BytesIO:
        assert timeout == 60
        return io.BytesIO(HERMES_INSTALLER_FIXTURE)

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="digest mismatch"):
        module._fetch_hermes_installer(
            {
                "HERMES_VERSION": "3c27eb6234bf91b8ceee9e9071591b31e9b148cb",
                "HERMES_INSTALLER_SHA256": "0" * 64,
            }
        )


def test_a_non_numeric_uv_version_is_rejected() -> None:
    with pytest.raises(SystemExit, match="numeric semantic version"):
        _load_generator_module()._uv_version({"UV_VERSION": "latest"})


def test_an_unpinned_uv_image_is_rejected() -> None:
    with pytest.raises(SystemExit, match="pinned by a SHA-256 digest"):
        _load_generator_module()._uv_image({"UV_IMAGE": "ghcr.io/astral-sh/uv:latest"})


def test_the_hermes_pin_is_applied_by_commit_and_forced() -> None:
    """`--branch` reaches `git clone --branch`, which rejects a SHA outright.

    `--force-commit` is what makes the pin take effect. Without it the installer skips
    the checkout whenever the commit is an ancestor of the freshly cloned HEAD, warns,
    and leaves the image on the tip of main — the drift the pin exists to prevent,
    arriving as a warning rather than a build failure.
    """
    sha = "3c27eb6234bf91b8ceee9e9071591b31e9b148cb"
    pins = {
        "CLAUDE_CODE_VERSION": "1",
        "CODEX_VERSION": "2",
        "OPENCODE_VERSION": "3",
        "NODE_VERSION": "4",
        "HERMES_VERSION": sha,
        "UV_VERSION": "0.12.9",
        "UV_IMAGE": UV_IMAGE,
    }

    layer = _load_generator_module()._install_layer(pins)

    assert f"--commit {sha}" in layer
    assert "--force-commit" in layer
    assert "--branch" not in layer


def test_the_alpine_branch_installs_the_shell_the_installer_needs() -> None:
    """The installer is piped into bash, which Alpine does not ship by default."""
    pins = {
        "CLAUDE_CODE_VERSION": "1",
        "CODEX_VERSION": "2",
        "OPENCODE_VERSION": "3",
        "NODE_VERSION": "4",
        "HERMES_VERSION": "3c27eb6234bf91b8ceee9e9071591b31e9b148cb",
        "UV_VERSION": "0.12.9",
        "UV_IMAGE": UV_IMAGE,
    }

    layer = _load_generator_module()._install_layer(pins)

    assert "apk add --no-cache bash git ripgrep xz" in layer


def test_a_missing_hermes_pin_is_reported_with_the_other_pins(tmp_path: Path) -> None:
    """Absent, it must fail the shared pin check rather than crash reading the layer."""
    module = _load_generator_module()
    versions = tmp_path / "agent-versions.env"
    versions.write_text(
        "CLAUDE_CODE_VERSION=1\nCODEX_VERSION=2\nOPENCODE_VERSION=3\nNODE_VERSION=4\n"
    )
    module.AGENT_VERSIONS_FILE = versions
    source = tmp_path / "source"
    _write_task(source, "task-a", "[environment]\n", "FROM ubuntu:22.04\n")

    with pytest.raises(ValueError, match="missing pins.*HERMES_VERSION"):
        module.prepare_dataset(
            source_dataset="openthoughts-tblite@2.0",
            source_dir=source,
            output_dir=tmp_path / "prepared",
            harbor_command="harbor",
            overwrite=False,
        )


def test_a_missing_uv_pin_is_reported_with_the_other_pins(tmp_path: Path) -> None:
    module = _load_generator_module()
    versions = tmp_path / "agent-versions.env"
    versions.write_text(
        "CLAUDE_CODE_VERSION=1\n"
        "CODEX_VERSION=2\n"
        "OPENCODE_VERSION=3\n"
        "NODE_VERSION=4\n"
        "HERMES_VERSION=3c27eb6234bf91b8ceee9e9071591b31e9b148cb\n"
        "HERMES_INSTALLER_SHA256="
        "45f589461248c7a6ec3aecd7522a69dd49c5c8dbf4798ba1296af5c0c5e7ccd3\n"
    )
    module.AGENT_VERSIONS_FILE = versions
    source = tmp_path / "source"
    _write_task(source, "task-a", "[environment]\n", "FROM ubuntu:22.04\n")

    with pytest.raises(ValueError, match="missing pins.*UV_IMAGE.*UV_VERSION"):
        module.prepare_dataset(
            source_dataset="openthoughts-tblite@2.0",
            source_dir=source,
            output_dir=tmp_path / "prepared",
            harbor_command="harbor",
            overwrite=False,
        )


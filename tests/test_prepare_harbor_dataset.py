# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "benchmark" / "prepare_harbor_dataset.py"


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
) -> Path:
    module = _load_generator_module()
    output = tmp_path / "prepared"
    return module.prepare_dataset(
        source_dataset=source_dataset,
        source_dir=source,
        output_dir=output,
        harbor_command="harbor",
        overwrite=False,
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
    assert dockerfile.startswith("FROM python:3.12-slim\nUSER root\n")
    assert "@anthropic-ai/claude-code@2.1.211" in dockerfile
    assert "@openai/codex@0.144.5" in dockerfile
    assert "opencode-ai@1.18.3" in dockerfile


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

    assert dockerfile.startswith("FROM ubuntu:22.04\nRUN echo task\n")
    assert "SWITCHYARD_PREBAKED_AGENT_VERSIONS" in dockerfile
    assert "/usr/local/lib/node_modules/npm" in dockerfile
    assert "node-v20.11.1-linux-$node_arch.tar.gz" in dockerfile


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
    assert (proxy_assets / "rewriter.py").is_file()
    assert not (proxy_assets / "verifier_proxy.py").exists()
    healthcheck = "\n".join(compose["services"]["proxy"]["healthcheck"]["test"])
    assert "3128" in healthcheck
    assert "3129" in healthcheck


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
        "NODE_VERSION": "20.11.1",
        "OPENCODE_VERSION": "1.18.3",
    }
    assert manifest["closed_book"]["proxy_asset_digest"].startswith("sha256:")
    assert manifest["closed_book"]["verifier_egress"] == "open-via-authenticated-proxy"
    assert {task["name"] for task in manifest["tasks"]} == {"task-a", "task-b"}
    assert all(task["dockerfile_digest"].startswith("sha256:") for task in manifest["tasks"])
    assert all(task["compose_digest"].startswith("sha256:") for task in manifest["tasks"])


def test_generated_compose_bakes_task_id_into_proxy_env(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_task(source, "task-id-check", "[environment]\n", "FROM ubuntu:22.04\n")

    output = _prepare(tmp_path, source)
    compose = yaml.safe_load(
        (output / "task-id-check" / "environment" / "docker-compose.yaml").read_text()
    )

    proxy_env = "\n".join(compose["services"]["proxy"]["environment"])
    assert "SWITCHYARD_TASK_ID=task-id-check" in proxy_env

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a local closed-book Harbor dataset with prebaked coding agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in a broken environment
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DATASET = "openthoughts-tblite@2.0"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "datasets" / "openthoughts-tblite-closed-book"
AGENT_VERSIONS_FILE = SCRIPT_DIR / "agent-versions.env"
PROXY_ASSET_DIR = SCRIPT_DIR / "closed_book_proxy" / "proxy"
AGENT_ENTRYPOINT = "switchyard-agent-entrypoint.sh"
TERMINAL_BENCH_2_SOURCE_DATASET = "terminal-bench/terminal-bench-2"
TERMINAL_BENCH_2_1_SOURCE_DATASET = "terminal-bench/terminal-bench-2-1"
# Shared across the TB2 family (2.0 + the 2.1 verified iteration): 2.1 tweaks
# timeouts/resources on existing tasks, so its Oracle solutions reach the same hosts.
TERMINAL_BENCH_2_PROXY_ALLOWLIST_HOSTS = (
    "api.github.com",
    "api.launchpad.net",
    "archive.ubuntu.com",
    "cloud.r-project.org",
    "coq.inria.fr",
    "codeload.github.com",
    "data.rcsb.org",
    "deb.debian.org",
    "download.pytorch.org",
    "download-r2.pytorch.org",
    "download.qemu.org",
    "files.pythonhosted.org",
    "github.com",
    "githubusercontent.com",
    "gitlab.inria.fr",
    "hf.co",
    "hf.space",
    "huggingface.co",
    "keyserver.ubuntu.com",
    "objects.githubusercontent.com",
    "ocaml.org",
    "opam.ocaml.org",
    "ports.ubuntu.com",
    "ppa.launchpadcontent.net",
    "pubchem.ncbi.nlm.nih.gov",
    "pypi.org",
    "pypi.python.org",
    "release-assets.githubusercontent.com",
    "security.ubuntu.com",
    "storage.googleapis.com",
    "transfer.xethub.hf.co",
    "www.cs.toronto.edu",
    "www.fpbase.org",
    "www.povray.org",
    "www.rcsb.org",
)


def _iso_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _path_digest(path: Path) -> str:
    try:
        resolved = path.resolve()
        hasher = hashlib.sha256()
        if resolved.is_file():
            hasher.update(resolved.name.encode())
            with resolved.open("rb") as fh:
                hasher.update(hashlib.file_digest(fh, "sha256").digest())
            return f"sha256:{hasher.hexdigest()}"
        if resolved.is_dir():
            for item in sorted(p for p in resolved.rglob("*") if p.is_file()):
                rel = item.relative_to(resolved).as_posix()
                with item.open("rb") as fh:
                    file_hash = hashlib.file_digest(fh, "sha256").hexdigest()
                hasher.update(f"{rel}\n{file_hash}\n".encode())
            return f"sha256:{hasher.hexdigest()}"
    except OSError:
        return "sha256:unknown"
    return "sha256:missing"


def _task_dirs(dataset_root: Path) -> list[Path]:
    if (dataset_root / "task.toml").is_file():
        return [dataset_root]
    return sorted(path.parent for path in dataset_root.rglob("task.toml"))


def _exported_dataset_candidates(download_root: Path, source_dataset: str) -> list[Path]:
    source_name = source_dataset.split("@", 1)[0]
    names = [
        source_name.split("/")[-1],
        source_name.replace("/", "__"),
        source_name,
    ]
    candidates: list[Path] = []
    for name in names:
        candidate = download_root / name
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _proxy_allowlist_hosts_for_dataset(source_dataset: str) -> tuple[str, ...]:
    source_name = source_dataset.split("@", 1)[0]
    if source_name in (TERMINAL_BENCH_2_SOURCE_DATASET, TERMINAL_BENCH_2_1_SOURCE_DATASET):
        return TERMINAL_BENCH_2_PROXY_ALLOWLIST_HOSTS
    return ()


def _find_exported_dataset_root(download_root: Path, source_dataset: str) -> Path:
    for candidate in _exported_dataset_candidates(download_root, source_dataset):
        if candidate.is_dir() and _task_dirs(candidate):
            return candidate

    dirs_with_tasks = sorted(
        {task.parent.parent for task in download_root.rglob("task.toml") if task.parent != download_root}
    )
    if len(dirs_with_tasks) == 1:
        return dirs_with_tasks[0]
    if _task_dirs(download_root):
        return download_root
    raise FileNotFoundError(f"could not find exported tasks under {download_root}")


def _run_download(source_dataset: str, download_root: Path, harbor_command: str, overwrite: bool) -> Path:
    download_root.mkdir(parents=True, exist_ok=True)
    command = [
        *shlex.split(harbor_command),
        "download",
        source_dataset,
        "--output-dir",
        str(download_root),
        "--export",
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    return _find_exported_dataset_root(download_root, source_dataset)


def _remove_toml_key_from_table(text: str, table: str, key: str) -> str:
    out: list[str] = []
    in_table = False
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    table_name = f"[{table}]"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_table = stripped == table_name
        if in_table and key_re.match(line):
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _install_layer(pins: dict[str, str]) -> str:
    node_version = pins["NODE_VERSION"]
    claude_version = pins["CLAUDE_CODE_VERSION"]
    codex_version = pins["CODEX_VERSION"]
    opencode_version = pins["OPENCODE_VERSION"]
    return f"""

# Switchyard benchmark prebaked coding agents.
ENV SWITCHYARD_PREBAKED_AGENT_VERSIONS="claude-code={claude_version},codex={codex_version},opencode={opencode_version},node={node_version}"
RUN set -eux; \\
    if command -v apt-get >/dev/null 2>&1; then \\
        apt-get update; \\
        apt-get install -y --no-install-recommends ca-certificates curl gzip tar xz-utils; \\
        rm -rf /var/lib/apt/lists/*; \\
    elif command -v apk >/dev/null 2>&1; then \\
        apk add --no-cache ca-certificates curl gzip tar xz; \\
    elif command -v yum >/dev/null 2>&1; then \\
        yum install -y ca-certificates curl gzip tar xz; \\
        yum clean all; \\
    else \\
        echo "No supported package manager found; expecting curl, tar, and gzip to exist."; \\
    fi; \\
    arch="$(uname -m)"; \\
    case "$arch" in \\
        x86_64|amd64) node_arch="x64" ;; \\
        aarch64|arm64) node_arch="arm64" ;; \\
        *) echo "Unsupported Node.js architecture: $arch" >&2; exit 1 ;; \\
    esac; \\
    rm -rf \\
        /usr/local/bin/node \\
        /usr/local/bin/npm \\
        /usr/local/bin/npx \\
        /usr/local/bin/corepack \\
        /usr/local/include/node \\
        /usr/local/lib/node_modules/npm \\
        /usr/local/lib/node_modules/corepack; \\
    curl -fsSL "https://nodejs.org/dist/v{node_version}/node-v{node_version}-linux-$node_arch.tar.gz" -o /tmp/node.tgz; \\
    tar -xzf /tmp/node.tgz -C /usr/local --strip-components=1; \\
    rm -f /tmp/node.tgz; \\
    npm config set fetch-retries 5; \\
    npm config set fetch-retry-mintimeout 20000; \\
    npm config set fetch-retry-maxtimeout 120000; \\
    npm install -g \\
        "@anthropic-ai/claude-code@{claude_version}" \\
        "@openai/codex@{codex_version}" \\
        "opencode-ai@{opencode_version}"; \\
    claude --version; \\
    codex --version; \\
    opencode --version
"""


def _agent_entrypoint_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

PROXY_CA="/etc/proxy-ca/ca-cert.pem"
test -f "${PROXY_CA}"
mkdir -p /usr/local/share/ca-certificates
cp "${PROXY_CA}" /usr/local/share/ca-certificates/switchyard-closed-book-proxy.crt
update-ca-certificates >/dev/null
echo "switchyard-agent-entrypoint: installed proxy CA into system trust store"

exec "$@"
"""


def _entrypoint_layer() -> str:
    return f"""

# Switchyard benchmark proxy CA trust bootstrap.
COPY {AGENT_ENTRYPOINT} /usr/local/bin/{AGENT_ENTRYPOINT}
RUN chmod +x /usr/local/bin/{AGENT_ENTRYPOINT}
ENTRYPOINT ["/usr/local/bin/{AGENT_ENTRYPOINT}"]
CMD ["bash", "-lc", "sleep infinity"]
"""


def _load_task_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def _rewrite_task_image(task_dir: Path, pins: dict[str, str]) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    data = _load_task_toml(task_toml)
    environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
    docker_image = environment.get("docker_image") if isinstance(environment, dict) else None
    dockerfile = task_dir / "environment" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    entrypoint = dockerfile.parent / AGENT_ENTRYPOINT
    entrypoint.write_text(_agent_entrypoint_script())
    entrypoint.chmod(0o755)
    layer = _install_layer(pins)
    entrypoint_layer = _entrypoint_layer()

    if docker_image:
        dockerfile.write_text(
            f"FROM {docker_image}\nUSER root\n{layer.lstrip()}{entrypoint_layer}"
        )
        task_toml.write_text(
            _remove_toml_key_from_table(task_toml.read_text(), "environment", "docker_image")
        )
        image_source = docker_image
        removed = True
    else:
        if not dockerfile.is_file():
            dockerfile.write_text("FROM ubuntu:22.04\n")
        dockerfile.write_text(dockerfile.read_text().rstrip() + "\n" + layer + entrypoint_layer)
        image_source = None
        removed = False

    return {
        "docker_image_source": image_source,
        "docker_image_removed": removed,
        "dockerfile_digest": _path_digest(dockerfile),
    }


def _as_env_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [f"{key}={val}" for key, val in value.items()]
    return []


def _merge_env(existing: Any, updates: dict[str, str]) -> list[str]:
    env: dict[str, str] = {}
    for item in _as_env_list(existing):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    env.update(updates)
    return [f"{key}={value}" for key, value in sorted(env.items())]


def _merge_networks(existing: Any, network: str) -> list[str]:
    if isinstance(existing, list):
        networks = [str(item) for item in existing]
    elif isinstance(existing, dict):
        networks = [str(item) for item in existing]
    elif isinstance(existing, str):
        networks = [existing]
    else:
        networks = []
    if network not in networks:
        networks.append(network)
    return networks


def _append_proxy_allowlist(proxy_assets: Path, hosts: tuple[str, ...]) -> None:
    if not hosts:
        return
    allowlist_path = proxy_assets / "allowlist-base.txt"
    base = allowlist_path.read_text().rstrip()
    additions = "\n".join(dict.fromkeys(hosts))
    allowlist_path.write_text(f"{base}\n\n# Dataset-required package and data sources.\n{additions}\n")


def _merge_compose(task_dir: Path, proxy_allowlist_hosts: tuple[str, ...]) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to generate closed-book docker-compose overrides")

    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    compose_path = env_dir / "docker-compose.yaml"
    if compose_path.is_file():
        compose = yaml.safe_load(compose_path.read_text()) or {}
    else:
        compose = {}
    if not isinstance(compose, dict):
        raise ValueError(f"{compose_path} must contain a YAML object")

    services = compose.setdefault("services", {})
    if not isinstance(services, dict):
        raise ValueError(f"{compose_path}: services must be a YAML object")

    proxy_assets = env_dir / "proxy"
    if proxy_assets.exists():
        shutil.rmtree(proxy_assets)
    shutil.copytree(PROXY_ASSET_DIR, proxy_assets)
    _append_proxy_allowlist(proxy_assets, proxy_allowlist_hosts)

    main = services.setdefault("main", {})
    if not isinstance(main, dict):
        raise ValueError(f"{compose_path}: services.main must be a YAML object")

    for name, service in list(services.items()):
        if name == "proxy":
            continue
        if not isinstance(service, dict):
            continue
        service.pop("network_mode", None)
        service["networks"] = _merge_networks(service.get("networks"), "agent-internal")

    main["environment"] = _merge_env(
        main.get("environment"),
        {
            "HTTP_PROXY": "http://proxy:3128",
            "HTTPS_PROXY": "http://proxy:3128",
            "NO_PROXY": "localhost,127.0.0.1,proxy",
            "NODE_EXTRA_CA_CERTS": "/etc/proxy-ca/ca-cert.pem",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
            "GIT_SSL_CAINFO": "/etc/ssl/certs/ca-certificates.crt",
            "http_proxy": "http://proxy:3128",
            "https_proxy": "http://proxy:3128",
            "no_proxy": "localhost,127.0.0.1,proxy",
        },
    )
    main["depends_on"] = {
        **(main.get("depends_on") if isinstance(main.get("depends_on"), dict) else {}),
        "proxy": {"condition": "service_healthy"},
    }
    main["volumes"] = [
        *([str(item) for item in main.get("volumes", [])] if isinstance(main.get("volumes"), list) else []),
        "proxy-ca-public:/etc/proxy-ca:ro",
    ]

    services["proxy"] = {
        "build": {"context": "./proxy"},
        "environment": [
            "ALLOWED_HOSTS=${ALLOWED_HOSTS:-}",
            "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-}",
            "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-}",
            "CLOSED_BOOK_MODE=${CLOSED_BOOK_MODE:-1}",
            "OPENAI_BASE_URL=${OPENAI_BASE_URL:-}",
            "SWITCHYARD_BASE_URL=${SWITCHYARD_BASE_URL:-}",
            "VERIFIER_PROXY_TOKEN=${SWITCHYARD_VERIFIER_PROXY_TOKEN:-}",
        ],
        "healthcheck": {
            "test": [
                "CMD",
                "python",
                "-c",
                (
                    "import socket\n"
                    "for port in (3128, 3129):\n"
                    "    s=socket.create_connection(('127.0.0.1', port), 2)\n"
                    "    s.close()\n"
                ),
            ],
            "interval": "2s",
            "timeout": "2s",
            "retries": 30,
        },
        "networks": ["agent-internal", "switchyard-egress"],
        "volumes": [
            "proxy-ca-public:/etc/proxy-public",
        ],
    }

    compose.setdefault("networks", {})
    compose["networks"]["agent-internal"] = {"driver": "bridge", "internal": True}
    compose["networks"]["switchyard-egress"] = {
        "external": True,
        "name": "${SWITCHYARD_DOCKER_NETWORK:?set SWITCHYARD_DOCKER_NETWORK}",
    }
    compose.setdefault("volumes", {})
    compose["volumes"]["proxy-ca-public"] = {}

    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    return {
        "compose_path": compose_path,
        "compose_digest": _path_digest(compose_path),
        "proxy_asset_digest": _path_digest(proxy_assets),
    }


def prepare_dataset(
    *,
    source_dataset: str,
    source_dir: Path | None,
    output_dir: Path,
    harbor_command: str,
    overwrite: bool,
) -> Path:
    pins = _read_env_file(AGENT_VERSIONS_FILE)
    required = {"CLAUDE_CODE_VERSION", "CODEX_VERSION", "OPENCODE_VERSION", "NODE_VERSION"}
    missing = sorted(required - pins.keys())
    if missing:
        raise ValueError(f"missing pins in {AGENT_VERSIONS_FILE}: {', '.join(missing)}")

    if source_dir is None:
        download_root = output_dir.parent / "_downloads"
        source_dir = _run_download(source_dataset, download_root, harbor_command, overwrite)

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)

    tasks = []
    proxy_digest = _path_digest(PROXY_ASSET_DIR)
    proxy_allowlist_hosts = _proxy_allowlist_hosts_for_dataset(source_dataset)
    for task_dir in _task_dirs(output_dir):
        image = _rewrite_task_image(task_dir, pins)
        compose = _merge_compose(task_dir, proxy_allowlist_hosts)
        tasks.append(
            {
                "name": task_dir.name,
                "path": str(task_dir.relative_to(output_dir)),
                **image,
                "compose_digest": compose["compose_digest"],
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at": _iso_timestamp(),
        "source_dataset": source_dataset,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "task_count": len(tasks),
        "agent_versions": pins,
        "closed_book": {
            "gateway_enforcement": "docker-compose sidecar proxy",
            "verifier_egress": "open-via-authenticated-proxy",
            "proxy_asset_digest": proxy_digest,
            "proxy_allowlist_hosts": list(proxy_allowlist_hosts),
            "proxy_strip_log_path": "/etc/proxy-public/strip.jsonl",
            "agent_internal_network": "agent-internal",
            "proxy_egress_network": "proxy-egress",
        },
        "tasks": tasks,
    }
    (output_dir / "switchyard_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )
    return output_dir


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_harbor_dataset")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--harbor-command", default=os.environ.get("HARBOR_COMMAND", "uv run --no-sync harbor"))
    parser.add_argument("--overwrite", action="store_true")
    ns = parser.parse_args(argv)

    prepared = prepare_dataset(
        source_dataset=ns.source_dataset,
        source_dir=ns.source_dir,
        output_dir=ns.output_dir,
        harbor_command=ns.harbor_command,
        overwrite=ns.overwrite,
    )
    print(f"Prepared closed-book Harbor dataset: {prepared}")
    print(f"Manifest: {prepared / 'switchyard_dataset_manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())

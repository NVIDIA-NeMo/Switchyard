# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dockerfile construction for all track adapters.

Produces a self-contained Dockerfile for a Harbor task with a digest-pinned
base image, pinned uv installer, and mandatory ``pinned_requirements`` on
the candidate. All three adapters (planning, search-native, tools) should
route through ``build_dockerfile()`` so reproducibility is a
single-point-of-change.

See ``docs/reference/adapter-reproducibility.md`` for the migration guide that other
track adapter owners should follow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1"


_REFERENCES_DIR = Path(__file__).resolve().parents[2].parent / "references"
_DOCKERFILE_DIR = _REFERENCES_DIR / "dockerfiles"

PINNED_REQUIREMENTS_FILENAME = "requirements.lock"

# Agent runtime versions, pinned for reproducibility. Bump together with a
# matching update to craft_taskgen/config.py::CC_VERSION. When bake_agents
# is True on a DockerBuildSpec, the Dockerfile installs these exact versions
# via npm inside the task image so agent behavior is reproducible too.
CLAUDE_CODE_VERSION = "2.1.118"
CODEX_VERSION = "0.121.0"
OPENCODE_VERSION = "1.4.9"
# openhands-sdk is installed by harbor's openhands-sdk install-template, not
# baked into our adapter Dockerfiles. Pinned here so run-baselines.sh can
# pass version=<this> via --agent-kwarg, matching the Ultra evaluator-sidecar
# config from lvega's handoff.
OPENHANDS_SDK_VERSION = "1.17.0"
# qwen-coder is likewise installed at run-time by harbor's install-qwen-code.sh.j2
# (npm install -g @qwen-code/qwen-code@<version>), not baked. Pinned here so
# run-baselines.sh can forward version=<this> via --agent-kwarg.
QWEN_CODE_VERSION = "0.16.0"
# pi (mariozechner/earendil-works pi-coding-agent) — installed at run-time by
# harbor's pi.py::install() (npm install -g @earendil-works/pi-coding-agent@<v>).
# Patched in patches/harbor-agent-patches.diff to add an `nvidia` provider that
# writes a per-model models.json planting OPENAI_BASE_URL as baseUrl.
PI_VERSION = "0.75.5"
OPENHANDS_SDK_MAX_ITERATIONS = 200
NODE_SETUP_URL = "https://deb.nodesource.com/setup_22.x"

# Shared Ubuntu base digest for adapters that need an OS-level image
# (search-native, future agent-only tracks). Planning uses python:{ver}
# bases per the Dockerfile.pyXXX fragments.
UBUNTU_22_04_DIGEST = "sha256:14be402d3f1eeeb5e7da73d3260322c68e7b51c88388f53e88eb21d6450bd520"

# Harbor framework version pinned via pyproject (`harbor @ git+...@<sha>`).
# Re-exported here as a constant so the per-task manifest can record it.
HARBOR_COMMIT = "46bb68c"


@dataclass
class DockerBuildSpec:
    """Everything the Dockerfile builder needs for one task.

    ``python`` and ``install`` are required. ``pre_install`` and
    ``test_deps`` are optional hooks the candidate's docker block can
    declare. ``pinned_requirements`` is the raw requirements.lock content
    (e.g. output of ``uv pip freeze``). When set, the Dockerfile installs
    the frozen set with ``--no-deps`` so the resolver runs once at lock
    time, not at build time.
    """

    repo: str
    parent_sha: str
    python: str
    install: str
    main_package: str = ""
    pre_install: list[str] = field(default_factory=list)
    test_deps: str = ""
    pinned_requirements: str = ""
    bake_agents: bool = False
    """When True, bakes Claude Code + codex (pinned versions) into the
    image. Use for tracks where harbor runs the agent inside the task
    container. Leave False for tracks that run the agent on the host.
    """


def _read_base(python_version: str) -> str:
    """Return the base Dockerfile fragment for a python version."""
    specific = _DOCKERFILE_DIR / f"Dockerfile.py{python_version.replace('.', '')}"
    if specific.is_file():
        return specific.read_text()
    template = _DOCKERFILE_DIR / "Dockerfile.template"
    if not template.is_file():
        raise FileNotFoundError(
            f"no base Dockerfile for python {python_version} and no template at {template}"
        )
    return template.read_text().replace("PYVERSION", python_version)


def main_package_name(repo: str) -> str:
    """Best-effort Python package name from a repo slug."""
    return repo.split("/")[1].replace("-", "_").split(".")[0].lower()


def _agent_bake_layer() -> list[str]:
    """Dockerfile lines that install pinned agent CLIs via npm.

    Bakes node 22 (via NodeSource), then pins Claude Code, codex, and
    opencode to exact npm versions. Track adopters who need in-container
    agent execution (search-native, etc.) flip ``bake_agents=True``.
    """
    return [
        f"RUN curl -fsSL {NODE_SETUP_URL} | bash - && "
        "apt-get install -y --no-install-recommends nodejs && "
        "rm -rf /var/lib/apt/lists/*",
        f"RUN npm install -g @anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}",
        f"RUN npm install -g @openai/codex@{CODEX_VERSION}",
        f"RUN npm install -g opencode-ai@{OPENCODE_VERSION}",
    ]


_REPO_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


def _require_single_line(value: str, field: str) -> str:
    """Reject newlines in a candidate-derived value before it is interpolated
    into a Dockerfile ``RUN`` directive. A value containing a newline could
    otherwise inject additional Dockerfile instructions (FROM/COPY/USER/...),
    i.e. Dockerfile directive injection (CWE-94)."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"Refusing to build Dockerfile: {field} contains a newline: {value!r}")
    return value


def build_dockerfile(spec: DockerBuildSpec) -> str:
    """Return the full Dockerfile content for one task.

    Structure (top to bottom):
      1. Base layer: python{ver}-slim + pinned uv.
      2. Clone the repo at parent_sha.
      3. Optional pre_install patches.
      4. Install the repo (spec.install, e.g. ``uv pip install --system -e .``).
      5. Dependency layer:
         - If pinned_requirements is set: COPY the lock file + install with --no-deps.
         - Else if test_deps is set: install them unpinned.
         - Else: no extra deps.
      6. Sanity import check.
      7. Re-init git (agent diffs against a clean tree).
      8. Ensure /repo/output exists.
    """
    base = _read_base(spec.python).rstrip()
    lines = [base]

    # Candidate-derived values flow into shell RUN directives; validate repo/ref
    # against strict character sets and reject newlines in free-form command
    # fields so neither can inject shell commands or extra Dockerfile directives.
    if not _REPO_RE.match(spec.repo):
        raise ValueError(f"Refusing to build Dockerfile: invalid repo {spec.repo!r}")
    if not _GIT_REF_RE.match(spec.parent_sha):
        raise ValueError(f"Refusing to build Dockerfile: invalid parent_sha {spec.parent_sha!r}")
    lines.append(
        f"RUN git clone https://github.com/{spec.repo}.git /repo "
        f"&& cd /repo && git checkout {spec.parent_sha}"
    )

    for patch in spec.pre_install:
        lines.append(f"RUN {_require_single_line(patch, 'pre_install')}")

    lines.append(f"RUN {_require_single_line(spec.install, 'install')}")

    if spec.pinned_requirements.strip():
        lines.append(f"COPY {PINNED_REQUIREMENTS_FILENAME} /tmp/{PINNED_REQUIREMENTS_FILENAME}")
        lines.append(f"RUN uv pip install --system --no-deps -r /tmp/{PINNED_REQUIREMENTS_FILENAME}")
    elif spec.test_deps.strip():
        lines.append(f"RUN uv pip install --system {_require_single_line(spec.test_deps, 'test_deps')}")

    if spec.bake_agents:
        lines.extend(_agent_bake_layer())

    main_pkg = spec.main_package or main_package_name(spec.repo)
    if main_pkg:
        lines.append(f"RUN python -c \"import {main_pkg}; print('{main_pkg} OK')\"")

    lines.append(
        'RUN rm -rf .git && git init && git config user.email "agent@test" '
        "&& git config user.name \"Agent\" && git add -A && git commit -m 'initial commit'"
    )
    lines.append("RUN mkdir -p /repo/output")
    return "\n".join(lines) + "\n"


def write_environment(
    env_dir: Path,
    dockerfile: str,
    pinned_requirements: str = "",
    manifest: dict | None = None,
) -> None:
    """Write the Dockerfile, the requirements lock file, and (if given) the
    per-task reproducibility manifest.

    ``env_dir`` is the ``environment/`` subdirectory of a Harbor task.
    """
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile)
    if pinned_requirements.strip():
        text = pinned_requirements if pinned_requirements.endswith("\n") else pinned_requirements + "\n"
        (env_dir / PINNED_REQUIREMENTS_FILENAME).write_text(text)
    if manifest is not None:
        (env_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _extract_base_image_digest(dockerfile: str) -> dict[str, str]:
    """Parse the FROM line and split reference from digest."""
    match = re.search(r"^FROM\s+(\S+)", dockerfile, re.MULTILINE)
    if not match:
        return {"reference": "", "digest": ""}
    ref = match.group(1)
    if "@" in ref:
        name, digest = ref.split("@", 1)
        return {"reference": name, "digest": digest}
    return {"reference": ref, "digest": ""}


def produce_manifest(
    spec: DockerBuildSpec,
    adapter: str,
    dockerfile: str,
    extra: dict | None = None,
) -> dict:
    """Build the machine-readable reproducibility manifest for one task.

    Captures every pinned version the adapter controlled at build time.
    Harbor / eval-time tooling can merge this with its own per-run manifest
    (agent config flags, sampling params, serving stack) to satisfy the
    NeurIPS Q4/Q5/Q6/Q8 reproducibility questions.
    """
    base = _extract_base_image_digest(dockerfile)
    lock_sha = ""
    if spec.pinned_requirements.strip():
        lock_sha = hashlib.sha256(spec.pinned_requirements.encode()).hexdigest()

    agents: dict[str, str] = {}
    if spec.bake_agents:
        agents = {
            "claude_code": CLAUDE_CODE_VERSION,
            "codex": CODEX_VERSION,
            "opencode": OPENCODE_VERSION,
        }

    manifest: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "adapter": adapter,
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_image": base,
        "installer": {"name": "uv", "version": "0.7.12"},
        "python": spec.python,
        "agents": agents,
        "harbor_commit": HARBOR_COMMIT,
        "repo": spec.repo,
        "parent_sha": spec.parent_sha,
        "pinned_requirements_sha256": lock_sha,
    }
    if extra:
        manifest.update(extra)
    return manifest


def spec_from_candidate(
    candidate: dict,
    pinned_requirements: str = "",
) -> DockerBuildSpec:
    """Convenience constructor for the shared candidate JSON schema.

    Reads the candidate's ``docker`` block: ``python`` (default 3.11),
    ``install`` (required), optional ``pre_install`` and ``test_deps``.

    Reproducibility is required: raises if the candidate has no
    ``pinned_requirements`` field (and none passed in explicitly). Every
    task must ship with a full dep lock; no opt-out.
    """
    docker = candidate.get("docker") or {}
    install = docker.get("install")
    if not install:
        raise ValueError(
            f"candidate missing docker.install (task_name={candidate.get('task_name', '<unknown>')})"
        )
    resolved_pins = pinned_requirements or candidate.get("pinned_requirements", "")
    if not resolved_pins.strip():
        raise ValueError(
            f"candidate missing pinned_requirements "
            f"(task_name={candidate.get('task_name', '<unknown>')}). "
            f"Every task must ship with a dep lock. Run craft-taskgen-lock-deps "
            f"(or equivalent) to populate the field."
        )
    return DockerBuildSpec(
        repo=candidate["repo"],
        parent_sha=candidate["parent_sha"],
        python=str(docker.get("python", "3.11")),
        install=install,
        main_package=main_package_name(candidate["repo"]),
        pre_install=list(docker.get("pre_install", []) or []),
        test_deps=docker.get("test_deps", "") or "",
        pinned_requirements=resolved_pins,
    )

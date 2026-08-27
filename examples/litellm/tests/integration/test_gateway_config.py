# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
import tomllib
from pathlib import Path

import yaml
from switchyard_litellm import StageRoutingPlugin

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
DEPLOYMENT_ROOT = PACKAGE_ROOT / "deployment"


def test_example_targets_python_312() -> None:
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["project"]["dependencies"] == [
        "nemo-switchyard>=0.1.0",
        "litellm==1.97.0",
    ]


def test_lockfile_matches_root_package_version() -> None:
    root_project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((PACKAGE_ROOT / "uv.lock").read_text())
    locked_root = next(
        package for package in lock["package"] if package["name"] == "nemo-switchyard"
    )

    assert locked_root["version"] == root_project["project"]["version"]


def test_compose_builds_the_pinned_plugin_image_and_selects_a_profile() -> None:
    compose = yaml.safe_load((DEPLOYMENT_ROOT / "compose.yaml").read_text())
    service = compose["services"]["litellm"]

    assert service["build"] == {
        "context": "../../..",
        "dockerfile": "examples/litellm/deployment/Dockerfile",
    }
    assert service["image"] == "switchyard-litellm:1.97.0"
    assert service["command"] == ["--config", "/app/deployment/litellm.yaml"]
    assert service["environment"] == {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}",
        "SWITCHYARD_LITELLM_CONFIG": "/app/deployment/switchyard.toml",
    }
    assert service["volumes"] == [
        "./profiles/${SWITCHYARD_LITELLM_PROFILE:-stage}:/app/deployment:ro"
    ]
    assert (DEPLOYMENT_ROOT / ".env.example").read_text() == (
        "OPENROUTER_API_KEY=\nSWITCHYARD_LITELLM_PROFILE=stage\n"
    )


def test_dockerfile_runs_the_final_image_as_litellm_user() -> None:
    """Require the final runtime stage to select the dedicated non-root account."""
    instructions = [
        line.strip()
        for line in (DEPLOYMENT_ROOT / "Dockerfile").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    final_stage = max(index for index, line in enumerate(instructions) if line.startswith("FROM "))
    runtime_users = [
        line.split(maxsplit=1)[1]
        for line in instructions[final_stage + 1 :]
        if line.startswith("USER ")
    ]

    assert runtime_users == ["litellm"]


def test_deployment_profiles_contain_only_model_and_policy_configuration() -> None:
    for profile in ("stage", "random"):
        assert {path.name for path in (DEPLOYMENT_ROOT / "profiles" / profile).iterdir()} == {
            "litellm.yaml",
            "switchyard.toml",
        }


def test_python_router_example_owns_its_programmatic_stage_plugin() -> None:
    namespace = runpy.run_path(str(PACKAGE_ROOT / "examples" / "python_router.py"))

    assert isinstance(namespace["STAGE_ROUTING_PLUGIN"], StageRoutingPlugin)
    assert [entry["model_name"] for entry in namespace["MODEL_LIST"]] == [
        "switchyard",
        "switchyard",
    ]


def test_docker_context_excludes_local_environment_files() -> None:
    patterns = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".dockerignore").read_text().splitlines()
        if line.strip()
    }

    assert {".env", "**/.env", ".env.*", "**/.env.*"} <= patterns
    assert {"!.env.example", "!**/.env.example"} <= patterns

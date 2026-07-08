# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "trialqa_local_runner.py"
DESCRIPTION = "Use for answering TrialQA questions with targeted ToolUniverse lookups."


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("switchyard_trialqa_local_runner", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidate(
    root: Path,
    *,
    name: str = "tooluniverse-trialqa",
    description: str = DESCRIPTION,
    version_status: str = "passed",
) -> Path:
    index = b"# TrialQA candidate index\n"
    skill = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        "# TrialQA\n\nUse targeted trial-registry searches.\n"
    ).encode()
    (root / name).mkdir(parents=True)
    (root / "SKILL.md").write_bytes(index)
    (root / name / "SKILL.md").write_bytes(skill)
    manifest = {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "validation": {"status": version_status},
        "skills": [
            {"path": "SKILL.md", "sha256": hashlib.sha256(index).hexdigest()},
            {
                "path": f"{name}/SKILL.md",
                "sha256": hashlib.sha256(skill).hexdigest(),
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _write_profile(path: Path, *, first: str = "sd-executor", model: str | None = None) -> Path:
    model = model or "nvidia/nvidia/nemotron-3-ultra"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "defaults:\n"
        "  api_key: ${NVIDIA_API_KEY}\n"
        "  base_url: https://inference-api.nvidia.com/v1\n"
        "routes:\n"
        f"  {first}:\n"
        "    type: model\n"
        "    target:\n"
        f"      model: {model}\n"
        "      format: openai\n"
        "  sd-judge:\n"
        "    type: model\n"
        "    target:\n"
        "      model: aws/anthropic/bedrock-claude-opus-4-8\n",
        encoding="utf-8",
    )
    return path


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_tooluniverse(venv: Path, *, version: str = "1.1.11") -> Path:
    binary = _write_executable(venv / "bin" / "tooluniverse-smcp-stdio")
    _write_executable(venv / "bin" / "python")
    metadata = (
        venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / f"tooluniverse-{version}.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Name: tooluniverse\nVersion: {version}\n", encoding="utf-8")
    return binary


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "candidate": _write_candidate(tmp_path / "candidate"),
        "profile": _write_profile(tmp_path / "routes.yaml"),
        "switchyard": _write_executable(tmp_path / "bin" / "switchyard"),
        "tooluniverse": _write_tooluniverse(tmp_path / "tooluniverse-venv"),
        "codex": _write_executable(tmp_path / "bin" / "codex"),
    }


def _build_pair(module: ModuleType, tmp_path: Path, candidate: Path) -> object:
    return module.build_trial_workspace_pair(
        capture_cwd=tmp_path / "capture",
        task_id="trialqa-001-r000",
        prompt=(
            "# TrialQA\n\nAnswer the clinical-trial question. Write only the answer to "
            "`answer.txt`.\n"
        ),
        candidate_root=candidate,
    )


def _payload_files(root: Path, managed: Path | None = None) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if managed is not None and path == managed:
            continue
        if path.is_file() and not path.is_symlink():
            result[relative.as_posix()] = path.read_bytes()
    return result


def test_workspace_pair_isolated_git_and_only_treatment_has_managed_skill(
    tmp_path: Path,
) -> None:
    module = _load()
    paths = _fixture_paths(tmp_path)

    pair = _build_pair(module, tmp_path, paths["candidate"])

    assert (pair.baseline.root / ".git").is_dir()
    assert (pair.treatment.root / ".git").is_dir()
    assert list((pair.baseline.root / ".agents" / "skills").iterdir()) == []
    managed = pair.treatment.managed_skill_path
    assert managed is not None
    assert managed.is_symlink()
    assert managed.resolve() == paths["candidate"] / "tooluniverse-trialqa"
    assert pair.baseline.prompt_path.read_bytes() == pair.treatment.prompt_path.read_bytes()
    assert _payload_files(pair.baseline.root) == _payload_files(pair.treatment.root, managed)
    config = json.loads(
        (pair.switchyard_config_dir / "config.json").read_text(encoding="utf-8")
    )
    assert config == {"skill_distillation": {"namespace": "tooluniverse-trialqa"}}
    assert pair.prompt_sha256.startswith("sha256:")


def test_build_run_spec_has_exact_profile_mcp_and_noninteractive_codex_contract(
    tmp_path: Path,
) -> None:
    module = _load()
    paths = _fixture_paths(tmp_path)
    pair = _build_pair(module, tmp_path, paths["candidate"])

    spec = module.build_run_spec(
        pair=pair,
        arm_name="treatment",
        switchyard_bin=paths["switchyard"],
        codex_bin=paths["codex"],
        routing_profile=paths["profile"],
        tooluniverse_bin=paths["tooluniverse"],
    )

    assert spec.cwd == pair.capture_cwd
    assert spec.stdin_path == pair.treatment.prompt_path
    assert spec.stdout_path == (
        pair.treatment.root / "outputs" / "switchyard-codex.stdout.log"
    )
    assert spec.stderr_path == pair.treatment.root / "outputs" / "switchyard.stderr.log"
    assert spec.initial_route == "sd-executor"
    assert spec.executor_model == "nvidia/nvidia/nemotron-3-ultra"
    assert spec.tooluniverse_version == "1.1.11"
    assert spec.codex_bin == paths["codex"].resolve()
    assert spec.env["SWITCHYARD_CODEX_BIN"] == str(paths["codex"].resolve())
    assert spec.argv[:7] == (
        str(paths["switchyard"]),
        "--routing-profiles",
        str(paths["profile"]),
        "--",
        "launch",
        "codex",
        "--",
    )
    assert (
        f'mcp_servers.tooluniverse.command="{paths["tooluniverse"].parent / "python"}"'
        in spec.argv
    )
    adapter = REPO / "benchmark" / "trialqa_tooluniverse_mcp.py"
    assert (
        "mcp_servers.tooluniverse.args="
        f'["{adapter}","--tooluniverse-bin","{paths["tooluniverse"]}",'
        f'"--skill-path","{pair.candidate.skill_path}"]'
        in spec.argv
    )
    assert 'mcp_servers.tooluniverse.env={PYTHONIOENCODING="utf-8"}' in spec.argv
    assert "mcp_servers.tooluniverse.required=true" in spec.argv
    assert 'mcp_servers.tooluniverse.default_tools_approval_mode="approve"' in spec.argv
    enabled = "mcp_servers.tooluniverse.enabled_tools=" + json.dumps(
        list(module.TRIALQA_MCP_TOOLS), separators=(",", ":")
    )
    assert enabled in spec.argv
    assert "mcp_servers.tooluniverse.startup_timeout_sec=60" in spec.argv
    assert "mcp_servers.tooluniverse.tool_timeout_sec=60" in spec.argv
    assert 'otel.exporter="none"' in spec.argv
    assert 'otel.metrics_exporter="none"' in spec.argv
    assert 'web_search="disabled"' in spec.argv
    exec_index = spec.argv.index("exec")
    assert spec.argv[exec_index - 12 : exec_index] == (
        "-a",
        "never",
        "-s",
        "workspace-write",
        "--disable",
        "multi_agent",
        "--disable",
        "plugins",
        "--disable",
        "shell_tool",
        "-C",
        str(pair.treatment.root),
    )
    assert spec.argv[exec_index:] == (
        "exec",
        "--ephemeral",
        "--ignore-user-config",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
        str(pair.treatment.final_output_path),
        "-",
    )
    assert "--model" not in spec.argv
    assert "-m" not in spec.argv
    assert "--output-schema" not in spec.argv
    assert spec.env == {
        "HOME": str(pair.runtime_root / "treatment" / "home"),
        "CODEX_HOME": str(pair.runtime_root / "treatment" / "codex-home"),
        "SWITCHYARD_CONFIG_DIR": str(pair.switchyard_config_dir),
        "SWITCHYARD_CODEX_BIN": str(paths["codex"].resolve()),
    }


@pytest.mark.parametrize("task_id", ["../escape", "a/b", ".", "", "two words"])
def test_workspace_builder_rejects_unsafe_task_ids(tmp_path: Path, task_id: str) -> None:
    module = _load()
    candidate = _write_candidate(tmp_path / "candidate")

    with pytest.raises(module.TrialQaLocalRunnerError, match="unsafe task id"):
        module.build_trial_workspace_pair(
            capture_cwd=tmp_path / "capture",
            task_id=task_id,
            prompt="question",
            candidate_root=candidate,
        )


def test_workspace_builder_refuses_existing_pair_collision(tmp_path: Path) -> None:
    module = _load()
    candidate = _write_candidate(tmp_path / "candidate")
    _build_pair(module, tmp_path, candidate)

    with pytest.raises(module.TrialQaLocalRunnerError, match="collision"):
        _build_pair(module, tmp_path, candidate)


def test_candidate_hash_mismatch_is_rejected_before_workspace_creation(tmp_path: Path) -> None:
    module = _load()
    candidate = _write_candidate(tmp_path / "candidate")
    (candidate / "tooluniverse-trialqa" / "SKILL.md").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(module.TrialQaLocalRunnerError, match="hash mismatch"):
        _build_pair(module, tmp_path, candidate)
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize(
    ("name", "description", "message"),
    [
        ("wrong-name", DESCRIPTION, "does not contain"),
        ("tooluniverse-trialqa", "", "nonempty description"),
    ],
)
def test_candidate_frontmatter_identity_is_fail_closed(
    tmp_path: Path,
    name: str,
    description: str,
    message: str,
) -> None:
    module = _load()
    candidate = _write_candidate(
        tmp_path / "candidate",
        name=name,
        description=description,
    )

    with pytest.raises(module.TrialQaLocalRunnerError, match=message):
        module.validate_candidate_skill(candidate, "tooluniverse-trialqa")


def test_candidate_requires_passed_validation(tmp_path: Path) -> None:
    module = _load()
    candidate = _write_candidate(tmp_path / "candidate", version_status="failed")

    with pytest.raises(module.TrialQaLocalRunnerError, match="status must be passed"):
        module.validate_candidate_skill(candidate, "tooluniverse-trialqa")


@pytest.mark.parametrize(
    ("first", "model", "message"),
    [
        ("another-route", None, "first routing profile route"),
        ("sd-executor", "nvidia/nvidia/nemotron-3-super-v3", "pinned model"),
    ],
)
def test_routing_profile_must_start_with_pinned_ultra_executor(
    tmp_path: Path,
    first: str,
    model: str | None,
    message: str,
) -> None:
    module = _load()
    profile = _write_profile(tmp_path / "routes.yaml", first=first, model=model)

    with pytest.raises(module.TrialQaLocalRunnerError, match=message):
        module.validate_routing_profile(profile)


def test_tooluniverse_distribution_must_be_exactly_pinned(tmp_path: Path) -> None:
    module = _load()
    binary = _write_tooluniverse(tmp_path / "venv", version="1.1.12")

    with pytest.raises(module.TrialQaLocalRunnerError, match="pinned to 1.1.11"):
        module.validate_tooluniverse_binary(binary)


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "evil"],
        ["--model=evil"],
        ["-m", "evil"],
        ["-mevil"],
        ["--cd", "/tmp/elsewhere"],
        ["--cd=/tmp/elsewhere"],
        ["-C", "/tmp/elsewhere"],
        ["-C/tmp/elsewhere"],
        ["-c", 'model="evil"'],
        ["--config", 'model_provider="other"'],
        ["-c", 'mcp_servers.tooluniverse.command="other"'],
        ["--enable", "shell_tool"],
        ["--enable", "plugins"],
        ["--enable", "multi_agent"],
        ["-c", "features.shell_tool=true"],
        ["-c", "features.plugins=true"],
        ["-c", "features.multi_agent=true"],
        ["-c", 'web_search="live"'],
        ["-c", 'otel.metrics_exporter="statsig"'],
    ],
)
def test_run_spec_rejects_model_cwd_and_protected_mcp_overrides(
    tmp_path: Path,
    extra: list[str],
) -> None:
    module = _load()
    paths = _fixture_paths(tmp_path)
    pair = _build_pair(module, tmp_path, paths["candidate"])

    with pytest.raises(module.TrialQaLocalRunnerError, match="forbidden"):
        module.build_run_spec(
            pair=pair,
            arm_name="baseline",
            switchyard_bin=paths["switchyard"],
            codex_bin=paths["codex"],
            routing_profile=paths["profile"],
            tooluniverse_bin=paths["tooluniverse"],
            extra_codex_args=extra,
        )


def _prompt_input_json(entries: list[tuple[str, str, Path]]) -> str:
    rendered = "## Skills\n\n### Available skills\n" + "\n".join(
        f"- {name}: {description} (file: {path})" for name, description, path in entries
    )
    return json.dumps(
        [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": rendered}],
            }
        ]
    )


def test_prompt_input_attestation_uses_no_model_debug_and_exact_candidate_metadata(
    tmp_path: Path,
) -> None:
    module = _load()
    paths = _fixture_paths(tmp_path)
    pair = _build_pair(module, tmp_path, paths["candidate"])
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        cwd = Path(cast(str | os.PathLike[str], kwargs["cwd"]))
        entries = [("system-skill", "System helper.", Path("/opt/codex/system/SKILL.md"))]
        if cwd == pair.treatment.root:
            assert pair.treatment.managed_skill_path is not None
            entries.append(
                (
                    pair.candidate.name,
                    pair.candidate.description,
                    pair.candidate.skill_path,
                )
            )
        return subprocess.CompletedProcess(command, 0, stdout=_prompt_input_json(entries), stderr="")

    # Local import keeps the fixture's annotations valid without leaking it into runtime code.
    from typing import cast

    attestation = module.attest_trial_workspace_pair(
        pair=pair,
        codex_bin=paths["codex"],
        run=fake_run,
        base_environment={"PATH": "/usr/bin"},
    )

    assert len(calls) == 2
    for arm, (command, kwargs) in zip((pair.baseline, pair.treatment), calls, strict=True):
        assert command == [
            str(paths["codex"]),
            "-C",
            str(arm.root),
            "debug",
            "prompt-input",
            module.ATTESTATION_PROMPT,
        ]
        assert "exec" not in command
        assert "--model" not in command
        assert kwargs["cwd"] == arm.root
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["HOME"] == str(pair.runtime_root / arm.name / "home")
        assert environment["CODEX_HOME"] == str(pair.runtime_root / arm.name / "codex-home")
    assert all(skill.name != pair.candidate.name for skill in attestation.baseline_skills)
    candidate = next(
        skill for skill in attestation.treatment_skills if skill.name == pair.candidate.name
    )
    assert candidate.description == DESCRIPTION
    assert candidate.path == pair.candidate.skill_path


@pytest.mark.parametrize("wrong_field", ["description", "path"])
def test_prompt_input_attestation_rejects_inexact_treatment_metadata(
    tmp_path: Path,
    wrong_field: str,
) -> None:
    module = _load()
    paths = _fixture_paths(tmp_path)
    pair = _build_pair(module, tmp_path, paths["candidate"])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = Path(os.fspath(kwargs["cwd"]))
        entries: list[tuple[str, str, Path]] = []
        if cwd == pair.treatment.root:
            assert pair.treatment.managed_skill_path is not None
            description = "Wrong description" if wrong_field == "description" else DESCRIPTION
            path = (
                Path("/tmp/wrong/SKILL.md")
                if wrong_field == "path"
                else pair.candidate.skill_path
            )
            entries.append((pair.candidate.name, description, path))
        return subprocess.CompletedProcess(command, 0, stdout=_prompt_input_json(entries), stderr="")

    with pytest.raises(module.TrialQaLocalRunnerError, match=wrong_field):
        module.attest_trial_workspace_pair(
            pair=pair,
            codex_bin=paths["codex"],
            run=fake_run,
            base_environment={},
        )


def test_prompt_input_parser_expands_skill_root_aliases(tmp_path: Path) -> None:
    module = _load()
    skill_root = tmp_path / "skills"
    payload = json.dumps(
        {
            "context": (
                "### Skill roots\n"
                f"- r0: `{skill_root}`\n"
                "### Available skills\n"
                f"- tooluniverse-trialqa: {DESCRIPTION} "
                "(path: r0/tooluniverse-trialqa/SKILL.md)\n"
            )
        }
    )

    parsed = module.parse_prompt_input_skills(payload)

    assert parsed == (
        module.PromptSkill(
            name="tooluniverse-trialqa",
            description=DESCRIPTION,
            path=skill_root / "tooluniverse-trialqa" / "SKILL.md",
        ),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build isolated local TrialQA/Codex A/B workspaces and launch specifications.

This module intentionally does not execute a TrialQA model run.  It prepares two
inner Git repositories, validates an immutable Switchyard skill candidate, and
returns the exact argv/environment a caller may execute after review.  The only
Codex subprocess helper is a no-model ``debug prompt-input`` attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

TOOLUNIVERSE_VERSION = "1.1.11"
TOOLUNIVERSE_COMMAND = "tooluniverse-smcp-stdio"
TOOLUNIVERSE_ARGS = ("--compact-mode",)
TOOLUNIVERSE_ADAPTER_PATH = Path(__file__).with_name("trialqa_tooluniverse_mcp.py")
TRIALQA_MCP_TOOLS = (
    "trialqa_load_active_skill",
    "list_tools",
    "grep_tools",
    "get_tool_info",
    "execute_tool",
    "find_tools",
)
TRIALQA_EVIDENCE_TOOL = "execute_tool"
CODEX_DISABLED_FEATURES = ("multi_agent", "plugins", "shell_tool")
NAMESPACE = "tooluniverse-trialqa"
EXECUTOR_ROUTE = "sd-executor"
EXECUTOR_MODEL = "nvidia/nvidia/nemotron-3-ultra"
SCHEMA_VERSION = "switchyard.trialqa_local_runner.v1"
ATTESTATION_PROMPT = "Inspect session context for skill-discovery attestation only."

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BANNED_EXACT_ARGS = frozenset({"-m", "--model", "-C", "--cd"})
_BANNED_PREFIX_ARGS = ("--model=", "--cd=")
_BANNED_CONFIG_KEYS = frozenset(
    {"model", "model_provider", "cwd", "working_dir", "web_search"}
)

class TrialQaLocalRunnerError(RuntimeError):
    """Raised when a local TrialQA run cannot be proven safe and reproducible."""


@dataclass(frozen=True)
class CandidateSkill:
    """Validated immutable candidate skill metadata."""

    candidate_root: Path
    skill_dir: Path
    skill_path: Path
    name: str
    description: str
    sha256: str


@dataclass(frozen=True)
class TrialArm:
    """One prepared no-skill or skilled trial repository."""

    name: Literal["baseline", "treatment"]
    root: Path
    prompt_path: Path
    answer_path: Path
    final_output_path: Path
    stdout_path: Path
    stderr_path: Path
    managed_skill_path: Path | None


@dataclass(frozen=True)
class TrialWorkspacePair:
    """Prepared A/B repositories plus their shared capture/runtime roots."""

    task_id: str
    capture_cwd: Path
    pair_root: Path
    runtime_root: Path
    switchyard_config_dir: Path
    baseline: TrialArm
    treatment: TrialArm
    candidate: CandidateSkill
    prompt_sha256: str


@dataclass(frozen=True)
class RunSpec:
    """A reviewed command specification; no command is executed by this object."""

    arm: Literal["baseline", "treatment"]
    cwd: Path
    argv: tuple[str, ...]
    env: Mapping[str, str]
    stdin_path: Path
    stdout_path: Path
    stderr_path: Path
    answer_path: Path
    final_output_path: Path
    codex_bin: Path
    initial_route: str
    executor_model: str
    tooluniverse_version: str

    def json_document(self) -> dict[str, object]:
        """Return a JSON-safe, secret-free representation."""

        value = asdict(self)
        for key in (
            "cwd",
            "stdin_path",
            "stdout_path",
            "stderr_path",
            "answer_path",
            "final_output_path",
            "codex_bin",
        ):
            value[key] = str(value[key])
        value["argv"] = list(self.argv)
        value["env"] = dict(self.env)
        value["schema_version"] = SCHEMA_VERSION
        return cast(dict[str, object], value)


@dataclass(frozen=True)
class PromptSkill:
    """One skill entry parsed from Codex's model-visible prompt input."""

    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class PromptInputAttestation:
    """Parsed no-model Codex skill-discovery evidence for both arms."""

    baseline_skills: tuple[PromptSkill, ...]
    treatment_skills: tuple[PromptSkill, ...]


CompletedTextProcess = subprocess.CompletedProcess[str]
SubprocessRunner = Callable[..., CompletedTextProcess]


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise TrialQaLocalRunnerError(f"unsafe {label}: {value!r}")
    return value


def _absolute(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise TrialQaLocalRunnerError(f"{label} must be absolute: {path}")
    return expanded.absolute()


def _existing_real_directory(path: Path, label: str) -> Path:
    absolute = _absolute(path, label)
    if absolute.is_symlink() or not absolute.is_dir():
        raise TrialQaLocalRunnerError(f"{label} must be a real directory: {absolute}")
    return absolute.resolve(strict=True)


def _existing_real_file(
    path: Path,
    label: str,
    *,
    executable: bool = False,
    allow_symlink: bool = False,
) -> Path:
    absolute = _absolute(path, label)
    if absolute.is_symlink() and not allow_symlink:
        raise TrialQaLocalRunnerError(f"{label} must be a real file: {absolute}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise TrialQaLocalRunnerError(f"{label} does not resolve to a file: {absolute}") from exc
    if not resolved.is_file():
        raise TrialQaLocalRunnerError(f"{label} must be a real file: {absolute}")
    if executable and not os.access(resolved, os.X_OK):
        raise TrialQaLocalRunnerError(f"{label} is not executable: {absolute}")
    return resolved


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialQaLocalRunnerError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TrialQaLocalRunnerError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _normalized_manifest_skill_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TrialQaLocalRunnerError("candidate manifest skill path must be a string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
        or pure.name != "SKILL.md"
    ):
        raise TrialQaLocalRunnerError(f"unsafe candidate manifest skill path: {value!r}")
    return value


def _skill_frontmatter(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TrialQaLocalRunnerError(f"candidate skill is not valid UTF-8: {path}") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise TrialQaLocalRunnerError(f"candidate skill is missing YAML frontmatter: {path}")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise TrialQaLocalRunnerError(f"candidate skill has unclosed YAML frontmatter: {path}") from exc
    try:
        raw: object = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise TrialQaLocalRunnerError(f"candidate skill has invalid YAML frontmatter: {path}") from exc
    if not isinstance(raw, dict):
        raise TrialQaLocalRunnerError(f"candidate skill frontmatter must be a mapping: {path}")
    name = raw.get("name")
    description = raw.get("description")
    if not isinstance(name, str):
        raise TrialQaLocalRunnerError(f"candidate skill frontmatter needs a string name: {path}")
    name = _safe_component(name.strip(), "skill name")
    if not isinstance(description, str) or not description.strip():
        raise TrialQaLocalRunnerError(
            f"candidate skill frontmatter needs a nonempty description: {path}"
        )
    description = description.strip()
    if "\n" in description or "\r" in description:
        raise TrialQaLocalRunnerError("candidate skill description must be one line")
    return name, description


def validate_candidate_skill(candidate_root: Path, skill_directory: str) -> CandidateSkill:
    """Validate every candidate hash and return one mountable skill."""

    root = _existing_real_directory(candidate_root, "candidate root")
    selected_directory = _safe_component(skill_directory, "candidate skill directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TrialQaLocalRunnerError(f"candidate manifest is missing or symlinked: {manifest_path}")
    manifest = _read_json_object(manifest_path, "candidate manifest")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise TrialQaLocalRunnerError("candidate validation status must be passed")
    entries = manifest.get("skills")
    if not isinstance(entries, list) or not entries:
        raise TrialQaLocalRunnerError("candidate manifest must hash at least one skill")

    manifest_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TrialQaLocalRunnerError("candidate manifest skill entries must be objects")
        relative = _normalized_manifest_skill_path(entry.get("path"))
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise TrialQaLocalRunnerError(f"invalid candidate SHA-256 for {relative}")
        if relative in manifest_hashes:
            raise TrialQaLocalRunnerError(f"duplicate candidate skill path: {relative}")
        manifest_hashes[relative] = digest

    actual_skills: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise TrialQaLocalRunnerError(f"candidate bundle cannot contain symlinks: {item}")
        if item.is_file() and item.name == "SKILL.md":
            actual_skills.add(item.relative_to(root).as_posix())
    if actual_skills != set(manifest_hashes):
        raise TrialQaLocalRunnerError(
            "candidate manifest does not exactly cover its SKILL.md files: "
            f"manifest={sorted(manifest_hashes)} actual={sorted(actual_skills)}"
        )

    for relative, expected_digest in manifest_hashes.items():
        skill_path = root / Path(*PurePosixPath(relative).parts)
        if skill_path.is_symlink() or not skill_path.is_file():
            raise TrialQaLocalRunnerError(f"candidate skill is missing or symlinked: {skill_path}")
        actual_digest = hashlib.sha256(skill_path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise TrialQaLocalRunnerError(f"candidate skill hash mismatch: {relative}")

    skill_dir = root / selected_directory
    skill_path = skill_dir / "SKILL.md"
    selected_relative = skill_path.relative_to(root).as_posix()
    if selected_relative not in manifest_hashes or not skill_dir.is_dir():
        raise TrialQaLocalRunnerError(
            f"candidate manifest does not contain {selected_directory}/SKILL.md"
        )
    name, description = _skill_frontmatter(skill_path)
    if name != selected_directory:
        raise TrialQaLocalRunnerError(
            f"skill frontmatter name {name!r} does not match directory {selected_directory!r}"
        )
    return CandidateSkill(
        candidate_root=root,
        skill_dir=skill_dir,
        skill_path=skill_path,
        name=name,
        description=description,
        sha256=f"sha256:{manifest_hashes[selected_relative]}",
    )


def validate_routing_profile(path: Path) -> str:
    """Return and validate the profile's first route for the Ultra executor."""

    profile = _existing_real_file(path, "routing profile")
    try:
        document: object = yaml.safe_load(profile.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrialQaLocalRunnerError(f"invalid routing profile: {profile}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("routes"), dict):
        raise TrialQaLocalRunnerError("routing profile must contain a routes mapping")
    routes = cast(dict[object, object], document["routes"])
    if not routes:
        raise TrialQaLocalRunnerError("routing profile has no routes")
    first = next(iter(routes))
    if not isinstance(first, str):
        raise TrialQaLocalRunnerError("routing profile route names must be strings")
    _safe_component(first, "routing profile route")
    if first != EXECUTOR_ROUTE:
        raise TrialQaLocalRunnerError(
            f"first routing profile route must be {EXECUTOR_ROUTE!r}, got {first!r}"
        )
    route = routes[first]
    if not isinstance(route, dict) or route.get("type") != "model":
        raise TrialQaLocalRunnerError(f"{EXECUTOR_ROUTE} must be a model route")
    target = route.get("target")
    if not isinstance(target, dict) or target.get("model") != EXECUTOR_MODEL:
        raise TrialQaLocalRunnerError(
            f"{EXECUTOR_ROUTE} must target pinned model {EXECUTOR_MODEL!r}"
        )
    return first


def validate_tooluniverse_binary(path: Path) -> Path:
    """Require the pinned ToolUniverse console script from an inspectable venv."""

    binary = _existing_real_file(path, "ToolUniverse MCP binary", executable=True)
    if binary.name != TOOLUNIVERSE_COMMAND:
        raise TrialQaLocalRunnerError(
            f"ToolUniverse MCP binary must be named {TOOLUNIVERSE_COMMAND!r}"
        )
    venv = binary.parent.parent
    metadata_paths = sorted(
        venv.glob("lib/python*/site-packages/tooluniverse-*.dist-info/METADATA")
    )
    if len(metadata_paths) != 1:
        raise TrialQaLocalRunnerError(
            f"could not uniquely attest ToolUniverse distribution metadata under {venv}"
        )
    metadata = metadata_paths[0].read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Name", "Version"} and key not in fields:
            fields[key] = value.strip()
    if fields.get("Name", "").lower() != "tooluniverse":
        raise TrialQaLocalRunnerError("ToolUniverse distribution metadata has the wrong name")
    if fields.get("Version") != TOOLUNIVERSE_VERSION:
        raise TrialQaLocalRunnerError(
            f"ToolUniverse must be pinned to {TOOLUNIVERSE_VERSION}, "
            f"got {fields.get('Version')!r}"
        )
    return binary


def _git_init(path: Path) -> None:
    try:
        subprocess.run(
            ["git", "init", "--quiet", "--initial-branch", "main", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrialQaLocalRunnerError(f"failed to initialize inner Git repository: {path}") from exc


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_arm(stage_root: Path, name: Literal["baseline", "treatment"], prompt: bytes) -> None:
    root = stage_root / "arms" / name
    root.mkdir(parents=True)
    (root / ".agents" / "skills").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "prompt.md").write_bytes(prompt)
    (root / ".gitignore").write_text("/answer.txt\n/outputs/\n", encoding="utf-8")
    _git_init(root)


def _arm_payload_snapshot(root: Path, *, ignored_skill: Path | None) -> dict[str, tuple[str, int]]:
    snapshot: dict[str, tuple[str, int]] = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if ignored_skill is not None and item == ignored_skill:
            continue
        if item.is_symlink():
            raise TrialQaLocalRunnerError(f"unexpected trial workspace symlink: {item}")
        if item.is_file():
            snapshot[relative.as_posix()] = (
                hashlib.sha256(item.read_bytes()).hexdigest(),
                stat.S_IMODE(item.stat().st_mode),
            )
    return snapshot


def _arm_from_root(
    name: Literal["baseline", "treatment"], root: Path, managed_skill: Path | None
) -> TrialArm:
    return TrialArm(
        name=name,
        root=root,
        prompt_path=root / "prompt.md",
        answer_path=root / "answer.txt",
        final_output_path=root / "outputs" / "codex-final.json",
        # ``switchyard launch`` writes its human-readable banner to the same
        # stdout stream as Codex's ``--json`` events, so this is deliberately a
        # mixed launch log rather than a file advertised as strict JSONL.
        stdout_path=root / "outputs" / "switchyard-codex.stdout.log",
        stderr_path=root / "outputs" / "switchyard.stderr.log",
        managed_skill_path=managed_skill,
    )


def build_trial_workspace_pair(
    *,
    capture_cwd: Path,
    task_id: str,
    prompt: str,
    candidate_root: Path,
    candidate_skill_directory: str = NAMESPACE,
) -> TrialWorkspacePair:
    """Create isolated inner-Git baseline/treatment repositories atomically."""

    task_id = _safe_component(task_id, "task id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise TrialQaLocalRunnerError("TrialQA prompt must be nonempty")
    try:
        prompt_bytes = prompt.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TrialQaLocalRunnerError("TrialQA prompt must be valid UTF-8") from exc
    candidate = validate_candidate_skill(candidate_root, candidate_skill_directory)
    capture = _absolute(capture_cwd, "capture cwd")
    if capture.is_symlink():
        raise TrialQaLocalRunnerError(f"capture cwd cannot be a symlink: {capture}")
    capture.mkdir(parents=True, exist_ok=True)
    if not capture.is_dir():
        raise TrialQaLocalRunnerError(f"capture cwd is not a directory: {capture}")
    capture = capture.resolve(strict=True)

    trials_root = capture / "trialqa-local"
    pair_root = trials_root / task_id
    if pair_root.exists() or pair_root.is_symlink():
        raise TrialQaLocalRunnerError(f"trial workspace collision: {pair_root}")
    trials_root.mkdir(parents=True, exist_ok=True)
    stage = trials_root / f".{task_id}-{uuid.uuid4().hex}.tmp"
    if stage.exists() or stage.is_symlink():
        raise TrialQaLocalRunnerError(f"trial staging collision: {stage}")

    try:
        stage.mkdir()
        _build_arm(stage, "baseline", prompt_bytes)
        _build_arm(stage, "treatment", prompt_bytes)
        treatment_link = stage / "arms" / "treatment" / ".agents" / "skills" / candidate.name
        treatment_link.symlink_to(candidate.skill_dir, target_is_directory=True)
        if not treatment_link.is_symlink() or treatment_link.resolve(strict=True) != candidate.skill_dir:
            raise TrialQaLocalRunnerError("managed treatment skill symlink did not resolve exactly")

        baseline_root = stage / "arms" / "baseline"
        treatment_root = stage / "arms" / "treatment"
        baseline_snapshot = _arm_payload_snapshot(baseline_root, ignored_skill=None)
        treatment_snapshot = _arm_payload_snapshot(
            treatment_root,
            ignored_skill=treatment_link,
        )
        if baseline_snapshot != treatment_snapshot:
            raise TrialQaLocalRunnerError(
                "baseline and treatment payloads differ beyond the managed skill symlink"
            )
        if any((baseline_root / ".agents" / "skills").iterdir()):
            raise TrialQaLocalRunnerError("baseline unexpectedly contains a project skill")
        treatment_entries = list((treatment_root / ".agents" / "skills").iterdir())
        if treatment_entries != [treatment_link]:
            raise TrialQaLocalRunnerError("treatment must contain exactly one managed skill")

        runtime = stage / "runtime"
        for arm_name in ("baseline", "treatment"):
            (runtime / arm_name / "home").mkdir(parents=True)
            (runtime / arm_name / "codex-home").mkdir()
        config_dir = runtime / "switchyard-config"
        _write_json(config_dir / "config.json", {"skill_distillation": {"namespace": NAMESPACE}})
        stage.rename(pair_root)
    except BaseException:
        if stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise

    baseline_root = pair_root / "arms" / "baseline"
    treatment_root = pair_root / "arms" / "treatment"
    managed_skill = treatment_root / ".agents" / "skills" / candidate.name
    return TrialWorkspacePair(
        task_id=task_id,
        capture_cwd=capture,
        pair_root=pair_root,
        runtime_root=pair_root / "runtime",
        switchyard_config_dir=pair_root / "runtime" / "switchyard-config",
        baseline=_arm_from_root("baseline", baseline_root, None),
        treatment=_arm_from_root("treatment", treatment_root, managed_skill),
        candidate=candidate,
        prompt_sha256=f"sha256:{hashlib.sha256(prompt_bytes).hexdigest()}",
    )


def _config_override_key(argument: str) -> str | None:
    key, separator, _value = argument.partition("=")
    return key.strip() if separator else None


def _validate_extra_codex_args(arguments: Sequence[str]) -> tuple[str, ...]:
    extra = tuple(arguments)
    index = 0
    while index < len(extra):
        argument = extra[index]
        if argument in _BANNED_EXACT_ARGS or argument.startswith(_BANNED_PREFIX_ARGS):
            raise TrialQaLocalRunnerError(f"Codex model/CWD override is forbidden: {argument}")
        if argument.startswith("-m") and argument != "--":
            raise TrialQaLocalRunnerError(f"Codex model override is forbidden: {argument}")
        if argument.startswith("-C") and argument != "--":
            raise TrialQaLocalRunnerError(f"Codex CWD override is forbidden: {argument}")
        if argument in {"--enable", "--disable"}:
            if index + 1 >= len(extra):
                raise TrialQaLocalRunnerError(f"missing value for Codex option {argument}")
            if extra[index + 1] in CODEX_DISABLED_FEATURES:
                raise TrialQaLocalRunnerError("protected Codex feature override is forbidden")
            index += 2
            continue
        if argument in {"-c", "--config"}:
            if index + 1 >= len(extra):
                raise TrialQaLocalRunnerError(f"missing value for Codex option {argument}")
            key = _config_override_key(extra[index + 1])
            if key is None:
                raise TrialQaLocalRunnerError("Codex config override must contain key=value")
            if (
                key in _BANNED_CONFIG_KEYS
                or key.startswith("model_providers.")
                or key.startswith("mcp_servers.tooluniverse.")
                or key.startswith("otel.")
                or key in {
                    "features.multi_agent",
                    "features.plugins",
                    "features.shell_tool",
                }
            ):
                raise TrialQaLocalRunnerError(f"protected Codex config override is forbidden: {key}")
            index += 2
            continue
        index += 1
    return extra


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _arm_environment(
    pair: TrialWorkspacePair,
    arm: TrialArm,
    codex_bin: Path,
) -> dict[str, str]:
    runtime = pair.runtime_root / arm.name
    return {
        "HOME": str(runtime / "home"),
        "CODEX_HOME": str(runtime / "codex-home"),
        "SWITCHYARD_CONFIG_DIR": str(pair.switchyard_config_dir),
        "SWITCHYARD_CODEX_BIN": str(codex_bin),
    }


def build_run_spec(
    *,
    pair: TrialWorkspacePair,
    arm_name: Literal["baseline", "treatment"],
    switchyard_bin: Path,
    codex_bin: Path,
    routing_profile: Path,
    tooluniverse_bin: Path,
    extra_codex_args: Sequence[str] = (),
) -> RunSpec:
    """Build the exact reviewed Switchyard/Codex command without executing it."""

    switchyard = _existing_real_file(switchyard_bin, "Switchyard binary", executable=True)
    codex = _existing_real_file(
        codex_bin,
        "Codex binary",
        executable=True,
        allow_symlink=True,
    )
    profile = _existing_real_file(routing_profile, "routing profile")
    initial_route = validate_routing_profile(profile)
    tooluniverse = validate_tooluniverse_binary(tooluniverse_bin)
    adapter = _existing_real_file(TOOLUNIVERSE_ADAPTER_PATH, "TrialQA MCP adapter")
    tooluniverse_python = tooluniverse.parent / "python"
    if (
        not tooluniverse_python.exists()
        or not tooluniverse_python.is_file()
        or not os.access(tooluniverse_python, os.X_OK)
    ):
        raise TrialQaLocalRunnerError(
            f"ToolUniverse venv Python is not executable: {tooluniverse_python}"
        )
    extra = _validate_extra_codex_args(extra_codex_args)
    arm = pair.baseline if arm_name == "baseline" else pair.treatment
    if arm_name not in {"baseline", "treatment"}:
        raise TrialQaLocalRunnerError(f"unknown TrialQA arm: {arm_name!r}")
    if not arm.root.is_dir() or not (arm.root / ".git").is_dir():
        raise TrialQaLocalRunnerError(f"trial arm is not an isolated Git workspace: {arm.root}")

    adapter_args = [
        str(adapter),
        "--tooluniverse-bin",
        str(tooluniverse),
    ]
    if arm.name == "treatment":
        adapter_args.extend(("--skill-path", str(pair.candidate.skill_path)))

    argv = (
        str(switchyard),
        "--routing-profiles",
        str(profile),
        "--",
        "launch",
        "codex",
        "--",
        "-c",
        f"mcp_servers.tooluniverse.command={_toml_string(str(tooluniverse_python))}",
        "-c",
        "mcp_servers.tooluniverse.args="
        + json.dumps(adapter_args, separators=(",", ":")),
        "-c",
        'mcp_servers.tooluniverse.env={PYTHONIOENCODING="utf-8"}',
        "-c",
        "mcp_servers.tooluniverse.required=true",
        "-c",
        'mcp_servers.tooluniverse.default_tools_approval_mode="approve"',
        "-c",
        "mcp_servers.tooluniverse.enabled_tools="
        + json.dumps(list(TRIALQA_MCP_TOOLS), separators=(",", ":")),
        "-c",
        "mcp_servers.tooluniverse.startup_timeout_sec=60",
        "-c",
        "mcp_servers.tooluniverse.tool_timeout_sec=60",
        "-c",
        'otel.exporter="none"',
        "-c",
        'otel.metrics_exporter="none"',
        "-c",
        'web_search="disabled"',
        *extra,
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
        str(arm.root),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(arm.final_output_path),
        "-",
    )
    _assert_command_contract(argv, arm.root, profile)
    return RunSpec(
        arm=arm.name,
        cwd=pair.capture_cwd,
        argv=argv,
        env=_arm_environment(pair, arm, codex),
        stdin_path=arm.prompt_path,
        stdout_path=arm.stdout_path,
        stderr_path=arm.stderr_path,
        answer_path=arm.answer_path,
        final_output_path=arm.final_output_path,
        codex_bin=codex,
        initial_route=initial_route,
        executor_model=EXECUTOR_MODEL,
        tooluniverse_version=TOOLUNIVERSE_VERSION,
    )


def _assert_command_contract(argv: tuple[str, ...], trial: Path, profile: Path) -> None:
    if argv[1:7] != (
        "--routing-profiles",
        str(profile),
        "--",
        "launch",
        "codex",
        "--",
    ):
        raise TrialQaLocalRunnerError("Switchyard routing profile must precede launch codex")
    if any(arg == "--model" or arg.startswith("--model=") or arg == "-m" for arg in argv):
        raise TrialQaLocalRunnerError("run command unexpectedly overrides the profile model")
    if argv.count("-C") != 1 or argv[argv.index("-C") + 1] != str(trial):
        raise TrialQaLocalRunnerError("run command must contain exactly one pinned Codex cwd")
    exec_index = argv.index("exec")
    if argv.index("-a") > exec_index or argv.index("-s") > exec_index or argv.index("-C") > exec_index:
        raise TrialQaLocalRunnerError("Codex global approval/sandbox/cwd flags must precede exec")
    disabled = [
        argv[index + 1]
        for index, argument in enumerate(argv[:-1])
        if argument == "--disable"
    ]
    if disabled != list(CODEX_DISABLED_FEATURES):
        raise TrialQaLocalRunnerError(
            "Codex multi-agent, plugins, and shell tool must be disabled exactly"
        )
    if any(index > exec_index for index, argument in enumerate(argv) if argument == "--disable"):
        raise TrialQaLocalRunnerError("Codex feature disables must precede exec")
    expected_tail = (
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
    )
    if argv[exec_index : exec_index + len(expected_tail)] != expected_tail:
        raise TrialQaLocalRunnerError("Codex exec safety/output flags changed unexpectedly")
    if "--output-schema" in argv:
        raise TrialQaLocalRunnerError(
            "strict Codex output schemas are incompatible with reasoning-only model responses"
        )
    if "--output-last-message" not in argv:
        raise TrialQaLocalRunnerError("Codex must persist its final assistant message")
    if argv[-1] != "-":
        raise TrialQaLocalRunnerError("Codex prompt must be read from inherited stdin")


def _iter_json_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_json_strings(item)


def _strip_path_markup(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and (
        (stripped[0] == stripped[-1] == "`")
        or (stripped[0] == "<" and stripped[-1] == ">")
    ):
        return stripped[1:-1]
    return stripped


def parse_prompt_input_skills(payload: str) -> tuple[PromptSkill, ...]:
    """Parse exact rendered skill metadata from ``debug prompt-input`` JSON."""

    try:
        document: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TrialQaLocalRunnerError("Codex prompt-input output is not valid JSON") from exc
    strings = tuple(_iter_json_strings(document))
    lines = [line for value in strings for line in value.splitlines()]
    roots: dict[str, Path] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- ") or ": " not in stripped:
            continue
        alias, raw_path = stripped[2:].split(": ", maxsplit=1)
        raw_path = _strip_path_markup(raw_path)
        if re.fullmatch(r"r\d+", alias) and Path(raw_path).is_absolute():
            roots[alias] = Path(os.path.normpath(raw_path))

    parsed: list[PromptSkill] = []
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- ") or ": " not in stripped:
            continue
        body = stripped[2:]
        name, separator, remainder = body.partition(": ")
        if not separator:
            continue
        marker = " (file: " if " (file: " in remainder else " (path: "
        marker_index = remainder.rfind(marker)
        if marker_index < 0 or not remainder.endswith(")"):
            continue
        description = remainder[:marker_index]
        raw_path = _strip_path_markup(remainder[marker_index + len(marker) : -1])
        root_alias, slash, relative = raw_path.partition("/")
        if not Path(raw_path).is_absolute() and slash and root_alias in roots:
            path = roots[root_alias] / relative
        else:
            path = Path(raw_path)
        if not path.is_absolute():
            continue
        normalized = Path(os.path.normpath(str(path)))
        key = (name, description, str(normalized))
        if key in seen:
            continue
        seen.add(key)
        parsed.append(PromptSkill(name=name, description=description, path=normalized))
    return tuple(parsed)


def _run_prompt_input_debug(
    *,
    codex_bin: Path,
    arm: TrialArm,
    environment: Mapping[str, str],
    run: SubprocessRunner,
    base_environment: Mapping[str, str] | None,
) -> tuple[PromptSkill, ...]:
    # Homebrew and npm commonly expose Codex through a stable executable symlink.
    codex = _existing_real_file(
        codex_bin,
        "Codex binary",
        executable=True,
        allow_symlink=True,
    )
    command = [
        str(codex),
        "-C",
        str(arm.root),
        "debug",
        "prompt-input",
        ATTESTATION_PROMPT,
    ]
    merged_environment = dict(os.environ if base_environment is None else base_environment)
    merged_environment.update(environment)
    result = run(
        command,
        cwd=arm.root,
        env=merged_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise TrialQaLocalRunnerError(f"Codex prompt-input attestation failed: {detail}")
    return parse_prompt_input_skills(result.stdout)


def attest_trial_workspace_pair(
    *,
    pair: TrialWorkspacePair,
    codex_bin: Path,
    run: SubprocessRunner = subprocess.run,
    base_environment: Mapping[str, str] | None = None,
) -> PromptInputAttestation:
    """Prove only treatment exposes the exact managed candidate to Codex.

    ``codex debug prompt-input`` renders local context and exits without making a
    model request.  The injected runner keeps this behavior unit-testable.
    """

    codex = _existing_real_file(
        codex_bin,
        "Codex binary",
        executable=True,
        allow_symlink=True,
    )

    baseline_skills = _run_prompt_input_debug(
        codex_bin=codex,
        arm=pair.baseline,
        environment=_arm_environment(pair, pair.baseline, codex),
        run=run,
        base_environment=base_environment,
    )
    treatment_skills = _run_prompt_input_debug(
        codex_bin=codex,
        arm=pair.treatment,
        environment=_arm_environment(pair, pair.treatment, codex),
        run=run,
        base_environment=base_environment,
    )
    managed_skill = pair.treatment.managed_skill_path
    assert managed_skill is not None
    managed_path = managed_skill / "SKILL.md"
    if (
        not managed_skill.is_symlink()
        or managed_path.resolve(strict=True) != pair.candidate.skill_path
    ):
        raise TrialQaLocalRunnerError(
            "treatment managed skill no longer resolves to the validated candidate"
        )
    # Codex canonicalizes repo skill symlinks before rendering prompt input, so
    # the model-visible path is the immutable candidate target rather than the
    # managed link under the trial workspace.  The check above proves how that
    # target entered the treatment arm.
    expected_prompt_path = pair.candidate.skill_path

    baseline_collisions = [
        skill
        for skill in baseline_skills
        if skill.name == pair.candidate.name
        or skill.path in {managed_path, expected_prompt_path}
    ]
    if baseline_collisions:
        raise TrialQaLocalRunnerError(
            "baseline Codex prompt unexpectedly exposes the treatment skill"
        )
    named = [skill for skill in treatment_skills if skill.name == pair.candidate.name]
    if len(named) != 1:
        raise TrialQaLocalRunnerError(
            "treatment Codex prompt must expose exactly one candidate skill entry"
        )
    actual = named[0]
    if actual.description != pair.candidate.description:
        raise TrialQaLocalRunnerError(
            "treatment skill description differs from candidate frontmatter"
        )
    if actual.path != expected_prompt_path:
        raise TrialQaLocalRunnerError(
            f"treatment skill path is not the validated candidate target: {actual.path}"
        )
    return PromptInputAttestation(
        baseline_skills=baseline_skills,
        treatment_skills=treatment_skills,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reviewed local TrialQA/Codex A/B run specifications; never run a model."
    )
    parser.add_argument("--capture-cwd", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-skill-directory", default=NAMESPACE)
    parser.add_argument("--switchyard-bin", type=Path, required=True)
    parser.add_argument("--codex-bin", type=Path, required=True)
    parser.add_argument("--routing-profile", type=Path, required=True)
    parser.add_argument("--tooluniverse-bin", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare both arms and print secret-free run specs without executing them."""

    args = _parser().parse_args(argv)
    try:
        prompt_path = _existing_real_file(args.prompt_file, "prompt file")
        prompt = prompt_path.read_text(encoding="utf-8")
        pair = build_trial_workspace_pair(
            capture_cwd=args.capture_cwd,
            task_id=args.task_id,
            prompt=prompt,
            candidate_root=args.candidate_root,
            candidate_skill_directory=args.candidate_skill_directory,
        )
        arms: tuple[Literal["baseline", "treatment"], ...] = (
            "baseline",
            "treatment",
        )
        specs = [
            build_run_spec(
                pair=pair,
                arm_name=arm,
                switchyard_bin=args.switchyard_bin,
                codex_bin=args.codex_bin,
                routing_profile=args.routing_profile,
                tooluniverse_bin=args.tooluniverse_bin,
            ).json_document()
            for arm in arms
        ]
    except (OSError, UnicodeDecodeError, TrialQaLocalRunnerError) as exc:
        print(f"trialqa-local-runner: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema_version": SCHEMA_VERSION, "runs": specs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

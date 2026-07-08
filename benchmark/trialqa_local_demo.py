# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-free, fail-closed LABBench2 TrialQA orchestration.

The module deliberately separates four trust domains:

* ``doctor`` performs only local integrity checks and Codex's no-model
``debug prompt-input`` attestation;
* generation runs Codex through a reviewed :class:`RunSpec` and permits no
  access to the pinned parquet while the child process is alive;
* semantic judging uses a dedicated Switchyard server whose only requested
  route is ``sd-judge`` and whose routing statistics are checked per call;
* native Switchyard sessions are imported only after a real judge outcome is
  available.

No downloader is provided.  The caller must supply the exact pinned parquet
already validated by :func:`load_pinned_trialqa_parquet`, or an explicitly
hash-bound non-official TrialQA-compatible prospective parquet for fast canary
iteration.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.machinery
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the CLI itself.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

yaml = cast(Any, importlib.import_module("yaml"))

from benchmark.trialqa_local_dataset import (  # noqa: E402
    SERGEI_TEST_COUNT,
    SERGEI_TRAIN_COUNT,
    TRIALQA_DATASET_CONFIG,
    TRIALQA_DATASET_ID,
    TRIALQA_DATASET_REVISION,
    TRIALQA_PARQUET_SHA256,
    JudgeOutcome,
    TrialQADataError,
    TrialQADataset,
    TrialQAJudgeError,
    TrialQARow,
    build_reward_record,
    create_all_test_split_manifest,
    create_split_manifest,
    load_pinned_trialqa_parquet,
    load_trialqa_compatible_parquet,
    question_group_key,
    score_semantic_answer,
    task_name,
    validate_all_test_split_manifest,
    validate_split_manifest,
)
from benchmark.trialqa_local_runner import (  # noqa: E402
    CODEX_DISABLED_FEATURES,
    EXECUTOR_MODEL,
    EXECUTOR_ROUTE,
    NAMESPACE,
    TOOLUNIVERSE_ADAPTER_PATH,
    TOOLUNIVERSE_VERSION,
    TRIALQA_EVIDENCE_TOOL,
    TRIALQA_MCP_TOOLS,
    CandidateSkill,
    RunSpec,
    TrialArm,
    TrialQaLocalRunnerError,
    TrialWorkspacePair,
    attest_trial_workspace_pair,
    build_run_spec,
    build_trial_workspace_pair,
    validate_candidate_skill,
    validate_routing_profile,
    validate_tooluniverse_binary,
)
from benchmark.trialqa_tooluniverse_mcp import describe_tools_document  # noqa: E402
from switchyard.lib.skill_distillation_native import (  # noqa: E402
    NativeTrialQAEvidenceImportResult,
    import_native_trialqa_evidence,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore  # noqa: E402

SCHEMA_VERSION = "switchyard.trialqa_local_demo.v1"
MANIFEST_SCHEMA_VERSION = "switchyard.trialqa_experiment_manifest.v1"
LEDGER_SCHEMA_VERSION = "switchyard.trialqa_resumable_ledger.v1"
GENERATION_SCHEMA_VERSION = "switchyard.trialqa_generation.v1"
DOCTOR_SCHEMA_VERSION = "switchyard.trialqa_doctor.v1"

JUDGE_ROUTE = "sd-judge"
JUDGE_MODEL = "aws/anthropic/bedrock-claude-opus-4-8"
JUDGE_VERIFIER = "trialqa-semantic-judge-v1"
FULL_REPEATS = 5
PILOT_REPEATS = 1
MAX_GENERATION_CONCURRENCY = 4
CURRENT_EXPOSED_HELDOUT_QUESTION_COUNT = 88
PRIMARY_HELDOUT_QUESTION_START = CURRENT_EXPOSED_HELDOUT_QUESTION_COUNT
PRIMARY_HELDOUT_QUESTION_COUNT = SERGEI_TEST_COUNT - PRIMARY_HELDOUT_QUESTION_START
FINAL_ANSWER_JSON_SOURCE = "codex-output-last-message-json"
FINAL_ANSWER_TEXT_SOURCE = "codex-output-last-message-text"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXECUTION_SOURCE_PATHS = (
    Path("benchmark/trialqa_local_dataset.py"),
    Path("benchmark/trialqa_tooluniverse_mcp.py"),
    Path("benchmark/trialqa_local_batch.py"),
    Path("benchmark/trialqa_local_gate.py"),
    Path("benchmark/trialqa_local_regression.py"),
    Path("benchmark/trialqa_local_search_gate.py"),
    Path("benchmark/trialqa_local_runner.py"),
    Path("benchmark/trialqa_local_demo.py"),
    Path("crates/switchyard-components/src/lib.rs"),
    Path("crates/switchyard-components/src/backends/openai.rs"),
    Path("crates/switchyard-components/src/backends/stats.rs"),
    Path("crates/switchyard-components/src/stats/accumulator.rs"),
    Path("crates/switchyard-components/src/stats/mod.rs"),
    Path("crates/switchyard-translation/src/codecs/responses/buffered.rs"),
    Path("crates/switchyard-translation/src/codecs/responses/stream.rs"),
    Path("crates/switchyard-translation/src/codecs/stream.rs"),
    Path("crates/switchyard-translation/src/lib.rs"),
    Path("crates/switchyard-translation/src/namespace_tools.rs"),
    Path("switchyard/cli/launchers/codex_cli_launcher.py"),
    Path("switchyard/cli/launchers/skill_distillation.py"),
    Path("switchyard/lib/skill_distillation_native.py"),
    Path("switchyard/lib/skill_distillation_store.py"),
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_CHAT_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TERMINAL_CODEX_EVENT_TYPES = frozenset({"turn.failed", "item.failed", "thread.failed", "fatal"})

JsonObject = dict[str, Any]
Condition = Literal["donor", "baseline", "treatment"]
PlanKind = Literal["donor", "development", "pilot", "full"]
PROSPECTIVE_DATASET_ID = "trialqa-compatible-prospective"
PROSPECTIVE_DATASET_CONFIG = "clinicaltrials-gov"
PROSPECTIVE_DATASET_SPLIT = "prospective"


class TrialQADemoError(RuntimeError):
    """An orchestration invariant failed; callers must not continue silently."""


class GenerationTimeoutError(TrialQADemoError):
    """One Switchyard/Codex generation exceeded its wall-clock deadline."""

    def __init__(
        self,
        timeout_seconds: float,
        *,
        process_group_terminated: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.process_group_terminated = process_group_terminated
        super().__init__(f"Switchyard/Codex generation timed out after {timeout_seconds:g}s")


class HttpTransport(Protocol):
    """Small injectable HTTP boundary used by the local judge client."""

    def __call__(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        timeout: float,
    ) -> tuple[int, bytes]: ...


@dataclass(frozen=True)
class CandidateAttestation:
    candidate_id: str
    manifest_sha256: str
    skill_sha256: str


@dataclass(frozen=True)
class PlannedGeneration:
    task: JsonObject
    row: TrialQARow
    pair: TrialWorkspacePair
    spec: RunSpec


@dataclass(frozen=True)
class GenerationResult:
    manifest_id: str
    task_id: str
    pair_id: str
    row_id: str
    dataset_row_index: int
    partition: str
    condition: str
    repeat_index: int
    n_repeats: int
    answer: str
    answer_source: str
    session_dir: Path
    stats_path: Path
    trajectory_path: Path
    codex_events_path: Path
    final_output_path: Path
    generation_path: Path
    stats: JsonObject
    usage: JsonObject
    artifact_sha256: Mapping[str, str]

    def json_document(self) -> JsonObject:
        value = asdict(self)
        for key in (
            "session_dir",
            "stats_path",
            "trajectory_path",
            "codex_events_path",
            "final_output_path",
            "generation_path",
        ):
            value[key] = str(value[key])
        value["schema_version"] = GENERATION_SCHEMA_VERSION
        return value


@dataclass(frozen=True)
class ScoredGeneration:
    generation: GenerationResult
    outcome: JudgeOutcome
    reward: JsonObject
    evidence: NativeTrialQAEvidenceImportResult


@dataclass(frozen=True)
class TrialResultRecord:
    """Gold-free terminal record; failures remain explicit zero-score trials."""

    manifest_id: str
    task_id: str
    pair_id: str
    row_id: str
    question_group_key: str
    condition: str
    repeat_index: int
    n_repeats: int
    status: Literal["scored", "error"]
    score: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    evidence_id: str | None
    error_stage: str | None
    error_type: str | None

    def json_document(self) -> JsonObject:
        return {
            "schema_version": "switchyard.trialqa_result.v1",
            **asdict(self),
        }


@dataclass(frozen=True)
class JudgeProcessSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    base_url: str
    stdout_path: Path
    stderr_path: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _execution_source_sha256() -> JsonObject:
    """Bind an experiment manifest to the exact local execution code."""

    return {path.as_posix(): _sha256_file(_PROJECT_ROOT / path) for path in _EXECUTION_SOURCE_PATHS}


def _adapter_schema_attestation(document: Mapping[str, object] | None = None) -> JsonObject:
    value = dict(document or describe_tools_document())
    tools = value.get("tools")
    if not isinstance(tools, list):
        raise TrialQADemoError("TrialQA MCP adapter description has no tools list")
    names: list[str] = []
    for raw in tools:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise TrialQADemoError("TrialQA MCP adapter description has an invalid tool")
        names.append(cast(str, raw["name"]))
    if tuple(names) != TRIALQA_MCP_TOOLS:
        raise TrialQADemoError(
            f"TrialQA MCP adapter tools differ from the runner contract: {names}"
        )
    return {
        "schema_version": value.get("schema_version"),
        "tool_names": names,
        "description_sha256": _sha256_bytes(_canonical_json(value)),
        "adapter_sha256": _sha256_file(TOOLUNIVERSE_ADAPTER_PATH),
    }


def _runtime_binary_attestation(path: Path, label: str) -> JsonObject:
    return _runtime_file_attestation(path, label, executable=True)


def _runtime_file_attestation(
    path: Path,
    label: str,
    *,
    executable: bool = False,
) -> JsonObject:
    requested = path.expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise TrialQADemoError(f"{label} does not resolve: {requested}") from exc
    if not resolved.is_file():
        raise TrialQADemoError(f"{label} is not a file: {requested}")
    if executable and not os.access(resolved, os.X_OK):
        raise TrialQADemoError(f"{label} is not an executable file: {requested}")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
    }


def _native_extension_attestation() -> JsonObject:
    """Hash the extension file backing the loaded Rust translation engine."""

    module_name = "switchyard_rust._switchyard_rust"
    try:
        native = importlib.import_module(module_name)
    except ImportError as exc:
        raise TrialQADemoError(
            "Switchyard's native Rust extension is not importable; rebuild it before doctor"
        ) from exc
    module_file = getattr(native, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise TrialQADemoError("loaded Switchyard Rust extension has no file identity")
    if not any(module_file.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise TrialQADemoError(
            "switchyard_rust._switchyard_rust did not load from a native extension file"
        )
    return {
        "module": module_name,
        **_runtime_file_attestation(
            Path(module_file),
            "loaded Switchyard Rust extension",
        ),
    }


def _validate_namespace_translation_attestation(value: object) -> None:
    """Validate the concise result of the local namespace round-trip probe."""

    if not isinstance(value, dict):
        raise TrialQADemoError("TrialQA doctor report has no namespace translation probe")
    flattened_name = value.get("flattened_name")
    if (
        not isinstance(flattened_name, str)
        or _CHAT_FUNCTION_NAME.fullmatch(flattened_name) is None
        or flattened_name == "trialqa_load_active_skill"
    ):
        raise TrialQADemoError(
            "TrialQA doctor namespace probe did not produce a flat Chat function name"
        )
    expected = {
        "schema_version": "switchyard.trialqa_namespace_translation.v1",
        "source_format": "openai_responses",
        "target_format": "openai_chat",
        "namespace": "mcp__tooluniverse",
        "child_name": "trialqa_load_active_skill",
        "flattened_name": flattened_name,
        "flattened_name_sha256": _sha256_bytes(flattened_name.encode("ascii")),
        "request_flattened_tool_count": 1,
        "response_namespace": "mcp__tooluniverse",
        "response_child_name": "trialqa_load_active_skill",
        "response_call_id": "trialqa-doctor-call",
        "model_calls": 0,
    }
    if value != expected:
        raise TrialQADemoError("TrialQA doctor report has a stale namespace translation probe")


def _validate_doctor_report(
    path: Path,
    *,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
) -> str:
    report = _read_json_object(path, "TrialQA doctor report")
    if (
        report.get("schema_version") != DOCTOR_SCHEMA_VERSION
        or report.get("status") != "passed"
        or isinstance(report.get("model_calls"), bool)
        or report.get("model_calls") != 0
    ):
        raise TrialQADemoError("TrialQA doctor report is not a zero-call pass")
    dataset = report.get("dataset")
    if not isinstance(dataset, dict) or {
        "id": dataset.get("id"),
        "config": dataset.get("config"),
        "revision": dataset.get("revision"),
        "parquet_sha256": dataset.get("parquet_sha256"),
        "row_count": dataset.get("row_count"),
        "split_counts": dataset.get("split_counts"),
    } != {
        "id": TRIALQA_DATASET_ID,
        "config": TRIALQA_DATASET_CONFIG,
        "revision": TRIALQA_DATASET_REVISION,
        "parquet_sha256": TRIALQA_PARQUET_SHA256,
        "row_count": SERGEI_TRAIN_COUNT + SERGEI_TEST_COUNT,
        "split_counts": {"train": SERGEI_TRAIN_COUNT, "test": SERGEI_TEST_COUNT},
    }:
        raise TrialQADemoError("TrialQA doctor report has the wrong dataset attestation")
    routing = report.get("routing")
    if not isinstance(routing, dict) or {
        "first_route": routing.get("first_route"),
        "executor_model": routing.get("executor_model"),
        "judge_route": routing.get("judge_route"),
        "judge_model": routing.get("judge_model"),
    } != {
        "first_route": EXECUTOR_ROUTE,
        "executor_model": EXECUTOR_MODEL,
        "judge_route": JUDGE_ROUTE,
        "judge_model": JUDGE_MODEL,
    }:
        raise TrialQADemoError("TrialQA doctor report has the wrong routing attestation")
    if report.get("implementation") != {"source_sha256": _execution_source_sha256()}:
        raise TrialQADemoError("TrialQA doctor report is stale for the current implementation")
    if report.get("mcp_adapter") != _adapter_schema_attestation():
        raise TrialQADemoError("TrialQA doctor report has stale MCP adapter schemas")
    if report.get("codex_safety") != _codex_safety_attestation():
        raise TrialQADemoError("TrialQA doctor report has stale Codex safety settings")
    _validate_namespace_translation_attestation(report.get("namespace_translation"))
    expected_runtime = {
        "switchyard": _runtime_binary_attestation(switchyard_bin, "Switchyard binary"),
        "codex": _runtime_binary_attestation(codex_bin, "Codex binary"),
        "switchyard_rust_native_extension": _native_extension_attestation(),
        "tooluniverse": {
            **_runtime_binary_attestation(tooluniverse_bin, "ToolUniverse binary"),
            "version": TOOLUNIVERSE_VERSION,
            "python": _runtime_binary_attestation(
                tooluniverse_bin.parent / "python",
                "ToolUniverse venv Python",
            ),
        },
    }
    if report.get("runtime_artifacts") != expected_runtime:
        raise TrialQADemoError("TrialQA doctor report is stale for the selected runtime")
    return _sha256_file(path)


def _dataset_is_official_labbench2(manifest: Mapping[str, object]) -> bool:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise TrialQADemoError("experiment manifest has no dataset metadata")
    value = dataset.get("official_labbench2")
    return True if value is None else value is True


def _manifest_declared_test_count(manifest: Mapping[str, object]) -> int:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise TrialQADemoError("experiment manifest has no dataset metadata")
    value = dataset.get("test_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrialQADemoError("experiment manifest has invalid dataset test_count")
    return value


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise TrialQADemoError(f"unsafe {label}: {value!r}")
    return value


def _read_json_object(path: Path, label: str) -> JsonObject:
    if path.is_symlink() or not path.is_file():
        raise TrialQADemoError(f"{label} must be a real file: {path}")
    file_stat = path.stat()
    if file_stat.st_nlink != 1:
        raise TrialQADemoError(f"{label} must have exactly one hard link: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialQADemoError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrialQADemoError(f"{label} must be a JSON object: {path}")
    return cast(JsonObject, value)


def _write_json_atomic(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if exclusive and (path.exists() or path.is_symlink()):
        existing = _read_json_object(path, "immutable JSON document")
        if _canonical_json(existing) != _canonical_json(value):
            raise TrialQADemoError(f"immutable JSON document differs: {path}")
        return
    staging = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if staging.exists() or staging.is_symlink():
        raise TrialQADemoError(f"JSON staging collision: {staging}")
    try:
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and (path.exists() or path.is_symlink()):
            existing = _read_json_object(path, "immutable JSON document")
            if _canonical_json(existing) != _canonical_json(value):
                raise TrialQADemoError(f"immutable JSON document differs: {path}")
            return
        staging.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging.unlink()


def _candidate_attestation(candidate: CandidateSkill) -> CandidateAttestation:
    manifest_path = candidate.candidate_root / "manifest.json"
    manifest = _read_json_object(manifest_path, "candidate manifest")
    candidate_id = _safe_component(manifest.get("candidate_id"), "candidate id")
    actual_skill_hash = _sha256_file(candidate.skill_path)
    expected_skill_hash = candidate.sha256
    if _SHA256.fullmatch(expected_skill_hash) is None:
        raise TrialQADemoError("candidate runner returned an invalid skill hash")
    if actual_skill_hash != expected_skill_hash:
        raise TrialQADemoError("candidate skill changed after runner validation")
    return CandidateAttestation(
        candidate_id=candidate_id,
        manifest_sha256=_sha256_file(manifest_path),
        skill_sha256=actual_skill_hash,
    )


def _validate_judge_route(path: Path) -> None:
    try:
        document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrialQADemoError(f"invalid routing profile: {path}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("routes"), dict):
        raise TrialQADemoError("routing profile has no routes mapping")
    route = document["routes"].get(JUDGE_ROUTE)
    if not isinstance(route, dict) or route.get("type") != "model":
        raise TrialQADemoError(f"routing profile must define model route {JUDGE_ROUTE!r}")
    target = route.get("target")
    if not isinstance(target, dict) or target.get("model") != JUDGE_MODEL:
        raise TrialQADemoError(f"{JUDGE_ROUTE} must target the pinned judge {JUDGE_MODEL!r}")


def render_trial_prompt(row: TrialQARow) -> str:
    """Render the only model-visible task prompt; no gold field is accepted."""

    suffix = row.prompt_suffix.strip()
    suffix_section = f"\n\nDataset instruction:\n{suffix}" if suffix else ""
    return (
        "# LABBench2 TrialQA\n\n"
        "Answer the clinical-trial question using only the available ToolUniverse MCP "
        "tools. Call `trialqa_load_active_skill` first and follow the returned skill "
        "when available. Its explicit stop conditions, call bounds, and `never` rules "
        "are requirements. Before "
        "answering, verify that retrieved evidence explicitly identifies the field the "
        "question asks for. If it does not, inspect operation definitions and retrieve "
        "another relevant read-only evidence slice; never guess a specific value. "
        "Discover ToolUniverse operations through the compact meta-tools "
        "and invoke clinical-trial operations only through `execute_tool`; never call a "
        "raw `ClinicalTrials_*` tool directly. Do not use shell commands, web tools, "
        "generic MCP resource tools, or inspect files. Return only a JSON object "
        "with one string field named `answer`.\n\nQuestion:\n"
        f"{row.question}{suffix_section}\n"
    )


def _manifest_task(
    row: TrialQARow,
    *,
    condition: Condition,
    partition: Literal["train", "test"],
    repeat_index: int,
    n_repeats: int,
) -> JsonObject:
    group = question_group_key(row)
    pair = task_name(row, repeat_index)
    phase = (
        "donor" if condition == "donor" else "development" if partition == "train" else "evaluation"
    )
    pair_id = pair
    arm = "treatment" if condition == "treatment" else "baseline"
    return {
        "task_id": f"{pair_id}-{condition}",
        "pair_id": pair_id,
        "row_id": row.id,
        "dataset_row_index": row.dataset_row_index,
        "question_group_key": group,
        "partition": partition,
        "phase": phase,
        "condition": condition,
        "arm": arm,
        "repeat_index": repeat_index,
        "n_repeats": n_repeats,
    }


def build_experiment_manifest(
    *,
    dataset: TrialQADataset,
    split_manifest: Mapping[str, object],
    kind: PlanKind,
    candidate: CandidateSkill | None,
    routing_profile: Path,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
    doctor_report: Path,
    primary_question_start: int | None = None,
    primary_question_count: int | None = None,
) -> JsonObject:
    """Create a deterministic, gold-free protocol-stage manifest.

    ``donor`` is intentionally candidate-free and must be completed before a
    distilled candidate exists. ``development`` reuses the train question IDs
    with a candidate-attested treatment arm so it can be compared with the
    immutable historical donor control. ``pilot`` and ``full`` are held-out A/B
    stages. Keeping these stages separate prevents development results from
    becoming performance evidence.
    """

    assignments = validate_split_manifest(dataset, split_manifest)
    if kind not in {"donor", "development", "pilot", "full"}:
        raise TrialQADemoError(f"unknown plan kind: {kind!r}")
    validate_routing_profile(routing_profile)
    _validate_judge_route(routing_profile)
    validate_tooluniverse_binary(tooluniverse_bin)
    doctor_sha256 = _validate_doctor_report(
        doctor_report,
        switchyard_bin=switchyard_bin,
        codex_bin=codex_bin,
        tooluniverse_bin=tooluniverse_bin,
    )
    if kind == "donor":
        if candidate is not None:
            raise TrialQADemoError("donor manifests must not reference a candidate")
        candidate_document: JsonObject | None = None
    else:
        if candidate is None:
            raise TrialQADemoError(f"{kind} manifests require an immutable candidate")
        candidate_document = asdict(_candidate_attestation(candidate))

    train_rows = [row for row in dataset.rows if assignments[row.id] == "train"]
    test_rows = [row for row in dataset.rows if assignments[row.id] == "test"]
    if len(train_rows) != SERGEI_TRAIN_COUNT or len(test_rows) != SERGEI_TEST_COUNT:
        raise TrialQADemoError("TrialQA split is not the pinned 24/96 protocol")
    heldout_question_groups = [question_group_key(row) for row in test_rows]
    heldout_ordering = {
        "question_count": SERGEI_TEST_COUNT,
        "question_group_keys": heldout_question_groups,
        "question_group_keys_sha256": _sha256_bytes(_canonical_json(heldout_question_groups)),
    }
    if (primary_question_start is None) is not (primary_question_count is None):
        raise TrialQADemoError("primary question start and count must be supplied together")
    primary_evaluation_scope: JsonObject | None = None
    heldout_quarantine: JsonObject | None = None
    selected_test_rows = test_rows
    if primary_question_start is not None and primary_question_count is not None:
        if kind != "full":
            raise TrialQADemoError("primary evaluation scope is valid only for full manifests")
        primary_question_start, primary_question_count = _validate_primary_question_suffix(
            primary_question_start,
            primary_question_count,
        )
        selected_test_rows = test_rows[
            primary_question_start : primary_question_start + primary_question_count
        ]
        if len(selected_test_rows) != primary_question_count:
            raise TrialQADemoError("primary evaluation scope exceeds held-out rows")
        primary_evaluation_scope = {
            "question_start": primary_question_start,
            "question_count": primary_question_count,
            "repeat_count": FULL_REPEATS,
            "task_count": primary_question_count * FULL_REPEATS * 2,
            "question_group_keys_sha256": _sha256_bytes(
                _canonical_json([question_group_key(row) for row in selected_test_rows])
            ),
        }
        quarantined_groups = [question_group_key(row) for row in test_rows[:primary_question_start]]
        heldout_quarantine = {
            "question_start": 0,
            "question_count": primary_question_start,
            "disposition": "excluded-exposed-heldout",
            "question_group_keys_sha256": _sha256_bytes(_canonical_json(quarantined_groups)),
        }

    tasks: list[JsonObject] = []
    if kind == "donor":
        for row in train_rows:
            for repeat in range(1, FULL_REPEATS + 1):
                tasks.append(
                    _manifest_task(
                        row,
                        condition="donor",
                        partition="train",
                        repeat_index=repeat,
                        n_repeats=FULL_REPEATS,
                    )
                )
    elif kind == "development":
        for row in train_rows:
            for repeat in range(1, FULL_REPEATS + 1):
                tasks.append(
                    _manifest_task(
                        row,
                        condition="treatment",
                        partition="train",
                        repeat_index=repeat,
                        n_repeats=FULL_REPEATS,
                    )
                )
    elif kind == "pilot":
        manifest_rows = cast(list[dict[str, object]], split_manifest["rows"])
        split_hash = {
            cast(str, item["row_id"]): cast(str, item["split_hash"]) for item in manifest_rows
        }
        selected = min(test_rows, key=lambda row: (split_hash[row.id], row.dataset_row_index))
        for condition in ("baseline", "treatment"):
            tasks.append(
                _manifest_task(
                    selected,
                    condition=condition,
                    partition="test",
                    repeat_index=1,
                    n_repeats=PILOT_REPEATS,
                )
            )
    else:
        for row in selected_test_rows:
            for repeat in range(1, FULL_REPEATS + 1):
                for condition in ("baseline", "treatment"):
                    tasks.append(
                        _manifest_task(
                            row,
                            condition=cast(Condition, condition),
                            partition="test",
                            repeat_index=repeat,
                            n_repeats=FULL_REPEATS,
                        )
                    )

    seed: JsonObject = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "dataset": {
            "id": TRIALQA_DATASET_ID,
            "config": TRIALQA_DATASET_CONFIG,
            "revision": TRIALQA_DATASET_REVISION,
            "parquet_sha256": TRIALQA_PARQUET_SHA256,
            "split_manifest_sha256": _sha256_bytes(_canonical_json(split_manifest)),
            "train_count": SERGEI_TRAIN_COUNT,
            "test_count": SERGEI_TEST_COUNT,
            "heldout_ordering": heldout_ordering,
        },
        "candidate": candidate_document,
        "routing": {
            "profile_sha256": _sha256_file(routing_profile),
            "executor_route": EXECUTOR_ROUTE,
            "executor_model": EXECUTOR_MODEL,
            "judge_route": JUDGE_ROUTE,
            "judge_model": JUDGE_MODEL,
        },
        "implementation": {
            "source_sha256": _execution_source_sha256(),
        },
        "preflight": {
            "doctor_report_sha256": doctor_sha256,
        },
        "runtime": {
            "switchyard": _runtime_binary_attestation(switchyard_bin, "Switchyard binary"),
            "codex": _runtime_binary_attestation(codex_bin, "Codex binary"),
            "tooluniverse": {
                **_runtime_binary_attestation(tooluniverse_bin, "ToolUniverse binary"),
                "version": TOOLUNIVERSE_VERSION,
                "python": _runtime_binary_attestation(
                    tooluniverse_bin.parent / "python",
                    "ToolUniverse venv Python",
                ),
            },
        },
        "protocol": {
            "train_repeats": (FULL_REPEATS if kind in {"donor", "development"} else 0),
            "test_repeats": (
                FULL_REPEATS if kind == "full" else PILOT_REPEATS if kind == "pilot" else 0
            ),
            "conditions": (
                ["donor"]
                if kind == "donor"
                else ["treatment"]
                if kind == "development"
                else ["baseline", "treatment"]
            ),
            "performance_eligible": (kind == "full" and primary_evaluation_scope is not None),
            "primary_evaluation_scope": primary_evaluation_scope,
            "heldout_quarantine": heldout_quarantine,
            "control_design": (
                "historical-cached-donor"
                if kind == "development"
                else "candidate-free-donor"
                if kind == "donor"
                else "concurrent-paired"
            ),
            "gold_in_manifest": False,
            "executor_draw_policy": "one-completed-draw-per-task-v1",
            "max_generation_concurrency": MAX_GENERATION_CONCURRENCY,
            "arm_order_policy": (
                "manifest-order-single-arm-v1"
                if kind in {"donor", "development"}
                else "deterministic-balanced-crossover-v1"
            ),
            "final_answer_policy": "raw-last-message-with-exact-json-unwrapping-v1",
            "passive_codex_items": ["todo_list"],
            "recovered_trialqa_tool_errors": "retained-as-telemetry",
            "recovered_executor_errors": "retained-as-telemetry",
            "recovered_codex_error_events": "telemetry-before-turn.completed",
            "batch_driver": "benchmark/trialqa_local_batch.py",
        },
        "tasks": tasks,
    }
    manifest_id = f"trialqa-{kind}-{hashlib.sha256(_canonical_json(seed)).hexdigest()[:20]}"
    manifest = {"manifest_id": manifest_id, **seed}
    validate_manifest_pairing(manifest)
    return manifest


def _validate_prospective_population_report(
    report_path: Path,
    *,
    dataset: TrialQADataset,
) -> str:
    """Validate the zero-spend provenance report for a compatible population."""

    report = _read_json_object(report_path, "prospective population report")
    if (
        report.get("schema_version") != "switchyard.trialqa_prospective_population.v1"
        or report.get("status") != "passed"
    ):
        raise TrialQADemoError("prospective population report is not a pass")
    population = report.get("population")
    if not isinstance(population, dict):
        raise TrialQADemoError("prospective population report has no population metadata")
    if (
        population.get("kind") != "trialqa-compatible-clinicaltrials-gov-prospective"
        or population.get("official_labbench2") is not False
        or population.get("sha256") != dataset.parquet_sha256
        or population.get("row_count") != len(dataset.rows)
    ):
        raise TrialQADemoError("prospective population report does not match the dataset")
    exclusion = report.get("official_trialqa_exclusion")
    if not isinstance(exclusion, dict) or exclusion.get("selected_ncts_overlap_official_trialqa") != []:
        raise TrialQADemoError("prospective population overlaps official TrialQA NCTs")
    constraints = report.get("use_constraints")
    if not isinstance(constraints, dict) or constraints != {
        "model_calls": 0,
        "must_not_be_reported_as_official_labbench2_trialqa": True,
        "performance_eligible_only_if_manifest_is_frozen_before_generation": True,
    }:
        raise TrialQADemoError("prospective population report has invalid use constraints")
    return _sha256_file(report_path)


def build_prospective_experiment_manifest(
    *,
    dataset: TrialQADataset,
    population_report: Path,
    candidate: CandidateSkill,
    routing_profile: Path,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
    doctor_report: Path,
) -> JsonObject:
    """Create a full paired manifest for a non-official prospective population."""

    if not dataset.rows:
        raise TrialQADemoError("prospective population must not be empty")
    validate_routing_profile(routing_profile)
    _validate_judge_route(routing_profile)
    validate_tooluniverse_binary(tooluniverse_bin)
    doctor_sha256 = _validate_doctor_report(
        doctor_report,
        switchyard_bin=switchyard_bin,
        codex_bin=codex_bin,
        tooluniverse_bin=tooluniverse_bin,
    )
    population_report_sha256 = _validate_prospective_population_report(
        population_report,
        dataset=dataset,
    )
    split_manifest = create_all_test_split_manifest(
        dataset,
        dataset_id=PROSPECTIVE_DATASET_ID,
        dataset_config=PROSPECTIVE_DATASET_CONFIG,
        split=PROSPECTIVE_DATASET_SPLIT,
    )
    validate_all_test_split_manifest(
        dataset,
        split_manifest,
        dataset_id=PROSPECTIVE_DATASET_ID,
        dataset_config=PROSPECTIVE_DATASET_CONFIG,
        split=PROSPECTIVE_DATASET_SPLIT,
    )
    heldout_question_groups = [question_group_key(row) for row in dataset.rows]
    heldout_ordering = {
        "question_count": len(dataset.rows),
        "question_group_keys": heldout_question_groups,
        "question_group_keys_sha256": _sha256_bytes(_canonical_json(heldout_question_groups)),
    }
    tasks: list[JsonObject] = []
    for row in dataset.rows:
        for repeat in range(1, FULL_REPEATS + 1):
            for condition in ("baseline", "treatment"):
                tasks.append(
                    _manifest_task(
                        row,
                        condition=condition,
                        partition="test",
                        repeat_index=repeat,
                        n_repeats=FULL_REPEATS,
                    )
                )

    primary_evaluation_scope = {
        "question_start": 0,
        "question_count": len(dataset.rows),
        "repeat_count": FULL_REPEATS,
        "task_count": len(dataset.rows) * FULL_REPEATS * 2,
        "question_group_keys_sha256": _sha256_bytes(_canonical_json(heldout_question_groups)),
    }
    seed: JsonObject = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "full",
        "dataset": {
            "id": PROSPECTIVE_DATASET_ID,
            "config": PROSPECTIVE_DATASET_CONFIG,
            "split": PROSPECTIVE_DATASET_SPLIT,
            "revision": dataset.revision,
            "parquet_sha256": dataset.parquet_sha256,
            "population_report_sha256": population_report_sha256,
            "split_manifest_sha256": _sha256_bytes(_canonical_json(split_manifest)),
            "row_count": len(dataset.rows),
            "train_count": 0,
            "test_count": len(dataset.rows),
            "official_labbench2": False,
            "heldout_ordering": heldout_ordering,
        },
        "candidate": asdict(_candidate_attestation(candidate)),
        "routing": {
            "profile_sha256": _sha256_file(routing_profile),
            "executor_route": EXECUTOR_ROUTE,
            "executor_model": EXECUTOR_MODEL,
            "judge_route": JUDGE_ROUTE,
            "judge_model": JUDGE_MODEL,
        },
        "implementation": {
            "source_sha256": _execution_source_sha256(),
        },
        "preflight": {
            "doctor_report_sha256": doctor_sha256,
            "doctor_dataset_attestation": "official-labbench2-runtime-preflight",
        },
        "runtime": {
            "switchyard": _runtime_binary_attestation(switchyard_bin, "Switchyard binary"),
            "codex": _runtime_binary_attestation(codex_bin, "Codex binary"),
            "tooluniverse": {
                **_runtime_binary_attestation(tooluniverse_bin, "ToolUniverse binary"),
                "version": TOOLUNIVERSE_VERSION,
                "python": _runtime_binary_attestation(
                    tooluniverse_bin.parent / "python",
                    "ToolUniverse venv Python",
                ),
            },
        },
        "protocol": {
            "train_repeats": 0,
            "test_repeats": FULL_REPEATS,
            "conditions": ["baseline", "treatment"],
            "performance_eligible": True,
            "primary_evaluation_scope": primary_evaluation_scope,
            "heldout_quarantine": {
                "question_start": 0,
                "question_count": 0,
                "disposition": "none-new-prospective-population",
                "question_group_keys_sha256": _sha256_bytes(_canonical_json([])),
            },
            "control_design": "concurrent-paired",
            "gold_in_manifest": False,
            "executor_draw_policy": "one-completed-draw-per-task-v1",
            "max_generation_concurrency": MAX_GENERATION_CONCURRENCY,
            "arm_order_policy": "deterministic-balanced-crossover-v1",
            "final_answer_policy": "raw-last-message-with-exact-json-unwrapping-v1",
            "passive_codex_items": ["todo_list"],
            "recovered_trialqa_tool_errors": "retained-as-telemetry",
            "recovered_executor_errors": "retained-as-telemetry",
            "recovered_codex_error_events": "telemetry-before-turn.completed",
            "batch_driver": "benchmark/trialqa_local_batch.py",
            "prospective_population_kind": "trialqa-compatible-clinicaltrials-gov",
        },
        "tasks": tasks,
    }
    manifest_id = f"trialqa-full-{hashlib.sha256(_canonical_json(seed)).hexdigest()[:20]}"
    manifest = {"manifest_id": manifest_id, **seed}
    validate_manifest_pairing(manifest)
    return manifest


def _manifest_heldout_question_groups(manifest: Mapping[str, object]) -> tuple[str, ...]:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise TrialQADemoError("experiment manifest has no dataset metadata")
    official_labbench2 = dataset.get("official_labbench2")
    expected_question_count = (
        SERGEI_TEST_COUNT
        if official_labbench2 is not False
        else _manifest_declared_test_count(manifest)
    )
    ordering = dataset.get("heldout_ordering")
    if not isinstance(ordering, dict) or set(ordering) != {
        "question_count",
        "question_group_keys",
        "question_group_keys_sha256",
    }:
        raise TrialQADemoError("held-out question ordering attestation is invalid")
    raw_groups = ordering.get("question_group_keys")
    if (
        ordering.get("question_count") != expected_question_count
        or not isinstance(raw_groups, list)
        or len(raw_groups) != expected_question_count
        or not all(isinstance(group, str) and group for group in raw_groups)
        or len(set(cast(list[str], raw_groups))) != expected_question_count
    ):
        raise TrialQADemoError("held-out question ordering attestation is invalid")
    groups = tuple(cast(list[str], raw_groups))
    if ordering.get("question_group_keys_sha256") != _sha256_bytes(_canonical_json(list(groups))):
        raise TrialQADemoError("held-out question ordering digest is invalid")
    return groups


def _validate_manifest_id(manifest: Mapping[str, object], kind: str) -> None:
    seed = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected = f"trialqa-{kind}-{hashlib.sha256(_canonical_json(seed)).hexdigest()[:20]}"
    if manifest.get("manifest_id") != expected:
        raise TrialQADemoError("experiment manifest ID does not match its canonical contents")


def validate_manifest_pairing(manifest: Mapping[str, object]) -> None:
    """Require exact baseline/treatment evaluation pairs and unique donor tasks."""

    tasks = manifest.get("tasks")
    kind = manifest.get("kind")
    if kind not in {"donor", "development", "pilot", "full"}:
        raise TrialQADemoError(f"experiment manifest has an invalid kind: {kind!r}")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise TrialQADemoError("experiment manifest has no protocol metadata")
    manifest_max_generation_concurrency(manifest)
    heldout_question_groups = _manifest_heldout_question_groups(manifest)
    if not isinstance(tasks, list) or not tasks:
        raise TrialQADemoError("experiment manifest has no tasks")
    task_ids: set[str] = set()
    eval_arms: dict[tuple[str, int], dict[str, JsonObject]] = {}
    donor_keys: set[tuple[str, int]] = set()
    for raw in tasks:
        if not isinstance(raw, dict):
            raise TrialQADemoError("experiment manifest tasks must be objects")
        task_id = _safe_component(raw.get("task_id"), "task id")
        if task_id in task_ids:
            raise TrialQADemoError(f"duplicate manifest task: {task_id}")
        task_ids.add(task_id)
        group = _safe_component(raw.get("question_group_key"), "question group key")
        repeat = raw.get("repeat_index")
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            raise TrialQADemoError(f"invalid repeat index for {task_id}")
        condition = raw.get("condition")
        row_id = raw.get("row_id")
        row_index = raw.get("dataset_row_index")
        if (
            not isinstance(row_id, str)
            or not row_id
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
        ):
            raise TrialQADemoError(f"manifest task has invalid row identity: {task_id}")
        expected_group = (
            f"trialqa-{row_index:04d}-{hashlib.sha256(row_id.encode()).hexdigest()[:12]}"
        )
        expected_pair = f"{expected_group}-r{repeat:03d}"
        partition = raw.get("partition")
        expected_phase = (
            "donor"
            if condition == "donor"
            else "development"
            if partition == "train"
            else "evaluation"
        )
        expected_arm = "treatment" if condition == "treatment" else "baseline"
        if (
            group != expected_group
            or raw.get("pair_id") != expected_pair
            or task_id != f"{expected_pair}-{condition}"
            or raw.get("phase") != expected_phase
            or raw.get("arm") != expected_arm
        ):
            raise TrialQADemoError(f"manifest task identity is inconsistent: {task_id}")
        key = (group, repeat)
        if condition == "donor":
            if raw.get("partition") != "train" or key in donor_keys:
                raise TrialQADemoError("donor tasks must be unique train tasks")
            donor_keys.add(key)
        elif condition == "treatment" and kind == "development":
            if (
                raw.get("partition") != "train"
                or raw.get("phase") != "development"
                or key in donor_keys
            ):
                raise TrialQADemoError("development tasks must be unique train treatment tasks")
            donor_keys.add(key)
        elif condition in {"baseline", "treatment"}:
            if raw.get("partition") != "test" or kind not in {"pilot", "full"}:
                raise TrialQADemoError("evaluation tasks must use the held-out partition")
            expected_repeats = FULL_REPEATS if kind == "full" else PILOT_REPEATS
            if raw.get("n_repeats") != expected_repeats:
                raise TrialQADemoError("evaluation task has an invalid repeat count")
            arms = eval_arms.setdefault(key, {})
            if condition in arms:
                raise TrialQADemoError(f"duplicate evaluation arm for {key!r}: {condition}")
            arms[cast(str, condition)] = raw
        else:
            raise TrialQADemoError(f"invalid task condition for {task_id}: {condition!r}")
    incomplete = [key for key, arms in eval_arms.items() if set(arms) != {"baseline", "treatment"}]
    if incomplete:
        raise TrialQADemoError(f"manifest contains unpaired evaluation tasks: {incomplete[0]!r}")
    for key, arms in eval_arms.items():
        baseline = arms["baseline"]
        treatment = arms["treatment"]
        for field in (
            "pair_id",
            "row_id",
            "dataset_row_index",
            "question_group_key",
            "repeat_index",
            "n_repeats",
        ):
            if baseline.get(field) != treatment.get(field):
                raise TrialQADemoError(f"evaluation pair differs at {field}: {key!r}")
    conditions = {cast(str, task["condition"]) for task in cast(list[JsonObject], tasks)}
    expected_conditions = {
        "donor": {"donor"},
        "development": {"treatment"},
        "pilot": {"baseline", "treatment"},
        "full": {"baseline", "treatment"},
    }[kind]
    if conditions != expected_conditions:
        raise TrialQADemoError(f"{kind} manifest has invalid conditions: {conditions}")

    group_order = list(
        dict.fromkeys(
            cast(str, task["question_group_key"])
            for task in cast(list[JsonObject], tasks)
            if task.get("condition") in {"baseline", "treatment"}
        )
    )
    if kind in {"pilot", "full"}:
        expected_repeats = FULL_REPEATS if kind == "full" else PILOT_REPEATS
        expected_eval_keys = {
            (group, repeat) for group in group_order for repeat in range(1, expected_repeats + 1)
        }
        if set(eval_arms) != expected_eval_keys:
            raise TrialQADemoError(
                "evaluation tasks do not contain exact per-question repeat coverage"
            )

    if kind == "full":
        primary = protocol.get("primary_evaluation_scope")
        quarantine = protocol.get("heldout_quarantine")
        official_labbench2 = _dataset_is_official_labbench2(manifest)
        heldout_question_count = _manifest_declared_test_count(manifest)
        minimum_primary_start = PRIMARY_HELDOUT_QUESTION_START if official_labbench2 else 0
        if primary is None:
            if quarantine is not None or protocol.get("performance_eligible") is not False:
                raise TrialQADemoError(
                    "descriptive full manifests must be nonperformance and unquarantined"
                )
            if group_order != list(heldout_question_groups):
                raise TrialQADemoError(
                    "descriptive full manifest must contain the ordered held-out questions"
                )
        else:
            if not isinstance(primary, dict):
                raise TrialQADemoError("full manifest has an invalid primary evaluation scope")
            primary_start, primary_count = _validate_primary_question_suffix(
                primary.get("question_start"),
                primary.get("question_count"),
                minimum_start=minimum_primary_start,
                heldout_question_count=heldout_question_count,
            )
            primary_task_count = primary_count * FULL_REPEATS * 2
            quarantine_disposition = (
                "excluded-exposed-heldout"
                if official_labbench2
                else "none-new-prospective-population"
            )
            expected_primary = {
                "question_start": primary_start,
                "question_count": primary_count,
                "repeat_count": FULL_REPEATS,
                "task_count": primary_task_count,
                "question_group_keys_sha256": _sha256_bytes(
                    _canonical_json(list(heldout_question_groups[primary_start:]))
                ),
            }
            if primary != expected_primary:
                raise TrialQADemoError("full manifest has an invalid primary evaluation scope")
            if (
                not isinstance(quarantine, dict)
                or set(quarantine)
                != {
                    "question_start",
                    "question_count",
                    "disposition",
                    "question_group_keys_sha256",
                }
                or quarantine.get("question_start") != 0
                or quarantine.get("question_count") != primary_start
                or quarantine.get("disposition") != quarantine_disposition
                or quarantine.get("question_group_keys_sha256")
                != _sha256_bytes(_canonical_json(list(heldout_question_groups[:primary_start])))
                or protocol.get("performance_eligible") is not True
                or group_order != list(heldout_question_groups[primary_start:])
                or len(tasks) != primary_task_count
            ):
                raise TrialQADemoError("full manifest quarantine attestation is invalid")
    _validate_manifest_id(manifest, kind)


def _validate_primary_question_suffix(
    start: object,
    count: object,
    *,
    minimum_start: int = PRIMARY_HELDOUT_QUESTION_START,
    heldout_question_count: int = SERGEI_TEST_COUNT,
) -> tuple[int, int]:
    """Require a nonempty contiguous held-out suffix after the exposed prefix."""

    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        raise TrialQADemoError("primary evaluation window is not integral")
    if minimum_start < 0:
        raise TrialQADemoError("primary evaluation minimum start must not be negative")
    if heldout_question_count < 1:
        raise TrialQADemoError("primary evaluation held-out count must be positive")
    if start < minimum_start:
        raise TrialQADemoError(
            "primary evaluation must start at or after held-out ordinal "
            f"{minimum_start}"
        )
    if start >= heldout_question_count:
        raise TrialQADemoError("primary evaluation scope must be nonempty")
    if count != heldout_question_count - start:
        raise TrialQADemoError(
            f"primary evaluation scope must be a contiguous suffix through ordinal "
            f"{heldout_question_count - 1}"
        )
    return start, count


def manifest_max_generation_concurrency(manifest: Mapping[str, object]) -> int:
    """Return the immutable generation-worker ceiling declared by a manifest."""

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise TrialQADemoError("experiment manifest has no protocol metadata")
    value = protocol.get("max_generation_concurrency")
    if isinstance(value, bool) or not isinstance(value, int) or value != MAX_GENERATION_CONCURRENCY:
        raise TrialQADemoError(
            f"experiment manifest max_generation_concurrency must be {MAX_GENERATION_CONCURRENCY}"
        )
    return value


def primary_evaluation_window(
    manifest: Mapping[str, object],
) -> tuple[int | None, int | None]:
    """Return the immutable primary held-out window declared by a manifest."""

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise TrialQADemoError("experiment manifest has no protocol metadata")
    value = protocol.get("primary_evaluation_scope")
    if value is None:
        return None, None
    if not isinstance(value, dict) or set(value) != {
        "question_start",
        "question_count",
        "repeat_count",
        "task_count",
        "question_group_keys_sha256",
    }:
        raise TrialQADemoError("primary evaluation scope metadata is invalid")
    start = value.get("question_start")
    count = value.get("question_count")
    official_labbench2 = _dataset_is_official_labbench2(manifest)
    heldout_question_count = (
        SERGEI_TEST_COUNT
        if official_labbench2
        else _manifest_declared_test_count(manifest)
    )
    start, count = _validate_primary_question_suffix(
        start,
        count,
        minimum_start=PRIMARY_HELDOUT_QUESTION_START if official_labbench2 else 0,
        heldout_question_count=heldout_question_count,
    )
    if value.get("repeat_count") != FULL_REPEATS or value.get("task_count") != (
        count * FULL_REPEATS * 2
    ):
        raise TrialQADemoError("primary evaluation scope task counts are invalid")
    heldout_question_groups = _manifest_heldout_question_groups(manifest)
    if value.get("question_group_keys_sha256") != _sha256_bytes(
        _canonical_json(list(heldout_question_groups[start:]))
    ):
        raise TrialQADemoError("primary evaluation scope ordering digest is invalid")
    return start, count


def load_manifest_dataset(dataset_path: Path, manifest: Mapping[str, object]) -> TrialQADataset:
    """Load the dataset declared by ``manifest`` from ``dataset_path``.

    Official LABBench2 manifests stay pinned to EdisonScientific/labbench2.
    Prospective canaries are intentionally non-official, so their manifest must
    carry the compatible population identity, row count, revision, and parquet
    digest that bind the local artifact.
    """

    if _dataset_is_official_labbench2(manifest):
        return load_pinned_trialqa_parquet(dataset_path)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise TrialQADemoError("experiment manifest has no dataset metadata")
    if (
        dataset.get("id") != PROSPECTIVE_DATASET_ID
        or dataset.get("config") != PROSPECTIVE_DATASET_CONFIG
        or dataset.get("split") != PROSPECTIVE_DATASET_SPLIT
    ):
        raise TrialQADemoError("prospective manifest has invalid dataset identity")
    revision = dataset.get("revision")
    parquet_sha256 = dataset.get("parquet_sha256")
    row_count = dataset.get("row_count")
    if not isinstance(revision, str) or not revision:
        raise TrialQADemoError("prospective manifest has invalid dataset revision")
    if not isinstance(parquet_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", parquet_sha256):
        raise TrialQADemoError("prospective manifest has invalid parquet sha256")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise TrialQADemoError("prospective manifest has invalid row count")
    return load_trialqa_compatible_parquet(
        dataset_path,
        expected_sha256=parquet_sha256,
        expected_row_count=row_count,
        revision=revision,
    )


def create_manifest_split(dataset: TrialQADataset, manifest: Mapping[str, object]) -> JsonObject:
    """Return the split manifest required by ``manifest`` for ``dataset``."""

    if _dataset_is_official_labbench2(manifest):
        return create_split_manifest(dataset)
    return create_all_test_split_manifest(
        dataset,
        dataset_id=PROSPECTIVE_DATASET_ID,
        dataset_config=PROSPECTIVE_DATASET_CONFIG,
        split=PROSPECTIVE_DATASET_SPLIT,
    )


def build_reproducible_manifest_from_supplied(
    *,
    supplied: Mapping[str, object],
    dataset: TrialQADataset,
    split_manifest: Mapping[str, object],
    candidate: CandidateSkill | None,
    routing_profile: Path,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
    doctor_report: Path,
    population_report: Path | None = None,
) -> JsonObject:
    """Rebuild ``supplied`` from current local inputs for fail-closed execution."""

    kind = supplied.get("kind")
    if kind not in {"donor", "development", "pilot", "full"}:
        raise TrialQADemoError("experiment manifest has an invalid kind")
    if _dataset_is_official_labbench2(supplied):
        return build_experiment_manifest(
            dataset=dataset,
            split_manifest=split_manifest,
            kind=cast(PlanKind, kind),
            candidate=candidate,
            routing_profile=routing_profile,
            switchyard_bin=switchyard_bin,
            codex_bin=codex_bin,
            tooluniverse_bin=tooluniverse_bin,
            doctor_report=doctor_report,
            primary_question_start=primary_evaluation_window(supplied)[0],
            primary_question_count=primary_evaluation_window(supplied)[1],
        )
    if kind != "full":
        raise TrialQADemoError("prospective canary manifests must use kind 'full'")
    if candidate is None:
        raise TrialQADemoError("prospective canary manifests require an immutable candidate")
    if population_report is None:
        raise TrialQADemoError("prospective canary manifests require --population-report")
    expected_split = create_manifest_split(dataset, supplied)
    if dict(split_manifest) != dict(expected_split):
        raise TrialQADemoError("prospective split manifest differs from current dataset")
    return build_prospective_experiment_manifest(
        dataset=dataset,
        population_report=population_report,
        candidate=candidate,
        routing_profile=routing_profile,
        switchyard_bin=switchyard_bin,
        codex_bin=codex_bin,
        tooluniverse_bin=tooluniverse_bin,
        doctor_report=doctor_report,
    )


def write_experiment_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    validate_manifest_pairing(manifest)
    _write_json_atomic(path, dict(manifest), exclusive=True)


def _manifest_task_by_id(manifest: Mapping[str, object], task_id: str) -> JsonObject:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise TrialQADemoError("manifest tasks are missing")
    matches = [task for task in tasks if isinstance(task, dict) and task.get("task_id") == task_id]
    if len(matches) != 1:
        raise TrialQADemoError(f"manifest does not contain exactly one task {task_id!r}")
    return cast(JsonObject, matches[0])


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _meaningful_gold_markers(
    dataset: TrialQADataset, partition: str, split: Mapping[str, object]
) -> tuple[str, ...]:
    try:
        assignments = validate_split_manifest(dataset, split)
    except TrialQADataError:
        assignments = validate_all_test_split_manifest(
            dataset,
            split,
            dataset_id=PROSPECTIVE_DATASET_ID,
            dataset_config=PROSPECTIVE_DATASET_CONFIG,
            split=PROSPECTIVE_DATASET_SPLIT,
        )
    markers: set[str] = set()
    for row in dataset.rows:
        if assignments[row.id] != partition:
            continue
        for value in (row.ideal, row.key_passage, *row.sources):
            normalized = value.strip()
            if len(normalized) >= 16:
                markers.add(normalized)
    return tuple(sorted(markers, key=len, reverse=True))


def assert_trial_inputs_gold_free(
    *,
    dataset: TrialQADataset,
    split_manifest: Mapping[str, object],
    pair: TrialWorkspacePair,
    row: TrialQARow,
    partition: str,
) -> None:
    """Prove the executor-visible prompt/skill contain no held-out gold text."""

    expected = render_trial_prompt(row).encode("utf-8")
    for arm in (pair.baseline, pair.treatment):
        if arm.prompt_path.read_bytes() != expected:
            raise TrialQADemoError(f"trial prompt differs from gold-free renderer: {arm.name}")
        if _is_relative_to(dataset.path.resolve(), arm.root.resolve()):
            raise TrialQADemoError("pinned parquet is inside an executor workspace")
        for item in arm.root.rglob("*.parquet"):
            if item.is_file():
                raise TrialQADemoError(f"executor workspace contains parquet input: {item}")

    prompt_text = expected.decode("utf-8")
    direct_gold = (
        row.ideal.strip(),
        row.key_passage.strip(),
        *[source.strip() for source in row.sources],
    )
    for marker in direct_gold:
        if len(marker) >= 16 and marker in prompt_text:
            raise TrialQADemoError("model-visible prompt contains a gold-only field")

    if partition == "test":
        skill_text = pair.candidate.skill_path.read_text(encoding="utf-8")
        for marker in _meaningful_gold_markers(dataset, "test", split_manifest):
            if marker in skill_text:
                raise TrialQADemoError("candidate skill contains held-out gold text")


class GoldArtifactLockdown:
    """Temporarily remove every permission bit from the sole parquet artifact."""

    def __init__(self, path: Path, *, expected_sha256: str = TRIALQA_PARQUET_SHA256) -> None:
        self.path = path.expanduser().absolute()
        self.expected_sha256 = expected_sha256.removeprefix("sha256:")
        self._mode: int | None = None

    def __enter__(self) -> GoldArtifactLockdown:
        if self.path.is_symlink() or not self.path.is_file():
            raise TrialQADemoError(f"gold artifact must be a real file: {self.path}")
        file_stat = self.path.stat()
        if file_stat.st_nlink != 1:
            raise TrialQADemoError("gold artifact must have exactly one hard link")
        if _sha256_file(self.path) != f"sha256:{self.expected_sha256}":
            raise TrialQADemoError("gold artifact hash changed before lockdown")
        self._mode = stat.S_IMODE(file_stat.st_mode)
        os.chmod(self.path, 0, follow_symlinks=False)
        if stat.S_IMODE(self.path.stat().st_mode) != 0:
            raise TrialQADemoError("failed to remove parquet permissions")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._mode is None:
            return
        os.chmod(self.path, self._mode, follow_symlinks=False)
        if _sha256_file(self.path) != f"sha256:{self.expected_sha256}":
            raise TrialQADemoError("gold artifact changed while locked down")


def assert_unique_gold_artifact(
    dataset_path: Path,
    search_root: Path,
    *,
    expected_sha256: str = TRIALQA_PARQUET_SHA256,
) -> None:
    """Reject a second readable copy of the pinned parquet under experiment scope."""

    dataset_real = dataset_path.resolve(strict=True)
    root = search_root.resolve(strict=True)
    matches: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in {".git", "__pycache__"}]
        for filename in files:
            candidate = Path(directory) / filename
            if candidate.is_symlink():
                continue
            try:
                if (
                    candidate.stat().st_size == dataset_real.stat().st_size
                    and _sha256_file(candidate)
                    == f"sha256:{expected_sha256.removeprefix('sha256:')}"
                ):
                    matches.append(candidate.resolve())
            except OSError:
                continue
    if matches != [dataset_real]:
        raise TrialQADemoError(
            f"expected one pinned gold artifact under {root}, found {len(matches)}"
        )


def _arm(root: Path, name: Literal["baseline", "treatment"], managed: Path | None) -> TrialArm:
    return TrialArm(
        name=name,
        root=root,
        prompt_path=root / "prompt.md",
        answer_path=root / "answer.txt",
        final_output_path=root / "outputs" / "codex-final.json",
        stdout_path=root / "outputs" / "switchyard-codex.stdout.log",
        stderr_path=root / "outputs" / "switchyard.stderr.log",
        managed_skill_path=managed,
    )


def _initialize_inner_git(path: Path) -> None:
    result = subprocess.run(
        ["git", "init", "--quiet", "--initial-branch", "main", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise TrialQADemoError(f"failed to initialize donor Git workspace: {path}")


def _bootstrap_candidate(runtime_root: Path) -> CandidateSkill:
    """Return a non-mountable marker used only to satisfy the pair data shape."""

    root = runtime_root / "bootstrap-no-candidate"
    marker = root / "NO_SKILL"
    if not marker.exists():
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "This donor run is explicitly unskilled; this file is not a Codex skill.\n",
            encoding="utf-8",
        )
        _write_json_atomic(
            root / "attestation.json",
            {"schema_version": SCHEMA_VERSION, "loaded": False, "candidate_id": None},
            exclusive=True,
        )
    return CandidateSkill(
        candidate_root=root,
        skill_dir=root,
        skill_path=marker,
        name="bootstrap-no-candidate",
        description="Non-mountable donor bootstrap marker.",
        sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )


def _build_donor_workspace_pair(
    *, capture_cwd: Path, pair_id: str, prompt: str
) -> TrialWorkspacePair:
    """Build a baseline-only donor workspace without any real candidate."""

    capture = capture_cwd.absolute()
    capture.mkdir(parents=True, exist_ok=True)
    capture = capture.resolve(strict=True)
    pair_root = capture / "trialqa-local" / pair_id
    if pair_root.exists() or pair_root.is_symlink():
        raise TrialQADemoError(f"donor workspace collision: {pair_root}")
    pair_root.parent.mkdir(parents=True, exist_ok=True)
    stage = pair_root.parent / f".{pair_id}.donor-{os.getpid()}-{threading.get_ident()}.tmp"
    if stage.exists() or stage.is_symlink():
        raise TrialQADemoError(f"donor staging collision: {stage}")
    try:
        for arm_name in ("baseline", "treatment"):
            root = stage / "arms" / arm_name
            (root / ".agents" / "skills").mkdir(parents=True)
            (root / "outputs").mkdir()
            (root / "prompt.md").write_text(prompt, encoding="utf-8")
            (root / ".gitignore").write_text(
                "/answer.txt\n/outputs/\n/.switchyard-trialqa/\n", encoding="utf-8"
            )
            _initialize_inner_git(root)
        runtime = stage / "runtime"
        for arm_name in ("baseline", "treatment"):
            (runtime / arm_name / "home").mkdir(parents=True)
            (runtime / arm_name / "codex-home").mkdir()
        config = runtime / "switchyard-config" / "config.json"
        _write_json_atomic(config, {"skill_distillation": {"namespace": NAMESPACE}})
        bootstrap = _bootstrap_candidate(runtime)
        stage.rename(pair_root)
    except BaseException:
        if stage.is_dir() and not stage.is_symlink():
            import shutil

            shutil.rmtree(stage)
        raise
    runtime_root = pair_root / "runtime"
    bootstrap = _bootstrap_candidate(runtime_root)
    baseline_root = pair_root / "arms" / "baseline"
    treatment_root = pair_root / "arms" / "treatment"
    return TrialWorkspacePair(
        task_id=pair_id,
        capture_cwd=capture,
        pair_root=pair_root,
        runtime_root=runtime_root,
        switchyard_config_dir=runtime_root / "switchyard-config",
        baseline=_arm(baseline_root, "baseline", None),
        treatment=_arm(treatment_root, "treatment", None),
        candidate=bootstrap,
        prompt_sha256=_sha256_bytes(prompt.encode("utf-8")),
    )


def _load_existing_donor_pair(
    *, capture_cwd: Path, pair_id: str, prompt: str
) -> TrialWorkspacePair:
    capture = capture_cwd.resolve(strict=True)
    pair_root = capture / "trialqa-local" / pair_id
    runtime_root = pair_root / "runtime"
    baseline_root = pair_root / "arms" / "baseline"
    treatment_root = pair_root / "arms" / "treatment"
    for root in (baseline_root, treatment_root):
        if root.is_symlink() or not root.is_dir() or not (root / ".git").is_dir():
            raise TrialQADemoError(f"existing donor arm is unsafe: {root}")
        if list((root / ".agents" / "skills").iterdir()):
            raise TrialQADemoError("donor arm unexpectedly exposes a project skill")
        if (root / "prompt.md").read_text(encoding="utf-8") != prompt:
            raise TrialQADemoError("existing donor prompt differs from the manifest task")
    bootstrap = _bootstrap_candidate(runtime_root)
    return TrialWorkspacePair(
        task_id=pair_id,
        capture_cwd=capture,
        pair_root=pair_root,
        runtime_root=runtime_root,
        switchyard_config_dir=runtime_root / "switchyard-config",
        baseline=_arm(baseline_root, "baseline", None),
        treatment=_arm(treatment_root, "treatment", None),
        candidate=bootstrap,
        prompt_sha256=_sha256_bytes(prompt.encode("utf-8")),
    )


def _load_existing_pair(
    *,
    capture_cwd: Path,
    pair_id: str,
    prompt: str,
    candidate_root: Path,
) -> TrialWorkspacePair:
    candidate = validate_candidate_skill(candidate_root, NAMESPACE)
    capture = capture_cwd.resolve(strict=True)
    pair_root = capture / "trialqa-local" / pair_id
    baseline_root = pair_root / "arms" / "baseline"
    treatment_root = pair_root / "arms" / "treatment"
    managed = treatment_root / ".agents" / "skills" / candidate.name
    for root in (baseline_root, treatment_root):
        if root.is_symlink() or not root.is_dir() or not (root / ".git").is_dir():
            raise TrialQADemoError(f"existing trial arm is unsafe: {root}")
    if list((baseline_root / ".agents" / "skills").iterdir()):
        raise TrialQADemoError("existing baseline unexpectedly exposes a skill")
    if not managed.is_symlink() or managed.resolve(strict=True) != candidate.skill_dir:
        raise TrialQADemoError("existing treatment skill link differs from candidate")
    treatment_entries = list((treatment_root / ".agents" / "skills").iterdir())
    if treatment_entries != [managed]:
        raise TrialQADemoError("existing treatment has unexpected project skills")
    expected_prompt = prompt.encode("utf-8")
    for root in (baseline_root, treatment_root):
        if (root / "prompt.md").read_bytes() != expected_prompt:
            raise TrialQADemoError("existing pair prompt differs from manifest task")
    return TrialWorkspacePair(
        task_id=pair_id,
        capture_cwd=capture,
        pair_root=pair_root,
        runtime_root=pair_root / "runtime",
        switchyard_config_dir=pair_root / "runtime" / "switchyard-config",
        baseline=_arm(baseline_root, "baseline", None),
        treatment=_arm(treatment_root, "treatment", managed),
        candidate=candidate,
        prompt_sha256=_sha256_bytes(expected_prompt),
    )


def prepare_generation(
    *,
    manifest: Mapping[str, object],
    task_id: str,
    dataset: TrialQADataset,
    split_manifest: Mapping[str, object],
    capture_cwd: Path,
    candidate_root: Path | None,
    switchyard_bin: Path,
    codex_bin: Path,
    routing_profile: Path,
    tooluniverse_bin: Path,
) -> PlannedGeneration:
    """Prepare or safely reopen one A/B pair and return its reviewed RunSpec."""

    validate_manifest_pairing(manifest)
    task = _manifest_task_by_id(manifest, task_id)
    row = dataset.row_by_id(cast(str, task["row_id"]))
    if row.dataset_row_index != task.get("dataset_row_index"):
        raise TrialQADemoError("manifest row identity differs from pinned dataset")
    prompt = render_trial_prompt(row)
    pair_id = _safe_component(task.get("pair_id"), "pair id")
    pair_path = capture_cwd.absolute() / "trialqa-local" / pair_id
    is_donor = task.get("condition") == "donor"
    if is_donor:
        if candidate_root is not None:
            raise TrialQADemoError("donor generation must not be given a candidate root")
        if pair_path.exists() or pair_path.is_symlink():
            pair = _load_existing_donor_pair(
                capture_cwd=capture_cwd,
                pair_id=pair_id,
                prompt=prompt,
            )
        else:
            pair = _build_donor_workspace_pair(
                capture_cwd=capture_cwd,
                pair_id=pair_id,
                prompt=prompt,
            )
    else:
        if candidate_root is None:
            raise TrialQADemoError("evaluation generation requires a candidate root")
        if pair_path.exists() or pair_path.is_symlink():
            pair = _load_existing_pair(
                capture_cwd=capture_cwd,
                pair_id=pair_id,
                prompt=prompt,
                candidate_root=candidate_root,
            )
        else:
            pair = build_trial_workspace_pair(
                capture_cwd=capture_cwd.absolute(),
                task_id=pair_id,
                prompt=prompt,
                candidate_root=candidate_root.absolute(),
                candidate_skill_directory=NAMESPACE,
            )
    assert_trial_inputs_gold_free(
        dataset=dataset,
        split_manifest=split_manifest,
        pair=pair,
        row=row,
        partition=cast(str, task["partition"]),
    )
    arm_name = cast(Literal["baseline", "treatment"], task["arm"])
    spec = build_run_spec(
        pair=pair,
        arm_name=arm_name,
        switchyard_bin=switchyard_bin.absolute(),
        codex_bin=codex_bin.absolute(),
        routing_profile=routing_profile.absolute(),
        tooluniverse_bin=tooluniverse_bin.absolute(),
    )
    return PlannedGeneration(task=task, row=row, pair=pair, spec=spec)


def _copy_stream(source: Any, destination: Any) -> None:
    try:
        while chunk := source.read(64 * 1024):
            destination.write(chunk)
            destination.flush()
    finally:
        source.close()


def _terminate_timed_out_process(
    process: subprocess.Popen[bytes],
    threads: Sequence[threading.Thread],
) -> None:
    """Terminate the isolated launch group and prove its capture pipes drained."""

    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - Windows fallback.
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    # The group may retain descendants after its leader exits on SIGTERM. Send
    # SIGKILL unconditionally so inherited stdout/stderr pipes cannot keep the
    # capture threads alive after the deadline.
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows fallback.
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as exc:
        raise TrialQADemoError(
            "timed-out Switchyard/Codex process group did not exit after SIGKILL"
        ) from exc

    for thread in threads:
        thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads):
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        for thread in threads:
            thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads):
        raise TrialQADemoError("timed-out Switchyard/Codex capture streams did not terminate")


def _raise_generation_timeout(
    process: subprocess.Popen[bytes],
    threads: Sequence[threading.Thread],
    *,
    timeout_seconds: float,
    cause: BaseException,
) -> NoReturn:
    try:
        _terminate_timed_out_process(process, threads)
    except Exception as cleanup_error:
        raise GenerationTimeoutError(
            timeout_seconds,
            process_group_terminated=False,
        ) from cleanup_error
    raise GenerationTimeoutError(timeout_seconds) from cause


def run_streaming_subprocess(
    spec: RunSpec,
    extra_environment: Mapping[str, str],
    *,
    timeout_seconds: float = 1800.0,
) -> int:
    """Run a reviewed spec while concurrently streaming both pipes to disk."""

    environment = dict(os.environ)
    environment.update(spec.env)
    environment.update(extra_environment)
    spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        spec.stdin_path.open("rb") as stdin,
        spec.stdout_path.open("xb") as stdout,
        spec.stderr_path.open("xb") as stderr,
    ):
        process = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=environment,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=(os.name == "posix"),
        )
        if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract.
            process.kill()
            raise TrialQADemoError("subprocess pipes were not created")
        threads = (
            threading.Thread(target=_copy_stream, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=_copy_stream, args=(process.stderr, stderr), daemon=True),
        )
        for thread in threads:
            thread.start()
        wait_started = time.monotonic()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _raise_generation_timeout(
                process,
                threads,
                timeout_seconds=timeout_seconds,
                cause=exc,
            )
        for thread in threads:
            remaining = max(
                0.0,
                timeout_seconds - (time.monotonic() - wait_started),
            )
            thread.join(timeout=remaining)
        if any(thread.is_alive() for thread in threads):
            _raise_generation_timeout(
                process,
                threads,
                timeout_seconds=timeout_seconds,
                cause=TimeoutError("Switchyard/Codex descendants retained capture streams"),
            )
    return return_code


def _validate_codex_tool_events(
    events: Sequence[Mapping[str, object]],
    *,
    require_skill_load: bool,
) -> None:
    successful_evidence_calls = 0
    successful_skill_loads_before_tool_use = 0
    saw_non_loader_tool = False
    for event in events:
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "todo_list":
            if set(item) != {"id", "type", "items"} or not isinstance(item.get("id"), str):
                raise TrialQADemoError("Codex emitted an invalid passive todo list")
            todo_items = item.get("items")
            if not isinstance(todo_items, list) or any(
                not isinstance(todo, dict)
                or set(todo) != {"text", "completed"}
                or not isinstance(todo.get("text"), str)
                or not todo["text"].strip()
                or not isinstance(todo.get("completed"), bool)
                for todo in todo_items
            ):
                raise TrialQADemoError("Codex emitted an invalid passive todo list")
            continue
        if item_type not in {"agent_message", "reasoning", "mcp_tool_call"}:
            raise TrialQADemoError(f"Codex emitted a forbidden or unknown item type: {item_type!r}")
        if item_type != "mcp_tool_call":
            continue
        if item.get("server") != "tooluniverse" or item.get("tool") not in TRIALQA_MCP_TOOLS:
            raise TrialQADemoError("Codex used an MCP tool outside the TrialQA adapter")
        if item.get("tool") != "trialqa_load_active_skill":
            saw_non_loader_tool = True
        if event.get("type") == "item.completed":
            # A failed call to an allowlisted read-only adapter tool is model
            # telemetry, not an infrastructure failure.  Count only successful
            # calls as evidence and let recovered mistakes affect tokens/quality.
            if item.get("status") != "completed" or item.get("error") is not None:
                continue
            if item.get("tool") == "trialqa_load_active_skill":
                if not saw_non_loader_tool:
                    successful_skill_loads_before_tool_use += 1
            elif item.get("tool") == TRIALQA_EVIDENCE_TOOL:
                successful_evidence_calls += 1
    if successful_evidence_calls == 0:
        raise TrialQADemoError("Codex completed without a TrialQA evidence-tool call")
    if require_skill_load and successful_skill_loads_before_tool_use < 1:
        raise TrialQADemoError(
            "treatment must successfully load its active TrialQA skill before tool use"
        )


def read_codex_events(path: Path) -> tuple[JsonObject, ...]:
    """Read the JSON objects from Codex's mixed human/JSON launch log."""

    if path.is_symlink() or not path.is_file():
        raise TrialQADemoError("Codex JSON event stream is missing")
    events: list[JsonObject] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            # ``switchyard launch`` writes its human banner to the same stream
            # as Codex JSON events.  Preserve the complete mixed log, but only
            # interpret lines that are standalone JSON objects.
            if not line.lstrip().startswith("{"):
                continue
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                raise TrialQADemoError(f"Codex event {line_number} is not an object")
            events.append(cast(JsonObject, value))
    except (OSError, UnicodeDecodeError) as exc:
        raise TrialQADemoError("Codex mixed launch log is not readable UTF-8") from exc
    return tuple(events)


def codex_tool_metrics(path: Path) -> JsonObject:
    """Count reference-aligned TrialQA calls from unique Codex item IDs.

    The published skill-distillation workflow treats environment/tool calls as
    the operational cost.  Skill loading is setup overhead, so report it
    separately rather than counting it as a TrialQA evidence operation.
    """

    events = read_codex_events(path)
    started: dict[str, tuple[str, str]] = {}
    successful: set[str] = set()
    for event in events:
        if event.get("type") not in {"item.started", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "mcp_tool_call":
            continue
        item_id = item.get("id")
        server = item.get("server")
        tool = item.get("tool")
        if (
            not isinstance(item_id, str)
            or not item_id
            or server != "tooluniverse"
            or tool not in TRIALQA_MCP_TOOLS
        ):
            raise TrialQADemoError("Codex tool metrics found an invalid TrialQA call")
        identity = (cast(str, server), cast(str, tool))
        if event.get("type") == "item.started":
            previous = started.setdefault(item_id, identity)
            if previous != identity:
                raise TrialQADemoError("Codex reused a tool-call ID for another operation")
            continue
        if item_id not in started:
            raise TrialQADemoError("Codex completed a TrialQA call without starting it")
        if started[item_id] != identity:
            raise TrialQADemoError("Codex tool-call identity changed before completion")
        if item.get("status") == "completed" and item.get("error") is None:
            successful.add(item_id)

    skill_ids = {
        item_id
        for item_id, (_server, tool) in started.items()
        if tool == "trialqa_load_active_skill"
    }
    operational_ids = set(started) - skill_ids
    return {
        "operational_calls": len(operational_ids),
        "successful_operational_calls": len(operational_ids & successful),
        "skill_load_calls": len(skill_ids),
        "successful_skill_load_calls": len(skill_ids & successful),
    }


def _parse_codex_events(
    path: Path,
    *,
    require_skill_load: bool = False,
    enforce_tool_policy: bool = True,
) -> JsonObject:
    events = read_codex_events(path)
    types = [event.get("type") for event in events]
    terminal = [event_type for event_type in types if event_type in _TERMINAL_CODEX_EVENT_TYPES]
    if terminal:
        raise TrialQADemoError(f"Codex event stream reports failure: {terminal[0]}")
    if "thread.started" not in types or "turn.completed" not in types:
        raise TrialQADemoError("Codex event stream lacks successful lifecycle events")
    completed = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == "turn.completed"
    ]
    if len(completed) != 1:
        raise TrialQADemoError("Codex run must contain exactly one completed turn")
    completed_index, completed_event = completed[0]
    for index, event in enumerate(events):
        if event.get("type") != "error":
            continue
        message = event.get("message")
        if index >= completed_index or not isinstance(message, str) or not message.strip():
            raise TrialQADemoError(
                "Codex error telemetry must be nonempty and precede turn.completed"
            )
    if enforce_tool_policy:
        _validate_codex_tool_events(events, require_skill_load=require_skill_load)
    usage = completed_event.get("usage", {})
    if not isinstance(usage, dict):
        raise TrialQADemoError("Codex completed event has invalid usage")
    return cast(JsonObject, usage)


def completed_model_draw_usage(path: Path) -> JsonObject | None:
    """Return usage only when Codex attests one successfully completed turn."""

    try:
        return _parse_codex_events(path, enforce_tool_policy=False)
    except TrialQADemoError:
        return None


def _parse_final_answer_with_source(path: Path) -> tuple[str, str]:
    """Read Codex's authoritative last message without imposing an envelope task.

    Sergei's TrialQA verifier scores the answer text written by the agent (with a
    trajectory fallback); JSON is not part of the benchmark.  We still unwrap
    the requested ``{"answer": ...}`` form when it is exact, but otherwise score
    the nonempty final assistant text verbatim instead of retrying until a model
    happens to serialize the preferred envelope.
    """

    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise TrialQADemoError(f"Codex final output must be a real file: {path}")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise TrialQADemoError(f"Codex final output is not readable UTF-8: {path}") from exc
    if not raw:
        raise TrialQADemoError("Codex final answer is empty")
    try:
        document: object = json.loads(raw)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and set(document) == {"answer"}:
        answer = document.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip(), FINAL_ANSWER_JSON_SOURCE
    return raw, FINAL_ANSWER_TEXT_SOURCE


def _parse_final_answer(path: Path) -> str:
    return _parse_final_answer_with_source(path)[0]


_OPENAI_TRANSPORT_FIELDS = {
    "physical_attempts",
    "null_eof_retries",
    "retry_usage_charges",
    "unpriced_null_eof_retries",
    "retry_token_sensitivity",
}
_TOKEN_TOTAL_FIELDS = {
    "prompt",
    "completion",
    "cached",
    "cache_creation",
    "reasoning",
    "total",
}


def _validate_openai_transport_stats(
    stats: Mapping[str, object],
    *,
    total_requests: int,
    require_priced: bool,
) -> JsonObject:
    """Validate and normalize physical retry accounting for one session."""

    transport = stats.get("openai_transport")
    if not isinstance(transport, dict) or set(transport) != _OPENAI_TRANSPORT_FIELDS:
        raise TrialQADemoError("Switchyard stats lack exact OpenAI transport accounting")
    counters: dict[str, int] = {}
    for field in _OPENAI_TRANSPORT_FIELDS - {"retry_token_sensitivity"}:
        value = transport.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrialQADemoError(f"Switchyard OpenAI transport {field} is invalid")
        counters[field] = value
    sensitivity = transport.get("retry_token_sensitivity")
    if not isinstance(sensitivity, dict) or set(sensitivity) != _TOKEN_TOTAL_FIELDS:
        raise TrialQADemoError("Switchyard retry token sensitivity is invalid")
    normalized_tokens: dict[str, int] = {}
    for field in _TOKEN_TOTAL_FIELDS:
        value = sensitivity.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrialQADemoError(f"Switchyard retry token sensitivity {field} is invalid")
        normalized_tokens[field] = value
    if normalized_tokens["total"] != (
        normalized_tokens["prompt"] + normalized_tokens["completion"]
    ):
        raise TrialQADemoError("Switchyard retry token sensitivity total is inconsistent")

    physical_attempts = counters["physical_attempts"]
    retries = counters["null_eof_retries"]
    charges = counters["retry_usage_charges"]
    unpriced = counters["unpriced_null_eof_retries"]
    if (
        physical_attempts != total_requests + retries
        or retries > total_requests
        or charges + unpriced != retries
        or (charges == 0 and any(normalized_tokens.values()))
        or (charges > 0 and normalized_tokens["total"] == 0)
    ):
        raise TrialQADemoError("Switchyard OpenAI transport accounting is inconsistent")
    if require_priced and unpriced:
        raise TrialQADemoError("Switchyard session contains an unpriced null-EOF retry")
    return {
        **counters,
        "retry_token_sensitivity": normalized_tokens,
    }


def _validate_executor_stats(stats: JsonObject) -> None:
    total_requests = stats.get("total_requests")
    total_errors = stats.get("total_errors")
    if (
        not isinstance(total_requests, int)
        or isinstance(total_requests, bool)
        or total_requests < 1
    ):
        raise TrialQADemoError("Switchyard stats contain no executor requests")
    if (
        not isinstance(total_errors, int)
        or isinstance(total_errors, bool)
        or total_errors < 0
        or total_errors >= total_requests
    ):
        raise TrialQADemoError("Switchyard stats contain invalid executor error counts")
    for subsystem_name in ("classifier", "planner"):
        subsystem = stats.get(subsystem_name)
        if (
            not isinstance(subsystem, dict)
            or subsystem.get("total_requests") != 0
            or isinstance(subsystem.get("total_requests"), bool)
            or subsystem.get("total_errors") != 0
            or isinstance(subsystem.get("total_errors"), bool)
        ):
            raise TrialQADemoError(f"Switchyard {subsystem_name} recorded non-executor activity")
    models = stats.get("models")
    if not isinstance(models, dict):
        raise TrialQADemoError("Switchyard stats lack per-model attribution")
    attempts_by_model: dict[str, int] = {}
    successful_calls = 0
    model_errors = 0
    for model, value in models.items():
        if not isinstance(model, str) or not isinstance(value, dict):
            raise TrialQADemoError("Switchyard stats model entries are invalid")
        calls = value.get("calls", 0)
        errors = value.get("errors", 0)
        if (
            not isinstance(calls, int)
            or isinstance(calls, bool)
            or calls < 0
            or not isinstance(errors, int)
            or isinstance(errors, bool)
            or errors < 0
        ):
            raise TrialQADemoError("Switchyard stats contain invalid model attempt counts")
        attempts = calls + errors
        if attempts:
            attempts_by_model[model] = attempts
        successful_calls += calls
        model_errors += errors
    if (
        attempts_by_model != {EXECUTOR_MODEL: total_requests}
        or successful_calls < 1
        or model_errors != total_errors
    ):
        raise TrialQADemoError(
            "executor attempts were not exclusively attributed to "
            f"{EXECUTOR_MODEL}: {attempts_by_model}"
        )
    _validate_openai_transport_stats(
        stats,
        total_requests=total_requests,
        require_priced=True,
    )


def _validate_executor_usage(stats: Mapping[str, object], usage: Mapping[str, object]) -> None:
    tokens = stats.get("total_tokens")
    if (
        not isinstance(tokens, dict)
        or usage.get("input_tokens") != tokens.get("prompt")
        or usage.get("output_tokens") != tokens.get("completion")
    ):
        raise TrialQADemoError("Codex usage differs from Switchyard executor stats")


def _session_for_context(
    *,
    store: SkillDistillationStore,
    previous: set[str],
    run_context: Mapping[str, object],
    active_evidence: Mapping[str, object],
) -> Path:
    matches: list[Path] = []
    for session_dir in store.sessions_path.iterdir():
        if session_dir.name in previous or session_dir.is_symlink() or not session_dir.is_dir():
            continue
        session_path = session_dir / "session.json"
        try:
            document = _read_json_object(session_path, "Switchyard session")
        except TrialQADemoError:
            continue
        if document.get("run_context") == dict(run_context):
            if document.get("active_skill") != dict(active_evidence):
                raise TrialQADemoError(
                    "captured active-skill evidence differs from launch attestation"
                )
            matches.append(session_dir)
    if len(matches) != 1:
        raise TrialQADemoError(
            f"generation produced {len(matches)} matching Switchyard sessions, expected one"
        )
    store.validate_session_evidence(matches[0])
    return matches[0]


RunExecutor = Callable[[RunSpec, Mapping[str, str]], int]


def execute_generation(
    *,
    manifest: Mapping[str, object],
    planned: PlannedGeneration,
    dataset: TrialQADataset,
    executor: RunExecutor = run_streaming_subprocess,
) -> GenerationResult:
    """Execute one generation arm and validate every output before returning."""

    manifest_id = _safe_component(manifest.get("manifest_id"), "manifest id")
    task = planned.task
    task_id = _safe_component(task.get("task_id"), "task id")
    arm = planned.pair.baseline if planned.spec.arm == "baseline" else planned.pair.treatment
    generation_path = arm.root / "outputs" / "generation.json"
    collisions = [
        path
        for path in (
            planned.spec.stdout_path,
            planned.spec.stderr_path,
            planned.spec.final_output_path,
            planned.spec.answer_path,
            generation_path,
        )
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise TrialQADemoError(f"generation output collision: {collisions[0]}")

    candidate_proof = (
        _candidate_attestation(planned.pair.candidate) if planned.spec.arm == "treatment" else None
    )
    run_context: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "task_id": task_id,
        "pair_id": task["pair_id"],
        "row_id": task["row_id"],
        "dataset_row_index": task["dataset_row_index"],
        "question_group_key": task["question_group_key"],
        "partition": task["partition"],
        "phase": task["phase"],
        "condition": task["condition"],
        "repeat_index": task["repeat_index"],
        "n_repeats": task["n_repeats"],
        "executor_model": EXECUTOR_MODEL,
        "route": EXECUTOR_ROUTE,
        "skill_loaded": planned.spec.arm == "treatment",
        "candidate_id": candidate_proof.candidate_id if candidate_proof else None,
        "candidate_manifest_sha256": (candidate_proof.manifest_sha256 if candidate_proof else None),
        "candidate_skill_sha256": candidate_proof.skill_sha256 if candidate_proof else None,
    }
    if planned.spec.arm == "treatment":
        assert candidate_proof is not None
        active_evidence: JsonObject = {
            "loaded": True,
            "candidate_id": candidate_proof.candidate_id,
            "manifest_sha256": candidate_proof.manifest_sha256,
            "skill_sha256": candidate_proof.skill_sha256,
            "path": str(planned.pair.candidate.skill_path),
        }
    else:
        active_evidence = {
            "loaded": False,
            "candidate_id": None,
            "manifest_sha256": None,
            "skill_sha256": None,
            "path": None,
        }
    # Launcher metadata must be inside the central Switchyard project (the
    # launch cwd) but outside Codex's model-visible ``-C`` trial repository.
    metadata_dir = planned.pair.runtime_root / arm.name / "launch-metadata"
    context_path = metadata_dir / "run-context.json"
    active_path = metadata_dir / "active-evidence.json"
    _write_json_atomic(context_path, run_context, exclusive=True)
    _write_json_atomic(active_path, active_evidence, exclusive=True)

    store = SkillDistillationStore(NAMESPACE, planned.spec.cwd)
    previous_sessions = {path.name for path in store.sessions_path.iterdir()}
    extra_environment = {
        "SWITCHYARD_SKILL_DISTILLATION_RUN_CONTEXT_PATH": str(context_path),
        "SWITCHYARD_SKILL_DISTILLATION_ACTIVE_EVIDENCE_PATH": str(active_path),
    }
    assert_unique_gold_artifact(
        dataset.path,
        dataset.path.parents[2],
        expected_sha256=dataset.parquet_sha256,
    )
    before_candidate = (
        _candidate_attestation(planned.pair.candidate) if planned.spec.arm == "treatment" else None
    )
    with GoldArtifactLockdown(dataset.path, expected_sha256=dataset.parquet_sha256):
        return_code = executor(planned.spec, extra_environment)
    if (
        before_candidate is not None
        and _candidate_attestation(planned.pair.candidate) != before_candidate
    ):
        raise TrialQADemoError("treatment candidate changed during generation")
    if return_code != 0:
        raise TrialQADemoError(f"Switchyard/Codex generation exited with {return_code}")

    usage = _parse_codex_events(
        planned.spec.stdout_path,
        require_skill_load=planned.spec.arm == "treatment",
    )
    answer, answer_source = _parse_final_answer_with_source(planned.spec.final_output_path)
    planned.spec.answer_path.write_text(answer + "\n", encoding="utf-8")
    session_dir = _session_for_context(
        store=store,
        previous=previous_sessions,
        run_context=run_context,
        active_evidence=active_evidence,
    )
    stats_path = session_dir / "stats.json"
    trajectory_path = session_dir / "turns.jsonl"
    stats_document = _read_json_object(stats_path, "Switchyard session stats")
    _validate_executor_stats(stats_document)
    _validate_executor_usage(stats_document, usage)
    artifact_hashes = {
        "answer": _sha256_file(planned.spec.answer_path),
        "codex_events": _sha256_file(planned.spec.stdout_path),
        "final_output": _sha256_file(planned.spec.final_output_path),
        "stats": _sha256_file(stats_path),
        "trajectory": _sha256_file(trajectory_path),
    }
    result = GenerationResult(
        manifest_id=manifest_id,
        task_id=task_id,
        pair_id=cast(str, task["pair_id"]),
        row_id=planned.row.id,
        dataset_row_index=planned.row.dataset_row_index,
        partition=cast(str, task["partition"]),
        condition=cast(str, task["condition"]),
        repeat_index=cast(int, task["repeat_index"]),
        n_repeats=cast(int, task["n_repeats"]),
        answer=answer,
        answer_source=answer_source,
        session_dir=session_dir,
        stats_path=stats_path,
        trajectory_path=trajectory_path,
        codex_events_path=planned.spec.stdout_path,
        final_output_path=planned.spec.final_output_path,
        generation_path=generation_path,
        stats=stats_document,
        usage=usage,
        artifact_sha256=artifact_hashes,
    )
    _write_json_atomic(generation_path, result.json_document(), exclusive=True)
    return result


def validate_generation_for_import(
    generation: GenerationResult,
    *,
    project_dir: Path,
) -> None:
    """Re-attest immutable outputs, capture identity, model, and stats before import."""

    expected_paths = {
        "answer": generation.final_output_path.parent.parent / "answer.txt",
        "codex_events": generation.codex_events_path,
        "final_output": generation.final_output_path,
        "stats": generation.stats_path,
        "trajectory": generation.trajectory_path,
    }
    if set(generation.artifact_sha256) != set(expected_paths):
        raise TrialQADemoError("generation artifact hash set is incomplete")
    for name, path in expected_paths.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise TrialQADemoError(f"generation artifact is unsafe before import: {name}")
        if _sha256_file(path) != generation.artifact_sha256[name]:
            raise TrialQADemoError(f"generation artifact changed before import: {name}")

    captured_answer, captured_answer_source = _parse_final_answer_with_source(
        generation.final_output_path
    )
    try:
        saved_answer = expected_paths["answer"].read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TrialQADemoError("saved answer is not readable UTF-8") from exc
    if (
        generation.answer_source != captured_answer_source
        or generation.answer != captured_answer
        or saved_answer != captured_answer + "\n"
    ):
        raise TrialQADemoError("generation answer differs from captured Codex output")
    captured_usage = _parse_codex_events(
        generation.codex_events_path,
        enforce_tool_policy=False,
    )
    if generation.usage != captured_usage:
        raise TrialQADemoError("generation usage differs from captured Codex events")

    store = SkillDistillationStore(NAMESPACE, project_dir)
    store.validate_session_evidence(generation.session_dir)
    session = _read_json_object(generation.session_dir / "session.json", "Switchyard session")
    context = session.get("run_context")
    if not isinstance(context, dict):
        raise TrialQADemoError("Switchyard session lost its run context")
    expected_context = {
        "manifest_id": generation.manifest_id,
        "task_id": generation.task_id,
        "pair_id": generation.pair_id,
        "row_id": generation.row_id,
        "dataset_row_index": generation.dataset_row_index,
        "question_group_key": generation.pair_id.rsplit("-r", maxsplit=1)[0],
        "partition": generation.partition,
        "condition": generation.condition,
        "repeat_index": generation.repeat_index,
        "n_repeats": generation.n_repeats,
        "executor_model": EXECUTOR_MODEL,
        "route": EXECUTOR_ROUTE,
        "skill_loaded": generation.condition == "treatment",
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise TrialQADemoError(f"Switchyard session run context differs at {key}")
    active = session.get("active_skill")
    if not isinstance(active, dict) or active.get("loaded") is not (
        generation.condition == "treatment"
    ):
        raise TrialQADemoError("Switchyard session has the wrong active-skill condition")
    if context.get("candidate_id") != active.get("candidate_id"):
        raise TrialQADemoError("run context and active evidence disagree on candidate ID")
    if context.get("candidate_manifest_sha256") != active.get("manifest_sha256"):
        raise TrialQADemoError("run context and active evidence disagree on candidate manifest")
    if context.get("candidate_skill_sha256") != active.get("skill_sha256"):
        raise TrialQADemoError("run context and active evidence disagree on candidate skill hash")
    stats = _read_json_object(generation.stats_path, "Switchyard session stats")
    _validate_executor_stats(stats)
    _validate_executor_usage(stats, generation.usage)
    if generation.stats != stats:
        raise TrialQADemoError("generation stats differ from captured Switchyard stats")

    try:
        lines = generation.trajectory_path.read_text(encoding="utf-8").splitlines()
        turns = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialQADemoError("Switchyard trajectory is invalid before import") from exc
    if not turns or not all(isinstance(turn, dict) for turn in turns):
        raise TrialQADemoError("Switchyard trajectory contains no turns")
    served_models = {cast(dict[str, object], turn).get("served_model") for turn in turns}
    accepted_models = {EXECUTOR_MODEL, "nvidia/nemotron-3-ultra"}
    if not served_models or not served_models.issubset(accepted_models):
        raise TrialQADemoError(
            f"Switchyard trajectory contains non-Ultra served models: {served_models}"
        )


def load_generation_result(path: Path) -> GenerationResult:
    """Load a persisted generation record without trusting its referenced files."""

    document = _read_json_object(path, "generation record")
    if document.pop("schema_version", None) != GENERATION_SCHEMA_VERSION:
        raise TrialQADemoError("generation record has the wrong schema")
    path_fields = (
        "session_dir",
        "stats_path",
        "trajectory_path",
        "codex_events_path",
        "final_output_path",
        "generation_path",
    )
    for field in path_fields:
        value = document.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise TrialQADemoError(f"generation record has invalid path field {field}")
        document[field] = Path(value)
    if cast(Path, document["generation_path"]).resolve(strict=True) != path.resolve(strict=True):
        raise TrialQADemoError("generation record path does not match its own identity")
    for field in (
        "manifest_id",
        "task_id",
        "pair_id",
        "row_id",
        "partition",
        "condition",
        "answer",
        "answer_source",
    ):
        if not isinstance(document.get(field), str):
            raise TrialQADemoError(f"generation record has invalid field {field}")
    for field in ("dataset_row_index", "repeat_index", "n_repeats"):
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TrialQADemoError(f"generation record has invalid field {field}")
    for field in ("stats", "usage", "artifact_sha256"):
        if not isinstance(document.get(field), dict):
            raise TrialQADemoError(f"generation record has invalid field {field}")
    try:
        return GenerationResult(**document)
    except TypeError as exc:
        raise TrialQADemoError("generation record fields do not match its schema") from exc


def _stdlib_http_transport(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
    timeout: float,
) -> tuple[int, bytes]:
    data = _canonical_json(payload) if payload is not None else None
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-switchyard",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TrialQAJudgeError(f"local judge HTTP request failed: {url}") from exc


def _decode_http_json(status: int, body: bytes, label: str) -> JsonObject:
    if status != 200:
        detail = body.decode("utf-8", errors="replace")[:500]
        raise TrialQAJudgeError(f"{label} returned HTTP {status}: {detail}")
    try:
        value: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialQAJudgeError(f"{label} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrialQAJudgeError(f"{label} response must be a JSON object")
    if value.get("error") is not None:
        raise TrialQAJudgeError(f"{label} response contains an error")
    return cast(JsonObject, value)


def _model_calls(stats: Mapping[str, object]) -> dict[str, int]:
    models = stats.get("models", {})
    if not isinstance(models, dict):
        raise TrialQAJudgeError("judge stats lack model attribution")
    result: dict[str, int] = {}
    for model, raw in models.items():
        if not isinstance(model, str) or not isinstance(raw, dict):
            raise TrialQAJudgeError("judge stats contain invalid model entries")
        calls = raw.get("calls", 0)
        if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
            raise TrialQAJudgeError("judge stats contain invalid call counts")
        result[model] = calls
    return result


class DedicatedJudgeClient:
    """Strict callback for :func:`score_semantic_answer` using one local server."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: HttpTransport = _stdlib_http_transport,
        timeout: float = 3600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.timeout = timeout
        self.call_count = 0

    def stats(self) -> JsonObject:
        status, body = self.transport(
            "GET", f"{self.base_url}/v1/routing/stats", None, min(self.timeout, 30.0)
        )
        return _decode_http_json(status, body, "judge routing stats")

    def __call__(self, payload: Mapping[str, object]) -> str:
        if payload.get("model") != JUDGE_ROUTE:
            raise TrialQAJudgeError(f"judge request must use route {JUDGE_ROUTE!r}")
        before = self.stats()
        status, body = self.transport(
            "POST", f"{self.base_url}/v1/chat/completions", payload, self.timeout
        )
        response = _decode_http_json(status, body, "semantic judge")
        after = self.stats()
        before_requests = before.get("total_requests", 0)
        after_requests = after.get("total_requests")
        before_errors = before.get("total_errors", 0)
        after_errors = after.get("total_errors")
        if not isinstance(before_requests, int) or not isinstance(after_requests, int):
            raise TrialQAJudgeError("judge stats request totals are invalid")
        if not isinstance(before_errors, int) or not isinstance(after_errors, int):
            raise TrialQAJudgeError("judge stats error totals are invalid")
        if after_requests - before_requests != 1 or after_errors - before_errors != 0:
            raise TrialQAJudgeError("judge request did not produce one error-free routed call")
        before_calls = _model_calls(before)
        after_calls = _model_calls(after)
        deltas = {
            model: after_calls.get(model, 0) - before_calls.get(model, 0)
            for model in set(before_calls) | set(after_calls)
        }
        positive = {model: count for model, count in deltas.items() if count}
        if positive != {JUDGE_MODEL: 1}:
            raise TrialQAJudgeError(
                f"judge call was not exclusively routed to {JUDGE_MODEL}: {positive}"
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TrialQAJudgeError("judge response must contain exactly one choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TrialQAJudgeError("judge response choice has no text content")
        self.call_count += 1
        return cast(str, message["content"])


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


def build_judge_process_spec(
    *,
    switchyard_bin: Path,
    routing_profile: Path,
    runtime_root: Path,
    port: int | None = None,
) -> JudgeProcessSpec:
    validate_routing_profile(routing_profile)
    _validate_judge_route(routing_profile)
    switchyard = switchyard_bin.resolve(strict=True)
    if not switchyard.is_file() or not os.access(switchyard, os.X_OK):
        raise TrialQADemoError("Switchyard judge binary is not executable")
    selected_port = port or _free_local_port()
    if not 1 <= selected_port <= 65535:
        raise TrialQADemoError("judge port is outside the valid range")
    runtime = runtime_root.absolute()
    runtime.mkdir(parents=True, exist_ok=True)
    home = runtime / "home"
    config = runtime / "switchyard-config"
    home.mkdir(exist_ok=True)
    config.mkdir(exist_ok=True)
    return JudgeProcessSpec(
        argv=(
            str(switchyard),
            "--routing-profiles",
            str(routing_profile.resolve(strict=True)),
            "--",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(selected_port),
            "--inbound",
            "openai",
        ),
        cwd=runtime,
        env={"HOME": str(home), "SWITCHYARD_CONFIG_DIR": str(config)},
        base_url=f"http://127.0.0.1:{selected_port}",
        stdout_path=runtime / "judge.stdout.log",
        stderr_path=runtime / "judge.stderr.log",
    )


class DedicatedJudgeProcess:
    """Lifecycle wrapper for one isolated local Switchyard judge server."""

    def __init__(
        self,
        spec: JudgeProcessSpec,
        *,
        transport: HttpTransport = _stdlib_http_transport,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        startup_timeout: float = 30.0,
    ) -> None:
        self.spec = spec
        self.transport = transport
        self.popen = popen
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._stdout: Any = None
        self._stderr: Any = None

    def __enter__(self) -> DedicatedJudgeClient:
        environment = dict(os.environ)
        environment.update(self.spec.env)
        self._stdout = self.spec.stdout_path.open("xb")
        self._stderr = self.spec.stderr_path.open("xb")
        try:
            self.process = self.popen(
                self.spec.argv,
                cwd=self.spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._stdout,
                stderr=self._stderr,
                close_fds=True,
                start_new_session=(os.name == "posix"),
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise TrialQAJudgeError(
                        f"dedicated judge exited during startup: {self.process.returncode}"
                    )
                try:
                    status, _body = self.transport("GET", f"{self.spec.base_url}/health", None, 1.0)
                except TrialQAJudgeError:
                    time.sleep(0.05)
                    continue
                if status == 200:
                    return DedicatedJudgeClient(self.spec.base_url, transport=self.transport)
                time.sleep(0.05)
            raise TrialQAJudgeError("dedicated judge did not become healthy")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.process is not None and self.process.poll() is None:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows fallback.
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows fallback.
                    self.process.kill()
                self.process.wait(timeout=5)
        if self._stdout is not None:
            self._stdout.close()
        if self._stderr is not None:
            self._stderr.close()


def score_and_import_generation(
    *,
    generation: GenerationResult,
    row: TrialQARow,
    judge: Callable[[Mapping[str, object]], str],
    project_dir: Path,
) -> ScoredGeneration:
    """Judge without fallback, build the reference reward, and import evidence."""

    if generation.row_id != row.id or generation.dataset_row_index != row.dataset_row_index:
        raise TrialQADemoError("generation identity differs from the score row")
    validate_generation_for_import(generation, project_dir=project_dir)
    outcome = score_semantic_answer(
        question=row.question,
        ideal=row.ideal,
        answer=generation.answer,
        model=JUDGE_ROUTE,
        judge=judge,
    )
    from benchmark.trialqa_local_dataset import ValidatedPrediction

    prediction = ValidatedPrediction(
        row_id=row.id,
        dataset_row_index=row.dataset_row_index,
        repeat_index=generation.repeat_index,
        condition=generation.condition,
        answer=generation.answer,
        answer_source=generation.answer_source,
        partition=cast(Any, generation.partition),
        question_group_key=question_group_key(row),
        task_name=task_name(row, generation.repeat_index),
        trajectory_path=str(generation.trajectory_path),
    )
    reward = cast(
        JsonObject,
        build_reward_record(
            row=row,
            prediction=prediction,
            n_repeats=generation.n_repeats,
            outcome=outcome,
            process_metrics={
                "switchyard_total_requests": generation.stats["total_requests"],
                "switchyard_total_tokens": generation.stats.get("total_tokens", {}),
                "codex_usage": generation.usage,
            },
        ),
    )
    session_document = _read_json_object(
        generation.session_dir / "session.json", "Switchyard session"
    )
    run_context = session_document.get("run_context")
    if not isinstance(run_context, dict):
        raise TrialQADemoError("validated Switchyard session lost its run context")
    evidence_outcome: JsonObject = {
        "score": outcome.score,
        "raw_score": outcome.score,
        "source_scale": "0_to_1",
        "label": outcome.judge_result,
        "judge_rationale": outcome.rationale,
        "verifier": JUDGE_VERIFIER,
        "metrics": {"semantic_equivalence": outcome.score},
        "partition": generation.partition,
        "row_id": row.id,
        "question": row.question,
        "question_group_key": question_group_key(row),
        "repeat_index": generation.repeat_index,
        "n_repeats": generation.n_repeats,
        "task_name": generation.task_id,
        "condition": generation.condition,
    }
    # Sergei's trajectory analyst receives expected-vs-actual supervision for
    # donor/train traces. Held-out answers remain excluded from native evidence
    # so they can never enter the distilled candidate.
    if generation.condition == "donor" and generation.partition == "train":
        evidence_outcome.update(
            {
                "ideal_answer": row.ideal,
                "submitted_answer": generation.answer,
            }
        )
    evidence = import_native_trialqa_evidence(
        generation.session_dir,
        namespace=NAMESPACE,
        task={
            "id": generation.task_id,
            "question_id": row.id,
            "row_id": row.id,
            "question": row.question,
            "question_group_key": question_group_key(row),
            "task_name": generation.task_id,
            "condition": generation.condition,
            "partition": generation.partition,
            "repeat_index": generation.repeat_index,
            "n_repeats": generation.n_repeats,
        },
        outcome=evidence_outcome,
        run={
            "run_id": generation.manifest_id,
            "phase": "donor" if generation.condition == "donor" else "evaluation",
            "model": EXECUTOR_MODEL,
            "executor_model": EXECUTOR_MODEL,
            "route": EXECUTOR_ROUTE,
            "harness": "switchyard-codex-local",
            "skill_loaded": run_context["skill_loaded"],
            "candidate_id": run_context["candidate_id"],
            "candidate_manifest_sha256": run_context["candidate_manifest_sha256"],
            "candidate_skill_sha256": run_context["candidate_skill_sha256"],
        },
        project_dir=project_dir,
    )
    return ScoredGeneration(
        generation=generation,
        outcome=outcome,
        reward=reward,
        evidence=evidence,
    )


def _token_counts(generation: GenerationResult) -> dict[str, int]:
    raw = generation.stats.get("total_tokens")
    if not isinstance(raw, dict):
        raise TrialQADemoError("generation stats lack total token counts")
    counts: dict[str, int] = {}
    for field in ("prompt", "completion", "total"):
        value = raw.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TrialQADemoError(f"generation has invalid {field} token count")
        counts[field] = value
    if counts["total"] != counts["prompt"] + counts["completion"]:
        raise TrialQADemoError("generation total tokens do not equal prompt plus completion")
    return counts


def scored_result_record(scored: ScoredGeneration) -> TrialResultRecord:
    generation = scored.generation
    counts = _token_counts(generation)
    return TrialResultRecord(
        manifest_id=generation.manifest_id,
        task_id=generation.task_id,
        pair_id=generation.pair_id,
        row_id=generation.row_id,
        question_group_key=generation.pair_id.rsplit("-r", maxsplit=1)[0],
        condition=generation.condition,
        repeat_index=generation.repeat_index,
        n_repeats=generation.n_repeats,
        status="scored",
        score=scored.outcome.score,
        prompt_tokens=counts["prompt"],
        completion_tokens=counts["completion"],
        total_tokens=counts["total"],
        evidence_id=scored.evidence.evidence_id,
        error_stage=None,
        error_type=None,
    )


def failure_result_record(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    stage: str,
    error: BaseException,
    generation: GenerationResult | None = None,
    usage: Mapping[str, object] | None = None,
) -> TrialResultRecord:
    """Retain one failed trial as zero/error without inventing judge success."""

    if generation is not None:
        counts = _token_counts(generation)
    elif usage is not None:
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        if (
            not isinstance(prompt, int)
            or isinstance(prompt, bool)
            or prompt < 0
            or not isinstance(completion, int)
            or isinstance(completion, bool)
            or completion < 0
        ):
            raise TrialQADemoError("failed generation has invalid Codex token usage")
        counts = {
            "prompt": prompt,
            "completion": completion,
            "total": prompt + completion,
        }
    else:
        counts = {"prompt": 0, "completion": 0, "total": 0}
    row_id = task.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise TrialQADemoError("failure task has an invalid row ID")
    return TrialResultRecord(
        manifest_id=_safe_component(manifest.get("manifest_id"), "manifest id"),
        task_id=_safe_component(task.get("task_id"), "task id"),
        pair_id=_safe_component(task.get("pair_id"), "pair id"),
        row_id=row_id,
        question_group_key=_safe_component(task.get("question_group_key"), "question group key"),
        condition=cast(str, task["condition"]),
        repeat_index=cast(int, task["repeat_index"]),
        n_repeats=cast(int, task["n_repeats"]),
        status="error",
        score=0.0,
        prompt_tokens=counts["prompt"],
        completion_tokens=counts["completion"],
        total_tokens=counts["total"],
        evidence_id=None,
        error_stage=stage,
        error_type=type(error).__name__,
    )


def write_trial_result(path: Path, record: TrialResultRecord) -> None:
    _write_json_atomic(path, record.json_document(), exclusive=True)


def write_failure_result(results_root: Path, record: TrialResultRecord) -> Path:
    if record.status != "error":
        raise TrialQADemoError("failure result writer requires error status")
    digest = hashlib.sha256(_canonical_json(record.json_document())).hexdigest()[:16]
    path = results_root / "failures" / f"{record.task_id}-{digest}.json"
    write_trial_result(path, record)
    return path


def load_trial_result(path: Path) -> TrialResultRecord:
    document = _read_json_object(path, "trial result")
    if document.pop("schema_version", None) != "switchyard.trialqa_result.v1":
        raise TrialQADemoError("trial result has the wrong schema")
    try:
        record = TrialResultRecord(**document)
    except TypeError as exc:
        raise TrialQADemoError("trial result fields do not match its schema") from exc
    if record.status not in {"scored", "error"} or record.score not in {0.0, 1.0}:
        raise TrialQADemoError("trial result has an invalid status or score")
    if (
        any(
            value < 0
            for value in (record.prompt_tokens, record.completion_tokens, record.total_tokens)
        )
        or record.total_tokens != record.prompt_tokens + record.completion_tokens
    ):
        raise TrialQADemoError("trial result has invalid token counts")
    return record


def collect_protocol_results(
    results_root: Path, manifest: Mapping[str, object]
) -> tuple[TrialResultRecord, ...]:
    """Collect one terminal record per task, preferring a successful retry."""

    records: list[TrialResultRecord] = []
    for task in cast(list[dict[str, object]], manifest.get("tasks", [])):
        task_id = cast(str, task["task_id"])
        success_path = results_root / f"{task_id}.json"
        failure_paths = sorted((results_root / "failures").glob(f"{task_id}-*.json"))
        failures = {load_trial_result(path) for path in failure_paths}
        if success_path.is_file() and not success_path.is_symlink():
            if any(failure.error_stage == "generation" for failure in failures):
                raise TrialQADemoError(
                    f"task {task_id} has both a success and terminal generation failure"
                )
            records.append(load_trial_result(success_path))
            continue
        if len(failures) != 1:
            raise TrialQADemoError(
                f"task {task_id} needs exactly one terminal success/error record, got {len(failures)}"
            )
        records.append(next(iter(failures)))
    return tuple(records)


def validate_protocol_result_ledger(
    results_root: Path,
    records: Sequence[TrialResultRecord],
    ledger: ResumableLedger,
) -> None:
    """Bind every terminal error result to its hash-chained completion record."""

    for record in records:
        completion = ledger.event_record(record.task_id, "completed")
        payload = completion.get("payload")
        if not isinstance(payload, dict):
            raise TrialQADemoError("completed ledger payload is invalid")
        if record.status == "scored":
            if payload.get("terminal_error") is True:
                raise TrialQADemoError(
                    f"scored task {record.task_id} is marked as a terminal error"
                )
            result_path = (results_root / f"{record.task_id}.json").absolute()
            if (
                result_path.is_symlink()
                or not result_path.is_file()
                or payload.get("result_path") != str(result_path)
                or payload.get("result_sha256") != _sha256_file(result_path)
                or payload.get("score") != record.score
                or payload.get("evidence_id") != record.evidence_id
            ):
                raise TrialQADemoError(
                    f"scored task {record.task_id} result differs from its completion binding"
                )
            scored_event = ledger.event_record(record.task_id, "scored")
            evidence_event = ledger.event_record(record.task_id, "evidence_imported")
            scored_payload = scored_event.get("payload")
            evidence_payload = evidence_event.get("payload")
            if (
                not isinstance(scored_payload, dict)
                or not isinstance(evidence_payload, dict)
                or scored_payload.get("score") != record.score
                or scored_payload.get("result_path") != str(result_path)
                or scored_payload.get("result_sha256") != payload.get("result_sha256")
                or evidence_payload.get("evidence_id") != record.evidence_id
                or evidence_payload.get("result_sha256") != payload.get("result_sha256")
                or payload.get("scored_record_sha256") != scored_event.get("record_sha256")
                or payload.get("evidence_record_sha256") != evidence_event.get("record_sha256")
            ):
                raise TrialQADemoError(
                    f"scored task {record.task_id} lacks hash-bound score provenance"
                )
            continue
        if record.error_stage != "generation" or payload.get("terminal_error") is not True:
            raise TrialQADemoError(
                f"error task {record.task_id} lacks terminal generation provenance"
            )
        failure_paths = sorted((results_root / "failures").glob(f"{record.task_id}-*.json"))
        matches = [path for path in failure_paths if load_trial_result(path) == record]
        if len(matches) != 1:
            raise TrialQADemoError(
                f"terminal task {record.task_id} has ambiguous failure artifacts"
            )
        result_path = matches[0].absolute()
        ledgered_path = payload.get("failure_result_path")
        if (
            not isinstance(ledgered_path, str)
            or Path(ledgered_path).absolute() != result_path
            or payload.get("failure_result_sha256") != _sha256_file(result_path)
        ):
            raise TrialQADemoError(
                f"terminal task {record.task_id} result differs from its ledger binding"
            )
        failed_sha256 = payload.get("failed_record_sha256")
        failures = [
            item
            for item in ledger.records()
            if item.get("task_id") == record.task_id
            and item.get("event") == "failed"
            and item.get("record_sha256") == failed_sha256
        ]
        if len(failures) != 1:
            raise TrialQADemoError(f"terminal task {record.task_id} lacks its failed ledger record")
        failure_payload = failures[0].get("payload")
        proof = failure_payload.get("session_proof") if isinstance(failure_payload, dict) else None
        ordinary_terminal = (
            isinstance(failure_payload, dict) and failure_payload.get("terminal") is True
        )
        exhausted_zero_request = (
            isinstance(failure_payload, dict)
            and payload.get("retry_exhausted") is True
            and failure_payload.get("terminal") is False
            and failure_payload.get("retry_permitted") is True
            and isinstance(proof, dict)
            and proof.get("total_requests") == 0
        )
        if (
            not isinstance(failure_payload, dict)
            or not (ordinary_terminal or exhausted_zero_request)
            or not isinstance(proof, dict)
        ):
            raise TrialQADemoError(f"terminal task {record.task_id} lacks Switchyard session proof")
        session_path_value = proof.get("session_path")
        hashes = proof.get("artifact_sha256")
        quarantine_value = payload.get("quarantined_attempt")
        if (
            not isinstance(session_path_value, str)
            or not isinstance(hashes, dict)
            or not isinstance(quarantine_value, str)
        ):
            raise TrialQADemoError(f"terminal task {record.task_id} has malformed session proof")
        session_path = Path(session_path_value)
        quarantine = Path(quarantine_value)
        session_id = proof.get("session_id")
        expected_sessions = (
            results_root.parent / ".switchyard" / "skill-distillation" / NAMESPACE / "sessions"
        )
        expected_quarantine_parent = results_root.parent / "failed-attempts" / record.task_id
        if (
            not isinstance(session_id, str)
            or _SAFE_COMPONENT.fullmatch(session_id) is None
            or session_path.absolute() != (expected_sessions / session_id).absolute()
            or session_path.is_symlink()
            or not session_path.is_dir()
            or quarantine.parent.absolute() != expected_quarantine_parent.absolute()
            or quarantine.is_symlink()
            or not quarantine.is_dir()
        ):
            raise TrialQADemoError(f"terminal task {record.task_id} has unsafe proof paths")
        output_hashes = failure_payload.get("artifact_sha256")
        allowed_outputs = {
            "codex-final.json",
            "switchyard-codex.stdout.log",
            "switchyard.stderr.log",
        }
        if (
            not isinstance(output_hashes, dict)
            or not output_hashes
            or set(output_hashes).difference(allowed_outputs)
        ):
            raise TrialQADemoError(f"terminal task {record.task_id} lacks raw output hashes")
        for name, expected_hash in output_hashes.items():
            output_path = quarantine / "outputs" / name
            if (
                not isinstance(name, str)
                or not isinstance(expected_hash, str)
                or output_path.is_symlink()
                or not output_path.is_file()
                or output_path.stat().st_nlink != 1
                or _sha256_file(output_path) != expected_hash
            ):
                raise TrialQADemoError(f"terminal task {record.task_id} changed raw output {name}")
        proof_paths = {
            "session.json": session_path / "session.json",
            "stats.json": session_path / "stats.json",
            "run-context.json": quarantine / "runtime" / "launch-metadata" / "run-context.json",
            "active-evidence.json": (
                quarantine / "runtime" / "launch-metadata" / "active-evidence.json"
            ),
        }
        if proof.get("turns_present") is True:
            proof_paths["turns.jsonl"] = session_path / "turns.jsonl"
        elif hashes.get("turns.jsonl") is not None:
            raise TrialQADemoError(
                f"terminal task {record.task_id} has inconsistent trajectory proof"
            )
        for name, path in proof_paths.items():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_nlink != 1
                or hashes.get(name) != _sha256_file(path)
            ):
                raise TrialQADemoError(
                    f"terminal task {record.task_id} changed session artifact {name}"
                )
        requests = proof.get("total_requests")
        served_models = proof.get("served_models")
        if (
            not isinstance(requests, int)
            or isinstance(requests, bool)
            or requests < 0
            or not isinstance(served_models, list)
            or not all(isinstance(model, str) for model in served_models)
            or (requests > 0 and not served_models)
            or (requests == 0 and bool(served_models))
            or set(served_models).difference({EXECUTOR_MODEL, "nvidia/nemotron-3-ultra"})
        ):
            raise TrialQADemoError(
                f"terminal task {record.task_id} has invalid executor attribution"
            )
        stats_document = _read_json_object(
            session_path / "stats.json", "terminal Switchyard session stats"
        )
        try:
            openai_transport = _validate_openai_transport_stats(
                stats_document,
                total_requests=requests,
                require_priced=False,
            )
        except TrialQADemoError as exc:
            raise TrialQADemoError(
                f"terminal task {record.task_id} has invalid transport accounting"
            ) from exc
        if proof.get("openai_transport") != openai_transport:
            raise TrialQADemoError(
                f"terminal task {record.task_id} transport proof differs from stats"
            )
        usage = failure_payload.get("usage")
        if usage is None:
            expected_counts = (0, 0, 0)
        elif isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if (
                not isinstance(input_tokens, int)
                or isinstance(input_tokens, bool)
                or input_tokens < 0
                or not isinstance(output_tokens, int)
                or isinstance(output_tokens, bool)
                or output_tokens < 0
            ):
                raise TrialQADemoError(
                    f"terminal task {record.task_id} has invalid token usage proof"
                )
            expected_counts = (
                input_tokens,
                output_tokens,
                input_tokens + output_tokens,
            )
        else:
            raise TrialQADemoError(f"terminal task {record.task_id} has invalid token usage proof")
        if (requests > 0) is (usage is None):
            raise TrialQADemoError(
                f"terminal task {record.task_id} request and usage proof disagree"
            )
        if expected_counts != (
            record.prompt_tokens,
            record.completion_tokens,
            record.total_tokens,
        ):
            raise TrialQADemoError(
                f"terminal task {record.task_id} token usage differs from session proof"
            )


def build_protocol_report(
    manifest: Mapping[str, object],
    records: Sequence[TrialResultRecord],
) -> JsonObject:
    """Compute Sergei's exact replicate metrics with full-count gates."""

    validate_manifest_pairing(manifest)
    kind = manifest.get("kind")
    if kind not in {"pilot", "full"}:
        raise TrialQADemoError("A/B protocol reports require a pilot or full manifest")
    expected_tasks = {
        cast(str, task["task_id"]): cast(JsonObject, task)
        for task in cast(list[dict[str, object]], manifest["tasks"])
    }
    by_task: dict[str, TrialResultRecord] = {}
    for record in records:
        if record.manifest_id != manifest.get("manifest_id"):
            raise TrialQADemoError("trial result belongs to another manifest")
        if record.task_id not in expected_tasks or record.task_id in by_task:
            raise TrialQADemoError("trial results contain unknown or duplicate task IDs")
        task = expected_tasks[record.task_id]
        for field in (
            "pair_id",
            "row_id",
            "question_group_key",
            "condition",
            "repeat_index",
            "n_repeats",
        ):
            if getattr(record, field) != task.get(field):
                raise TrialQADemoError(f"trial result differs from manifest at {field}")
        by_task[record.task_id] = record
    if set(by_task) != set(expected_tasks):
        raise TrialQADemoError(
            f"report is incomplete: expected {len(expected_tasks)} records, got {len(by_task)}"
        )

    expected_questions = len(
        {
            cast(str, task["question_group_key"])
            for task in cast(list[JsonObject], manifest["tasks"])
        }
    )
    expected_repeats = FULL_REPEATS if kind == "full" else PILOT_REPEATS
    condition_report: JsonObject = {}
    for condition in ("baseline", "treatment"):
        condition_records = [record for record in records if record.condition == condition]
        expected_trials = expected_questions * expected_repeats
        if len(condition_records) != expected_trials:
            raise TrialQADemoError(
                f"{condition} requires exactly {expected_trials} records, got {len(condition_records)}"
            )
        per_question: dict[str, list[TrialResultRecord]] = {}
        for record in condition_records:
            per_question.setdefault(record.question_group_key, []).append(record)
        if len(per_question) != expected_questions:
            raise TrialQADemoError(f"{condition} requires exactly {expected_questions} questions")
        incomplete = sum(
            len({record.repeat_index for record in question_records}) < expected_repeats
            for question_records in per_question.values()
        )
        if incomplete:
            raise TrialQADemoError(f"{condition} has {incomplete} incomplete questions")
        all_scores = [record.score for record in condition_records]
        question_means = [
            sum(record.score for record in question_records) / len(question_records)
            for question_records in per_question.values()
        ]
        question_mins = [
            min(record.score for record in question_records)
            for question_records in per_question.values()
        ]
        question_maxes = [
            max(record.score for record in question_records)
            for question_records in per_question.values()
        ]
        tokens = {
            "prompt": sum(record.prompt_tokens for record in condition_records),
            "completion": sum(record.completion_tokens for record in condition_records),
            "total": sum(record.total_tokens for record in condition_records),
        }
        condition_report[condition] = {
            "records": len(condition_records),
            "questions": len(per_question),
            "repeats": expected_repeats,
            "trial_mean": sum(all_scores) / len(all_scores),
            "question_macro_mean": sum(question_means) / len(question_means),
            "worst_case": sum(question_mins) / len(question_mins),
            "oracle": sum(question_maxes) / len(question_maxes),
            "incomplete": incomplete,
            "error_records": sum(record.status == "error" for record in condition_records),
            "tokens": tokens,
        }
    baseline = cast(JsonObject, condition_report["baseline"])
    treatment = cast(JsonObject, condition_report["treatment"])
    baseline_tokens = cast(JsonObject, baseline["tokens"])["total"]
    treatment_tokens = cast(JsonObject, treatment["tokens"])["total"]
    assert isinstance(baseline_tokens, int) and isinstance(treatment_tokens, int)
    return {
        "schema_version": "switchyard.trialqa_protocol_report.v1",
        "manifest_id": manifest["manifest_id"],
        "kind": kind,
        "conditions": condition_report,
        "benefit": {
            "trial_mean_delta": cast(float, treatment["trial_mean"])
            - cast(float, baseline["trial_mean"]),
            "question_macro_mean_delta": cast(float, treatment["question_macro_mean"])
            - cast(float, baseline["question_macro_mean"]),
            "worst_case_delta": cast(float, treatment["worst_case"])
            - cast(float, baseline["worst_case"]),
            "oracle_delta": cast(float, treatment["oracle"]) - cast(float, baseline["oracle"]),
            "total_token_delta": treatment_tokens - baseline_tokens,
            "token_reduction_fraction": (
                (baseline_tokens - treatment_tokens) / baseline_tokens if baseline_tokens else None
            ),
        },
        "count_gate": {
            "passed": True,
            "expected_questions_per_arm": expected_questions,
            "expected_records_per_arm": expected_questions * expected_repeats,
        },
    }


def build_comparison_report(scored: Sequence[ScoredGeneration]) -> JsonObject:
    """Aggregate exact paired accuracy and token benefit for held-out A/B runs."""

    pairs: dict[tuple[str, int], dict[str, ScoredGeneration]] = {}
    for item in scored:
        generation = item.generation
        if generation.condition not in {"baseline", "treatment"}:
            raise TrialQADemoError("comparison report accepts held-out A/B generations only")
        key = (generation.row_id, generation.repeat_index)
        condition_items = pairs.setdefault(key, {})
        if generation.condition in condition_items:
            raise TrialQADemoError(f"duplicate scored condition for pair {key!r}")
        condition_items[generation.condition] = item
    incomplete = [key for key, items in pairs.items() if set(items) != {"baseline", "treatment"}]
    if incomplete:
        raise TrialQADemoError(f"comparison report contains an incomplete pair: {incomplete[0]!r}")
    if not pairs:
        raise TrialQADemoError("comparison report contains no pairs")

    conditions: JsonObject = {}
    for condition in ("baseline", "treatment"):
        items = [pair[condition] for pair in pairs.values()]
        tokens = {"prompt": 0, "completion": 0, "total": 0}
        for item in items:
            counts = _token_counts(item.generation)
            for field, value in counts.items():
                tokens[field] += value
        correct = sum(item.outcome.score for item in items)
        conditions[condition] = {
            "trials": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "tokens": tokens,
            "mean_tokens_per_trial": tokens["total"] / len(items),
        }
    baseline = cast(JsonObject, conditions["baseline"])
    treatment = cast(JsonObject, conditions["treatment"])
    baseline_tokens = cast(JsonObject, baseline["tokens"])["total"]
    treatment_tokens = cast(JsonObject, treatment["tokens"])["total"]
    if not isinstance(baseline_tokens, int) or not isinstance(treatment_tokens, int):
        raise TrialQADemoError("comparison report token aggregation failed")
    reduction = (baseline_tokens - treatment_tokens) / baseline_tokens if baseline_tokens else None
    return {
        "schema_version": "switchyard.trialqa_comparison_report.v1",
        "pair_count": len(pairs),
        "conditions": conditions,
        "benefit": {
            "accuracy_delta": cast(float, treatment["accuracy"])
            - cast(float, baseline["accuracy"]),
            "total_token_delta": treatment_tokens - baseline_tokens,
            "token_reduction_fraction": reduction,
        },
        "pairing": {
            "key": ["row_id", "repeat_index"],
            "complete": True,
        },
    }


class ResumableLedger:
    """Append-only hash-chained task ledger for one immutable manifest."""

    _TRANSITIONS: Mapping[str | None, frozenset[str]] = {
        None: frozenset({"generation_started"}),
        "generation_started": frozenset({"generation_completed", "failed"}),
        "generation_completed": frozenset({"scored", "failed"}),
        "scored": frozenset({"evidence_imported", "failed"}),
        "evidence_imported": frozenset({"completed", "failed"}),
        "failed": frozenset({"generation_started", "score_retry_started", "completed"}),
        "score_retry_started": frozenset({"scored", "failed"}),
        "completed": frozenset(),
    }

    def __init__(self, path: Path, manifest: Mapping[str, object]) -> None:
        self.path = path.absolute()
        self.manifest = dict(manifest)
        self.manifest_id = _safe_component(manifest.get("manifest_id"), "manifest id")
        self.task_ids = {
            cast(str, task["task_id"])
            for task in cast(list[dict[str, object]], manifest.get("tasks", []))
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> tuple[JsonObject, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_nlink != 1:
            raise TrialQADemoError(f"unsafe resumable ledger: {self.path}")
        records: list[JsonObject] = []
        previous = "sha256:" + "0" * 64
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrialQADemoError(f"invalid ledger JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise TrialQADemoError(f"ledger line {line_number} is not an object")
            record = cast(JsonObject, value)
            supplied_hash = record.pop("record_sha256", None)
            if (
                record.get("schema_version") != LEDGER_SCHEMA_VERSION
                or record.get("sequence") != len(records)
                or record.get("previous_sha256") != previous
                or record.get("manifest_id") != self.manifest_id
            ):
                raise TrialQADemoError(f"ledger chain is invalid at line {line_number}")
            expected_hash = _sha256_bytes(_canonical_json(record))
            record["record_sha256"] = supplied_hash
            if supplied_hash != expected_hash:
                raise TrialQADemoError(f"ledger record hash differs at line {line_number}")
            previous = expected_hash
            records.append(record)
        return tuple(records)

    def states(self) -> dict[str, str]:
        states: dict[str, str] = {}
        for record in self.records():
            task_id = record.get("task_id")
            event = record.get("event")
            if (
                not isinstance(task_id, str)
                or task_id not in self.task_ids
                or not isinstance(event, str)
            ):
                raise TrialQADemoError("ledger record references an unknown task or event")
            previous = states.get(task_id)
            if event not in self._TRANSITIONS.get(previous, frozenset()):
                raise TrialQADemoError(
                    f"invalid ledger transition for {task_id}: {previous!r} -> {event!r}"
                )
            states[task_id] = event
        return states

    def append(
        self, task_id: str, event: str, payload: Mapping[str, object] | None = None
    ) -> JsonObject:
        _safe_component(task_id, "ledger task id")
        if task_id not in self.task_ids:
            raise TrialQADemoError(f"ledger task is not in manifest: {task_id}")
        records = self.records()
        states = self.states()
        previous_state = states.get(task_id)
        if event not in self._TRANSITIONS.get(previous_state, frozenset()):
            raise TrialQADemoError(
                f"invalid ledger transition for {task_id}: {previous_state!r} -> {event!r}"
            )
        previous_hash = records[-1]["record_sha256"] if records else "sha256:" + "0" * 64
        record: JsonObject = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": len(records),
            "previous_sha256": previous_hash,
            "manifest_id": self.manifest_id,
            "task_id": task_id,
            "event": event,
            "recorded_at": _utc_now(),
            "payload": dict(payload or {}),
        }
        record["record_sha256"] = _sha256_bytes(_canonical_json(record))
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.write(descriptor, _canonical_json(record) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record

    def pending_task_ids(self) -> tuple[str, ...]:
        states = self.states()
        running = [task_id for task_id, state in states.items() if state == "generation_started"]
        if running:
            raise TrialQADemoError(
                f"ledger contains interrupted running task; mark failed before resume: {running[0]}"
            )
        ordered = [
            cast(str, task["task_id"])
            for task in cast(list[dict[str, object]], self.manifest["tasks"])
        ]
        return tuple(task_id for task_id in ordered if states.get(task_id) != "completed")

    def event_record(self, task_id: str, event: str) -> JsonObject:
        """Return the sole ledger event for one task, rejecting ambiguity."""

        matches = [
            record
            for record in self.records()
            if record.get("task_id") == task_id and record.get("event") == event
        ]
        if len(matches) != 1:
            raise TrialQADemoError(
                f"ledger contains {len(matches)} {event!r} records for {task_id!r}"
            )
        return matches[0]


def _probe_local_command(
    command: Sequence[str],
    *,
    marker: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    result = run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise TrialQADemoError(f"local preflight command failed: {' '.join(command)}")
    output = (result.stdout + "\n" + result.stderr).strip()
    if marker.lower() not in output.lower():
        raise TrialQADemoError(f"local preflight output lacks {marker!r}: {' '.join(command)}")
    return output.splitlines()[0]


def _codex_safety_attestation() -> JsonObject:
    return {
        "disabled_features": list(CODEX_DISABLED_FEATURES),
        "otel_exporter": "none",
        "otel_metrics_exporter": "none",
        "web_search": "disabled",
    }


def _probe_codex_safety(
    codex_bin: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> JsonObject:
    command = [str(codex_bin)]
    for feature in CODEX_DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.extend(
        (
            "-c",
            'otel.exporter="none"',
            "-c",
            'otel.metrics_exporter="none"',
            "-c",
            'web_search="disabled"',
            "features",
            "list",
        )
    )
    result = run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise TrialQADemoError("Codex safety-feature probe failed")
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 3 and columns[0] in CODEX_DISABLED_FEATURES:
            states[columns[0]] = columns[-1]
    if states != dict.fromkeys(CODEX_DISABLED_FEATURES, "false"):
        raise TrialQADemoError(f"Codex safety features are not disabled: {states}")
    return _codex_safety_attestation()


def _probe_adapter_schemas(
    tooluniverse_bin: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> JsonObject:
    python = tooluniverse_bin.parent / "python"
    command = [str(python), str(TOOLUNIVERSE_ADAPTER_PATH), "--describe-tools"]
    result = run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise TrialQADemoError("local TrialQA MCP adapter schema probe failed")
    try:
        value: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TrialQADemoError("TrialQA MCP adapter schema probe returned invalid JSON") from exc
    expected = describe_tools_document()
    if not isinstance(value, dict) or value != expected:
        raise TrialQADemoError("TrialQA MCP adapter schema probe differs from local source")
    return _adapter_schema_attestation(cast(dict[str, object], value))


def _probe_namespace_translation(
    *,
    engine_factory: Callable[[], Any] | None = None,
) -> JsonObject:
    """Prove namespace-tool request and response translation without I/O."""

    if engine_factory is None:
        from switchyard_rust.translation import TranslationEngine

        engine_factory = TranslationEngine
    engine = engine_factory()
    namespace = "mcp__tooluniverse"
    child_name = "trialqa_load_active_skill"
    request = {
        "model": EXECUTOR_ROUTE,
        "input": "offline TrialQA namespace translation probe",
        "tools": [
            {
                "type": "namespace",
                "name": namespace,
                "description": "TrialQA ToolUniverse tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": child_name,
                        "description": "Load the active TrialQA skill.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ],
            }
        ],
    }
    try:
        translated_request = engine.translate_request(
            "openai_responses",
            "openai_chat",
            request,
        )
    except Exception as exc:
        raise TrialQADemoError(
            "Responses namespace tools cannot be translated to OpenAI Chat"
        ) from exc
    tools = translated_request.get("tools")
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        raise TrialQADemoError("namespace translation did not emit exactly one Chat function tool")
    translated_tool = cast(dict[str, object], tools[0])
    function = translated_tool.get("function")
    if translated_tool.get("type") != "function" or not isinstance(function, dict):
        raise TrialQADemoError("namespace translation did not emit a Chat function")
    flattened_name = function.get("name")
    if (
        not isinstance(flattened_name, str)
        or _CHAT_FUNCTION_NAME.fullmatch(flattened_name) is None
        or flattened_name == child_name
    ):
        raise TrialQADemoError(
            "namespace translation did not encode the namespace in a flat Chat name"
        )

    completion = {
        "id": "chatcmpl-trialqa-doctor",
        "object": "chat.completion",
        "created": 0,
        "model": EXECUTOR_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "trialqa-doctor-call",
                            "type": "function",
                            "function": {
                                "name": flattened_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    try:
        translated_response = engine.translate_response(
            "openai_chat",
            "openai_responses",
            completion,
        )
    except Exception as exc:
        raise TrialQADemoError(
            "OpenAI Chat namespace call cannot be translated back to Responses"
        ) from exc
    output = translated_response.get("output")
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
        raise TrialQADemoError(
            "namespace response translation did not emit exactly one function call"
        )
    function_call = cast(dict[str, object], output[0])
    if {
        "type": function_call.get("type"),
        "namespace": function_call.get("namespace"),
        "name": function_call.get("name"),
        "call_id": function_call.get("call_id"),
        "arguments": function_call.get("arguments"),
    } != {
        "type": "function_call",
        "namespace": namespace,
        "name": child_name,
        "call_id": "trialqa-doctor-call",
        "arguments": "{}",
    }:
        raise TrialQADemoError(
            "namespace response translation did not reconstruct the exact namespace and child"
        )
    attestation = {
        "schema_version": "switchyard.trialqa_namespace_translation.v1",
        "source_format": "openai_responses",
        "target_format": "openai_chat",
        "namespace": namespace,
        "child_name": child_name,
        "flattened_name": flattened_name,
        "flattened_name_sha256": _sha256_bytes(flattened_name.encode("ascii")),
        "request_flattened_tool_count": len(tools),
        "response_namespace": function_call["namespace"],
        "response_child_name": function_call["name"],
        "response_call_id": function_call["call_id"],
        "model_calls": 0,
    }
    _validate_namespace_translation_attestation(attestation)
    return attestation


def run_doctor(
    *,
    dataset_path: Path,
    experiment_root: Path,
    candidate_root: Path,
    switchyard_bin: Path,
    codex_bin: Path,
    tooluniverse_bin: Path,
    routing_profile: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> JsonObject:
    """Perform the complete local no-spend preflight and return its evidence."""

    dataset_path = dataset_path.absolute()
    candidate_root = candidate_root.absolute()
    switchyard_bin = switchyard_bin.absolute()
    codex_bin = codex_bin.absolute()
    tooluniverse_bin = tooluniverse_bin.absolute()
    routing_profile = routing_profile.absolute()
    dataset = load_pinned_trialqa_parquet(dataset_path)
    split = create_split_manifest(dataset)
    assignments = validate_split_manifest(dataset, split)
    counts = {
        "train": sum(partition == "train" for partition in assignments.values()),
        "test": sum(partition == "test" for partition in assignments.values()),
    }
    if counts != {"train": SERGEI_TRAIN_COUNT, "test": SERGEI_TEST_COUNT}:
        raise TrialQADemoError(f"unexpected TrialQA split counts: {counts}")
    validate_routing_profile(routing_profile)
    _validate_judge_route(routing_profile)
    candidate = validate_candidate_skill(candidate_root, NAMESPACE)
    candidate_proof = _candidate_attestation(candidate)
    validate_tooluniverse_binary(tooluniverse_bin)
    switchyard_version = _probe_local_command(
        [str(switchyard_bin), "--version"], marker="switchyard", run=run
    )
    codex_version = _probe_local_command([str(codex_bin), "--version"], marker="codex", run=run)
    codex_safety = _probe_codex_safety(codex_bin, run=run)
    tool_help = _probe_local_command(
        [str(tooluniverse_bin), "--help"], marker="include-tools", run=run
    )
    adapter_proof = _probe_adapter_schemas(tooluniverse_bin, run=run)
    namespace_translation = _probe_namespace_translation()
    experiment = experiment_root.absolute()
    experiment.mkdir(parents=True, exist_ok=True)
    assert_unique_gold_artifact(
        dataset.path,
        experiment,
        expected_sha256=dataset.parquet_sha256,
    )
    test_row = next(row for row in dataset.rows if assignments[row.id] == "test")
    with tempfile.TemporaryDirectory(prefix="trialqa-doctor-", dir=experiment) as temporary:
        capture = Path(temporary) / "capture"
        pair = build_trial_workspace_pair(
            capture_cwd=capture.absolute(),
            task_id="trialqa-doctor",
            prompt=render_trial_prompt(test_row),
            candidate_root=candidate_root.absolute(),
            candidate_skill_directory=NAMESPACE,
        )
        assert_trial_inputs_gold_free(
            dataset=dataset,
            split_manifest=split,
            pair=pair,
            row=test_row,
            partition="test",
        )
        attestation = attest_trial_workspace_pair(
            pair=pair,
            codex_bin=codex_bin.absolute(),
            run=run,
            base_environment=os.environ,
        )
        with GoldArtifactLockdown(dataset.path, expected_sha256=dataset.parquet_sha256):
            locked_mode = stat.S_IMODE(dataset.path.stat().st_mode)
        if locked_mode != 0:
            raise TrialQADemoError("gold lockdown attestation did not observe mode 000")
        treatment = [skill for skill in attestation.treatment_skills if skill.name == NAMESPACE]
        if len(treatment) != 1:
            raise TrialQADemoError("treatment skill attestation disappeared")
        attestation_document = {
            "baseline_has_candidate": False,
            "treatment_has_candidate": True,
            "treatment_path": str(treatment[0].path),
            "candidate_skill_sha256": candidate_proof.skill_sha256,
        }
    return {
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "status": "passed",
        "model_calls": 0,
        "dataset": {
            "id": TRIALQA_DATASET_ID,
            "config": TRIALQA_DATASET_CONFIG,
            "revision": dataset.revision,
            "parquet_sha256": dataset.parquet_sha256,
            "row_count": len(dataset.rows),
            "split_counts": counts,
        },
        "runtime": {
            "switchyard": switchyard_version,
            "codex": codex_version,
            "tooluniverse": tool_help,
        },
        "implementation": {
            "source_sha256": _execution_source_sha256(),
        },
        "mcp_adapter": adapter_proof,
        "codex_safety": codex_safety,
        "namespace_translation": namespace_translation,
        "runtime_artifacts": {
            "switchyard": _runtime_binary_attestation(
                switchyard_bin,
                "Switchyard binary",
            ),
            "codex": _runtime_binary_attestation(codex_bin, "Codex binary"),
            "switchyard_rust_native_extension": _native_extension_attestation(),
            "tooluniverse": {
                **_runtime_binary_attestation(
                    tooluniverse_bin,
                    "ToolUniverse binary",
                ),
                "version": TOOLUNIVERSE_VERSION,
                "python": _runtime_binary_attestation(
                    tooluniverse_bin.parent / "python",
                    "ToolUniverse venv Python",
                ),
            },
        },
        "routing": {
            "first_route": EXECUTOR_ROUTE,
            "executor_model": EXECUTOR_MODEL,
            "judge_route": JUDGE_ROUTE,
            "judge_model": JUDGE_MODEL,
        },
        "candidate": asdict(candidate_proof),
        "ab_attestation": attestation_document,
        "gold_guard": {
            "unique_pinned_artifact": True,
            "permission_lockdown_verified": True,
            "executor_inputs_gold_free": True,
        },
    }


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--switchyard-bin", type=Path, required=True)
    parser.add_argument("--codex-bin", type=Path, required=True)
    parser.add_argument("--tooluniverse-bin", type=Path, required=True)
    parser.add_argument("--routing-profile", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="run local checks; never make a model call")
    _common_paths(doctor)
    doctor.add_argument("--output", type=Path)
    plan = commands.add_parser("plan", help="write an immutable pilot/full manifest")
    _common_paths(plan)
    plan.add_argument("--kind", choices=("donor", "development", "pilot", "full"), required=True)
    plan.add_argument("--primary-question-start", type=int)
    plan.add_argument("--primary-question-count", type=int)
    plan.add_argument("--doctor-report", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan_prospective = commands.add_parser(
        "plan-prospective",
        help="write a non-official TrialQA-compatible prospective canary manifest",
    )
    _common_paths(plan_prospective)
    plan_prospective.add_argument("--dataset-sha256", required=True)
    plan_prospective.add_argument("--dataset-row-count", type=int, required=True)
    plan_prospective.add_argument("--dataset-revision", required=True)
    plan_prospective.add_argument("--population-report", type=Path, required=True)
    plan_prospective.add_argument("--doctor-report", type=Path, required=True)
    plan_prospective.add_argument("--output", type=Path, required=True)
    run_one = commands.add_parser("run-one", help="execute one paid Ultra generation task")
    _common_paths(run_one)
    run_one.add_argument("--manifest", type=Path, required=True)
    run_one.add_argument("--doctor-report", type=Path, required=True)
    run_one.add_argument("--population-report", type=Path)
    run_one.add_argument("--task-id", required=True)
    run_one.add_argument("--yes-spend", action="store_true")
    score_one = commands.add_parser(
        "score-one", help="judge one generated answer and import its native evidence"
    )
    _common_paths(score_one)
    score_one.add_argument("--manifest", type=Path, required=True)
    score_one.add_argument("--doctor-report", type=Path, required=True)
    score_one.add_argument("--population-report", type=Path)
    score_one.add_argument("--task-id", required=True)
    score_one.add_argument("--generation", type=Path, required=True)
    score_one.add_argument("--judge-port", type=int)
    score_one.add_argument("--yes-spend", action="store_true")
    report = commands.add_parser("report", help="build the exact-count paired A/B report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--results-root", type=Path, required=True)
    report.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            supplied = _read_json_object(args.manifest, "experiment manifest")
            results_root = args.results_root.absolute()
            ledger = ResumableLedger(results_root.parent / "ledger.jsonl", supplied)
            states = ledger.states()
            task_ids = {
                cast(str, task["task_id"])
                for task in cast(list[dict[str, object]], supplied.get("tasks", []))
            }
            if set(states) != task_ids or any(
                states.get(task_id) != "completed" for task_id in task_ids
            ):
                raise TrialQADemoError(
                    "protocol report requires every manifest task to be ledger-completed"
                )
            records = collect_protocol_results(results_root, supplied)
            validate_protocol_result_ledger(results_root, records, ledger)
            report = build_protocol_report(supplied, records)
            if args.output is not None:
                _write_json_atomic(args.output, report, exclusive=True)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "doctor":
            if args.candidate_root is None:
                raise TrialQADemoError("doctor requires --candidate-root")
            report = run_doctor(
                dataset_path=args.dataset,
                experiment_root=args.experiment_root,
                candidate_root=args.candidate_root,
                switchyard_bin=args.switchyard_bin,
                codex_bin=args.codex_bin,
                tooluniverse_bin=args.tooluniverse_bin,
                routing_profile=args.routing_profile,
            )
            if args.output is not None:
                _write_json_atomic(args.output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        if args.command == "plan":
            dataset = load_pinned_trialqa_parquet(args.dataset)
            split = create_split_manifest(dataset)
            candidate = None
            if args.kind != "donor":
                if args.candidate_root is None:
                    raise TrialQADemoError(f"{args.kind} plan requires --candidate-root")
                candidate = validate_candidate_skill(args.candidate_root.absolute(), NAMESPACE)
            expected = build_experiment_manifest(
                dataset=dataset,
                split_manifest=split,
                kind=cast(PlanKind, args.kind),
                candidate=candidate,
                routing_profile=args.routing_profile.absolute(),
                switchyard_bin=args.switchyard_bin.absolute(),
                codex_bin=args.codex_bin.absolute(),
                tooluniverse_bin=args.tooluniverse_bin.absolute(),
                doctor_report=args.doctor_report.absolute(),
                primary_question_start=args.primary_question_start,
                primary_question_count=args.primary_question_count,
            )
            write_experiment_manifest(args.output, expected)
            print(json.dumps(expected, indent=2, sort_keys=True))
            return 0
        if args.command == "plan-prospective":
            if args.candidate_root is None:
                raise TrialQADemoError("plan-prospective requires --candidate-root")
            dataset = load_trialqa_compatible_parquet(
                args.dataset,
                expected_sha256=args.dataset_sha256,
                expected_row_count=args.dataset_row_count,
                revision=args.dataset_revision,
            )
            candidate = validate_candidate_skill(args.candidate_root.absolute(), NAMESPACE)
            expected = build_prospective_experiment_manifest(
                dataset=dataset,
                population_report=args.population_report.absolute(),
                candidate=candidate,
                routing_profile=args.routing_profile.absolute(),
                switchyard_bin=args.switchyard_bin.absolute(),
                codex_bin=args.codex_bin.absolute(),
                tooluniverse_bin=args.tooluniverse_bin.absolute(),
                doctor_report=args.doctor_report.absolute(),
            )
            write_experiment_manifest(args.output, expected)
            print(json.dumps(expected, indent=2, sort_keys=True))
            return 0
        if not args.yes_spend:
            raise TrialQADemoError(
                f"{args.command} requires --yes-spend because it makes a live model call"
            )
        supplied = _read_json_object(args.manifest, "experiment manifest")
        dataset = load_manifest_dataset(args.dataset, supplied)
        split = create_manifest_split(dataset, supplied)
        # A live run accepts only an exactly reproducible donor/pilot/full manifest.
        kind = supplied.get("kind")
        if kind not in {"donor", "development", "pilot", "full"}:
            raise TrialQADemoError("experiment manifest has an invalid kind")
        candidate = None
        if kind != "donor":
            if args.candidate_root is None:
                raise TrialQADemoError(f"{kind} run requires --candidate-root")
            candidate = validate_candidate_skill(args.candidate_root.absolute(), NAMESPACE)
        expected = build_reproducible_manifest_from_supplied(
            supplied=supplied,
            dataset=dataset,
            split_manifest=split,
            candidate=candidate,
            routing_profile=args.routing_profile.absolute(),
            switchyard_bin=args.switchyard_bin.absolute(),
            codex_bin=args.codex_bin.absolute(),
            tooluniverse_bin=args.tooluniverse_bin.absolute(),
            doctor_report=args.doctor_report.absolute(),
            population_report=(
                args.population_report.absolute()
                if getattr(args, "population_report", None) is not None
                else None
            ),
        )
        if supplied != expected:
            raise TrialQADemoError("experiment manifest differs from current pinned inputs")
        capture = args.experiment_root.absolute() / cast(str, supplied["manifest_id"])
        capture.mkdir(parents=True, exist_ok=True)
        if args.command == "score-one":
            ledger = ResumableLedger(capture / "ledger.jsonl", supplied)
            score_state = ledger.states().get(args.task_id)
            if score_state == "failed":
                latest = [
                    record
                    for record in ledger.records()
                    if record.get("task_id") == args.task_id and record.get("event") == "failed"
                ][-1]
                payload = latest.get("payload")
                if not isinstance(payload, dict) or payload.get("stage") != "score-import":
                    raise TrialQADemoError(
                        "score-one can retry only a ledgered score/import failure"
                    )
                ledger.append(
                    args.task_id,
                    "score_retry_started",
                    {"failed_record_sha256": latest["record_sha256"]},
                )
            elif score_state != "generation_completed":
                raise TrialQADemoError("score-one requires a ledgered generation or score retry")
            completion = ledger.event_record(args.task_id, "generation_completed")
            payload = completion.get("payload")
            if not isinstance(payload, dict):
                raise TrialQADemoError("generation_completed ledger payload is missing")
            ledgered_path = payload.get("generation_path")
            ledgered_sha256 = payload.get("generation_sha256")
            if (
                not isinstance(ledgered_path, str)
                or Path(ledgered_path).resolve(strict=True) != args.generation.resolve(strict=True)
                or not isinstance(ledgered_sha256, str)
                or _sha256_file(args.generation) != ledgered_sha256
            ):
                raise TrialQADemoError("generation record differs from its ledgered identity")
            generation = load_generation_result(args.generation)
            if (
                generation.manifest_id != supplied["manifest_id"]
                or generation.task_id != args.task_id
            ):
                raise TrialQADemoError("generation record does not match manifest/task selection")
            if payload.get("artifact_sha256") != dict(generation.artifact_sha256):
                raise TrialQADemoError("generation artifact hashes differ from the ledger")
            task = _manifest_task_by_id(supplied, args.task_id)
            row = dataset.row_by_id(cast(str, task["row_id"]))
            judge_spec = build_judge_process_spec(
                switchyard_bin=args.switchyard_bin,
                routing_profile=args.routing_profile,
                runtime_root=capture / "judge" / args.task_id,
                port=args.judge_port,
            )
            try:
                with DedicatedJudgeProcess(judge_spec) as judge:
                    scored = score_and_import_generation(
                        generation=generation,
                        row=row,
                        judge=judge,
                        project_dir=capture,
                    )
            except Exception as exc:
                write_failure_result(
                    capture / "results",
                    failure_result_record(
                        manifest=supplied,
                        task=task,
                        stage="score-import",
                        error=exc,
                        generation=generation,
                    ),
                )
                ledger.append(
                    args.task_id,
                    "failed",
                    {"stage": "score-import", "error": type(exc).__name__},
                )
                raise
            ledger.append(
                args.task_id,
                "scored",
                {
                    "score": scored.outcome.score,
                    "judge_result": scored.outcome.judge_result,
                    "judge_model": JUDGE_MODEL,
                },
            )
            ledger.append(
                args.task_id,
                "evidence_imported",
                {"evidence_id": scored.evidence.evidence_id},
            )
            ledger.append(args.task_id, "completed")
            write_trial_result(
                capture / "results" / f"{args.task_id}.json",
                scored_result_record(scored),
            )
            print(
                json.dumps(
                    {
                        "task_id": args.task_id,
                        "score": scored.outcome.score,
                        "judge_result": scored.outcome.judge_result,
                        "evidence_id": scored.evidence.evidence_id,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        planned = prepare_generation(
            manifest=supplied,
            task_id=args.task_id,
            dataset=dataset,
            split_manifest=split,
            capture_cwd=capture,
            candidate_root=args.candidate_root if kind != "donor" else None,
            switchyard_bin=args.switchyard_bin,
            codex_bin=args.codex_bin,
            routing_profile=args.routing_profile,
            tooluniverse_bin=args.tooluniverse_bin,
        )
        ledger = ResumableLedger(capture / "ledger.jsonl", supplied)
        ledger.append(args.task_id, "generation_started")
        try:
            result = execute_generation(
                manifest=supplied,
                planned=planned,
                dataset=dataset,
            )
        except Exception as exc:
            usage = completed_model_draw_usage(planned.spec.stdout_path)
            failure_payload: JsonObject = {
                "stage": "generation",
                "error": type(exc).__name__,
                "completed_model_draw": usage is not None,
            }
            if usage is not None:
                failure_payload["usage"] = usage
                failure_path = write_failure_result(
                    capture / "results",
                    failure_result_record(
                        manifest=supplied,
                        task=planned.task,
                        stage="generation",
                        error=exc,
                        usage=usage,
                    ),
                )
                failure_payload.update(
                    {
                        "terminal_result_path": str(failure_path),
                        "terminal_result_sha256": _sha256_file(failure_path),
                    }
                )
            ledger.append(args.task_id, "failed", failure_payload)
            if usage is not None:
                ledger.append(
                    args.task_id,
                    "completed",
                    {
                        "terminal_error": True,
                        "stage": "generation",
                        "failure_result_path": str(failure_path),
                        "failure_result_sha256": _sha256_file(failure_path),
                    },
                )
            raise
        ledger.append(
            args.task_id,
            "generation_completed",
            {
                "generation_path": str(result.generation_path),
                "generation_sha256": _sha256_file(result.generation_path),
                "artifact_sha256": dict(result.artifact_sha256),
            },
        )
        print(json.dumps(result.json_document(), indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        TrialQADataError,
        TrialQADemoError,
        TrialQaLocalRunnerError,
        TrialQAJudgeError,
        ValueError,
    ) as exc:
        print(f"trialqa-local-demo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

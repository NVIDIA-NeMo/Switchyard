# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent, resumable driver for immutable TrialQA manifests.

Completed model draws are terminal. A generation exception is retried only
when its unique native Switchyard session affirmatively proves zero executor
requests. All hash-chain ledger writes remain serialized in this parent
process, and each worker gets an identical private copy of the pinned parquet
so permission lockdowns never race.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark.trialqa_local_demo as demo  # noqa: E402
from benchmark.trialqa_local_dataset import TrialQADataset  # noqa: E402
from benchmark.trialqa_local_runner import (  # noqa: E402
    EXECUTOR_MODEL,
    EXECUTOR_ROUTE,
    NAMESPACE,
    validate_candidate_skill,
)
from switchyard.lib.skill_distillation_store import SkillDistillationStore  # noqa: E402


@dataclass(frozen=True)
class Runtime:
    experiment_root: Path
    switchyard: Path
    codex: Path
    tooluniverse: Path
    profile: Path
    doctor: Path
    candidate: Path | None
    generation_timeout_seconds: int
    generation_timeout_policy: str


@dataclass(frozen=True)
class RecordedFailure:
    """One append-only failed event and any proven completed-draw usage."""

    ledger_record: dict[str, object]
    usage: dict[str, object] | None
    terminal_result_path: Path | None
    retry_permitted: bool
    manual_review: bool
    completed_model_draw: bool


@dataclass(frozen=True)
class SessionProof:
    """Immutable proof of the one native Switchyard session for an attempt."""

    ledger_payload: dict[str, object]
    total_requests: int
    usage: dict[str, object] | None


class SessionProofError(RuntimeError):
    """The attempt cannot be classified safely enough for an automatic retry."""


BatchStage = Literal["all", "generation", "score"]
ConditionScope = Literal["both", "baseline", "treatment"]
DEFAULT_GENERATION_TIMEOUT_SECONDS = 1800
MIN_DEVELOPMENT_GENERATION_TIMEOUT_SECONDS = 60
MAX_DEVELOPMENT_GENERATION_TIMEOUT_SECONDS = 1799
MIN_CANARY_GENERATION_TIMEOUT_SECONDS = 120
MAX_CANARY_GENERATION_TIMEOUT_SECONDS = 900
DEFAULT_GENERATION_TIMEOUT_POLICY = "protocol-default-v1"
DEVELOPMENT_GENERATION_TIMEOUT_POLICY = "development-terminal-v1"
CANARY_GENERATION_TIMEOUT_POLICY = "exposed-treatment-canary-v1"
SCOPE_ATTESTATION_SCHEMA_VERSION = 1
SCOPE_SELECTOR_VERSION = "manifest-question-window-v1"
EXPOSED_HELDOUT_QUARANTINE_QUESTIONS = demo.PRIMARY_HELDOUT_QUESTION_START


def _manifest_heldout_quarantine_questions(manifest: Mapping[str, object]) -> int:
    """Return the global held-out prefix excluded from this manifest's evidence."""

    if manifest.get("kind") != "full":
        return 0
    primary_start, _primary_count = demo.primary_evaluation_window(manifest)
    return primary_start if primary_start is not None else EXPOSED_HELDOUT_QUARANTINE_QUESTIONS


@dataclass(frozen=True)
class TaskScope:
    """Immutable task membership plus its deterministic selection evidence."""

    tasks: tuple[dict[str, object], ...]
    selector: str
    manifest_question_start: int
    question_start: int
    question_limit: int | None
    available_question_count: int | None
    selected_question_groups: tuple[str, ...]
    selected_repeat_indices: tuple[int, ...]
    condition: ConditionScope
    heldout_quarantine_questions: int
    heldout_classification: str

    def metadata(self, manifest_id: object) -> dict[str, object]:
        """Return a hash-bound JSON document for emitted batch metadata."""

        if not isinstance(manifest_id, str) or not manifest_id:
            raise RuntimeError("scope attestation requires a manifest ID")
        task_ids = [str(task["task_id"]) for task in self.tasks]
        pair_ids = list(dict.fromkeys(str(task["pair_id"]) for task in self.tasks))
        document: dict[str, object] = {
            "schema_version": SCOPE_ATTESTATION_SCHEMA_VERSION,
            "selector": self.selector,
            "manifest_id": manifest_id,
            "manifest_question_start": self.manifest_question_start,
            "question_start": self.question_start,
            "question_limit": self.question_limit,
            "question_end_exclusive": (
                self.question_start + len(self.selected_question_groups)
                if self.available_question_count is not None
                else None
            ),
            "available_question_count": self.available_question_count,
            "selected_question_count": len(self.selected_question_groups),
            "selected_question_groups": list(self.selected_question_groups),
            "selected_repeat_indices": list(self.selected_repeat_indices),
            "condition": self.condition,
            "heldout_quarantine_questions": self.heldout_quarantine_questions,
            "heldout_classification": self.heldout_classification,
            "selected_pair_count": len(pair_ids),
            "selected_task_count": len(task_ids),
            "selected_pair_ids_sha256": _canonical_sha256(pair_ids),
            "selected_task_ids_sha256": _canonical_sha256(task_ids),
        }
        return {
            **document,
            "attestation_sha256": _canonical_sha256(document),
        }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _resolve_generation_timeout(
    requested: int | None,
    *,
    canary_requested: int | None = None,
    single_arm_canary: bool = False,
    kind: object,
    max_generation_attempts: int,
) -> tuple[int, str]:
    """Resolve the default deadline or one narrowly scoped terminal override."""

    if requested is not None and canary_requested is not None:
        raise RuntimeError("generation timeout overrides are mutually exclusive")
    if canary_requested is not None:
        if not (
            MIN_CANARY_GENERATION_TIMEOUT_SECONDS
            <= canary_requested
            <= MAX_CANARY_GENERATION_TIMEOUT_SECONDS
        ):
            raise RuntimeError("canary generation timeout must be between 120 and 900 seconds")
        if not single_arm_canary:
            raise RuntimeError(
                "--canary-generation-timeout-seconds requires a reviewed single-arm canary"
            )
        if max_generation_attempts != 1:
            raise RuntimeError("canary generation timeout requires --max-generation-attempts 1")
        return canary_requested, CANARY_GENERATION_TIMEOUT_POLICY
    if requested is None:
        return DEFAULT_GENERATION_TIMEOUT_SECONDS, DEFAULT_GENERATION_TIMEOUT_POLICY
    if not (
        MIN_DEVELOPMENT_GENERATION_TIMEOUT_SECONDS
        <= requested
        <= MAX_DEVELOPMENT_GENERATION_TIMEOUT_SECONDS
    ):
        raise RuntimeError("development generation timeout must be between 60 and 1799 seconds")
    if kind != "development":
        raise RuntimeError(
            "--development-generation-timeout-seconds requires a development manifest"
        )
    if max_generation_attempts != 1:
        raise RuntimeError("development generation timeout requires --max-generation-attempts 1")
    return requested, DEVELOPMENT_GENERATION_TIMEOUT_POLICY


def _validate_single_arm_execution(
    manifest: Mapping[str, object],
    *,
    condition: ConditionScope,
    stage: BatchStage,
    limit: int | None,
    question_start: int,
    question_limit: int | None,
    max_generation_attempts: int,
) -> None:
    """Allow only the exposed-heldout treatment mechanism canary.

    Evaluation remains paired everywhere else.  The sole exception is a
    generation-only treatment canary for one of the three reviewed, already
    exposed held-out questions of a descriptive manifest.
    """

    if condition == "both":
        return
    if condition != "treatment":
        raise RuntimeError("single-arm baseline execution is not permitted")
    protocol = manifest.get("protocol")
    descriptive_full = (
        manifest.get("kind") == "full"
        and isinstance(protocol, Mapping)
        and protocol.get("primary_evaluation_scope") is None
        and protocol.get("performance_eligible") is False
    )
    if not descriptive_full:
        raise RuntimeError(
            "single-arm treatment execution requires a descriptive nonperformance full manifest"
        )
    if stage != "generation":
        raise RuntimeError("single-arm treatment execution is generation-only")
    if max_generation_attempts != 1:
        raise RuntimeError("single-arm treatment execution requires --max-generation-attempts 1")
    if limit is not None:
        raise RuntimeError("single-arm treatment execution does not permit --limit")
    if question_limit is None:
        raise RuntimeError("single-arm treatment execution requires an explicit --question-limit")
    if question_limit != 1 or question_start not in {2, 5, 7}:
        raise RuntimeError(
            "single-arm treatment question window must select exactly one reviewed "
            "held-out ordinal: 2, 5, or 7"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _worker_datasets(
    source: Path,
    manifest: Mapping[str, object],
    manifest_id: str,
    workers: int,
) -> list[TrialQADataset]:
    root = Path("/private/tmp") / f"switchyard-{manifest_id}-datasets"
    datasets: list[TrialQADataset] = []
    expected = demo.load_manifest_dataset(source, manifest)
    for index in range(workers):
        target = root / f"worker-{index}" / "source" / "trialqa" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"worker dataset is missing or unsafe: {target}")
            # A forcibly stopped generation can leave its private copy under
            # the protocol's mode-000 gold lockdown. Restore owner access only
            # long enough to re-attest the immutable hash before reuse.
            target.chmod(0o600)
            if _sha256(target) != f"sha256:{expected.parquet_sha256}":
                raise RuntimeError(f"worker dataset hash mismatch: {target}")
        else:
            shutil.copy2(source, target)
            target.chmod(0o600)
        datasets.append(demo.load_manifest_dataset(target, manifest))
    return datasets


def _generation(
    manifest: dict[str, Any],
    task_id: str,
    dataset: TrialQADataset,
    split: dict[str, object],
    runtime: Runtime,
    capture: Path,
) -> demo.GenerationResult:
    planned = demo.prepare_generation(
        manifest=manifest,
        task_id=task_id,
        dataset=dataset,
        split_manifest=split,
        capture_cwd=capture,
        candidate_root=runtime.candidate,
        switchyard_bin=runtime.switchyard,
        codex_bin=runtime.codex,
        routing_profile=runtime.profile,
        tooluniverse_bin=runtime.tooluniverse,
    )
    executor = partial(
        demo.run_streaming_subprocess,
        timeout_seconds=runtime.generation_timeout_seconds,
    )
    return demo.execute_generation(
        manifest=manifest,
        planned=planned,
        dataset=dataset,
        executor=executor,
    )


def _score(
    manifest: dict[str, Any],
    task: dict[str, object],
    generation: demo.GenerationResult,
    dataset: TrialQADataset,
    runtime: Runtime,
    capture: Path,
    worker_index: int,
) -> demo.ScoredGeneration:
    row = dataset.row_by_id(str(task["row_id"]))
    judge_spec = demo.build_judge_process_spec(
        switchyard_bin=runtime.switchyard,
        routing_profile=runtime.profile,
        runtime_root=capture / "judge" / str(task["task_id"]),
        port=9300 + worker_index,
    )
    with demo.DedicatedJudgeProcess(judge_spec) as judge:
        return demo.score_and_import_generation(
            generation=generation,
            row=row,
            judge=judge,
            project_dir=capture,
        )


def _generation_outputs(capture: Path, task: Mapping[str, object]) -> Path:
    arm = "treatment" if task.get("condition") == "treatment" else "baseline"
    return capture / "trialqa-local" / str(task["pair_id"]) / "arms" / arm / "outputs"


def _launch_metadata(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    capture: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    arm = "treatment" if task.get("condition") == "treatment" else "baseline"
    metadata = (
        capture / "trialqa-local" / str(task["pair_id"]) / "runtime" / arm / "launch-metadata"
    )
    context_path = metadata / "run-context.json"
    active_path = metadata / "active-evidence.json"
    try:
        context = demo._read_json_object(context_path, "generation run context")
        active = demo._read_json_object(active_path, "generation active-skill evidence")
    except demo.TrialQADemoError as exc:
        raise SessionProofError("generation launch metadata is unverifiable") from exc
    identity = {
        "manifest_id": manifest.get("manifest_id"),
        "executor_model": EXECUTOR_MODEL,
        "route": EXECUTOR_ROUTE,
        "skill_loaded": task.get("condition") == "treatment",
    }
    for key in (
        "task_id",
        "pair_id",
        "row_id",
        "dataset_row_index",
        "question_group_key",
        "partition",
        "phase",
        "condition",
        "repeat_index",
        "n_repeats",
    ):
        identity[key] = task.get(key)
    mismatches = [key for key, expected in identity.items() if context.get(key) != expected]
    if mismatches:
        raise SessionProofError(f"generation launch context differs at {mismatches[0]}")
    expected_loaded = task.get("condition") == "treatment"
    if active.get("loaded") is not expected_loaded:
        raise SessionProofError("generation active-skill evidence has the wrong condition")
    candidate = manifest.get("candidate")
    expected_candidate: dict[str, object]
    if expected_loaded:
        if not isinstance(candidate, dict):
            raise SessionProofError("treatment manifest lacks candidate attestation")
        expected_candidate = {
            "candidate_id": candidate.get("candidate_id"),
            "candidate_manifest_sha256": candidate.get("manifest_sha256"),
            "candidate_skill_sha256": candidate.get("skill_sha256"),
        }
        if not isinstance(active.get("path"), str) or not active["path"]:
            raise SessionProofError("treatment active-skill evidence lacks a path")
    else:
        expected_candidate = {
            "candidate_id": None,
            "candidate_manifest_sha256": None,
            "candidate_skill_sha256": None,
        }
        if active.get("path") is not None:
            raise SessionProofError("unskilled arm has an active-skill path")
    for key, expected in expected_candidate.items():
        if context.get(key) != expected:
            raise SessionProofError(f"generation candidate attestation differs at {key}")
    for context_key, active_key in (
        ("candidate_id", "candidate_id"),
        ("candidate_manifest_sha256", "manifest_sha256"),
        ("candidate_skill_sha256", "skill_sha256"),
    ):
        if context.get(context_key) != active.get(active_key):
            raise SessionProofError(
                f"generation context and active evidence differ at {context_key}"
            )
    return (
        dict(context),
        dict(active),
        {
            "run-context.json": demo._sha256_file(context_path),
            "active-evidence.json": demo._sha256_file(active_path),
        },
    )


def _bound_session_ids(
    ledger: demo.ResumableLedger,
    task_id: str,
    sessions_path: Path,
) -> set[str]:
    bound: set[str] = set()
    for record in ledger.records():
        if record.get("task_id") != task_id:
            continue
        payload = record.get("payload")
        proof = payload.get("session_proof") if isinstance(payload, dict) else None
        if not isinstance(proof, dict):
            continue
        session_id = proof.get("session_id")
        if isinstance(session_id, str):
            session_value = proof.get("session_path")
            artifacts = proof.get("artifact_sha256")
            expected_dir = sessions_path / session_id
            if (
                not isinstance(session_value, str)
                or Path(session_value).absolute() != expected_dir.absolute()
                or expected_dir.is_symlink()
                or not expected_dir.is_dir()
                or not isinstance(artifacts, dict)
            ):
                raise SessionProofError(
                    f"ledger-bound Switchyard session is unverifiable: {session_id}"
                )
            for name in ("session.json", "stats.json"):
                expected_hash = artifacts.get(name)
                path = expected_dir / name
                if (
                    not isinstance(expected_hash, str)
                    or path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_nlink != 1
                    or demo._sha256_file(path) != expected_hash
                ):
                    raise SessionProofError(
                        f"ledger-bound Switchyard artifact differs: {session_id}/{name}"
                    )
            turns_hash = artifacts.get("turns.jsonl")
            turns_path = expected_dir / "turns.jsonl"
            if turns_hash is None:
                if turns_path.exists() or turns_path.is_symlink():
                    raise SessionProofError(
                        f"ledger-bound zero-request trajectory differs: {session_id}"
                    )
            elif (
                not isinstance(turns_hash, str)
                or turns_path.is_symlink()
                or not turns_path.is_file()
                or turns_path.stat().st_nlink != 1
                or demo._sha256_file(turns_path) != turns_hash
            ):
                raise SessionProofError(f"ledger-bound Switchyard trajectory differs: {session_id}")
            bound.add(session_id)
    return bound


def _stats_usage(stats: Mapping[str, object]) -> dict[str, object]:
    tokens = stats.get("total_tokens")
    if not isinstance(tokens, dict):
        raise SessionProofError("Switchyard session stats lack total token counts")
    prompt = tokens.get("prompt")
    completion = tokens.get("completion")
    total = tokens.get("total")
    if (
        not isinstance(prompt, int)
        or isinstance(prompt, bool)
        or prompt < 0
        or not isinstance(completion, int)
        or isinstance(completion, bool)
        or completion < 0
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total != prompt + completion
    ):
        raise SessionProofError("Switchyard session stats contain invalid token counts")
    return {"input_tokens": prompt, "output_tokens": completion}


def _validate_session_proof(
    *,
    session_dir: Path,
    expected_context: Mapping[str, object],
    expected_active: Mapping[str, object],
    launch_sha256: Mapping[str, str],
) -> SessionProof:
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise SessionProofError("Switchyard session is not a real directory")
    session_path = session_dir / "session.json"
    stats_path = session_dir / "stats.json"
    turns_path = session_dir / "turns.jsonl"
    try:
        session = demo._read_json_object(session_path, "Switchyard session")
        stats = demo._read_json_object(stats_path, "Switchyard session stats")
    except demo.TrialQADemoError as exc:
        raise SessionProofError("Switchyard session artifacts are unverifiable") from exc
    session_id = session.get("session_id")
    turn_count = session.get("turn_count")
    exit_code = session.get("exit_code")
    if (
        not isinstance(session_id, str)
        or session_id != session_dir.name
        or isinstance(session.get("schema_version"), bool)
        or session.get("schema_version") != 1
        or session.get("namespace") != NAMESPACE
        or session.get("launch_target") != "codex"
        or session.get("display_model") != EXECUTOR_ROUTE
        or session.get("status") not in {"completed", "failed", "interrupted"}
        or not isinstance(session.get("ended_at"), str)
        or not session["ended_at"]
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not isinstance(turn_count, int)
        or isinstance(turn_count, bool)
        or turn_count < 0
        or session.get("turns_path") != "turns.jsonl"
        or session.get("stats_path") != "stats.json"
        or session.get("run_context") != dict(expected_context)
        or session.get("active_skill") != dict(expected_active)
    ):
        raise SessionProofError("Switchyard session metadata is invalid")

    total_requests = stats.get("total_requests")
    total_errors = stats.get("total_errors")
    models = stats.get("models")
    if (
        not isinstance(total_requests, int)
        or isinstance(total_requests, bool)
        or total_requests < 0
        or not isinstance(total_errors, int)
        or isinstance(total_errors, bool)
        or total_errors < 0
        or not isinstance(models, dict)
    ):
        raise SessionProofError("Switchyard session request stats are invalid")
    try:
        openai_transport = demo._validate_openai_transport_stats(
            stats,
            total_requests=total_requests,
            require_priced=False,
        )
    except demo.TrialQADemoError as exc:
        raise SessionProofError("Switchyard OpenAI transport stats are invalid") from exc
    for subsystem_name in ("classifier", "planner"):
        subsystem = stats.get(subsystem_name)
        if (
            not isinstance(subsystem, dict)
            or subsystem.get("total_requests") != 0
            or isinstance(subsystem.get("total_requests"), bool)
            or subsystem.get("total_errors") != 0
            or isinstance(subsystem.get("total_errors"), bool)
        ):
            raise SessionProofError(f"Switchyard {subsystem_name} recorded non-executor activity")
    model_attempts: dict[str, int] = {}
    successful_calls = 0
    model_errors = 0
    for model, value in models.items():
        if not isinstance(model, str) or not isinstance(value, dict):
            raise SessionProofError("Switchyard session model stats are invalid")
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
            raise SessionProofError("Switchyard session model counts are invalid")
        attempts = calls + errors
        if attempts:
            model_attempts[model] = attempts
        successful_calls += calls
        model_errors += errors
    usage = _stats_usage(stats)
    turns_sha256: str | None = None
    served_models: set[str] = set()
    if total_requests == 0:
        if (
            model_attempts
            or model_errors != 0
            or turn_count != 0
            or usage != {"input_tokens": 0, "output_tokens": 0}
            or turns_path.exists()
            or turns_path.is_symlink()
            or session.get("trajectory_sha256") is not None
        ):
            raise SessionProofError("zero-request Switchyard session is inconsistent")
        result_usage = None
    else:
        accepted_models = {
            EXECUTOR_MODEL,
            "nvidia/nemotron-3-ultra",
        }
        unpriced_lazy_failures = int(openai_transport["unpriced_null_eof_retries"])
        uncaptured_successes = successful_calls - turn_count
        if (
            set(model_attempts).difference(accepted_models)
            or sum(model_attempts.values()) != total_requests
            or model_errors != total_errors
            or uncaptured_successes < 0
            or uncaptured_successes > unpriced_lazy_failures
        ):
            raise SessionProofError(
                "Switchyard session requests were not exclusively attributed to Ultra"
            )
        if turn_count == 0:
            if (
                uncaptured_successes < 1
                or turns_path.exists()
                or turns_path.is_symlink()
                or session.get("trajectory_sha256") is not None
            ):
                raise SessionProofError("paid Switchyard session lacks a valid trajectory")
            served_models.update(model_attempts)
            result_usage = usage
            return SessionProof(
                ledger_payload={
                    "session_id": session_id,
                    "session_path": str(session_dir),
                    "status": session["status"],
                    "exit_code": exit_code,
                    "turn_count": turn_count,
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "openai_transport": openai_transport,
                    "served_models": sorted(served_models),
                    "turns_present": False,
                    "artifact_sha256": {
                        "session.json": demo._sha256_file(session_path),
                        "stats.json": demo._sha256_file(stats_path),
                        "turns.jsonl": None,
                        **dict(launch_sha256),
                    },
                },
                total_requests=total_requests,
                usage=result_usage,
            )
        if (
            turns_path.is_symlink()
            or not turns_path.is_file()
            or turns_path.stat().st_nlink != 1
            or turn_count > total_requests
        ):
            raise SessionProofError("paid Switchyard session lacks a valid trajectory")
        try:
            lines = turns_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise SessionProofError("Switchyard session trajectory is unreadable") from exc
        if len(lines) != turn_count:
            raise SessionProofError("Switchyard session trajectory count differs")
        for expected_index, line in enumerate(lines):
            try:
                turn: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionProofError(
                    "Switchyard session trajectory contains invalid JSON"
                ) from exc
            if (
                not isinstance(turn, dict)
                or isinstance(turn.get("schema_version"), bool)
                or turn.get("schema_version") != 1
                or turn.get("session_id") != session_id
                or isinstance(turn.get("turn_index"), bool)
                or turn.get("turn_index") != expected_index
                or turn.get("served_model") not in accepted_models
            ):
                raise SessionProofError(
                    "Switchyard session trajectory is not contiguous Ultra-only evidence"
                )
            request = turn.get("request")
            if not isinstance(request, dict) or request.get("model") != EXECUTOR_ROUTE:
                raise SessionProofError(
                    "Switchyard session trajectory does not use the pinned executor route"
                )
            expected_candidate_id = expected_active.get("candidate_id")
            expected_manifest_sha256 = expected_active.get("manifest_sha256")
            if (
                turn.get("active_skill_version") != expected_candidate_id
                or turn.get("active_skill_candidate_id") != expected_candidate_id
                or turn.get("active_skill_manifest_sha256") != expected_manifest_sha256
            ):
                raise SessionProofError(
                    "Switchyard session trajectory has the wrong active-skill evidence"
                )
            served_models.add(str(turn["served_model"]))
        turns_sha256 = demo._sha256_file(turns_path)
        if session.get("trajectory_sha256") != turns_sha256:
            raise SessionProofError("Switchyard session trajectory hash differs")
        result_usage = usage

    return SessionProof(
        ledger_payload={
            "session_id": session_id,
            "session_path": str(session_dir),
            "status": session["status"],
            "exit_code": exit_code,
            "turn_count": turn_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "openai_transport": openai_transport,
            "served_models": sorted(served_models),
            "turns_present": turns_sha256 is not None,
            "artifact_sha256": {
                "session.json": demo._sha256_file(session_path),
                "stats.json": demo._sha256_file(stats_path),
                "turns.jsonl": turns_sha256,
                **dict(launch_sha256),
            },
        },
        total_requests=total_requests,
        usage=result_usage,
    )


def _unbound_session_proof(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    capture: Path,
    ledger: demo.ResumableLedger,
) -> SessionProof:
    context, active, launch_sha256 = _launch_metadata(manifest, task, capture)
    try:
        store = SkillDistillationStore(NAMESPACE, capture)
    except (OSError, ValueError) as exc:
        raise SessionProofError("Switchyard session store is unavailable") from exc
    task_id = str(task["task_id"])
    try:
        bound = _bound_session_ids(ledger, task_id, store.sessions_path)
        session_entries = sorted(store.sessions_path.iterdir())
    except OSError as exc:
        raise SessionProofError("Switchyard session store is unreadable") from exc
    matches: list[Path] = []
    for session_dir in session_entries:
        if session_dir.name in bound or session_dir.is_symlink() or not session_dir.is_dir():
            continue
        try:
            session = demo._read_json_object(session_dir / "session.json", "Switchyard session")
        except demo.TrialQADemoError:
            continue
        if session.get("run_context") == context:
            matches.append(session_dir)
    if len(matches) != 1:
        raise SessionProofError(
            f"generation has {len(matches)} unbound matching Switchyard sessions, expected one"
        )
    return _validate_session_proof(
        session_dir=matches[0],
        expected_context=context,
        expected_active=active,
        launch_sha256=launch_sha256,
    )


def _completed_draw_usage(capture: Path, task: Mapping[str, object]) -> dict[str, object] | None:
    """Return usage only when Codex proves one successfully completed turn."""

    events_path = _generation_outputs(capture, task) / "switchyard-codex.stdout.log"
    if not events_path.is_file() or events_path.is_symlink():
        return None
    try:
        usage = dict(demo._parse_codex_events(events_path, enforce_tool_policy=False))
    except demo.TrialQADemoError:
        return None
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
        raise SessionProofError("Codex completed-turn usage is invalid")
    return usage


def _write_failure_result(
    *,
    manifest: dict[str, Any],
    task: dict[str, object],
    capture: Path,
    stage: str,
    error: BaseException,
    generation: demo.GenerationResult | None = None,
    usage: Mapping[str, object] | None = None,
) -> Path:
    # Generation failures use completed-turn Codex usage; score/import failures
    # retain their validated GenerationResult as the authoritative token source.
    record = demo.failure_result_record(
        manifest=manifest,
        task=task,
        stage=stage,
        error=error,
        generation=generation,
        usage=usage,
    )
    digest = hashlib.sha256(demo._canonical_json(record.json_document())).hexdigest()[:16]
    failure_path = capture / "results" / "failures" / f"{record.task_id}-{digest}.json"
    if failure_path.exists():
        if demo.load_trial_result(failure_path) != record:
            raise RuntimeError(f"existing failure record differs: {failure_path}")
    else:
        demo.write_failure_result(capture / "results", record)
    return failure_path


def _record_generation_failure(
    *,
    manifest: dict[str, Any],
    task: dict[str, object],
    capture: Path,
    ledger: demo.ResumableLedger,
    error: BaseException,
    retry_exhausted: bool,
) -> RecordedFailure:
    """Classify one attempt from native proof before allowing any retry."""

    codex_usage: dict[str, object] | None = None
    codex_proof_error: SessionProofError | None = None
    try:
        codex_usage = _completed_draw_usage(capture, task)
    except SessionProofError as exc:
        codex_proof_error = exc
    proof: SessionProof | None = None
    proof_error: SessionProofError | None = None
    try:
        proof = _unbound_session_proof(
            manifest=manifest,
            task=task,
            capture=capture,
            ledger=ledger,
        )
    except SessionProofError as exc:
        proof_error = exc

    completed_model_draw = codex_usage is not None or (
        proof is not None and proof.total_requests > 0
    )
    contradiction = codex_usage is not None and proof is not None and proof.total_requests == 0
    usage_mismatch = (
        codex_usage is not None
        and proof is not None
        and proof.usage is not None
        and any(
            codex_usage.get(key) != proof.usage.get(key)
            for key in ("input_tokens", "output_tokens")
        )
    )
    if isinstance(error, demo.GenerationTimeoutError):
        timeout_seconds: float | None = error.timeout_seconds
        process_group_terminated: bool | None = error.process_group_terminated
    else:
        timeout_seconds = None
        process_group_terminated = None
    timed_out = timeout_seconds is not None
    timeout_cleanup_failed = timed_out and process_group_terminated is not True
    manual_review = (
        codex_proof_error is not None
        or proof_error is not None
        or contradiction
        or usage_mismatch
        or timeout_cleanup_failed
    )
    retry_permitted = (
        not manual_review
        and proof is not None
        and proof.total_requests == 0
        and codex_usage is None
        and not retry_exhausted
    )
    terminal = not manual_review and not retry_permitted
    usage = codex_usage or (proof.usage if proof is not None else None)
    terminal_result_path = None
    if terminal:
        terminal_result_path = _write_failure_result(
            manifest=manifest,
            task=task,
            capture=capture,
            stage="generation",
            error=error,
            usage=usage,
        )
    failure_payload: dict[str, object] = {
        "stage": "generation",
        "error": type(error).__name__,
        "message": str(error),
        "completed_model_draw": completed_model_draw,
        "retry_exhausted": retry_exhausted,
        "retry_permitted": retry_permitted,
        "manual_review": manual_review,
        "terminal": terminal,
    }
    if timed_out:
        failure_payload.update(
            {
                "timed_out": True,
                "wall_clock_timeout_seconds": timeout_seconds,
                "process_group_terminated": process_group_terminated,
            }
        )
    if proof is not None:
        failure_payload["session_proof"] = proof.ledger_payload
    if proof_error is not None:
        failure_payload["session_proof_error"] = str(proof_error)
    if codex_proof_error is not None:
        failure_payload["codex_proof_error"] = str(codex_proof_error)
    if contradiction:
        failure_payload["session_proof_error"] = (
            "Codex reports a completed turn but Switchyard reports zero requests"
        )
    if usage_mismatch:
        failure_payload["session_proof_error"] = (
            "Codex completed-turn usage differs from Switchyard session usage"
        )
    outputs = _generation_outputs(capture, task)
    artifacts: dict[str, str] = {}
    for name in (
        "codex-final.json",
        "switchyard-codex.stdout.log",
        "switchyard.stderr.log",
    ):
        path = outputs / name
        if path.is_file() and not path.is_symlink():
            artifacts[name] = demo._sha256_file(path)
    if artifacts:
        failure_payload["artifact_sha256"] = artifacts
    if usage is not None:
        failure_payload["usage"] = usage
    if terminal_result_path is not None:
        failure_payload.update(
            {
                "terminal_result_path": str(terminal_result_path),
                "terminal_result_sha256": demo._sha256_file(terminal_result_path),
            }
        )
    ledger_record = ledger.append(str(task["task_id"]), "failed", failure_payload)
    return RecordedFailure(
        ledger_record=ledger_record,
        usage=usage,
        terminal_result_path=terminal_result_path,
        retry_permitted=retry_permitted,
        manual_review=manual_review,
        completed_model_draw=completed_model_draw,
    )


def _record_score_failure(
    *,
    manifest: dict[str, Any],
    task: dict[str, object],
    capture: Path,
    ledger: demo.ResumableLedger,
    error: BaseException,
    generation: demo.GenerationResult,
) -> None:
    failure_path = _write_failure_result(
        manifest=manifest,
        task=task,
        capture=capture,
        stage="score-import",
        error=error,
        generation=generation,
    )
    ledger.append(
        str(task["task_id"]),
        "failed",
        {
            "stage": "score-import",
            "error": type(error).__name__,
            "message": str(error),
            "terminal_result_path": str(failure_path),
            "terminal_result_sha256": demo._sha256_file(failure_path),
        },
    )


def _scored_result_path(capture: Path, task_id: str) -> Path:
    return capture / "results" / f"{task_id}.json"


def _validate_scored_result(
    record: demo.TrialResultRecord,
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    generation: demo.GenerationResult,
) -> None:
    token_counts = generation.stats.get("total_tokens")
    if not isinstance(token_counts, dict):
        raise RuntimeError(f"generation token counts are invalid for {record.task_id}")
    expected = {
        "manifest_id": manifest.get("manifest_id"),
        "task_id": task.get("task_id"),
        "pair_id": task.get("pair_id"),
        "row_id": task.get("row_id"),
        "question_group_key": task.get("question_group_key"),
        "condition": task.get("condition"),
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
        "prompt_tokens": token_counts.get("prompt"),
        "completion_tokens": token_counts.get("completion"),
        "total_tokens": token_counts.get("total"),
    }
    if record.status != "scored" or record.evidence_id is None:
        raise RuntimeError(f"successful score result is invalid for {record.task_id}")
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise RuntimeError(
                f"successful score result differs from generation at {field}: {record.task_id}"
            )


def _commit_scored_result(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    capture: Path,
    ledger: demo.ResumableLedger,
    scored: demo.ScoredGeneration,
) -> Path:
    task_id = str(task["task_id"])
    record = demo.scored_result_record(scored)
    _validate_scored_result(
        record,
        manifest=manifest,
        task=task,
        generation=scored.generation,
    )
    result_path = _scored_result_path(capture, task_id)
    demo.write_trial_result(result_path, record)
    result_sha256 = demo._sha256_file(result_path)
    scored_event = ledger.append(
        task_id,
        "scored",
        {
            "score": scored.outcome.score,
            "judge_result": scored.outcome.judge_result,
            "judge_model": demo.JUDGE_MODEL,
            "result_path": str(result_path),
            "result_sha256": result_sha256,
        },
    )
    evidence_event = ledger.append(
        task_id,
        "evidence_imported",
        {
            "evidence_id": scored.evidence.evidence_id,
            "result_sha256": result_sha256,
        },
    )
    ledger.append(
        task_id,
        "completed",
        {
            "result_path": str(result_path),
            "result_sha256": result_sha256,
            "score": record.score,
            "evidence_id": record.evidence_id,
            "scored_record_sha256": scored_event["record_sha256"],
            "evidence_record_sha256": evidence_event["record_sha256"],
        },
    )
    return result_path


def _finish_partial_scored_result(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    capture: Path,
    ledger: demo.ResumableLedger,
    generation: demo.GenerationResult,
) -> demo.TrialResultRecord:
    """Finish a score whose immutable result was written before an interruption."""

    task_id = str(task["task_id"])
    result_path = _scored_result_path(capture, task_id)
    record = demo.load_trial_result(result_path)
    _validate_scored_result(
        record,
        manifest=manifest,
        task=task,
        generation=generation,
    )
    result_sha256 = demo._sha256_file(result_path)
    state = ledger.states().get(task_id)
    if state in {"generation_completed", "score_retry_started"}:
        ledger.append(
            task_id,
            "scored",
            {
                "score": record.score,
                "judge_result": "recovered-from-immutable-result",
                "judge_model": demo.JUDGE_MODEL,
                "result_path": str(result_path),
                "result_sha256": result_sha256,
                "recovered": True,
            },
        )
        state = "scored"
    if state == "scored":
        scored_event = ledger.event_record(task_id, "scored")
        scored_payload = scored_event.get("payload")
        if (
            not isinstance(scored_payload, dict)
            or scored_payload.get("score") != record.score
            or scored_payload.get("result_path") != str(result_path)
            or scored_payload.get("result_sha256") != result_sha256
        ):
            raise RuntimeError(f"scored ledger binding is invalid for {task_id}")
        ledger.append(
            task_id,
            "evidence_imported",
            {
                "evidence_id": record.evidence_id,
                "result_sha256": result_sha256,
                "recovered": True,
            },
        )
        state = "evidence_imported"
    if state != "evidence_imported":
        raise RuntimeError(f"score recovery has an invalid state for {task_id}: {state!r}")
    scored_event = ledger.event_record(task_id, "scored")
    evidence_event = ledger.event_record(task_id, "evidence_imported")
    evidence_payload = evidence_event.get("payload")
    if (
        not isinstance(evidence_payload, dict)
        or evidence_payload.get("evidence_id") != record.evidence_id
        or evidence_payload.get("result_sha256") != result_sha256
    ):
        raise RuntimeError(f"evidence ledger binding is invalid for {task_id}")
    ledger.append(
        task_id,
        "completed",
        {
            "result_path": str(result_path),
            "result_sha256": result_sha256,
            "score": record.score,
            "evidence_id": record.evidence_id,
            "scored_record_sha256": scored_event["record_sha256"],
            "evidence_record_sha256": evidence_event["record_sha256"],
            "recovered": True,
        },
    )
    return record


def _load_completed_generation(
    ledger: demo.ResumableLedger,
    task_id: str,
) -> demo.GenerationResult:
    event = _latest_generation_record(ledger, task_id)
    payload = event.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("generation_path"), str):
        raise RuntimeError(f"ledger generation payload is missing for {task_id}")
    path = Path(payload["generation_path"])
    if payload.get("generation_sha256") != demo._sha256_file(path):
        raise RuntimeError(f"ledger generation hash mismatch for {task_id}")
    generation = demo.load_generation_result(path)
    if generation.manifest_id != ledger.manifest_id or generation.task_id != task_id:
        raise RuntimeError(f"ledgered generation identity mismatch for {task_id}")
    if payload.get("artifact_sha256") != dict(generation.artifact_sha256):
        raise RuntimeError(f"ledgered artifact hashes mismatch for {task_id}")
    return generation


def _latest_generation_record(
    ledger: demo.ResumableLedger,
    task_id: str,
) -> dict[str, object]:
    matches = [
        record
        for record in ledger.records()
        if record.get("task_id") == task_id and record.get("event") == "generation_completed"
    ]
    if not matches:
        raise RuntimeError(f"ledger generation payload is missing for {task_id}")
    return matches[-1]


def _latest_failure(ledger: demo.ResumableLedger, task_id: str) -> dict[str, object]:
    matches = [
        record
        for record in ledger.records()
        if record.get("task_id") == task_id and record.get("event") == "failed"
    ]
    if not matches:
        raise RuntimeError(f"ledger failure payload is missing for {task_id}")
    return matches[-1]


def _latest_failure_stage(ledger: demo.ResumableLedger, task_id: str) -> object:
    payload = _latest_failure(ledger, task_id).get("payload")
    return payload.get("stage") if isinstance(payload, dict) else None


def _generation_attempt_count(ledger: demo.ResumableLedger, task_id: str) -> int:
    return sum(
        record.get("task_id") == task_id and record.get("event") == "generation_started"
        for record in ledger.records()
    )


def _validate_generation_timeout_history(
    ledger: demo.ResumableLedger,
    *,
    timeout_seconds: int,
    timeout_policy: str,
) -> None:
    """Reject deadline changes after the first generation attempt is ledgered."""

    for record in ledger.records():
        if record.get("event") != "generation_started":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("generation_started ledger payload is invalid")
        recorded_timeout = payload.get("wall_clock_timeout_seconds")
        recorded_policy = payload.get("timeout_policy")
        if recorded_timeout is None and recorded_policy is None:
            recorded_timeout = DEFAULT_GENERATION_TIMEOUT_SECONDS
            recorded_policy = DEFAULT_GENERATION_TIMEOUT_POLICY
        if (
            not isinstance(recorded_timeout, int)
            or isinstance(recorded_timeout, bool)
            or not isinstance(recorded_policy, str)
        ):
            raise RuntimeError("ledgered generation timeout policy is invalid")
        if recorded_timeout != timeout_seconds or recorded_policy != timeout_policy:
            raise RuntimeError(
                "generation timeout policy differs from prior attempts in this capture"
            )


def _quarantine_failed_generation(
    *,
    capture: Path,
    task: dict[str, object],
    failure: dict[str, object],
) -> Path:
    pair_id = str(task["pair_id"])
    task_id = str(task["task_id"])
    arm = "treatment" if task.get("condition") == "treatment" else "baseline"
    sequence = failure.get("sequence")
    record_sha256 = failure.get("record_sha256")
    if not isinstance(sequence, int) or not isinstance(record_sha256, str):
        raise RuntimeError(f"failed ledger record is invalid for {task_id}")
    destination = (
        capture
        / "failed-attempts"
        / task_id
        / f"ledger-{sequence:06d}-{record_sha256.removeprefix('sha256:')[:16]}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pair_root = capture / "trialqa-local" / pair_id
    if destination.is_dir() and not destination.is_symlink():
        return destination
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"failed-attempt quarantine collision: {destination}")
    destination.mkdir(parents=True)
    if not pair_root.exists():
        return destination
    if pair_root.is_symlink() or not pair_root.is_dir():
        raise RuntimeError(f"failed generation workspace is unsafe: {pair_root}")

    arm_root = pair_root / "arms" / arm
    runtime_arm = pair_root / "runtime" / arm
    if arm_root.is_symlink() or not arm_root.is_dir():
        raise RuntimeError(f"failed generation arm is missing or unsafe: {arm_root}")
    for source, relative in (
        (arm_root / "outputs", Path("outputs")),
        (arm_root / "answer.txt", Path("answer.txt")),
        (runtime_arm, Path("runtime")),
    ):
        if not source.exists():
            continue
        if source.is_symlink():
            raise RuntimeError(f"failed generation artifact is symlinked: {source}")
        source.rename(destination / relative)
    (arm_root / "outputs").mkdir()
    (runtime_arm / "home").mkdir(parents=True)
    (runtime_arm / "codex-home").mkdir()
    return destination


def _quarantine_failed_judge(
    *,
    capture: Path,
    task_id: str,
    failure: dict[str, object],
) -> Path | None:
    sequence = failure.get("sequence")
    record_sha256 = failure.get("record_sha256")
    if not isinstance(sequence, int) or not isinstance(record_sha256, str):
        raise RuntimeError(f"failed ledger record is invalid for {task_id}")
    source = capture / "judge" / task_id
    destination = (
        capture
        / "failed-attempts"
        / task_id
        / f"judge-ledger-{sequence:06d}-{record_sha256.removeprefix('sha256:')[:16]}"
    )
    if destination.is_dir() and not destination.is_symlink() and not source.exists():
        return destination
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"failed judge workspace is unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"failed-judge quarantine collision: {destination}")
    source.rename(destination)
    return destination


def _finish_terminal_generation_failure(
    *,
    capture: Path,
    task: dict[str, object],
    ledger: demo.ResumableLedger,
    failure: dict[str, object],
) -> Path:
    """Quarantine one terminal attempt and advance its ledger state to completed."""

    payload = failure.get("payload")
    if not isinstance(payload, dict) or payload.get("terminal") is not True:
        raise RuntimeError(f"generation failure is not terminal: {task['task_id']}")
    result_value = payload.get("terminal_result_path")
    result_sha256 = payload.get("terminal_result_sha256")
    if not isinstance(result_value, str) or not isinstance(result_sha256, str):
        raise RuntimeError(f"terminal failure result is missing: {task['task_id']}")
    result_path = Path(result_value)
    if (
        result_path.is_symlink()
        or not result_path.is_file()
        or demo._sha256_file(result_path) != result_sha256
    ):
        raise RuntimeError(f"terminal failure result is invalid: {task['task_id']}")
    quarantine = _quarantine_failed_generation(
        capture=capture,
        task=task,
        failure=failure,
    )
    # Terminal zero-score draws use the append-only ``failed -> completed``
    # transition and never return to ``generation_started`` on a later run.
    ledger.append(
        str(task["task_id"]),
        "completed",
        {
            "terminal_error": True,
            "failure_result_path": str(result_path),
            "failure_result_sha256": result_sha256,
            "quarantined_attempt": str(quarantine),
            "failed_record_sha256": failure["record_sha256"],
        },
    )
    return quarantine


def _finish_exhausted_generation_failure(
    *,
    manifest: dict[str, Any],
    capture: Path,
    task: dict[str, object],
    ledger: demo.ResumableLedger,
    failure: dict[str, object],
    max_attempts: int,
) -> Path:
    """Terminalize a bound zero-request failure when its global budget is spent."""

    payload = failure.get("payload")
    proof = payload.get("session_proof") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("retry_permitted") is not True
        or payload.get("terminal") is not False
        or not isinstance(proof, dict)
        or proof.get("total_requests") != 0
    ):
        raise RuntimeError(f"exhausted generation failure is invalid: {task['task_id']}")
    error = RuntimeError(f"generation exhausted its cumulative {max_attempts}-attempt budget")
    result_path = _write_failure_result(
        manifest=manifest,
        task=task,
        capture=capture,
        stage="generation",
        error=error,
    )
    result_sha256 = demo._sha256_file(result_path)
    quarantine = _quarantine_failed_generation(
        capture=capture,
        task=task,
        failure=failure,
    )
    ledger.append(
        str(task["task_id"]),
        "completed",
        {
            "terminal_error": True,
            "retry_exhausted": True,
            "generation_attempts": max_attempts,
            "failure_result_path": str(result_path),
            "failure_result_sha256": result_sha256,
            "quarantined_attempt": str(quarantine),
            "failed_record_sha256": failure["record_sha256"],
        },
    )
    return quarantine


def _pair_safe_chunks(
    values: list[dict[str, object]],
    size: int,
    *,
    pair_positions: Mapping[str, int] | None = None,
) -> Iterator[list[dict[str, object]]]:
    """Yield pair-safe waves in a deterministic balanced crossover order.

    Fresh A/B pairs alternate which arm runs in the first wave, and the second
    wave uses the complementary arm. This keeps each concurrent wave balanced
    (within one task for odd sizes) instead of confounding every baseline with
    an earlier provider-time window than its treatment peer.
    """

    pair_order: list[str] = []
    by_pair: dict[str, list[dict[str, object]]] = {}
    for task in values:
        pair_id = str(task["pair_id"])
        if pair_id not in by_pair:
            pair_order.append(pair_id)
            by_pair[pair_id] = []
        by_pair[pair_id].append(task)
    if any(len(tasks) > 2 for tasks in by_pair.values()):
        raise RuntimeError("a TrialQA pair contains more than two pending arms")
    scheduled_by_pair: dict[str, list[dict[str, object]]] = {}
    for pending_position, pair_id in enumerate(pair_order):
        if pair_positions is None:
            position = pending_position
        else:
            try:
                position = pair_positions[pair_id]
            except KeyError as exc:
                raise RuntimeError("a pending TrialQA pair lacks its manifest position") from exc
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise RuntimeError("a TrialQA pair has an invalid manifest position")
        pair_tasks = by_pair[pair_id]
        if len(pair_tasks) < 2:
            scheduled_by_pair[pair_id] = pair_tasks
            continue
        by_condition = {str(task.get("condition")): task for task in pair_tasks}
        if set(by_condition) != {"baseline", "treatment"}:
            raise RuntimeError("a TrialQA A/B pair must contain one arm per condition")
        first = "baseline" if position % 2 == 0 else "treatment"
        second = "treatment" if first == "baseline" else "baseline"
        scheduled_by_pair[pair_id] = [by_condition[first], by_condition[second]]
    for pair_index in range(0, len(pair_order), size):
        selected = pair_order[pair_index : pair_index + size]
        max_arms = max(len(scheduled_by_pair[pair_id]) for pair_id in selected)
        for arm_index in range(max_arms):
            wave = [
                scheduled_by_pair[pair_id][arm_index]
                for pair_id in selected
                if arm_index < len(scheduled_by_pair[pair_id])
            ]
            if len({str(task["pair_id"]) for task in wave}) != len(wave):
                raise RuntimeError("pair-safe wave contains duplicate workspaces")
            yield wave


def _build_task_scope(
    tasks: list[dict[str, object]],
    *,
    limit: int | None,
    manifest_question_start: int = 0,
    question_start: int = 0,
    question_limit: int | None,
    repeat_limit: int | None,
    condition: ConditionScope,
    heldout_quarantine_questions: int = 0,
    allow_descriptive_mixed_heldout: bool = False,
) -> TaskScope:
    """Freeze and attest deterministic task membership before resume filtering."""

    if isinstance(question_start, bool) or not isinstance(question_start, int):
        raise RuntimeError("question start must be an integer")
    if (
        isinstance(manifest_question_start, bool)
        or not isinstance(manifest_question_start, int)
        or manifest_question_start < 0
    ):
        raise RuntimeError("manifest question start must not be negative")
    if question_start < 0:
        raise RuntimeError("question start must not be negative")
    if (
        isinstance(heldout_quarantine_questions, bool)
        or not isinstance(heldout_quarantine_questions, int)
        or heldout_quarantine_questions < 0
    ):
        raise RuntimeError("held-out quarantine question count must not be negative")
    if limit is not None and (
        question_start != 0 or question_limit is not None or repeat_limit is not None
    ):
        raise RuntimeError("--limit cannot be combined with question/repeat cohort limits")
    if question_limit is not None and (
        isinstance(question_limit, bool)
        or not isinstance(question_limit, int)
        or question_limit < 1
    ):
        raise RuntimeError("question limit must be positive")
    if repeat_limit is not None and (
        isinstance(repeat_limit, bool) or not isinstance(repeat_limit, int) or repeat_limit < 1
    ):
        raise RuntimeError("repeat limit must be positive")
    if not isinstance(allow_descriptive_mixed_heldout, bool):
        raise RuntimeError("descriptive held-out policy must be boolean")
    descriptive_mixed_heldout = (
        heldout_quarantine_questions > 0
        and question_limit is None
        and question_start == manifest_question_start == 0
        and repeat_limit is None
        and limit is None
        and allow_descriptive_mixed_heldout
    )
    if heldout_quarantine_questions and question_limit is None and not descriptive_mixed_heldout:
        raise RuntimeError(
            "held-out execution requires an explicit --question-limit so quarantined "
            "questions cannot be mixed with evaluation questions"
        )

    uses_question_selector = (
        question_start != 0 or question_limit is not None or repeat_limit is not None
    )
    available_question_count: int | None = None
    selected_groups: list[str] = []
    selected_repeat_indices: tuple[int, ...] = ()
    heldout_classification = "not-applicable"
    if not uses_question_selector:
        scoped = list(tasks if limit is None else tasks[:limit])
        selector = "manifest-all-tasks-v1" if limit is None else "manifest-task-prefix-v1"
        for task in scoped:
            group = task.get("question_group_key")
            if isinstance(group, str) and group and group not in selected_groups:
                selected_groups.append(group)
        selected_repeat_indices = tuple(
            sorted(
                {
                    repeat
                    for task in scoped
                    if isinstance((repeat := task.get("repeat_index")), int)
                    and not isinstance(repeat, bool)
                }
            )
        )
        if descriptive_mixed_heldout:
            available_question_count = len(selected_groups)
            heldout_classification = "descriptive-mixed-heldout"
    else:
        selector = SCOPE_SELECTOR_VERSION
        group_order: list[str] = []
        by_group_repeat: dict[tuple[str, int], list[dict[str, object]]] = {}
        repeats_by_group: dict[str, set[int]] = {}
        for task in tasks:
            group = str(task.get("question_group_key") or "")
            repeat = task.get("repeat_index")
            if not group or not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
                raise RuntimeError("question/repeat cohort selection requires manifest metadata")
            if group not in repeats_by_group:
                group_order.append(group)
                repeats_by_group[group] = set()
            repeats_by_group[group].add(repeat)
            by_group_repeat.setdefault((group, repeat), []).append(task)
        available_question_count = len(group_order)
        local_question_start = question_start - manifest_question_start
        if local_question_start < 0:
            raise RuntimeError("question start precedes the manifest question range")
        if local_question_start >= available_question_count:
            raise RuntimeError("question start exceeds the manifest question range")
        local_question_end = (
            available_question_count
            if question_limit is None
            else local_question_start + question_limit
        )
        if local_question_end > available_question_count:
            raise RuntimeError("question limit exceeds the manifest question count")
        question_end = manifest_question_start + local_question_end
        selected_groups = group_order[local_question_start:local_question_end]
        if not selected_groups:
            raise RuntimeError("question selection is empty")
        repeat_sets = {tuple(sorted(repeats_by_group[group])) for group in selected_groups}
        if len(repeat_sets) != 1:
            raise RuntimeError("selected questions have inconsistent repeat coverage")
        available_repeat_indices = next(iter(repeat_sets), ())
        if available_repeat_indices != tuple(range(1, len(available_repeat_indices) + 1)):
            raise RuntimeError("selected questions have incomplete repeat coverage")
        available_repeats = len(available_repeat_indices)
        selected_repeat_count = repeat_limit or available_repeats
        if selected_repeat_count > available_repeats:
            raise RuntimeError("repeat limit exceeds the selected question repeats")
        selected_repeat_indices = tuple(range(1, selected_repeat_count + 1))
        if heldout_quarantine_questions:
            global_question_end = manifest_question_start + available_question_count
            if heldout_quarantine_questions > global_question_end:
                raise RuntimeError("held-out quarantine exceeds the manifest question count")
            if (
                manifest_question_start > 0
                and heldout_quarantine_questions != manifest_question_start
            ):
                raise RuntimeError("held-out quarantine must equal the manifest question start")
            if question_start < heldout_quarantine_questions < question_end:
                raise RuntimeError(
                    "question selection cannot mix quarantined exposed held-out "
                    "questions with evaluation questions"
                )
            heldout_classification = (
                "exposed-heldout-quarantine"
                if question_end <= heldout_quarantine_questions
                else "unexposed-heldout-evaluation"
            )
        scoped = []
        for repeat in selected_repeat_indices:
            for group in selected_groups:
                pair = by_group_repeat.get((group, repeat))
                if not pair:
                    raise RuntimeError("selected cohort has a missing question/repeat combination")
                scoped.extend(pair)

    if condition != "both":
        scoped = [task for task in scoped if task.get("condition") == condition]

    tasks_by_pair: dict[str, list[dict[str, object]]] = {}
    for task in scoped:
        tasks_by_pair.setdefault(str(task["pair_id"]), []).append(task)
    for pair_tasks in tasks_by_pair.values():
        conditions = [str(task.get("condition")) for task in pair_tasks]
        if any(condition in {"baseline", "treatment"} for condition in conditions):
            development_pair = all(
                task.get("condition") == "treatment"
                and task.get("partition") == "train"
                and task.get("phase") == "development"
                for task in pair_tasks
            )
            expected = (
                ["treatment"]
                if development_pair
                else ["baseline", "treatment"]
                if condition == "both"
                else [condition]
            )
            if sorted(conditions) != expected:
                if condition == "both":
                    raise RuntimeError(
                        "an evaluation scope must contain complete baseline/treatment pairs"
                    )
                raise RuntimeError(
                    "an evaluation scope must contain the requested condition exactly once"
                )
    if not scoped:
        raise RuntimeError("task scope is empty")
    return TaskScope(
        tasks=tuple(scoped),
        selector=selector,
        manifest_question_start=manifest_question_start,
        question_start=question_start,
        question_limit=question_limit,
        available_question_count=available_question_count,
        selected_question_groups=tuple(selected_groups),
        selected_repeat_indices=selected_repeat_indices,
        condition=condition,
        heldout_quarantine_questions=heldout_quarantine_questions,
        heldout_classification=heldout_classification,
    )


def _select_task_scope(
    tasks: list[dict[str, object]],
    *,
    limit: int | None,
    manifest_question_start: int = 0,
    question_start: int = 0,
    question_limit: int | None,
    repeat_limit: int | None,
    condition: ConditionScope,
    heldout_quarantine_questions: int = 0,
    allow_descriptive_mixed_heldout: bool = False,
) -> list[dict[str, object]]:
    """Compatibility wrapper returning the frozen task membership only."""

    return list(
        _build_task_scope(
            tasks,
            limit=limit,
            manifest_question_start=manifest_question_start,
            question_start=question_start,
            question_limit=question_limit,
            repeat_limit=repeat_limit,
            condition=condition,
            heldout_quarantine_questions=heldout_quarantine_questions,
            allow_descriptive_mixed_heldout=allow_descriptive_mixed_heldout,
        ).tasks
    )


def _build_manifest_task_scope(
    manifest: Mapping[str, object],
    tasks: list[dict[str, object]],
    *,
    limit: int | None,
    question_start: int,
    question_limit: int | None,
    repeat_limit: int | None,
    condition: ConditionScope,
) -> TaskScope:
    """Build a scope using the manifest's global primary/quarantine coordinates."""

    kind = manifest.get("kind")
    primary_start, _primary_count = (
        demo.primary_evaluation_window(manifest) if kind == "full" else (None, None)
    )
    return _build_task_scope(
        tasks,
        limit=limit,
        manifest_question_start=primary_start or 0,
        question_start=question_start,
        question_limit=question_limit,
        repeat_limit=repeat_limit,
        condition=condition,
        heldout_quarantine_questions=_manifest_heldout_quarantine_questions(manifest),
        allow_descriptive_mixed_heldout=(kind == "full" and primary_start is None),
    )


def _pending_tasks_for_stage(
    scoped: list[dict[str, object]],
    states: Mapping[str, str],
    stage: BatchStage,
) -> list[dict[str, object]]:
    """Select stage work without changing the immutable cohort membership."""

    if stage == "generation":
        return [
            task
            for task in scoped
            if states.get(str(task["task_id"])) in {None, "generation_started", "failed"}
        ]
    if stage == "score":
        invalid = [
            str(task["task_id"])
            for task in scoped
            if states.get(str(task["task_id"]))
            not in {
                "completed",
                "generation_completed",
                "failed",
                "score_retry_started",
                "scored",
                "evidence_imported",
            }
        ]
        if invalid:
            raise RuntimeError(
                f"score stage requires ledgered generations for every selected task: {invalid[0]}"
            )
    return [task for task in scoped if states.get(str(task["task_id"])) != "completed"]


def _scoped_pending_tasks(
    tasks: list[dict[str, object]],
    states: Mapping[str, str],
    limit: int | None,
) -> list[dict[str, object]]:
    """Backward-compatible manifest-prefix scope used by the full protocol."""

    scoped = _select_task_scope(
        tasks,
        limit=limit,
        question_limit=None,
        repeat_limit=None,
        condition="both",
    )
    return _pending_tasks_for_stage(scoped, states, "all")


def _validate_manifest_generation_workers(
    manifest: Mapping[str, object],
    *,
    stage: BatchStage,
    workers: int,
) -> None:
    """Enforce the frozen generation ceiling without limiting score-only work."""

    limit = demo.manifest_max_generation_concurrency(manifest)
    if stage != "score" and workers > limit:
        raise RuntimeError(
            f"--workers {workers} exceeds manifest max_generation_concurrency {limit} "
            f"for stage {stage!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--population-report", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--switchyard", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--tooluniverse", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--repeat-limit", type=int)
    parser.add_argument(
        "--condition",
        choices=("both", "baseline", "treatment"),
        default="both",
        help="treatment-only is restricted to exposed descriptive canary generation",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "generation", "score"),
        default="all",
        help="defer judging with generation, or score hash-bound generations later",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true")
    parser.add_argument(
        "--development-generation-timeout-seconds",
        type=int,
        help=(
            "development-only per-generation wall-clock deadline; requires "
            "--max-generation-attempts 1 and is never protocol evidence"
        ),
    )
    parser.add_argument(
        "--canary-generation-timeout-seconds",
        type=int,
        help=(
            "120..900 second deadline allowed only for a reviewed q2/q5/q7 "
            "treatment-only descriptive canary with --max-generation-attempts 1"
        ),
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=3,
        help=(
            "cumulative cap for attempts whose native session proves zero executor "
            "requests; completed/paid runs are never retried"
        ),
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise RuntimeError("workers must be between 1 and 16")
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("limit must be positive")
    if args.question_start < 0:
        raise RuntimeError("question-start must not be negative")
    if args.question_limit is not None and args.question_limit < 1:
        raise RuntimeError("question-limit must be positive")
    if args.repeat_limit is not None and args.repeat_limit < 1:
        raise RuntimeError("repeat-limit must be positive")
    if args.max_generation_attempts < 1 or args.max_generation_attempts > 5:
        raise RuntimeError("max-generation-attempts must be between 1 and 5")
    manifest = demo._read_json_object(args.manifest.absolute(), "experiment manifest")
    kind = manifest.get("kind")
    if kind not in {"donor", "development", "pilot", "full"}:
        raise RuntimeError(f"invalid manifest kind: {kind!r}")
    _validate_manifest_generation_workers(
        manifest,
        stage=args.stage,
        workers=args.workers,
    )
    _validate_single_arm_execution(
        manifest,
        condition=args.condition,
        stage=args.stage,
        limit=args.limit,
        question_start=args.question_start,
        question_limit=args.question_limit,
        max_generation_attempts=args.max_generation_attempts,
    )
    generation_timeout_seconds, generation_timeout_policy = _resolve_generation_timeout(
        args.development_generation_timeout_seconds,
        canary_requested=args.canary_generation_timeout_seconds,
        single_arm_canary=args.condition == "treatment",
        kind=kind,
        max_generation_attempts=args.max_generation_attempts,
    )
    runtime = Runtime(
        experiment_root=args.experiment_root.absolute(),
        switchyard=args.switchyard.absolute(),
        codex=args.codex.absolute(),
        tooluniverse=args.tooluniverse.absolute(),
        profile=args.profile.absolute(),
        doctor=args.doctor.absolute(),
        candidate=args.candidate.absolute() if args.candidate else None,
        generation_timeout_seconds=generation_timeout_seconds,
        generation_timeout_policy=generation_timeout_policy,
    )
    dataset = demo.load_manifest_dataset(args.dataset.absolute(), manifest)
    split = demo.create_manifest_split(dataset, manifest)
    candidate = None
    if kind != "donor":
        if runtime.candidate is None:
            raise RuntimeError(f"{kind} manifest requires --candidate")
        candidate = validate_candidate_skill(runtime.candidate, NAMESPACE)
    expected = demo.build_reproducible_manifest_from_supplied(
        supplied=manifest,
        dataset=dataset,
        split_manifest=split,
        candidate=candidate,
        routing_profile=runtime.profile,
        switchyard_bin=runtime.switchyard,
        codex_bin=runtime.codex,
        tooluniverse_bin=runtime.tooluniverse,
        doctor_report=runtime.doctor,
        population_report=args.population_report.absolute() if args.population_report else None,
    )
    if manifest != expected:
        raise RuntimeError("manifest differs from the current pinned inputs")

    capture = runtime.experiment_root / str(manifest["manifest_id"])
    capture.mkdir(parents=True, exist_ok=True)
    lock_path = capture / "batch.lock"
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_descriptor)
        raise RuntimeError(f"another batch driver holds {lock_path}") from exc
    ledger = demo.ResumableLedger(capture / "ledger.jsonl", manifest)
    _validate_generation_timeout_history(
        ledger,
        timeout_seconds=runtime.generation_timeout_seconds,
        timeout_policy=runtime.generation_timeout_policy,
    )
    tasks = [dict(task) for task in manifest["tasks"]]
    task_scope = _build_manifest_task_scope(
        manifest,
        tasks,
        limit=args.limit,
        question_start=args.question_start,
        question_limit=args.question_limit,
        repeat_limit=args.repeat_limit,
        condition=args.condition,
    )
    scoped = list(task_scope.tasks)
    scope_metadata = task_scope.metadata(manifest["manifest_id"])
    states = ledger.states()
    pair_positions: dict[str, int] = {}
    for task in tasks:
        pair_id = str(task["pair_id"])
        if pair_id not in pair_positions:
            pair_positions[pair_id] = len(pair_positions)
    interrupted = [
        task for task in scoped if states.get(str(task["task_id"])) == "generation_started"
    ]
    if interrupted and not args.recover_interrupted:
        raise RuntimeError(
            "interrupted generation requires --recover-interrupted before resume: "
            f"{interrupted[0]['task_id']}"
        )
    if interrupted and args.recover_interrupted:
        for task in interrupted:
            task_id = str(task["task_id"])
            recorded = _record_generation_failure(
                manifest=manifest,
                task=task,
                capture=capture,
                ledger=ledger,
                error=InterruptedError(
                    "previous batch stopped before generation completion was ledgered"
                ),
                retry_exhausted=False,
            )
            if recorded.terminal_result_path is not None:
                _finish_terminal_generation_failure(
                    capture=capture,
                    task=task,
                    ledger=ledger,
                    failure=recorded.ledger_record,
                )
                states[task_id] = "completed"
            else:
                states[task_id] = "failed"
    if args.stage != "generation":
        interrupted_scores: list[dict[str, object]] = []
        for task in scoped:
            task_id = str(task["task_id"])
            state = states.get(task_id)
            result_path = _scored_result_path(capture, task_id)
            judge_path = capture / "judge" / task_id
            if result_path.is_file() and not result_path.is_symlink():
                continue
            if state == "score_retry_started" or (
                state == "generation_completed" and judge_path.exists()
            ):
                interrupted_scores.append(task)
        if interrupted_scores and not args.recover_interrupted:
            raise RuntimeError(
                "interrupted scoring requires --recover-interrupted and --retry-failed: "
                f"{interrupted_scores[0]['task_id']}"
            )
        if interrupted_scores and not args.retry_failed:
            raise RuntimeError("interrupted scoring recovery also requires --retry-failed")
        for task in interrupted_scores:
            task_id = str(task["task_id"])
            generation = _load_completed_generation(ledger, task_id)
            _record_score_failure(
                manifest=manifest,
                task=task,
                capture=capture,
                ledger=ledger,
                error=InterruptedError(
                    "previous batch stopped before scoring was durably committed"
                ),
                generation=generation,
            )
            states[task_id] = "failed"
    pending = _pending_tasks_for_stage(scoped, states, args.stage)
    recovered_scores: list[str] = []
    if args.stage != "generation":
        for task in pending:
            task_id = str(task["task_id"])
            state = states.get(task_id)
            result_path = _scored_result_path(capture, task_id)
            if state not in {
                "generation_completed",
                "score_retry_started",
                "scored",
                "evidence_imported",
            }:
                continue
            if not result_path.is_file() or result_path.is_symlink():
                if state in {"scored", "evidence_imported"}:
                    raise RuntimeError(f"partial score state lacks its immutable result: {task_id}")
                continue
            generation = _load_completed_generation(ledger, task_id)
            _finish_partial_scored_result(
                manifest=manifest,
                task=task,
                capture=capture,
                ledger=ledger,
                generation=generation,
            )
            states[task_id] = "completed"
            recovered_scores.append(task_id)
        pending = [task for task in pending if states.get(str(task["task_id"])) != "completed"]
    if args.stage == "generation":
        pending = [
            task
            for task in pending
            if states.get(str(task["task_id"])) != "failed"
            or _latest_failure_stage(ledger, str(task["task_id"])) == "generation"
        ]
    unexpected: list[tuple[str, object]] = []
    failed_stages: dict[str, str] = {}
    terminal_failures: dict[str, dict[str, object]] = {}
    exhausted_failures: dict[str, dict[str, object]] = {}
    for task in pending:
        task_id = str(task["task_id"])
        state = states.get(task_id)
        if state in {None, "generation_completed"}:
            continue
        if state == "failed" and args.retry_failed:
            failure = _latest_failure(ledger, task_id)
            failure_payload = failure.get("payload")
            stage = failure_payload.get("stage") if isinstance(failure_payload, dict) else None
            if stage in {"generation", "score-import"}:
                if args.stage == "generation" and stage != "generation":
                    unexpected.append((task_id, f"failed-{stage}-outside-generation-stage"))
                    continue
                if args.stage == "score" and stage != "score-import":
                    unexpected.append((task_id, f"failed-{stage}-outside-score-stage"))
                    continue
                if stage == "generation" and isinstance(failure_payload, dict):
                    if failure_payload.get("terminal") is True:
                        terminal_failures[task_id] = failure
                        continue
                    if failure_payload.get("retry_permitted") is not True:
                        unexpected.append((task_id, "failed-manual-review"))
                        continue
                    if _generation_attempt_count(ledger, task_id) >= args.max_generation_attempts:
                        exhausted_failures[task_id] = failure
                        continue
                failed_stages[task_id] = stage
                continue
        unexpected.append((task_id, state))
    if unexpected:
        raise RuntimeError(f"manual recovery required for non-resumable states: {unexpected[:3]}")
    worker_datasets = _worker_datasets(
        args.dataset.absolute(),
        manifest,
        str(manifest["manifest_id"]),
        args.workers,
    )
    completed_before = sum(state == "completed" for state in states.values())
    print(
        json.dumps(
            {
                "event": "batch_started",
                "manifest_id": manifest["manifest_id"],
                "kind": kind,
                "stage": args.stage,
                "condition": args.condition,
                "question_start": args.question_start,
                "question_limit": args.question_limit,
                "repeat_limit": args.repeat_limit,
                "scope": scope_metadata,
                "workers": args.workers,
                "generation_timeout_seconds": runtime.generation_timeout_seconds,
                "generation_timeout_policy": runtime.generation_timeout_policy,
                "completed_before": completed_before,
                "scope_size": len(scoped),
                "selected": len(pending),
                "recovered_scores": recovered_scores,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chunk_index, chunk in enumerate(
            _pair_safe_chunks(
                pending,
                args.workers,
                pair_positions=pair_positions,
            ),
            start=1,
        ):
            generations: dict[str, demo.GenerationResult] = {}
            new_tasks = (
                []
                if args.stage == "score"
                else [
                    task
                    for task in chunk
                    if states.get(str(task["task_id"])) is None
                    or failed_stages.get(str(task["task_id"])) == "generation"
                ]
            )
            for task in chunk:
                task_id = str(task["task_id"])
                if task_id in terminal_failures:
                    _finish_terminal_generation_failure(
                        capture=capture,
                        task=task,
                        ledger=ledger,
                        failure=terminal_failures[task_id],
                    )
                    states[task_id] = "completed"
                elif task_id in exhausted_failures:
                    _finish_exhausted_generation_failure(
                        manifest=manifest,
                        capture=capture,
                        task=task,
                        ledger=ledger,
                        failure=exhausted_failures[task_id],
                        max_attempts=args.max_generation_attempts,
                    )
                    states[task_id] = "completed"
                elif states.get(task_id) == "generation_completed":
                    generations[task_id] = _load_completed_generation(ledger, task_id)
                elif failed_stages.get(task_id) == "score-import":
                    generation = _load_completed_generation(ledger, task_id)
                    failure = _latest_failure(ledger, task_id)
                    judge_quarantine = _quarantine_failed_judge(
                        capture=capture,
                        task_id=task_id,
                        failure=failure,
                    )
                    recovery_payload: dict[str, object] = {
                        "recovery": "reuse-generation-after-score-failure",
                        "failed_record_sha256": failure["record_sha256"],
                        "generation_record_sha256": _latest_generation_record(ledger, task_id)[
                            "record_sha256"
                        ],
                    }
                    if judge_quarantine is not None:
                        recovery_payload["quarantined_judge_attempt"] = str(judge_quarantine)
                    ledger.append(
                        task_id,
                        "score_retry_started",
                        {
                            **recovery_payload,
                            "generation_path": str(generation.generation_path),
                            "generation_sha256": demo._sha256_file(generation.generation_path),
                            "artifact_sha256": dict(generation.artifact_sha256),
                        },
                    )
                    states[task_id] = "score_retry_started"
                    generations[task_id] = generation

            terminalized: list[str] = []
            manual_failures: list[str] = []
            generation_queue = list(new_tasks)
            for _round in range(1, args.max_generation_attempts + 1):
                futures: dict[
                    Future[demo.GenerationResult],
                    tuple[dict[str, object], int, int],
                ] = {}
                for worker_index, task in enumerate(generation_queue):
                    task_id = str(task["task_id"])
                    prior_attempts = _generation_attempt_count(ledger, task_id)
                    if prior_attempts >= args.max_generation_attempts:
                        raise RuntimeError(
                            f"generation retry budget was already exhausted: {task_id}"
                        )
                    generation_attempt = prior_attempts + 1
                    start_payload: dict[str, object] = {
                        "generation_attempt": generation_attempt,
                        "wall_clock_timeout_seconds": (runtime.generation_timeout_seconds),
                        "timeout_policy": runtime.generation_timeout_policy,
                    }
                    if states.get(task_id) == "failed":
                        failure = _latest_failure(ledger, task_id)
                        prior_quarantine = _quarantine_failed_generation(
                            capture=capture,
                            task=task,
                            failure=failure,
                        )
                        start_payload.update(
                            {
                                "recovery": "retry-failed-generation",
                                "quarantined_attempt": str(prior_quarantine),
                                "failed_record_sha256": failure["record_sha256"],
                            }
                        )
                    elif states.get(task_id) is not None:
                        raise RuntimeError(
                            f"generation queue has invalid state for {task_id}: "
                            f"{states.get(task_id)!r}"
                        )
                    ledger.append(task_id, "generation_started", start_payload)
                    states[task_id] = "generation_started"
                    future = pool.submit(
                        _generation,
                        manifest,
                        task_id,
                        worker_datasets[worker_index],
                        split,
                        runtime,
                        capture,
                    )
                    futures[future] = (task, worker_index, generation_attempt)
                retry_queue: list[dict[str, object]] = []
                for future in as_completed(futures):
                    task, _worker_index, generation_attempt = futures[future]
                    task_id = str(task["task_id"])
                    try:
                        generation = future.result()
                    except BaseException as exc:
                        recorded = _record_generation_failure(
                            manifest=manifest,
                            task=task,
                            capture=capture,
                            ledger=ledger,
                            error=exc,
                            retry_exhausted=(generation_attempt >= args.max_generation_attempts),
                        )
                        completed_draw = recorded.completed_model_draw
                        if recorded.terminal_result_path is not None:
                            terminal_quarantine: Path | None = _finish_terminal_generation_failure(
                                capture=capture,
                                task=task,
                                ledger=ledger,
                                failure=recorded.ledger_record,
                            )
                            states[task_id] = "completed"
                            terminalized.append(task_id)
                            event = "generation_terminal_error"
                        elif recorded.manual_review:
                            states[task_id] = "failed"
                            manual_failures.append(task_id)
                            terminal_quarantine = None
                            event = "generation_manual_review"
                        elif recorded.retry_permitted:
                            states[task_id] = "failed"
                            retry_queue.append(task)
                            terminal_quarantine = None
                            event = "generation_retryable_error"
                        else:
                            raise RuntimeError(
                                f"generation failure classification is incomplete: {task_id}"
                            ) from exc
                        print(
                            json.dumps(
                                {
                                    "event": event,
                                    "task_id": task_id,
                                    "generation_attempt": generation_attempt,
                                    "completed_model_draw": completed_draw,
                                    "retry_permitted": recorded.retry_permitted,
                                    "manual_review": recorded.manual_review,
                                    "quarantined_attempt": (
                                        str(terminal_quarantine)
                                        if terminal_quarantine is not None
                                        else None
                                    ),
                                    "error": repr(exc),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
                    generations[task_id] = generation
                    ledger.append(
                        task_id,
                        "generation_completed",
                        {
                            "generation_path": str(generation.generation_path),
                            "generation_sha256": demo._sha256_file(generation.generation_path),
                            "artifact_sha256": dict(generation.artifact_sha256),
                        },
                    )
                    states[task_id] = "generation_completed"
                    print(
                        json.dumps(
                            {
                                "event": "generation_completed",
                                "task_id": task_id,
                                "answer": generation.answer,
                                "requests": generation.stats["total_requests"],
                                "tokens": generation.stats["total_tokens"]["total"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if not retry_queue:
                    break
                print(
                    json.dumps(
                        {
                            "event": "generation_retry_scheduled",
                            "task_ids": [str(task["task_id"]) for task in retry_queue],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                generation_queue = retry_queue

            if args.stage == "generation":
                if manual_failures:
                    raise RuntimeError(
                        f"task failures require review: generation={manual_failures}"
                    )
                print(
                    json.dumps(
                        {
                            "event": "generation_chunk_completed",
                            "chunk": chunk_index,
                            "generation_completed_total": sum(
                                state == "generation_completed" for state in states.values()
                            ),
                            "completed_total": sum(
                                state == "completed" for state in states.values()
                            ),
                            "terminal_generation_errors": len(terminalized),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            score_futures: dict[Future[demo.ScoredGeneration], tuple[dict[str, object], int]] = {}
            score_tasks = [task for task in chunk if str(task["task_id"]) in generations]
            for worker_index, task in enumerate(score_tasks):
                task_id = str(task["task_id"])
                score_futures[
                    pool.submit(
                        _score,
                        manifest,
                        task,
                        generations[task_id],
                        worker_datasets[worker_index],
                        runtime,
                        capture,
                        worker_index,
                    )
                ] = (task, worker_index)
            score_failures: list[str] = []
            for score_future in as_completed(score_futures):
                task, _worker_index = score_futures[score_future]
                task_id = str(task["task_id"])
                generation = generations[task_id]
                try:
                    scored = score_future.result()
                except BaseException as exc:
                    _record_score_failure(
                        manifest=manifest,
                        task=task,
                        capture=capture,
                        ledger=ledger,
                        error=exc,
                        generation=generation,
                    )
                    states[task_id] = "failed"
                    score_failures.append(task_id)
                    print(
                        json.dumps(
                            {"event": "score_failed", "task_id": task_id, "error": repr(exc)}
                        ),
                        flush=True,
                    )
                    continue
                _commit_scored_result(
                    manifest=manifest,
                    task=task,
                    capture=capture,
                    ledger=ledger,
                    scored=scored,
                )
                states[task_id] = "completed"
                print(
                    json.dumps(
                        {
                            "event": "task_completed",
                            "task_id": task_id,
                            "score": scored.outcome.score,
                            "judge_result": scored.outcome.judge_result,
                            "evidence_id": scored.evidence.evidence_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if manual_failures or score_failures:
                raise RuntimeError(
                    "task failures require review: "
                    f"generation={manual_failures}, score={score_failures}"
                )
            print(
                json.dumps(
                    {
                        "event": "chunk_completed",
                        "chunk": chunk_index,
                        "completed_total": sum(state == "completed" for state in states.values()),
                        "terminal_generation_errors": len(terminalized),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "batch_completed",
                "manifest_id": manifest["manifest_id"],
                "stage": args.stage,
                "condition": args.condition,
                "question_start": args.question_start,
                "question_limit": args.question_limit,
                "repeat_limit": args.repeat_limit,
                "scope": scope_metadata,
                "completed_total": sum(state == "completed" for state in states.values()),
                "generation_completed_total": sum(
                    state == "generation_completed" for state in states.values()
                ),
                "selected": len(pending),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    os.close(lock_descriptor)


if __name__ == "__main__":
    main()

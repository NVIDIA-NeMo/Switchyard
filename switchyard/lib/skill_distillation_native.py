# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import finalized native sessions as immutable TrialQA evidence."""

import hashlib
import json
import math
import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from switchyard.lib.skill_distillation_store import (
    SKILL_DISTILLATION_SCHEMA_VERSION,
    SkillDistillationStore,
)

NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION = 1
NATIVE_TRIALQA_SOURCE_KIND = "switchyard_native_session"
_NATIVE_EVIDENCE_ID = re.compile(r"native-[0-9a-f]{32}\Z")
_EVENT_KINDS = {"message", "tool_call", "tool_result", "final_output"}
_TRIALQA_EXECUTOR_ROUTE = "sd-executor"
_TRIALQA_EXECUTOR_MODEL = "nvidia/nvidia/nemotron-3-ultra"
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

JsonObject = dict[str, Any]


class NativeTrialQAEvidenceError(RuntimeError):
    """Raised when a native session cannot become lossless TrialQA evidence."""


class NativeTrialQAEvidenceConflictError(NativeTrialQAEvidenceError):
    """Raised when immutable native evidence has changed or is unsafe."""


@dataclass(frozen=True)
class NativeTrialQAEvidenceImportResult:
    """Stable identity and path produced by one native evidence import."""

    evidence_id: str
    evidence_path: Path
    imported: bool


@dataclass(frozen=True)
class _Artifact:
    name: str
    path: PurePosixPath
    content: bytes

    @property
    def marker(self) -> JsonObject:
        return {
            "path": self.path.as_posix(),
            "sha256": f"sha256:{hashlib.sha256(self.content).hexdigest()}",
            "size_bytes": len(self.content),
        }


@dataclass(frozen=True)
class _PreparedEvidence:
    evidence_id: str
    document: JsonObject
    manifest: JsonObject
    artifacts: tuple[_Artifact, ...]


def import_native_trialqa_evidence(
    session_dir: Path,
    *,
    namespace: str,
    task: Mapping[str, Any],
    outcome: Mapping[str, Any],
    run: Mapping[str, Any],
    project_dir: Path | None = None,
) -> NativeTrialQAEvidenceImportResult:
    """Publish one completed native session as content-addressed TrialQA evidence.

    Task identity, verifier outcome, and run identity are required inputs rather
    than inferred from model text. Re-importing an identical snapshot is a no-op;
    any mutation of an existing evidence bundle is a conflict.
    """

    normalized_task = _normalize_task(task)
    normalized_outcome = _normalize_outcome(outcome)
    normalized_run = _normalize_run(run)
    project_root = (project_dir or Path.cwd()).expanduser().absolute()
    store = SkillDistillationStore(namespace, project_root)
    session_dir = session_dir.expanduser().absolute()

    with store.exclusive_lock():
        session_id = store.validate_session_evidence(session_dir)
        prepared = _prepare_evidence(
            session_dir,
            session_id=session_id,
            namespace=store.namespace,
            task=normalized_task,
            outcome=normalized_outcome,
            run=normalized_run,
        )
        evidence_path = store.evidence_path / prepared.evidence_id
        if evidence_path.is_symlink():
            raise NativeTrialQAEvidenceConflictError(
                f"Refusing symlinked native evidence destination: {evidence_path}"
            )
        if evidence_path.exists():
            _validate_existing_evidence(evidence_path, prepared)
            return NativeTrialQAEvidenceImportResult(
                evidence_id=prepared.evidence_id,
                evidence_path=evidence_path,
                imported=False,
            )

        _write_evidence_directory(evidence_path, prepared)
        validate_native_trialqa_evidence_directory(
            evidence_path,
            expected_evidence_id=prepared.evidence_id,
        )
        return NativeTrialQAEvidenceImportResult(
            evidence_id=prepared.evidence_id,
            evidence_path=evidence_path,
            imported=True,
        )


def validate_native_trialqa_evidence_directory(
    path: Path,
    *,
    expected_evidence_id: str,
) -> None:
    """Validate one immutable native TrialQA evidence directory without mutation."""

    if _NATIVE_EVIDENCE_ID.fullmatch(expected_evidence_id) is None:
        raise NativeTrialQAEvidenceError(
            f"Invalid native TrialQA evidence ID: {expected_evidence_id!r}"
        )
    evidence_dir = path.expanduser().absolute()
    _reject_symlink_components(evidence_dir, label="native evidence directory")
    if not evidence_dir.is_dir():
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence path is not a real directory: {evidence_dir}"
        )
    if evidence_dir.name != expected_evidence_id:
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence directory name does not match {expected_evidence_id}"
        )

    manifest_path = evidence_dir / "manifest.json"
    evidence_path = evidence_dir / "evidence.json"
    manifest = _read_json_object_file(
        manifest_path,
        label="native evidence manifest",
    )
    evidence = _read_json_object_file(
        evidence_path,
        label="native evidence document",
    )
    if (
        manifest.get("schema_version") != NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION
        or manifest.get("kind") != "switchyard_skill_distillation_evidence"
        or manifest.get("source_type") != NATIVE_TRIALQA_SOURCE_KIND
        or manifest.get("evidence_id") != expected_evidence_id
    ):
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence manifest identity is invalid: {manifest_path}"
        )

    evidence_marker = _mapping(manifest.get("evidence"))
    if evidence_marker.get("path") != "evidence.json":
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence manifest has an invalid evidence path: {manifest_path}"
        )
    _validate_file_marker(evidence_path, evidence_marker)
    _validate_evidence_document(evidence, expected_evidence_id=expected_evidence_id)

    evidence_seed = dict(evidence)
    evidence_seed.pop("evidence_id", None)
    content_evidence_id = f"native-{_digest_json(evidence_seed)[:32]}"
    if content_evidence_id != expected_evidence_id:
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence ID does not match document content: {evidence_path}"
        )

    artifacts = _mapping(manifest.get("artifacts"))
    if not artifacts or artifacts != _mapping(evidence.get("artifacts")):
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence artifact manifest is missing or inconsistent: {manifest_path}"
        )
    expected_files = {"manifest.json", "evidence.json"}
    expected_directories: set[str] = set()
    for name, raw_marker in artifacts.items():
        marker = _mapping(raw_marker)
        relative = _artifact_path(marker.get("path"), name=name)
        relative_text = relative.as_posix()
        if relative_text in expected_files:
            raise NativeTrialQAEvidenceConflictError(
                f"Native artifact paths are not unique: {relative_text}"
            )
        expected_files.add(relative_text)
        expected_directories.update(
            parent.as_posix() for parent in relative.parents if parent != PurePosixPath(".")
        )
        artifact_path = evidence_dir.joinpath(*relative.parts)
        _validate_file_marker(artifact_path, marker)
    _validate_directory_shape(
        evidence_dir,
        expected_files=expected_files,
        expected_directories=expected_directories,
    )


def _prepare_evidence(
    session_dir: Path,
    *,
    session_id: str,
    namespace: str,
    task: JsonObject,
    outcome: JsonObject,
    run: JsonObject,
) -> _PreparedEvidence:
    session_bytes = _read_regular_file(
        session_dir / "session.json",
        label="native session metadata",
    )
    turns_bytes = _read_regular_file(
        session_dir / "turns.jsonl",
        label="native session trajectory",
    )
    stats_bytes = _read_regular_file(
        session_dir / "stats.json",
        label="native session stats",
    )
    session = _read_json_object_bytes(session_bytes, label="native session metadata")
    turns = _read_json_lines_bytes(turns_bytes, label="native session trajectory")
    stats = _read_json_object_bytes(stats_bytes, label="native session stats")
    _validate_session_snapshot(
        session,
        turns,
        turns_bytes=turns_bytes,
        session_id=session_id,
        namespace=namespace,
    )
    _validate_trialqa_bindings(
        session=session,
        turns=turns,
        stats=stats,
        task=task,
        outcome=outcome,
        run=run,
    )
    events = _normalize_events(turns)

    task_bytes = _json_document(task).encode("utf-8")
    outcome_bytes = _json_document(outcome).encode("utf-8")
    run_bytes = _json_document(run).encode("utf-8")
    artifacts = (
        _Artifact("session", PurePosixPath("raw/session.json"), session_bytes),
        _Artifact("trajectory", PurePosixPath("raw/turns.jsonl"), turns_bytes),
        _Artifact("stats", PurePosixPath("raw/stats.json"), stats_bytes),
        _Artifact("task", PurePosixPath("raw/task.json"), task_bytes),
        _Artifact("outcome", PurePosixPath("raw/outcome.json"), outcome_bytes),
        _Artifact("run", PurePosixPath("raw/run.json"), run_bytes),
    )
    artifact_markers = {artifact.name: artifact.marker for artifact in artifacts}
    served_models = _ordered_text_values(turns, "served_model")
    execution: JsonObject = {
        **run,
        "run_id": run["run_id"],
        "phase": run["phase"],
        "model": run.get("model") or session.get("display_model"),
        "harness": run.get("harness") or session.get("launch_target"),
        "route": run.get("route") or session.get("strategy_summary"),
        "session_id": session_id,
        "started_at": session.get("started_at"),
        "ended_at": session.get("ended_at"),
        "served_models": served_models,
    }
    source: JsonObject = {
        "kind": NATIVE_TRIALQA_SOURCE_KIND,
        "session_id": session_id,
        "session_path": f"sessions/{session_id}",
        "trajectory_sha256": session["trajectory_sha256"],
        "artifact_hashes": {name: marker["sha256"] for name, marker in artifact_markers.items()},
    }
    evidence_seed: JsonObject = {
        "schema_version": NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION,
        "task": task,
        "execution": execution,
        "outcome": outcome,
        "source": source,
        "events": events,
        "artifacts": artifact_markers,
    }
    evidence_id = f"native-{_digest_json(evidence_seed)[:32]}"
    document = {"evidence_id": evidence_id, **evidence_seed}
    manifest = _evidence_manifest(evidence_id, document, artifact_markers)
    return _PreparedEvidence(
        evidence_id=evidence_id,
        document=document,
        manifest=manifest,
        artifacts=artifacts,
    )


def _normalize_task(value: Mapping[str, Any]) -> JsonObject:
    task = _normalize_json_object(value, label="task")
    task_id = _required_text(task.get("id"), field="task.id")
    question = _required_text(task.get("question"), field="task.question")
    condition = _required_text(task.get("condition"), field="task.condition")
    task_type = task.get("type", "trialqa")
    if task_type != "trialqa":
        raise NativeTrialQAEvidenceError("task.type must be 'trialqa'")
    task.update(
        {
            "id": task_id,
            "question": question,
            "condition": condition,
            "type": "trialqa",
        }
    )
    _validate_optional_index(task, "repeat_index", minimum=1)
    _validate_optional_index(task, "n_repeats", minimum=1)
    return task


def _normalize_outcome(value: Mapping[str, Any]) -> JsonObject:
    outcome = _normalize_json_object(value, label="outcome")
    score = _finite_number(outcome.get("score"), field="outcome.score")
    if not 0.0 <= score <= 1.0:
        raise NativeTrialQAEvidenceError("outcome.score must be between 0 and 1")
    outcome["score"] = score
    outcome["verifier"] = _required_text(
        outcome.get("verifier"),
        field="outcome.verifier",
    )
    if "raw_score" in outcome:
        outcome["raw_score"] = _finite_number(
            outcome["raw_score"],
            field="outcome.raw_score",
        )
    metrics = outcome.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            raise NativeTrialQAEvidenceError("outcome.metrics must be a JSON object")
        outcome["metrics"] = {
            name: _finite_number(metric, field=f"outcome.metrics.{name}")
            for name, metric in metrics.items()
        }
    return outcome


def _normalize_run(value: Mapping[str, Any]) -> JsonObject:
    run = _normalize_json_object(value, label="run")
    run["run_id"] = _required_text(run.get("run_id"), field="run.run_id")
    run["phase"] = _required_text(run.get("phase"), field="run.phase")
    for field in ("model", "harness", "route"):
        if field in run and run[field] is not None:
            run[field] = _required_text(run[field], field=f"run.{field}")
    return run


def _validate_openai_transport_stats(stats: Mapping[str, object], total_requests: int) -> None:
    transport = stats.get("openai_transport")
    if not isinstance(transport, dict) or set(transport) != _OPENAI_TRANSPORT_FIELDS:
        raise NativeTrialQAEvidenceError(
            "native TrialQA stats lack exact OpenAI transport accounting"
        )
    counters: dict[str, int] = {}
    for field in _OPENAI_TRANSPORT_FIELDS - {"retry_token_sensitivity"}:
        value = transport.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise NativeTrialQAEvidenceError(f"native TrialQA OpenAI transport {field} is invalid")
        counters[field] = value
    sensitivity = transport.get("retry_token_sensitivity")
    if not isinstance(sensitivity, dict) or set(sensitivity) != _TOKEN_TOTAL_FIELDS:
        raise NativeTrialQAEvidenceError("native TrialQA retry token sensitivity is invalid")
    tokens: dict[str, int] = {}
    for field in _TOKEN_TOTAL_FIELDS:
        value = sensitivity.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise NativeTrialQAEvidenceError(
                f"native TrialQA retry token sensitivity {field} is invalid"
            )
        tokens[field] = value
    if tokens["total"] != tokens["prompt"] + tokens["completion"]:
        raise NativeTrialQAEvidenceError(
            "native TrialQA retry token sensitivity total is inconsistent"
        )
    retries = counters["null_eof_retries"]
    charges = counters["retry_usage_charges"]
    if (
        counters["physical_attempts"] != total_requests + retries
        or retries > total_requests
        or charges + counters["unpriced_null_eof_retries"] != retries
        or counters["unpriced_null_eof_retries"] != 0
        or (charges == 0 and any(tokens.values()))
        or (charges > 0 and tokens["total"] == 0)
    ):
        raise NativeTrialQAEvidenceError(
            "native TrialQA OpenAI transport accounting is inconsistent or unpriced"
        )


def _validate_trialqa_bindings(
    *,
    session: JsonObject,
    turns: Sequence[JsonObject],
    stats: JsonObject,
    task: JsonObject,
    outcome: JsonObject,
    run: JsonObject,
) -> None:
    """Bind caller metadata to the captured Ultra session before publication."""

    context = _mapping(session.get("run_context"))
    active = _mapping(session.get("active_skill"))
    if not context or not active:
        raise NativeTrialQAEvidenceError(
            "native TrialQA sessions require captured run_context and active_skill evidence"
        )
    if (
        session.get("status") != "completed"
        or session.get("exit_code") != 0
        or session.get("launch_target") != "codex"
        or session.get("display_model") != _TRIALQA_EXECUTOR_ROUTE
    ):
        raise NativeTrialQAEvidenceError(
            "native TrialQA evidence requires a completed Codex sd-executor session"
        )

    task_bindings = {
        "task_id": task.get("id"),
        "row_id": task.get("row_id", task.get("question_id")),
        "question_group_key": task.get("question_group_key"),
        "partition": task.get("partition"),
        "condition": task.get("condition"),
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
    }
    for field, expected in task_bindings.items():
        if expected is not None and context.get(field) != expected:
            raise NativeTrialQAEvidenceError(
                f"task metadata does not match session.run_context.{field}"
            )

    run_bindings = {
        "manifest_id": run.get("run_id"),
        "phase": run.get("phase"),
        "executor_model": run.get("executor_model", run.get("model")),
        "route": run.get("route"),
        "skill_loaded": run.get("skill_loaded"),
        "candidate_id": run.get("candidate_id"),
        "candidate_manifest_sha256": run.get("candidate_manifest_sha256"),
        "candidate_skill_sha256": run.get("candidate_skill_sha256"),
    }
    for field, expected in run_bindings.items():
        if context.get(field) != expected:
            raise NativeTrialQAEvidenceError(
                f"run metadata does not match session.run_context.{field}"
            )
    if (
        run.get("model") != _TRIALQA_EXECUTOR_MODEL
        or run.get("executor_model", run.get("model")) != _TRIALQA_EXECUTOR_MODEL
        or run.get("route") != _TRIALQA_EXECUTOR_ROUTE
    ):
        raise NativeTrialQAEvidenceError(
            "native TrialQA evidence must be attributed to the pinned Ultra executor route"
        )

    loaded = run.get("skill_loaded")
    if not isinstance(loaded, bool) or active.get("loaded") is not loaded:
        raise NativeTrialQAEvidenceError(
            "run skill_loaded does not match captured active_skill evidence"
        )
    active_bindings = {
        "candidate_id": run.get("candidate_id"),
        "manifest_sha256": run.get("candidate_manifest_sha256"),
        "skill_sha256": run.get("candidate_skill_sha256"),
    }
    for field, expected in active_bindings.items():
        if active.get(field) != expected:
            raise NativeTrialQAEvidenceError(
                f"run candidate metadata does not match active_skill.{field}"
            )
    candidate_values = tuple(active_bindings.values())
    if loaded:
        if not all(isinstance(value, str) and value for value in candidate_values):
            raise NativeTrialQAEvidenceError(
                "a loaded TrialQA skill requires complete candidate hash evidence"
            )
    elif any(value is not None for value in candidate_values):
        raise NativeTrialQAEvidenceError("an unskilled TrialQA run cannot name a candidate")

    outcome_bindings = {
        "row_id": task.get("row_id", task.get("question_id")),
        "question": task.get("question"),
        "question_group_key": task.get("question_group_key"),
        "partition": task.get("partition"),
        "condition": task.get("condition"),
        "repeat_index": task.get("repeat_index"),
        "n_repeats": task.get("n_repeats"),
        "task_name": task.get("id"),
    }
    for field, expected in outcome_bindings.items():
        if field in outcome and outcome.get(field) != expected:
            raise NativeTrialQAEvidenceError(
                f"outcome.{field} does not match trusted task metadata"
            )

    total_requests = stats.get("total_requests")
    total_errors = stats.get("total_errors")
    models = stats.get("models")
    if (
        isinstance(total_requests, bool)
        or not isinstance(total_requests, int)
        or total_requests < 1
        or isinstance(total_errors, bool)
        or not isinstance(total_errors, int)
        or total_errors < 0
        or total_errors >= total_requests
        or not isinstance(models, dict)
    ):
        raise NativeTrialQAEvidenceError(
            "native TrialQA stats contain invalid request or recovered-error counts"
        )
    for subsystem_name in ("classifier", "planner"):
        subsystem = stats.get(subsystem_name)
        if (
            not isinstance(subsystem, dict)
            or subsystem.get("total_requests") != 0
            or isinstance(subsystem.get("total_requests"), bool)
            or subsystem.get("total_errors") != 0
            or isinstance(subsystem.get("total_errors"), bool)
        ):
            raise NativeTrialQAEvidenceError(
                f"native TrialQA {subsystem_name} stats contain non-executor activity"
            )
    attempts_by_model: dict[str, int] = {}
    successful_calls = 0
    model_errors = 0
    for model, raw_model_stats in models.items():
        if not isinstance(model, str) or not isinstance(raw_model_stats, dict):
            raise NativeTrialQAEvidenceError("native TrialQA model stats are malformed")
        calls = raw_model_stats.get("calls", 0)
        errors = raw_model_stats.get("errors", 0)
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(errors, bool)
            or not isinstance(errors, int)
            or errors < 0
        ):
            raise NativeTrialQAEvidenceError("native TrialQA model stats are invalid")
        attempts = calls + errors
        if attempts:
            attempts_by_model[model] = attempts
        successful_calls += calls
        model_errors += errors
    if (
        attempts_by_model != {_TRIALQA_EXECUTOR_MODEL: total_requests}
        or successful_calls != total_requests - total_errors
        or model_errors != total_errors
        or len(turns) != successful_calls
    ):
        raise NativeTrialQAEvidenceError(
            "native TrialQA attempts were not exclusively attributed to the pinned Ultra "
            "model, or recovered errors and captured turns are inconsistent"
        )
    _validate_openai_transport_stats(stats, total_requests)

    for index, turn in enumerate(turns):
        request = _mapping(turn.get("request"))
        if (
            request.get("model") != _TRIALQA_EXECUTOR_ROUTE
            or turn.get("served_model") != _TRIALQA_EXECUTOR_MODEL
        ):
            raise NativeTrialQAEvidenceError(
                f"native TrialQA turn {index} does not attest the Ultra executor route"
            )
        expected_candidate = run.get("candidate_id") if loaded else None
        expected_manifest = run.get("candidate_manifest_sha256") if loaded else None
        if (
            turn.get("active_skill_candidate_id") != expected_candidate
            or turn.get("active_skill_manifest_sha256") != expected_manifest
        ):
            raise NativeTrialQAEvidenceError(
                f"native TrialQA turn {index} has inconsistent active-skill evidence"
            )


def _normalize_events(turns: Sequence[JsonObject]) -> list[JsonObject]:
    events: list[JsonObject] = []
    previous_messages: list[JsonObject] = []
    for turn_index, turn in enumerate(turns):
        request = turn.get("request")
        response = turn.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise NativeTrialQAEvidenceError(
                f"native session turn {turn_index} must contain request and response objects"
            )
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list):
            raise NativeTrialQAEvidenceError(
                f"native session turn {turn_index} request.messages must be a list"
            )
        request_messages = [
            _normalize_json_object(message, label=f"turn {turn_index} request message")
            for message in raw_messages
            if isinstance(message, Mapping)
        ]
        if len(request_messages) != len(raw_messages):
            raise NativeTrialQAEvidenceError(
                f"native session turn {turn_index} request messages must be objects"
            )
        common_prefix = 0
        while (
            common_prefix < len(previous_messages)
            and common_prefix < len(request_messages)
            and previous_messages[common_prefix] == request_messages[common_prefix]
        ):
            common_prefix += 1
        for message in request_messages[common_prefix:]:
            _append_message_events(
                events,
                message,
                turn=turn,
                turn_index=turn_index,
                source="request",
                final=False,
            )

        response_message = _response_message(response, turn_index=turn_index)
        _append_message_events(
            events,
            response_message,
            turn=turn,
            turn_index=turn_index,
            source="response",
            final=not bool(response_message.get("tool_calls")),
        )
        previous_messages = [*request_messages, response_message]

    if not events:
        raise NativeTrialQAEvidenceError("native session has no normalizable events")
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    return events


def _append_message_events(
    events: list[JsonObject],
    message: JsonObject,
    *,
    turn: JsonObject,
    turn_index: int,
    source: str,
    final: bool,
) -> None:
    role = _required_text(message.get("role"), field="message.role")
    metadata: JsonObject = {"turn_index": turn_index, "source": source}
    if source == "response":
        for field in ("served_model", "usage", "routing"):
            if field in turn:
                metadata[field] = turn[field]
    timestamp = turn.get("recorded_at")
    timestamp = timestamp if isinstance(timestamp, str) and timestamp.strip() else None

    if role == "tool":
        _append_event(
            events,
            kind="tool_result",
            payload=message,
            timestamp=timestamp,
            metadata=metadata,
        )
        return

    tool_calls = message.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise NativeTrialQAEvidenceError("assistant message tool_calls must be a list")
    has_content = message.get("content") not in (None, "", [])
    if has_content or not tool_calls:
        _append_event(
            events,
            kind="final_output" if final else "message",
            payload=message,
            timestamp=timestamp,
            metadata=metadata,
        )
    for call in tool_calls or []:
        if not isinstance(call, Mapping):
            raise NativeTrialQAEvidenceError("assistant tool calls must be JSON objects")
        _append_event(
            events,
            kind="tool_call",
            payload=_normalize_json_object(call, label="assistant tool call"),
            timestamp=timestamp,
            metadata=metadata,
        )


def _append_event(
    events: list[JsonObject],
    *,
    kind: str,
    payload: JsonObject,
    timestamp: str | None,
    metadata: JsonObject,
) -> None:
    event: JsonObject = {
        "sequence": len(events),
        "kind": kind,
        "payload": payload,
        "metadata": metadata,
    }
    if timestamp is not None:
        event["timestamp"] = timestamp
    events.append(event)


def _response_message(response: JsonObject, *, turn_index: int) -> JsonObject:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise NativeTrialQAEvidenceError(
            f"native session turn {turn_index} response has no completion choice"
        )
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise NativeTrialQAEvidenceError(
            f"native session turn {turn_index} response has no assistant message"
        )
    normalized = _normalize_json_object(message, label=f"turn {turn_index} response message")
    if normalized.get("role") != "assistant":
        raise NativeTrialQAEvidenceError(
            f"native session turn {turn_index} response message must be from assistant"
        )
    return normalized


def _validate_session_snapshot(
    session: JsonObject,
    turns: Sequence[JsonObject],
    *,
    turns_bytes: bytes,
    session_id: str,
    namespace: str,
) -> None:
    turn_count = session.get("turn_count")
    if (
        session.get("schema_version") != SKILL_DISTILLATION_SCHEMA_VERSION
        or session.get("session_id") != session_id
        or session.get("namespace") != namespace
        or session.get("status") != "completed"
        or isinstance(turn_count, bool)
        or not isinstance(turn_count, int)
        or turn_count <= 0
        or turn_count != len(turns)
    ):
        raise NativeTrialQAEvidenceError(
            "native session evidence must be completed, non-empty, and match its store"
        )
    for index, turn in enumerate(turns):
        if (
            turn.get("schema_version") != SKILL_DISTILLATION_SCHEMA_VERSION
            or turn.get("session_id") != session_id
            or turn.get("turn_index") != index
        ):
            raise NativeTrialQAEvidenceError(
                "native session turns must have contiguous indexes and matching identity"
            )
    actual_hash = f"sha256:{hashlib.sha256(turns_bytes).hexdigest()}"
    if session.get("trajectory_sha256") != actual_hash:
        raise NativeTrialQAEvidenceConflictError(
            "native session trajectory changed while evidence was being imported"
        )


def _validate_evidence_document(evidence: JsonObject, *, expected_evidence_id: str) -> None:
    if (
        evidence.get("schema_version") != NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION
        or evidence.get("evidence_id") != expected_evidence_id
        or _mapping(evidence.get("source")).get("kind") != NATIVE_TRIALQA_SOURCE_KIND
    ):
        raise NativeTrialQAEvidenceConflictError("Native evidence document identity is invalid")
    _normalize_task(_mapping(evidence.get("task")))
    _normalize_outcome(_mapping(evidence.get("outcome")))
    execution = _mapping(evidence.get("execution"))
    _normalize_run(execution)
    events = evidence.get("events")
    if not isinstance(events, list) or not events:
        raise NativeTrialQAEvidenceConflictError("Native evidence events are missing")
    for sequence, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or event.get("sequence") != sequence
            or event.get("kind") not in _EVENT_KINDS
            or not isinstance(event.get("payload"), dict)
            or not isinstance(event.get("metadata"), dict)
        ):
            raise NativeTrialQAEvidenceConflictError(
                "Native evidence events must be contiguous source-neutral records"
            )


def _evidence_manifest(
    evidence_id: str,
    document: JsonObject,
    artifacts: JsonObject,
) -> JsonObject:
    evidence_bytes = _json_document(document).encode("utf-8")
    return {
        "schema_version": NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION,
        "kind": "switchyard_skill_distillation_evidence",
        "source_type": NATIVE_TRIALQA_SOURCE_KIND,
        "evidence_id": evidence_id,
        "evidence": {
            "path": "evidence.json",
            "sha256": f"sha256:{hashlib.sha256(evidence_bytes).hexdigest()}",
            "size_bytes": len(evidence_bytes),
        },
        "artifacts": artifacts,
    }


def _write_evidence_directory(
    evidence_dir: Path,
    prepared: _PreparedEvidence,
) -> None:
    staging = evidence_dir.parent / f".{prepared.evidence_id}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        raw_dir = staging / "raw"
        raw_dir.mkdir(mode=0o700)
        for artifact in prepared.artifacts:
            destination = staging.joinpath(*artifact.path.parts)
            _write_file_exclusive(destination, artifact.content)
        _write_file_exclusive(
            staging / "evidence.json",
            _json_document(prepared.document).encode("utf-8"),
        )
        _write_file_exclusive(
            staging / "manifest.json",
            _json_document(prepared.manifest).encode("utf-8"),
        )
        staging.replace(evidence_dir)
    except Exception:
        _remove_staging_path(staging)
        raise


def _validate_existing_evidence(
    evidence_dir: Path,
    prepared: _PreparedEvidence,
) -> None:
    validate_native_trialqa_evidence_directory(
        evidence_dir,
        expected_evidence_id=prepared.evidence_id,
    )
    existing_document = _read_json_object_file(
        evidence_dir / "evidence.json",
        label="native evidence document",
    )
    existing_manifest = _read_json_object_file(
        evidence_dir / "manifest.json",
        label="native evidence manifest",
    )
    if existing_document != prepared.document or existing_manifest != prepared.manifest:
        raise NativeTrialQAEvidenceConflictError(
            f"Immutable native evidence {prepared.evidence_id} contains different metadata"
        )
    for artifact in prepared.artifacts:
        existing = _read_regular_file(
            evidence_dir.joinpath(*artifact.path.parts),
            label=f"native evidence artifact {artifact.name!r}",
        )
        if existing != artifact.content:
            raise NativeTrialQAEvidenceConflictError(
                f"Immutable native evidence artifact changed: {artifact.name}"
            )


def _validate_file_marker(path: Path, marker: Mapping[str, Any]) -> None:
    expected_hash = marker.get("sha256")
    expected_size = marker.get("size_bytes")
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise NativeTrialQAEvidenceConflictError(f"Invalid integrity marker for {path}")
    content = _read_regular_file(path, label=f"native evidence file {path.name!r}")
    actual_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
    if actual_hash != expected_hash or len(content) != expected_size:
        raise NativeTrialQAEvidenceConflictError(f"Integrity check failed for {path}")


def _validate_directory_shape(
    evidence_dir: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
) -> None:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for child in evidence_dir.rglob("*"):
        relative = child.relative_to(evidence_dir).as_posix()
        if child.is_symlink():
            raise NativeTrialQAEvidenceConflictError(
                f"Native evidence cannot contain symlinks: {relative}"
            )
        if child.is_dir():
            actual_directories.add(relative)
        elif child.is_file():
            actual_files.add(relative)
        else:
            raise NativeTrialQAEvidenceConflictError(
                f"Native evidence contains an unsupported entry: {relative}"
            )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise NativeTrialQAEvidenceConflictError(
            f"Native evidence directory contains unexpected entries: {evidence_dir}"
        )


def _artifact_path(value: Any, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise NativeTrialQAEvidenceConflictError(f"Native artifact {name!r} has no path")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise NativeTrialQAEvidenceConflictError(
            f"Native artifact {name!r} has an unsafe path: {value}"
        )
    return path


def _read_json_object_file(path: Path, *, label: str) -> JsonObject:
    return _read_json_object_bytes(_read_regular_file(path, label=label), label=label)


def _read_json_object_bytes(payload: bytes, *, label: str) -> JsonObject:
    try:
        value: object = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeTrialQAEvidenceError(f"Invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise NativeTrialQAEvidenceError(f"Expected a JSON object in {label}")
    return cast(JsonObject, value)


def _read_json_lines_bytes(payload: bytes, *, label: str) -> list[JsonObject]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise NativeTrialQAEvidenceError(f"Invalid UTF-8 in {label}") from exc
    values: list[JsonObject] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NativeTrialQAEvidenceError(f"Invalid JSON in {label} line {line_number}") from exc
        if not isinstance(value, dict):
            raise NativeTrialQAEvidenceError(
                f"Expected a JSON object in {label} line {line_number}"
            )
        values.append(cast(JsonObject, value))
    return values


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise NativeTrialQAEvidenceError(f"Missing or unreadable {label}: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise NativeTrialQAEvidenceConflictError(
            f"{label.capitalize()} is not a single-link regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise NativeTrialQAEvidenceConflictError(f"Could not safely open {label}: {path}") from exc
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise NativeTrialQAEvidenceConflictError(
                f"{label.capitalize()} changed while it was opened: {path}"
            )
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_nlink != 1
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise NativeTrialQAEvidenceConflictError(
                f"{label.capitalize()} changed while it was read: {path}"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _write_file_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise NativeTrialQAEvidenceConflictError(
                f"Could not create a private regular evidence file: {path}"
            )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(f"Short write while creating native evidence: {path}")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_staging_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _reject_symlink_components(path: Path, *, label: str) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise NativeTrialQAEvidenceConflictError(f"Refusing symlinked {label}: {candidate}")


def _normalize_json_object(value: Mapping[str, Any], *, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise NativeTrialQAEvidenceError(f"{label} must be a JSON object")
    normalized = _normalize_json_value(value, label=label)
    return cast(JsonObject, normalized)


def _normalize_json_value(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        normalized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NativeTrialQAEvidenceError(f"{label} contains a non-string key")
            normalized[key] = _normalize_json_value(item, label=f"{label}.{key}")
        return normalized
    if isinstance(value, list | tuple):
        return [
            _normalize_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NativeTrialQAEvidenceError(f"{label} must be finite")
        return value
    raise NativeTrialQAEvidenceError(f"{label} is not JSON serializable")


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeTrialQAEvidenceError(f"{field} is required and must be non-empty")
    return value.strip()


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NativeTrialQAEvidenceError(f"{field} is required and must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise NativeTrialQAEvidenceError(f"{field} must be finite")
    return number


def _validate_optional_index(value: JsonObject, field: str, *, minimum: int) -> None:
    if field not in value:
        return
    index = value[field]
    if isinstance(index, bool) or not isinstance(index, int) or index < minimum:
        raise NativeTrialQAEvidenceError(f"task.{field} must be an integer >= {minimum}")


def _ordered_text_values(values: Sequence[JsonObject], field: str) -> list[str]:
    result: list[str] = []
    for value in values:
        item = value.get(field)
        if isinstance(item, str) and item.strip() and item not in result:
            result.append(item)
    return result


def _mapping(value: Any) -> JsonObject:
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_document(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "NATIVE_TRIALQA_EVIDENCE_SCHEMA_VERSION",
    "NATIVE_TRIALQA_SOURCE_KIND",
    "NativeTrialQAEvidenceConflictError",
    "NativeTrialQAEvidenceError",
    "NativeTrialQAEvidenceImportResult",
    "import_native_trialqa_evidence",
    "validate_native_trialqa_evidence_directory",
]

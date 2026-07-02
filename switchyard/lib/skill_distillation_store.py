# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-local storage helpers for skill distillation sessions."""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchyard.lib.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)

SKILL_DISTILLATION_ROOT = Path(".switchyard") / "skill-distillation"
SKILL_DISTILLATION_SCHEMA_VERSION = 1

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SkillDistillationStoreSummary:
    """Small status snapshot for one namespace's local store."""

    path: Path
    session_count: int
    active_skill_path: Path
    active_skill_exists: bool


def resolve_skill_distillation_store_path(
    namespace: str,
    project_dir: Path | None = None,
) -> Path:
    """Return the project-local store path for *namespace*."""

    root = project_dir or Path.cwd()
    return root / SKILL_DISTILLATION_ROOT / namespace


def summarize_skill_distillation_store(
    namespace: str,
    project_dir: Path | None = None,
) -> SkillDistillationStoreSummary:
    """Return inspectable state without creating the store."""

    path = resolve_skill_distillation_store_path(namespace, project_dir)
    sessions_dir = path / "sessions"
    session_count = 0
    if sessions_dir.is_dir():
        session_count = sum(1 for child in sessions_dir.iterdir() if child.is_dir())
    active_skill_path = path / "active" / "SKILL.md"
    return SkillDistillationStoreSummary(
        path=path,
        session_count=session_count,
        active_skill_path=active_skill_path,
        active_skill_exists=active_skill_path.is_file(),
    )


class SkillDistillationSessionCapture:
    """Owns the files for one saved launcher session."""

    def __init__(
        self,
        *,
        namespace: str,
        launch_target: str,
        display_model: str,
        strategy_summary: str | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self.namespace = namespace
        self.launch_target = launch_target
        self.display_model = display_model
        self.strategy_summary = strategy_summary
        self.started_at = _utc_now()
        self.session_id = _new_session_id(launch_target)
        self.store_path = resolve_skill_distillation_store_path(namespace, project_dir)
        self.session_dir = self.store_path / "sessions" / self.session_id
        self.turns_path = self.session_dir / "turns.jsonl"
        self.stats_path = self.session_dir / "stats.json"
        self.session_path = self.session_dir / "session.json"
        self.ledger_path = self.store_path / "distillation-ledger.jsonl"
        self._turn_count = 0
        self._finished = False
        self._active_skill = self._active_skill_metadata()
        self._ensure_layout()
        self._write_session(status="running")

    @property
    def active_skill_version(self) -> str | None:
        version = self._active_skill.get("version")
        return version if isinstance(version, str) else None

    def record_turn(self, entry: JsonObject) -> None:
        """Append one normalized request/response turn."""

        if self._finished:
            return
        body = {
            "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "turn_index": self._turn_count,
            "recorded_at": _utc_now(),
            "active_skill_version": self.active_skill_version,
            **entry,
        }
        try:
            with self.turns_path.open("a", encoding="utf-8") as handle:
                json.dump(body, handle, sort_keys=True, default=str)
                handle.write("\n")
        except OSError as exc:
            logger.warning(
                "Skill distillation: failed to write turn to %s: %s",
                self.turns_path,
                exc,
            )
            return
        self._turn_count += 1

    def finish(
        self,
        *,
        exit_code: int | None,
        stats: StatsAccumulator,
        error: BaseException | None = None,
    ) -> None:
        """Finalize session metadata and stats without raising."""

        if self._finished:
            return
        self._finished = True
        stats_snapshot = _stats_snapshot(stats)
        status = _status_for_exit(exit_code, error)
        try:
            _write_json(self.stats_path, stats_snapshot)
            self._append_ledger(status=status, exit_code=exit_code)
            self._write_session(
                status=status,
                exit_code=exit_code,
                ended_at=_utc_now(),
                error=repr(error) if error is not None else None,
            )
        except OSError as exc:
            logger.warning(
                "Skill distillation: failed to finalize session %s: %s",
                self.session_id,
                exc,
            )

    def _ensure_layout(self) -> None:
        for directory in (
            self.store_path / "active",
            self.store_path / "candidates",
            self.store_path / "history",
            self.store_path / "reports",
            self.store_path / "sessions",
            self.session_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _active_skill_metadata(self) -> JsonObject:
        active_skill_path = self.store_path / "active" / "SKILL.md"
        version: str | None = None
        if active_skill_path.is_file():
            try:
                stat = active_skill_path.stat()
            except OSError:
                version = None
            else:
                version = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        return {
            "loaded": False,
            "path": str(active_skill_path),
            "version": version,
        }

    def _session_document(
        self,
        *,
        status: str,
        exit_code: int | None = None,
        ended_at: str | None = None,
        error: str | None = None,
    ) -> JsonObject:
        distillation_status = "pending" if self._turn_count else "skipped"
        distillation_reason = (
            "session captured for end-of-session distillation"
            if self._turn_count
            else "no completed turns were captured"
        )
        document: JsonObject = {
            "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "launch_target": self.launch_target,
            "display_model": self.display_model,
            "strategy_summary": self.strategy_summary,
            "store_path": str(self.store_path),
            "session_path": str(self.session_dir),
            "turns_path": "turns.jsonl",
            "stats_path": "stats.json",
            "started_at": self.started_at,
            "ended_at": ended_at,
            "status": status,
            "exit_code": exit_code,
            "turn_count": self._turn_count,
            "active_skill": self._active_skill,
            "distillation": {
                "status": distillation_status,
                "reason": distillation_reason,
                "ledger_path": str(self.ledger_path),
            },
        }
        if error is not None:
            document["error"] = error
        return document

    def _write_session(
        self,
        *,
        status: str,
        exit_code: int | None = None,
        ended_at: str | None = None,
        error: str | None = None,
    ) -> None:
        _write_json(
            self.session_path,
            self._session_document(
                status=status,
                exit_code=exit_code,
                ended_at=ended_at,
                error=error,
            ),
        )

    def _append_ledger(self, *, status: str, exit_code: int | None) -> None:
        entry = {
            "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            "session_id": self.session_id,
            "session_path": str(self.session_dir),
            "status": "pending" if self._turn_count else "skipped",
            "session_status": status,
            "exit_code": exit_code,
            "turn_count": self._turn_count,
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle, sort_keys=True, default=str)
            handle.write("\n")


def _stats_snapshot(stats: StatsAccumulator) -> JsonObject:
    try:
        snapshot = stats.snapshot_sync()
    except Exception as exc:  # pragma: no cover - defensive fail-open path.
        logger.warning("Skill distillation: failed to snapshot stats: %s", exc)
        return {"error": repr(exc)}
    return dict(snapshot)


def _status_for_exit(exit_code: int | None, error: BaseException | None) -> str:
    if error is not None:
        return "failed"
    if exit_code is None:
        return "unknown"
    if exit_code == 0:
        return "completed"
    if exit_code == 130:
        return "interrupted"
    return "failed"


def _write_json(path: Path, data: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    tmp_path.replace(path)


def _new_session_id(target: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{target}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "SKILL_DISTILLATION_SCHEMA_VERSION",
    "SkillDistillationSessionCapture",
    "SkillDistillationStoreSummary",
    "resolve_skill_distillation_store_path",
    "summarize_skill_distillation_store",
]

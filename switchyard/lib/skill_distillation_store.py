# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-local storage helpers for skill distillation sessions."""

import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import stat
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from switchyard.lib.stats_accumulator import StatsAccumulator

try:
    _fcntl: Any = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - Windows.
    _fcntl = None

try:
    _msvcrt: Any = importlib.import_module("msvcrt")
except ImportError:  # pragma: no cover - POSIX.
    _msvcrt = None

logger = logging.getLogger(__name__)

SKILL_DISTILLATION_ROOT = Path(".switchyard") / "skill-distillation"
SKILL_DISTILLATION_SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_NAME = "manifest.json"
ACTIVATION_LEDGER_NAME = "activation-ledger.jsonl"
STORE_LOCK_NAME = ".store.lock"
TRANSACTION_JOURNAL_NAME = ".active-transaction.json"

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")
_PROCESS_STORE_LOCK = threading.RLock()
_LOCKED_STORE_PATHS: set[str] = set()

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SkillDistillationStoreSummary:
    """Small status snapshot for one namespace's local store."""

    path: Path
    session_count: int
    active_skill_path: Path
    active_skill_exists: bool


@dataclass(frozen=True)
class SkillActivationRecord:
    """Result of publishing or restoring one active skill bundle."""

    namespace: str
    operation: Literal["activate", "rollback"]
    active_candidate_id: str
    previous_candidate_id: str | None
    recorded_at: str
    history_path: Path | None


class SkillDistillationMigrationError(RuntimeError):
    """Raised when an old active bundle cannot participate in safe history."""


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


class SkillDistillationStore:
    """Persist immutable candidates and transactionally publish active skill bundles."""

    def __init__(
        self,
        namespace: str,
        project_dir: Path | None = None,
    ) -> None:
        _validate_safe_component(namespace, "namespace")
        self.namespace = namespace
        self.project_path = (project_dir or Path.cwd()).absolute()
        self.store_path = resolve_skill_distillation_store_path(namespace, self.project_path)
        self.candidates_path = self.store_path / "candidates"
        self.active_path = self.store_path / "active"
        self.history_path = self.store_path / "history"
        self.sessions_path = self.store_path / "sessions"
        self.evidence_path = self.store_path / "evidence"
        self.activation_ledger_path = self.history_path / ACTIVATION_LEDGER_NAME
        self.lock_path = self.store_path / STORE_LOCK_NAME
        self.transaction_journal_path = self.store_path / TRANSACTION_JOURNAL_NAME
        self._ensure_layout()

    def save_candidate(
        self,
        *,
        candidate_id: str,
        skills: Mapping[str, str],
        generator: str,
        evidence_ids: Sequence[str],
        validation: Mapping[str, Any],
        created_at: str | None = None,
    ) -> Path:
        """Save an immutable candidate, accepting a repeat only when it is identical."""

        with self._exclusive_lock():
            return self._save_candidate(
                candidate_id=candidate_id,
                skills=skills,
                generator=generator,
                evidence_ids=evidence_ids,
                validation=validation,
                created_at=created_at,
            )

    def _save_candidate(
        self,
        *,
        candidate_id: str,
        skills: Mapping[str, str],
        generator: str,
        evidence_ids: Sequence[str],
        validation: Mapping[str, Any],
        created_at: str | None,
    ) -> Path:
        _validate_safe_component(candidate_id, "candidate id")
        generator = _required_text(generator, "generator")
        normalized_evidence_ids = _normalize_evidence_ids(evidence_ids)
        normalized_skills = _normalize_skills(skills)
        normalized_validation = _normalize_json_object(validation, "validation")
        _validation_status(normalized_validation)

        candidate_path = self.candidates_path / candidate_id
        manifest_created_at = created_at
        if manifest_created_at is None and _path_exists(candidate_path):
            manifest_created_at = _existing_candidate_created_at(
                candidate_path,
                namespace=self.namespace,
                candidate_id=candidate_id,
            )
        manifest_created_at = _required_text(
            manifest_created_at or _utc_now(),
            "created_at",
        )
        manifest: JsonObject = {
            "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
            "namespace": self.namespace,
            "candidate_id": candidate_id,
            "generator": generator,
            "provenance": {"source_evidence_ids": normalized_evidence_ids},
            "validation": normalized_validation,
            "created_at": manifest_created_at,
            "skills": [
                {
                    "path": relative_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for relative_path, content in normalized_skills.items()
            ],
        }

        staged_path = self.candidates_path / f".{candidate_id}-{uuid.uuid4().hex}.tmp"
        _require_absent_path(staged_path, "candidate staging path")
        try:
            _write_candidate_bundle(staged_path, normalized_skills, manifest)
            _validate_candidate_bundle(
                staged_path,
                namespace=self.namespace,
                expected_candidate_id=candidate_id,
            )
            if _path_exists(candidate_path):
                return self._accept_identical_candidate(
                    candidate_path,
                    staged_path,
                    candidate_id,
                )
            try:
                _rename_directory(staged_path, candidate_path)
            except OSError as exc:
                if _path_exists(candidate_path):
                    try:
                        return self._accept_identical_candidate(
                            candidate_path,
                            staged_path,
                            candidate_id,
                        )
                    except FileExistsError as conflict:
                        raise conflict from exc
                raise
            return candidate_path
        finally:
            _remove_staging_path(staged_path)

    def activate(self, candidate_id: str) -> SkillActivationRecord:
        """Publish a validated candidate after confirming all local evidence exists."""

        with self._exclusive_lock():
            return self._activate(candidate_id)

    def _activate(self, candidate_id: str) -> SkillActivationRecord:
        _validate_safe_component(candidate_id, "candidate id")
        candidate_path = self.candidates_path / candidate_id
        _validate_candidate_bundle(
            candidate_path,
            namespace=self.namespace,
            expected_candidate_id=candidate_id,
            require_passed=True,
            evidence_validator=self._validate_evidence,
        )

        previous_candidate_id: str | None = None
        history_path: Path | None = None
        active_has_content = self._active_has_content()
        if active_has_content:
            active_manifest = _validate_candidate_bundle(
                self.active_path,
                namespace=self.namespace,
            )
            previous_candidate_id = _manifest_candidate_id(active_manifest)
            if previous_candidate_id == candidate_id:
                if not _directories_identical(self.active_path, candidate_path):
                    raise ValueError(
                        f"active candidate {candidate_id!r} differs from its immutable source"
                    )
                return SkillActivationRecord(
                    namespace=self.namespace,
                    operation="activate",
                    active_candidate_id=candidate_id,
                    previous_candidate_id=None,
                    recorded_at=_utc_now(),
                    history_path=None,
                )
            history_path = self._new_history_path(previous_candidate_id)

        staged_path = self.store_path / f".active-{candidate_id}-{uuid.uuid4().hex}.tmp"
        backup_path = history_path or (self.store_path / f".active-empty-{uuid.uuid4().hex}.tmp")
        _require_absent_path(staged_path, "active staging path")
        _require_absent_path(backup_path, "active backup path")
        try:
            shutil.copytree(candidate_path, staged_path)
            _validate_candidate_bundle(
                staged_path,
                namespace=self.namespace,
                expected_candidate_id=candidate_id,
                require_passed=True,
                evidence_validator=self._validate_evidence,
            )
        except BaseException:
            _remove_staging_path(staged_path)
            raise

        transaction_id = uuid.uuid4().hex
        self._write_transaction_journal(
            {
                "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
                "namespace": self.namespace,
                "operation": "activate",
                "transaction_id": transaction_id,
                "staged_path": self._relative_store_path(staged_path),
                "backup_path": self._relative_store_path(backup_path),
                "preserve_backup": history_path is not None,
            }
        )
        try:
            self._publish_active(staged_path, backup_path)
        except BaseException:
            self._recover_transaction()
            raise

        record = SkillActivationRecord(
            namespace=self.namespace,
            operation="activate",
            active_candidate_id=candidate_id,
            previous_candidate_id=previous_candidate_id,
            recorded_at=_utc_now(),
            history_path=history_path,
        )
        try:
            self._append_activation_record(record, transaction_id=transaction_id)
        except BaseException:
            self._restore_active_after_failed_commit(backup_path)
            self._remove_transaction_journal()
            raise
        if history_path is None:
            _remove_staging_path(backup_path)
        self._remove_transaction_journal()
        return record

    def rollback(self) -> SkillActivationRecord:
        """Restore the immediately preceding bundle, reversing immediate failures."""

        with self._exclusive_lock():
            return self._rollback()

    def _rollback(self) -> SkillActivationRecord:
        history_path = self._latest_history_path()
        restored_manifest = _validate_candidate_bundle(
            history_path,
            namespace=self.namespace,
            require_passed=True,
            evidence_validator=self._validate_evidence,
        )
        restored_candidate_id = _manifest_candidate_id(restored_manifest)
        if not self._active_has_content():
            raise FileNotFoundError("cannot roll back without an active skill bundle")
        active_manifest = _validate_candidate_bundle(
            self.active_path,
            namespace=self.namespace,
        )
        previous_candidate_id = _manifest_candidate_id(active_manifest)
        displaced_path = self.store_path / f".rollback-{uuid.uuid4().hex}.tmp"
        _require_absent_path(displaced_path, "rollback staging path")
        transaction_id = uuid.uuid4().hex
        self._write_transaction_journal(
            {
                "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
                "namespace": self.namespace,
                "operation": "rollback",
                "transaction_id": transaction_id,
                "history_path": self._relative_store_path(history_path),
                "displaced_path": self._relative_store_path(displaced_path),
            }
        )

        _rename_directory(self.active_path, displaced_path)
        try:
            _rename_directory(history_path, self.active_path)
        except BaseException:
            try:
                _rename_directory(displaced_path, self.active_path)
            except BaseException as restore_error:
                raise RuntimeError(
                    "rollback failed and the prior active bundle could not be restored"
                ) from restore_error
            self._recover_transaction()
            raise
        record = SkillActivationRecord(
            namespace=self.namespace,
            operation="rollback",
            active_candidate_id=restored_candidate_id,
            previous_candidate_id=previous_candidate_id,
            recorded_at=_utc_now(),
            history_path=history_path,
        )
        try:
            self._append_activation_record(record, transaction_id=transaction_id)
        except BaseException:
            self._restore_after_failed_rollback(history_path, displaced_path)
            self._remove_transaction_journal()
            raise
        _remove_staging_path(displaced_path)
        self._remove_transaction_journal()
        return record

    def _ensure_layout(self) -> None:
        for index, directory in enumerate(self._required_directories()):
            _ensure_real_directory(directory, create=index > 0)

    def _required_directories(self) -> tuple[Path, ...]:
        directories = list(_store_root_directories(self.project_path, self.namespace))
        directories.extend(
            (
                self.candidates_path,
                self.active_path,
                self.history_path,
                self.sessions_path,
                self.evidence_path,
            )
        )
        return tuple(directories)

    def _assert_store_layout(self) -> None:
        for directory in self._required_directories():
            _ensure_real_directory(directory, create=False)

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Serialize one adapter transaction against candidate store operations.

        This context is not reentrant. Adapters must not call candidate mutation
        methods or acquire the same namespace lock while it is held.
        """

        with self._exclusive_lock():
            yield

    def validate_session_evidence(self, session_path: Path) -> str:
        """Validate a finalized native session owned by this store.

        Adapters may call this while :meth:`exclusive_lock` is held. The session
        must be a direct child of this namespace's ``sessions`` directory so a
        caller cannot validate data from another project or namespace by path.
        """

        session_path = session_path.expanduser().absolute()
        if session_path.parent != self.sessions_path:
            raise ValueError(
                "native session evidence must be a direct child of this store's "
                f"sessions directory: {session_path}"
            )
        evidence_id = _validate_safe_component(session_path.name, "session id")
        _validate_session_evidence(
            session_path,
            evidence_id=evidence_id,
            namespace=self.namespace,
        )
        return evidence_id

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with _PROCESS_STORE_LOCK:
            lock_key = os.path.normcase(str(self.lock_path))
            if lock_key in _LOCKED_STORE_PATHS:
                raise RuntimeError(
                    "skill-distillation store locks are not reentrant; do not call "
                    "candidate operations from an adapter lock"
                )
            _LOCKED_STORE_PATHS.add(lock_key)
            try:
                self._assert_store_layout()
                descriptor = _open_regular_no_follow(
                    self.lock_path,
                    os.O_RDWR | os.O_CREAT,
                    mode=0o600,
                )
                lock_kind: Literal["fcntl", "msvcrt", "local"] | None = None
                try:
                    lock_kind = _lock_descriptor(descriptor)
                    self._assert_store_layout()
                    self._recover_transaction()
                    yield
                finally:
                    if lock_kind is not None:
                        _unlock_descriptor(descriptor, lock_kind)
                    os.close(descriptor)
            finally:
                _LOCKED_STORE_PATHS.remove(lock_key)

    def _validate_evidence(self, evidence_id: str) -> None:
        session_path = self.sessions_path / evidence_id
        if _path_exists(session_path):
            _validate_session_evidence(
                session_path,
                evidence_id=evidence_id,
                namespace=self.namespace,
            )
            return
        imported_path = self.evidence_path / evidence_id
        if _path_exists(imported_path):
            if evidence_id.startswith("native-"):
                from switchyard.lib.skill_distillation_native import (
                    validate_native_trialqa_evidence_directory,
                )

                validate_native_trialqa_evidence_directory(
                    imported_path,
                    expected_evidence_id=evidence_id,
                )
            else:
                raise ValueError(
                    f"candidate references unsupported imported evidence {evidence_id!r}"
                )
            return
        raise FileNotFoundError(f"candidate references missing local evidence {evidence_id!r}")

    def _accept_identical_candidate(
        self,
        candidate_path: Path,
        staged_path: Path,
        candidate_id: str,
    ) -> Path:
        if not _directories_identical(candidate_path, staged_path):
            raise FileExistsError(
                f"candidate {candidate_id!r} is immutable and already contains different data"
            )
        _validate_candidate_bundle(
            candidate_path,
            namespace=self.namespace,
            expected_candidate_id=candidate_id,
        )
        return candidate_path

    def _active_has_content(self) -> bool:
        if not _path_exists(self.active_path):
            return False
        if self.active_path.is_symlink() or not self.active_path.is_dir():
            raise ValueError(f"active path is not a local directory: {self.active_path}")
        has_content = next(self.active_path.iterdir(), None) is not None
        if (
            has_content
            and _path_exists(self.active_path / "SKILL.md")
            and not _path_exists(self.active_path / CANDIDATE_MANIFEST_NAME)
        ):
            raise SkillDistillationMigrationError(
                "legacy active/SKILL.md has no candidate manifest; migrate or remove the "
                "legacy bundle before activation"
            )
        return has_content

    def _new_history_path(self, candidate_id: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return self.history_path / f"{timestamp}-{candidate_id}-{uuid.uuid4().hex[:8]}"

    def _latest_history_path(self) -> Path:
        candidates = sorted(
            (
                path
                for path in self.history_path.iterdir()
                if path.is_dir() and not path.is_symlink()
            ),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("no previous active skill bundle is available")
        return candidates[0]

    def _publish_active(self, staged_path: Path, backup_path: Path) -> None:
        active_existed = _path_exists(self.active_path)
        if active_existed:
            if self.active_path.is_symlink() or not self.active_path.is_dir():
                raise ValueError(f"active path is not a local directory: {self.active_path}")
            _rename_directory(self.active_path, backup_path)
        try:
            _rename_directory(staged_path, self.active_path)
        except BaseException:
            if active_existed:
                try:
                    _rename_directory(backup_path, self.active_path)
                except BaseException as restore_error:
                    raise RuntimeError(
                        "activation failed and the prior active bundle could not be restored"
                    ) from restore_error
            raise

    def _restore_active_after_failed_commit(self, backup_path: Path) -> None:
        displaced_path = self.store_path / f".failed-activation-{uuid.uuid4().hex}.tmp"
        _require_absent_path(displaced_path, "failed activation staging path")
        _rename_directory(self.active_path, displaced_path)
        try:
            _rename_directory(backup_path, self.active_path)
        except BaseException:
            try:
                _rename_directory(displaced_path, self.active_path)
            except BaseException as restore_error:
                raise RuntimeError(
                    "activation ledger failed and the active bundle could not be restored"
                ) from restore_error
            raise
        _remove_staging_path(displaced_path)

    def _restore_after_failed_rollback(
        self,
        history_path: Path,
        displaced_path: Path,
    ) -> None:
        failed_path = self.store_path / f".failed-rollback-{uuid.uuid4().hex}.tmp"
        _require_absent_path(failed_path, "failed rollback staging path")
        _rename_directory(self.active_path, failed_path)
        try:
            _rename_directory(displaced_path, self.active_path)
        except BaseException:
            try:
                _rename_directory(failed_path, self.active_path)
            except BaseException as restore_error:
                raise RuntimeError(
                    "rollback ledger failed and the active bundle could not be restored"
                ) from restore_error
            raise
        try:
            _rename_directory(failed_path, history_path)
        except BaseException as restore_error:
            raise RuntimeError(
                "rollback ledger failed and the history bundle could not be restored"
            ) from restore_error

    def _relative_store_path(self, path: Path) -> str:
        return path.relative_to(self.store_path).as_posix()

    def _write_transaction_journal(self, journal: JsonObject) -> None:
        _require_absent_path(self.transaction_journal_path, "activation transaction journal")
        _write_json_atomic_no_follow(self.transaction_journal_path, journal)

    def _remove_transaction_journal(self) -> None:
        if not _path_exists(self.transaction_journal_path):
            return
        if self.transaction_journal_path.is_symlink():
            raise ValueError(
                f"refusing symlinked activation transaction journal: "
                f"{self.transaction_journal_path}"
            )
        self.transaction_journal_path.unlink()

    def _recover_transaction(self) -> None:
        if not _path_exists(self.transaction_journal_path):
            return
        journal = _read_json_object_no_follow(self.transaction_journal_path)
        if (
            journal.get("schema_version") != SKILL_DISTILLATION_SCHEMA_VERSION
            or journal.get("namespace") != self.namespace
        ):
            raise ValueError("activation transaction journal identity is invalid")
        transaction_id = journal.get("transaction_id")
        if (
            not isinstance(transaction_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
        ):
            raise ValueError("activation transaction journal id is invalid")
        committed = self._activation_ledger_contains(transaction_id)
        operation = journal.get("operation")
        if operation == "activate":
            self._recover_activation_transaction(journal, committed=committed)
        elif operation == "rollback":
            self._recover_rollback_transaction(journal, committed=committed)
        else:
            raise ValueError("activation transaction journal operation is invalid")

    def _recover_activation_transaction(
        self,
        journal: JsonObject,
        *,
        committed: bool,
    ) -> None:
        staged_path = self._journal_store_path(journal, "staged_path")
        backup_path = self._journal_store_path(journal, "backup_path")
        preserve_backup = journal.get("preserve_backup")
        if not isinstance(preserve_backup, bool):
            raise ValueError("activation transaction preserve_backup must be boolean")
        if committed:
            _ensure_real_directory(self.active_path, create=False)
            _remove_staging_path(staged_path)
            if not preserve_backup:
                _remove_staging_path(backup_path)
            self._remove_transaction_journal()
            return

        if backup_path.is_dir() and not backup_path.is_symlink():
            displaced_path: Path | None = None
            if self.active_path.is_dir() and not self.active_path.is_symlink():
                if staged_path.is_dir() and not staged_path.is_symlink():
                    if next(self.active_path.iterdir(), None) is not None:
                        raise RuntimeError("cannot recover activation with conflicting active data")
                    self.active_path.rmdir()
                else:
                    displaced_path = self.store_path / f".recovery-{uuid.uuid4().hex}.tmp"
                    _rename_directory(self.active_path, displaced_path)
            _rename_directory(backup_path, self.active_path)
            if displaced_path is not None:
                _remove_staging_path(displaced_path)
        else:
            _ensure_real_directory(self.active_path, create=False)
        _remove_staging_path(staged_path)
        self._remove_transaction_journal()

    def _recover_rollback_transaction(
        self,
        journal: JsonObject,
        *,
        committed: bool,
    ) -> None:
        history_path = self._journal_store_path(journal, "history_path")
        displaced_path = self._journal_store_path(journal, "displaced_path")
        if committed:
            _ensure_real_directory(self.active_path, create=False)
            _remove_staging_path(displaced_path)
            self._remove_transaction_journal()
            return

        if displaced_path.is_dir() and not displaced_path.is_symlink():
            if history_path.is_dir() and not history_path.is_symlink():
                if self.active_path.is_dir() and not self.active_path.is_symlink():
                    if next(self.active_path.iterdir(), None) is not None:
                        raise RuntimeError("cannot recover rollback with conflicting active data")
                    self.active_path.rmdir()
                _rename_directory(displaced_path, self.active_path)
            else:
                _ensure_real_directory(self.active_path, create=False)
                _rename_directory(self.active_path, history_path)
                _rename_directory(displaced_path, self.active_path)
        else:
            _ensure_real_directory(self.active_path, create=False)
            _ensure_real_directory(history_path, create=False)
        self._remove_transaction_journal()

    def _journal_store_path(self, journal: JsonObject, field: str) -> Path:
        value = journal.get(field)
        if not isinstance(value, str):
            raise ValueError(f"activation transaction {field} must be a string")
        relative = PurePosixPath(value)
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"activation transaction {field} is unsafe")
        return self.store_path.joinpath(*relative.parts)

    def _activation_ledger_contains(self, transaction_id: str) -> bool:
        if not _path_exists(self.activation_ledger_path):
            return False
        for entry in _read_json_lines_no_follow(self.activation_ledger_path):
            if entry.get("transaction_id") == transaction_id:
                return True
        return False

    def _append_activation_record(
        self,
        record: SkillActivationRecord,
        *,
        transaction_id: str | None = None,
    ) -> None:
        entry = {
            "schema_version": SKILL_DISTILLATION_SCHEMA_VERSION,
            "namespace": record.namespace,
            "operation": record.operation,
            "active_candidate_id": record.active_candidate_id,
            "previous_candidate_id": record.previous_candidate_id,
            "recorded_at": record.recorded_at,
            "history_path": str(record.history_path) if record.history_path is not None else None,
        }
        if transaction_id is not None:
            entry["transaction_id"] = transaction_id
        self._assert_store_layout()
        _append_json_line_no_follow(self.activation_ledger_path, entry)


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
        run_context: Mapping[str, Any] | None = None,
        active_skill_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        namespace = _validate_safe_component(namespace, "namespace")
        launch_target = _validate_safe_component(launch_target, "launch target")
        self.namespace = namespace
        self.launch_target = launch_target
        self.display_model = display_model
        self.strategy_summary = strategy_summary
        self.run_context = (
            _normalize_json_object(run_context, "run context")
            if run_context is not None
            else None
        )
        normalized_active_skill = (
            _normalize_json_object(active_skill_evidence, "active skill evidence")
            if active_skill_evidence is not None
            else None
        )
        if normalized_active_skill is not None and not isinstance(
            normalized_active_skill.get("loaded"),
            bool,
        ):
            raise ValueError("active skill evidence must contain boolean loaded")
        self.started_at = _utc_now()
        self.session_id = _new_session_id(launch_target)
        self.project_path = (project_dir or Path.cwd()).absolute()
        self.store_path = resolve_skill_distillation_store_path(namespace, self.project_path)
        self.session_dir = self.store_path / "sessions" / self.session_id
        self.turns_path = self.session_dir / "turns.jsonl"
        self.stats_path = self.session_dir / "stats.json"
        self.session_path = self.session_dir / "session.json"
        self.ledger_path = self.store_path / "distillation-ledger.jsonl"
        self._turn_count = 0
        self._finished = False
        try:
            self._ensure_layout()
        except (OSError, ValueError) as exc:
            logger.warning(
                "Skill distillation: failed to initialize local session store: %s",
                exc,
            )
            raise OSError("unsafe or unavailable skill-distillation session store") from exc
        self._active_skill = normalized_active_skill or self._active_skill_metadata()
        self._write_session(status="running")

    @property
    def active_skill_version(self) -> str | None:
        version = self._active_skill.get("candidate_id") or self._active_skill.get("version")
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
            "active_skill_candidate_id": self._active_skill.get("candidate_id"),
            "active_skill_manifest_sha256": self._active_skill.get("manifest_sha256"),
            **entry,
        }
        try:
            _append_json_line_no_follow(self.turns_path, body)
        except (OSError, ValueError) as exc:
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
            trajectory_sha256 = (
                f"sha256:{_sha256_regular_file(self.turns_path)}" if self._turn_count else None
            )
            _write_json(self.stats_path, stats_snapshot)
            self._append_ledger(status=status, exit_code=exit_code)
            self._write_session(
                status=status,
                exit_code=exit_code,
                ended_at=_utc_now(),
                error=repr(error) if error is not None else None,
                trajectory_sha256=trajectory_sha256,
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Skill distillation: failed to finalize session %s: %s",
                self.session_id,
                exc,
            )

    def _ensure_layout(self) -> None:
        directories = (
            *_store_root_directories(self.project_path, self.namespace),
            self.store_path / "active",
            self.store_path / "candidates",
            self.store_path / "history",
            self.store_path / "reports",
            self.store_path / "sessions",
            self.session_dir,
        )
        for index, directory in enumerate(directories):
            _ensure_real_directory(directory, create=index > 0)

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
        trajectory_sha256: str | None = None,
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
        if self.run_context is not None:
            document["run_context"] = self.run_context
        if error is not None:
            document["error"] = error
        if trajectory_sha256 is not None:
            document["trajectory_sha256"] = trajectory_sha256
        return document

    def _write_session(
        self,
        *,
        status: str,
        exit_code: int | None = None,
        ended_at: str | None = None,
        error: str | None = None,
        trajectory_sha256: str | None = None,
    ) -> None:
        _write_json(
            self.session_path,
            self._session_document(
                status=status,
                exit_code=exit_code,
                ended_at=ended_at,
                error=error,
                trajectory_sha256=trajectory_sha256,
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
        _append_json_line_no_follow(self.ledger_path, entry)


def _validate_safe_component(value: str, kind: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or value in {"", ".", ".."}
        or _SAFE_COMPONENT.fullmatch(value) is None
    ):
        raise ValueError(
            f"{kind} must be a non-empty safe local path component containing only "
            "letters, numbers, dot, underscore, and hyphen"
        )
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _normalize_evidence_ids(evidence_ids: Sequence[str]) -> list[str]:
    if isinstance(evidence_ids, (str, bytes)) or not evidence_ids:
        raise ValueError("provenance requires at least one source evidence id")
    normalized = [
        _validate_safe_component(evidence_id, "source evidence id") for evidence_id in evidence_ids
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("source evidence ids must be unique")
    return normalized


def _normalize_skill_path(relative_path: str) -> str:
    message = "skill bundle paths must be safe relative paths ending in SKILL.md"
    if not isinstance(relative_path, str):
        raise ValueError(message)
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or relative_path != path.as_posix()
        or path.name != "SKILL.md"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(message)
    try:
        for component in path.parts[:-1]:
            _validate_safe_component(component, "skill directory")
    except ValueError:
        raise ValueError(message) from None
    return path.as_posix()


def _normalize_skills(skills: Mapping[str, str]) -> dict[str, bytes]:
    normalized: dict[str, bytes] = {}
    for relative_path, content in skills.items():
        path = _normalize_skill_path(relative_path)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"skill document {path!r} must contain non-empty text")
        normalized[path] = content.encode("utf-8")
    if "SKILL.md" not in normalized:
        raise ValueError("skill bundle must include a top-level SKILL.md index")
    return dict(sorted(normalized.items()))


def _normalize_json_object(value: Mapping[str, Any], field: str) -> JsonObject:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object with string keys")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
        decoded: object = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{field} must be a JSON object")
    return cast(JsonObject, decoded)


def _validation_status(validation: JsonObject) -> str:
    status = validation.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("validation.status must be a non-empty string")
    return status


def _write_candidate_bundle(
    bundle_path: Path,
    skills: Mapping[str, bytes],
    manifest: JsonObject,
) -> None:
    bundle_path.mkdir()
    _ensure_real_directory(bundle_path, create=False)
    for relative_path, content in skills.items():
        skill_path = bundle_path.joinpath(*PurePosixPath(relative_path).parts)
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_bytes(content)
        skill_path.chmod(0o644)
    _write_json(bundle_path / CANDIDATE_MANIFEST_NAME, manifest)
    (bundle_path / CANDIDATE_MANIFEST_NAME).chmod(0o644)


def _validate_candidate_bundle(
    bundle_path: Path,
    *,
    namespace: str,
    expected_candidate_id: str | None = None,
    require_passed: bool = False,
    evidence_validator: Callable[[str], None] | None = None,
) -> JsonObject:
    if bundle_path.is_symlink() or not bundle_path.is_dir():
        raise FileNotFoundError(f"skill candidate bundle does not exist: {bundle_path}")
    manifest_path = bundle_path / CANDIDATE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"candidate manifest is missing: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != SKILL_DISTILLATION_SCHEMA_VERSION:
        raise ValueError("candidate manifest has an unsupported schema version")
    if manifest.get("namespace") != namespace:
        raise ValueError("candidate manifest namespace does not match its store")
    candidate_id = _manifest_candidate_id(manifest)
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise ValueError("candidate manifest id does not match its immutable path")
    _required_text(manifest.get("generator"), "generator")
    _required_text(manifest.get("created_at"), "created_at")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("candidate manifest provenance must be an object")
    evidence_ids = provenance.get("source_evidence_ids")
    if not isinstance(evidence_ids, list):
        raise ValueError("candidate manifest provenance must list source evidence ids")
    normalized_evidence_ids = _normalize_evidence_ids(evidence_ids)

    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("candidate manifest validation must be an object")
    status = _validation_status(validation)
    if require_passed and status != "passed":
        raise PermissionError(
            f"candidate {candidate_id!r} cannot activate until validation status is passed"
        )

    expected_hashes = _manifest_skill_hashes(manifest)
    actual_hashes, actual_directories = _bundle_skill_hashes(bundle_path)
    expected_directories = _skill_parent_directories(expected_hashes)
    if actual_directories != expected_directories:
        raise ValueError("candidate bundle contains unexpected directories")
    if expected_hashes.keys() != actual_hashes.keys():
        raise ValueError("candidate bundle files do not match its skill manifest")
    for relative_path, expected_hash in expected_hashes.items():
        if actual_hashes[relative_path] != expected_hash:
            raise ValueError(f"candidate skill hash mismatch: {relative_path}")

    if evidence_validator is not None:
        for evidence_id in normalized_evidence_ids:
            evidence_validator(evidence_id)
    return manifest


def _validate_session_evidence(
    session_path: Path,
    *,
    evidence_id: str,
    namespace: str,
) -> None:
    if session_path.is_symlink() or not session_path.is_dir():
        raise ValueError(f"native session evidence is not a real directory: {session_path}")
    metadata_path = session_path / "session.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"native session evidence has no real session.json: {session_path}")
    metadata = _read_json_object_no_follow(metadata_path)
    schema_version = metadata.get("schema_version")
    turn_count = metadata.get("turn_count")
    if (
        isinstance(schema_version, bool)
        or schema_version != SKILL_DISTILLATION_SCHEMA_VERSION
        or metadata.get("session_id") != evidence_id
        or metadata.get("namespace") != namespace
        or metadata.get("status") != "completed"
        or isinstance(turn_count, bool)
        or not isinstance(turn_count, int)
        or turn_count <= 0
    ):
        raise ValueError(
            "native session evidence must be completed, non-empty, and match its store: "
            f"{metadata_path}"
        )

    turns_path = session_path / "turns.jsonl"
    if turns_path.is_symlink() or not turns_path.is_file():
        raise ValueError(f"native session evidence has no real turns.jsonl: {session_path}")
    turns_content = _read_regular_file_no_follow(turns_path)
    try:
        turn_lines = turns_content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"native session trajectory is not UTF-8: {turns_path}") from exc
    if len(turn_lines) != turn_count:
        raise ValueError(
            f"native session trajectory count does not match session.json: {turns_path}"
        )
    for expected_index, line in enumerate(turn_lines):
        try:
            turn: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"native session trajectory contains invalid JSON: {turns_path}"
            ) from exc
        turn_schema = turn.get("schema_version") if isinstance(turn, dict) else None
        turn_index = turn.get("turn_index") if isinstance(turn, dict) else None
        if (
            not isinstance(turn, dict)
            or isinstance(turn_schema, bool)
            or turn_schema != SKILL_DISTILLATION_SCHEMA_VERSION
            or turn.get("session_id") != evidence_id
            or isinstance(turn_index, bool)
            or not isinstance(turn_index, int)
            or turn_index != expected_index
        ):
            raise ValueError(
                "native session trajectory records must match the session and have contiguous "
                f"turn indexes: {turns_path}"
            )

    expected_hash = metadata.get("trajectory_sha256")
    actual_hash = f"sha256:{hashlib.sha256(turns_content).hexdigest()}"
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
        or expected_hash != actual_hash
    ):
        raise ValueError(f"native session trajectory hash mismatch: {turns_path}")


def _manifest_candidate_id(manifest: JsonObject) -> str:
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ValueError("candidate manifest candidate_id must be a string")
    return _validate_safe_component(candidate_id, "candidate id")


def _manifest_skill_hashes(manifest: JsonObject) -> dict[str, str]:
    raw_entries = manifest.get("skills")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("candidate manifest must hash every SKILL.md document")
    hashes: dict[str, str] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("candidate manifest skill entries must be objects")
        relative_path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(relative_path, str):
            raise ValueError("candidate manifest skill path must be a string")
        relative_path = _normalize_skill_path(relative_path)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"candidate manifest has an invalid SHA-256 hash: {relative_path}")
        if relative_path in hashes:
            raise ValueError(f"candidate manifest repeats skill path: {relative_path}")
        hashes[relative_path] = digest
    if "SKILL.md" not in hashes:
        raise ValueError("candidate manifest must include the top-level SKILL.md index")
    return hashes


def _bundle_skill_hashes(bundle_path: Path) -> tuple[dict[str, str], set[str]]:
    hashes: dict[str, str] = {}
    directories: set[str] = set()
    for path in bundle_path.rglob("*"):
        relative_path = path.relative_to(bundle_path).as_posix()
        if path.is_symlink():
            raise ValueError(f"candidate bundle cannot contain symlinks: {relative_path}")
        if path.is_dir():
            directories.add(relative_path)
            continue
        if not path.is_file():
            raise ValueError(f"candidate bundle contains an unsupported entry: {relative_path}")
        if path.stat().st_mode & 0o111:
            raise ValueError(f"candidate bundle files cannot be executable: {relative_path}")
        if relative_path == CANDIDATE_MANIFEST_NAME:
            continue
        relative_path = _normalize_skill_path(relative_path)
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes, directories


def _skill_parent_directories(skill_hashes: Mapping[str, str]) -> set[str]:
    directories: set[str] = set()
    for relative_path in skill_hashes:
        path = PurePosixPath(relative_path)
        for parent in path.parents:
            if parent != PurePosixPath("."):
                directories.add(parent.as_posix())
    return directories


def _existing_candidate_created_at(
    candidate_path: Path,
    *,
    namespace: str,
    candidate_id: str,
) -> str | None:
    manifest_path = candidate_path / CANDIDATE_MANIFEST_NAME
    if candidate_path.is_symlink() or manifest_path.is_symlink():
        return None
    try:
        manifest = _read_json_object(manifest_path)
    except (OSError, ValueError):
        return None
    if manifest.get("namespace") != namespace or manifest.get("candidate_id") != candidate_id:
        return None
    created_at = manifest.get("created_at")
    return created_at if isinstance(created_at, str) and created_at.strip() else None


def _read_json_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(JsonObject, value)


def _read_json_object_no_follow(path: Path) -> JsonObject:
    try:
        value: object = json.loads(_read_regular_file_no_follow(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(JsonObject, value)


def _read_json_lines_no_follow(path: Path) -> list[JsonObject]:
    try:
        lines = _read_regular_file_no_follow(path).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 JSON lines file: {path}") from exc
    entries: list[JsonObject] = []
    for line in lines:
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON lines file: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON lines entries must be objects: {path}")
        entries.append(cast(JsonObject, value))
    return entries


def _directories_identical(left: Path, right: Path) -> bool:
    left_snapshot = _directory_snapshot(left)
    right_snapshot = _directory_snapshot(right)
    return left_snapshot is not None and left_snapshot == right_snapshot


def _directory_snapshot(path: Path) -> tuple[set[str], dict[str, bytes]] | None:
    if path.is_symlink() or not path.is_dir():
        return None
    directories: set[str] = set()
    files: dict[str, bytes] = {}
    for child in path.rglob("*"):
        if child.is_symlink():
            return None
        relative_path = child.relative_to(path).as_posix()
        if child.is_dir():
            directories.add(relative_path)
        elif child.is_file():
            files[relative_path] = child.read_bytes()
        else:
            return None
    return directories, files


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _store_root_directories(project_path: Path, namespace: str) -> tuple[Path, ...]:
    current = project_path
    directories = [current]
    for component in (*SKILL_DISTILLATION_ROOT.parts, namespace):
        current /= component
        directories.append(current)
    return tuple(directories)


def _ensure_real_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked skill-distillation directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"skill-distillation path is not a directory: {path}")
        return
    if not create:
        raise FileNotFoundError(f"skill-distillation directory does not exist: {path}")
    try:
        path.mkdir()
    except FileExistsError:
        pass
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"could not create a real skill-distillation directory: {path}")


def _require_absent_path(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked {label}: {path}")
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def _remove_staging_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _open_regular_no_follow(path: Path, flags: int, *, mode: int) -> int:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked skill-distillation file: {path}")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags | no_follow | close_on_exec, mode)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError(f"refusing symlinked skill-distillation file: {path}") from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"skill-distillation file is not a private regular file: {path}")
    return descriptor


def _lock_descriptor(descriptor: int) -> Literal["fcntl", "msvcrt", "local"]:
    if _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_EX)
        return "fcntl"
    if _msvcrt is not None:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_LOCK, 1)
        return "msvcrt"
    return "local"


def _unlock_descriptor(
    descriptor: int,
    lock_kind: Literal["fcntl", "msvcrt", "local"],
) -> None:
    if lock_kind == "fcntl" and _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_UN)
    elif lock_kind == "msvcrt" and _msvcrt is not None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_UNLCK, 1)


def _append_json_line_no_follow(path: Path, entry: JsonObject) -> None:
    payload = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode("utf-8")
    descriptor = _open_regular_no_follow(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        mode=0o600,
    )
    initial_size = os.lseek(descriptor, 0, os.SEEK_END)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short write while appending {path}")
        os.fsync(descriptor)
    except BaseException:
        os.ftruncate(descriptor, initial_size)
        os.fsync(descriptor)
        raise
    finally:
        os.close(descriptor)


def _read_regular_file_no_follow(path: Path) -> bytes:
    descriptor = _open_regular_no_follow(path, os.O_RDONLY, mode=0o600)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _sha256_regular_file(path: Path) -> str:
    return hashlib.sha256(_read_regular_file_no_follow(path)).hexdigest()


def _rename_directory(source: Path, target: Path) -> None:
    source.rename(target)


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


def _write_json_atomic_no_follow(path: Path, data: JsonObject) -> None:
    _require_absent_path(path, "activation transaction journal")
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    _require_absent_path(temporary, "activation transaction staging file")
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = _open_regular_no_follow(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode=0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_session_id(target: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{target}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ACTIVATION_LEDGER_NAME",
    "CANDIDATE_MANIFEST_NAME",
    "SKILL_DISTILLATION_SCHEMA_VERSION",
    "SkillActivationRecord",
    "SkillDistillationMigrationError",
    "SkillDistillationSessionCapture",
    "SkillDistillationStore",
    "SkillDistillationStoreSummary",
    "resolve_skill_distillation_store_path",
    "summarize_skill_distillation_store",
]

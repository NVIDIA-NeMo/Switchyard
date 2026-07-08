# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable skill candidates and active-bundle history."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import switchyard.lib.skill_distillation_store as store_module
from switchyard.lib.skill_distillation_store import (
    SkillActivationRecord,
    SkillDistillationMigrationError,
    SkillDistillationSessionCapture,
    SkillDistillationStore,
    summarize_skill_distillation_store,
)
from switchyard.lib.stats_accumulator import StatsAccumulator

NAMESPACE = "tooluniverse-trialqa"
EVIDENCE_ID = "codex-session-1"


def _create_evidence(store: SkillDistillationStore, evidence_id: str = EVIDENCE_ID) -> None:
    evidence_path = store.store_path / "sessions" / evidence_id
    evidence_path.mkdir(parents=True)
    turns_content = (
        json.dumps(
            {
                "schema_version": 1,
                "session_id": evidence_id,
                "turn_index": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (evidence_path / "turns.jsonl").write_text(turns_content, encoding="utf-8")
    (evidence_path / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": evidence_id,
                "namespace": NAMESPACE,
                "status": "completed",
                "turn_count": 1,
                "trajectory_sha256": (
                    f"sha256:{hashlib.sha256(turns_content.encode()).hexdigest()}"
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _save_candidate(
    store: SkillDistillationStore,
    candidate_id: str,
    *,
    status: str = "passed",
    evidence_ids: list[str] | None = None,
    index: str | None = None,
) -> Path:
    return store.save_candidate(
        candidate_id=candidate_id,
        skills={
            "SKILL.md": index or f"# {candidate_id}\n",
            "trialqa/SKILL.md": f"# TrialQA {candidate_id}\n",
        },
        generator="nvidia/nemotron-3-ultra",
        evidence_ids=evidence_ids or [EVIDENCE_ID],
        validation={"status": status, "checks": ["answer-leakage"]},
        created_at="2026-07-06T12:00:00Z",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_bundles(store: SkillDistillationStore) -> list[Path]:
    return sorted(path for path in store.history_path.iterdir() if path.is_dir())


def test_save_candidate_writes_explicit_manifest_and_hashes_every_skill(
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)

    candidate_path = _save_candidate(store, "candidate-1")

    manifest = _read_json(candidate_path / "manifest.json")
    assert set(manifest) == {
        "candidate_id",
        "created_at",
        "generator",
        "namespace",
        "provenance",
        "schema_version",
        "skills",
        "validation",
    }
    assert manifest["schema_version"] == 1
    assert manifest["namespace"] == NAMESPACE
    assert manifest["candidate_id"] == "candidate-1"
    assert manifest["generator"] == "nvidia/nemotron-3-ultra"
    assert manifest["provenance"] == {"source_evidence_ids": [EVIDENCE_ID]}
    assert manifest["validation"]["status"] == "passed"
    assert manifest["created_at"] == "2026-07-06T12:00:00Z"
    assert manifest["skills"] == [
        {
            "path": "SKILL.md",
            "sha256": hashlib.sha256(b"# candidate-1\n").hexdigest(),
        },
        {
            "path": "trialqa/SKILL.md",
            "sha256": hashlib.sha256(b"# TrialQA candidate-1\n").hexdigest(),
        },
    ]
    assert {path.name for path in candidate_path.rglob("*") if path.is_file()} == {
        "SKILL.md",
        "manifest.json",
    }
    assert all(
        path.stat().st_mode & 0o111 == 0 for path in candidate_path.rglob("*") if path.is_file()
    )


@pytest.mark.parametrize(
    ("candidate_id", "evidence_ids"),
    [
        ("../candidate", [EVIDENCE_ID]),
        ("candidate/one", [EVIDENCE_ID]),
        ("candidate-1", ["../session"]),
        ("candidate-1", ["session/one"]),
    ],
)
def test_save_candidate_rejects_unsafe_identifiers(
    tmp_path: Path,
    candidate_id: str,
    evidence_ids: list[str],
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)

    with pytest.raises(ValueError, match="safe local path component"):
        _save_candidate(store, candidate_id, evidence_ids=evidence_ids)


def test_save_candidate_requires_source_evidence(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)

    with pytest.raises(ValueError, match="at least one source evidence id"):
        store.save_candidate(
            candidate_id="candidate-1",
            skills={"SKILL.md": "# Index\n"},
            generator="generator",
            evidence_ids=[],
            validation={"status": "passed"},
        )


@pytest.mark.parametrize(
    "skills",
    [
        {"task/SKILL.md": "# Task\n"},
        {"SKILL.md": "# Index\n", "task/helper.py": "print('no')\n"},
        {"SKILL.md": "# Index\n", "../SKILL.md": "# Escape\n"},
    ],
)
def test_save_candidate_accepts_only_skill_documents_with_top_level_index(
    tmp_path: Path,
    skills: dict[str, str],
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)

    with pytest.raises(ValueError, match="SKILL.md"):
        store.save_candidate(
            candidate_id="candidate-1",
            skills=skills,
            generator="generator",
            evidence_ids=[EVIDENCE_ID],
            validation={"status": "passed"},
        )


def test_candidate_save_is_idempotent_only_for_identical_content(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    first = store.save_candidate(
        candidate_id="candidate-1",
        skills={"SKILL.md": "# Index\n"},
        generator="generator",
        evidence_ids=[EVIDENCE_ID],
        validation={"status": "passed"},
    )
    original_manifest = (first / "manifest.json").read_bytes()

    second = store.save_candidate(
        candidate_id="candidate-1",
        skills={"SKILL.md": "# Index\n"},
        generator="generator",
        evidence_ids=[EVIDENCE_ID],
        validation={"status": "passed"},
    )

    assert second == first
    assert (first / "manifest.json").read_bytes() == original_manifest
    with pytest.raises(FileExistsError, match="immutable"):
        store.save_candidate(
            candidate_id="candidate-1",
            skills={"SKILL.md": "# Changed\n"},
            generator="generator",
            evidence_ids=[EVIDENCE_ID],
            validation={"status": "passed"},
        )
    assert (first / "SKILL.md").read_text(encoding="utf-8") == "# Index\n"


def test_activation_requires_passed_validation_and_existing_evidence(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _save_candidate(store, "failed-candidate", status="failed")
    _save_candidate(
        store,
        "missing-evidence",
        evidence_ids=["missing-session"],
    )

    with pytest.raises(PermissionError, match="validation status is passed"):
        store.activate("failed-candidate")
    with pytest.raises(FileNotFoundError, match="missing-session"):
        store.activate("missing-evidence")
    assert not (store.active_path / "SKILL.md").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("session_id", "another-session"),
        ("namespace", "another-namespace"),
        ("status", "failed"),
        ("turn_count", 0),
    ],
)
def test_activation_rejects_invalid_native_session_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    metadata_path = store.sessions_path / EVIDENCE_ID / "session.json"
    metadata = _read_json(metadata_path)
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    _save_candidate(store, "candidate-1")

    with pytest.raises(ValueError, match="completed, non-empty, and match"):
        store.activate("candidate-1")


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "symlink",
        "invalid-json",
        "wrong-session",
        "noncontiguous",
        "count-mismatch",
        "hash-mismatch",
    ],
)
def test_activation_rejects_invalid_native_trajectory(
    tmp_path: Path,
    corruption: str,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    session_path = store.sessions_path / EVIDENCE_ID
    turns_path = session_path / "turns.jsonl"
    metadata_path = session_path / "session.json"
    metadata = _read_json(metadata_path)

    if corruption == "missing":
        turns_path.unlink()
    elif corruption == "symlink":
        turns_path.unlink()
        victim_path = tmp_path / "turns.jsonl"
        victim_path.write_text("{}\n", encoding="utf-8")
        turns_path.symlink_to(victim_path)
    elif corruption == "invalid-json":
        turns_path.write_text("not-json\n", encoding="utf-8")
    elif corruption == "wrong-session":
        content = (
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "another-session",
                    "turn_index": 0,
                }
            )
            + "\n"
        )
        turns_path.write_text(content, encoding="utf-8")
        metadata["trajectory_sha256"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    elif corruption == "noncontiguous":
        content = "".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": EVIDENCE_ID,
                    "turn_index": index,
                }
            )
            + "\n"
            for index in (0, 2)
        )
        turns_path.write_text(content, encoding="utf-8")
        metadata["turn_count"] = 2
        metadata["trajectory_sha256"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    elif corruption == "count-mismatch":
        metadata["turn_count"] = 2
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    else:
        content = (
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": EVIDENCE_ID,
                    "turn_index": 0,
                    "tampered": True,
                }
            )
            + "\n"
        )
        turns_path.write_text(content, encoding="utf-8")

    _save_candidate(store, "candidate-1")
    with pytest.raises(ValueError, match="native session"):
        store.activate("candidate-1")


def test_store_rejects_symlinked_root_without_writing_target(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    external_path = tmp_path / "external"
    project_path.mkdir()
    external_path.mkdir()
    (project_path / ".switchyard").symlink_to(external_path, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked"):
        SkillDistillationStore(NAMESPACE, project_path)

    assert list(external_path.iterdir()) == []


def test_session_capture_rejects_symlinked_root_without_writing_target(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    external_path = tmp_path / "external"
    project_path.mkdir()
    external_path.mkdir()
    (project_path / ".switchyard").symlink_to(external_path, target_is_directory=True)

    with pytest.raises(OSError, match="unsafe or unavailable"):
        SkillDistillationSessionCapture(
            namespace=NAMESPACE,
            launch_target="codex",
            display_model="model",
            project_dir=project_path,
        )

    assert list(external_path.iterdir()) == []


@pytest.mark.parametrize(
    ("namespace", "launch_target"),
    [
        ("bad/namespace", "codex"),
        ("..", "codex"),
        (NAMESPACE, "bad/target"),
        (NAMESPACE, ".."),
    ],
)
def test_session_capture_validates_path_identities_before_writing(
    tmp_path: Path,
    namespace: str,
    launch_target: str,
) -> None:
    with pytest.raises(ValueError, match="safe local path component"):
        SkillDistillationSessionCapture(
            namespace=namespace,
            launch_target=launch_target,
            display_model="model",
            project_dir=tmp_path,
        )

    assert not (tmp_path / ".switchyard").exists()


def test_save_rejects_symlinked_staging_path_without_writing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    external_path = tmp_path / "external"
    external_path.mkdir()
    staging_path = store.candidates_path / ".candidate-1-fixed.tmp"
    staging_path.symlink_to(external_path, target_is_directory=True)

    class FixedUuid:
        hex = "fixed"

    monkeypatch.setattr(store_module.uuid, "uuid4", FixedUuid)

    with pytest.raises(ValueError, match="symlinked candidate staging"):
        _save_candidate(store, "candidate-1")

    assert list(external_path.iterdir()) == []


def test_process_local_lock_fallback_keeps_store_usable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(store_module, "_fcntl", None)
    monkeypatch.setattr(store_module, "_msvcrt", None)
    store = SkillDistillationStore(NAMESPACE, tmp_path)

    candidate_path = _save_candidate(store, "candidate-1")

    assert candidate_path.is_dir()
    assert store.lock_path.is_file()


def test_public_adapter_lock_rejects_nested_candidate_operations(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)

    with store.exclusive_lock():
        with pytest.raises(RuntimeError, match="not reentrant"):
            _save_candidate(store, "candidate-1")
        second_handle = SkillDistillationStore(NAMESPACE, tmp_path)
        with pytest.raises(RuntimeError, match="not reentrant"):
            with second_handle.exclusive_lock():
                pass

    assert _save_candidate(store, "candidate-1").is_dir()


def test_session_ledger_symlink_fails_open_without_writing_target(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    session = SkillDistillationSessionCapture(
        namespace=NAMESPACE,
        launch_target="codex",
        display_model="model",
        project_dir=tmp_path,
    )
    session.record_turn({"messages": []})
    victim_path = tmp_path / "victim.txt"
    victim_path.write_text("unchanged\n", encoding="utf-8")
    session.ledger_path.symlink_to(victim_path)

    session.finish(exit_code=0, stats=StatsAccumulator())

    assert victim_path.read_text(encoding="utf-8") == "unchanged\n"
    assert "failed to finalize session" in caplog.text


def test_session_turns_symlink_fails_open_without_writing_target(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    session = SkillDistillationSessionCapture(
        namespace=NAMESPACE,
        launch_target="codex",
        display_model="model",
        project_dir=tmp_path,
    )
    victim_path = tmp_path / "victim.txt"
    victim_path.write_text("unchanged\n", encoding="utf-8")
    session.turns_path.symlink_to(victim_path)

    session.record_turn({"messages": []})

    assert victim_path.read_text(encoding="utf-8") == "unchanged\n"
    assert "failed to write turn" in caplog.text


def test_finalized_capture_records_trajectory_hash(tmp_path: Path) -> None:
    session = SkillDistillationSessionCapture(
        namespace=NAMESPACE,
        launch_target="codex",
        display_model="model",
        project_dir=tmp_path,
    )
    session.record_turn({"messages": []})

    session.finish(exit_code=0, stats=StatsAccumulator())

    metadata = _read_json(session.session_path)
    assert metadata["trajectory_sha256"] == (
        f"sha256:{hashlib.sha256(session.turns_path.read_bytes()).hexdigest()}"
    )


def test_legacy_active_skill_requires_migration_before_activation(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    (store.active_path / "SKILL.md").write_text("# Legacy\n", encoding="utf-8")

    with pytest.raises(SkillDistillationMigrationError, match="legacy active/SKILL.md"):
        store.activate("candidate-1")

    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# Legacy\n"
    assert _history_bundles(store) == []


def test_activation_preserves_history_and_rollback_restores_bundle(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    first_path = _save_candidate(store, "candidate-1")
    second_path = _save_candidate(store, "candidate-2")

    first_activation = store.activate("candidate-1")
    second_activation = store.activate("candidate-2")

    assert first_activation.active_candidate_id == "candidate-1"
    assert first_activation.previous_candidate_id is None
    assert second_activation.active_candidate_id == "candidate-2"
    assert second_activation.previous_candidate_id == "candidate-1"
    assert second_activation.history_path is not None
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-2\n"
    assert (second_activation.history_path / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "# candidate-1\n"
    assert first_path.is_dir()
    assert second_path.is_dir()

    rollback = store.rollback()

    assert rollback.operation == "rollback"
    assert rollback.active_candidate_id == "candidate-1"
    assert rollback.previous_candidate_id == "candidate-2"
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert _history_bundles(store) == []
    ledger = [
        json.loads(line)
        for line in store.activation_ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["operation"] for entry in ledger] == ["activate", "activate", "rollback"]
    summary = summarize_skill_distillation_store(NAMESPACE, tmp_path)
    assert summary.active_skill_path == store.active_path / "SKILL.md"
    assert summary.active_skill_exists is True


@pytest.mark.parametrize("crash_point", ["after-backup", "after-publish"])
def test_store_recovers_interrupted_active_publication(
    tmp_path: Path,
    crash_point: str,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    candidate_path = _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    staged_path = store.store_path / ".active-candidate-2-crash.tmp"
    backup_path = store.history_path / "crash-candidate-1"
    shutil.copytree(candidate_path, staged_path)
    store._write_transaction_journal(
        {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "operation": "activate",
            "transaction_id": "a" * 32,
            "staged_path": store._relative_store_path(staged_path),
            "backup_path": store._relative_store_path(backup_path),
            "preserve_backup": True,
        }
    )
    store.active_path.rename(backup_path)
    if crash_point == "after-publish":
        staged_path.rename(store.active_path)

    recovered = SkillDistillationStore(NAMESPACE, tmp_path)
    with recovered.exclusive_lock():
        pass

    assert (recovered.active_path / "SKILL.md").read_text(encoding="utf-8") == ("# candidate-1\n")
    assert not recovered.transaction_journal_path.exists()
    assert not staged_path.exists()
    assert not backup_path.exists()


def test_store_commits_journaled_publication_recorded_in_ledger(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    candidate_path = _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    staged_path = store.store_path / ".active-candidate-2-crash.tmp"
    backup_path = store.history_path / "crash-candidate-1"
    transaction_id = "b" * 32
    shutil.copytree(candidate_path, staged_path)
    store._write_transaction_journal(
        {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "operation": "activate",
            "transaction_id": transaction_id,
            "staged_path": store._relative_store_path(staged_path),
            "backup_path": store._relative_store_path(backup_path),
            "preserve_backup": True,
        }
    )
    store.active_path.rename(backup_path)
    staged_path.rename(store.active_path)
    store._append_activation_record(
        SkillActivationRecord(
            namespace=NAMESPACE,
            operation="activate",
            active_candidate_id="candidate-2",
            previous_candidate_id="candidate-1",
            recorded_at="2026-07-06T12:00:00Z",
            history_path=backup_path,
        ),
        transaction_id=transaction_id,
    )

    recovered = SkillDistillationStore(NAMESPACE, tmp_path)
    with recovered.exclusive_lock():
        pass

    assert (recovered.active_path / "SKILL.md").read_text(encoding="utf-8") == ("# candidate-2\n")
    assert (backup_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert not recovered.transaction_journal_path.exists()


@pytest.mark.parametrize("crash_point", ["after-backup", "after-publish"])
def test_store_recovers_interrupted_rollback(
    tmp_path: Path,
    crash_point: str,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    store.activate("candidate-2")
    history_path = _history_bundles(store)[0]
    displaced_path = store.store_path / ".rollback-crash.tmp"
    store._write_transaction_journal(
        {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "operation": "rollback",
            "transaction_id": "c" * 32,
            "history_path": store._relative_store_path(history_path),
            "displaced_path": store._relative_store_path(displaced_path),
        }
    )
    store.active_path.rename(displaced_path)
    if crash_point == "after-publish":
        history_path.rename(store.active_path)

    recovered = SkillDistillationStore(NAMESPACE, tmp_path)
    with recovered.exclusive_lock():
        pass

    assert (recovered.active_path / "SKILL.md").read_text(encoding="utf-8") == ("# candidate-2\n")
    assert (history_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert not displaced_path.exists()
    assert not recovered.transaction_journal_path.exists()


def test_activation_rejects_skill_content_that_no_longer_matches_manifest(tmp_path: Path) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    candidate_path = _save_candidate(store, "candidate-1")
    (candidate_path / "trialqa" / "SKILL.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        store.activate("candidate-1")


def test_symlinked_activation_ledger_cannot_escape_and_restores_active_state(
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    victim_path = tmp_path / "victim.txt"
    victim_path.write_text("unchanged\n", encoding="utf-8")
    store.activation_ledger_path.symlink_to(victim_path)

    with pytest.raises(ValueError, match="symlinked skill-distillation file"):
        store.activate("candidate-1")

    assert victim_path.read_text(encoding="utf-8") == "unchanged\n"
    assert list(store.active_path.iterdir()) == []
    assert _history_bundles(store) == []


def test_activation_ledger_failure_restores_previous_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    original_ledger = store.activation_ledger_path.read_bytes()

    def fail_ledger(_record: object, **_kwargs: object) -> None:
        raise OSError("simulated ledger failure")

    monkeypatch.setattr(store, "_append_activation_record", fail_ledger)

    with pytest.raises(OSError, match="simulated ledger failure"):
        store.activate("candidate-2")
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert _history_bundles(store) == []
    assert store.activation_ledger_path.read_bytes() == original_ledger


def test_failed_active_publish_restores_previous_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    real_rename = store_module._rename_directory

    def fail_new_active(source: Path, target: Path) -> None:
        if source.name.startswith(".active-") and target == store.active_path:
            raise OSError("simulated publish failure")
        real_rename(source, target)

    monkeypatch.setattr(store_module, "_rename_directory", fail_new_active)

    with pytest.raises(OSError, match="simulated publish failure"):
        store.activate("candidate-2")
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert _history_bundles(store) == []


def test_rollback_ledger_failure_restores_active_and_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    store.activate("candidate-2")
    history_path = _history_bundles(store)[0]
    original_ledger = store.activation_ledger_path.read_bytes()

    def fail_ledger(_record: object, **_kwargs: object) -> None:
        raise OSError("simulated ledger failure")

    monkeypatch.setattr(store, "_append_activation_record", fail_ledger)

    with pytest.raises(OSError, match="simulated ledger failure"):
        store.rollback()
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-2\n"
    assert history_path.is_dir()
    assert (history_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-1\n"
    assert store.activation_ledger_path.read_bytes() == original_ledger


def test_failed_rollback_restores_current_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SkillDistillationStore(NAMESPACE, tmp_path)
    _create_evidence(store)
    _save_candidate(store, "candidate-1")
    _save_candidate(store, "candidate-2")
    store.activate("candidate-1")
    store.activate("candidate-2")
    history_path = _history_bundles(store)[0]
    real_rename = store_module._rename_directory

    def fail_history_restore(source: Path, target: Path) -> None:
        if source == history_path and target == store.active_path:
            raise OSError("simulated rollback failure")
        real_rename(source, target)

    monkeypatch.setattr(store_module, "_rename_directory", fail_history_restore)

    with pytest.raises(OSError, match="simulated rollback failure"):
        store.rollback()
    assert (store.active_path / "SKILL.md").read_text(encoding="utf-8") == "# candidate-2\n"
    assert history_path.is_dir()

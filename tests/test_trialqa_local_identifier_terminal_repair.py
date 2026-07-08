# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import benchmark.trialqa_local_candidate_repair as base
import benchmark.trialqa_local_distiller as distiller
import benchmark.trialqa_local_identifier_terminal_repair as repair
from benchmark.trialqa_local_dataset import TrialQADataset
from switchyard.lib.skill_distillation_store import SkillDistillationStore


def _leaf_catalog() -> dict[str, Any]:
    return {
        "schema_version": "test",
        "tool_rules": [
            {"tool_name": "trialqa_search", "rules": [{"rule": "old search"}]},
            {"tool_name": "trialqa_get_study", "rules": [{"rule": "unchanged"}]},
        ],
        "workflow_rules": [{"rule": "unchanged workflow"}],
        "search_discipline_repair_mode": distiller.SEARCH_DISCIPLINE_REPAIR_MODE,
    }


def _repaired_leaf_catalog() -> dict[str, Any]:
    catalog = copy.deepcopy(_leaf_catalog())
    catalog["tool_rules"][0]["rules"] = [{"rule": "identifier terminal"}]
    catalog["identifier_terminal_repair_mode"] = distiller.IDENTIFIER_TERMINAL_REPAIR_MODE
    return catalog


def test_identifier_leaf_guard_rejects_non_search_change() -> None:
    parent = _leaf_catalog()
    repaired = _repaired_leaf_catalog()

    repair._assert_identifier_leaf_only(parent, repaired)
    repaired["workflow_rules"][0]["rule"] = "tampered workflow"

    with pytest.raises(distiller.TrialQADistillationError, match="outside the search leaf"):
        repair._assert_identifier_leaf_only(parent, repaired)


def test_v10_parent_loader_requires_exact_mode_stage_calls_and_122_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    supplied: dict[str, object] = {}

    def fake_load_parent(**kwargs: object) -> object:
        supplied.update(kwargs)
        return sentinel

    monkeypatch.setattr(base, "_load_parent", fake_load_parent)

    loaded = repair._load_parent(
        project_dir=tmp_path / "project",
        store_dir=tmp_path / "store",
        candidate_id="trialqa-v10",
        work_dir=tmp_path / "work",
    )

    assert loaded is sentinel
    assert supplied["expected_mode"] == distiller.SEARCH_DISCIPLINE_REPAIR_MODE
    assert supplied["expected_stage"] == "search_discipline_repair"
    assert supplied["expected_new_calls"] == repair.ZERO_CALLS
    assert supplied["expected_catalog_mode_field"] == "search_discipline_repair_mode"
    assert supplied["expected_evidence_count"] == 122


def _gate_report(semantic_result: dict[str, object]) -> dict[str, Any]:
    result = {
        "task_id": "q2-treatment",
        "question_ordinal": 2,
        "decision": "kill",
        "semantic_result": semantic_result,
        "checks": {
            "semantic_replay_passed": True,
            "search_arguments_valid": True,
            "at_most_three_searches": False,
            "unique_canonical_arguments": True,
            "unique_normalized_queries": True,
            "unique_title_resolution_found": True,
            "no_search_after_first_resolution": False,
            "next_call_is_expected_evidence_getter": False,
        },
        "kill_reasons": repair.EXPECTED_SEARCH_KILL_REASONS,
        "search_count": 7,
        "successful_execute_tool_count": 8,
        "resolution_index": 0,
        "post_resolution_search_count": 6,
        "repeated_argument_count": 0,
        "repeated_normalized_query_count": 0,
        "next_operation": repair.search_gate.SEARCH_OPERATION,
    }
    unsigned: dict[str, Any] = {
        "schema_version": repair.search_gate.SEARCH_GATE_SCHEMA_VERSION,
        "policy": {
            "name": repair.search_gate.SEARCH_GATE_POLICY,
            "performance_eligible": False,
            "condition": "treatment",
            "max_searches": 3,
            "model_calls": 0,
            "judge_calls": 0,
            "evidence_imports": 0,
            "network_calls": 0,
        },
        "manifest_id": "trialqa-full-test",
        "manifest_sha256": f"sha256:{'1' * 64}",
        "dataset": {
            "id": "EdisonScientific/labbench2",
            "revision": "rev",
            "parquet_sha256": "p",
        },
        "semantic_report_sha256": f"sha256:{'2' * 64}",
        "results": [result],
        "summary": {
            "checked_tasks": 1,
            "passed_tasks": 0,
            "killed_tasks": 1,
            "decision": "kill",
        },
    }
    return {**unsigned, "report_sha256": base._canonical_sha256(unsigned)}


def test_rehashed_v10_search_count_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic_result: dict[str, object] = {"task_id": "q2-treatment", "decision": "pass"}
    report = _gate_report(semantic_result)
    path = tmp_path / "search.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        repair.search_gate,
        "build_search_gate_report",
        lambda **_kwargs: report,
    )
    dataset = TrialQADataset(tmp_path / "data.parquet", "rev", "p", ())
    common = {
        "path": path,
        "semantic_report": {"report_sha256": f"sha256:{'2' * 64}"},
        "semantic_result": semantic_result,
        "descriptive": {
            "manifest_id": "trialqa-full-test",
            "dataset": {"id": "EdisonScientific/labbench2"},
        },
        "descriptive_binding": {"canonical_sha256": f"sha256:{'1' * 64}"},
        "dataset": dataset,
        "capture": tmp_path,
        "task": {"task_id": "q2-treatment"},
    }

    repair._load_search_gate_report(**common)
    report["results"][0]["post_resolution_search_count"] = 5
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = base._canonical_sha256(unsigned)
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(distiller.TrialQADistillationError, match="seven-search kill"):
        repair._load_search_gate_report(**common)


def test_execute_saves_once_inactive_and_reports_after_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SkillDistillationStore(distiller.NAMESPACE, project)
    evidence_ids = tuple(f"native-{index:032x}" for index in range(122))
    parent = base.ParentCandidate(
        project_dir=project.resolve(),
        store_dir=store.store_path.resolve(),
        candidate_id="trialqa-v10",
        candidate_path=store.candidates_path / "trialqa-v10",
        binding={
            "candidate_id": "trialqa-v10",
            "manifest_sha256": f"sha256:{'1' * 64}",
            "skill_sha256": f"sha256:{'2' * 64}",
        },
        catalog_binding={
            "run_id": "trialqa-search-repair-parent",
            "sha256": f"sha256:{'3' * 64}",
            "integrity_sha256": f"sha256:{'4' * 64}",
            "size_bytes": 1,
        },
        catalog={},
        evidence_ids=evidence_ids,
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    skill = "---\nname: test\n---\n"
    candidate_id = "trialqa-v11"
    validation: dict[str, Any] = {
        "status": "passed",
        "candidate_id": candidate_id,
        "source_evidence_ids": list(evidence_ids),
        "new_calls": dict(repair.ZERO_CALLS),
        "checks": {"candidate_remains_inactive": True},
        "artifacts": {
            "skill_sha256": f"sha256:{repair.hashlib.sha256(skill.encode()).hexdigest()}"
        },
    }
    plan = repair.IdentifierTerminalPlan(
        run_id="trialqa-identifier-terminal-test",
        run_path=tmp_path / "work" / "trialqa-identifier-terminal-test",
        parent=parent,
        manifest={
            "input_bindings": {
                "manifests": {"primary_untouched": {"content_id": "trialqa-full-primary"}}
            }
        },
        catalog={"schema_version": "test"},
        skill=skill,
        candidate_id=candidate_id,
        validation=validation,
        dataset_path=tmp_path / "data.parquet",
        descriptive_manifest_path=tmp_path / "descriptive.json",
        primary_manifest_path=tmp_path / "primary.json",
        capture_path=capture,
        semantic_report_path=tmp_path / "semantic.json",
        search_gate_report_path=tmp_path / "search.json",
    )
    monkeypatch.setattr(repair, "build_identifier_terminal_plan", lambda **_kwargs: plan)
    lock_calls = 0
    original_lock = SkillDistillationStore.exclusive_lock

    @contextmanager
    def counted_lock(instance: SkillDistillationStore) -> Iterator[None]:
        nonlocal lock_calls
        lock_calls += 1
        with original_lock(instance):
            yield

    monkeypatch.setattr(SkillDistillationStore, "exclusive_lock", counted_lock)

    result = repair.execute_identifier_terminal(plan)

    saved = json.loads((result.candidate_path / "manifest.json").read_text())
    assert lock_calls == 1
    assert saved["validation"]["new_calls"] == repair.ZERO_CALLS
    assert saved["provenance"]["source_evidence_ids"] == list(evidence_ids)
    assert not (store.active_path / "manifest.json").exists()
    assert result.report_path.exists()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import benchmark.trialqa_local_candidate_repair as base
import benchmark.trialqa_local_distiller as distiller
import benchmark.trialqa_local_search_repair as repair
from benchmark.trialqa_local_dataset import TrialQADataset


def _leaf_catalog() -> dict[str, Any]:
    return {
        "schema_version": "test",
        "tool_rules": [
            {"tool_name": "trialqa_search", "rules": [{"rule": "old search"}]},
            {"tool_name": "trialqa_get_study", "rules": [{"rule": "unchanged"}]},
        ],
        "workflow_rules": [
            {"rule": "Use trialqa_get_outcome_measures for q5."},
            {"rule": "Use trialqa_extract_adverse_events for q7."},
        ],
        "failure_modes": [],
        "gotchas": [],
        "mechanism_repair_mode": distiller.MECHANISM_REPAIR_MODE,
    }


def _repaired_leaf_catalog() -> dict[str, Any]:
    catalog = copy.deepcopy(_leaf_catalog())
    catalog["tool_rules"][0]["rules"] = [{"rule": "new bounded search"}]
    catalog["search_discipline_repair_mode"] = distiller.SEARCH_DISCIPLINE_REPAIR_MODE
    return catalog


def test_search_leaf_guard_rejects_any_non_search_change() -> None:
    parent = _leaf_catalog()
    repaired = _repaired_leaf_catalog()

    repair._assert_search_leaf_only(parent, repaired)
    repaired["workflow_rules"][0]["rule"] = "tampered q5"

    with pytest.raises(distiller.TrialQADistillationError, match="outside the search leaf"):
        repair._assert_search_leaf_only(parent, repaired)


def test_v9_parent_loader_passes_exact_stage_mode_and_evidence_contract(
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
        candidate_id="trialqa-parent",
        work_dir=tmp_path / "work",
    )

    assert loaded is sentinel
    assert supplied["expected_mode"] == distiller.MECHANISM_REPAIR_MODE
    assert supplied["expected_stage"] == "mechanism_repair"
    assert supplied["expected_new_calls"] == repair.ZERO_CALLS
    assert supplied["expected_catalog_mode_field"] == "mechanism_repair_mode"
    assert supplied["expected_evidence_count"] == 122


def _search_report(semantic_result: dict[str, object]) -> dict[str, Any]:
    result = {
        "task_id": "q2-treatment",
        "question_ordinal": 2,
        "decision": "kill",
        "semantic_result": semantic_result,
        "checks": {
            "semantic_replay_passed": True,
            "search_arguments_valid": True,
            "at_most_three_searches": False,
            "unique_canonical_arguments": False,
            "unique_normalized_queries": False,
            "unique_title_resolution_found": True,
            "no_search_after_first_resolution": False,
            "next_call_is_expected_evidence_getter": False,
        },
        "kill_reasons": repair.EXPECTED_SEARCH_KILL_REASONS,
        "search_count": 8,
        "successful_execute_tool_count": 9,
        "resolution_index": 0,
        "post_resolution_search_count": 7,
        "repeated_argument_count": 1,
        "repeated_normalized_query_count": 2,
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
        "dataset": {"id": "EdisonScientific/labbench2", "revision": "rev", "parquet_sha256": "p"},
        "semantic_report_sha256": f"sha256:{'2' * 64}",
        "results": [result],
        "summary": {"checked_tasks": 1, "passed_tasks": 0, "killed_tasks": 1, "decision": "kill"},
    }
    return {**unsigned, "report_sha256": base._canonical_sha256(unsigned)}


def test_rehashed_search_gate_count_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantic_result: dict[str, object] = {"task_id": "q2-treatment", "decision": "pass"}
    report = _search_report(semantic_result)
    path = tmp_path / "search.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        repair.search_gate,
        "build_search_gate_report",
        lambda **_kwargs: report,
    )
    dataset = TrialQADataset(tmp_path / "data.parquet", "rev", "p", ())
    descriptive = {
        "manifest_id": "trialqa-full-test",
        "dataset": {"id": "EdisonScientific/labbench2"},
    }
    common = {
        "path": path,
        "semantic_report": {"report_sha256": f"sha256:{'2' * 64}"},
        "semantic_result": semantic_result,
        "descriptive": descriptive,
        "descriptive_binding": {"canonical_sha256": f"sha256:{'1' * 64}"},
        "dataset": dataset,
        "capture": tmp_path,
        "task": {"task_id": "q2-treatment"},
    }

    repair._load_search_gate_report(**common)
    report["results"][0]["post_resolution_search_count"] = 6
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = base._canonical_sha256(unsigned)
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(distiller.TrialQADistillationError, match="eight-search kill"):
        repair._load_search_gate_report(**common)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import benchmark.trialqa_local_candidate_repair as repair
import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_distiller as distiller
from switchyard.lib.skill_distillation_store import SkillDistillationStore


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _catalog() -> dict[str, Any]:
    catalog = {
        "schema_version": distiller.FINAL_MERGE_SCHEMA,
        "skill_name": distiller.SKILL_NAME,
        "summary": "Resolve one trial with bounded search and targeted evidence retrieval.",
        "tool_rules": [
            {
                "tool_name": "trialqa_search",
                "rules": [
                    {
                        "when": "resolving the trial identifier before evidence retrieval",
                        "rule": (
                            "Use at most 3 semantically distinct searches; never repeat the same "
                            "query or arguments, never invent an acronym expansion, and stop "
                            "searching after an exact title match."
                        ),
                        "confidence": 1.0,
                        "source_patch_count": 1,
                    }
                ],
            }
        ],
        "workflow_rules": [
            {
                "rule": (
                    "After trialqa_search resolves the NCT id, select the field-specific getter: "
                    "trialqa_get_eligibility for criteria or thresholds, "
                    "trialqa_get_outcome_measures for outcome counts or timeFrames, "
                    "trialqa_get_descriptions for narrative or per-arm detail, and "
                    "trialqa_get_study for enrollment or structured summary fields."
                ),
                "rationale": "Search metadata does not contain every requested field.",
                "confidence": 1.0,
                "source_patch_count": 1,
            },
            {
                "rule": (
                    "Treat the listed field routes as non-exhaustive. If the selected slice lacks "
                    "the requested field, call another relevant getter whose documented output "
                    "can contain it."
                ),
                "rationale": "A second relevant slice can hold a missing field.",
                "confidence": 1.0,
                "source_patch_count": 1,
            },
            {
                "rule": (
                    "For intervention starting-dose, regimen, or ordered arm/group questions, if "
                    "the selected slice lacks direct support, use trialqa_extract_adverse_events "
                    "as a fallback evidence slice; inspect group titles/descriptions. Do not infer "
                    "starting, lowest, or highest values from outcome timeFrames or cohort labels. "
                    "Answer only when retrieved evidence directly supports every requested field. "
                    "Once it does, finalize; otherwise use another relevant getter."
                ),
                "rationale": "The exposed failure lacked direct intervention-group evidence.",
                "confidence": 1.0,
                "source_patch_count": 1,
                "provenance_stratum": "exposed-development",
            },
        ],
        "failure_modes": [],
        "gotchas": [],
        "compaction_mode": distiller.CACHED_CATALOG_TRANSPORT_MODE,
        "development_layer_mode": distiller.DEVELOPMENT_LAYER_MODE,
    }
    return distiller.adapt_compact_tool_contract(catalog, tool_contract="compact")


def _parent(tmp_path: Path) -> repair.ParentCandidate:
    project = tmp_path / "project"
    project.mkdir()
    store = SkillDistillationStore(distiller.NAMESPACE, project)
    catalog = _catalog()
    digest = "a" * 64
    return repair.ParentCandidate(
        project_dir=project.resolve(),
        store_dir=store.store_path.resolve(),
        candidate_id="trialqa-parent",
        candidate_path=store.candidates_path / "trialqa-parent",
        binding={
            "candidate_id": "trialqa-parent",
            "manifest_sha256": f"sha256:{digest}",
            "skill_sha256": f"sha256:{'b' * 64}",
        },
        catalog_binding={
            "run_id": "trialqa-development-parent",
            "sha256": f"sha256:{'c' * 64}",
            "integrity_sha256": f"sha256:{'d' * 64}",
            "size_bytes": 1,
        },
        catalog=catalog,
        evidence_ids=(f"native-{'1' * 32}", f"native-{'2' * 32}"),
    )


def _success_result() -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": '{"status":"success"}'}],
        "structured_content": None,
    }


def _events(operations: list[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": "thread"},
        {"type": "turn.started"},
    ]
    loader = {
        "id": "load",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "trialqa_load_active_skill",
        "arguments": {},
    }
    events.extend(
        [
            {"type": "item.started", "item": {**loader, "status": "in_progress"}},
            {
                "type": "item.completed",
                "item": {
                    **loader,
                    "status": "completed",
                    "error": None,
                    "result": _success_result(),
                },
            },
        ]
    )
    for index, operation in enumerate(operations):
        item = {
            "id": f"op-{index}",
            "type": "mcp_tool_call",
            "server": "tooluniverse",
            "tool": "execute_tool",
            "arguments": {
                "tool_name": operation,
                "arguments_json": '{"query_term":"SYNTHETIC-TRIAL"}',
            },
        }
        events.extend(
            [
                {"type": "item.started", "item": {**item, "status": "in_progress"}},
                {
                    "type": "item.completed",
                    "item": {
                        **item,
                        "status": "completed",
                        "error": None,
                        "result": _success_result(),
                    },
                },
            ]
        )
    events.append({"type": "turn.completed", "usage": {"input_tokens": 1}})
    return events


def _q2_capture(
    tmp_path: Path, parent: repair.ParentCandidate
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    capture = tmp_path / "trialqa-full-descriptive"
    pair_id = "trialqa-0002-abcdef123456-r001"
    task_id = f"{pair_id}-treatment"
    output = capture / "trialqa-local" / pair_id / "arms" / "treatment" / "outputs"
    output.mkdir(parents=True)
    events_path = output / "switchyard-codex.stdout.log"
    operations = ["ClinicalTrials_search_studies"] * 6 + ["ClinicalTrials_get_study"]
    events_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in _events(operations)) + "\n"
    )
    final_path = output / "codex-final.json"
    final_path.write_text('{"answer":"12 weeks and 4 weeks"}\n')
    answer_path = output.parent / "answer.txt"
    answer_path.write_text("12 weeks and 4 weeks\n")
    session = (
        capture
        / ".switchyard"
        / "skill-distillation"
        / distiller.NAMESPACE
        / "sessions"
        / "session-1"
    )
    session.mkdir(parents=True)
    stats = {
        "total_requests": 9,
        "total_errors": 0,
        "models": {
            distiller.EXECUTOR_MODEL: {"calls": 9, "errors": 0},
        },
        "openai_transport": {
            "physical_attempts": 9,
            "null_eof_retries": 0,
            "retry_usage_charges": 0,
            "unpriced_null_eof_retries": 0,
            "retry_token_sensitivity": {
                "prompt": 0,
                "completion": 0,
                "cached": 0,
                "cache_creation": 0,
                "reasoning": 0,
                "total": 0,
            },
        },
    }
    stats_path = session / "stats.json"
    _write_json(stats_path, stats)
    trajectory_path = session / "turns.jsonl"
    trajectory_path.write_text("{}\n")
    task = {
        "task_id": task_id,
        "pair_id": pair_id,
        "row_id": "row-2",
        "dataset_row_index": 2,
        "repeat_index": 1,
        "n_repeats": 5,
        "condition": "treatment",
    }
    _write_json(
        session / "session.json",
        {
            "status": "completed",
            "exit_code": 0,
            "turn_count": 9,
            "active_skill": {"loaded": True, **parent.binding},
            "run_context": {
                "task_id": task_id,
                "candidate_id": parent.candidate_id,
                "candidate_manifest_sha256": parent.binding["manifest_sha256"],
                "candidate_skill_sha256": parent.binding["skill_sha256"],
            },
        },
    )
    generation_path = output / "generation.json"
    artifact_paths = {
        "answer": answer_path,
        "codex_events": events_path,
        "final_output": final_path,
        "stats": stats_path,
        "trajectory": trajectory_path,
    }
    generation = {
        "schema_version": demo.GENERATION_SCHEMA_VERSION,
        "manifest_id": "trialqa-full-descriptive",
        "task_id": task_id,
        "pair_id": pair_id,
        "row_id": "row-2",
        "dataset_row_index": 2,
        "partition": "test",
        "condition": "treatment",
        "repeat_index": 1,
        "n_repeats": 5,
        "answer": "12 weeks and 4 weeks",
        "answer_source": "codex-output-last-message-json",
        "session_dir": str(session.resolve()),
        "stats_path": str(stats_path.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "codex_events_path": str(events_path.resolve()),
        "final_output_path": str(final_path.resolve()),
        "generation_path": str(generation_path.resolve()),
        "stats": stats,
        "usage": {},
        "artifact_sha256": {name: demo._sha256_file(path) for name, path in artifact_paths.items()},
    }
    _write_json(generation_path, generation)
    result = repair.regression._evaluate_generation(
        demo.load_generation_result(generation_path), repair.regression.MECHANISM_SPECS[2]
    )
    return capture, task, result


def test_q2_generation_recounts_operations_and_transport(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    capture, task, result = _q2_capture(tmp_path, parent)

    binding, literals = repair._validate_q2_generation(
        capture=capture.resolve(),
        manifest_id="trialqa-full-descriptive",
        task=task,
        result=result,
        parent=parent,
    )

    assert binding["logical_requests"] == binding["physical_attempts"] == 9
    assert binding["null_eof_retries"] == 0
    assert binding["operation_counts"] == repair.EXPECTED_OPERATIONS
    assert "12 weeks" in literals
    assert "SYNTHETIC-TRIAL" in literals


def test_operation_recount_exposes_eligibility_calls() -> None:
    events = _events(
        ["ClinicalTrials_search_studies"] * 5
        + ["ClinicalTrials_get_study", "get_clinical_trial_eligibility_criteria"]
    )

    counts = repair._successful_operation_counts(events)

    assert counts["ClinicalTrials_search_studies"] == 5
    assert counts["get_clinical_trial_eligibility_criteria"] == 1


def test_q2_generation_rejects_rehashed_report_with_wrong_answer(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    capture, task, result = _q2_capture(tmp_path, parent)
    generation_path = (
        capture
        / "trialqa-local"
        / str(task["pair_id"])
        / "arms"
        / "treatment"
        / "outputs"
        / "generation.json"
    )
    generation = json.loads(generation_path.read_text())
    generation["answer"] = "8 weeks and 2 weeks"
    _write_json(generation_path, generation)
    result["bindings"]["generation_sha256"] = demo._sha256_file(generation_path)

    with pytest.raises(distiller.TrialQADistillationError, match="recomputed generation"):
        repair._validate_q2_generation(
            capture=capture.resolve(),
            manifest_id="trialqa-full-descriptive",
            task=task,
            result=result,
            parent=parent,
        )


def test_q7_generation_rejects_report_only_pass_tamper(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    capture, task, _result = _q2_capture(tmp_path, parent)
    generation_path = (
        capture
        / "trialqa-local"
        / str(task["pair_id"])
        / "arms"
        / "treatment"
        / "outputs"
        / "generation.json"
    )
    generation = demo.load_generation_result(generation_path)
    tampered = repair.regression._evaluate_generation(
        generation, repair.regression.MECHANISM_SPECS[7]
    )
    tampered["decision"] = "pass"
    tampered["checks"] = dict.fromkeys(repair.EXPECTED_Q2_CHECKS, True)
    tampered["kill_reasons"] = []

    with pytest.raises(distiller.TrialQADistillationError, match="q7 result differs"):
        repair._validate_q7_generations(
            capture=capture.resolve(),
            manifest_id="trialqa-full-descriptive",
            tasks=[task],
            results=[tampered],
        )


def test_report_self_hash_is_required(tmp_path: Path) -> None:
    report = {
        "schema_version": "test",
        "policy": {
            "name": repair.regression.REGRESSION_POLICY,
            "performance_eligible": False,
            "allowed_question_ordinals": [2, 5, 7],
            "condition": "treatment",
            "model_calls": 0,
            "judge_calls": 0,
            "evidence_imports": 0,
        },
    }
    report["report_sha256"] = repair._canonical_sha256(report)
    path = tmp_path / "report.json"
    _write_json(path, report)
    report["schema_version"] = "tampered"
    _write_json(path, report)

    with pytest.raises(distiller.TrialQADistillationError, match="self-hash"):
        repair._load_report(path, "report")


def test_build_and_execute_saves_inactive_zero_call_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _parent(tmp_path)
    capture = tmp_path / "capture"
    capture.mkdir()
    groups = tuple(f"group-{index:03d}" for index in range(96))
    q2_task = {"task_id": "q2-treatment", "pair_id": "q2", "row_id": "row-2"}
    monkeypatch.setattr(repair, "_load_parent", lambda **_kwargs: parent)
    monkeypatch.setattr(
        repair,
        "_load_dataset",
        lambda _path: (
            object(),
            {
                "dataset_id": "EdisonScientific/labbench2",
                "file_sha256": f"sha256:{'4' * 64}",
            },
        ),
    )
    monkeypatch.setattr(
        repair,
        "_load_manifests",
        lambda *_args, **_kwargs: (
            {"manifest_id": "trialqa-full-descriptive"},
            {"manifest_id": "trialqa-full-primary"},
            {
                "descriptive": {"canonical_sha256": f"sha256:{'3' * 64}"},
                "primary_untouched": {"content_id": "trialqa-full-primary"},
                "primary_capture_started": False,
            },
            groups,
            {},
        ),
    )
    monkeypatch.setattr(
        repair,
        "_validate_reports",
        lambda **_kwargs: (
            q2_task,
            {"bindings": {}},
            {"q7_pass": {}, "q2_kill": {}},
            {},
        ),
    )
    monkeypatch.setattr(
        repair,
        "_expected_task",
        lambda _tasks, group, *, repeat: {
            "task_id": f"{group}-r{repeat}",
            "pair_id": f"{group}-r{repeat}",
            "repeat_index": repeat,
        },
    )
    monkeypatch.setattr(
        repair,
        "_validate_ledger_scope",
        lambda **_kwargs: {
            "file_sha256": f"sha256:{'5' * 64}",
            "terminal_states": {},
        },
    )
    monkeypatch.setattr(
        repair,
        "_validate_q2_generation",
        lambda **_kwargs: (
            {
                "logical_requests": 9,
                "physical_attempts": 9,
                "null_eof_retries": 0,
                "operation_counts": repair.EXPECTED_OPERATIONS,
            },
            ("NCT00000000", "12 weeks"),
        ),
    )
    monkeypatch.setattr(
        repair,
        "_validate_q7_generations",
        lambda **_kwargs: ([{"repeat_index": index} for index in range(1, 6)], ()),
    )

    plan = repair.build_candidate_repair_plan(
        parent_project_dir=parent.project_dir,
        parent_store_dir=parent.store_dir,
        parent_candidate_id=parent.candidate_id,
        dataset_path=tmp_path / "trialqa.parquet",
        descriptive_manifest=tmp_path / "descriptive.json",
        primary_manifest=tmp_path / "primary.json",
        capture_dir=capture,
        q7_pass_report=tmp_path / "q7.json",
        q2_kill_report=tmp_path / "q2.json",
        work_dir=tmp_path / "work",
    )
    result = repair.execute_candidate_repair(plan)

    manifest = json.loads((result.candidate_path / "manifest.json").read_text())
    assert result.model_call_count == 0
    assert manifest["provenance"]["source_evidence_ids"] == list(parent.evidence_ids)
    assert manifest["validation"]["new_calls"] == {
        "model": 0,
        "judge": 0,
        "evidence_import": 0,
        "network": 0,
    }
    assert not (parent.store_dir / "active" / "manifest.json").exists()
    assert (
        "trialqa_get_eligibility first"
        in (result.candidate_path / distiller.SKILL_PATH).read_text()
    )
    assert (
        "trialqa_get_outcome_measures for outcome counts"
        in (result.candidate_path / distiller.SKILL_PATH).read_text()
    )


def test_mechanism_repair_preserves_q7_rule_and_rejects_literals() -> None:
    parent = _catalog()
    parent_q7 = next(
        item
        for item in parent["workflow_rules"]
        if "trialqa_extract_adverse_events" in item["rule"]
    )

    catalog = distiller.layer_exposed_mechanism_repair_catalog(parent)
    skill = distiller.render_skill_markdown(catalog, tool_contract="compact")
    repaired_q7 = next(
        item
        for item in catalog["workflow_rules"]
        if "trialqa_extract_adverse_events" in item["rule"]
    )

    assert repaired_q7 == parent_q7
    assert "never search after resolution" in skill
    assert "never trialqa_get_study" in skill
    with pytest.raises(distiller.TrialQADistillationError, match="task-specific"):
        distiller._assert_no_sensitive(skill + " NCT03249792", (), "repair")

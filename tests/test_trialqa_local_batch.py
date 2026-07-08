# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import benchmark.trialqa_local_batch as batch
import benchmark.trialqa_local_demo as demo


def _openai_transport(
    total_requests: int,
    *,
    retries: int = 0,
    charges: int = 0,
    unpriced: int = 0,
    prompt: int = 0,
    completion: int = 0,
) -> dict[str, object]:
    return {
        "physical_attempts": total_requests + retries,
        "null_eof_retries": retries,
        "retry_usage_charges": charges,
        "unpriced_null_eof_retries": unpriced,
        "retry_token_sensitivity": {
            "prompt": prompt,
            "completion": completion,
            "cached": 0,
            "cache_creation": 0,
            "reasoning": 0,
            "total": prompt + completion,
        },
    }


def _task() -> dict[str, object]:
    return {
        "task_id": "pair-baseline",
        "pair_id": "pair",
        "row_id": "row-1",
        "dataset_row_index": 1,
        "question_group_key": "pair",
        "partition": "test",
        "phase": "evaluation",
        "condition": "baseline",
        "repeat_index": 1,
        "n_repeats": 5,
    }


def _ab_tasks(pair_count: int) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"pair-{index}-{condition}",
            "pair_id": f"pair-{index}",
            "condition": condition,
        }
        for index in range(pair_count)
        for condition in ("baseline", "treatment")
    ]


def _clustered_ab_tasks(question_count: int, repeat_count: int) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"question-{question}-r{repeat}-{condition}",
            "pair_id": f"question-{question}-r{repeat}",
            "question_group_key": f"question-{question}",
            "repeat_index": repeat,
            "condition": condition,
        }
        for question in range(question_count)
        for repeat in range(1, repeat_count + 1)
        for condition in ("baseline", "treatment")
    ]


def _development_tasks(question_count: int, repeat_count: int) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"question-{question}-r{repeat}-treatment",
            "pair_id": f"question-{question}-r{repeat}",
            "question_group_key": f"question-{question}",
            "repeat_index": repeat,
            "condition": "treatment",
            "partition": "train",
            "phase": "development",
        }
        for question in range(question_count)
        for repeat in range(1, repeat_count + 1)
    ]


def _full_manifest(*, primary: bool = False) -> dict[str, object]:
    heldout_groups = [f"question-{index}" for index in range(96)]
    return {
        "kind": "full",
        "dataset": {
            "heldout_ordering": {
                "question_count": 96,
                "question_group_keys": heldout_groups,
                "question_group_keys_sha256": demo._sha256_bytes(
                    demo._canonical_json(heldout_groups)
                ),
            }
        },
        "protocol": {
            "max_generation_concurrency": demo.MAX_GENERATION_CONCURRENCY,
            "primary_evaluation_scope": (
                {
                    "question_start": 88,
                    "question_count": 8,
                    "repeat_count": 5,
                    "task_count": 80,
                    "question_group_keys_sha256": demo._sha256_bytes(
                        demo._canonical_json(heldout_groups[88:])
                    ),
                }
                if primary
                else None
            ),
            "heldout_quarantine": (
                {
                    "question_start": 0,
                    "question_count": 88,
                    "disposition": "excluded-exposed-heldout",
                    "question_group_keys_sha256": demo._sha256_bytes(
                        demo._canonical_json(heldout_groups[:88])
                    ),
                }
                if primary
                else None
            ),
            "performance_eligible": primary,
        },
    }


def test_pair_safe_chunks_use_balanced_crossover_order() -> None:
    waves = list(batch._pair_safe_chunks(_ab_tasks(8), 8))

    assert len(waves) == 2
    assert [task["condition"] for task in waves[0]] == [
        "baseline",
        "treatment",
        "baseline",
        "treatment",
        "baseline",
        "treatment",
        "baseline",
        "treatment",
    ]
    assert [task["condition"] for task in waves[1]] == [
        "treatment",
        "baseline",
        "treatment",
        "baseline",
        "treatment",
        "baseline",
        "treatment",
        "baseline",
    ]
    assert [task["pair_id"] for task in waves[0]] == [task["pair_id"] for task in waves[1]]
    assert all(len({task["pair_id"] for task in wave}) == len(wave) for wave in waves)


def test_pair_safe_chunks_balance_odd_worker_waves_and_ignore_arm_input_order() -> None:
    tasks = _ab_tasks(6)
    tasks = [
        task
        for pair_index in range(6)
        for task in reversed(tasks[pair_index * 2 : pair_index * 2 + 2])
    ]

    waves = list(batch._pair_safe_chunks(tasks, 3))

    assert len(waves) == 4
    for first, second in zip(waves[::2], waves[1::2], strict=True):
        assert [task["pair_id"] for task in first] == [task["pair_id"] for task in second]
        assert all(
            first_task["condition"] != second_task["condition"]
            for first_task, second_task in zip(first, second, strict=True)
        )
    assert all(
        abs(
            sum(task["condition"] == "baseline" for task in wave)
            - sum(task["condition"] == "treatment" for task in wave)
        )
        <= 1
        for wave in waves
    )


def test_pair_safe_chunks_keep_manifest_crossover_position_after_resume() -> None:
    all_tasks = _ab_tasks(4)
    pair_positions = {f"pair-{index}": index for index in range(4)}
    pending = [task for task in all_tasks if task["pair_id"] in {"pair-1", "pair-2"}]

    waves = list(
        batch._pair_safe_chunks(
            pending,
            2,
            pair_positions=pair_positions,
        )
    )

    assert [task["condition"] for task in waves[0]] == ["treatment", "baseline"]
    assert [task["condition"] for task in waves[1]] == ["baseline", "treatment"]


def test_pair_safe_chunks_reject_duplicate_ab_condition() -> None:
    tasks = _ab_tasks(1)
    tasks[1]["condition"] = "baseline"

    with pytest.raises(RuntimeError, match="one arm per condition"):
        list(batch._pair_safe_chunks(tasks, 2))


def test_limit_scope_is_stable_across_partial_and_completed_canary_resume() -> None:
    tasks = _ab_tasks(12)
    positions = {f"pair-{index}": index for index in range(12)}
    scoped = tasks[:16]
    first_wave = next(batch._pair_safe_chunks(scoped, 8, pair_positions=positions))
    states = {str(task["task_id"]): "completed" for task in first_wave}

    remaining = batch._scoped_pending_tasks(tasks, states, 16)

    assert len(remaining) == 8
    assert {task["pair_id"] for task in remaining} == {task["pair_id"] for task in scoped}
    states.update({str(task["task_id"]): "completed" for task in remaining})
    assert batch._scoped_pending_tasks(tasks, states, 16) == []


def test_evaluation_limit_rejects_incomplete_pair_but_donor_limit_is_unrestricted() -> None:
    with pytest.raises(RuntimeError, match="complete baseline/treatment pairs"):
        batch._scoped_pending_tasks(_ab_tasks(8), {}, 15)

    donor = [
        {"task_id": f"donor-{index}", "pair_id": f"donor-{index}", "condition": "donor"}
        for index in range(3)
    ]
    assert batch._scoped_pending_tasks(donor, {}, 1) == donor[:1]


def test_question_scope_round_robins_distinct_clusters_before_repeats() -> None:
    scoped = batch._select_task_scope(
        _clustered_ab_tasks(4, 3),
        limit=None,
        question_limit=3,
        repeat_limit=2,
        condition="both",
    )

    assert [task["pair_id"] for task in scoped[::2]] == [
        "question-0-r1",
        "question-1-r1",
        "question-2-r1",
        "question-0-r2",
        "question-1-r2",
        "question-2-r2",
    ]
    assert all(
        [first["condition"], second["condition"]] == ["baseline", "treatment"]
        for first, second in zip(scoped[::2], scoped[1::2], strict=True)
    )


@pytest.mark.parametrize("question_limit", [1, 4, 8])
def test_heldout_question_windows_start_after_exposed_quarantine(
    question_limit: int,
) -> None:
    scope = batch._build_task_scope(
        _clustered_ab_tasks(96, 2),
        limit=None,
        question_start=88,
        question_limit=question_limit,
        repeat_limit=1,
        condition="both",
        heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
    )
    metadata = scope.metadata("trialqa-full-selector-test")

    assert scope.selected_question_groups[0] == "question-88"
    assert scope.selected_question_groups[-1] == f"question-{87 + question_limit}"
    assert scope.selected_repeat_indices == (1,)
    assert scope.heldout_classification == "unexposed-heldout-evaluation"
    assert len(scope.tasks) == question_limit * 2
    assert all(
        [first["condition"], second["condition"]] == ["baseline", "treatment"]
        for first, second in zip(scope.tasks[::2], scope.tasks[1::2], strict=True)
    )
    assert metadata["question_start"] == 88
    assert metadata["question_limit"] == question_limit
    assert metadata["question_end_exclusive"] == 88 + question_limit
    assert metadata["selected_task_count"] == question_limit * 2
    assert metadata["attestation_sha256"] == batch._canonical_sha256(
        {key: value for key, value in metadata.items() if key != "attestation_sha256"}
    )


def test_primary_manifest_window_keeps_global_heldout_ordinals() -> None:
    scope = batch._build_task_scope(
        _clustered_ab_tasks(8, 1),
        limit=None,
        manifest_question_start=88,
        question_start=88,
        question_limit=8,
        repeat_limit=1,
        condition="both",
        heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
    )
    metadata = scope.metadata("trialqa-primary-window-test")

    assert len(scope.selected_question_groups) == 8
    assert metadata["manifest_question_start"] == 88
    assert metadata["question_start"] == 88
    assert metadata["question_end_exclusive"] == 96
    assert metadata["heldout_classification"] == "unexposed-heldout-evaluation"
    with pytest.raises(RuntimeError, match="precedes the manifest"):
        batch._build_task_scope(
            _clustered_ab_tasks(8, 1),
            limit=None,
            manifest_question_start=88,
            question_start=0,
            question_limit=8,
            repeat_limit=1,
            condition="both",
            heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
        )


@pytest.mark.parametrize(("question_start", "question_count"), [(89, 7), (95, 1)])
def test_later_primary_scope_uses_global_quarantine_coordinates(
    question_start: int,
    question_count: int,
) -> None:
    scope = batch._build_task_scope(
        _clustered_ab_tasks(question_count, 1),
        limit=None,
        manifest_question_start=question_start,
        question_start=question_start,
        question_limit=question_count,
        repeat_limit=1,
        condition="both",
        heldout_quarantine_questions=question_start,
    )
    metadata = scope.metadata("trialqa-later-primary-window-test")

    assert metadata["manifest_question_start"] == question_start
    assert metadata["question_start"] == question_start
    assert metadata["question_end_exclusive"] == 96
    assert metadata["heldout_quarantine_questions"] == question_start
    assert metadata["heldout_classification"] == "unexposed-heldout-evaluation"


def test_manifest_quarantine_boundary_is_dynamic_for_primary_and_descriptive() -> None:
    descriptive = _full_manifest()
    primary = _full_manifest(primary=True)
    later = json.loads(json.dumps(primary))
    heldout_groups = later["dataset"]["heldout_ordering"]["question_group_keys"]
    later["protocol"]["primary_evaluation_scope"] = {
        "question_start": 89,
        "question_count": 7,
        "repeat_count": 5,
        "task_count": 70,
        "question_group_keys_sha256": demo._sha256_bytes(demo._canonical_json(heldout_groups[89:])),
    }

    assert batch._manifest_heldout_quarantine_questions(descriptive) == 88
    assert batch._manifest_heldout_quarantine_questions(primary) == 88
    assert batch._manifest_heldout_quarantine_questions(later) == 89

    descriptive_scope = batch._build_manifest_task_scope(
        descriptive,
        _clustered_ab_tasks(96, 1),
        limit=None,
        question_start=0,
        question_limit=None,
        repeat_limit=None,
        condition="both",
    )
    primary_scope = batch._build_manifest_task_scope(
        primary,
        _clustered_ab_tasks(8, 1),
        limit=None,
        question_start=88,
        question_limit=8,
        repeat_limit=1,
        condition="both",
    )
    later_scope = batch._build_manifest_task_scope(
        later,
        _clustered_ab_tasks(7, 1),
        limit=None,
        question_start=89,
        question_limit=7,
        repeat_limit=1,
        condition="both",
    )

    assert descriptive_scope.metadata("descriptive")["heldout_quarantine_questions"] == 88
    assert descriptive_scope.heldout_classification == "descriptive-mixed-heldout"
    assert primary_scope.metadata("primary")["heldout_quarantine_questions"] == 88
    assert later_scope.metadata("later")["heldout_quarantine_questions"] == 89

    with pytest.raises(RuntimeError, match="must equal the manifest question start"):
        batch._build_task_scope(
            _clustered_ab_tasks(7, 1),
            limit=None,
            manifest_question_start=89,
            question_start=89,
            question_limit=7,
            repeat_limit=1,
            condition="both",
            heldout_quarantine_questions=88,
        )


def test_heldout_quarantine_scope_is_separate_and_crossing_is_rejected() -> None:
    tasks = _clustered_ab_tasks(96, 1)
    quarantine = batch._build_task_scope(
        tasks,
        limit=None,
        question_start=0,
        question_limit=88,
        repeat_limit=1,
        condition="both",
        heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
    )

    assert quarantine.selected_question_groups == tuple(f"question-{index}" for index in range(88))
    assert quarantine.heldout_classification == "exposed-heldout-quarantine"
    with pytest.raises(RuntimeError, match="cannot mix quarantined"):
        batch._build_task_scope(
            tasks,
            limit=None,
            question_start=87,
            question_limit=2,
            repeat_limit=1,
            condition="both",
            heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
        )
    with pytest.raises(RuntimeError, match="explicit --question-limit"):
        batch._build_task_scope(
            tasks,
            limit=None,
            question_start=0,
            question_limit=None,
            repeat_limit=None,
            condition="both",
            heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
        )


def test_default_full_scope_is_explicitly_descriptive_when_mixed() -> None:
    scope = batch._build_task_scope(
        _clustered_ab_tasks(96, 1),
        limit=None,
        question_start=0,
        question_limit=None,
        repeat_limit=None,
        condition="both",
        heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
        allow_descriptive_mixed_heldout=True,
    )
    metadata = scope.metadata("trialqa-descriptive-full-test")

    assert len(scope.tasks) == 96 * 2
    assert metadata["heldout_classification"] == "descriptive-mixed-heldout"
    assert metadata["question_end_exclusive"] == 96
    assert metadata["heldout_quarantine_questions"] == 88


@pytest.mark.parametrize("repeat_limit", [1, 2])
def test_development_regression_selects_question_seven_and_repeat_prefix(
    repeat_limit: int,
) -> None:
    scope = batch._build_task_scope(
        _development_tasks(24, 5),
        limit=None,
        question_start=7,
        question_limit=1,
        repeat_limit=repeat_limit,
        condition="both",
    )
    metadata = scope.metadata("trialqa-development-selector-test")

    assert [task["pair_id"] for task in scope.tasks] == [
        f"question-7-r{repeat}" for repeat in range(1, repeat_limit + 1)
    ]
    assert {task["condition"] for task in scope.tasks} == {"treatment"}
    assert metadata["question_start"] == 7
    assert metadata["question_end_exclusive"] == 8
    assert metadata["selected_repeat_indices"] == list(range(1, repeat_limit + 1))
    assert metadata["heldout_classification"] == "not-applicable"


def test_question_window_is_frozen_before_resume_filtering() -> None:
    scope = batch._build_task_scope(
        _clustered_ab_tasks(12, 2),
        limit=None,
        question_start=8,
        question_limit=2,
        repeat_limit=2,
        condition="both",
    )
    metadata = scope.metadata("trialqa-scope-resume-test")
    completed_pair = str(scope.tasks[0]["pair_id"])
    states = {
        str(task["task_id"]): "completed"
        for task in scope.tasks
        if task["pair_id"] == completed_pair
    }

    pending = batch._pending_tasks_for_stage(list(scope.tasks), states, "all")

    assert len(pending) == len(scope.tasks) - 2
    assert metadata == scope.metadata("trialqa-scope-resume-test")
    assert metadata["selected_task_count"] == len(scope.tasks)
    assert {task["question_group_key"] for task in pending} == {
        "question-8",
        "question-9",
    }


def test_question_window_rejects_out_of_range_and_incomplete_repeats() -> None:
    tasks = _clustered_ab_tasks(4, 3)
    with pytest.raises(RuntimeError, match="question start exceeds"):
        batch._build_task_scope(
            tasks,
            limit=None,
            question_start=4,
            question_limit=1,
            repeat_limit=1,
            condition="both",
        )
    with pytest.raises(RuntimeError, match="question limit exceeds"):
        batch._build_task_scope(
            tasks,
            limit=None,
            question_start=3,
            question_limit=2,
            repeat_limit=1,
            condition="both",
        )
    incomplete = [
        task
        for task in tasks
        if not (task["question_group_key"] == "question-1" and task["repeat_index"] == 2)
    ]
    with pytest.raises(RuntimeError, match="inconsistent repeat coverage"):
        batch._build_task_scope(
            incomplete,
            limit=None,
            question_start=0,
            question_limit=2,
            repeat_limit=1,
            condition="both",
        )


def test_question_scope_can_select_treatment_only_for_development() -> None:
    scoped = batch._select_task_scope(
        _clustered_ab_tasks(4, 3),
        limit=None,
        question_limit=2,
        repeat_limit=1,
        condition="treatment",
    )

    assert [task["pair_id"] for task in scoped] == [
        "question-0-r1",
        "question-1-r1",
    ]
    assert {task["condition"] for task in scoped} == {"treatment"}


def test_exposed_descriptive_generation_allows_treatment_only_q7() -> None:
    manifest = _full_manifest()

    batch._validate_single_arm_execution(
        manifest,
        condition="treatment",
        stage="generation",
        limit=None,
        question_start=7,
        question_limit=1,
        max_generation_attempts=1,
    )
    scope = batch._build_task_scope(
        _clustered_ab_tasks(96, 5),
        limit=None,
        question_start=7,
        question_limit=1,
        repeat_limit=1,
        condition="treatment",
        heldout_quarantine_questions=batch.EXPOSED_HELDOUT_QUARANTINE_QUESTIONS,
    )

    assert [task["task_id"] for task in scope.tasks] == ["question-7-r1-treatment"]
    assert scope.heldout_classification == "exposed-heldout-quarantine"


@pytest.mark.parametrize(
    (
        "manifest",
        "condition",
        "stage",
        "limit",
        "question_start",
        "question_limit",
        "max_generation_attempts",
        "match",
    ),
    [
        (_full_manifest(primary=True), "treatment", "generation", None, 7, 1, 1, "descriptive"),
        (_full_manifest(), "baseline", "generation", None, 7, 1, 1, "baseline"),
        (_full_manifest(), "treatment", "score", None, 7, 1, 1, "generation-only"),
        (_full_manifest(), "treatment", "all", None, 7, 1, 1, "generation-only"),
        (
            _full_manifest(),
            "treatment",
            "generation",
            None,
            7,
            1,
            2,
            "max-generation-attempts 1",
        ),
        (_full_manifest(), "treatment", "generation", 2, 7, 1, 1, "--limit"),
        (_full_manifest(), "treatment", "generation", None, 7, None, 1, "question-limit"),
        (_full_manifest(), "treatment", "generation", None, 8, 1, 1, "2, 5, or 7"),
        (_full_manifest(), "treatment", "generation", None, 7, 2, 1, "2, 5, or 7"),
        (_full_manifest(), "treatment", "generation", None, 3, 1, 1, "2, 5, or 7"),
    ],
)
def test_single_arm_execution_fails_closed_outside_exposed_generation(
    manifest: dict[str, object],
    condition: batch.ConditionScope,
    stage: batch.BatchStage,
    limit: int | None,
    question_start: int,
    question_limit: int | None,
    max_generation_attempts: int,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        batch._validate_single_arm_execution(
            manifest,
            condition=condition,
            stage=stage,
            limit=limit,
            question_start=question_start,
            question_limit=question_limit,
            max_generation_attempts=max_generation_attempts,
        )


def test_question_scope_accepts_explicit_development_manifest_without_arm_filter() -> None:
    tasks = [
        {
            **task,
            "partition": "train",
            "phase": "development",
        }
        for task in _clustered_ab_tasks(4, 3)
        if task["condition"] == "treatment"
    ]

    scoped = batch._select_task_scope(
        tasks,
        limit=None,
        question_limit=2,
        repeat_limit=1,
        condition="both",
    )

    assert [task["pair_id"] for task in scoped] == [
        "question-0-r1",
        "question-1-r1",
    ]


def test_question_scope_rejects_prefix_limit_combination() -> None:
    with pytest.raises(RuntimeError, match="cannot be combined"):
        batch._select_task_scope(
            _clustered_ab_tasks(2, 2),
            limit=2,
            question_limit=1,
            repeat_limit=None,
            condition="both",
        )


def test_stage_pending_tasks_separates_generation_from_scoring() -> None:
    scoped = _clustered_ab_tasks(2, 1)
    states = {
        "question-0-r1-baseline": "completed",
        "question-0-r1-treatment": "generation_completed",
    }

    assert batch._pending_tasks_for_stage(scoped, states, "generation") == [
        scoped[2],
        scoped[3],
    ]
    assert batch._pending_tasks_for_stage(scoped[:2], states, "score") == [scoped[1]]
    assert batch._pending_tasks_for_stage(scoped, states, "all") == scoped[1:]


@pytest.mark.parametrize("stage", ["generation", "all"])
def test_generation_stages_reject_workers_above_manifest_limit(
    stage: batch.BatchStage,
) -> None:
    with pytest.raises(RuntimeError, match="max_generation_concurrency 4"):
        batch._validate_manifest_generation_workers(
            _full_manifest(),
            stage=stage,
            workers=5,
        )


def test_manifest_worker_limit_allows_generation_at_limit_and_score_only_above_it() -> None:
    manifest = _full_manifest()

    batch._validate_manifest_generation_workers(manifest, stage="generation", workers=4)
    batch._validate_manifest_generation_workers(manifest, stage="all", workers=4)
    batch._validate_manifest_generation_workers(manifest, stage="score", workers=16)


def test_batch_rejects_excess_generation_workers_before_capture_or_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / "experiment"
    monkeypatch.setattr(
        batch.demo,
        "_read_json_object",
        lambda *_args: _full_manifest(),
    )
    monkeypatch.setattr(
        batch.demo,
        "ResumableLedger",
        lambda *_args, **_kwargs: pytest.fail("ledger must not be constructed"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trialqa_local_batch.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--dataset",
            str(tmp_path / "dataset.parquet"),
            "--experiment-root",
            str(experiment_root),
            "--doctor",
            str(tmp_path / "doctor.json"),
            "--switchyard",
            str(tmp_path / "switchyard"),
            "--codex",
            str(tmp_path / "codex"),
            "--tooluniverse",
            str(tmp_path / "tooluniverse"),
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--workers",
            "5",
        ],
    )

    with pytest.raises(RuntimeError, match="max_generation_concurrency 4"):
        batch.main()

    assert not experiment_root.exists()


def test_generation_stage_ignores_partial_scoring_states() -> None:
    scoped = _clustered_ab_tasks(3, 1)
    states = {
        str(scoped[0]["task_id"]): "score_retry_started",
        str(scoped[1]["task_id"]): "scored",
        str(scoped[2]["task_id"]): "evidence_imported",
        str(scoped[3]["task_id"]): "failed",
    }

    assert batch._pending_tasks_for_stage(scoped, states, "generation") == [
        scoped[3],
        scoped[4],
        scoped[5],
    ]


def test_score_stage_rejects_tasks_without_generation() -> None:
    with pytest.raises(RuntimeError, match="requires ledgered generations"):
        batch._pending_tasks_for_stage(_clustered_ab_tasks(1, 1), {}, "score")


def test_development_timeout_defaults_to_protocol_deadline() -> None:
    assert batch._resolve_generation_timeout(
        None,
        kind="full",
        max_generation_attempts=3,
    ) == (
        batch.DEFAULT_GENERATION_TIMEOUT_SECONDS,
        batch.DEFAULT_GENERATION_TIMEOUT_POLICY,
    )


def test_development_timeout_requires_development_and_one_attempt() -> None:
    with pytest.raises(RuntimeError, match="requires a development manifest"):
        batch._resolve_generation_timeout(
            600,
            kind="full",
            max_generation_attempts=1,
        )
    with pytest.raises(RuntimeError, match="max-generation-attempts 1"):
        batch._resolve_generation_timeout(
            600,
            kind="development",
            max_generation_attempts=2,
        )
    assert batch._resolve_generation_timeout(
        600,
        kind="development",
        max_generation_attempts=1,
    ) == (600, batch.DEVELOPMENT_GENERATION_TIMEOUT_POLICY)


def test_exposed_treatment_canary_timeout_is_narrow_and_terminal() -> None:
    assert batch._resolve_generation_timeout(
        None,
        canary_requested=600,
        single_arm_canary=True,
        kind="full",
        max_generation_attempts=1,
    ) == (600, batch.CANARY_GENERATION_TIMEOUT_POLICY)
    with pytest.raises(RuntimeError, match="reviewed single-arm canary"):
        batch._resolve_generation_timeout(
            None,
            canary_requested=600,
            single_arm_canary=False,
            kind="full",
            max_generation_attempts=1,
        )
    with pytest.raises(RuntimeError, match="max-generation-attempts 1"):
        batch._resolve_generation_timeout(
            None,
            canary_requested=600,
            single_arm_canary=True,
            kind="full",
            max_generation_attempts=2,
        )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        batch._resolve_generation_timeout(
            600,
            canary_requested=600,
            single_arm_canary=True,
            kind="development",
            max_generation_attempts=1,
        )


@pytest.mark.parametrize("timeout", [119, 901])
def test_exposed_treatment_canary_timeout_rejects_out_of_range(timeout: int) -> None:
    with pytest.raises(RuntimeError, match="between 120 and 900"):
        batch._resolve_generation_timeout(
            None,
            canary_requested=timeout,
            single_arm_canary=True,
            kind="full",
            max_generation_attempts=1,
        )


@pytest.mark.parametrize("timeout", [59, 1800])
def test_development_timeout_rejects_out_of_range(timeout: int) -> None:
    with pytest.raises(RuntimeError, match="between 60 and 1799"):
        batch._resolve_generation_timeout(
            timeout,
            kind="development",
            max_generation_attempts=1,
        )


def test_generation_uses_runtime_wall_clock_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(batch.demo, "prepare_generation", lambda **_kwargs: planned)

    def execute_generation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(batch.demo, "execute_generation", execute_generation)
    runtime = batch.Runtime(
        experiment_root=tmp_path,
        switchyard=tmp_path / "switchyard",
        codex=tmp_path / "codex",
        tooluniverse=tmp_path / "tooluniverse",
        profile=tmp_path / "profile.yaml",
        doctor=tmp_path / "doctor.json",
        candidate=tmp_path / "candidate",
        generation_timeout_seconds=600,
        generation_timeout_policy=batch.DEVELOPMENT_GENERATION_TIMEOUT_POLICY,
    )

    result = batch._generation({}, "task", object(), {}, runtime, tmp_path)

    assert result is not None
    executor = captured["executor"]
    assert executor.func is batch.demo.run_streaming_subprocess
    assert executor.keywords == {"timeout_seconds": 600}


def test_generation_timeout_history_is_immutable(tmp_path: Path) -> None:
    manifest = {
        "manifest_id": "trialqa-development-timeout-test",
        "tasks": [{"task_id": "task-treatment"}],
    }
    ledger = demo.ResumableLedger(tmp_path / "ledger.jsonl", manifest)
    ledger.append(
        "task-treatment",
        "generation_started",
        {
            "generation_attempt": 1,
            "wall_clock_timeout_seconds": 600,
            "timeout_policy": batch.DEVELOPMENT_GENERATION_TIMEOUT_POLICY,
        },
    )

    batch._validate_generation_timeout_history(
        ledger,
        timeout_seconds=600,
        timeout_policy=batch.DEVELOPMENT_GENERATION_TIMEOUT_POLICY,
    )
    with pytest.raises(RuntimeError, match="differs from prior attempts"):
        batch._validate_generation_timeout_history(
            ledger,
            timeout_seconds=batch.DEFAULT_GENERATION_TIMEOUT_SECONDS,
            timeout_policy=batch.DEFAULT_GENERATION_TIMEOUT_POLICY,
        )


def test_legacy_generation_timeout_history_uses_protocol_default(tmp_path: Path) -> None:
    manifest = {
        "manifest_id": "trialqa-full-legacy-timeout-test",
        "tasks": [{"task_id": "task-baseline"}],
    }
    ledger = demo.ResumableLedger(tmp_path / "ledger.jsonl", manifest)
    ledger.append("task-baseline", "generation_started", {"generation_attempt": 1})

    batch._validate_generation_timeout_history(
        ledger,
        timeout_seconds=batch.DEFAULT_GENERATION_TIMEOUT_SECONDS,
        timeout_policy=batch.DEFAULT_GENERATION_TIMEOUT_POLICY,
    )


def _capture(tmp_path: Path) -> tuple[Path, dict[str, object], demo.ResumableLedger]:
    capture = tmp_path / "capture"
    task = _task()
    arm = capture / "trialqa-local" / "pair" / "arms" / "baseline"
    (arm / "outputs").mkdir(parents=True)
    metadata = capture / "trialqa-local" / "pair" / "runtime" / "baseline" / "launch-metadata"
    metadata.mkdir(parents=True)
    (metadata / "run-context.json").write_text("{}\n")
    (metadata / "active-evidence.json").write_text("{}\n")
    (arm / "outputs" / "switchyard-codex.stdout.log").write_text("events\n")
    (arm / "outputs" / "switchyard.stderr.log").write_text("stderr\n")
    manifest = {"manifest_id": "trialqa-full-batch-test", "tasks": [task]}
    ledger = demo.ResumableLedger(capture / "ledger.jsonl", manifest)
    ledger.append("pair-baseline", "generation_started")
    return capture, task, ledger


def _scored_generation(tmp_path: Path, ledger: demo.ResumableLedger) -> demo.ScoredGeneration:
    task = ledger.manifest["tasks"][0]
    assert isinstance(task, dict)
    generation = demo.GenerationResult(
        manifest_id=ledger.manifest_id,
        task_id=str(task["task_id"]),
        pair_id=str(task["pair_id"]),
        row_id=str(task["row_id"]),
        dataset_row_index=int(task["dataset_row_index"]),
        partition=str(task["partition"]),
        condition=str(task["condition"]),
        repeat_index=int(task["repeat_index"]),
        n_repeats=int(task["n_repeats"]),
        answer="six months",
        answer_source=demo.FINAL_ANSWER_TEXT_SOURCE,
        session_dir=tmp_path / "session",
        stats_path=tmp_path / "stats.json",
        trajectory_path=tmp_path / "turns.jsonl",
        codex_events_path=tmp_path / "codex.jsonl",
        final_output_path=tmp_path / "final.txt",
        generation_path=tmp_path / "generation.json",
        stats={
            "total_requests": 2,
            "total_tokens": {"prompt": 100, "completion": 20, "total": 120},
        },
        usage={"input_tokens": 100, "output_tokens": 20},
        artifact_sha256={},
    )
    return demo.ScoredGeneration(
        generation=generation,
        outcome=demo.JudgeOutcome(
            judge_result="correct",
            score=1.0,
            rationale="Equivalent.",
            judge_available=True,
            judge_model=demo.JUDGE_MODEL,
        ),
        reward={"judge_result": "correct"},
        evidence=demo.NativeTrialQAEvidenceImportResult(
            evidence_id="native-" + "1" * 32,
            evidence_path=tmp_path / "evidence.json",
            imported=True,
        ),
    )


def test_scored_result_is_written_then_hash_bound_to_terminal_ledger(
    tmp_path: Path,
) -> None:
    capture, task, ledger = _capture(tmp_path)
    ledger.append("pair-baseline", "generation_completed")
    scored = _scored_generation(tmp_path, ledger)

    result_path = batch._commit_scored_result(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        scored=scored,
    )

    assert result_path.is_file()
    assert ledger.states()["pair-baseline"] == "completed"
    records = demo.collect_protocol_results(capture / "results", ledger.manifest)
    demo.validate_protocol_result_ledger(capture / "results", records, ledger)
    changed = json.loads(result_path.read_text())
    changed["score"] = 0.0
    result_path.write_text(json.dumps(changed) + "\n")
    tampered = demo.collect_protocol_results(capture / "results", ledger.manifest)
    with pytest.raises(demo.TrialQADemoError, match="completion binding"):
        demo.validate_protocol_result_ledger(capture / "results", tampered, ledger)


def test_partial_scored_result_recovers_without_rescoring(tmp_path: Path) -> None:
    capture, task, ledger = _capture(tmp_path)
    ledger.append("pair-baseline", "generation_completed")
    scored = _scored_generation(tmp_path, ledger)
    result_path = batch._scored_result_path(capture, "pair-baseline")
    demo.write_trial_result(result_path, demo.scored_result_record(scored))

    record = batch._finish_partial_scored_result(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        generation=scored.generation,
    )

    assert record.score == 1.0
    assert ledger.states()["pair-baseline"] == "completed"
    records = demo.collect_protocol_results(capture / "results", ledger.manifest)
    demo.validate_protocol_result_ledger(capture / "results", records, ledger)


def _proof(requests: int) -> batch.SessionProof:
    usage = {"input_tokens": 100, "output_tokens": 20} if requests else None
    return batch.SessionProof(
        ledger_payload={
            "session_id": "session-1",
            "total_requests": requests,
            "artifact_sha256": {},
        },
        total_requests=requests,
        usage=usage,
    )


def _real_paid_proof(capture: Path) -> batch.SessionProof:
    session = (
        capture / ".switchyard" / "skill-distillation" / demo.NAMESPACE / "sessions" / "session-1"
    )
    session.mkdir(parents=True)
    for name in ("session.json", "turns.jsonl"):
        (session / name).write_text(name + "\n")
    transport = _openai_transport(1)
    (session / "stats.json").write_text(
        json.dumps(
            {
                "total_requests": 1,
                "openai_transport": transport,
            }
        )
        + "\n"
    )
    metadata = capture / "trialqa-local" / "pair" / "runtime" / "baseline" / "launch-metadata"
    hashes = {
        name: demo._sha256_file(session / name)
        for name in ("session.json", "stats.json", "turns.jsonl")
    }
    hashes.update(
        {
            name: demo._sha256_file(metadata / name)
            for name in ("run-context.json", "active-evidence.json")
        }
    )
    return batch.SessionProof(
        ledger_payload={
            "session_id": "session-1",
            "session_path": str(session),
            "total_requests": 1,
            "openai_transport": transport,
            "served_models": [demo.EXECUTOR_MODEL],
            "turns_present": True,
            "artifact_sha256": hashes,
        },
        total_requests=1,
        usage={"input_tokens": 100, "output_tokens": 20},
    )


def test_completed_executor_draw_terminalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    monkeypatch.setattr(
        batch, "_unbound_session_proof", lambda **_kwargs: _real_paid_proof(capture)
    )
    monkeypatch.setattr(
        batch,
        "_completed_draw_usage",
        lambda *_args: {"input_tokens": 100, "output_tokens": 20},
    )

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=demo.TrialQADemoError("no evidence call"),
        retry_exhausted=False,
    )

    assert recorded.completed_model_draw is True
    assert recorded.retry_permitted is False
    assert recorded.manual_review is False
    assert recorded.terminal_result_path is not None
    result = demo.load_trial_result(recorded.terminal_result_path)
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (
        100,
        20,
        120,
    )
    batch._finish_terminal_generation_failure(
        capture=capture,
        task=task,
        ledger=ledger,
        failure=recorded.ledger_record,
    )
    assert ledger.states()["pair-baseline"] == "completed"
    records = demo.collect_protocol_results(capture / "results", ledger.manifest)
    demo.validate_protocol_result_ledger(capture / "results", records, ledger)


def test_retry_requires_affirmative_zero_request_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    monkeypatch.setattr(batch, "_unbound_session_proof", lambda **_kwargs: _proof(0))
    monkeypatch.setattr(batch, "_completed_draw_usage", lambda *_args: None)

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=RuntimeError("pre-model launch failure"),
        retry_exhausted=False,
    )

    assert recorded.completed_model_draw is False
    assert recorded.retry_permitted is True
    assert recorded.manual_review is False
    assert recorded.terminal_result_path is None
    assert ledger.states()["pair-baseline"] == "failed"


def test_generation_timeout_never_retries_zero_request_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    ledger.manifest["kind"] = "development"
    monkeypatch.setattr(batch, "_unbound_session_proof", lambda **_kwargs: _proof(0))
    monkeypatch.setattr(batch, "_completed_draw_usage", lambda *_args: None)

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=demo.GenerationTimeoutError(600),
        retry_exhausted=True,
    )

    payload = recorded.ledger_record["payload"]
    assert recorded.completed_model_draw is False
    assert recorded.retry_permitted is False
    assert recorded.manual_review is False
    assert recorded.terminal_result_path is not None
    assert payload["timed_out"] is True
    assert payload["wall_clock_timeout_seconds"] == 600
    assert payload["process_group_terminated"] is True
    batch._finish_terminal_generation_failure(
        capture=capture,
        task=task,
        ledger=ledger,
        failure=recorded.ledger_record,
    )
    assert ledger.states()["pair-baseline"] == "completed"
    assert batch._generation_attempt_count(ledger, "pair-baseline") == 1


def test_development_default_timeout_preserves_zero_request_retry_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    ledger.manifest["kind"] = "development"
    monkeypatch.setattr(batch, "_unbound_session_proof", lambda **_kwargs: _proof(0))
    monkeypatch.setattr(batch, "_completed_draw_usage", lambda *_args: None)

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=demo.GenerationTimeoutError(batch.DEFAULT_GENERATION_TIMEOUT_SECONDS),
        retry_exhausted=False,
    )

    assert recorded.retry_permitted is True
    assert recorded.terminal_result_path is None
    assert ledger.states()["pair-baseline"] == "failed"


def test_generation_timeout_cleanup_failure_requires_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    ledger.manifest["kind"] = "development"
    monkeypatch.setattr(batch, "_unbound_session_proof", lambda **_kwargs: _proof(0))
    monkeypatch.setattr(batch, "_completed_draw_usage", lambda *_args: None)

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=demo.GenerationTimeoutError(
            600,
            process_group_terminated=False,
        ),
        retry_exhausted=True,
    )

    assert recorded.retry_permitted is False
    assert recorded.manual_review is True
    assert recorded.terminal_result_path is None
    assert recorded.ledger_record["payload"]["process_group_terminated"] is False
    assert ledger.states()["pair-baseline"] == "failed"


def test_missing_session_proof_requires_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, task, ledger = _capture(tmp_path)
    monkeypatch.setattr(
        batch,
        "_unbound_session_proof",
        lambda **_kwargs: (_ for _ in ()).throw(batch.SessionProofError("missing")),
    )
    monkeypatch.setattr(batch, "_completed_draw_usage", lambda *_args: None)

    recorded = batch._record_generation_failure(
        manifest=ledger.manifest,
        task=task,
        capture=capture,
        ledger=ledger,
        error=RuntimeError("ambiguous failure"),
        retry_exhausted=False,
    )

    assert recorded.retry_permitted is False
    assert recorded.manual_review is True
    assert recorded.terminal_result_path is None


def test_paid_session_proof_requires_contiguous_ultra_trajectory(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-1"
    session_dir.mkdir()
    context: dict[str, object] = {"task_id": "pair-baseline"}
    active: dict[str, object] = {"loaded": False}
    turn = {
        "schema_version": 1,
        "session_id": "session-1",
        "turn_index": 0,
        "served_model": demo.EXECUTOR_MODEL,
        "request": {"model": demo.EXECUTOR_ROUTE},
        "active_skill_version": None,
        "active_skill_candidate_id": None,
        "active_skill_manifest_sha256": None,
    }
    turns_path = session_dir / "turns.jsonl"
    turns_path.write_text(json.dumps(turn) + "\n")
    stats = {
        "total_requests": 1,
        "total_errors": 0,
        "models": {demo.EXECUTOR_MODEL: {"calls": 1, "errors": 0}},
        "total_tokens": {"prompt": 100, "completion": 20, "total": 120},
        "classifier": {"total_requests": 0, "total_errors": 0},
        "planner": {"total_requests": 0, "total_errors": 0},
        "openai_transport": _openai_transport(
            1,
            retries=1,
            charges=1,
            prompt=100,
            completion=20,
        ),
    }
    (session_dir / "stats.json").write_text(json.dumps(stats))
    session = {
        "schema_version": 1,
        "session_id": "session-1",
        "namespace": demo.NAMESPACE,
        "launch_target": "codex",
        "display_model": demo.EXECUTOR_ROUTE,
        "status": "completed",
        "ended_at": "2026-07-07T00:00:00Z",
        "exit_code": 0,
        "turn_count": 1,
        "turns_path": "turns.jsonl",
        "stats_path": "stats.json",
        "run_context": context,
        "active_skill": active,
        "trajectory_sha256": demo._sha256_file(turns_path),
    }
    (session_dir / "session.json").write_text(json.dumps(session))

    proof = batch._validate_session_proof(
        session_dir=session_dir,
        expected_context=context,
        expected_active=active,
        launch_sha256={
            "run-context.json": "sha256:" + "1" * 64,
            "active-evidence.json": "sha256:" + "2" * 64,
        },
    )

    assert proof.total_requests == 1
    assert proof.usage == {"input_tokens": 100, "output_tokens": 20}
    assert proof.ledger_payload["served_models"] == [demo.EXECUTOR_MODEL]
    assert proof.ledger_payload["openai_transport"] == _openai_transport(
        1,
        retries=1,
        charges=1,
        prompt=100,
        completion=20,
    )


def test_unpriced_lazy_retry_failure_is_bound_without_synthetic_turn(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-null-eof"
    session_dir.mkdir()
    context: dict[str, object] = {"task_id": "pair-treatment"}
    active: dict[str, object] = {"loaded": True}
    transport = _openai_transport(1, retries=1, unpriced=1)
    (session_dir / "stats.json").write_text(
        json.dumps(
            {
                "total_requests": 1,
                "total_errors": 0,
                "models": {demo.EXECUTOR_MODEL: {"calls": 1, "errors": 0}},
                "total_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "classifier": {"total_requests": 0, "total_errors": 0},
                "planner": {"total_requests": 0, "total_errors": 0},
                "openai_transport": transport,
            }
        )
    )
    session = {
        "schema_version": 1,
        "session_id": "session-null-eof",
        "namespace": demo.NAMESPACE,
        "launch_target": "codex",
        "display_model": demo.EXECUTOR_ROUTE,
        "status": "failed",
        "ended_at": "2026-07-07T00:00:00Z",
        "exit_code": 1,
        "turn_count": 0,
        "turns_path": "turns.jsonl",
        "stats_path": "stats.json",
        "run_context": context,
        "active_skill": active,
        "trajectory_sha256": None,
    }
    (session_dir / "session.json").write_text(json.dumps(session))

    proof = batch._validate_session_proof(
        session_dir=session_dir,
        expected_context=context,
        expected_active=active,
        launch_sha256={
            "run-context.json": "sha256:" + "1" * 64,
            "active-evidence.json": "sha256:" + "2" * 64,
        },
    )

    assert proof.total_requests == 1
    assert proof.usage == {"input_tokens": 0, "output_tokens": 0}
    assert proof.ledger_payload["turns_present"] is False
    assert proof.ledger_payload["served_models"] == [demo.EXECUTOR_MODEL]
    assert proof.ledger_payload["openai_transport"] == transport

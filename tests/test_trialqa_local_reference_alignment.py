# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_reference_alignment as alignment


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _tasks(*, questions: int, repeats: int) -> list[dict[str, object]]:
    tasks = []
    for index in range(questions):
        row_id = f"row-{index}"
        group = f"trialqa-{index:04d}-{hashlib.sha256(row_id.encode()).hexdigest()[:12]}"
        for repeat in range(1, repeats + 1):
            for arm in ("baseline", "treatment"):
                tasks.append(
                    {
                        "task_id": f"{group}-r{repeat:03d}-{arm}",
                        "pair_id": f"{group}-r{repeat:03d}",
                        "row_id": row_id,
                        "dataset_row_index": index,
                        "question_group_key": group,
                        "partition": "test",
                        "phase": "evaluation",
                        "condition": arm,
                        "arm": arm,
                        "repeat_index": repeat,
                        "n_repeats": repeats,
                    }
                )
    return tasks


def _manifest(
    path: Path,
    *,
    official_labbench2: bool = False,
    questions: int = 8,
    repeats: int = 5,
) -> Path:
    tasks = _tasks(questions=questions, repeats=repeats)
    groups = [
        f"trialqa-{index:04d}-{hashlib.sha256(f'row-{index}'.encode()).hexdigest()[:12]}"
        for index in range(questions)
    ]
    group_digest = demo._sha256_bytes(demo._canonical_json(groups))
    empty_digest = demo._sha256_bytes(demo._canonical_json([]))
    manifest = {
        "schema_version": "switchyard.trialqa_experiment_manifest.v1",
        "kind": "full",
        "dataset": {
            "official_labbench2": official_labbench2,
            "test_count": questions,
            "heldout_ordering": {
                "question_count": questions,
                "question_group_keys": groups,
                "question_group_keys_sha256": group_digest,
            },
        },
        "protocol": {
            "batch_driver": "benchmark/trialqa_local_batch.py",
            "performance_eligible": True,
            "primary_evaluation_scope": {
                "question_start": 0,
                "question_count": questions,
                "repeat_count": repeats,
                "task_count": len(tasks),
                "question_group_keys_sha256": group_digest,
            },
            "heldout_quarantine": {
                "question_start": 0,
                "question_count": 0,
                "disposition": (
                    "excluded-exposed-heldout"
                    if official_labbench2
                    else "none-new-prospective-population"
                ),
                "question_group_keys_sha256": empty_digest,
            },
            "prospective_population_kind": "trialqa-compatible-clinicaltrials-gov",
            "max_generation_concurrency": 4,
        },
        "routing": {
            "executor_model": "nvidia/nvidia/nemotron-3-ultra",
        },
        "runtime": {
            "tooluniverse": {
                "version": "1.1.11",
            },
        },
        "tasks": tasks,
    }
    manifest = {
        "manifest_id": f"trialqa-full-{hashlib.sha256(demo._canonical_json(manifest)).hexdigest()[:20]}",
        **manifest,
    }
    return _write_json(path, manifest)


def _reference(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "switchyard.trialqa_reference_targets.v1",
            "population": {
                "dataset": "LABBench2 TrialQA",
                "heldout_questions": 96,
                "repeats_per_question": 5,
                "trials": 480,
                "tool_provider": "ToolUniverse MCP",
                "injected_context": False,
            },
            "super": {
                "r1": {
                    "accuracy": 0.738,
                    "token_reduction": 0.3,
                    "operational_call_reduction": 0.45,
                }
            },
        },
    )


def _skills_repo(path: Path, *, ultra_placeholder: bool = True) -> Path:
    docs = path / "docs"
    configs = path / "configs"
    scripts = path / "scripts"
    docs.mkdir(parents=True)
    configs.mkdir()
    scripts.mkdir()
    (docs / "exp1.md").write_text(
        "\n".join(
            [
                "Trials: 480",
                "Mean: 0.610",
                "Mean: 0.738",
                "Test (distilled):",
                "trial_mean 0.737500",
                "Mean tokens / trial",
                "549,406",
                "384,654",
                "Operational tool calls / trial",
                "15.5",
                "8.6",
            ]
        ),
        encoding="utf-8",
    )
    ultra_body = (
        "nvidia/nvidia/nvidia/nemotron-3-ultra\n"
        "<<PLACEHOLDER>>\n"
        "Success criterion: heldout accuracy improves or stays flat\n"
        if ultra_placeholder
        else "nvidia/nvidia/nvidia/nemotron-3-ultra\nTrials: 480\nMean: 0.800\n"
    )
    (docs / "exp1_ultra.md").write_text(ultra_body, encoding="utf-8")
    (configs / "trialqa-opencode.harbor.yaml").write_text(
        "\n".join(
            [
                "dataset_config: trialqa",
                "train_fraction: 0.2",
                "split_seed: trace2skill-trialqa",
                "n_repeats: 5",
                "tooluniverse_mcp: true",
            ]
        ),
        encoding="utf-8",
    )
    (scripts / "aggregate_trialqa_replicate_metrics.py").write_text(
        "\n".join(
            [
                '"trial_mean"',
                '"question_macro_mean"',
                '"worst_case"',
                '"oracle"',
                '"token_metrics"',
                "def aggregate_token_metrics(): pass",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _requirements(report: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = report["requirements"]
    assert isinstance(raw, list)
    return {str(item["id"]): item for item in raw if isinstance(item, dict)}


def _refresh_manifest_id(payload: dict[str, object]) -> None:
    seed = {key: value for key, value in payload.items() if key != "manifest_id"}
    payload["manifest_id"] = f"trialqa-full-{hashlib.sha256(demo._canonical_json(seed)).hexdigest()[:20]}"


def test_reference_alignment_marks_proxy_canary_but_not_official_reproduction(
    tmp_path: Path,
) -> None:
    report = alignment.build_reference_alignment(
        alignment.ReferenceAlignmentConfig(
            manifest=_manifest(tmp_path / "manifest.json"),
            reference_targets=_reference(tmp_path / "reference.json"),
        )
    )
    requirements = _requirements(report)

    assert report["schema_version"] == alignment.SCHEMA_VERSION
    assert report["canary_alignment_status"] == "proved"
    assert report["official_reproduction_status"] == "missing"
    assert report["claim_scope"] == "prospective_transfer_canary"
    assert report["current_scope"] == {
        "questions": 8,
        "repeats_per_question": 5,
        "paired_tasks": 80,
    }
    assert report["reference_scope"] == {
        "questions": 96,
        "repeats_per_question": 5,
        "unpaired_trials": 480,
        "paired_tasks": 960,
    }
    assert requirements["paired_off_on_shape_matches_reference"]["status"] == "proved"
    assert requirements["nemotron_ultra_switchyard_runtime_bound"]["status"] == "proved"
    assert requirements["tooluniverse_trialqa_interface_bound"]["status"] == "proved"
    assert requirements["prospective_proxy_scope_explicit"]["status"] == "proved"
    assert requirements["official_96_question_reproduction_bound"]["status"] == "missing"
    assert (
        requirements["official_96_question_reproduction_bound"][
            "required_for_official_reproduction"
        ]
        is True
    )
    assert requirements["official_96_question_reproduction_bound"]["required_for_canary"] is False


def test_reference_alignment_fails_wrong_sized_proxy_canary(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path / "wrong-sized-proxy.json",
        questions=6,
        repeats=5,
    )

    report = alignment.build_reference_alignment(
        alignment.ReferenceAlignmentConfig(
            manifest=manifest,
            reference_targets=_reference(tmp_path / "reference.json"),
        )
    )
    requirements = _requirements(report)

    assert report["canary_alignment_status"] == "failed"
    assert report["official_reproduction_status"] == "missing"
    assert report["claim_scope"] == "not_aligned"
    assert requirements["prospective_proxy_scope_explicit"]["status"] == "failed"


def test_reference_alignment_fails_when_runtime_is_not_ultra(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["routing"]["executor_model"] = "nvidia/nvidia/nemotron-3-super"
    _refresh_manifest_id(payload)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    report = alignment.build_reference_alignment(
        alignment.ReferenceAlignmentConfig(
            manifest=manifest,
            reference_targets=_reference(tmp_path / "reference.json"),
        )
    )
    requirements = _requirements(report)

    assert report["canary_alignment_status"] == "failed"
    assert requirements["nemotron_ultra_switchyard_runtime_bound"]["status"] == "failed"


def test_reference_alignment_binds_cloned_workflow_evidence(tmp_path: Path) -> None:
    report = alignment.build_reference_alignment(
        alignment.ReferenceAlignmentConfig(
            manifest=_manifest(tmp_path / "manifest.json"),
            reference_targets=_reference(tmp_path / "reference.json"),
            skills_distillation_repo=_skills_repo(tmp_path / "skills-distillation"),
        )
    )
    requirements = _requirements(report)
    workflow = report["reference_workflow_evidence"]
    assert isinstance(workflow, dict)

    assert report["canary_alignment_status"] == "proved"
    assert requirements["reference_workflow_source_evidence_bound"]["status"] == "proved"
    assert workflow["super_reference_status"] == "complete"
    assert workflow["ultra_trialqa_reference_status"] == "placeholder_only"


def test_reference_alignment_fails_when_ultra_trialqa_reference_is_overclaimed(
    tmp_path: Path,
) -> None:
    report = alignment.build_reference_alignment(
        alignment.ReferenceAlignmentConfig(
            manifest=_manifest(tmp_path / "manifest.json"),
            reference_targets=_reference(tmp_path / "reference.json"),
            skills_distillation_repo=_skills_repo(
                tmp_path / "skills-distillation",
                ultra_placeholder=False,
            ),
        )
    )
    requirements = _requirements(report)

    assert report["canary_alignment_status"] == "failed"
    assert requirements["reference_workflow_source_evidence_bound"]["status"] == "failed"

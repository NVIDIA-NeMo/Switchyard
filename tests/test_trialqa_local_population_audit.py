# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import benchmark.trialqa_local_dataset as trialqa
import benchmark.trialqa_local_population_audit as audit


def _dataset(tmp_path: Path) -> trialqa.TrialQADataset:
    rows = tuple(
        trialqa.TrialQARow(
            dataset_row_index=index,
            id=f"trialqa-{index:03d}",
            tag="trialqa",
            version="1.0",
            question=f"Question {index}?",
            ideal=f"Ideal {index}",
            files="",
            sources=(f"NCT{index:08d}",),
            key_passage=f"Passage {index}",
            canary="",
            is_opensource=True,
            ground_truth=True,
            prompt_suffix="",
            trialqa_type="text",
            mode=trialqa.TrialQAMode(file=False, retrieve=True, inject=False),
            validator_params="{}",
            answer_regex="",
        )
        for index in range(trialqa.TRIALQA_ROW_COUNT)
    )
    return trialqa.TrialQADataset(
        path=tmp_path / "trialqa.parquet",
        revision=trialqa.TRIALQA_DATASET_REVISION,
        parquet_sha256=trialqa.TRIALQA_PARQUET_SHA256,
        rows=rows,
    )


def _hf_metadata(*, extra_trialqa_file: bool = False) -> dict[str, object]:
    siblings = [{"rfilename": trialqa.TRIALQA_PARQUET_NAME}]
    if extra_trialqa_file:
        siblings.append({"rfilename": "trialqa/validation-00000-of-00001.parquet"})
    return {
        "id": trialqa.TRIALQA_DATASET_ID,
        "sha": trialqa.TRIALQA_DATASET_REVISION,
        "lastModified": "2026-03-13T23:17:33.000Z",
        "cardData": {
            "dataset_info": [
                {
                    "config_name": trialqa.TRIALQA_DATASET_CONFIG,
                    "splits": [
                        {
                            "name": trialqa.TRIALQA_DATASET_SPLIT,
                            "num_examples": trialqa.TRIALQA_ROW_COUNT,
                        }
                    ],
                }
            ],
            "configs": [
                {
                    "config_name": trialqa.TRIALQA_DATASET_CONFIG,
                    "data_files": [
                        {
                            "split": trialqa.TRIALQA_DATASET_SPLIT,
                            "path": "trialqa/train-*",
                        }
                    ],
                }
            ],
        },
        "siblings": siblings,
    }


def test_scan_experiment_exposure_is_partition_aware(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    split = trialqa.create_split_manifest(dataset)
    split_rows = split["rows"]
    assert isinstance(split_rows, list)
    train_row = next(row for row in split_rows if row["partition"] == "train")
    test_row = next(row for row in split_rows if row["partition"] == "test")

    root = tmp_path / "experiments"
    root.mkdir()
    manifest = {
        "tasks": [
            {
                "task_id": "train-task",
                "row_id": train_row["row_id"],
                "dataset_row_index": train_row["dataset_row_index"],
            },
            {
                "task_id": "test-task",
                "row_id": test_row["row_id"],
                "dataset_row_index": test_row["dataset_row_index"],
            },
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    train_output = root / "run" / "train" / "outputs"
    test_output = root / "run" / "test" / "outputs"
    train_output.mkdir(parents=True)
    test_output.mkdir(parents=True)
    (train_output / "generation.json").write_text(
        json.dumps({"task_id": "train-task", "row_id": train_row["row_id"]})
    )
    (test_output / "generation.json").write_text(
        json.dumps({"task_id": "test-task", "row_id": test_row["row_id"]})
    )

    report = audit.scan_experiment_exposure(dataset, root)

    assert report["planned_rows"] == {
        "max_dataset_row_index": max(
            train_row["dataset_row_index"], test_row["dataset_row_index"]
        ),
        "min_dataset_row_index": min(
            train_row["dataset_row_index"], test_row["dataset_row_index"]
        ),
        "test": 1,
        "total": 2,
        "train": 1,
    }
    assert report["generation_exposed_rows"]["train"] == 1
    assert report["generation_exposed_rows"]["test"] == 1
    assert report["unexposed_rows"]["total"] == 118
    assert len(report["unexposed_train_row_ids"]) == 23
    assert len(report["unexposed_test_row_ids"]) == 95
    assert report["local_trialqa_performance_population_available"] is True


def test_build_population_audit_marks_exhausted_local_population(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    root = tmp_path / "experiments"
    root.mkdir()
    tasks = []
    for row in dataset.rows:
        tasks.append(
            {
                "task_id": f"trialqa-{row.dataset_row_index:04d}-abcdefabcdef-r001-baseline",
                "row_id": row.id,
                "dataset_row_index": row.dataset_row_index,
            }
        )
        output_dir = root / "run" / row.id / "outputs"
        output_dir.mkdir(parents=True)
        (output_dir / "generation.json").write_text(
            json.dumps({"task_id": tasks[-1]["task_id"], "row_id": row.id})
        )
    (root / "full-manifest.json").write_text(json.dumps({"tasks": tasks}))

    report = audit.build_population_audit_report(
        dataset,
        root,
        huggingface_metadata=_hf_metadata(),
    )

    assert report["local_exposure"]["generation_exposed_rows"] == {
        "max_dataset_row_index": 119,
        "min_dataset_row_index": 0,
        "test": 96,
        "total": 120,
        "train": 24,
    }
    assert report["huggingface_metadata"]["trialqa_only_one_official_split_file"] is True
    assert report["decision"] == {
        "local_existing_parquet_supports_new_performance_claim": False,
        "official_unseen_trialqa_population_available": False,
        "next_step": "create_or_fetch_new_trialqa_compatible_population",
    }


def test_huggingface_metadata_detects_extra_trialqa_population() -> None:
    summary = audit.summarize_huggingface_metadata(_hf_metadata(extra_trialqa_file=True))

    assert summary["trialqa_only_one_official_split_file"] is False
    assert summary["official_trialqa_unseen_split_available"] is True

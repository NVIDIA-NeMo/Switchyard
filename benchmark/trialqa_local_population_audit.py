# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-spend population and exposure audit for the local TrialQA workflow.

The audit answers one pre-benchmark question: can a local TrialQA row still be
used for a prospective performance claim?  It deliberately makes no model
calls and performs no network access.  Optional Hugging Face metadata can be
passed as a saved JSON file when the caller wants the report to include the
official upstream dataset shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import benchmark.trialqa_local_dataset as trialqa

SCHEMA_VERSION = "switchyard.trialqa_population_audit.v1"
SKIPPED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "hf-cache",
    "node_modules",
    "sessions",
    "tooluniverse-venv",
}
TASK_ID_DATASET_INDEX = re.compile(r"\btrialqa-(\d{4})-[0-9a-f]+-r\d{3}-")
JsonObject = dict[str, Any]


class TrialQAPopulationAuditError(RuntimeError):
    """Population audit inputs are malformed or inconsistent."""


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrialQAPopulationAuditError(f"{field} must be an object")
    return value


def _as_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TrialQAPopulationAuditError(f"{field} must be an array")
    return value


def _sorted_row_ids(dataset: trialqa.TrialQADataset, row_indices: set[int]) -> list[str]:
    by_index = {row.dataset_row_index: row.id for row in dataset.rows}
    return [by_index[index] for index in sorted(row_indices)]


def _partition_index_sets(dataset: trialqa.TrialQADataset) -> dict[str, set[int]]:
    split = trialqa.create_split_manifest(dataset)
    rows = cast(list[dict[str, object]], split["rows"])
    by_row_id = {row.id: row.dataset_row_index for row in dataset.rows}
    partitions = {"train": set[int](), "test": set[int]()}
    for row in rows:
        row_id = cast(str, row["row_id"])
        partition = cast(str, row["partition"])
        partitions[partition].add(by_row_id[row_id])
    return partitions


def _summarize_indices(row_indices: set[int], partitions: Mapping[str, set[int]]) -> JsonObject:
    return {
        "total": len(row_indices),
        "train": len(row_indices & partitions["train"]),
        "test": len(row_indices & partitions["test"]),
        "min_dataset_row_index": min(row_indices) if row_indices else None,
        "max_dataset_row_index": max(row_indices) if row_indices else None,
    }


def _index_from_record(
    record: Mapping[str, object],
    *,
    row_id_to_index: Mapping[str, int],
    task_id_to_index: Mapping[str, int],
    row_count: int,
) -> int | None:
    raw_index = record.get("dataset_row_index")
    if isinstance(raw_index, int) and 0 <= raw_index < row_count:
        return raw_index

    raw_row_id = record.get("row_id")
    if isinstance(raw_row_id, str) and raw_row_id in row_id_to_index:
        return row_id_to_index[raw_row_id]

    raw_task_id = record.get("task_id")
    if isinstance(raw_task_id, str):
        if raw_task_id in task_id_to_index:
            return task_id_to_index[raw_task_id]
        match = TASK_ID_DATASET_INDEX.search(raw_task_id)
        if match is not None:
            parsed = int(match.group(1))
            if 0 <= parsed < row_count:
                return parsed
    return None


def _scan_manifest(
    path: Path,
    *,
    row_id_to_index: Mapping[str, int],
    task_id_to_index: dict[str, int],
    planned: set[int],
) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, Mapping):
        return False
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return False
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        row_index = _index_from_record(
            task,
            row_id_to_index=row_id_to_index,
            task_id_to_index=task_id_to_index,
            row_count=len(row_id_to_index),
        )
        if row_index is None:
            continue
        planned.add(row_index)
        task_id = task.get("task_id")
        if isinstance(task_id, str):
            task_id_to_index[task_id] = row_index
    return True


def _scan_generation(
    path: Path,
    *,
    row_id_to_index: Mapping[str, int],
    task_id_to_index: Mapping[str, int],
    exposed: set[int],
) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, Mapping):
        return False
    row_index = _index_from_record(
        data,
        row_id_to_index=row_id_to_index,
        task_id_to_index=task_id_to_index,
        row_count=len(row_id_to_index),
    )
    if row_index is None:
        return False
    exposed.add(row_index)
    return True


def scan_experiment_exposure(
    dataset: trialqa.TrialQADataset,
    experiment_root: Path,
) -> JsonObject:
    """Scan manifests and generation summaries for local row exposure."""

    if not experiment_root.exists():
        raise TrialQAPopulationAuditError(f"experiment root does not exist: {experiment_root}")

    row_id_to_index = {row.id: row.dataset_row_index for row in dataset.rows}
    task_id_to_index: dict[str, int] = {}
    planned: set[int] = set()
    exposed: set[int] = set()
    files_scanned = {
        "generation_json": 0,
        "task_manifest_json": 0,
    }
    files_used = {
        "generation_json": 0,
        "task_manifest_json": 0,
    }
    skipped_directories = 0

    for dirpath, dirnames, filenames in os.walk(experiment_root):
        retained = []
        for dirname in dirnames:
            if dirname in SKIPPED_DIR_NAMES:
                skipped_directories += 1
            else:
                retained.append(dirname)
        dirnames[:] = retained

        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            if filename == "generation.json":
                files_scanned["generation_json"] += 1
                if _scan_generation(
                    path,
                    row_id_to_index=row_id_to_index,
                    task_id_to_index=task_id_to_index,
                    exposed=exposed,
                ):
                    files_used["generation_json"] += 1
                continue
            if "manifest" in filename and filename.endswith(".json"):
                files_scanned["task_manifest_json"] += 1
                if _scan_manifest(
                    path,
                    row_id_to_index=row_id_to_index,
                    task_id_to_index=task_id_to_index,
                    planned=planned,
                ):
                    files_used["task_manifest_json"] += 1

    partitions = _partition_index_sets(dataset)
    all_indices = {row.dataset_row_index for row in dataset.rows}
    unplanned = all_indices - planned
    unexposed = all_indices - exposed

    return {
        "experiment_root": str(experiment_root),
        "files_scanned": files_scanned,
        "files_used": files_used,
        "skipped_directories": skipped_directories,
        "planned_rows": _summarize_indices(planned, partitions),
        "generation_exposed_rows": _summarize_indices(exposed, partitions),
        "unplanned_rows": _summarize_indices(unplanned, partitions),
        "unexposed_rows": _summarize_indices(unexposed, partitions),
        "unexposed_train_row_ids": _sorted_row_ids(dataset, unexposed & partitions["train"]),
        "unexposed_test_row_ids": _sorted_row_ids(dataset, unexposed & partitions["test"]),
        "local_trialqa_performance_population_available": bool(unexposed & partitions["test"]),
    }


def summarize_huggingface_metadata(metadata: Mapping[str, object]) -> JsonObject:
    """Summarize saved Hugging Face dataset metadata for the TrialQA config."""

    card_data = _as_mapping(metadata.get("cardData"), "cardData")
    dataset_info = _as_sequence(card_data.get("dataset_info"), "cardData.dataset_info")
    configs = _as_sequence(card_data.get("configs"), "cardData.configs")
    siblings = _as_sequence(metadata.get("siblings"), "siblings")

    trialqa_info: Mapping[str, object] | None = None
    for item in dataset_info:
        if isinstance(item, Mapping) and item.get("config_name") == trialqa.TRIALQA_DATASET_CONFIG:
            trialqa_info = item
            break
    if trialqa_info is None:
        raise TrialQAPopulationAuditError("HF metadata has no trialqa dataset_info entry")

    trialqa_config: Mapping[str, object] | None = None
    for item in configs:
        if isinstance(item, Mapping) and item.get("config_name") == trialqa.TRIALQA_DATASET_CONFIG:
            trialqa_config = item
            break
    if trialqa_config is None:
        raise TrialQAPopulationAuditError("HF metadata has no trialqa config entry")

    splits = [
        {
            "name": split.get("name"),
            "num_examples": split.get("num_examples"),
        }
        for split in _as_sequence(trialqa_info.get("splits"), "trialqa.splits")
        if isinstance(split, Mapping)
    ]
    data_files = [
        {
            "split": data_file.get("split"),
            "path": data_file.get("path"),
        }
        for data_file in _as_sequence(trialqa_config.get("data_files"), "trialqa.data_files")
        if isinstance(data_file, Mapping)
    ]
    trialqa_siblings = sorted(
        cast(str, sibling.get("rfilename"))
        for sibling in siblings
        if isinstance(sibling, Mapping)
        and isinstance(sibling.get("rfilename"), str)
        and cast(str, sibling["rfilename"]).startswith("trialqa/")
    )

    has_only_expected_split = splits == [
        {"name": trialqa.TRIALQA_DATASET_SPLIT, "num_examples": trialqa.TRIALQA_ROW_COUNT}
    ]
    has_only_expected_file = trialqa_siblings == [trialqa.TRIALQA_PARQUET_NAME]
    has_only_expected_data_pattern = data_files == [
        {"split": trialqa.TRIALQA_DATASET_SPLIT, "path": "trialqa/train-*"}
    ]

    return {
        "dataset_id": metadata.get("id"),
        "dataset_sha": metadata.get("sha"),
        "last_modified": metadata.get("lastModified"),
        "trialqa_splits": splits,
        "trialqa_data_files": data_files,
        "trialqa_siblings": trialqa_siblings,
        "trialqa_only_one_official_split_file": (
            has_only_expected_split and has_only_expected_file and has_only_expected_data_pattern
        ),
        "official_trialqa_unseen_split_available": not (
            has_only_expected_split and has_only_expected_file and has_only_expected_data_pattern
        ),
    }


def build_population_audit_report(
    dataset: trialqa.TrialQADataset,
    experiment_root: Path,
    *,
    huggingface_metadata: Mapping[str, object] | None = None,
) -> JsonObject:
    """Build the full TrialQA population audit report."""

    split = trialqa.create_split_manifest(dataset)
    report: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "id": trialqa.TRIALQA_DATASET_ID,
            "config": trialqa.TRIALQA_DATASET_CONFIG,
            "split": trialqa.TRIALQA_DATASET_SPLIT,
            "revision": dataset.revision,
            "parquet_path": str(dataset.path),
            "parquet_sha256": dataset.parquet_sha256,
            "row_count": len(dataset.rows),
            "split_seed": split["split_seed"],
            "split_counts": split["counts"],
        },
        "local_exposure": scan_experiment_exposure(dataset, experiment_root),
    }
    if huggingface_metadata is not None:
        report["huggingface_metadata"] = summarize_huggingface_metadata(huggingface_metadata)
    local_available = cast(
        bool,
        cast(Mapping[str, object], report["local_exposure"])[
            "local_trialqa_performance_population_available"
        ],
    )
    official_available = True
    if "huggingface_metadata" in report:
        official_available = cast(
            bool,
            cast(Mapping[str, object], report["huggingface_metadata"])[
                "official_trialqa_unseen_split_available"
            ],
        )
    report["decision"] = {
        "local_existing_parquet_supports_new_performance_claim": local_available,
        "official_unseen_trialqa_population_available": official_available,
        "next_step": (
            "create_or_fetch_new_trialqa_compatible_population"
            if not local_available and not official_available
            else "plan_prospective_trialqa_canary"
        ),
    }
    return report


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialQAPopulationAuditError(f"could not read JSON from {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TrialQAPopulationAuditError(f"JSON root must be an object: {path}")
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the pinned TrialQA parquet artifact.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Experiment root to scan for manifests and generation summaries.",
    )
    parser.add_argument(
        "--huggingface-metadata",
        type=Path,
        help="Optional saved Hugging Face dataset API JSON for upstream shape validation.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = trialqa.load_pinned_trialqa_parquet(args.dataset)
    metadata = _load_json(args.huggingface_metadata) if args.huggingface_metadata else None
    report = build_population_audit_report(
        dataset,
        args.experiment_root,
        huggingface_metadata=metadata,
    )
    payload = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

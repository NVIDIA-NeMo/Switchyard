# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline integrity, split, prediction, and judging helpers for TrialQA.

This module deliberately performs no downloads and no model calls.  Callers must
provide the pinned parquet file and inject a judge callback when scoring a
non-empty answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import pyarrow as pa
import pyarrow.parquet as pq

TRIALQA_DATASET_ID = "EdisonScientific/labbench2"
TRIALQA_DATASET_CONFIG = "trialqa"
TRIALQA_DATASET_SPLIT = "train"
TRIALQA_DATASET_REVISION = "27d12d72af24e3f70db8a99df63e567366cbdb80"
TRIALQA_PARQUET_NAME = "trialqa/train-00000-of-00001.parquet"
TRIALQA_PARQUET_SHA256 = "b571c93dce7497f678e019c17b1d4bc230da7a4d180c3cb9f22343ecc2efcd42"
TRIALQA_PARQUET_SIZE_BYTES = 71_452
TRIALQA_ROW_COUNT = 120

SERGEI_SPLIT_SEED = "trace2skill-trialqa"
SERGEI_TRAIN_FRACTION = 0.2
SERGEI_TRAIN_COUNT = 24
SERGEI_TEST_COUNT = 96
PROSPECTIVE_ALL_TEST_SPLIT_SEED = "trialqa-compatible-prospective-all-test-v1"

TRIALQA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("tag", pa.string()),
        pa.field("version", pa.string()),
        pa.field("question", pa.string()),
        pa.field("ideal", pa.string()),
        pa.field("files", pa.string()),
        pa.field("sources", pa.list_(pa.string())),
        pa.field("key_passage", pa.string()),
        pa.field("canary", pa.string()),
        pa.field("is_opensource", pa.bool_()),
        pa.field("ground_truth", pa.bool_()),
        pa.field("prompt_suffix", pa.string()),
        pa.field("type", pa.string()),
        pa.field(
            "mode",
            pa.struct(
                [
                    pa.field("file", pa.bool_()),
                    pa.field("retrieve", pa.bool_()),
                    pa.field("inject", pa.bool_()),
                ]
            ),
        ),
        pa.field("validator_params", pa.string()),
        pa.field("answer_regex", pa.string()),
    ]
)

JUDGE_SYSTEM_PROMPT = (
    "You are grading answers to TrialQA clinical-trial questions. "
    "Decide whether the submitted answer is semantically equivalent to the ideal answer. "
    "Be strict about numeric thresholds, eligibility boundaries, washout periods, counts, and units. "
    "Return only JSON with keys judge_result, score, rationale. "
    "judge_result must be correct, incorrect, or unsure. score must be 1.0 for correct and 0.0 otherwise."
)

Partition: TypeAlias = Literal["train", "test"]
JudgeLabel: TypeAlias = Literal["correct", "incorrect", "unsure"]
JudgeCallback: TypeAlias = Callable[[Mapping[str, object]], str]


class TrialQADataError(ValueError):
    """The local dataset or split manifest failed an integrity check."""


class TrialQAPredictionError(ValueError):
    """A prediction set failed structural or pairing validation."""


class TrialQAJudgeError(RuntimeError):
    """A non-empty answer could not be authoritatively judged."""


@dataclass(frozen=True)
class TrialQAMode:
    """Mode flags embedded in one TrialQA row."""

    file: bool
    retrieve: bool
    inject: bool


@dataclass(frozen=True)
class TrialQARow:
    """A validated row from the pinned TrialQA parquet artifact."""

    dataset_row_index: int
    id: str
    tag: str
    version: str
    question: str
    ideal: str
    files: str
    sources: tuple[str, ...]
    key_passage: str
    canary: str
    is_opensource: bool
    ground_truth: bool
    prompt_suffix: str
    trialqa_type: str
    mode: TrialQAMode
    validator_params: str
    answer_regex: str


@dataclass(frozen=True)
class TrialQADataset:
    """The fully validated local TrialQA dataset."""

    path: Path
    revision: str
    parquet_sha256: str
    rows: tuple[TrialQARow, ...]

    def row_by_id(self, row_id: str) -> TrialQARow:
        """Return one row by ID, raising for an unknown ID."""

        for row in self.rows:
            if row.id == row_id:
                return row
        raise KeyError(row_id)


@dataclass(frozen=True)
class ValidatedPrediction:
    """A prediction whose identity fields match the pinned dataset and split."""

    row_id: str
    dataset_row_index: int
    repeat_index: int
    condition: str
    answer: str
    answer_source: str
    partition: Partition
    question_group_key: str
    task_name: str
    trajectory_path: str | None = None


@dataclass(frozen=True)
class JudgeOutcome:
    """A normalized semantic-judge decision."""

    judge_result: JudgeLabel
    score: float
    rationale: str
    judge_available: bool
    judge_model: str | None
    judge_error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TrialQADataError(f"{field} must be a string")
    return value


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TrialQADataError(f"{field} must be a boolean")
    return value


def _row_from_mapping(raw: Mapping[str, object], index: int) -> TrialQARow:
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not all(isinstance(item, str) for item in sources_raw):
        raise TrialQADataError(f"row {index} sources must be a list of strings")
    mode_raw = raw.get("mode")
    if not isinstance(mode_raw, Mapping):
        raise TrialQADataError(f"row {index} mode must be an object")

    row = TrialQARow(
        dataset_row_index=index,
        id=_required_string(raw.get("id"), f"row {index} id"),
        tag=_required_string(raw.get("tag"), f"row {index} tag"),
        version=_required_string(raw.get("version"), f"row {index} version"),
        question=_required_string(raw.get("question"), f"row {index} question"),
        ideal=_required_string(raw.get("ideal"), f"row {index} ideal"),
        files=_required_string(raw.get("files"), f"row {index} files"),
        sources=tuple(cast(list[str], sources_raw)),
        key_passage=_required_string(raw.get("key_passage"), f"row {index} key_passage"),
        canary=_required_string(raw.get("canary"), f"row {index} canary"),
        is_opensource=_required_bool(raw.get("is_opensource"), f"row {index} is_opensource"),
        ground_truth=_required_bool(raw.get("ground_truth"), f"row {index} ground_truth"),
        prompt_suffix=_required_string(raw.get("prompt_suffix"), f"row {index} prompt_suffix"),
        trialqa_type=_required_string(raw.get("type"), f"row {index} type"),
        mode=TrialQAMode(
            file=_required_bool(mode_raw.get("file"), f"row {index} mode.file"),
            retrieve=_required_bool(mode_raw.get("retrieve"), f"row {index} mode.retrieve"),
            inject=_required_bool(mode_raw.get("inject"), f"row {index} mode.inject"),
        ),
        validator_params=_required_string(
            raw.get("validator_params"), f"row {index} validator_params"
        ),
        answer_regex=_required_string(raw.get("answer_regex"), f"row {index} answer_regex"),
    )
    if not row.id:
        raise TrialQADataError(f"row {index} has an empty id")
    if row.tag != "trialqa":
        raise TrialQADataError(f"row {index} has unexpected tag {row.tag!r}")
    if row.version != "1.0":
        raise TrialQADataError(f"row {index} has unexpected version {row.version!r}")
    return row


def _load_trialqa_parquet(
    path: Path,
    *,
    expected_sha256: str,
    expected_row_count: int,
    revision: str,
    expected_size_bytes: int | None = None,
) -> TrialQADataset:
    """Load a parquet file against explicit expectations (used by tests and the pinned wrapper)."""

    path = path.expanduser()
    if not path.is_file():
        raise TrialQADataError(f"TrialQA parquet does not exist: {path}")
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise TrialQADataError(
            f"TrialQA parquet size mismatch: expected {expected_size_bytes}, got {path.stat().st_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise TrialQADataError(
            f"TrialQA parquet SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise TrialQADataError(f"unable to read TrialQA parquet: {path}") from exc
    if not table.schema.equals(TRIALQA_SCHEMA, check_metadata=False):
        raise TrialQADataError(
            f"TrialQA schema mismatch: expected {TRIALQA_SCHEMA}, got {table.schema}"
        )
    if table.num_rows != expected_row_count:
        raise TrialQADataError(
            f"TrialQA row count mismatch: expected {expected_row_count}, got {table.num_rows}"
        )
    null_columns = [field.name for field in table.schema if table[field.name].null_count]
    if null_columns:
        raise TrialQADataError(f"TrialQA columns contain nulls: {', '.join(null_columns)}")

    rows = tuple(_row_from_mapping(raw, index) for index, raw in enumerate(table.to_pylist()))
    row_ids = [row.id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise TrialQADataError("TrialQA row IDs are not unique")
    return TrialQADataset(
        path=path.resolve(), revision=revision, parquet_sha256=actual_sha256, rows=rows
    )


def load_pinned_trialqa_parquet(path: Path) -> TrialQADataset:
    """Load only the exact pinned 120-row TrialQA parquet artifact."""

    return _load_trialqa_parquet(
        path,
        expected_sha256=TRIALQA_PARQUET_SHA256,
        expected_row_count=TRIALQA_ROW_COUNT,
        revision=TRIALQA_DATASET_REVISION,
        expected_size_bytes=TRIALQA_PARQUET_SIZE_BYTES,
    )


def load_trialqa_compatible_parquet(
    path: Path,
    *,
    expected_sha256: str,
    expected_row_count: int,
    revision: str,
) -> TrialQADataset:
    """Load a same-schema, non-official TrialQA-compatible parquet artifact.

    This intentionally does not replace :func:`load_pinned_trialqa_parquet`.
    Use it only for explicitly non-official prospective populations whose
    artifact hash and row count have been frozen before model generation.
    """

    if expected_row_count < 1:
        raise TrialQADataError("TrialQA-compatible row count must be positive")
    if not revision.strip():
        raise TrialQADataError("TrialQA-compatible revision must not be empty")
    return _load_trialqa_parquet(
        path,
        expected_sha256=expected_sha256,
        expected_row_count=expected_row_count,
        revision=revision,
    )


def stable_partition_key(row_id: str, dataset_row_index: int, seed: str) -> str:
    """Return the exact hash used by Sergei's TrialQA materializer."""

    return hashlib.sha256(f"{seed}\0{row_id}\0{dataset_row_index}".encode()).hexdigest()


def create_split_manifest(
    dataset: TrialQADataset,
    *,
    split_seed: str = SERGEI_SPLIT_SEED,
    train_fraction: float = SERGEI_TRAIN_FRACTION,
) -> dict[str, object]:
    """Create the deterministic 20/80 TrialQA split manifest."""

    if not split_seed:
        raise TrialQADataError("split seed must not be empty")
    if not 0.0 < train_fraction < 1.0:
        raise TrialQADataError("train fraction must be between 0 and 1")
    keyed = sorted(
        (
            stable_partition_key(row.id, row.dataset_row_index, split_seed),
            row.id,
        )
        for row in dataset.rows
    )
    train_n = int(round(len(dataset.rows) * train_fraction))
    train_ids = {row_id for _, row_id in keyed[:train_n]}
    manifest_rows: list[dict[str, object]] = []
    for row in dataset.rows:
        partition: Partition = "train" if row.id in train_ids else "test"
        manifest_rows.append(
            {
                "row_id": row.id,
                "dataset_row_index": row.dataset_row_index,
                "partition": partition,
                "split_hash": stable_partition_key(row.id, row.dataset_row_index, split_seed),
            }
        )
    train_count = len(train_ids)
    return {
        "schema_version": "trialqa_split_manifest.v1",
        "dataset_id": TRIALQA_DATASET_ID,
        "dataset_config": TRIALQA_DATASET_CONFIG,
        "split": TRIALQA_DATASET_SPLIT,
        "dataset_revision": dataset.revision,
        "parquet_sha256": dataset.parquet_sha256,
        "row_count": len(dataset.rows),
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "rows": manifest_rows,
        "counts": {"test": len(dataset.rows) - train_count, "train": train_count},
    }


def create_all_test_split_manifest(
    dataset: TrialQADataset,
    *,
    dataset_id: str = "trialqa-compatible-prospective",
    dataset_config: str = "clinicaltrials-gov",
    split: str = "prospective",
    split_seed: str = PROSPECTIVE_ALL_TEST_SPLIT_SEED,
) -> dict[str, object]:
    """Create an all-held-out split manifest for a frozen compatible population."""

    if not dataset_id.strip():
        raise TrialQADataError("dataset id must not be empty")
    if not dataset_config.strip():
        raise TrialQADataError("dataset config must not be empty")
    if not split.strip():
        raise TrialQADataError("dataset split must not be empty")
    if not split_seed.strip():
        raise TrialQADataError("split seed must not be empty")
    manifest_rows: list[dict[str, object]] = []
    for row in dataset.rows:
        manifest_rows.append(
            {
                "row_id": row.id,
                "dataset_row_index": row.dataset_row_index,
                "partition": "test",
                "split_hash": stable_partition_key(row.id, row.dataset_row_index, split_seed),
            }
        )
    return {
        "schema_version": "trialqa_split_manifest.v1",
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "split": split,
        "dataset_revision": dataset.revision,
        "parquet_sha256": dataset.parquet_sha256,
        "row_count": len(dataset.rows),
        "split_seed": split_seed,
        "train_fraction": 0.0,
        "rows": manifest_rows,
        "counts": {"test": len(dataset.rows), "train": 0},
    }


def validate_split_manifest(
    dataset: TrialQADataset,
    manifest: Mapping[str, object],
    *,
    split_seed: str = SERGEI_SPLIT_SEED,
    train_fraction: float = SERGEI_TRAIN_FRACTION,
) -> dict[str, Partition]:
    """Validate all split provenance and return row-ID-to-partition assignments."""

    expected = create_split_manifest(
        dataset, split_seed=split_seed, train_fraction=train_fraction
    )
    if dict(manifest) != expected:
        raise TrialQADataError("split manifest does not exactly match the pinned dataset and split")
    rows = cast(list[dict[str, object]], expected["rows"])
    return {cast(str, row["row_id"]): cast(Partition, row["partition"]) for row in rows}


def validate_all_test_split_manifest(
    dataset: TrialQADataset,
    manifest: Mapping[str, object],
    *,
    dataset_id: str = "trialqa-compatible-prospective",
    dataset_config: str = "clinicaltrials-gov",
    split: str = "prospective",
    split_seed: str = PROSPECTIVE_ALL_TEST_SPLIT_SEED,
) -> dict[str, Partition]:
    """Validate a frozen compatible all-test split manifest."""

    expected = create_all_test_split_manifest(
        dataset,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        split=split,
        split_seed=split_seed,
    )
    if dict(manifest) != expected:
        raise TrialQADataError(
            "all-test split manifest does not exactly match the compatible dataset"
        )
    rows = cast(list[dict[str, object]], expected["rows"])
    return {cast(str, row["row_id"]): cast(Partition, row["partition"]) for row in rows}


def question_group_key(row: TrialQARow) -> str:
    """Return Sergei's stable per-question grouping key."""

    task_id_hash = hashlib.sha256(row.id.encode()).hexdigest()[:12]
    return f"trialqa-{row.dataset_row_index:04d}-{task_id_hash}"


def task_name(row: TrialQARow, repeat_index: int) -> str:
    """Return Sergei's stable repeated-task name."""

    if repeat_index < 1:
        raise TrialQAPredictionError("repeat index must be >= 1")
    return f"{question_group_key(row)}-r{repeat_index:03d}"


def prediction_pair_key(prediction: ValidatedPrediction) -> tuple[str, int]:
    """Return the cross-condition join key for one prediction."""

    return prediction.question_group_key, prediction.repeat_index


def _prediction_string(raw: Mapping[str, object], field: str, *, nonempty: bool) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        raise TrialQAPredictionError(f"prediction {field} must be a string")
    value = value.strip()
    if nonempty and not value:
        raise TrialQAPredictionError(f"prediction {field} must not be empty")
    return value


def _prediction_int(raw: Mapping[str, object], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrialQAPredictionError(f"prediction {field} must be an integer")
    return value


def validate_predictions(
    records: Iterable[Mapping[str, object]],
    *,
    dataset: TrialQADataset,
    split_manifest: Mapping[str, object],
    partition: Partition,
    expected_repeats: int,
    required_condition: str | None = None,
    require_complete: bool = True,
) -> tuple[ValidatedPrediction, ...]:
    """Validate predictions and enforce unique, optionally complete pairing keys."""

    if partition not in {"train", "test"}:
        raise TrialQAPredictionError(f"invalid partition: {partition!r}")
    if expected_repeats < 1:
        raise TrialQAPredictionError("expected repeats must be >= 1")
    if required_condition is not None and not required_condition.strip():
        raise TrialQAPredictionError("required condition must not be empty")
    assignments = validate_split_manifest(dataset, split_manifest)
    rows_by_id = {row.id: row for row in dataset.rows}
    seen: set[tuple[str, int]] = set()
    validated: list[ValidatedPrediction] = []

    for raw in records:
        leaked_fields = {"ideal", "sources"}.intersection(raw)
        if leaked_fields:
            raise TrialQAPredictionError(
                f"prediction contains gold-only fields: {', '.join(sorted(leaked_fields))}"
            )
        row_id = _prediction_string(raw, "row_id", nonempty=True)
        row = rows_by_id.get(row_id)
        if row is None:
            raise TrialQAPredictionError(f"prediction has unknown row_id {row_id!r}")
        row_index = _prediction_int(raw, "dataset_row_index")
        if row_index != row.dataset_row_index:
            raise TrialQAPredictionError(f"prediction row index does not match row_id {row_id!r}")
        if assignments[row_id] != partition:
            raise TrialQAPredictionError(
                f"prediction row {row_id!r} belongs to {assignments[row_id]}, not {partition}"
            )
        repeat_index = _prediction_int(raw, "repeat_index")
        if not 1 <= repeat_index <= expected_repeats:
            raise TrialQAPredictionError(
                f"prediction repeat_index must be in [1, {expected_repeats}]"
            )
        condition = _prediction_string(raw, "condition", nonempty=True)
        if required_condition is not None and condition != required_condition:
            raise TrialQAPredictionError(
                f"prediction condition {condition!r} does not match {required_condition!r}"
            )
        expected_group = question_group_key(row)
        supplied_group = raw.get("question_group_key")
        if supplied_group is not None and supplied_group != expected_group:
            raise TrialQAPredictionError(f"prediction question_group_key does not match {row_id!r}")
        expected_task = task_name(row, repeat_index)
        supplied_task = raw.get("task_name")
        if supplied_task is not None and supplied_task != expected_task:
            raise TrialQAPredictionError(f"prediction task_name does not match {row_id!r}")
        supplied_partition = raw.get("partition")
        if supplied_partition is not None and supplied_partition != partition:
            raise TrialQAPredictionError(f"prediction partition does not match {row_id!r}")
        supplied_repeats = raw.get("n_repeats")
        if supplied_repeats is not None and supplied_repeats != expected_repeats:
            raise TrialQAPredictionError(f"prediction n_repeats does not match {row_id!r}")
        trajectory_path = raw.get("trajectory_path")
        if trajectory_path is not None and not isinstance(trajectory_path, str):
            raise TrialQAPredictionError("prediction trajectory_path must be a string or null")

        prediction = ValidatedPrediction(
            row_id=row_id,
            dataset_row_index=row_index,
            repeat_index=repeat_index,
            condition=condition,
            answer=_prediction_string(raw, "answer", nonempty=False),
            answer_source=_prediction_string(raw, "answer_source", nonempty=True),
            partition=partition,
            question_group_key=expected_group,
            task_name=expected_task,
            trajectory_path=trajectory_path,
        )
        pair_key = prediction_pair_key(prediction)
        if pair_key in seen:
            raise TrialQAPredictionError(f"duplicate prediction pair key: {pair_key!r}")
        seen.add(pair_key)
        validated.append(prediction)

    if require_complete:
        expected_keys = {
            (question_group_key(row), repeat_index)
            for row in dataset.rows
            if assignments[row.id] == partition
            for repeat_index in range(1, expected_repeats + 1)
        }
        if seen != expected_keys:
            missing = len(expected_keys - seen)
            unexpected = len(seen - expected_keys)
            raise TrialQAPredictionError(
                f"prediction pair keys are incomplete: {missing} missing, {unexpected} unexpected"
            )
    return tuple(validated)


def validate_paired_conditions(
    left: Iterable[ValidatedPrediction], right: Iterable[ValidatedPrediction]
) -> tuple[tuple[str, int], ...]:
    """Require two conditions to contain the exact same unique pair keys."""

    left_keys = [prediction_pair_key(prediction) for prediction in left]
    right_keys = [prediction_pair_key(prediction) for prediction in right]
    if len(left_keys) != len(set(left_keys)) or len(right_keys) != len(set(right_keys)):
        raise TrialQAPredictionError("condition contains duplicate prediction pair keys")
    if set(left_keys) != set(right_keys):
        raise TrialQAPredictionError("conditions do not contain identical prediction pair keys")
    return tuple(sorted(left_keys))


def build_judge_payload(
    *, question: str, ideal: str, answer: str, model: str
) -> dict[str, object]:
    """Build the exact deterministic semantic-judge payload used by the reference."""

    if not model.strip():
        raise TrialQAJudgeError("judge model must not be empty")
    user = {
        "question": question,
        "ideal_answer": ideal,
        "submitted_answer": answer,
    }
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }


def parse_judge_result(content: str, *, model: str) -> JudgeOutcome:
    """Parse the reference judge JSON, raising instead of silently falling back."""

    if not isinstance(content, str) or not content.strip():
        raise TrialQAJudgeError("judge returned empty or non-string content")
    match = re.search(r"\{.*\}", content, re.S)
    try:
        parsed = json.loads(match.group(0) if match else content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise TrialQAJudgeError("judge returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise TrialQAJudgeError("judge JSON must be an object")
    raw_result = parsed.get("judge_result")
    if not isinstance(raw_result, str):
        raise TrialQAJudgeError("judge_result must be a string")
    normalized_result = raw_result.lower()
    if normalized_result not in {"correct", "incorrect", "unsure"}:
        raise TrialQAJudgeError(f"judge returned invalid result {raw_result!r}")
    result = cast(JudgeLabel, normalized_result)
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        raise TrialQAJudgeError("judge rationale must be a string")
    return JudgeOutcome(
        judge_result=result,
        score=1.0 if result == "correct" else 0.0,
        rationale=rationale,
        judge_available=True,
        judge_model=model,
    )


def score_semantic_answer(
    *,
    question: str,
    ideal: str,
    answer: str,
    model: str,
    judge: JudgeCallback,
) -> JudgeOutcome:
    """Score an answer with an injected judge; empty answers deterministically score zero."""

    if not answer.strip():
        return JudgeOutcome(
            judge_result="incorrect",
            score=0.0,
            rationale="No answer was submitted.",
            judge_available=False,
            judge_model=None,
        )
    payload = build_judge_payload(question=question, ideal=ideal, answer=answer, model=model)
    try:
        content = judge(payload)
    except Exception as exc:
        raise TrialQAJudgeError("semantic judge call failed") from exc
    return parse_judge_result(content, model=model)


def build_reward_record(
    *,
    row: TrialQARow,
    prediction: ValidatedPrediction,
    n_repeats: int,
    outcome: JudgeOutcome,
    process_metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the reference-compatible ``trialqa_reward.v1`` record."""

    if prediction.row_id != row.id or prediction.dataset_row_index != row.dataset_row_index:
        raise TrialQAPredictionError("reward row and prediction identities do not match")
    if prediction.question_group_key != question_group_key(row):
        raise TrialQAPredictionError("reward prediction has an invalid question group key")
    if prediction.task_name != task_name(row, prediction.repeat_index):
        raise TrialQAPredictionError("reward prediction has an invalid task name")
    if n_repeats < prediction.repeat_index:
        raise TrialQAPredictionError("reward n_repeats is smaller than repeat_index")
    return {
        "schema_version": "trialqa_reward.v1",
        "reward": outcome.score,
        "score": outcome.score,
        "judge_result": outcome.judge_result,
        "judge_rationale": outcome.rationale,
        "judge_available": outcome.judge_available,
        "judge_model": outcome.judge_model,
        "judge_error": outcome.judge_error,
        "question": row.question,
        "ideal": row.ideal,
        "answer": prediction.answer,
        "answer_source": prediction.answer_source,
        "sources": list(row.sources),
        "row_id": row.id,
        "dataset_row_index": row.dataset_row_index,
        "partition": prediction.partition,
        "repeat_index": prediction.repeat_index,
        "n_repeats": n_repeats,
        "question_group_key": prediction.question_group_key,
        "task_name": prediction.task_name,
        "process_metrics": dict(process_metrics or {}),
    }

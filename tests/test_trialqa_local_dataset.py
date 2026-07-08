# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmark import trialqa_local_dataset as trialqa


def _fixture_rows(*, duplicate_id: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(trialqa.TRIALQA_ROW_COUNT):
        row_id = "trialqa-000" if duplicate_id and index == 1 else f"trialqa-{index:03d}"
        rows.append(
            {
                "id": row_id,
                "tag": "trialqa",
                "version": "1.0",
                "question": f"Question {index}?",
                "ideal": f"Ideal answer {index}",
                "files": "",
                "sources": [f"NCT{index:08d}"],
                "key_passage": f"Passage {index}",
                "canary": "",
                "is_opensource": True,
                "ground_truth": True,
                "prompt_suffix": "",
                "type": "text",
                "mode": {"file": False, "retrieve": True, "inject": False},
                "validator_params": "{}",
                "answer_regex": "",
            }
        )
    return rows


def _write_parquet(
    path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    schema: pa.Schema = trialqa.TRIALQA_SCHEMA,
) -> str:
    table = pa.Table.from_pylist(rows or _fixture_rows(), schema=schema)
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fixture(path: Path, *, rows: list[dict[str, object]] | None = None) -> trialqa.TrialQADataset:
    digest = _write_parquet(path, rows=rows)
    return trialqa._load_trialqa_parquet(
        path,
        expected_sha256=digest,
        expected_row_count=trialqa.TRIALQA_ROW_COUNT,
        revision=trialqa.TRIALQA_DATASET_REVISION,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> trialqa.TrialQADataset:
    return _load_fixture(tmp_path / "trialqa.parquet")


def test_pinned_artifact_constants_are_exact() -> None:
    assert trialqa.TRIALQA_DATASET_ID == "EdisonScientific/labbench2"
    assert trialqa.TRIALQA_DATASET_CONFIG == "trialqa"
    assert trialqa.TRIALQA_DATASET_SPLIT == "train"
    assert trialqa.TRIALQA_DATASET_REVISION == "27d12d72af24e3f70db8a99df63e567366cbdb80"
    assert trialqa.TRIALQA_PARQUET_NAME == "trialqa/train-00000-of-00001.parquet"
    assert (
        trialqa.TRIALQA_PARQUET_SHA256
        == "b571c93dce7497f678e019c17b1d4bc230da7a4d180c3cb9f22343ecc2efcd42"
    )
    assert trialqa.TRIALQA_PARQUET_SIZE_BYTES == 71_452
    assert trialqa.TRIALQA_ROW_COUNT == 120


def test_validated_loader_enforces_exact_schema_and_unique_rows(
    dataset: trialqa.TrialQADataset,
) -> None:
    assert len(dataset.rows) == 120
    assert len({row.id for row in dataset.rows}) == 120
    assert dataset.rows[0].sources == ("NCT00000000",)
    assert dataset.rows[0].mode == trialqa.TrialQAMode(
        file=False, retrieve=True, inject=False
    )
    assert dataset.revision == trialqa.TRIALQA_DATASET_REVISION


def test_trialqa_compatible_loader_is_hash_bound_but_not_pinned(tmp_path: Path) -> None:
    path = tmp_path / "prospective.parquet"
    digest = _write_parquet(path)

    loaded = trialqa.load_trialqa_compatible_parquet(
        path,
        expected_sha256=digest,
        expected_row_count=trialqa.TRIALQA_ROW_COUNT,
        revision="clinicaltrials-gov-prospective-v1",
    )

    assert loaded.parquet_sha256 == digest
    assert loaded.revision == "clinicaltrials-gov-prospective-v1"
    assert len(loaded.rows) == trialqa.TRIALQA_ROW_COUNT


def test_trialqa_compatible_loader_rejects_empty_revision(tmp_path: Path) -> None:
    path = tmp_path / "prospective.parquet"
    digest = _write_parquet(path)

    with pytest.raises(trialqa.TrialQADataError, match="revision must not be empty"):
        trialqa.load_trialqa_compatible_parquet(
            path,
            expected_sha256=digest,
            expected_row_count=trialqa.TRIALQA_ROW_COUNT,
            revision=" ",
        )


def test_loader_rejects_hash_tampering(tmp_path: Path) -> None:
    path = tmp_path / "trialqa.parquet"
    digest = _write_parquet(path)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(trialqa.TrialQADataError, match="SHA256 mismatch"):
        trialqa._load_trialqa_parquet(
            path,
            expected_sha256=digest,
            expected_row_count=120,
            revision=trialqa.TRIALQA_DATASET_REVISION,
        )


def test_loader_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "trialqa.parquet"
    drifted_schema = pa.schema(
        field for field in trialqa.TRIALQA_SCHEMA if field.name != "answer_regex"
    )
    digest = _write_parquet(path, schema=drifted_schema)

    with pytest.raises(trialqa.TrialQADataError, match="schema mismatch"):
        trialqa._load_trialqa_parquet(
            path,
            expected_sha256=digest,
            expected_row_count=120,
            revision=trialqa.TRIALQA_DATASET_REVISION,
        )


def test_loader_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    path = tmp_path / "trialqa.parquet"
    digest = _write_parquet(path, rows=_fixture_rows(duplicate_id=True))

    with pytest.raises(trialqa.TrialQADataError, match="not unique"):
        trialqa._load_trialqa_parquet(
            path,
            expected_sha256=digest,
            expected_row_count=120,
            revision=trialqa.TRIALQA_DATASET_REVISION,
        )


def test_sergei_split_is_exact_and_deterministic(dataset: trialqa.TrialQADataset) -> None:
    first = trialqa.create_split_manifest(dataset)
    second = trialqa.create_split_manifest(dataset)
    assert first == second
    assert first["counts"] == {"test": 96, "train": 24}
    assert first["split_seed"] == "trace2skill-trialqa"
    assert first["train_fraction"] == 0.2

    rows = first["rows"]
    assert isinstance(rows, list)
    train_ids = {row["row_id"] for row in rows if row["partition"] == "train"}
    assert train_ids == {
        "trialqa-005",
        "trialqa-007",
        "trialqa-012",
        "trialqa-014",
        "trialqa-017",
        "trialqa-024",
        "trialqa-026",
        "trialqa-028",
        "trialqa-029",
        "trialqa-031",
        "trialqa-035",
        "trialqa-036",
        "trialqa-042",
        "trialqa-044",
        "trialqa-055",
        "trialqa-088",
        "trialqa-092",
        "trialqa-095",
        "trialqa-100",
        "trialqa-101",
        "trialqa-104",
        "trialqa-106",
        "trialqa-108",
        "trialqa-109",
    }
    assert rows[0]["split_hash"] == (
        "7aab9c69cf6252dd26dec5d4e314b924ed0842cb101e2bf5b0a3aa42b76e3de5"
    )


def test_split_manifest_round_trips_and_rejects_tampering(
    dataset: trialqa.TrialQADataset,
) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    round_tripped = json.loads(json.dumps(manifest))
    assignments = trialqa.validate_split_manifest(dataset, round_tripped)
    assert list(assignments).index("trialqa-000") == 0
    assert sum(partition == "train" for partition in assignments.values()) == 24

    rows = round_tripped["rows"]
    rows[0]["partition"] = "train" if rows[0]["partition"] == "test" else "test"
    with pytest.raises(trialqa.TrialQADataError, match="does not exactly match"):
        trialqa.validate_split_manifest(dataset, round_tripped)


def _prediction_records(
    dataset: trialqa.TrialQADataset,
    manifest: dict[str, object],
    *,
    condition: str,
    repeats: int = 2,
) -> list[dict[str, object]]:
    assignments = trialqa.validate_split_manifest(dataset, manifest)
    records: list[dict[str, object]] = []
    for row in dataset.rows:
        if assignments[row.id] != "test":
            continue
        for repeat_index in range(1, repeats + 1):
            records.append(
                {
                    "row_id": row.id,
                    "dataset_row_index": row.dataset_row_index,
                    "repeat_index": repeat_index,
                    "n_repeats": repeats,
                    "condition": condition,
                    "partition": "test",
                    "question_group_key": trialqa.question_group_key(row),
                    "task_name": trialqa.task_name(row, repeat_index),
                    "answer": f"Answer {repeat_index}",
                    "answer_source": "file",
                }
            )
    return records


def test_prediction_validation_produces_complete_pair_keys(
    dataset: trialqa.TrialQADataset,
) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    records = _prediction_records(dataset, manifest, condition="baseline")

    predictions = trialqa.validate_predictions(
        records,
        dataset=dataset,
        split_manifest=manifest,
        partition="test",
        expected_repeats=2,
        required_condition="baseline",
    )

    assert len(predictions) == 192
    assert len({trialqa.prediction_pair_key(item) for item in predictions}) == 192
    assert predictions[0].answer_source == "file"


def test_prediction_validation_rejects_duplicates_and_incomplete_pairs(
    dataset: trialqa.TrialQADataset,
) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    records = _prediction_records(dataset, manifest, condition="baseline")

    with pytest.raises(trialqa.TrialQAPredictionError, match="duplicate"):
        trialqa.validate_predictions(
            [*records, records[0]],
            dataset=dataset,
            split_manifest=manifest,
            partition="test",
            expected_repeats=2,
        )

    with pytest.raises(trialqa.TrialQAPredictionError, match="incomplete"):
        trialqa.validate_predictions(
            records[:-1],
            dataset=dataset,
            split_manifest=manifest,
            partition="test",
            expected_repeats=2,
        )


def test_prediction_validation_rejects_gold_leakage(dataset: trialqa.TrialQADataset) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    records = _prediction_records(dataset, manifest, condition="baseline")
    records[0]["ideal"] = "leaked"

    with pytest.raises(trialqa.TrialQAPredictionError, match="gold-only"):
        trialqa.validate_predictions(
            records,
            dataset=dataset,
            split_manifest=manifest,
            partition="test",
            expected_repeats=2,
        )


def test_paired_conditions_require_identical_keys(dataset: trialqa.TrialQADataset) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    baseline = trialqa.validate_predictions(
        _prediction_records(dataset, manifest, condition="baseline"),
        dataset=dataset,
        split_manifest=manifest,
        partition="test",
        expected_repeats=2,
    )
    skilled = trialqa.validate_predictions(
        _prediction_records(dataset, manifest, condition="skilled"),
        dataset=dataset,
        split_manifest=manifest,
        partition="test",
        expected_repeats=2,
    )
    assert len(trialqa.validate_paired_conditions(baseline, skilled)) == 192
    with pytest.raises(trialqa.TrialQAPredictionError, match="identical"):
        trialqa.validate_paired_conditions(baseline, skilled[:-1])


def test_judge_payload_exactly_matches_reference() -> None:
    payload = trialqa.build_judge_payload(
        question="Who is eligible?",
        ideal="Adults aged 18 or older.",
        answer="People who are at least 18.",
        model="judge-model",
    )

    assert payload == {
        "model": "judge-model",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": trialqa.JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    '{"question": "Who is eligible?", '
                    '"ideal_answer": "Adults aged 18 or older.", '
                    '"submitted_answer": "People who are at least 18."}'
                ),
            },
        ],
    }


def test_judge_parser_extracts_json_and_derives_binary_score() -> None:
    outcome = trialqa.parse_judge_result(
        'prefix {"judge_result":"CORRECT","score":0.0,"rationale":"Equivalent"} suffix',
        model="judge-model",
    )

    assert outcome == trialqa.JudgeOutcome(
        judge_result="correct",
        score=1.0,
        rationale="Equivalent",
        judge_available=True,
        judge_model="judge-model",
    )


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "[]",
        '{"score": 1.0, "rationale": "missing label"}',
        '{"judge_result": "maybe", "score": 0.0, "rationale": "bad label"}',
        '{"judge_result": "correct", "score": 1.0, "rationale": []}',
    ],
)
def test_judge_parser_fails_closed_on_invalid_nonempty_results(content: str) -> None:
    with pytest.raises(trialqa.TrialQAJudgeError):
        trialqa.parse_judge_result(content, model="judge-model")


def test_empty_answer_is_deterministic_zero_without_calling_judge() -> None:
    def must_not_run(_payload: object) -> NoReturn:
        raise AssertionError("judge must not be called for an empty answer")

    outcome = trialqa.score_semantic_answer(
        question="Question",
        ideal="Ideal",
        answer="  ",
        model="judge-model",
        judge=must_not_run,
    )

    assert outcome.score == 0.0
    assert outcome.judge_result == "incorrect"
    assert outcome.judge_available is False


def test_nonempty_judge_exception_fails_closed() -> None:
    def failing_judge(_payload: object) -> NoReturn:
        raise TimeoutError("timeout")

    with pytest.raises(trialqa.TrialQAJudgeError, match="call failed") as exc_info:
        trialqa.score_semantic_answer(
            question="Question",
            ideal="Ideal",
            answer="Answer",
            model="judge-model",
            judge=failing_judge,
        )
    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_reward_record_matches_reference_schema(dataset: trialqa.TrialQADataset) -> None:
    manifest = trialqa.create_split_manifest(dataset)
    record = _prediction_records(dataset, manifest, condition="baseline", repeats=1)[0]
    prediction = trialqa.validate_predictions(
        [record],
        dataset=dataset,
        split_manifest=manifest,
        partition="test",
        expected_repeats=1,
        require_complete=False,
    )[0]
    row = dataset.row_by_id(prediction.row_id)
    outcome = trialqa.JudgeOutcome(
        judge_result="correct",
        score=1.0,
        rationale="Equivalent",
        judge_available=True,
        judge_model="judge-model",
    )

    reward = trialqa.build_reward_record(
        row=row,
        prediction=prediction,
        n_repeats=1,
        outcome=outcome,
        process_metrics={"tool_call_count": 3},
    )

    assert reward["schema_version"] == "trialqa_reward.v1"
    assert reward["reward"] == 1.0
    assert reward["judge_result"] == "correct"
    assert reward["row_id"] == row.id
    assert reward["question_group_key"] == trialqa.question_group_key(row)
    assert reward["process_metrics"] == {"tool_call_count": 3}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

import benchmark.trialqa_local_dataset as trialqa
import benchmark.trialqa_local_prospective_population as prospective


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
            sources=("https://clinicaltrials.gov/study/NCT00000001",)
            if index == 0
            else (f"https://clinicaltrials.gov/study/NCT{index + 10:08d}",),
            key_passage="",
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
        path=tmp_path / "official.parquet",
        revision=trialqa.TRIALQA_DATASET_REVISION,
        parquet_sha256=trialqa.TRIALQA_PARQUET_SHA256,
        rows=rows,
    )


def _study(nct_id: str, *, title: str = "A Fresh Trial") -> dict[str, object]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": title,
            },
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": 42},
            },
            "eligibilityModule": {
                "minimumAge": "18 Years",
                "sex": "ALL",
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Objective response rate",
                        "timeFrame": "Up to 24 weeks",
                    }
                ]
            },
        }
    }


def test_exposed_nct_ids_include_official_sources(tmp_path: Path) -> None:
    assert "NCT00000001" in prospective.exposed_nct_ids(_dataset(tmp_path))


def test_rows_from_studies_excludes_official_trialqa_ncts() -> None:
    rows, report = prospective.rows_from_studies(
        [
            _study("NCT00000001", title="Already Exposed"),
            _study("NCT99990001", title="Fresh One"),
            _study("NCT99990002", title="Fresh Two"),
        ],
        exclude_ncts={"NCT00000001"},
        limit=2,
    )

    assert [row["sources"] for row in rows] == [
        ["https://clinicaltrials.gov/study/NCT99990001"],
        ["https://clinicaltrials.gov/study/NCT99990002"],
    ]
    assert report["skipped"] == {"official_trialqa_source_nct": 1}
    assert report["selected_count"] == 2
    assert set(report["template_counts"]) <= {
        "enrollment_count",
        "minimum_age",
        "overall_status",
        "phase",
        "primary_outcome_measure",
        "primary_outcome_time_frame",
        "sex",
    }


def test_write_population_uses_trialqa_schema_and_report_marks_nonofficial(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    rows, selection = prospective.rows_from_studies(
        [_study("NCT99990001"), _study("NCT99990002")],
        exclude_ncts=prospective.exposed_nct_ids(dataset),
        limit=2,
    )
    output = tmp_path / "prospective.parquet"
    digest = prospective.write_population(rows, output)

    table = pq.read_table(output)
    assert table.schema.equals(trialqa.TRIALQA_SCHEMA, check_metadata=False)
    assert table.num_rows == 2

    report = prospective.build_report(
        rows=rows,
        output=output,
        output_sha256=digest,
        exclude_dataset=dataset,
        excluded_ncts=prospective.exposed_nct_ids(dataset),
        fetch_report={"network": False},
        selection_report=selection,
    )

    assert report["population"]["official_labbench2"] is False
    assert report["population"]["sha256"] == digest
    assert report["official_trialqa_exclusion"]["selected_ncts_overlap_official_trialqa"] == []
    assert report["use_constraints"] == {
        "model_calls": 0,
        "must_not_be_reported_as_official_labbench2_trialqa": True,
        "performance_eligible_only_if_manifest_is_frozen_before_generation": True,
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a fresh TrialQA-compatible prospective population from ClinicalTrials.gov.

This script is intentionally zero model-spend.  It creates rows with the same
schema as LABBench2 TrialQA, but it does not label them as official LABBench2
examples.  The purpose is to unblock a prospective Switchyard canary after the
official 120-row TrialQA population has been fully exposed locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

import benchmark.trialqa_local_dataset as trialqa

SCHEMA_VERSION = "switchyard.trialqa_prospective_population.v1"
CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_FIELDS = (
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "Phase",
    "EnrollmentCount",
    "MinimumAge",
    "Sex",
    "PrimaryOutcomeMeasure",
    "PrimaryOutcomeTimeFrame",
    "InterventionName",
)
DEFAULT_QUERY_CONDITIONS = (
    "cancer",
    "diabetes",
    "HIV",
    "heart failure",
    "asthma",
    "breast cancer",
)
NCT_ID = re.compile(r"\bNCT\d{8}\b", re.I)
JsonObject = dict[str, Any]


class TrialQAProspectivePopulationError(RuntimeError):
    """Prospective population generation failed a safety or provenance check."""


@dataclass(frozen=True)
class ProspectiveQuestion:
    template_id: str
    question: str
    ideal: str
    key_passage: str
    validator_params: JsonObject


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _get(mapping: Mapping[str, object], *path: str) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_list_item(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, list) or not value or not isinstance(value[0], Mapping):
        return None
    return cast(Mapping[str, object], value[0])


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int_string(value: object) -> str | None:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip().isdigit():
        return value.strip()
    return None


def _phase_label(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    labels: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        token = item.strip().upper()
        label = {
            "EARLY_PHASE1": "Early Phase 1",
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "NA": "Not applicable",
        }.get(token, item.strip())
        if label:
            labels.append(label)
    return ", ".join(labels) if labels else None


def _sex_label(value: object) -> str | None:
    token = _string(value)
    if token is None:
        return None
    return {"ALL": "All", "MALE": "Male", "FEMALE": "Female"}.get(token.upper(), token)


def _title(study: Mapping[str, object]) -> str | None:
    return _string(_get(study, "protocolSection", "identificationModule", "briefTitle")) or _string(
        _get(study, "protocolSection", "identificationModule", "officialTitle")
    )


def _nct_id(study: Mapping[str, object]) -> str | None:
    raw = _string(_get(study, "protocolSection", "identificationModule", "nctId"))
    return raw.upper() if raw and NCT_ID.fullmatch(raw) else None


def _primary_outcome(study: Mapping[str, object]) -> Mapping[str, object] | None:
    return _first_list_item(_get(study, "protocolSection", "outcomesModule", "primaryOutcomes"))


def prospective_questions_for_study(study: Mapping[str, object]) -> list[ProspectiveQuestion]:
    """Return direct-evidence question candidates for one ClinicalTrials.gov study."""

    nct = _nct_id(study)
    title = _title(study)
    if nct is None or title is None:
        return []
    protocol = cast(Mapping[str, object], study["protocolSection"])
    design = _get(protocol, "designModule")
    eligibility = _get(protocol, "eligibilityModule")
    status = _get(protocol, "statusModule")
    primary = _primary_outcome(study)

    candidates: list[ProspectiveQuestion] = []

    phase = _phase_label(_get(cast(Mapping[str, object], design or {}), "phases"))
    if phase:
        candidates.append(
            ProspectiveQuestion(
                template_id="phase",
                question=f"In the {title} trial ({nct}), what phase is listed for the study?",
                ideal=phase,
                key_passage=f"{nct} designModule.phases = {phase}",
                validator_params={"nct_id": nct, "field_path": "protocolSection.designModule.phases"},
            )
        )

    enrollment = _int_string(
        _get(cast(Mapping[str, object], design or {}), "enrollmentInfo", "count")
    )
    if enrollment:
        candidates.append(
            ProspectiveQuestion(
                template_id="enrollment_count",
                question=(
                    f"In the {title} trial ({nct}), how many participants are listed "
                    "in the enrollment count?"
                ),
                ideal=enrollment,
                key_passage=f"{nct} designModule.enrollmentInfo.count = {enrollment}",
                validator_params={
                    "nct_id": nct,
                    "field_path": "protocolSection.designModule.enrollmentInfo.count",
                },
            )
        )

    minimum_age = _string(_get(cast(Mapping[str, object], eligibility or {}), "minimumAge"))
    if minimum_age:
        candidates.append(
            ProspectiveQuestion(
                template_id="minimum_age",
                question=(
                    f"In the {title} trial ({nct}), what minimum age is listed in the "
                    "eligibility criteria?"
                ),
                ideal=minimum_age,
                key_passage=f"{nct} eligibilityModule.minimumAge = {minimum_age}",
                validator_params={
                    "nct_id": nct,
                    "field_path": "protocolSection.eligibilityModule.minimumAge",
                },
            )
        )

    sex = _sex_label(_get(cast(Mapping[str, object], eligibility or {}), "sex"))
    if sex:
        candidates.append(
            ProspectiveQuestion(
                template_id="sex",
                question=(
                    f"In the {title} trial ({nct}), what sex eligibility value is listed?"
                ),
                ideal=sex,
                key_passage=f"{nct} eligibilityModule.sex = {sex}",
                validator_params={"nct_id": nct, "field_path": "protocolSection.eligibilityModule.sex"},
            )
        )

    overall_status = _string(_get(cast(Mapping[str, object], status or {}), "overallStatus"))
    if overall_status:
        candidates.append(
            ProspectiveQuestion(
                template_id="overall_status",
                question=f"In the {title} trial ({nct}), what is the official overall status?",
                ideal=overall_status,
                key_passage=f"{nct} statusModule.overallStatus = {overall_status}",
                validator_params={
                    "nct_id": nct,
                    "field_path": "protocolSection.statusModule.overallStatus",
                },
            )
        )

    if primary is not None:
        measure = _string(primary.get("measure"))
        if measure:
            candidates.append(
                ProspectiveQuestion(
                    template_id="primary_outcome_measure",
                    question=(
                        f"In the {title} trial ({nct}), what is the first listed primary "
                        "outcome measure?"
                    ),
                    ideal=measure,
                    key_passage=f"{nct} outcomesModule.primaryOutcomes[0].measure = {measure}",
                    validator_params={
                        "nct_id": nct,
                        "field_path": "protocolSection.outcomesModule.primaryOutcomes[0].measure",
                    },
                )
            )
        timeframe = _string(primary.get("timeFrame"))
        if timeframe:
            candidates.append(
                ProspectiveQuestion(
                    template_id="primary_outcome_time_frame",
                    question=(
                        f"In the {title} trial ({nct}), what time frame is listed for "
                        "the first primary outcome measure?"
                    ),
                    ideal=timeframe,
                    key_passage=f"{nct} outcomesModule.primaryOutcomes[0].timeFrame = {timeframe}",
                    validator_params={
                        "nct_id": nct,
                        "field_path": "protocolSection.outcomesModule.primaryOutcomes[0].timeFrame",
                    },
                )
            )

    return candidates


def exposed_nct_ids(dataset: trialqa.TrialQADataset) -> set[str]:
    """Extract all NCT identifiers visible in the official local TrialQA source."""

    found: set[str] = set()
    for row in dataset.rows:
        haystacks = [row.question, row.ideal, row.key_passage, row.validator_params, *row.sources]
        for text in haystacks:
            found.update(match.group(0).upper() for match in NCT_ID.finditer(text))
    return found


def _choose_question(
    candidates: Sequence[ProspectiveQuestion],
    template_counts: Counter[str],
) -> ProspectiveQuestion | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: (template_counts[item.template_id], item.template_id))


def rows_from_studies(
    studies: Iterable[Mapping[str, object]],
    *,
    exclude_ncts: set[str],
    limit: int,
) -> tuple[list[JsonObject], JsonObject]:
    """Convert ClinicalTrials.gov studies into TrialQA-schema row mappings."""

    if limit < 1:
        raise TrialQAProspectivePopulationError("limit must be positive")
    rows: list[JsonObject] = []
    seen_ncts: set[str] = set()
    template_counts: Counter[str] = Counter()
    skipped = Counter[str]()

    for study in studies:
        nct = _nct_id(study)
        if nct is None:
            skipped["missing_nct"] += 1
            continue
        if nct in exclude_ncts:
            skipped["official_trialqa_source_nct"] += 1
            continue
        if nct in seen_ncts:
            skipped["duplicate_nct"] += 1
            continue
        chosen = _choose_question(prospective_questions_for_study(study), template_counts)
        if chosen is None:
            skipped["no_supported_question"] += 1
            continue
        seen_ncts.add(nct)
        template_counts[chosen.template_id] += 1
        row_id = f"prospective-ctgov-{nct.lower()}-{chosen.template_id}"
        validator_params = {
            **chosen.validator_params,
            "schema_version": "trialqa_prospective_validator_params.v1",
            "template_id": chosen.template_id,
            "source": "clinicaltrials.gov-api-v2",
        }
        rows.append(
            {
                "id": row_id,
                "tag": "trialqa",
                "version": "1.0",
                "question": chosen.question,
                "ideal": chosen.ideal,
                "files": "",
                "sources": [f"https://clinicaltrials.gov/study/{nct}"],
                "key_passage": chosen.key_passage,
                "canary": "",
                "is_opensource": True,
                "ground_truth": True,
                "prompt_suffix": "",
                "type": "text",
                "mode": {"file": False, "retrieve": True, "inject": False},
                "validator_params": json.dumps(validator_params, sort_keys=True),
                "answer_regex": "",
            }
        )
        if len(rows) >= limit:
            break

    return rows, {
        "selected_count": len(rows),
        "selected_nct_ids": [source["sources"][0].rsplit("/", 1)[-1] for source in rows],
        "template_counts": dict(sorted(template_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }


def _read_studies_json(path: Path) -> list[Mapping[str, object]]:
    data = json.loads(path.read_text())
    if isinstance(data, Mapping) and isinstance(data.get("studies"), list):
        studies = data["studies"]
    elif isinstance(data, list):
        studies = data
    else:
        raise TrialQAProspectivePopulationError("studies JSON must be a list or CT.gov response")
    if not all(isinstance(item, Mapping) for item in studies):
        raise TrialQAProspectivePopulationError("each study must be an object")
    return [cast(Mapping[str, object], item) for item in studies]


def fetch_studies(
    *,
    query_conditions: Sequence[str],
    page_size: int,
    max_pages: int,
    timeout_seconds: float,
) -> tuple[list[Mapping[str, object]], JsonObject]:
    """Fetch candidate studies from ClinicalTrials.gov API v2."""

    if page_size < 1 or page_size > 1000:
        raise TrialQAProspectivePopulationError("page size must be 1..1000")
    if max_pages < 1:
        raise TrialQAProspectivePopulationError("max pages must be positive")
    studies: list[Mapping[str, object]] = []
    requests: list[JsonObject] = []
    for condition in query_conditions:
        page_token: str | None = None
        for page_index in range(max_pages):
            params = {
                "query.cond": condition,
                "filter.overallStatus": "COMPLETED",
                "pageSize": str(page_size),
                "format": "json",
                "fields": ",".join(CTGOV_FIELDS),
            }
            if page_token:
                params["pageToken"] = page_token
            url = f"{CTGOV_API}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, Mapping) or not isinstance(payload.get("studies"), list):
                raise TrialQAProspectivePopulationError("ClinicalTrials.gov response is malformed")
            batch = payload["studies"]
            studies.extend(cast(list[Mapping[str, object]], batch))
            requests.append(
                {
                    "condition": condition,
                    "page_index": page_index,
                    "returned_study_count": len(batch),
                    "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                }
            )
            token = payload.get("nextPageToken")
            page_token = token if isinstance(token, str) and token else None
            if page_token is None:
                break
    return studies, {
        "api": CTGOV_API,
        "fields": list(CTGOV_FIELDS),
        "query_conditions": list(query_conditions),
        "page_size": page_size,
        "max_pages": max_pages,
        "requests": requests,
        "fetched_study_count": len(studies),
    }


def write_population(rows: Sequence[Mapping[str, object]], output: Path) -> str:
    """Write rows with the exact TrialQA parquet schema and return its SHA-256."""

    if not rows:
        raise TrialQAProspectivePopulationError("cannot write an empty prospective population")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=trialqa.TRIALQA_SCHEMA)
    pq.write_table(table, output)  # type: ignore[no-untyped-call]
    return _sha256_file(output)


def build_report(
    *,
    rows: Sequence[Mapping[str, object]],
    output: Path,
    output_sha256: str,
    exclude_dataset: trialqa.TrialQADataset,
    excluded_ncts: set[str],
    fetch_report: Mapping[str, object],
    selection_report: Mapping[str, object],
) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if rows else "failed",
        "population": {
            "kind": "trialqa-compatible-clinicaltrials-gov-prospective",
            "official_labbench2": False,
            "path": str(output),
            "sha256": output_sha256,
            "row_count": len(rows),
            "row_ids_sha256": hashlib.sha256(
                _canonical_json([cast(str, row["id"]) for row in rows])
            ).hexdigest(),
            "source_urls": [cast(list[str], row["sources"])[0] for row in rows],
        },
        "official_trialqa_exclusion": {
            "dataset_id": trialqa.TRIALQA_DATASET_ID,
            "dataset_revision": exclude_dataset.revision,
            "parquet_sha256": exclude_dataset.parquet_sha256,
            "excluded_nct_count": len(excluded_ncts),
            "selected_ncts_overlap_official_trialqa": sorted(
                set(cast(list[str], selection_report["selected_nct_ids"])) & excluded_ncts
            ),
        },
        "fetch": dict(fetch_report),
        "selection": dict(selection_report),
        "use_constraints": {
            "performance_eligible_only_if_manifest_is_frozen_before_generation": True,
            "must_not_be_reported_as_official_labbench2_trialqa": True,
            "model_calls": 0,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--input-studies-json", type=Path)
    parser.add_argument("--query-condition", action="append", dest="query_conditions")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exclude_dataset = trialqa.load_pinned_trialqa_parquet(args.exclude_dataset)
    excluded = exposed_nct_ids(exclude_dataset)
    if args.input_studies_json is not None:
        studies = _read_studies_json(args.input_studies_json)
        fetch_report: JsonObject = {
            "source": str(args.input_studies_json),
            "fetched_study_count": len(studies),
            "network": False,
        }
    else:
        studies, fetch_report = fetch_studies(
            query_conditions=tuple(args.query_conditions or DEFAULT_QUERY_CONDITIONS),
            page_size=args.page_size,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout_seconds,
        )
        fetch_report["network"] = True

    rows, selection_report = rows_from_studies(studies, exclude_ncts=excluded, limit=args.limit)
    if len(rows) < args.limit:
        raise TrialQAProspectivePopulationError(
            f"only generated {len(rows)} rows, fewer than requested limit {args.limit}"
        )
    digest = write_population(rows, args.output)
    report = build_report(
        rows=rows,
        output=args.output,
        output_sha256=digest,
        exclude_dataset=exclude_dataset,
        excluded_ncts=excluded,
        fetch_report=fetch_report,
        selection_report=selection_report,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

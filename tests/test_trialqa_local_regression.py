# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmark.trialqa_local_demo as demo
import benchmark.trialqa_local_regression as regression


def _events(
    path: Path,
    *,
    operation: str,
    nct_id: str,
    payload: object,
    child_status: str = "success",
) -> Path:
    loader = {
        "id": "load",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "trialqa_load_active_skill",
        "arguments": {},
        "result": {"content": [{"type": "text", "text": "loaded"}]},
        "error": None,
        "status": "completed",
    }
    evidence = {
        "id": "evidence",
        "type": "mcp_tool_call",
        "server": "tooluniverse",
        "tool": "execute_tool",
        "arguments": {
            "tool_name": operation,
            "arguments_json": json.dumps({"nct_ids": [nct_id]}),
        },
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"status": child_status, "data": payload}),
                }
            ],
            "structured_content": None,
        },
        "error": None,
        "status": "completed",
    }
    events = [
        {"type": "thread.started", "thread_id": "thread"},
        {"type": "item.started", "item": {**loader, "status": "in_progress"}},
        {"type": "item.completed", "item": loader},
        {"type": "item.started", "item": {**evidence, "status": "in_progress"}},
        {"type": "item.completed", "item": evidence},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _generation(
    tmp_path: Path, *, ordinal: int, answer: str, events: Path
) -> demo.GenerationResult:
    generation_path = tmp_path / f"generation-q{ordinal}.json"
    generation_path.write_text("{}\n", encoding="utf-8")
    return demo.GenerationResult(
        manifest_id="trialqa-full-regression",
        task_id=f"q{ordinal}-r1-treatment",
        pair_id=f"q{ordinal}-r1",
        row_id=f"row-{ordinal}",
        dataset_row_index=ordinal,
        partition="test",
        condition="treatment",
        repeat_index=1,
        n_repeats=5,
        answer=answer,
        answer_source=demo.FINAL_ANSWER_TEXT_SOURCE,
        session_dir=tmp_path,
        stats_path=tmp_path / "stats.json",
        trajectory_path=tmp_path / "trajectory.jsonl",
        codex_events_path=events,
        final_output_path=tmp_path / "final.json",
        generation_path=generation_path,
        stats={},
        usage={"input_tokens": 10, "output_tokens": 2},
        artifact_sha256={},
    )


@pytest.mark.parametrize(
    ("ordinal", "answer", "operation", "payload"),
    [
        (
            7,
            "The starting dose was 10 mg administered once daily (QD).",
            "extract_clinical_trial_adverse_events",
            {"NCT ID": "NCT01970865", "groups": [{"title": "10 mg QD"}]},
        ),
        (
            2,
            "The minimum periods are 12 weeks and 4 weeks, respectively.",
            "get_clinical_trial_eligibility_criteria",
            {
                "NCT ID": "NCT03249792",
                "eligibility_criteria": (
                    "HIV RNA suppression for >=12 weeks; stable drug regimen for >=4 weeks"
                ),
            },
        ),
        (
            5,
            "The next assessment is Dose 6.",
            "get_clinical_trial_outcome_measures",
            {
                "NCT ID": "NCT01693562",
                "primary_outcomes": [
                    {
                        "measure": "Anti-drug antibodies (ADA)",
                        "timeFrame": "even-numbered doses after D2",
                    }
                ],
            },
        ),
    ],
)
def test_reviewed_mechanism_answers_and_direct_evidence_pass(
    tmp_path: Path,
    ordinal: int,
    answer: str,
    operation: str,
    payload: object,
) -> None:
    spec = regression.MECHANISM_SPECS[ordinal]
    events = _events(
        tmp_path / "events.jsonl",
        operation=operation,
        nct_id=spec.nct_id,
        payload=payload,
    )
    generation = _generation(tmp_path, ordinal=ordinal, answer=answer, events=events)

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "pass"
    assert result["kill_reasons"] == []
    assert result["checks"] == {
        "normalized_answer": True,
        "no_unsupported_quantitative_value": True,
        "direct_successful_operation": True,
        "supporting_payload": True,
    }


@pytest.mark.parametrize(
    "answer",
    [
        "The minimum periods are 12 weeks and 4 weeks, respectively.",
        "12 weeks for HIV RNA <50 copies/mL and 4 weeks for stable drug/dose regimen",
        "HIV RNA <50 copies/mL for 12 weeks and stable drug/dose regimen for 4 weeks",
        "4 weeks for the stable drug/dose regimen and 12 weeks for HIV RNA <50 copies/mL",
        "For HIV RNA <50 copies/mL, 12 weeks; for the stable drug/dose regimen, 4 weeks",
        (
            "For HIV-infected participants in the MK-2118-001 trial (NCT03249792), the "
            "eligibility criteria require: (1) HIV RNA <50 copies/mL (or below the lower "
            "limit of quantification) for ≥12 weeks prior to screening, and (2) a stable "
            "antiretroviral regimen without drug or dose changes for ≥4 weeks prior to "
            "study entry (Day 1)."
        ),
    ],
)
def test_q2_answer_accepts_correct_labeled_or_declared_order(answer: str) -> None:
    assert regression._q2_answer(answer) == (True, True)


@pytest.mark.parametrize(
    "answer",
    [
        "12 weeks for the stable drug/dose regimen and 4 weeks for HIV RNA <50 copies/mL",
        "Stable drug regimen for 12 weeks and HIV RNA suppression for 4 weeks",
        "HIV RNA suppression was not required for 12 weeks and the stable regimen was 4 weeks",
        "HIV RNA suppression for 12 weeks, stable drug regimen for 4 weeks, and follow-up for 8 weeks",
        "The minimum periods are 4 weeks and 12 weeks, respectively.",
        "The minimum periods are 12 weeks and 4 weeks.",
        (
            "12 weeks for HIV RNA suppression and 4 weeks for the stable drug regimen; "
            "that is 12 weeks and 4 weeks"
        ),
        "HIV RNA suppression for <12 weeks and the stable drug regimen for 4 weeks",
        "HIV RNA suppression for less than 12 weeks and the stable drug regimen for 4 weeks",
        "HIV RNA suppression for at most 12 weeks and the stable drug regimen for 4 weeks",
        "HIV RNA suppression for no more than 12 weeks and the stable drug regimen for 4 weeks",
        "12 weeks for HIV RNA suppression or 4 weeks for the stable drug regimen",
        "HIV RNA suppression for 12 weeks or the stable drug regimen for 4 weeks",
        "The minimum periods are 12 weeks or 4 weeks, respectively.",
    ],
)
def test_q2_answer_rejects_reversed_negated_or_extra_week_values(answer: str) -> None:
    answer_ok, _quantity_ok = regression._q2_answer(answer)

    assert answer_ok is False


@pytest.mark.parametrize(
    ("ordinal", "answer"),
    [
        (7, "The starting dose was 25 mg once daily."),
        (2, "The minimum periods are 4 weeks and 12 weeks, respectively."),
        (5, "The next assessment is Dose 8."),
    ],
)
def test_incorrect_normalized_answers_kill(tmp_path: Path, ordinal: int, answer: str) -> None:
    spec = regression.MECHANISM_SPECS[ordinal]
    payloads = {
        7: {"NCT ID": "NCT01970865", "groups": [{"title": "10 mg once daily"}]},
        2: {
            "NCT ID": "NCT03249792",
            "eligibility_criteria": (
                "HIV RNA suppression for at least 12 weeks and stable drug regimen "
                "for at least 4 weeks"
            ),
        },
        5: {
            "NCT ID": "NCT01693562",
            "primary_outcomes": [
                {
                    "measure": "Anti-drug antibodies",
                    "timeFrame": "even-numbered doses after D2",
                }
            ],
        },
    }
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload=payloads[ordinal],
    )
    generation = _generation(tmp_path, ordinal=ordinal, answer=answer, events=events)

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert "normalized_answer" in result["kill_reasons"]


def test_q7_unsupported_quantitative_value_kills(tmp_path: Path) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "groups": [{"title": "10 mg QD"}]},
    )
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="The dose was 10 mg or 25 mg once daily.",
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert result["checks"]["no_unsupported_quantitative_value"] is False


def test_wrong_direct_operation_kills_even_with_correct_answer_and_payload(tmp_path: Path) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation="ClinicalTrials_get_study",
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "groups": [{"title": "10 mg QD"}]},
    )
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert result["checks"]["direct_successful_operation"] is False
    assert result["checks"]["supporting_payload"] is False


@pytest.mark.parametrize(
    ("payload", "child_status", "direct", "support"),
    [
        ("25 mg twice daily", "success", True, False),
        ("10 mg QD", "error", False, False),
    ],
)
def test_direct_operation_requires_successful_supporting_child_payload(
    tmp_path: Path,
    payload: str,
    child_status: str,
    direct: bool,
    support: bool,
) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "groups": [{"title": payload}]},
        child_status=child_status,
    )
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert result["checks"]["direct_successful_operation"] is direct
    assert result["checks"]["supporting_payload"] is support


def test_success_prefixed_truncated_direct_payload_remains_checkable(tmp_path: Path) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "groups": [{"title": "unused"}]},
    )
    raw = events.read_text(encoding="utf-8").splitlines()
    completed = json.loads(raw[4])
    completed["item"]["result"]["content"][0]["text"] = (
        '{"status": "success", "data": [{"NCT ID": "NCT01970865", '
        '"freq_threshold": "5%", "groups": ['
        '{"id": "EG000", "title": "10 mg QD (Phase 1)", '
        '"description": "10 mg was orally given once daily"}, '
        '{"id": "EG001", "title": "25 mg BID", '
        '"description": "A later escalation group"}]}'
        "... [TRUNCATED]"
    )
    raw[4] = json.dumps(completed)
    events.write_text("\n".join(raw) + "\n", encoding="utf-8")
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "pass"


@pytest.mark.parametrize(
    ("ordinal", "answer"),
    [
        (2, "The periods are not 12 weeks and 4 weeks."),
        (2, "The stable drug regimen was 12 weeks and HIV RNA suppression was 4 weeks."),
        (2, "The periods were 12 months and 4 weeks."),
        (5, "The next assessment is not Dose 6."),
        (7, "The starting dose was not 10 mg once daily."),
        (7, "It was 10 mg once daily, while 25 milligrams weekly was the starting dose."),
    ],
)
def test_negated_reversed_or_conflicting_answers_fail_closed(ordinal: int, answer: str) -> None:
    answer_ok, _quantity_ok = regression.MECHANISM_SPECS[ordinal].answer_check(answer)

    assert answer_ok is False


@pytest.mark.parametrize(
    ("ordinal", "payload"),
    [
        (
            2,
            {
                "NCT ID": "NCT03249792",
                "eligibility_criteria": (
                    "HIV RNA suppression was not required for 12 weeks; stable drug "
                    "regimen was required for 4 weeks"
                ),
            },
        ),
        (
            2,
            {
                "NCT ID": "NCT03249792",
                "eligibility_criteria": (
                    "HIV RNA suppression for 12 weeks or 8 weeks; stable drug regimen for 4 weeks"
                ),
            },
        ),
        (
            5,
            {
                "NCT ID": "NCT01693562",
                "primary_outcomes": [
                    {
                        "measure": "Anti-drug antibodies",
                        "timeFrame": "Dose 6 was not next; Dose 8 was next",
                    }
                ],
            },
        ),
        (
            7,
            {
                "NCT ID": "NCT01970865",
                "groups": [{"title": "10 mg QD was not starting; 25 mg weekly was"}],
            },
        ),
    ],
)
def test_negated_or_conflicting_supporting_payload_kills(
    tmp_path: Path, ordinal: int, payload: object
) -> None:
    spec = regression.MECHANISM_SPECS[ordinal]
    answers = {
        2: "The minimum periods are 12 weeks and 4 weeks, respectively.",
        5: "The next assessment is Dose 6.",
        7: "The starting dose was 10 mg once daily.",
    }
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload=payload,
    )
    generation = _generation(
        tmp_path,
        ordinal=ordinal,
        answer=answers[ordinal],
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert result["checks"]["supporting_payload"] is False


@pytest.mark.parametrize(
    ("ordinal", "payload"),
    [
        (
            2,
            {
                "NCT ID": "NCT03249792",
                "eligibility_criteria": (
                    "HIV RNA suppression for at least 12 weeks; stable drug regimen "
                    "for at least 4 weeks. Participants must not use another therapy."
                ),
            },
        ),
        (
            5,
            {
                "NCT ID": "NCT01693562",
                "primary_outcomes": [
                    {
                        "measure": "Anti-drug antibodies (ADA)",
                        "timeFrame": "even-numbered doses after D2",
                        "description": "An unrelated endpoint was not analyzed.",
                    }
                ],
            },
        ),
    ],
)
def test_unrelated_negation_outside_supporting_field_does_not_false_kill(
    tmp_path: Path, ordinal: int, payload: object
) -> None:
    spec = regression.MECHANISM_SPECS[ordinal]
    answers = {
        2: "The minimum periods are 12 weeks and 4 weeks, respectively.",
        5: "The next assessment is Dose 6.",
    }
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload=payload,
    )
    generation = _generation(
        tmp_path,
        ordinal=ordinal,
        answer=answers[ordinal],
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "pass"
    assert result["checks"]["supporting_payload"] is True


def test_direct_operation_requires_exact_argument_and_payload_nct(tmp_path: Path) -> None:
    spec = regression.MECHANISM_SPECS[7]
    supported_payload = {
        "NCT ID": spec.nct_id,
        "groups": [{"title": "10 mg QD"}],
    }
    wrong_argument_events = _events(
        tmp_path / "wrong-argument.jsonl",
        operation=spec.operation,
        nct_id="NCT00000000",
        payload=supported_payload,
    )
    wrong_argument = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=wrong_argument_events,
    )

    wrong_argument_result = regression._evaluate_generation(wrong_argument, spec)

    assert wrong_argument_result["checks"]["direct_successful_operation"] is False
    assert wrong_argument_result["checks"]["supporting_payload"] is False

    wrong_payload_events = _events(
        tmp_path / "wrong-payload.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": "NCT00000000", "groups": [{"title": "10 mg QD"}]},
    )
    wrong_payload = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=wrong_payload_events,
    )

    wrong_payload_result = regression._evaluate_generation(wrong_payload, spec)

    assert wrong_payload_result["checks"]["direct_successful_operation"] is True
    assert wrong_payload_result["checks"]["supporting_payload"] is False


def test_support_must_come_from_the_reviewed_response_field(tmp_path: Path) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "references": "10 mg once daily"},
    )
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=events,
    )

    result = regression._evaluate_generation(generation, spec)

    assert result["decision"] == "kill"
    assert result["checks"]["supporting_payload"] is False


def test_manifest_id_is_recomputed_from_every_manifest_field() -> None:
    seed: dict[str, object] = {"kind": "full", "candidate": {"candidate_id": "v8"}}
    manifest_id = "trialqa-full-" + hashlib.sha256(demo._canonical_json(seed)).hexdigest()[:20]
    manifest = {"manifest_id": manifest_id, **seed}

    assert regression._validate_content_addressed_manifest_id(manifest) == manifest_id
    manifest["candidate"] = {"candidate_id": "another-candidate"}
    with pytest.raises(regression.TrialQARegressionError, match="content-addressed"):
        regression._validate_content_addressed_manifest_id(manifest)


def test_manifest_session_attestation_uses_launch_and_native_proof_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = regression.MECHANISM_SPECS[7]
    events = _events(
        tmp_path / "events.jsonl",
        operation=spec.operation,
        nct_id=spec.nct_id,
        payload={"NCT ID": spec.nct_id, "groups": [{"title": "10 mg QD"}]},
    )
    generation = _generation(
        tmp_path,
        ordinal=7,
        answer="10 mg once daily",
        events=events,
    )
    context = {"candidate_id": "v8"}
    active = {"candidate_id": "v8"}
    hashes = {"run-context.json": "sha256:context"}
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        regression.batch,
        "_launch_metadata",
        lambda *_args, **_kwargs: (context, active, hashes),
    )

    def validate_proof(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(regression.batch, "_validate_session_proof", validate_proof)

    regression._validate_manifest_session_attestation(
        manifest={"candidate": {"candidate_id": "v8"}},
        task={"task_id": generation.task_id},
        capture=tmp_path,
        generation=generation,
    )

    assert observed == {
        "session_dir": generation.session_dir,
        "expected_context": context,
        "expected_active": active,
        "launch_sha256": hashes,
    }


def test_checker_refuses_other_ordinal_and_baseline_task() -> None:
    groups = [f"q{index}" for index in range(8)]
    with pytest.raises(regression.TrialQARegressionError, match="only q2/q5/q7"):
        regression._mechanism_spec_for_task(
            {"task_id": "q3", "question_group_key": "q3", "condition": "treatment"},
            groups,
        )
    with pytest.raises(regression.TrialQARegressionError, match="only q2/q5/q7"):
        regression._mechanism_spec_for_task(
            {"task_id": "q7", "question_group_key": "q7", "condition": "baseline"},
            groups,
        )

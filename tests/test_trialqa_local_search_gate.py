# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

import benchmark.trialqa_local_search_gate as gate


def _semantic_result() -> dict[str, object]:
    return {
        "task_id": "trialqa-q2-r001-treatment",
        "question_ordinal": 2,
        "decision": "pass",
        "bindings": {
            "generation_sha256": f"sha256:{'1' * 64}",
            "codex_events_sha256": f"sha256:{'2' * 64}",
        },
    }


def _call(tool_name: str, arguments: Mapping[str, object], data: object) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "id": f"call-{tool_name}-{json.dumps(arguments, sort_keys=True)}",
            "type": "mcp_tool_call",
            "server": "tooluniverse",
            "tool": "execute_tool",
            "status": "completed",
            "error": None,
            "arguments": {
                "tool_name": tool_name,
                "arguments_json": json.dumps(arguments),
            },
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "success", "data": data}),
                    }
                ],
                "structured_content": None,
            },
        },
    }


def _search(query: str, *, resolve: bool = False, title: str | None = None) -> dict[str, object]:
    studies = (
        [
            {
                "nct_id": "NCT03249792",
                "brief_title": title or f"Study of {query} in adults",
            }
        ]
        if resolve
        else []
    )
    return _call(
        gate.SEARCH_OPERATION,
        {"query_term": query},
        {
            "studies": studies,
            "total_count": 1 if resolve else 0,
            "next_page_token": None,
        },
    )


def _structured_search(arguments: Mapping[str, object]) -> dict[str, object]:
    return _call(
        gate.SEARCH_OPERATION,
        arguments,
        {
            "studies": [
                {
                    "nct_id": "NCT03249792",
                    "brief_title": "Study of MK-2118 in adults",
                }
            ],
            "total_count": 1,
            "next_page_token": None,
        },
    )


def _getter(*, nct_id: str = "NCT03249792") -> dict[str, object]:
    return _call(
        "get_clinical_trial_eligibility_criteria",
        {"nct_ids": [nct_id]},
        [{"NCT ID": nct_id, "eligibility_criteria": "12 weeks; 4 weeks"}],
    )


def test_single_unique_resolution_then_expected_getter_passes() -> None:
    result = gate.evaluate_search_gate(
        events=[_search("MK-2118-001", resolve=True), _getter()],
        semantic_result=_semantic_result(),
    )

    assert result["decision"] == "pass"
    assert result["search_count"] == 1
    assert result["resolution_index"] == 0
    assert result["post_resolution_search_count"] == 0
    assert result["repeated_argument_count"] == 0
    assert result["semantic_result"] == _semantic_result()


def test_query_condition_and_intervention_are_valid_and_resolve_by_preference() -> None:
    result = gate.evaluate_search_gate(
        events=[_structured_search({"query_intr": "MK-", "query_cond": "HIV"}), _getter()],
        semantic_result=_semantic_result(),
    )

    assert result["decision"] == "pass"
    assert result["checks"]["search_arguments_valid"] is True
    assert result["resolution"]["normalized_query"] == "mk"


def test_v9_q2_search_sequence_deterministically_kills_after_semantic_pass() -> None:
    queries = [
        "MK-2118-001",
        "MK-2118 HIV",
        "MK-2118",
        "HIV RNA <50 copies",
        "MK-2118-001 HIV",
        "MK-2118 HIV RNA",
        "MK-2118 001",
        "MK-2118",
    ]
    events = [_search(query, resolve=index in {0, 2, 6, 7}) for index, query in enumerate(queries)]
    events.append(_getter())

    result = gate.evaluate_search_gate(events=events, semantic_result=_semantic_result())

    assert result["semantic_result"]["decision"] == "pass"
    assert result["decision"] == "kill"
    assert result["search_count"] == 8
    assert result["resolution_index"] == 0
    assert result["repeated_argument_count"] == 1
    assert result["repeated_normalized_query_count"] == 2
    assert result["post_resolution_search_count"] == 7
    assert result["checks"]["search_arguments_valid"] is True
    assert "at_most_three_searches" in result["kill_reasons"]
    assert "no_search_after_first_resolution" in result["kill_reasons"]


@pytest.mark.parametrize(
    "events",
    [
        [
            _search(
                "MK-2118-001",
                resolve=True,
                title="An unrelated clinical trial",
            ),
            _getter(),
        ],
        [_search("MK-2118-001", resolve=True), _getter(nct_id="NCT00000000")],
    ],
)
def test_resolution_title_or_next_getter_tamper_kills(
    events: list[dict[str, object]],
) -> None:
    result = gate.evaluate_search_gate(events=events, semantic_result=_semantic_result())

    assert result["decision"] == "kill"

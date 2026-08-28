# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: validate that all records in skills_pr.json parse correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from craft_taskgen.importers.common import extract_pr_ref, load_records

SKILLS_PR_PATH = Path(__file__).resolve().parent.parent / "skills_pr.json"


@pytest.mark.skipif(not SKILLS_PR_PATH.exists(), reason="skills_pr.json not present")
def test_all_records_have_extractable_pr_refs():
    rows = load_records(SKILLS_PR_PATH, allow_json_object_map=True, record_id_key="source_record_id")
    assert len(rows) == 63

    failed = []
    repos = set()
    for row in rows:
        ref = extract_pr_ref(row)
        if ref is None:
            failed.append(row.get("source_record_id", row))
        else:
            repos.add(ref[0])

    assert not failed, f"Could not extract PR ref from: {failed}"
    assert len(repos) == 11

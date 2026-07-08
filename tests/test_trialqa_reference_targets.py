# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path


def test_trialqa_reference_targets_encode_expected_skill_distillation_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmark" / "fixtures" / "trialqa_reference_targets_v1.json"

    targets = json.loads(path.read_text(encoding="utf-8"))

    assert targets["schema_version"] == "switchyard.trialqa_reference_targets.v1"
    source = targets["source"]
    assert source["document"] == "2026-05-21-Skill-Distillation-Demo-V2.pptx.pdf"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", source["document_sha256"])
    assert source["slide_title"] == "Key Results Across Models and Distillation Conditions for TrialQA"
    assert source["pdf_page"] == 13
    assert targets["population"] == {
        "dataset": "LABBench2 TrialQA",
        "heldout_questions": 96,
        "repeats_per_question": 5,
        "trials": 480,
        "tool_provider": "ToolUniverse MCP",
        "injected_context": False,
    }
    assert targets["super"]["base"]["accuracy"] == 0.61
    assert targets["super"]["r1"]["accuracy"] >= 0.738
    assert targets["super"]["r1"]["token_reduction"] >= 0.30
    assert targets["super"]["r1"]["operational_call_reduction"] >= 0.45
    assert targets["super"]["r1b"]["token_reduction"] >= 0.29
    assert targets["nano"]["r1b"]["token_reduction"] >= 0.45
    assert targets["local_success_interpretation"]["first_canary"].startswith("directional")

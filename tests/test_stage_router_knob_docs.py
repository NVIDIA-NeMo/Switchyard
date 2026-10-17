# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-router knob reference drift tests.

The TOML schema and the stage-router guide each list every key a
``stage_router`` route accepts. These tests derive that key list from the Rust
config structs so a new or renamed knob fails here until both pages catch up.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_ALGORITHM = ROOT / "crates" / "switchyard-runner" / "src" / "algorithm.rs"
LIBSY_STAGE = ROOT / "crates" / "libsy" / "src" / "algorithms" / "util" / "stage.rs"
SCHEMA = ROOT / "docs" / "reference" / "toml_schema.md"
GUIDE = ROOT / "docs" / "routing_algorithms" / "stage_router_routing.md"

TOOL_SEMANTICS_CATEGORIES = ("observe", "mutate", "plan", "new")


def struct_fields(source: str, name: str) -> list[str]:
    match = re.search(rf"pub struct {name} \{{\n(.*?)\n\}}", source, re.S)
    assert match, f"struct {name} not found"
    return re.findall(r"^\s*(?:pub )?([a-z_]+): ", match.group(1), re.M)


def section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    stop = text.index(end, begin + len(start))
    return text[begin:stop]


def expected_keys() -> set[str]:
    runner = RUNNER_ALGORITHM.read_text()
    libsy = LIBSY_STAGE.read_text()
    keys = {"picker", "classifier", "handoff_notes", "subagents"}
    for field in struct_fields(runner, "StageTierConfig"):
        if field == "tool_semantics":
            keys.update(f"tool_semantics.{name}" for name in TOOL_SEMANTICS_CATEGORIES)
        elif field == "handoff_notes":
            keys.update(
                f"handoff_notes.{name}" for name in struct_fields(libsy, "HandoffNoteConfig")
            )
        else:
            keys.add(field)
    keys.update(f"classifier.{name}" for name in struct_fields(runner, "StageClassifierConfig"))
    return keys


def decision_sources() -> list[str]:
    libsy = LIBSY_STAGE.read_text()
    begin = libsy.index("impl DecisionSource")
    block = libsy[begin : libsy.index("\n}\n", begin)]
    names = re.findall(r'Self::\w+ => "([a-z_-]+)"', block)
    assert names, "DecisionSource::as_str arms not found"
    return names


def missing_keys(text: str) -> list[str]:
    return sorted(key for key in expected_keys() if f"`{key}`" not in text)


def test_expected_keys_cover_the_known_tables() -> None:
    keys = expected_keys()
    assert {"confidence_threshold", "classifier.target", "handoff_notes.escalation_note"} <= keys


def test_schema_lists_every_stage_router_key() -> None:
    text = section(SCHEMA.read_text(), "### `stage_router`", "### `auto`")
    assert missing_keys(text) == []


def test_guide_knob_reference_lists_every_stage_router_key() -> None:
    text = section(GUIDE.read_text(), "### Knob reference", "### Minimal route")
    assert missing_keys(text) == []


def test_guide_lists_every_decision_source() -> None:
    text = section(GUIDE.read_text(), "### Decision sources", "## When")
    missing = [name for name in decision_sources() if f"`{name}`" not in text]
    assert missing == []

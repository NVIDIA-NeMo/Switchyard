# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-router knob reference drift tests.

The TOML schema and the stage-router guide each list every key a
``stage_router`` route accepts, and name the shared classifier keys the route
rejects. These tests derive both lists from the Rust config types, so a new,
renamed, or removed knob fails here until both pages catch up, and a table row
for a key the code does not accept fails too.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_ALGORITHM = ROOT / "crates" / "switchyard-runner" / "src" / "algorithm.rs"
LIBSY_STAGE = ROOT / "crates" / "libsy" / "src" / "algorithms" / "util" / "stage.rs"
LIBSY_TOOL_SIGNALS = ROOT / "crates" / "libsy" / "src" / "algorithms" / "util" / "tool_signals.rs"
SCHEMA = ROOT / "docs" / "reference" / "toml_schema.md"
GUIDE = ROOT / "docs" / "routing_algorithms" / "stage_router_routing.md"

# `StageClassifierConfig` is shared with composite routes. A stage_router route
# rejects these fields (see `StageClassifierConfig::task_classifier_config`), so
# the pages name them as rejected rather than listing them as knobs.
REJECTED_CLASSIFIER_KEYS = {"classifier.response_schema", "classifier.policy"}

FIELD = re.compile(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?([a-z_][a-z0-9_]*)\s*:", re.M)
SERDE_NAME = re.compile(r'(?:rename|alias)\s*=\s*"([^"]+)"')


def block(source: str, start: str) -> str:
    """The body of the struct or enum variant whose declaration starts with `start`."""
    begin = source.index(start) + len(start)
    depth = 1
    for position in range(begin, len(source)):
        char = source[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[begin:position]
    raise AssertionError(f"unterminated block after {start!r}")


def toml_keys(body: str) -> list[str]:
    """TOML keys a serde struct body accepts: field names, renames, and aliases.

    Returns one entry per field; a field with aliases yields a `|`-joined entry.
    Fields marked `#[serde(flatten)]` are skipped, since their keys come from
    another struct.
    """
    keys: list[str] = []
    pending_attrs: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#["):
            pending_attrs.append(stripped)
            continue
        if stripped.startswith("//") or not stripped:
            continue
        match = FIELD.match(line)
        if match is None:
            pending_attrs = []
            continue
        attrs = " ".join(pending_attrs)
        pending_attrs = []
        if "flatten" in attrs:
            continue
        names = SERDE_NAME.findall(attrs)
        renamed = re.search(r'rename\s*=\s*"([^"]+)"', attrs)
        primary = renamed.group(1) if renamed else match.group(1)
        keys.append("|".join([primary, *(name for name in names if name != primary)]))
    return keys


def expected_keys() -> set[str]:
    runner = RUNNER_ALGORITHM.read_text()
    libsy = LIBSY_STAGE.read_text()
    signals = LIBSY_TOOL_SIGNALS.read_text()
    keys: set[str] = set()
    for key in toml_keys(block(runner, "StageRouter {")):
        keys.update(key.split("|"))
    for key in toml_keys(block(runner, "pub struct StageTierConfig {")):
        field = key.split("|")[0]
        if field == "tool_semantics":
            keys.update(
                f"tool_semantics.{name}"
                for name in toml_keys(block(signals, "pub struct ToolSemantics {"))
            )
        elif field == "handoff_notes":
            keys.add("handoff_notes")
            keys.update(
                f"handoff_notes.{name}"
                for name in toml_keys(block(libsy, "pub struct HandoffNoteConfig {"))
            )
        else:
            keys.update(key.split("|"))
    keys.add("classifier")
    for key in toml_keys(block(runner, "pub struct StageClassifierConfig {")):
        keys.update(f"classifier.{name}" for name in key.split("|"))
    return keys - REJECTED_CLASSIFIER_KEYS


def decision_sources() -> list[str]:
    body = block(LIBSY_STAGE.read_text(), "impl DecisionSource {")
    names = re.findall(r'Self::\w+ => "([a-z_-]+)"', body)
    assert names, "DecisionSource::as_str arms not found"
    return names


def section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    stop = text.index(end, begin + len(start))
    return text[begin:stop]


def table_keys(text: str) -> set[str]:
    """Keys named in the first column of the markdown tables in `text`."""
    return set(re.findall(r"^\| `([a-z_][a-z0-9_.]*)` \|", text, re.M))


def check_section(text: str) -> None:
    """Every accepted key is named somewhere in the section (a table row, or
    prose for aliases and keys the route rejects), and every table row names an
    accepted key."""
    expected = expected_keys()
    mentioned = {key for key in expected if f"`{key}`" in text}
    assert sorted(expected - mentioned) == [], "keys the code accepts but the page omits"
    unnamed = {key for key in REJECTED_CLASSIFIER_KEYS if f"`{key}`" not in text}
    assert sorted(unnamed) == [], "rejected keys the page does not name"
    unknown = table_keys(text) - expected
    assert sorted(unknown) == [], "keys the tables list but the code does not accept"


def test_derived_keys_include_the_known_shape() -> None:
    keys = expected_keys()
    assert {
        "picker",
        "confidence_threshold",
        "efficient_first_threshold",
        "tool_semantics.observe",
        "handoff_notes.escalation_note",
        "classifier.target",
        "subagents",
    } <= keys


def test_schema_lists_exactly_the_stage_router_keys() -> None:
    check_section(section(SCHEMA.read_text(), "### `stage_router`", "### `auto`"))


def test_guide_knob_reference_lists_exactly_the_stage_router_keys() -> None:
    check_section(section(GUIDE.read_text(), "### Knob reference", "### Minimal route"))


def test_guide_lists_every_decision_source() -> None:
    text = section(GUIDE.read_text(), "### Decision sources", "## When")
    missing = [name for name in decision_sources() if f"`{name}`" not in text]
    assert missing == []

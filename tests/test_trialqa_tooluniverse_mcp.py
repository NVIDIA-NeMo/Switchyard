# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused, zero-network tests for the TrialQA ToolUniverse MCP adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import types

import benchmark.trialqa_tooluniverse_mcp as module


class FakeChild:
    def __init__(self, result: types.CallToolResult | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = result or types.CallToolResult(
            content=[types.TextContent(type="text", text="child result")],
            structuredContent={"child": True},
            isError=False,
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        self.calls.append((name, dict(arguments or {})))
        return self.result


def _call(
    adapter: module.TrialQAToolUniverseAdapter,
    name: str,
    arguments: dict[str, Any],
) -> types.CallToolResult:
    async def run() -> types.CallToolResult:
        return await adapter.call_tool(name, arguments)

    return anyio.run(run)


def _walk_schema(value: object) -> None:
    if isinstance(value, dict):
        assert "anyOf" not in value
        assert "oneOf" not in value
        assert not isinstance(value.get("type"), list)
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _walk_schema(child)
    elif isinstance(value, list):
        for child in value:
            _walk_schema(child)


def _serialized_tools() -> list[dict[str, Any]]:
    return [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in module.advertised_tools()
    ]


def test_advertised_tools_are_exact_closed_read_only_codex_safe_schemas() -> None:
    tools = module.advertised_tools()

    assert [tool.name for tool in tools] == [
        "trialqa_load_active_skill",
        "list_tools",
        "grep_tools",
        "get_tool_info",
        "execute_tool",
        "find_tools",
    ]
    for tool in tools:
        _walk_schema(tool.inputSchema)
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.inputSchema["additionalProperties"] is False

    assert tools[0].inputSchema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    get_info_schema = tools[3].inputSchema
    assert get_info_schema["properties"]["tool_names"] == {
        "type": "array",
        "description": "One or more exact ToolUniverse tool names.",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
    }
    assert get_info_schema["required"] == ["tool_names"]
    execute_schema = tools[4].inputSchema
    assert execute_schema["required"] == ["tool_name", "arguments_json"]
    assert execute_schema["properties"]["tool_name"]["enum"] == list(
        module.ALLOWED_EXECUTION_TOOL_NAMES
    )
    assert execute_schema["properties"]["arguments_json"]["type"] == "string"


def test_skill_arm_does_not_change_advertised_tool_schemas() -> None:
    schemas_before = _serialized_tools()
    baseline = module.TrialQAToolUniverseAdapter(FakeChild())
    treatment = module.TrialQAToolUniverseAdapter(
        FakeChild(),
        skill=module.SkillSnapshot(True, "# TrialQA", "sha256:abc"),
    )

    assert (
        _call(baseline, module.SKILL_TOOL_NAME, {}).structuredContent
        != _call(treatment, module.SKILL_TOOL_NAME, {}).structuredContent
    )
    assert _serialized_tools() == schemas_before


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "list_tools",
            {
                "mode": "names",
                "categories": ["clinical_trials"],
                "limit": 20,
                "offset": 0,
            },
        ),
        (
            "grep_tools",
            {
                "pattern": "ClinicalTrials",
                "field": "name",
                "search_mode": "text",
                "limit": 20,
                "offset": 0,
            },
        ),
        (
            "get_tool_info",
            {
                "tool_names": [
                    "ClinicalTrials_search_studies",
                    "ClinicalTrials_get_study",
                ],
                "detail_level": "full",
            },
        ),
        (
            "find_tools",
            {
                "query": "search clinical trials",
                "categories": ["clinical_trials"],
                "limit": 5,
                "use_advanced_search": False,
                "search_method": "keyword",
            },
        ),
    ],
)
def test_discovery_tools_forward_exact_compact_child_requests(
    name: str, arguments: dict[str, Any]
) -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    result = _call(adapter, name, arguments)

    assert result is child.result
    assert child.calls == [(name, arguments)]


def test_discovery_results_expose_the_full_compact_catalog_without_filtering() -> None:
    child_result = types.CallToolResult(
        content=[types.TextContent(type="text", text="catalog")],
        structuredContent={"tools": ["ClinicalTrials_search_studies", "PubMed_search_articles"]},
        isError=False,
    )
    child = FakeChild(child_result)
    adapter = module.TrialQAToolUniverseAdapter(child)

    result = _call(adapter, "list_tools", {"mode": "names"})

    assert result is child_result
    assert result.structuredContent == child_result.structuredContent
    assert child.calls == [("list_tools", {"mode": "names"})]


def test_find_tools_forces_local_keyword_discovery() -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    _call(
        adapter,
        "find_tools",
        {
            "query": "search clinical trials",
            "use_advanced_search": True,
            "search_method": "auto",
        },
    )

    assert child.calls == [
        (
            "find_tools",
            {
                "query": "search clinical trials",
                "use_advanced_search": False,
                "search_method": "keyword",
            },
        )
    ]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("ClinicalTrials_search_studies", {"query_cond": "breast cancer"}),
        ("ClinicalTrials_get_study", {"nct_id": "NCT04280705"}),
        (
            "get_clinical_trial_eligibility_criteria",
            {"nct_ids": ["NCT04280705"]},
        ),
        (
            "get_clinical_trial_descriptions",
            {"nct_ids": ["NCT04280705"], "description_type": "full"},
        ),
        (
            "get_clinical_trial_status_and_dates",
            {"nct_ids": ["NCT04280705"]},
        ),
        (
            "get_clinical_trial_outcome_measures",
            {"nct_ids": ["NCT04280705"], "outcome_measures": "all"},
        ),
        ("get_clinical_trial_references", {"nct_ids": ["NCT04280705"]}),
        (
            "extract_clinical_trial_outcomes",
            {"nct_ids": ["NCT04280705"], "outcome_measure": "survival"},
        ),
        (
            "extract_clinical_trial_adverse_events",
            {
                "nct_ids": ["NCT04280705"],
                "organ_systems": ["Cardiac Disorders"],
                "adverse_event_type": "serious",
            },
        ),
    ],
)
def test_execute_tool_parses_json_for_each_allowed_read_only_operation(
    tool_name: str, arguments: dict[str, Any]
) -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    result = _call(
        adapter,
        "execute_tool",
        {"tool_name": tool_name, "arguments_json": json.dumps(arguments)},
    )

    assert result is child.result
    assert child.calls == [("execute_tool", {"tool_name": tool_name, "arguments": arguments})]


def test_child_call_tool_result_error_content_and_metadata_are_preserved() -> None:
    child_result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="upstream error text"),
            types.ImageContent(type="image", data="YWJj", mimeType="image/png"),
        ],
        structuredContent={"error": "not found"},
        isError=True,
        _meta={"child_request_id": "request-1"},
    )
    child = FakeChild(child_result)
    adapter = module.TrialQAToolUniverseAdapter(child)

    result = _call(
        adapter,
        "execute_tool",
        {
            "tool_name": "ClinicalTrials_get_study",
            "arguments_json": '{"nct_id":"NCT00000000"}',
        },
    )

    assert result is child_result
    assert result.isError is True
    assert result.content == child_result.content
    assert result.structuredContent == child_result.structuredContent
    assert result.meta == {"child_request_id": "request-1"}


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("unknown", {}, "unknown TrialQA adapter tool"),
        ("get_tool_info", {}, "'tool_names' is a required property"),
        (
            "get_tool_info",
            {"tool_names": "ClinicalTrials_get_study"},
            "is not of type 'array'",
        ),
        ("get_tool_info", {"tool_names": []}, "should be non-empty"),
        (
            "list_tools",
            {"mode": "names", "path": "/tmp/forbidden"},
            "Additional properties are not allowed",
        ),
        (
            "execute_tool",
            {
                "tool_name": "PubMed_search_articles",
                "arguments_json": "{}",
            },
            "is not one of",
        ),
        (
            "execute_tool",
            {"tool_name": "ClinicalTrials_get_study"},
            "'arguments_json' is a required property",
        ),
    ],
)
def test_invalid_or_arbitrary_arguments_are_rejected_before_child_call(
    name: str, arguments: dict[str, Any], message: str
) -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    with pytest.raises(module.TrialQAToolUniverseMCPError, match=message):
        _call(adapter, name, arguments)
    assert child.calls == []


@pytest.mark.parametrize("arguments_json", ["{", "[]", '"text"', "null"])
def test_execute_tool_requires_arguments_json_to_decode_to_an_object(
    arguments_json: str,
) -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    with pytest.raises(module.TrialQAToolUniverseMCPError, match="arguments_json"):
        _call(
            adapter,
            "execute_tool",
            {
                "tool_name": "ClinicalTrials_get_study",
                "arguments_json": arguments_json,
            },
        )
    assert child.calls == []


def test_active_skill_loader_is_local_and_baseline_is_explicitly_unavailable() -> None:
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child)

    result = _call(adapter, "trialqa_load_active_skill", {})

    assert result.isError is False
    assert result.structuredContent == {
        "available": False,
        "content": None,
        "sha256": None,
        "reason": "No active TrialQA skill was configured for this arm.",
    }
    assert child.calls == []


def test_active_skill_snapshot_reads_exact_real_utf8_skill_once(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_text = "---\nname: tooluniverse-trialqa\ndescription: test\n---\n\n# Rules\n"
    skill_path.write_text(skill_text, encoding="utf-8")
    snapshot = module.load_skill_snapshot(skill_path)
    skill_path.write_text("changed after startup", encoding="utf-8")
    child = FakeChild()
    adapter = module.TrialQAToolUniverseAdapter(child, skill=snapshot)

    result = _call(adapter, "trialqa_load_active_skill", {})

    assert result.structuredContent == {
        "available": True,
        "content": skill_text,
        "sha256": "sha256:" + __import__("hashlib").sha256(skill_text.encode("utf-8")).hexdigest(),
    }
    assert child.calls == []


def test_active_skill_path_rejects_wrong_name_symlink_and_non_utf8(tmp_path: Path) -> None:
    wrong = tmp_path / "other.md"
    wrong.write_text("content", encoding="utf-8")
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="name SKILL.md"):
        module.load_skill_snapshot(wrong)

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "SKILL.md"
    real.write_text("content", encoding="utf-8")
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="symlink"):
        module.load_skill_snapshot(linked_dir / "SKILL.md")

    invalid = tmp_path / "SKILL.md"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="UTF-8"):
        module.load_skill_snapshot(invalid)


def _runtime(tmp_path: Path, *, version: str = module.TOOLUNIVERSE_VERSION) -> tuple[Path, Path]:
    venv = tmp_path / "tooluniverse-venv"
    binary = venv / "bin" / module.TOOLUNIVERSE_BINARY_NAME
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    metadata = (
        venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / f"tooluniverse-{version}.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Name: tooluniverse\nVersion: {version}\n", encoding="utf-8")
    return venv, binary


def test_runtime_requires_pinned_venv_and_builds_exact_compact_child_argv(
    tmp_path: Path,
) -> None:
    venv, binary = _runtime(tmp_path)

    runtime = module.validate_runtime(binary, sys_prefix=venv)
    params = runtime.server_parameters()

    assert runtime.venv == venv.absolute()
    assert params.command == str(binary.absolute())
    assert params.args == ["--compact-mode"]
    assert params.env == {
        "PYTHONIOENCODING": "utf-8",
        "FASTMCP_CHECK_FOR_UPDATES": "off",
        "FASTMCP_SHOW_SERVER_BANNER": "false",
    }

    with pytest.raises(module.TrialQAToolUniverseMCPError, match="sys.prefix"):
        module.validate_runtime(binary, sys_prefix=tmp_path / "other-venv")


def test_runtime_rejects_wrong_version_binary_name_and_symlink(tmp_path: Path) -> None:
    venv, binary = _runtime(tmp_path / "wrong-version", version="1.1.12")
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="pinned"):
        module.validate_runtime(binary, sys_prefix=venv)

    wrong_name = binary.with_name("other")
    wrong_name.write_text("#!/bin/sh\n", encoding="utf-8")
    wrong_name.chmod(0o755)
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="named"):
        module.validate_runtime(wrong_name, sys_prefix=venv)

    good_venv, good_binary = _runtime(tmp_path / "good")
    link = good_binary.with_name("tooluniverse-smcp-stdio-link")
    link.symlink_to(good_binary)
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="named"):
        module.validate_runtime(link, sys_prefix=good_venv)


def _child_tool(name: str) -> types.Tool:
    return types.Tool(name=name, inputSchema={"type": "object", "properties": {}})


def test_child_tool_set_must_be_exact_five_compact_tools() -> None:
    tools = [_child_tool(name) for name in module.COMPACT_TOOL_NAMES]

    module.validate_child_tools(tools)

    with pytest.raises(module.TrialQAToolUniverseMCPError, match="missing"):
        module.validate_child_tools(tools[1:])
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="unexpected"):
        module.validate_child_tools([*tools, _child_tool("arbitrary_tool")])
    with pytest.raises(module.TrialQAToolUniverseMCPError, match="duplicate"):
        module.validate_child_tools([*tools, _child_tool(module.COMPACT_TOOL_NAMES[0])])


def test_describe_tools_is_complete_deterministic_and_never_spawns_child(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("describe-tools must not validate or spawn a child")

    monkeypatch.setattr(module, "validate_runtime", forbidden)
    monkeypatch.setattr(module.anyio, "run", forbidden)

    assert module.main(["--describe-tools"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    document = json.loads(captured.out)
    assert document == module.describe_tools_document()
    assert document["schema_version"] == "switchyard.trialqa_tooluniverse_mcp.v2"
    assert document["adapter"] == {
        "name": module.ADAPTER_NAME,
        "version": "2.0.0",
    }
    assert document["tooluniverse"] == {
        "version": module.TOOLUNIVERSE_VERSION,
        "binary_name": module.TOOLUNIVERSE_BINARY_NAME,
        "mode": "compact",
        "child_args": ["--compact-mode"],
        "compact_child_tools": list(module.COMPACT_TOOL_NAMES),
        "allowed_execution_tools": list(module.ALLOWED_EXECUTION_TOOL_NAMES),
    }
    assert len(document["tools"]) == 6
    for tool in document["tools"]:
        _walk_schema(tool["inputSchema"])


def test_server_registers_low_level_list_and_call_handlers() -> None:
    adapter = module.TrialQAToolUniverseAdapter(FakeChild())
    server = module.build_server(adapter)

    assert types.ListToolsRequest in server.request_handlers
    assert types.CallToolRequest in server.request_handlers
    assert server.name == module.ADAPTER_NAME


def test_module_does_not_embed_network_or_shell_execution_paths() -> None:
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "http://" not in text
    assert "https://" not in text
    assert "subprocess" not in text
    assert "shell=True" not in text
    assert os.path.basename(module.__file__) == "trialqa_tooluniverse_mcp.py"

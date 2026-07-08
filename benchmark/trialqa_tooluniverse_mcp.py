# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-safe, read-only TrialQA adapter for ToolUniverse's compact stdio MCP.

Run this module with the Python interpreter from the pinned ToolUniverse venv.
It forwards compact discovery operations to exactly one child process:

``tooluniverse-smcp-stdio --compact-mode``

ToolUniverse's compact schemas contain unions that some Codex clients reject.
The adapter re-advertises the same five-tool workflow with closed input schemas
that contain no ``anyOf``, ``oneOf``, or array-valued ``type``.  Its
``execute_tool`` accepts JSON text and permits only the nine pinned read-only
ClinicalTrials operations used by TrialQA.  Child
:class:`mcp.types.CallToolResult` objects are returned unchanged, including
their content, structured content, metadata, and error flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import anyio
import jsonschema
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server

SCHEMA_VERSION = "switchyard.trialqa_tooluniverse_mcp.v2"
ADAPTER_NAME = "switchyard-trialqa-tooluniverse"
ADAPTER_VERSION = "2.0.0"
TOOLUNIVERSE_VERSION = "1.1.11"
TOOLUNIVERSE_BINARY_NAME = "tooluniverse-smcp-stdio"
SKILL_TOOL_NAME = "trialqa_load_active_skill"

COMPACT_TOOL_NAMES = (
    "list_tools",
    "grep_tools",
    "get_tool_info",
    "execute_tool",
    "find_tools",
)
ALLOWED_EXECUTION_TOOL_NAMES = (
    "ClinicalTrials_search_studies",
    "ClinicalTrials_get_study",
    "get_clinical_trial_eligibility_criteria",
    "get_clinical_trial_descriptions",
    "get_clinical_trial_status_and_dates",
    "get_clinical_trial_outcome_measures",
    "get_clinical_trial_references",
    "extract_clinical_trial_outcomes",
    "extract_clinical_trial_adverse_events",
)
CHILD_ARGS = ("--compact-mode",)

JsonObject = dict[str, Any]
ArgumentMapper = Callable[[Mapping[str, Any]], JsonObject]


class TrialQAToolUniverseMCPError(RuntimeError):
    """Raised for an unsafe runtime or invalid adapter request."""


@dataclass(frozen=True)
class Runtime:
    """Validated pinned ToolUniverse process runtime."""

    tooluniverse_bin: Path
    venv: Path
    child_args: tuple[str, ...] = CHILD_ARGS

    def server_parameters(self) -> StdioServerParameters:
        """Return the exact child command contract."""

        return StdioServerParameters(
            command=str(self.tooluniverse_bin),
            args=list(self.child_args),
            env={
                "PYTHONIOENCODING": "utf-8",
                "FASTMCP_CHECK_FOR_UPDATES": "off",
                "FASTMCP_SHOW_SERVER_BANNER": "false",
            },
        )


@dataclass(frozen=True)
class SkillSnapshot:
    """Immutable in-memory copy of the optional active skill."""

    available: bool
    content: str | None
    sha256: str | None

    def document(self) -> JsonObject:
        if not self.available:
            return {
                "available": False,
                "content": None,
                "sha256": None,
                "reason": "No active TrialQA skill was configured for this arm.",
            }
        return {
            "available": True,
            "content": self.content,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ToolSpec:
    """One stable advertised tool and its optional child mapping."""

    name: str
    description: str
    input_schema: JsonObject
    child_name: str | None
    mapper: ArgumentMapper

    def mcp_tool(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            annotations=types.ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=self.child_name is not None,
            ),
        )


def _property(
    value_type: str,
    description: str,
    *,
    default: object | None = None,
    enum: Sequence[str] | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> JsonObject:
    value: JsonObject = {"type": value_type, "description": description}
    if value_type == "string":
        value["minLength"] = 1
    if default is not None:
        value["default"] = default
    if enum is not None:
        value["enum"] = list(enum)
    if minimum is not None:
        value["minimum"] = minimum
    if maximum is not None:
        value["maximum"] = maximum
    return value


def _string_array(description: str) -> JsonObject:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
    }


def _closed_schema(
    properties: Mapping[str, JsonObject],
    *,
    required: Sequence[str] = (),
) -> JsonObject:
    schema: JsonObject = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    _assert_schema_contract(schema)
    return schema


def _assert_schema_contract(value: object, path: str = "inputSchema") -> None:
    """Reject constructs that are incompatible with the target Codex client."""

    if isinstance(value, dict):
        for banned in ("anyOf", "oneOf"):
            if banned in value:
                raise TrialQAToolUniverseMCPError(f"{path} contains banned {banned}")
        raw_type = value.get("type")
        if isinstance(raw_type, list):
            raise TrialQAToolUniverseMCPError(f"{path}.type cannot be an array")
        if raw_type == "object" and value.get("additionalProperties") is not False:
            raise TrialQAToolUniverseMCPError(
                f"{path} object schemas must set additionalProperties=false"
            )
        for key, child in value.items():
            _assert_schema_contract(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_schema_contract(child, f"{path}[{index}]")


def _identity(arguments: Mapping[str, Any]) -> JsonObject:
    return dict(arguments)


def _keyword_find_arguments(arguments: Mapping[str, Any]) -> JsonObject:
    """Keep compact discovery local and deterministic; never invoke a finder LLM."""

    normalized = dict(arguments)
    normalized["use_advanced_search"] = False
    normalized["search_method"] = "keyword"
    return normalized


def _execute_arguments(arguments: Mapping[str, Any]) -> JsonObject:
    tool_name = arguments["tool_name"]
    if tool_name not in ALLOWED_EXECUTION_TOOL_NAMES:
        raise TrialQAToolUniverseMCPError(
            f"execute_tool cannot run non-TrialQA operation: {tool_name!r}"
        )
    raw_arguments = arguments["arguments_json"]
    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise TrialQAToolUniverseMCPError(
            f"invalid arguments_json for execute_tool: {exc.msg}"
        ) from exc
    if not isinstance(decoded, dict):
        raise TrialQAToolUniverseMCPError(
            "invalid arguments_json for execute_tool: JSON value must be an object"
        )
    return {"tool_name": tool_name, "arguments": decoded}


TOOL_SPECS = (
    ToolSpec(
        name=SKILL_TOOL_NAME,
        description=(
            "Load the arm's preconfigured immutable TrialQA skill. Returns available=false "
            "for baseline and the exact SKILL.md content for treatment."
        ),
        input_schema=_closed_schema({}),
        child_name=None,
        mapper=_identity,
    ),
    ToolSpec(
        name="list_tools",
        description=(
            "List the compact ToolUniverse catalog. Start with mode='names' or "
            "mode='categories', then inspect selected tools with get_tool_info."
        ),
        input_schema=_closed_schema(
            {
                "mode": _property(
                    "string",
                    "Output mode.",
                    default="names",
                    enum=("names", "basic", "categories", "by_category", "summary", "custom"),
                ),
                "categories": _string_array("Optional tool categories to include."),
                "fields": _string_array("Fields to include when mode='custom'."),
                "group_by_category": _property(
                    "boolean", "Group results by category.", default=False
                ),
                "brief": _property("boolean", "Truncate descriptions.", default=False),
                "limit": _property("integer", "Maximum tools to return.", minimum=1),
                "offset": _property("integer", "Number of tools to skip.", default=0, minimum=0),
            }
        ),
        child_name="list_tools",
        mapper=_identity,
    ),
    ToolSpec(
        name="grep_tools",
        description="Search the compact ToolUniverse catalog by text or regular expression.",
        input_schema=_closed_schema(
            {
                "pattern": _property("string", "Text or regular-expression search pattern."),
                "field": _property(
                    "string",
                    "Catalog field to search.",
                    default="name",
                    enum=("name", "description", "type", "category"),
                ),
                "search_mode": _property(
                    "string",
                    "Use text matching or regular expressions.",
                    default="text",
                    enum=("text", "regex"),
                ),
                "limit": _property("integer", "Maximum matches to return.", default=100, minimum=1),
                "offset": _property("integer", "Number of matches to skip.", default=0, minimum=0),
                "categories": _string_array("Optional tool categories to include."),
            },
            required=("pattern",),
        ),
        child_name="grep_tools",
        mapper=_identity,
    ),
    ToolSpec(
        name="get_tool_info",
        description=(
            "Get descriptions or full definitions for one or more tools discovered in the "
            "compact ToolUniverse catalog."
        ),
        input_schema=_closed_schema(
            {
                "tool_names": _string_array("One or more exact ToolUniverse tool names."),
                "detail_level": _property(
                    "string",
                    "Return descriptions or complete definitions.",
                    default="full",
                    enum=("description", "full"),
                ),
            },
            required=("tool_names",),
        ),
        child_name="get_tool_info",
        mapper=_identity,
    ),
    ToolSpec(
        name="execute_tool",
        description=(
            "Execute one pinned read-only TrialQA ClinicalTrials operation. Pass its arguments "
            "as a JSON object encoded in arguments_json."
        ),
        input_schema=_closed_schema(
            {
                "tool_name": _property(
                    "string",
                    "Exact read-only ClinicalTrials operation to execute.",
                    enum=ALLOWED_EXECUTION_TOOL_NAMES,
                ),
                "arguments_json": _property(
                    "string", "JSON object containing the selected operation's arguments."
                ),
            },
            required=("tool_name", "arguments_json"),
        ),
        child_name="execute_tool",
        mapper=_execute_arguments,
    ),
    ToolSpec(
        name="find_tools",
        description=(
            "Find ToolUniverse tools from a capability query using deterministic keyword "
            "discovery; this adapter never invokes an internal finder model."
        ),
        input_schema=_closed_schema(
            {
                "query": _property("string", "Capability description to search for."),
                "categories": _string_array("Optional tool categories to include."),
                "limit": _property("integer", "Maximum matches to return.", default=10, minimum=1),
                "use_advanced_search": _property(
                    "boolean", "Allow advanced search implementations.", default=True
                ),
                "search_method": _property(
                    "string",
                    "Search implementation preference.",
                    default="keyword",
                    enum=("auto", "keyword"),
                ),
            },
            required=("query",),
        ),
        child_name="find_tools",
        mapper=_keyword_find_arguments,
    ),
)
TOOL_SPEC_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def advertised_tools() -> list[types.Tool]:
    """Return the deterministic public tool definitions."""

    return [spec.mcp_tool() for spec in TOOL_SPECS]


def describe_tools_document() -> JsonObject:
    """Return a deterministic, no-child schema attestation document."""

    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
        "tooluniverse": {
            "version": TOOLUNIVERSE_VERSION,
            "binary_name": TOOLUNIVERSE_BINARY_NAME,
            "mode": "compact",
            "child_args": list(CHILD_ARGS),
            "compact_child_tools": list(COMPACT_TOOL_NAMES),
            "allowed_execution_tools": list(ALLOWED_EXECUTION_TOOL_NAMES),
        },
        "tools": [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in advertised_tools()
        ],
    }


def _json_result(document: Mapping[str, Any]) -> types.CallToolResult:
    payload = dict(document)
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
        ],
        structuredContent=payload,
        isError=False,
    )


class TrialQAToolUniverseAdapter:
    """Validate stable requests and forward them to exact child tools."""

    def __init__(self, child: Any, *, skill: SkillSnapshot | None = None) -> None:
        self._child = child
        self._skill = skill or SkillSnapshot(False, None, None)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> types.CallToolResult:
        spec = TOOL_SPEC_BY_NAME.get(name)
        if spec is None:
            raise TrialQAToolUniverseMCPError(f"unknown TrialQA adapter tool: {name}")
        normalized = dict(arguments or {})
        try:
            jsonschema.validate(instance=normalized, schema=spec.input_schema)
        except jsonschema.ValidationError as exc:
            raise TrialQAToolUniverseMCPError(
                f"invalid arguments for {name}: {exc.message}"
            ) from exc
        if spec.child_name is None:
            return _json_result(self._skill.document())
        child_arguments = spec.mapper(normalized)
        result: object = await self._child.call_tool(spec.child_name, child_arguments)
        if not isinstance(result, types.CallToolResult):
            raise TrialQAToolUniverseMCPError(
                f"child tool {spec.child_name!r} returned an invalid MCP result"
            )
        return result


def build_server(adapter: TrialQAToolUniverseAdapter) -> Server[Any]:
    """Build the low-level MCP server around an initialized adapter."""

    server: Server[Any] = Server(
        ADAPTER_NAME,
        version=ADAPTER_VERSION,
        instructions=(
            "Use trialqa_load_active_skill first. Discover tools with list_tools, grep_tools, "
            "get_tool_info, or find_tools, then call the read-only execute_tool wrapper with "
            "arguments_json."
        ),
    )

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        return advertised_tools()

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await adapter.call_tool(name, arguments)

    return server


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise TrialQAToolUniverseMCPError(f"missing path component: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TrialQAToolUniverseMCPError(f"symlink path components are forbidden: {current}")


def _read_utf8_regular_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrialQAToolUniverseMCPError(f"could not safely open --skill-path: {path}") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TrialQAToolUniverseMCPError(
                "--skill-path must be a single-link regular UTF-8 file"
            )
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise TrialQAToolUniverseMCPError("--skill-path changed while it was read")
    finally:
        os.close(descriptor)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrialQAToolUniverseMCPError("--skill-path is not valid UTF-8") from exc


def load_skill_snapshot(skill_path: Path | None) -> SkillSnapshot:
    """Read only the preconfigured real ``SKILL.md``; never accept a call-time path."""

    if skill_path is None:
        return SkillSnapshot(False, None, None)
    path = skill_path.expanduser().absolute()
    if path.name != "SKILL.md":
        raise TrialQAToolUniverseMCPError("--skill-path must name SKILL.md exactly")
    _reject_symlink_components(path)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise TrialQAToolUniverseMCPError("--skill-path must be a single-link regular UTF-8 file")
    content = _read_utf8_regular_no_follow(path)
    if not content.strip():
        raise TrialQAToolUniverseMCPError("--skill-path cannot be empty")
    return SkillSnapshot(
        True,
        content,
        f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}",
    )


def validate_runtime(
    tooluniverse_bin: Path,
    *,
    sys_prefix: Path | None = None,
) -> Runtime:
    """Require the exact binary, package version, and current venv interpreter."""

    requested = tooluniverse_bin.expanduser().absolute()
    if requested.name != TOOLUNIVERSE_BINARY_NAME:
        raise TrialQAToolUniverseMCPError(
            f"ToolUniverse binary must be named {TOOLUNIVERSE_BINARY_NAME!r}"
        )
    if requested.is_symlink() or not requested.is_file() or not os.access(requested, os.X_OK):
        raise TrialQAToolUniverseMCPError(
            f"ToolUniverse binary must be a real executable file: {requested}"
        )
    venv = requested.parent.parent.absolute()
    prefix = (sys_prefix or Path(sys.prefix)).expanduser().absolute()
    if prefix != venv:
        raise TrialQAToolUniverseMCPError(
            "adapter must run with the Python interpreter from the ToolUniverse venv: "
            f"expected sys.prefix={venv}, got {prefix}"
        )
    metadata_paths = sorted(
        venv.glob("lib/python*/site-packages/tooluniverse-*.dist-info/METADATA")
    )
    if len(metadata_paths) != 1:
        raise TrialQAToolUniverseMCPError(
            f"could not uniquely attest ToolUniverse metadata under {venv}"
        )
    fields: dict[str, str] = {}
    for line in metadata_paths[0].read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Name", "Version"} and key not in fields:
            fields[key] = value.strip()
    if (
        fields.get("Name", "").lower() != "tooluniverse"
        or fields.get("Version") != TOOLUNIVERSE_VERSION
    ):
        raise TrialQAToolUniverseMCPError(f"ToolUniverse must be pinned to {TOOLUNIVERSE_VERSION}")
    try:
        installed_mcp = version("mcp")
    except PackageNotFoundError as exc:  # pragma: no cover - startup-only defense.
        raise TrialQAToolUniverseMCPError("ToolUniverse venv has no mcp package") from exc
    if not installed_mcp:
        raise TrialQAToolUniverseMCPError("ToolUniverse venv has an invalid mcp package")
    return Runtime(tooluniverse_bin=requested, venv=venv)


def validate_child_tools(tools: Sequence[types.Tool]) -> None:
    """Require exactly the five compact ToolUniverse MCP tools."""

    by_name = {tool.name: tool for tool in tools}
    if len(by_name) != len(tools):
        raise TrialQAToolUniverseMCPError("pinned ToolUniverse child returned duplicate tools")
    required = set(COMPACT_TOOL_NAMES)
    missing = required.difference(by_name)
    unexpected = set(by_name).difference(required)
    if missing or unexpected:
        raise TrialQAToolUniverseMCPError(
            "pinned ToolUniverse compact tool set differs from the expected set: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )


async def run_adapter(runtime: Runtime, skill: SkillSnapshot) -> None:
    """Spawn the exact child and serve the adapter over this process's stdio."""

    async with stdio_client(runtime.server_parameters()) as (child_reader, child_writer):
        async with ClientSession(child_reader, child_writer) as child:
            await child.initialize()
            listed = await child.list_tools()
            validate_child_tools(listed.tools)
            adapter = TrialQAToolUniverseAdapter(child, skill=skill)
            server = build_server(adapter)
            async with stdio_server() as (reader, writer):
                await server.run(
                    reader,
                    writer,
                    server.create_initialization_options(),
                    raise_exceptions=True,
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tooluniverse-bin", type=Path)
    parser.add_argument("--skill-path", type=Path)
    parser.add_argument(
        "--describe-tools",
        action="store_true",
        help="print deterministic adapter schemas and exit without spawning ToolUniverse",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.describe_tools:
        print(
            json.dumps(
                describe_tools_document(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.tooluniverse_bin is None:
        print("trialqa_tooluniverse_mcp: error: --tooluniverse-bin is required", file=sys.stderr)
        return 2
    try:
        runtime = validate_runtime(args.tooluniverse_bin)
        skill = load_skill_snapshot(args.skill_path)
        anyio.run(run_adapter, runtime, skill)
    except TrialQAToolUniverseMCPError as exc:
        print(f"trialqa_tooluniverse_mcp: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

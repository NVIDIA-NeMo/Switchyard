# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, version-gated access to vLLM's tool-call and reasoning parsers.

The token-injection backend generates through vLLM's ``/v1/completions``
endpoint, which returns raw text — the chat endpoint's tool-call and
reasoning parsing never runs. This module re-runs that parsing in the proxy
using vLLM's own parser registries, so the parsed output matches what the
chat endpoint would have produced for the same serving version.

vLLM is an optional runtime dependency: it is imported on first use only, so
capture-only deployments and environments without vLLM are unaffected. The
import paths are gated on the installed vLLM version (verified boundaries:
tool parsers moved at 0.13.0, the OpenAI protocol module split at 0.15.0).
Patched vLLM builds may not match the upstream layout for their reported
version, so a failed import names the expected module path explicitly.
"""

from __future__ import annotations

import json
import logging
import uuid as uuid_lib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]


class VllmParserError(RuntimeError):
    """vLLM's parsers are unavailable or failed for this configuration."""


@dataclass
class ParsedGeneration:
    """Structured view of one raw completions-endpoint generation."""

    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[JsonObject] = field(default_factory=list)


@dataclass
class _ParserRuntime:
    """Resolved vLLM parser instances for one (model, config) pair."""

    tool_parser: Any | None
    reasoning_parser: Any | None
    chat_request_cls: Any


_runtime_cache: dict[tuple[str, str | None, str | None], _ParserRuntime] = {}


def parse_generation(
    text: str,
    *,
    model: str,
    tools: list[JsonObject] | None,
    tool_parser: str | None,
    reasoning_parser: str | None,
) -> ParsedGeneration:
    """Split raw generated text into reasoning, content, and structured tool calls.

    Parser names mirror the flags the vLLM server was launched with
    (``--tool-call-parser``, ``--reasoning-parser``). A ``None`` parser name
    skips that stage: content passes through unparsed. Raises
    :class:`VllmParserError` when a named parser cannot be loaded.
    """
    if tool_parser is None and reasoning_parser is None:
        return ParsedGeneration(content=text)

    runtime = _resolve_runtime(model, tool_parser, reasoning_parser)
    request = runtime.chat_request_cls(
        messages=[{"role": "user", "content": ""}],
        model=model,
        tools=tools or None,
    )

    reasoning_content: str | None = None
    content: str | None = text
    if runtime.reasoning_parser is not None:
        # Renamed upstream: extract_reasoning_content (<= 0.11.x) -> extract_reasoning.
        extract = getattr(runtime.reasoning_parser, "extract_reasoning_content", None)
        if extract is None:
            extract = runtime.reasoning_parser.extract_reasoning
        reasoning_content, content = extract(text, request=request)

    tool_calls: list[JsonObject] = []
    if runtime.tool_parser is not None and content:
        extracted = runtime.tool_parser.extract_tool_calls(content, request=request)
        if getattr(extracted, "tools_called", False):
            tool_calls = [
                _tool_call_to_dict(call) for call in getattr(extracted, "tool_calls", [])
            ]
            content = getattr(extracted, "content", None)
    return ParsedGeneration(
        content=content, reasoning_content=reasoning_content, tool_calls=tool_calls
    )


def parse_guided_tool_array(text: str) -> list[JsonObject]:
    """Parse a ``tool_choice: required``/named guided-decoding generation.

    Under guided decoding the generation is the schema-constrained JSON array
    of ``{name, parameters}`` objects itself (see :func:`tool_choice_json_schema`)
    — the tool parser never applies. Mirrors vLLM's chat-endpoint behavior.
    """
    calls = json.loads(text)
    if not isinstance(calls, list):
        raise VllmParserError("guided tool generation is not a JSON array")
    tool_calls: list[JsonObject] = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise VllmParserError("guided tool generation entry is not {name, parameters}")
        tool_calls.append({
            "id": f"chatcmpl-tool-{uuid_lib.uuid4().hex}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("parameters") or {}),
            },
        })
    return tool_calls


def tool_choice_json_schema(tools: list[JsonObject], tool_choice: object) -> JsonObject | None:
    """The guided-decoding JSON schema for ``required``/named tool choice, else ``None``.

    Ports vLLM's ``ChatCompletionRequest._get_json_schema_from_tool`` so the
    completions-endpoint request constrains generation exactly as the chat
    endpoint would. ``auto``/``none``/unset need no schema.
    """
    if not tools or tool_choice in (None, "auto", "none"):
        return None

    if isinstance(tool_choice, dict):
        name = ((tool_choice.get("function") or {}) or {}).get("name")
        by_name = {
            (tool.get("function") or {}).get("name"): (tool.get("function") or {})
            for tool in tools
        }
        if name not in by_name:
            raise VllmParserError(f"named tool {name!r} not present in tools")
        return by_name[name].get("parameters") or {"type": "object", "properties": {}}

    if tool_choice != "required":
        return None

    def tool_schema(function: JsonObject) -> JsonObject:
        return {
            "properties": {
                "name": {"type": "string", "enum": [function.get("name")]},
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
            },
            "required": ["name", "parameters"],
        }

    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": [tool_schema(tool.get("function") or {}) for tool in tools],
        },
    }


def _resolve_runtime(
    model: str, tool_parser: str | None, reasoning_parser: str | None
) -> _ParserRuntime:
    key = (model, tool_parser, reasoning_parser)
    cached = _runtime_cache.get(key)
    if cached is not None:
        return cached

    tool_parser_cls, reasoning_parser_cls, chat_request_cls = _import_vllm_parsers(
        tool_parser, reasoning_parser
    )
    tokenizer = _load_tokenizer(model)
    runtime = _ParserRuntime(
        tool_parser=tool_parser_cls(tokenizer) if tool_parser_cls is not None else None,
        reasoning_parser=(
            reasoning_parser_cls(tokenizer) if reasoning_parser_cls is not None else None
        ),
        chat_request_cls=chat_request_cls,
    )
    _runtime_cache[key] = runtime
    return runtime


def _import_vllm_parsers(
    tool_parser: str | None, reasoning_parser: str | None
) -> tuple[Any | None, Any | None, Any]:
    """Import parser classes from the installed vLLM, gated on its version.

    Verified layout boundaries: ``vllm.tool_parsers`` exists from 0.13.0
    (``vllm.entrypoints.openai.tool_parsers`` before), ``vllm.reasoning``
    never moved, and the monolithic ``vllm.entrypoints.openai.protocol``
    split into per-endpoint packages at 0.15.0.
    """
    vllm_version = _installed_vllm_version()

    if vllm_version >= (0, 13, 0):
        tool_parsers_module = "vllm.tool_parsers"
    else:
        tool_parsers_module = "vllm.entrypoints.openai.tool_parsers"
    if vllm_version >= (0, 15, 0):
        protocol_module = "vllm.entrypoints.openai.chat_completion.protocol"
    else:
        protocol_module = "vllm.entrypoints.openai.protocol"

    tool_parser_cls: Any | None = None
    if tool_parser is not None:
        manager = _import_module(tool_parsers_module, vllm_version).ToolParserManager
        tool_parser_cls = manager.get_tool_parser(tool_parser)
    reasoning_parser_cls: Any | None = None
    if reasoning_parser is not None:
        manager = _import_module("vllm.reasoning", vllm_version).ReasoningParserManager
        reasoning_parser_cls = manager.get_reasoning_parser(reasoning_parser)
    chat_request_cls = _import_module(protocol_module, vllm_version).ChatCompletionRequest
    return tool_parser_cls, reasoning_parser_cls, chat_request_cls


def _import_module(name: str, vllm_version: tuple[int, ...]) -> Any:
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        # Patched vLLM builds may not match the upstream layout for their
        # reported version — name the expected path so the mismatch is obvious.
        raise VllmParserError(
            f"cannot import {name!r} from the installed vLLM "
            f"(version {'.'.join(map(str, vllm_version))}); a patched build may "
            f"not match the upstream module layout for its version"
        ) from exc


def _installed_vllm_version() -> tuple[int, ...]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("vllm")
    except PackageNotFoundError as exc:
        raise VllmParserError(
            "vLLM is not installed; token injection with configured parsers "
            "requires the vllm package at runtime"
        ) from exc
    parts: list[int] = []
    for piece in raw.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _load_tokenizer(model: str) -> Any:
    """Fetch the model's tokenizer once; parser constructors require it."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise VllmParserError(
            "transformers is required to construct vLLM parsers (tokenizer)"
        ) from exc
    return AutoTokenizer.from_pretrained(model)


def _tool_call_to_dict(call: Any) -> JsonObject:
    """Normalize a vLLM ``ToolCall`` (pydantic model or dict) to a plain dict."""
    if isinstance(call, dict):
        return call
    dump = getattr(call, "model_dump", None)
    if callable(dump):
        result = dump()
        if isinstance(result, dict):
            return result
    raise VllmParserError(f"unsupported tool call shape: {type(call).__name__}")

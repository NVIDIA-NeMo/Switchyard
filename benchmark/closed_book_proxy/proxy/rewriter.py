# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-book egress gate and hosted-tool stripper for Harbor task containers."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mitmproxy import http

log = logging.getLogger(__name__)

HOSTED_TOOL_TYPES = {
    "browser",
    "computer",
    "mcp",
    "web_fetch",
    "web_search",
    "web_search_preview",
}
HOSTED_TOOL_PATTERNS = (
    re.compile(r"^web_search_20\d{6}$"),
    re.compile(r"^web_fetch_20\d{6}$"),
    re.compile(r"^code_execution_20\d{6}$"),
)
HOSTED_TOOL_KEYS = {
    "codeExecution",
    "googleSearch",
    "googleSearchRetrieval",
    "urlContext",
    "webSearch",
    "web_fetch",
    "web_search",
}
STRIP_PATH_SUFFIXES = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/responses",
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in {"", "0", "false", "no", "off"}


def _normalized_host(value: str | None) -> str:
    return (value or "").split(":", 1)[0].strip("[]").lower()


def _normalized_path(value: str) -> str:
    return value.split("?", 1)[0]


def _is_llm_request_path(value: str) -> bool:
    path = _normalized_path(value)
    return any(path.endswith(suffix) for suffix in STRIP_PATH_SUFFIXES)


class ClosedBookProxy:
    """Mitmproxy addon enforcing an explicit host allowlist."""

    def __init__(self) -> None:
        self.closed_book = _truthy(os.environ.get("CLOSED_BOOK_MODE", "1"))
        self.allowlist_path = Path(
            os.environ.get("SWITCHYARD_PROXY_ALLOWLIST", "/etc/proxy-public/allowed_domains.txt")
        )
        self.strip_log = Path(
            os.environ.get("SWITCHYARD_PROXY_STRIP_LOG", "/etc/proxy-public/strip.jsonl")
        )
        self.request_map = Path(
            os.environ.get(
                "SWITCHYARD_PROXY_REQUEST_MAP", "/etc/proxy-public/request_map.jsonl"
            )
        )
        self._request_map_lock = threading.Lock()
        self.allowed_hosts = self._load_allowed_hosts()

    def _load_allowed_hosts(self) -> set[str]:
        hosts: set[str] = set()
        try:
            for raw in self.allowlist_path.read_text().splitlines():
                value = raw.split("#", 1)[0].strip().lower()
                if value:
                    hosts.add(value)
        except OSError:
            pass
        return hosts

    def _allowed(self, host: str) -> bool:
        host = _normalized_host(host)
        if not host:
            return True
        if host in {"localhost", "127.0.0.1", "::1", "proxy"}:
            return True
        if host in self.allowed_hosts:
            return True
        return any(host.endswith(f".{allowed}") for allowed in self.allowed_hosts)

    def _deny_if_needed(self, flow: http.HTTPFlow) -> bool:
        if not self.closed_book:
            return False

        host = _normalized_host(flow.request.pretty_host or flow.request.host)
        if self._allowed(host):
            return False

        payload = {
            "error": "closed-book proxy denied outbound host",
            "host": host,
            "url": flow.request.pretty_url,
        }
        flow.response = http.Response.make(
            403,
            json.dumps(payload, sort_keys=True).encode(),
            {"content-type": "application/json"},
        )
        return True

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        if self._deny_if_needed(flow):
            return
        if _is_llm_request_path(flow.request.path):
            flow.request.headers["x-request-id"] = uuid.uuid4().hex

    def http_connect(self, flow: http.HTTPFlow) -> None:
        self._deny_if_needed(flow)

    def request(self, flow: http.HTTPFlow) -> None:
        if self._deny_if_needed(flow):
            return
        is_llm_request = _is_llm_request_path(flow.request.path)
        if is_llm_request:
            self._record_request(flow)
        if not self.closed_book or not is_llm_request:
            return
        if "application/json" not in flow.request.headers.get("content-type", ""):
            return
        try:
            body = json.loads(flow.request.get_text(strict=False) or "{}")
        except json.JSONDecodeError:
            return
        if not isinstance(body, dict):
            return

        removed = _strip_hosted_tools(body)
        if not removed:
            return

        flow.request.set_text(json.dumps(body, separators=(",", ":"), sort_keys=False))
        self._log_strip(flow, removed)

    def _record_request(self, flow: http.HTTPFlow) -> None:
        request_id = flow.request.headers.get("x-request-id") or uuid.uuid4().hex
        flow.request.headers["x-request-id"] = request_id

        with self._request_map_lock:
            try:
                with self.request_map.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"request_id": request_id}) + "\n")
            except OSError:
                log.exception("Failed to append proxy request map to %s", self.request_map)

    def _log_strip(self, flow: http.HTTPFlow, removed: list[str]) -> None:
        record = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": _normalized_host(flow.request.pretty_host or flow.request.host),
            "path": flow.request.path,
            "removed": removed,
        }
        try:
            with self.strip_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            return


def _is_hosted_tool(tool: Any) -> tuple[bool, str | None]:
    if isinstance(tool, str):
        tool_type = tool
    elif isinstance(tool, dict):
        if any(key in tool for key in HOSTED_TOOL_KEYS):
            return True, next(key for key in HOSTED_TOOL_KEYS if key in tool)
        tool_type = str(tool.get("type") or tool.get("name") or "")
    else:
        return False, None

    if tool_type in HOSTED_TOOL_TYPES:
        return True, tool_type
    if any(pattern.match(tool_type) for pattern in HOSTED_TOOL_PATTERNS):
        return True, tool_type
    return False, None


def _strip_tool_list(body: dict[str, Any], key: str) -> list[str]:
    tools = body.get(key)
    if not isinstance(tools, list):
        return []

    kept: list[Any] = []
    removed: list[str] = []
    for tool in tools:
        should_strip, label = _is_hosted_tool(tool)
        if should_strip:
            removed.append(label or key)
        else:
            kept.append(tool)
    body[key] = kept
    return removed


def _strip_hosted_tools(body: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    removed.extend(_strip_tool_list(body, "tools"))
    removed.extend(_strip_tool_list(body, "tool_choice"))

    for key in list(body):
        if key in HOSTED_TOOL_KEYS:
            body.pop(key, None)
            removed.append(key)
    if body.get("web_search_options") is not None:
        body.pop("web_search_options", None)
        removed.append("web_search_options")
    return removed


addons = [ClosedBookProxy()]

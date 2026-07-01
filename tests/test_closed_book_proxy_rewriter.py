# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark request-ID correlation."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REWRITER = REPO / "benchmark" / "closed_book_proxy" / "proxy" / "rewriter.py"
ENTRYPOINT = REPO / "benchmark" / "closed_book_proxy" / "proxy" / "entrypoint.sh"


class _Request:
    def __init__(self, path: str) -> None:
        self.pretty_host = self.host = "switchyard"
        self.path = path
        self.method = "post"
        self.pretty_url = f"http://switchyard{path}"
        self.headers = {"content-type": "application/json", "x-request-id": "client-id"}
        self._text = '{"messages":[]}'

    def get_text(self, *, strict: bool) -> str:
        return self._text

    def set_text(self, value: str) -> None:
        self._text = value


class _Flow:
    def __init__(self, path: str) -> None:
        self.request = _Request(path)
        self.response = None


def _proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("switchyard\n")
    request_map = tmp_path / "request_map.jsonl"
    monkeypatch.setenv("CLOSED_BOOK_MODE", "0")
    monkeypatch.setenv("SWITCHYARD_PROXY_ALLOWLIST", str(allowlist))
    monkeypatch.setenv("SWITCHYARD_PROXY_REQUEST_MAP", str(request_map))

    mitmproxy = types.ModuleType("mitmproxy")
    mitmproxy.http = types.SimpleNamespace(HTTPFlow=object, Response=object)
    monkeypatch.setitem(sys.modules, "mitmproxy", mitmproxy)
    spec = importlib.util.spec_from_file_location(f"proxy_{uuid.uuid4().hex}", REWRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClosedBookProxy(), request_map


def test_llm_request_gets_opaque_id_and_minimal_map_row(monkeypatch, tmp_path: Path) -> None:
    proxy, request_map = _proxy(monkeypatch, tmp_path)
    flow = _Flow("/v1/chat/completions")

    proxy.requestheaders(flow)
    proxy.request(flow)

    request_id = flow.request.headers["x-request-id"]
    assert uuid.UUID(request_id).hex == request_id
    assert json.loads(request_map.read_text()) == {"request_id": request_id}


def test_non_llm_request_is_not_recorded(monkeypatch, tmp_path: Path) -> None:
    proxy, request_map = _proxy(monkeypatch, tmp_path)
    flow = _Flow("/health")

    proxy.requestheaders(flow)
    proxy.request(flow)

    assert flow.request.headers["x-request-id"] == "client-id"
    assert not request_map.exists()


def test_entrypoint_initializes_request_map() -> None:
    source = ENTRYPOINT.read_text()
    assert 'touch "${STRIP_LOG_PATH}" "${REQUEST_MAP_PATH}"' in source
    assert 'export SWITCHYARD_PROXY_REQUEST_MAP="${REQUEST_MAP_PATH}"' in source

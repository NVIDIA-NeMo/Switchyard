# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
REWRITER = REPO / "benchmark" / "closed_book_proxy" / "proxy" / "rewriter.py"
ENTRYPOINT = REPO / "benchmark" / "closed_book_proxy" / "proxy" / "entrypoint.sh"


class _FakeResponse:
    @staticmethod
    def make(status: int, body: bytes, headers: dict[str, str]) -> tuple[object, ...]:
        return status, body, headers


class _FakeRequest:
    def __init__(self, path: str) -> None:
        self.pretty_host = "Switchyard:4000"
        self.host = "Switchyard"
        self.path = path
        self.method = "post"
        self.pretty_url = f"http://switchyard:4000{path}"
        self.headers = {
            "authorization": "Bearer private",
            "content-type": "application/json",
            "x-request-id": "client-supplied",
        }
        self._text = '{"messages":[{"role":"user","content":"private prompt"}]}'

    def get_text(self, *, strict: bool) -> str:
        del strict
        return self._text

    def set_text(self, value: str) -> None:
        self._text = value


class _FakeFlow:
    def __init__(self, path: str) -> None:
        self.request = _FakeRequest(path)
        self.response: object | None = None


def _load_rewriter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    closed_book: bool,
) -> tuple[ModuleType, Path]:
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("switchyard\n")
    request_map = tmp_path / "request_map.jsonl"

    monkeypatch.setenv("CLOSED_BOOK_MODE", "1" if closed_book else "0")
    monkeypatch.setenv("SWITCHYARD_PROXY_ALLOWLIST", str(allowlist))
    monkeypatch.setenv("SWITCHYARD_PROXY_REQUEST_MAP", str(request_map))
    monkeypatch.setenv("SWITCHYARD_PROXY_STRIP_LOG", str(tmp_path / "strip.jsonl"))

    mitmproxy = types.ModuleType("mitmproxy")
    mitmproxy.http = types.SimpleNamespace(HTTPFlow=object, Response=_FakeResponse)
    monkeypatch.setitem(sys.modules, "mitmproxy", mitmproxy)

    module_name = f"switchyard_closed_book_proxy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, REWRITER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, request_map


@pytest.mark.parametrize("closed_book", [False, True])
@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/v1/messages?beta=true",
        "/v1/responses",
    ],
)
def test_llm_request_gets_opaque_id_and_metadata_only_map_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closed_book: bool,
    path: str,
) -> None:
    module, request_map = _load_rewriter(monkeypatch, tmp_path, closed_book=closed_book)
    proxy = module.ClosedBookProxy()
    flow = _FakeFlow(path)

    proxy.requestheaders(flow)

    request_id = flow.request.headers["x-request-id"]
    assert request_id != "client-supplied"
    assert uuid.UUID(request_id).hex == request_id
    assert flow.response is None
    assert not request_map.exists()

    proxy.request(flow)

    rows = request_map.read_text().splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record == {
        "host": "switchyard",
        "method": "POST",
        "ordinal": 1,
        "path": path.split("?", 1)[0],
        "request_id": request_id,
        "timestamp": record["timestamp"],
    }
    assert record["timestamp"].endswith("Z")
    assert "private" not in rows[0]
    assert "authorization" not in rows[0]


def test_request_map_writes_are_serialized_and_ordinals_are_unique(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, request_map = _load_rewriter(monkeypatch, tmp_path, closed_book=False)
    proxy = module.ClosedBookProxy()
    flows = [_FakeFlow("/v1/responses") for _ in range(24)]

    def forward(flow: _FakeFlow) -> None:
        proxy.requestheaders(flow)
        proxy.request(flow)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(forward, flows))

    records = [json.loads(line) for line in request_map.read_text().splitlines()]
    assert sorted(record["ordinal"] for record in records) == list(range(1, 25))
    assert len({record["request_id"] for record in records}) == 24
    assert {flow.request.headers["x-request-id"] for flow in flows} == {
        record["request_id"] for record in records
    }


def test_non_llm_request_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, request_map = _load_rewriter(monkeypatch, tmp_path, closed_book=False)
    proxy = module.ClosedBookProxy()
    flow = _FakeFlow("/health")

    proxy.requestheaders(flow)
    proxy.request(flow)

    assert flow.request.headers["x-request-id"] == "client-supplied"
    assert not request_map.exists()


def test_denied_llm_request_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, request_map = _load_rewriter(monkeypatch, tmp_path, closed_book=True)
    proxy = module.ClosedBookProxy()
    proxy.allowed_hosts.clear()
    flow = _FakeFlow("/v1/messages")

    proxy.requestheaders(flow)
    proxy.request(flow)

    assert flow.response is not None
    assert flow.request.headers["x-request-id"] == "client-supplied"
    assert not request_map.exists()


def test_request_map_write_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module, _ = _load_rewriter(monkeypatch, tmp_path, closed_book=False)
    proxy = module.ClosedBookProxy()
    proxy.request_map = tmp_path
    flow = _FakeFlow("/v1/responses")

    proxy.requestheaders(flow)
    with caplog.at_level("ERROR"):
        proxy.request(flow)

    assert "Failed to append proxy request map" in caplog.text


def test_entrypoint_initializes_and_exports_request_map() -> None:
    source = ENTRYPOINT.read_text()

    assert 'REQUEST_MAP_PATH="${SWITCHYARD_PROXY_REQUEST_MAP:-${PUBLIC_DIR}/request_map.jsonl}"' in source
    assert 'touch "${STRIP_LOG_PATH}" "${REQUEST_MAP_PATH}"' in source
    assert 'export SWITCHYARD_PROXY_REQUEST_MAP="${REQUEST_MAP_PATH}"' in source

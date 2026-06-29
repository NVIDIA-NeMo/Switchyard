# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for launch_claude proxy event-loop safety and the sync probe.

Background
----------
The original bug: ``launch_claude`` called ``asyncio.run()`` to run the
startup probe, which created and then *closed* event loop L1.  The
``AsyncAnthropic`` / ``OpenAILLMClient`` async clients were constructed
during that same ``asyncio.run()`` call, so their internal httpx connection
pools could bind to L1.

When uvicorn later created event loop L2 in a background thread and a
request arrived, ``await client.send(...)`` attempted to use connection-pool
primitives that were (silently) associated with the closed L1.  The coroutine
never resolved — no exception, no timeout, just an indefinite hang.

Fix: the probe became a synchronous ``httpx.Client`` call (no event loop
created), and ``_build_claude_switchyard`` remains a plain ``def``.
Async clients are now constructed with *no* running loop; they bind their
connection pools to uvicorn's L2 on first use.

What these tests verify
-----------------------
1. Sync probe returns the correct True/False for each HTTP status class.
2. ``_build_claude_switchyard`` is a plain function (not ``async def``).
3. No ``asyncio.run()`` is called during switchyard construction.
4. A proxy built and started the correct way handles requests end-to-end
   without hanging against a loopback OpenAI-compatible upstream.
5. The "old" pattern — asyncio.run() before client construction — is
   documented as a regression scenario and checked for cross-loop hang.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest
import respx

from switchyard.cli.launchers.claude_code_launcher import (
    _build_claude_switchyard,
    _find_free_port,
    _spawn_proxy_thread,
    _wait_ready,
)
from switchyard.lib.backends.backend_format_resolver import (
    probe_anthropic_messages_support_sync,
)

_BASE_URL = "https://fake-inference.example.com/v1"
_MESSAGES_URL = "https://fake-inference.example.com/v1/messages"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_openai_completion(model: str = "test-model") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


class _OpenAICompatStub:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._requests: list[dict[str, object]] = []

    def __enter__(self) -> _OpenAICompatStub:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                owner._requests.append({"path": self.path, "body": body})
                content = json.dumps(_minimal_openai_completion()).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(content)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, object]]:
        return list(self._requests)


# ---------------------------------------------------------------------------
# Sync probe — unit tests for every status code branch
# ---------------------------------------------------------------------------


@respx.mock
def test_sync_probe_true_on_400():
    """400 means 'route exists, body invalid' — native path should be used."""
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(400))
    assert probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_true_on_422():
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(422))
    assert probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_true_on_200():
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(200, json={}))
    assert probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_false_on_404():
    """404 means the route doesn't exist — fall back to translation."""
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(404))
    assert not probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_false_on_401():
    """401 = bad credentials — warn and fall back."""
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(401))
    assert not probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_false_on_500():
    """5xx = backend error — safer to fall back than commit to a broken native path."""
    respx.post(_MESSAGES_URL).mock(return_value=httpx.Response(500))
    assert not probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_false_on_timeout():
    respx.post(_MESSAGES_URL).mock(side_effect=httpx.TimeoutException("timeout"))
    assert not probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


@respx.mock
def test_sync_probe_false_on_network_error():
    respx.post(_MESSAGES_URL).mock(side_effect=httpx.ConnectError("refused"))
    assert not probe_anthropic_messages_support_sync(base_url=_BASE_URL, api_key="k")


# ---------------------------------------------------------------------------
# Event-loop safety — structural assertions
# ---------------------------------------------------------------------------


def test_select_and_build_is_not_async():
    """_build_claude_switchyard must be a plain def, not async def.

    An async def (or a wrapper over asyncio.run()) creates and closes event
    loop L1.  Async clients built afterward may bind httpx connection pools to
    L1; when uvicorn's L2 awaits them the coroutine never resolves.
    """
    assert not asyncio.iscoroutinefunction(_build_claude_switchyard), (
        "_build_claude_switchyard must be a plain def. "
        "Using async def or asyncio.run() here creates a loop that closes "
        "before uvicorn starts. Connection-pool primitives bound to the dead "
        "loop cause indefinite hangs on first use in uvicorn's loop."
    )


def test_no_asyncio_run_during_switchyard_construction():
    """asyncio.run() must not be called during switchyard construction.

    This is the direct structural assertion for the regression: if
    asyncio.run() is called, a loop is created and closed before uvicorn
    starts, leading to the cross-loop hang described in the module docstring.
    """
    run_calls: list = []
    original_run = asyncio.run

    def spy_run(coro, **kwargs):
        run_calls.append(coro)
        return original_run(coro, **kwargs)

    with patch("asyncio.run", spy_run):
        _build_claude_switchyard(
            model="test-model",
            api_key="test-key",
            base_url=_BASE_URL,
            timeout=1.0,
        )

    assert not run_calls, (
        f"asyncio.run() was called {len(run_calls)} time(s) inside "
        "_build_claude_switchyard. This creates and closes event loop L1 "
        "before uvicorn's L2 starts. Async clients built here may bind their "
        "connection-pool primitives to L1; awaiting them in L2 hangs forever."
    )


# ---------------------------------------------------------------------------
# Integration test — proxy built correctly handles requests without hanging
# ---------------------------------------------------------------------------


def test_proxy_request_completes_not_hanging():
    """A proxy built without prior asyncio.run() must return a response, not hang.

    Regression test for event-loop contention.  We:
    1. Build the switchyard via the fixed code path (no asyncio.run).
    2. Start uvicorn in a background thread.
    3. Point the real Rust backend at a loopback OpenAI-compatible upstream.
    4. POST to /v1/messages with a 5-second client timeout.
    5. Assert a response arrives.  A hang would surface as httpx.ReadTimeout.

    If the event-loop contention bug were present (clients built inside
    asyncio.run()), step 4 would time out because the async call in uvicorn's
    loop would await primitives bound to the now-closed startup loop.
    """
    port = _find_free_port()

    with _OpenAICompatStub() as upstream:
        switchyard = _build_claude_switchyard(
            model="test-model",
            api_key="test-key",
            base_url=upstream.base_url,
            timeout=2.0,
        )
        server, thread = _spawn_proxy_thread(switchyard, port)
        try:
            assert _wait_ready(port, timeout_s=5.0), "proxy did not start within 5 s"
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "test-model",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )

            assert resp.status_code == 200, resp.text
            assert upstream.requests, "proxy did not call the loopback upstream"
        finally:
            server.should_exit = True
            thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Regression scenario — old pattern with asyncio.run() before construction
# ---------------------------------------------------------------------------


def test_proxy_built_after_closed_loop_documents_hang_risk():
    """Document the failure mode: build after asyncio.run() → hang risk.

    Reproduces the old code pattern:
      asyncio.run(noop)  →  L1 created then closed
      _build_claude_switchyard()  →  async clients constructed
      uvicorn in background thread  →  uses L2
      POST /v1/messages  →  client awaits in L2, but pools may reference L1

    Because an actual hang would block the test suite forever, we use a
    2-second deadline for the request thread and observe the outcome:

    - If the request times out (does NOT complete): the cross-loop hang is
      reproducible on this platform.  We mark the test as ``xfail`` with a
      meaningful message so CI reports it rather than blocking.
    - If it completes: the Python/httpx/anyio stack on this platform handles
      the cross-loop case gracefully — the test passes (xpass).

    Either way the test never actually hangs the suite.
    """
    # Step 1 — close loop L1, exactly as the old launch_claude did.
    async def _noop() -> None:
        pass

    asyncio.run(_noop())

    # Step 2 — build async clients (now with no running loop, but L1 is gone).
    port = _find_free_port()
    with _OpenAICompatStub() as upstream:
        switchyard = _build_claude_switchyard(
            model="test-model",
            api_key="test-key",
            base_url=upstream.base_url,
            timeout=1.0,
        )
        server, thread = _spawn_proxy_thread(switchyard, port)
        completed = threading.Event()
        status_holder: list[int] = []

        def _make_request() -> None:
            try:
                with httpx.Client(timeout=2.0) as c:
                    r = c.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "test-model",
                            "max_tokens": 10,
                            "messages": [{"role": "user", "content": "ping"}],
                        },
                    )
                status_holder.append(r.status_code)
            except Exception:
                pass
            finally:
                completed.set()

        try:
            assert _wait_ready(port, timeout_s=5.0), "proxy did not start"
            t = threading.Thread(target=_make_request, daemon=True)
            t.start()
            finished_in_time = completed.wait(timeout=2.5)

            if not finished_in_time:
                # Cross-loop hang reproduced on this platform — mark xfail so CI
                # reports it as a known failure rather than a blocking hang.
                pytest.xfail(
                    "Request did not complete within 2.5 s after asyncio.run() was "
                    "called before client construction. This confirms the cross-loop "
                    "hang bug is reproducible on this Python/httpx/anyio version. "
                    "The fix (sync probe, no asyncio.run before build) prevents this."
                )
            # If it finished — the platform doesn't exhibit the bug.  Pass.
            assert status_holder, "request thread exited without recording a status"
        finally:
            server.should_exit = True
            thread.join(timeout=3.0)

"""Tiny stdlib HTTP helpers.

The demo is deliberately dependency-free (stdlib only, Python 3.9+) so it runs
anywhere -- on a laptop, in an air-gapped sovereign environment, or in front of a
customer with no network. No pip install, no Rust toolchain, no venv.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

Route = Callable[["Ctx"], Tuple[int, Any]]


class Ctx:
    """Request context handed to a route handler."""

    def __init__(self, path: str, headers: Dict[str, str], body: Optional[dict]) -> None:
        self.path = path
        self.headers = headers          # lowercased names
        self.body = body or {}

    def header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        value = self.headers.get(name.lower())
        return value.strip() if isinstance(value, str) else default


class Service:
    """A minimal routed HTTP service backed by ThreadingHTTPServer."""

    def __init__(self, name: str, port: int) -> None:
        self.name = name
        self.port = port
        self._routes: Dict[Tuple[str, str], Route] = {}
        self._prefix: Dict[Tuple[str, str], Route] = {}
        self._server: Optional[ThreadingHTTPServer] = None

    def route(self, method: str, path: str) -> Callable[[Route], Route]:
        def decorate(fn: Route) -> Route:
            self._routes[(method.upper(), path)] = fn
            return fn

        return decorate

    def prefix(self, method: str, path: str) -> Callable[[Route], Route]:
        """Match any path starting with `path` (for /v1/spend-tree/{id})."""

        def decorate(fn: Route) -> Route:
            self._prefix[(method.upper(), path)] = fn
            return fn

        return decorate

    def _dispatch(self, method: str, path: str, headers: Dict[str, str], body: Optional[dict]):
        bare = path.split("?", 1)[0]
        handler = self._routes.get((method, bare))
        if handler is None:
            for (pmethod, ppath), candidate in self._prefix.items():
                if pmethod == method and bare.startswith(ppath):
                    handler = candidate
                    break
        if handler is None:
            return 404, {"error": {"message": "endpoint_not_found", "code": "endpoint_not_found"}}
        return handler(Ctx(path, headers, body))

    def start(self) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # silence access log
                pass

            def _headers(self) -> Dict[str, str]:
                return {k.lower(): v for k, v in self.headers.items()}

            def _run(self, method: str) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                try:
                    status, payload = service._dispatch(
                        method, self.path, self._headers(), body
                    )
                except Exception as error:  # noqa: BLE001 - demo surface
                    status, payload = 500, {"error": {"message": repr(error)[:200]}}
                self._send(status, payload)

            def _send(self, status: int, payload: Any) -> None:
                if isinstance(payload, str):
                    data = payload.encode("utf-8")
                    ctype = "text/html; charset=utf-8"
                elif isinstance(payload, tuple):        # (content_type, text)
                    ctype, text = payload
                    data = text.encode("utf-8")
                else:
                    data = json.dumps(payload).encode("utf-8")
                    ctype = "application/json"
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._run("GET")

            def do_POST(self) -> None:
                self._run("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def post_json(url: str, payload: dict, headers: Optional[Dict[str, str]] = None,
              timeout: float = 5.0) -> Tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("content-type", "application/json")
    for key, value in (headers or {}).items():
        if value is not None:
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as error:
        return error.code, _decode(error.read())


def get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return _decode(response.read())


def get_text(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")

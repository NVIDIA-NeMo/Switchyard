# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute configured native routes with Python-owned provider clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar, copy_context
from typing import Any, cast, final

from switchyard_rust._native import load_native

_Client = Callable[[dict[str, object]], Awaitable[Mapping[str, object]]]


class UpstreamError(RuntimeError):
    """A provider HTTP failure; its status determines native fallback behavior."""

    def __init__(self, status_code: int, body: str | Mapping[str, object]) -> None:
        """Keep provider detail available without including it in the exception message."""
        if not isinstance(status_code, int) or not 100 <= status_code <= 599:
            raise ValueError("status_code must be an HTTP status between 100 and 599")
        if not isinstance(body, (str, Mapping)):
            raise TypeError("body must be a string or mapping")
        self.status_code = status_code
        self.body = body if isinstance(body, str) else dict(body)
        super().__init__(f"upstream returned HTTP {status_code}")


class _Calls:
    def __init__(self) -> None:
        self.pending: set[asyncio.Future[Any]] = set()
        self.context = copy_context()
        self.closing: asyncio.Task[None] | None = None
        self.is_closed = False

    async def invoke(self, client: _Client, body: dict[str, object]) -> Mapping[str, object]:
        # Rust may schedule a callback after its enclosing Python request was cancelled.
        if self.is_closed:
            raise asyncio.CancelledError
        task = asyncio.current_task()
        if task is not None:
            self.pending.add(task)
        try:
            response = await client(body)
            if isinstance(response, AsyncIterator):
                if (aclose := getattr(response, "aclose", None)) is not None:
                    await aclose()
                raise TypeError("Python Runner callbacks must return buffered provider JSON")
            if not isinstance(response, Mapping):
                raise TypeError("Python Runner callbacks must return buffered provider JSON")
            return dict(response) if not isinstance(response, dict) else response
        finally:
            if task is not None:
                self.pending.discard(task)

    async def close(self) -> None:
        if self.closing is None:
            self.is_closed = True
            self.closing = self.context.run(asyncio.create_task, self.finish())
        # Repeated cancellation must not return before provider finalizers finish.
        cancelled = False
        while not self.closing.done():
            try:
                await asyncio.shield(self.closing)
            except asyncio.CancelledError:
                cancelled = True
        self.closing.result()
        if cancelled:
            raise asyncio.CancelledError

    async def finish(self) -> None:
        pending = tuple(self.pending)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


_calls: ContextVar[_Calls] = ContextVar("switchyard_runner_calls")


def _wrap(client: _Client) -> _Client:
    async def invoke(body: dict[str, object]) -> Mapping[str, object]:
        return await _calls.get().invoke(client, body)

    return invoke


@final
class Runner:
    """Configured routes sharing native routing state and host-owned clients."""

    def __init__(self, inner: Any) -> None:
        """Wrap a native runner; use ``from_toml`` to construct one."""
        self._inner = inner

    @staticmethod
    def from_toml(source: str, *, clients: Mapping[str, _Client]) -> Runner:
        """Load TOML; callbacks receive prepared bodies and return buffered provider JSON.

        Client names match ``llm_clients``. The host owns transport and credentials.
        Raise ``UpstreamError`` for HTTP failures or ``ContextWindowExceededError``
        from ``switchyard.libsy`` for context rejection. Other exceptions are terminal.
        Invalid configuration raises the native ``ServerConfigError``.
        """
        for name, client in clients.items():
            if not callable(client):
                raise TypeError(f"client {name!r} must be callable")
        native = load_native().runner.Runner.from_toml(
            source, clients={name: _wrap(client) for name, client in clients.items()}
        )
        return Runner(native)

    async def execute(
        self,
        request: Mapping[str, object],
        *,
        format: str = "openai_chat",
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """Execute a buffered request; its model names a configured route.

        ``format`` identifies the caller's request and returned response format.
        Cancellation waits for outstanding client work to finish cleanup.
        Use distinct session IDs for independent tasks; keep shared-session turns ordered.
        Streaming requests and callback results are not supported.
        """
        calls = _Calls()
        token = _calls.set(calls)
        try:
            if not isinstance(request, dict):
                request = dict(request)
            if headers is not None and not isinstance(headers, dict):
                headers = dict(headers)
            return cast(dict[str, object], await self._inner._execute(request, format, headers))
        finally:
            _calls.reset(token)
            await calls.close()


__all__ = ["Runner", "UpstreamError"]

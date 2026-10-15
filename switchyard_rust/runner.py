# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute configured native routes with Python-owned provider clients."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar, copy_context
from typing import Any, TypeVar, cast, final

from switchyard_rust._native import load_native

_Response = Mapping[str, object] | AsyncIterator[bytes]
_Client = Callable[[dict[str, object]], Awaitable[_Response]]
_T = TypeVar("_T")


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
        self.iterators: list[AsyncIterator[Any]] = []
        self.context = copy_context()
        self.closing: asyncio.Task[None] | None = None
        self.is_closed = False

    async def invoke(self, client: _Client, body: dict[str, object]) -> _Response:
        # Rust may schedule a callback after its enclosing Python request was cancelled.
        if self.is_closed:
            raise asyncio.CancelledError
        task = asyncio.current_task()
        if task is not None:
            self.pending.add(task)
        try:
            response = await client(body)
            if isinstance(response, AsyncIterator):
                iterator = _Source(aiter(response), self)
                self.iterators.append(iterator)
                if self.closing is not None:
                    await iterator.aclose()
                    raise asyncio.CancelledError
                return iterator
            if not isinstance(response, Mapping):
                raise TypeError(
                    "Python Runner callbacks must return provider JSON or an async byte iterator"
                )
            return dict(response) if not isinstance(response, dict) else response
        finally:
            if task is not None:
                self.pending.discard(task)

    async def wait(self, work: Awaitable[_T]) -> _T:
        task = asyncio.ensure_future(work)
        if self.is_closed:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise asyncio.CancelledError
        self.pending.add(task)
        try:
            return await task
        finally:
            self.pending.discard(task)

    async def close(self) -> None:
        if self.closing is None:
            self.is_closed = True
            self.closing = self.context.run(asyncio.create_task, self.finish())
        await _join_close(self.closing)

    async def finish(self) -> None:
        token = _calls.set(self)
        try:
            pending = tuple(self.pending)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            iterators = tuple(self.iterators)
            errors = await asyncio.gather(
                *(_close_iterator(iterator) for iterator in iterators),
                return_exceptions=True,
            )
            for error in errors:
                if isinstance(error, BaseException):
                    raise error
        finally:
            _calls.reset(token)


async def _close_iterator(iterator: AsyncIterator[Any]) -> None:
    if (aclose := getattr(iterator, "aclose", None)) is not None:
        await aclose()


async def _join_close(task: asyncio.Task[None]) -> None:
    # Repeated cancellation must not return before provider finalizers finish.
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


_calls: ContextVar[_Calls] = ContextVar("switchyard_runner_calls")


def _wrap(client: _Client) -> _Client:
    async def invoke(body: dict[str, object]) -> _Response:
        return await _calls.get().invoke(client, body)

    return invoke


class _Source:
    def __init__(self, inner: AsyncIterator[bytes], calls: _Calls) -> None:
        self.inner = inner
        self.calls = calls
        self.closing: asyncio.Task[None] | None = None

    def __aiter__(self) -> _Source:
        return self

    async def __anext__(self) -> bytes:
        if self.calls.is_closed:
            raise asyncio.CancelledError
        return await self.calls.wait(anext(self.inner))

    async def aclose(self) -> None:
        # Keep the source registered until request shutdown can join its cleanup.
        if self.closing is None:
            self.closing = self.calls.context.run(asyncio.create_task, _close_iterator(self.inner))
        try:
            await _join_close(self.closing)
        finally:
            if self.closing.done():
                self.calls.iterators = [item for item in self.calls.iterators if item is not self]


class _Stream:
    def __init__(self, inner: Any, calls: _Calls) -> None:
        self.inner = inner
        self.calls = calls
        calls.iterators.append(inner)

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> dict[str, object]:
        if self.calls.is_closed:
            raise StopAsyncIteration
        token = _calls.set(self.calls)
        try:
            return cast(dict[str, object], await self.calls.wait(self.inner.__anext__()))
        except BaseException:
            await self.aclose()
            raise
        finally:
            _calls.reset(token)

    async def aclose(self) -> None:
        await self.calls.close()

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()


@final
class Runner:
    """Configured routes sharing native routing state and host-owned clients."""

    def __init__(self, inner: Any) -> None:
        """Wrap a native runner; use ``from_toml`` to construct one."""
        self._inner = inner

    @staticmethod
    def from_toml(source: str, *, clients: Mapping[str, _Client]) -> Runner:
        """Load TOML; callbacks receive prepared bodies and return provider JSON or raw SSE bytes.

        Client names match ``llm_clients``. The host owns transport and credentials.
        Streaming callbacks return async byte iterators whose ``aclose`` releases the provider.
        Raise ``UpstreamError`` for HTTP failures or ``ContextWindowExceededError``
        from ``switchyard.libsy`` for context rejection. Other exceptions are terminal.
        In-band stream errors retain provider fields, not context-window fallback classification.
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
    ) -> dict[str, object] | _Stream:
        """Execute a provider request; its model names a configured route.

        ``format`` identifies the caller's request and returned response format.
        Cancellation waits for outstanding client work to finish cleanup.
        Streams yield provider-event dictionaries; use ``async with`` or ``aclose``
        when stopping early. The host frames events as SSE when serving HTTP.
        Use distinct session IDs for independent tasks; keep shared-session turns ordered.

        For a request with ``stream=True``::

            stream = await runner.execute(request, format=WireFormat.OPENAI_CHAT)
            async with stream:
                async for event in stream:
                    await send_sse_event(event)
        """
        calls = _Calls()
        token = _calls.set(calls)
        streaming = False
        try:
            if not isinstance(request, dict):
                request = dict(request)
            if headers is not None and not isinstance(headers, dict):
                headers = dict(headers)
            response = await self._inner._execute(request, format, headers)
            if isinstance(response, AsyncIterator):
                streaming = True
                return _Stream(response, calls)
            return cast(dict[str, object], response)
        finally:
            _calls.reset(token)
            if not streaming:
                await calls.close()


__all__ = ["Runner", "UpstreamError"]

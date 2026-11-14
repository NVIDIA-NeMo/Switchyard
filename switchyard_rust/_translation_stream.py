# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle of a translation stream and the input reads it initiates."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from types import TracebackType
from typing import Any


class _StreamInput:
    def __init__(self, source: AsyncIterable[Any]) -> None:
        self._iterator: AsyncIterator[Any] = aiter(source)
        self._task: asyncio.Future[Any] | None = None
        self._closed = False

    async def _read(self) -> Any:
        # Rust may have scheduled this coroutine just before aclose() started.
        if self._closed:
            raise asyncio.CancelledError
        self._task = asyncio.ensure_future(anext(self._iterator))
        try:
            return await self._task
        finally:
            self._task = None

    def cancel_read(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()


class _TranslationStream:
    def __init__(self, input_stream: _StreamInput, native: Any) -> None:
        self._input: _StreamInput | None = input_stream
        self._native: Any = native
        self._next_future: asyncio.Future[dict[str, Any]] | None = None
        self._closed = False

    def __aiter__(self) -> "_TranslationStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        if self._next_future is not None:
            raise RuntimeError("translation stream already has a pending read")
        pending: asyncio.Future[dict[str, Any]] = self._native.__anext__()
        self._next_future = pending
        try:
            # Keep the Rust poll alive until input cancellation and cleanup have completed.
            event = await asyncio.shield(pending)
            if self._closed:
                raise StopAsyncIteration
            return event
        except BaseException:
            await self.aclose()
            raise
        finally:
            self._next_future = None

    async def aclose(self) -> None:
        self._closed = True
        if self._input is not None:
            self._input.cancel_read()
        try:
            pending = self._next_future
            if pending is not None and not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # Only consume cancellation of the input read, not cancellation of aclose().
                    if not pending.done():
                        raise
                except StopAsyncIteration:
                    pass
        finally:
            self._native = None
            self._input = None

    async def __aenter__(self) -> "_TranslationStream":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

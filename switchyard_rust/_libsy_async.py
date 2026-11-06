# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Python call tasks alive only while their libsy reply is needed."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, Protocol


class _Call(Protocol):
    def _respond(self, task: asyncio.Future[Any]) -> Awaitable[None]: ...


async def respond(call: _Call, work: Awaitable[Any]) -> None:
    task = asyncio.ensure_future(work)
    try:
        await call._respond(task)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def fail(call: _Call, error: BaseException) -> None:
    async def raise_error() -> None:
        raise error

    await respond(call, raise_error())

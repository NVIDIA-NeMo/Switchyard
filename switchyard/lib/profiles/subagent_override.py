# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route delegated sub-agent requests to a fixed worker runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from switchyard.lib.request_metadata import CTX_PROFILE_REQUEST_HEADERS
from switchyard_rust.core import is_subagent_request

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.route_table import ChainRuntime
    from switchyard_rust.core import ChatRequest


class SubagentOverrideRuntime:
    """Wrap a runtime, sending delegated sub-agent work to ``worker`` instead.

    Normal traffic runs the wrapped runtime unchanged. A request whose headers carry a
    delegated sub-agent signal runs ``worker`` — a passthrough to the route's configured
    ``subagent_target``. The override never rewrites the request or response, and a worker
    failure surfaces as a normal target error rather than falling back to the wrapped
    runtime.

    Detection is :func:`~switchyard_rust.profiles.is_subagent_request`, the protocol
    crate's canonical policy, so harness maintenance turns (Codex ``compact``) stay on
    normal routing and every engine agrees on what counts as delegated work.
    """

    def __init__(self, inner: ChainRuntime, worker: ChainRuntime) -> None:
        self._inner = inner
        self._worker = worker

    def _branch(self, ctx: ProxyContext | None) -> ChainRuntime:
        """Select the worker for delegated work, else the wrapped runtime.

        Without a context there are no retained headers to read, so the request cannot be
        identified as delegated and routes normally.
        """
        if ctx is None:
            return self._inner
        headers = ctx.metadata.get(CTX_PROFILE_REQUEST_HEADERS) or {}
        return self._worker if is_subagent_request(headers) else self._inner

    async def call(self, request: ChatRequest, *, ctx: ProxyContext | None = None) -> Any:
        """Execute one request through the branch its headers select."""
        return await self._branch(ctx).call(request, ctx=ctx)

    def iter_components(self) -> list[Any]:
        """Return both branches' lifecycle components in startup order."""
        return [*self._inner.iter_components(), *self._worker.iter_components()]


__all__ = ["SubagentOverrideRuntime"]

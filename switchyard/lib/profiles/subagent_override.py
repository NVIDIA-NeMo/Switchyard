# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-agent override wrapper routing delegated worker requests to a fixed target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from switchyard.lib.profiles.protocols import (
    ContextAwareProfile,
    ProfileLifecycle,
    ProfileRunner,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatResponse
from switchyard_rust.profiles import ProfileInput, is_subagent_request


@dataclass(slots=True)
class SubagentProcessedRequest:
    """Pairs the branch selected for one request with its request-side state."""

    branch: Any
    processed: Any


class SubagentOverrideProfile:
    """Route recognized sub-agent requests to a fixed override branch.

    Wraps a built profile without changing its behavior for normal traffic. A
    request whose headers carry a delegated sub-agent signal runs through the
    override branch (a passthrough to the profile's configured
    ``subagent_target``); every other request runs the wrapped profile
    unchanged. The override never rewrites the request or response, and an
    override-branch failure surfaces as a normal target error — it is not
    re-routed through the wrapped profile.
    """

    def __init__(self, inner: ProfileRunner, override: ProfileRunner) -> None:
        """Wrap ``inner``, sending sub-agent requests to ``override`` instead."""
        self._inner = inner
        self._override = override

    def iter_components(self) -> list[object]:
        """Return lifecycle components of both branches in startup order."""
        return [*_components(self._inner), *_components(self._override)]

    def _branch(self, input: ProfileInput) -> Any:
        return self._override if is_subagent_request(input.metadata.headers) else self._inner

    async def run(self, input: ProfileInput) -> ChatResponse:
        """Execute the branch selected by the request's sub-agent signal."""
        response = await self._branch(input).run(input)
        return response  # type: ignore[no-any-return]

    async def run_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ChatResponse:
        """Execute the selected branch, preserving the caller-owned context."""
        branch = self._branch(input)
        if isinstance(branch, ContextAwareProfile):
            return await branch.run_with_context(input, ctx)
        response = await branch.run(input)
        return response  # type: ignore[no-any-return]

    async def process(self, input: ProfileInput) -> SubagentProcessedRequest:
        """Run the selected branch's request side, remembering the branch."""
        branch = self._branch(input)
        return SubagentProcessedRequest(branch, await branch.process(input))

    async def process_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> SubagentProcessedRequest:
        """Run the selected branch's request side with the caller's context."""
        branch = self._branch(input)
        if isinstance(branch, ContextAwareProfile):
            return SubagentProcessedRequest(branch, await branch.process_with_context(input, ctx))
        return SubagentProcessedRequest(branch, await branch.process(input))

    async def rprocess(
        self,
        processed: SubagentProcessedRequest,
        response: ChatResponse,
    ) -> ChatResponse:
        """Run the response side of the branch that processed the request."""
        result = await processed.branch.rprocess(processed.processed, response)
        return result  # type: ignore[no-any-return]


def _components(profile: object) -> list[object]:
    """Return a branch's lifecycle components, mirroring ``ProfileSwitchyard``."""
    if isinstance(profile, ProfileLifecycle):
        return profile.iter_components()
    return [profile]


__all__ = ["SubagentOverrideProfile", "SubagentProcessedRequest"]

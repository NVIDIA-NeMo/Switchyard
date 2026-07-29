# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-owned profile request values and runtime protocols."""

from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatResponse
from switchyard_rust.profiles import ProfileInput

ProcessedRequestT = TypeVar("ProcessedRequestT")
ProfileT_co = TypeVar("ProfileT_co", bound="Profile[Any]", covariant=True)


@runtime_checkable
class ProfileRunner(Protocol):
    """Object-safe profile execution surface.

    Generic callers, servers, and config loaders use this surface when they only
    need to execute a complete profile and receive the final response.
    """

    async def run(self, input: ProfileInput) -> ChatResponse:
        """Execute the complete profile lifecycle for one request."""
        ...


@runtime_checkable
class ProfileHooks(Protocol[ProcessedRequestT]):
    """Typed profile hook surface for library-style embedding.

    Profiles use ``ProcessedRequestT`` to expose profile-owned request-side
    state, such as a routing decision, without a generic metadata bag.
    """

    async def process(self, input: ProfileInput) -> ProcessedRequestT:
        """Prepare a request and return profile-owned request-side state."""
        ...

    async def rprocess(
        self,
        processed: ProcessedRequestT,
        response: ChatResponse,
    ) -> ChatResponse:
        """Process a backend response using the matching request-side state."""
        ...


@runtime_checkable
class Profile(ProfileRunner, ProfileHooks[ProcessedRequestT], Protocol[ProcessedRequestT]):
    """Full Python profile authoring contract.

    A profile config must build this complete surface, not just a run-only
    wrapper. The split remains useful because callers may either execute the
    whole profile through ``run`` or embed Switchyard around their own backend
    call by invoking ``process`` and ``rprocess`` directly.
    """


@runtime_checkable
class ProfileLifecycle(Protocol):
    """Optional lifecycle surface for profiles built from reusable components."""

    def iter_components(self) -> list[object]:
        """Return startup/shutdown components in order."""
        ...


@runtime_checkable
class ContextAwareProfile(Profile[ProcessedRequestT], Protocol[ProcessedRequestT]):
    """Optional component bridge surface for profiles that need ``ProxyContext``.

    Profiles can implement only ``run`` / ``process`` / ``rprocess``. Profiles
    that compose existing Python components also expose these methods so
    endpoint adapters can reuse the caller's context.
    """

    async def process_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ProcessedRequestT:
        """Prepare a request using the caller-owned compatibility context."""
        ...

    async def run_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ChatResponse:
        """Execute the complete lifecycle using the caller-owned context."""
        ...


class ProfileConfig(Protocol[ProfileT_co]):
    """Config contract implemented by each Python profile config dataclass."""

    PROFILE_TYPE: ClassVar[str]

    def build(self) -> ProfileT_co:
        """Build this config into a concrete profile runtime."""
        ...


__all__ = [
    "ProcessedRequestT",
    "ContextAwareProfile",
    "Profile",
    "ProfileConfig",
    "ProfileHooks",
    "ProfileInput",
    "ProfileLifecycle",
    "ProfileRunner",
    "ProfileT_co",
]

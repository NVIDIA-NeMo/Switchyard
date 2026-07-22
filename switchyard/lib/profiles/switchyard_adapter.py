# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility adapter that serves a Profile through the legacy endpoint contract."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from switchyard.lib.profiles.protocols import ContextAwareProfile, Profile, ProfileLifecycle
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import CTX_PROFILE_REQUEST_HEADERS
from switchyard.lib.roles import TranslatedResponse
from switchyard_rust.core import (
    ChatRequest,
    ChatResponse,
    SwitchyardProcessorError,
    request_type_value,
)
from switchyard_rust.profiles import ProfileInput, ProfileRequestMetadata
from switchyard_rust.translation import TranslationEngine


class ProfileSwitchyard:
    """Expose a Profile through the existing ``Switchyard.call`` protocol.

    Endpoints and route registries currently call ``obj.call(request, ctx=...)``
    and expect a response translated back to the inbound request format. Profiles
    return provider-native ``ChatResponse`` values instead, so this adapter keeps
    the serving contract stable while chain construction moves to Profile-owned
    runtimes.
    """

    state_key = "switchyard"

    def __init__(
        self,
        profile: Profile[Any],
        translator: TranslationEngine | None = None,
        response_processors: Sequence[Any] = (),
    ) -> None:
        """Create an adapter around one concrete Profile runtime."""
        self._profile = profile
        self._translator = translator or TranslationEngine()
        self._response_processors = tuple(response_processors)

    def iter_components(self) -> list[object]:
        """Return lifecycle components in startup order."""
        if isinstance(self._profile, ProfileLifecycle):
            components = self._profile.iter_components()
        else:
            components = [self._profile]
        return [*components, *self._response_processors, self._translator]

    async def call(
        self,
        request: ChatRequest,
        ctx: ProxyContext | None = None,
    ) -> TranslatedResponse:
        """Run the wrapped Profile and translate its response for the caller.

        Raises:
            SwitchyardProcessorError: A response processor raises or returns a value that is not
                a ``ChatResponse``.
        """
        context = ctx if ctx is not None else ProxyContext()
        profile_input = ProfileInput(
            request,
            metadata=_profile_metadata_from_context(context),
        )
        if isinstance(self._profile, ContextAwareProfile):
            response = await self._profile.run_with_context(profile_input, context)
        else:
            response = await self._profile.run(profile_input)
        current: Any = response
        for processor in self._response_processors:
            try:
                current = await processor.process(context, current)
            except Exception as error:
                raise SwitchyardProcessorError(str(error)) from error
            if not isinstance(current, ChatResponse):
                actual = type(current).__name__
                raise SwitchyardProcessorError(
                    f"Response processor returned {actual}, expected ChatResponse"
                )
        return await self._translator.translate(
            context,
            request,
            cast(ChatResponse, current),
        )


def _profile_metadata_from_context(context: ProxyContext) -> ProfileRequestMetadata:
    inbound_format = (
        request_type_value(context.inbound_format) if context.inbound_format is not None else None
    )
    headers = context.metadata.get(CTX_PROFILE_REQUEST_HEADERS)
    if isinstance(headers, Mapping):
        profile_headers: dict[str, str | list[str]] = {}
        for name, value in headers.items():
            if isinstance(value, list):
                profile_headers[str(name)] = [str(item) for item in value]
            else:
                profile_headers[str(name)] = str(value)
        return ProfileRequestMetadata.from_headers(
            profile_headers,
            inbound_format=inbound_format,
        )
    return ProfileRequestMetadata(
        request_id=context.request_id,
        inbound_format=inbound_format,
    )


__all__ = ["ProfileSwitchyard"]

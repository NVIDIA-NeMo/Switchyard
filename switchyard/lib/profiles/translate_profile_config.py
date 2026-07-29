# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned construction for host-owned format translation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import Any

from switchyard.lib.processors.format_translate import (
    FormatTranslateRequestProcessor,
    FormatTranslateResponseProcessor,
    ModelFormatLookupProcessor,
    StampOriginalFormatProcessor,
    TranslateConfig,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    SwitchyardBackendError,
)


@profile_config("translate")
class TranslateProfileConfig:
    """Dataclass profile config for IGW-style request/response translation hooks."""

    config: TranslateConfig = field(default_factory=TranslateConfig)
    model_selection_processors: Sequence[Any] = ()

    def build(self) -> ComponentChainProfile:
        """Build a hook-only profile around the existing translate processors.

        The returned profile is intended for hosts that call ``process()``,
        perform their own backend call, then call ``rprocess()``. ``run()`` is
        intentionally unsupported because this profile does not own upstream
        transport.
        """
        request_processors = [
            StampOriginalFormatProcessor(),
            *self.model_selection_processors,
            ModelFormatLookupProcessor(self.config),
            FormatTranslateRequestProcessor(),
        ]
        return ComponentChainProfile(
            request_processors=request_processors,
            backend=_TranslateHookOnlyBackend(),
            response_processors=[FormatTranslateResponseProcessor()],
        )


class _TranslateHookOnlyBackend(LLMBackend):
    """Backend placeholder that makes accidental translate ``run()`` fail clearly."""

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        """Advertise all request formats because this backend is never called."""
        return [
            ChatRequestType.OPENAI_CHAT,
            ChatRequestType.OPENAI_RESPONSES,
            ChatRequestType.ANTHROPIC,
        ]

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        """Reject complete execution; external hosts own the backend call."""
        del ctx, request
        raise SwitchyardBackendError(
            "TranslateProfileConfig is hook-only; call process(), invoke the host "
            "backend, then call rprocess()"
        )


__all__ = ["TranslateProfileConfig"]

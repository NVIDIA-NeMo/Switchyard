# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A programmatic Python profile that selects a tier from a request header."""

from dataclasses import dataclass

from switchyard.lib.backends.llm_target import LlmTarget
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.passthrough import PassthroughProfileConfig
from switchyard.lib.profiles.table import profile_config
from switchyard_rust.core import ChatRequest, ChatResponse
from switchyard_rust.profiles import ProfileInput

PROFILE_TYPE = "header-routing"
_DEFAULT_HEADER = "x-switchyard-tier"


@profile_config(PROFILE_TYPE, register=True)
@dataclass(frozen=True, slots=True)
class HeaderRoutingConfig:
    """Route to ``strong`` or ``weak`` based on a request header.

    A header value of ``strong`` selects the strong target; anything else, or a
    missing header, selects ``weak``.
    """

    strong: LlmTarget
    weak: LlmTarget
    header: str = _DEFAULT_HEADER

    def build(self) -> "HeaderRoutingProfile":
        return HeaderRoutingProfile(
            strong=PassthroughProfileConfig(target=self.strong).build(),
            weak=PassthroughProfileConfig(target=self.weak).build(),
            header=self.header,
        )


@dataclass(frozen=True, slots=True)
class HeaderRoutingDecision:
    """Profile-owned request-side state: the prepared request and chosen tier."""

    request: ChatRequest
    tier: str


class HeaderRoutingProfile:
    """Runtime that delegates each request to a passthrough by tier."""

    def __init__(
        self,
        strong: ComponentChainProfile,
        weak: ComponentChainProfile,
        header: str,
    ) -> None:
        self._strong = strong
        self._weak = weak
        self._header = header.lower()

    async def process(self, input: ProfileInput) -> HeaderRoutingDecision:
        return HeaderRoutingDecision(request=input.request, tier=self._tier(input))

    async def run(self, input: ProfileInput) -> ChatResponse:
        decision = await self.process(input)
        backend = self._strong if decision.tier == "strong" else self._weak
        response = await backend.run(ProfileInput(decision.request, input.metadata))
        return await self.rprocess(decision, response)

    async def rprocess(
        self, processed: HeaderRoutingDecision, response: ChatResponse
    ) -> ChatResponse:
        return response

    def _tier(self, input: ProfileInput) -> str:
        values = input.metadata.headers.get(self._header, [])
        chosen = values[0].strip().lower() if values else ""
        return "strong" if chosen == "strong" else "weak"


__all__ = [
    "HeaderRoutingConfig",
    "HeaderRoutingDecision",
    "HeaderRoutingProfile",
]

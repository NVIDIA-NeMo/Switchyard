# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned direct passthrough construction."""

from __future__ import annotations

from switchyard.lib.backends.llm_target import BackendFormat, EndpointConfig, LlmTarget
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend
from switchyard_rust.llm_client import LlmClient


@profile_config("passthrough")
class PassthroughProfileConfig:
    """Dataclass profile config for direct single-upstream passthrough profiles."""

    target: LlmTarget | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None

    def build(self) -> ComponentChainProfile:
        """Build a profile runtime for one direct upstream."""
        from switchyard.lib.llm_client_builder import build_target_llm_client

        backend: LLMBackend
        if self.target is None:
            backend = LlmClient(
                endpoint=EndpointConfig(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout_secs=self.timeout,
                ),
                format=BackendFormat.OPENAI,
            )
        else:
            backend = build_target_llm_client(self.target)

        return ComponentChainProfile(
            backend=backend,
        )


__all__ = ["PassthroughProfileConfig"]

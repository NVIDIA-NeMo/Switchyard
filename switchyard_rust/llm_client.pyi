# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from switchyard_rust.components import (
    BackendFormat,
    EndpointConfig,
    LlmTarget,
    StatsAccumulator,
)
from switchyard_rust.core import LLMBackend

class LlmClient(LLMBackend):
    def __init__(
        self,
        targets: Iterable[LlmTarget] | None = ...,
        *,
        default_target_id: str | None = ...,
        endpoint: EndpointConfig | None = ...,
        format: BackendFormat | str | None = ...,
    ) -> None: ...
    def target_ids(self) -> list[str]: ...
    @property
    def default_target_id(self) -> str | None: ...
    def attach_stats(self, stats: StatsAccumulator) -> None: ...

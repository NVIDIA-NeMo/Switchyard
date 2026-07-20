# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, final

from switchyard_rust.core import SwitchyardRuntimeError

class LlmClient(Protocol):
    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]: ...

class LibsyError(SwitchyardRuntimeError): ...

@final
class LlmTarget:
    def __init__(self, name: str, client: LlmClient) -> None: ...
    @property
    def name(self) -> str: ...

@final
class Algorithm:
    async def run(
        self,
        request: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]: ...

def noop() -> Algorithm: ...
def random(targets: Sequence[LlmTarget]) -> Algorithm: ...

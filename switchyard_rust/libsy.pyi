# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, final

class LlmClient(Protocol):
    async def call(
        self,
        request: Mapping[str, object],
        *,
        target: str,
    ) -> Mapping[str, object]: ...

class LibsyError(RuntimeError): ...

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

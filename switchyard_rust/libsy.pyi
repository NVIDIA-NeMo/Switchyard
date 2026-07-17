# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import final

from switchyard_rust.libsy_protocol import Context, Decision, Request, Response, RoutedLlmClient

class LibsyError(RuntimeError): ...

class LlmTarget:
    def __init__(
        self,
        semantic_name: str,
        *,
        llm_client: RoutedLlmClient | None = None,
    ) -> None: ...
    @property
    def semantic_name(self) -> str: ...
    @property
    def llm_client(self) -> RoutedLlmClient | None: ...
    @llm_client.setter
    def llm_client(self, value: RoutedLlmClient | None) -> None: ...

@final
class LlmTargetSet:
    def __init__(self, targets: list[LlmTarget]) -> None: ...
    @property
    def targets(self) -> list[LlmTarget]: ...
    def get_target(self, name: str) -> LlmTarget: ...
    def __len__(self) -> int: ...

@final
class Algorithm:
    async def run(
        self,
        request: Request,
        *,
        context: Context | None = None,
    ) -> tuple[list[Decision], Response]: ...

def noop() -> Algorithm: ...
def random(targets: LlmTargetSet) -> Algorithm: ...

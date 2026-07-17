# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import final

from switchyard_rust.libsy_protocol import (
    Context,
    Decision,
    Request,
    Response,
    RoutedLlmClient,
)

class LibsyError(RuntimeError): ...

_DecisionValue = Decision

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

class LlmCall:
    context: Context
    request: Request
    decision: Decision
    is_pending: bool

    def respond(self, response: Response) -> None: ...
    def fail(self, message: str) -> None: ...

class Step:
    class CallLlm:
        call: LlmCall

    class Decision:
        decision: _DecisionValue

    class ReturnToAgent:
        response: Response

class RunStream(AsyncIterator[Step]):
    def __aiter__(self) -> RunStream: ...
    async def __anext__(self) -> Step: ...
    async def aclose(self) -> None: ...

@final
class Algorithm:
    async def run(
        self,
        request: Request,
        *,
        context: Context | None = None,
    ) -> tuple[list[Decision], Response]: ...
    def run_stream(
        self,
        request: Request,
        *,
        context: Context | None = None,
    ) -> RunStream: ...

def noop() -> Algorithm: ...
def random(targets: LlmTargetSet) -> Algorithm: ...

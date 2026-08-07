# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal bindings for Rust-owned libsy algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from switchyard_rust.core import _load_native

_EXPORTS = frozenset(
    {
        "Algorithm",
        "LibsyError",
        "LlmFallback",
        "LlmTarget",
        "TaskClassifierConfig",
        "llm_task_classifier",
        "noop",
        "random",
        "stage_router",
    }
)


class LlmClient(Protocol):
    """Structural interface for a Python-hosted model client."""

    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Call the configured target and return an aggregate neutral response."""
        ...


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import final

    from switchyard_rust.core import SwitchyardRuntimeError

    class LibsyError(SwitchyardRuntimeError): ...

    @final
    class LlmTarget:
        def __init__(self, name: str, client: LlmClient) -> None: ...

        @property
        def name(self) -> str: ...

    @final
    class TaskClassifierConfig:
        def __init__(
            self,
            base_threshold: float,
            *,
            threshold_step: float = 0.0,
            session_affinity: bool = False,
            message_hash_fallback: bool = False,
            recent_turn_window: int | None = None,
            max_output_tokens: int = 4096,
            prompt: str | None = None,
        ) -> None: ...

    @final
    class LlmFallback:
        def __init__(
            self,
            judge_target: LlmTarget,
            *,
            config: TaskClassifierConfig,
        ) -> None: ...

    @final
    class Algorithm:
        async def run(
            self,
            request: Mapping[str, object],
            headers: Mapping[str, str] | None = None,
        ) -> tuple[list[dict[str, object]], dict[str, object]]: ...

    def noop() -> Algorithm: ...

    def random(targets: Sequence[LlmTarget]) -> Algorithm: ...

    def llm_task_classifier(
        judge_target: LlmTarget,
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        *,
        config: TaskClassifierConfig,
    ) -> Algorithm: ...

    def stage_router(
        capable_target: LlmTarget,
        efficient_target: LlmTarget,
        *,
        picker: str,
        confidence_threshold: float,
        recent_window: int | None = None,
        escalation_note: str | None = None,
        deescalation_note: str | None = None,
        only_on_wrong_signal_escalation: bool = True,
        capable_system_prompt: str | None = None,
        efficient_system_prompt: str | None = None,
        classifier: LlmFallback | None = None,
    ) -> Algorithm: ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = _load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*sorted(_EXPORTS), "LlmClient"]

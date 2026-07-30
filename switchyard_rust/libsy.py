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
        "LlmTarget",
        "llm_classifier",
        "noop",
        "passthrough",
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
    from typing import Literal, TypedDict, final

    from switchyard_rust.core import SwitchyardRuntimeError

    class LibsyError(SwitchyardRuntimeError): ...

    @final
    class LlmTarget:
        def __init__(self, name: str, client: LlmClient | None = None) -> None: ...

        @property
        def name(self) -> str: ...

    class DecisionResult(TypedDict):
        """Metadata returned by decision-only routing."""

        selected_model: str
        reasoning: str | None
        routing_tier: str | None

    @final
    class Algorithm:
        async def decide(
            self,
            request: Mapping[str, object],
            headers: Mapping[str, str] | None = None,
        ) -> DecisionResult: ...

        async def run(
            self,
            request: Mapping[str, object],
            headers: Mapping[str, str] | None = None,
        ) -> tuple[list[dict[str, object]], dict[str, object]]: ...

    def noop() -> Algorithm: ...

    def llm_classifier(
        *,
        judge: LlmTarget,
        efficient: LlmTarget,
        capable: LlmTarget,
        base_threshold: float,
        min_confidence: float = 0.0,
        capability_elevated_floor: float | None = None,
        session_affinity: bool = False,
        message_hash_fallback: bool = False,
        recent_turn_window: int | None = None,
    ) -> Algorithm: ...

    def passthrough(target: LlmTarget) -> Algorithm: ...

    def random(
        targets: Sequence[LlmTarget],
        *,
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> Algorithm: ...

    def stage_router(
        *,
        capable: LlmTarget,
        efficient: LlmTarget,
        picker: Literal["capable_first", "efficient_first"],
        confidence_threshold: float,
        recent_turn_window: int | None = None,
        handoff_escalation_note: str | None = None,
        handoff_deescalation_note: str | None = None,
        handoff_only_on_wrong_signal_escalation: bool = True,
        capable_system_prompt: str | None = None,
        efficient_system_prompt: str | None = None,
        judge: LlmTarget | None = None,
        classifier_base_threshold: float | None = None,
        classifier_min_confidence: float = 0.0,
        classifier_capability_elevated_floor: float | None = None,
        classifier_recent_turn_window: int | None = None,
    ) -> Algorithm: ...


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = _load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*sorted(_EXPORTS), "LlmClient"]

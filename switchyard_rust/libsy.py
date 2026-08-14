# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal bindings for Rust-owned libsy algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from switchyard_rust._native import load_native

_EXPORTS = frozenset(
    {
        "Algorithm",
        "ContextWindowExceededError",
        "Decision",
        "LibsyError",
        "LlmFallback",
        "ModelCall",
        "Step",
        "TaskClassifierConfig",
        "llm_task_classifier",
        "noop",
        "random",
        "stage_router",
    }
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from typing import ClassVar, Literal, final

    class LibsyError(RuntimeError): ...

    class ContextWindowExceededError(RuntimeError): ...

    @final
    class Decision:
        """A semantic routing choice produced by an algorithm."""

        @property
        def selected_model_id(self) -> str: ...

        @property
        def reasoning(self) -> str | None: ...

        @property
        def is_answer_call(self) -> bool: ...

    _RoutingDecision = Decision

    @final
    class ModelCall:
        @property
        def algorithm(self) -> str: ...

        @property
        def request(self) -> dict[str, object]: ...

        @property
        def models(self) -> list[str]: ...

        @property
        def decision(self) -> Decision: ...

        def into_parts(self) -> tuple[dict[str, object], Decision]: ...

        def respond(self, response: Mapping[str, object]) -> None: ...

        def fail(self, error: BaseException) -> None: ...

    class Step:
        @final
        class CallModel:
            __match_args__: ClassVar[tuple[Literal["call"]]] = ("call",)
            call: ModelCall

        @final
        class Decision:
            __match_args__: ClassVar[tuple[Literal["decision"]]] = ("decision",)
            decision: _RoutingDecision

        @final
        class Done:
            __match_args__: ClassVar[tuple[Literal["response"]]] = ("response",)
            response: dict[str, object]

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
            response_format_type: Literal["json_schema", "json_object"] = "json_schema",
        ) -> None: ...

    @final
    class LlmFallback:
        def __init__(
            self,
            judge_target: str,
            *,
            config: TaskClassifierConfig,
        ) -> None: ...

    @final
    class Algorithm:
        def run_stream(
            self,
            request: Mapping[str, object],
            headers: Mapping[str, str] | None = None,
        ) -> AsyncIterator[Step.CallModel | Step.Decision | Step.Done]: ...

    def noop() -> Algorithm: ...

    def random(
        targets: Sequence[str],
        *,
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> Algorithm: ...

    def llm_task_classifier(
        judge_target: str,
        efficient_target: str,
        capable_target: str,
        *,
        config: TaskClassifierConfig,
    ) -> Algorithm: ...

    def stage_router(
        capable_target: str,
        efficient_target: str,
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
        native: Any = load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)

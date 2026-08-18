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
        "CustomClassifierConfig",
        "Decision",
        "EscalationClassifierConfig",
        "LibsyError",
        "LlmClassifierConfig",
        "LlmFallback",
        "ModelCall",
        "Step",
        "TaskClassifierConfig",
        "llm_classifier",
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
    class CustomClassifierConfig:
        """Configure schema-validated routing across named targets.

        ``max_output_tokens`` must be positive. Enabling ``message_hash_fallback``
        requires ``session_affinity``.
        """

        def __init__(
            self,
            prompt: str,
            response_schema: Mapping[str, object],
            selector: str,
            *,
            session_affinity: bool = False,
            message_hash_fallback: bool = False,
            recent_turn_window: int | None = None,
            max_output_tokens: int = 4096,
        ) -> None: ...

    @final
    class EscalationClassifierConfig:
        """Configure response-based escalation between two targets.

        Counts and token limits must be positive, and ``window_message_chars``
        must be at least 50.
        """

        def __init__(
            self,
            *,
            confirmations: int = 2,
            recent_turn_window: int = 28,
            window_message_chars: int = 500,
            max_output_tokens: int = 4096,
            prompt: str | None = None,
            response_format_type: Literal["json_schema", "json_object"] = "json_schema",
        ) -> None: ...

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
        """Configure capability classification between efficient and capable targets.

        Thresholds must remain within ``[0, 1]``, ``max_output_tokens`` must be
        positive, and ``message_hash_fallback`` requires ``session_affinity``.
        """

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

    class LlmClassifierConfig:
        """Select one supported LLM classifier mode.

        Target names and each nested mode configuration must satisfy the selected
        classifier's invariants.
        """

        @staticmethod
        def capability(
            judge_target: str,
            efficient_target: str,
            capable_target: str,
            *,
            config: TaskClassifierConfig,
        ) -> LlmClassifierConfig:
            """Route by predicted task capability."""
            ...

        @staticmethod
        def escalation(
            judge_target: str,
            efficient_target: str,
            capable_target: str,
            *,
            config: EscalationClassifierConfig,
        ) -> LlmClassifierConfig:
            """Call the efficient target first and escalate judged responses."""
            ...

        @staticmethod
        def custom(
            judge_target: str,
            targets: Sequence[tuple[str, str]],
            *,
            default_target: str,
            config: CustomClassifierConfig,
        ) -> LlmClassifierConfig:
            """Route among named targets using a schema-selected label."""
            ...

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

    def llm_classifier(config: LlmClassifierConfig) -> Algorithm:
        """Build a classifier, raising ValueError when its configuration is invalid."""
        ...

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

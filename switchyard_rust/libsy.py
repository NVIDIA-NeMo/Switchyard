# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal bindings for Rust-owned libsy algorithms."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from switchyard_rust._native import load_native

_EXPORTS = frozenset(
    {
        "Algorithm",
        "ContextWindowExceededError",
        "CustomClassifierConfig",
        "EscalationClassifierConfig",
        "LibsyError",
        "LlmClassifierConfig",
        "LlmFallback",
        "LlmResponse",
        "ModelCall",
        "OutcomeMetadata",
        "RoutingOutcome",
        "Step",
        "TaskClassifierConfig",
        "drive",
        "llm_classifier",
        "llm_task_classifier",
        "noop",
        "random",
        "stage_router",
    }
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from typing import ClassVar, Generic, Literal, TypeVar, final

    _Stream = TypeVar("_Stream", bound=AsyncIterator[Mapping[str, object]])

    class LibsyError(RuntimeError): ...

    class ContextWindowExceededError(RuntimeError): ...

    @final
    class _LlmResponseStream(AsyncIterator[dict[str, object]]):
        """Native outcome stream. Host-supplied iterators need not support close."""

        def __aiter__(self) -> _LlmResponseStream: ...
        async def __anext__(self) -> dict[str, object]: ...
        async def aclose(self) -> None: ...

    class LlmResponse:
        """A normalized aggregate response or live normalized event stream."""

        @final
        class Agg:
            __match_args__: ClassVar[tuple[Literal["response"]]] = ("response",)
            response: dict[str, object]

            def __init__(self, response: Mapping[str, object]) -> None: ...

        @final
        class Stream(Generic[_Stream]):
            __match_args__: ClassVar[tuple[Literal["stream"]]] = ("stream",)
            stream: _Stream

            def __init__(self, stream: _Stream) -> None: ...

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
    class ModelCall:
        @property
        def algorithm(self) -> str: ...

        @property
        def request(self) -> dict[str, object]: ...

        @property
        def models(self) -> list[str]: ...

        def respond(
            self,
            response: LlmResponse.Agg | LlmResponse.Stream[Any],
            *,
            served_model: str | None = None,
            source_id: str | None = None,
        ) -> None: ...

        def fail(self, error: BaseException) -> None: ...

        def _is_completed(self) -> bool: ...

    @final
    class OutcomeMetadata:
        """Read-only outcome identity and optional algorithm evidence."""

        @property
        def outcome_id(self) -> str: ...

        @property
        def algorithm(self) -> str: ...

        @property
        def evidence(self) -> Any | None: ...

    @final
    class RoutingOutcome:
        @property
        def served_model(self) -> str | None: ...

        @property
        def source_id(self) -> str | None: ...

        @property
        def metadata(self) -> OutcomeMetadata | None: ...

        @property
        def selected_model_ids(self) -> list[str]: ...

        @property
        def request(self) -> dict[str, object]: ...

        @property
        def response(self) -> LlmResponse.Agg | LlmResponse.Stream[_LlmResponseStream] | None: ...

    class Step:
        @final
        class CallModel:
            __match_args__: ClassVar[tuple[Literal["call"]]] = ("call",)
            call: ModelCall

        @final
        class Done:
            __match_args__: ClassVar[tuple[Literal["outcome"]]] = ("outcome",)
            outcome: RoutingOutcome

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
        ) -> AsyncIterator[Step.CallModel | Step.Done]: ...

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


class _InputStream:
    """Keep Python reads alive only while Rust owns their stream."""

    def __init__(self, stream: Any) -> None:
        self.iterator = stream.__aiter__()
        self.loop = asyncio.get_running_loop()
        self.context = copy_context()
        self.read: asyncio.Future[Any] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.released = False

    async def __anext__(self) -> Any:
        if self.close_task is not None:
            raise StopAsyncIteration
        self.read = asyncio.ensure_future(self.iterator.__anext__())
        try:
            return await self.read
        finally:
            self.read = None

    def _release(self) -> None:
        self.released = True
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._start_close)

    def _start_close(self) -> asyncio.Task[None]:
        if self.close_task is None:
            self.close_task = self.context.run(self.loop.create_task, self._close())
        return self.close_task

    async def _close(self) -> None:
        iterator, self.iterator = self.iterator, None
        if self.read is not None:
            self.read.cancel()
            await asyncio.gather(self.read, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


async def _close_streams(streams: list[_InputStream], close_all: bool) -> None:
    results = await asyncio.gather(
        *(stream._start_close() for stream in streams if close_all or stream.released),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result


async def drive(
    algorithm: Algorithm,
    request: Mapping[str, object],
    serve: Callable[[ModelCall], Awaitable[None]],
    *,
    headers: Mapping[str, str] | None = None,
) -> RoutingOutcome:
    """Run the native driver with a host callback for each model call.

    ``serve`` must finish each call with ``respond`` or ``fail`` and return None.
    Record host receipts before completing the call. Complete it as the last
    action apart from resource cleanup. Callbacks must honor cancellation and
    release their resources in finally blocks. Retries and accounting belong
    to the host. This function does not make a final-answer call for route-only
    outcomes.

    A returned native response stream belongs to the caller. Consume it or
    await its ``aclose()``, including when a client disconnects. Host-supplied
    input iterators do not need an ``aclose`` method.
    """
    native: Any = load_native().libsy
    tasks: dict[asyncio.Task[Any], ModelCall] = {}
    cancelled: set[asyncio.Task[Any]] = set()
    streams: list[_InputStream] = []
    failure: BaseException | None = None
    stop = asyncio.Event()

    async def serve_owned(call: ModelCall) -> None:
        nonlocal failure
        if stop.is_set():
            return
        task = asyncio.current_task()
        assert task is not None
        tasks[task] = call
        try:
            await serve(call)
            if not call._is_completed():
                raise native.LibsyError("serve returned without completing its model call")
        except BaseException as error:
            if failure is None and not (
                task in cancelled and isinstance(error, asyncio.CancelledError)
            ):
                failure = error
            raise

    run = native._drive(
        algorithm,
        request,
        serve_owned,
        stop,
        streams,
        dict(headers) if headers is not None else None,
    )
    outcome: RoutingOutcome | None = None
    error: BaseException | None = None
    try:
        outcome = await asyncio.shield(run)
    except BaseException as caught:
        error = caught
    stop.set()

    async def cleanup() -> None:
        await asyncio.gather(run, return_exceptions=True)
        for task, call in tasks.items():
            if not task.done() and not call._is_completed():
                cancelled.add(task)
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _close_streams(streams, error is not None or failure is not None)

    # Repeated caller cancellation must not interrupt provider finally blocks.
    joined = asyncio.create_task(cleanup())
    closed_all = False
    while True:
        try:
            await asyncio.shield(joined)
        except asyncio.CancelledError as caught:
            error = caught
            if not joined.done():
                continue
        except BaseException as caught:
            if error is None:
                error = caught
        if (error is not None or failure is not None) and not closed_all:
            joined = asyncio.create_task(_close_streams(streams, True))
            closed_all = True
        else:
            break
    if isinstance(error, asyncio.CancelledError):
        raise error
    if failure is not None:
        raise native.LibsyError(f"Python host callback failed: {failure}") from failure
    if error is not None:
        raise error
    assert outcome is not None
    return outcome


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        native: Any = load_native()
        return getattr(native.libsy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain runtime helpers for profile-owned component pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.session_affinity import CTX_SESSION_KEY
from switchyard.lib.session_cache import SessionCache
from switchyard.lib.session_key import session_key_from_body
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard_rust.core import (
    ChatRequest,
    ChatResponse,
    SwitchyardBackendError,
    SwitchyardContextPoolExhaustedError,
    SwitchyardContextWindowExceededError,
    SwitchyardProcessorError,
    SwitchyardUpstreamError,
)
from switchyard_rust.profiles import ProfileInput


@dataclass(slots=True)
class ChainProcessedRequest:
    """Request-side state needed to finish a profile call.

    Existing Python processors and backends still communicate through
    ``ProxyContext``. The profile bridge snapshots the routing decision here
    and keeps the context synchronized so the decision is visible as typed
    profile-owned state while legacy components continue to run unchanged.
    """

    ctx: ProxyContext
    request: ChatRequest
    selected_target: str | None
    evicted_targets: tuple[str, ...]

    @classmethod
    def from_context(
        cls,
        ctx: ProxyContext,
        request: ChatRequest,
    ) -> ChainProcessedRequest:
        """Snapshot request-side routing state from the compatibility context."""
        return cls(
            ctx=ctx,
            request=request,
            selected_target=_context_selected_target(ctx),
            evicted_targets=tuple(ctx.evicted_targets or ()),
        )

    def sync_to_context(self) -> None:
        """Write profile-owned routing state back for legacy components."""
        self.ctx.selected_target = self.selected_target
        self.ctx.evicted_targets = list(self.evicted_targets) or None


class ComponentChainProfile:
    """Run existing processor/backend components as one Profile runtime.

    This is intentionally private. It lets Profile configs own construction
    directly while older component logic is still being moved into profiles.
    """

    #: LRU cap for the session eviction cache. Without a bound, a long-running
    #: server accumulates one entry per unique conversation indefinitely; the LRU
    #: evicts the oldest sessions when full, so active conversations are retained.
    _SESSION_CACHE_MAX = 1024

    def __init__(
        self,
        *,
        request_processors: Sequence[Any] = (),
        backend: LLMBackend,
        response_processors: Sequence[Any] = (),
        fallback_target_on_evict: str | None = None,
    ) -> None:
        """Create a profile runtime from already-built components."""
        self._request_processors = tuple(request_processors)
        self._backend = backend
        self._response_processors = tuple(response_processors)
        self._fallback_target_on_evict = fallback_target_on_evict
        # Maps session key → frozenset of target IDs that overflowed for that session.
        # Only allocated when eviction is configured; None otherwise.
        self._session_evictions: SessionCache | None = (
            SessionCache(self._SESSION_CACHE_MAX) if fallback_target_on_evict is not None else None
        )

    def iter_components(self) -> list[object]:
        """Return lifecycle components in startup order."""
        return [
            *self._request_processors,
            self._backend,
            *self._response_processors,
        ]

    def with_runtime_components(
        self,
        stats_accumulator: StatsAccumulator | None = None,
        enable_stats: bool = True,
        pre_request_processors: Sequence[Any] = (),
        post_request_processors: Sequence[Any] = (),
        response_processors: Sequence[Any] = (),
    ) -> ComponentChainProfile:
        """Return a copy with serving-level stats and processor hooks applied.

        Profile configs stay user-facing and parseable; shared serving resources
        such as one route-table accumulator or Intake processors are attached by
        the builder that hosts the profile.
        """
        from switchyard.lib.processors.stats_request_processor import (
            StatsRequestProcessor,
        )
        from switchyard.lib.processors.stats_response_processor_accumulator import (
            StatsResponseProcessor,
        )

        request_chain: list[Any] = []
        response_chain: list[Any] = list(self._response_processors)
        backend = self._backend
        stats: StatsAccumulator | None = None

        if enable_stats:
            stats = stats_accumulator or StatsAccumulator()
            request_chain.append(StatsRequestProcessor())
            backend = _attach_stats_to_backend(backend, stats)
            response_chain.append(StatsResponseProcessor(stats))

        request_chain.extend(pre_request_processors)
        request_chain.extend(self._request_processors)
        request_chain.extend(post_request_processors)
        response_chain.extend(response_processors)
        if stats is not None:
            _attach_stats_to_request_processors(request_chain, stats)

        return ComponentChainProfile(
            request_processors=request_chain,
            backend=backend,
            response_processors=response_chain,
            fallback_target_on_evict=self._fallback_target_on_evict,
        )

    async def process(self, input: ProfileInput) -> ChainProcessedRequest:
        """Run request-side components for Profile protocol conformance."""
        return await self.process_with_context(input, _context_from_input(input))

    async def process_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ChainProcessedRequest:
        """Run request-side components against the caller-supplied context."""
        current: Any = input.request
        for processor in self._request_processors:
            try:
                current = await processor.process(ctx, current)
            except Exception as error:
                raise SwitchyardProcessorError(str(error)) from error
            if not isinstance(current, ChatRequest):
                actual = type(current).__name__
                raise SwitchyardProcessorError(
                    f"Request processor returned {actual}, expected ChatRequest"
                )
        return ChainProcessedRequest.from_context(ctx, current)

    async def rprocess(
        self,
        processed: ChainProcessedRequest,
        response: ChatResponse,
    ) -> ChatResponse:
        """Run response-side components for a backend response."""
        current: Any = response
        for processor in self._response_processors:
            try:
                current = await processor.process(processed.ctx, current)
            except Exception as error:
                raise SwitchyardProcessorError(str(error)) from error
            if not isinstance(current, ChatResponse):
                actual = type(current).__name__
                raise SwitchyardProcessorError(
                    f"Response processor returned {actual}, expected ChatResponse"
                )
        return cast(ChatResponse, current)

    async def run(self, input: ProfileInput) -> ChatResponse:
        """Execute the complete Profile protocol lifecycle with a derived context."""
        return await self.run_with_context(input, _context_from_input(input))

    async def run_with_context(
        self,
        input: ProfileInput,
        ctx: ProxyContext,
    ) -> ChatResponse:
        """Execute the complete profile lifecycle with an existing context."""
        processed = await self.process_with_context(input, ctx)
        session_key: str | None = None
        if self._session_evictions is not None:
            session_key = _derive_session_key(ctx, input.request)
            session_evicted: frozenset[str] = self._session_evictions.get(session_key) or frozenset()
            if session_evicted:
                merged = set(processed.evicted_targets) | session_evicted
                processed.evicted_targets = tuple(sorted(merged))
                processed.sync_to_context()
                self._rewrite_evicted_pick(processed)
        try:
            return await self._call_backend_stage(processed)
        except SwitchyardContextWindowExceededError as error:
            if self._fallback_target_on_evict is None:
                raise
            return await self._retry_after_context_overflow(processed, error, session_key)

    async def _call_backend_stage(
        self,
        processed: ChainProcessedRequest,
    ) -> ChatResponse:
        """Call the backend and then response-side components."""
        try:
            response: Any = await self._backend.call(processed.ctx, processed.request)
        except SwitchyardContextWindowExceededError:
            raise
        except SwitchyardUpstreamError:
            raise
        except Exception as error:
            raise SwitchyardBackendError(str(error)) from error
        if not isinstance(response, ChatResponse):
            actual = type(response).__name__
            raise SwitchyardBackendError(
                f"Profile backend returned {actual}, expected ChatResponse"
            )
        return await self.rprocess(processed, response)

    async def _retry_after_context_overflow(
        self,
        processed: ChainProcessedRequest,
        error: BaseException,
        session_key: str | None = None,
    ) -> ChatResponse:
        """Record an evicted target, rewrite to fallback, and retry once."""

        def record_eviction(target_id: str | None) -> None:
            if target_id is None:
                return
            evicted = set(processed.evicted_targets)
            evicted.add(target_id)
            processed.evicted_targets = tuple(sorted(evicted))
            processed.sync_to_context()
            if session_key is not None and self._session_evictions is not None:
                prior: frozenset[str] = self._session_evictions.get(session_key) or frozenset()
                self._session_evictions.put(session_key, prior | {target_id})

        record_eviction(_overflow_target_id(processed, error))
        self._rewrite_evicted_pick(processed)
        try:
            return await self._call_backend_stage(processed)
        except SwitchyardContextWindowExceededError as second:
            second_target = _overflow_target_id(processed, second)
            record_eviction(second_target)
            last_target = second_target or "unknown"
            reason = "all attempted targets returned context-window overflow"
            pool_error = SwitchyardContextPoolExhaustedError(
                f"context pool exhausted after target {last_target}: {reason}"
            )
            pool_error.last_target_id = last_target  # type: ignore[attr-defined]
            pool_error.reason = reason  # type: ignore[attr-defined]
            raise pool_error from second

    def _rewrite_evicted_pick(self, processed: ChainProcessedRequest) -> None:
        """Rewrite an evicted or exception-only target to the configured fallback."""
        selected = processed.selected_target
        evicted = set(processed.evicted_targets)
        if (selected is not None and selected in evicted) or (not selected and evicted):
            processed.selected_target = self._fallback_target_on_evict
            processed.sync_to_context()


def _derive_session_key(ctx: ProxyContext, request: ChatRequest) -> str:
    """Return the session key for this request, memoized on ctx."""
    cached = ctx.metadata.get(CTX_SESSION_KEY)
    if isinstance(cached, str):
        return cached
    key = session_key_from_body(request.body)
    ctx.metadata[CTX_SESSION_KEY] = key
    return key


def _context_from_input(input: ProfileInput) -> ProxyContext:
    """Build a compatibility context from profile metadata."""
    ctx = ProxyContext()
    metadata = input.metadata
    if metadata.request_id is not None:
        ctx.request_id = metadata.request_id
    if metadata.inbound_format is not None:
        ctx.inbound_format = cast(Any, metadata.inbound_format)
    for key, values in metadata.headers.items():
        ctx.metadata[key] = list(values)
    return ctx


def _context_selected_target(ctx: ProxyContext) -> str | None:
    """Return a normalized selected target from the compatibility context."""
    selected = ctx.selected_target
    return selected if isinstance(selected, str) and selected else None


def _overflow_target_id(
    processed: ChainProcessedRequest,
    error: BaseException,
) -> str | None:
    """Return target id carried by an overflow error or processed state."""
    target_id = getattr(error, "target_id", None)
    if isinstance(target_id, str) and target_id:
        return target_id
    return processed.selected_target


def _attach_stats_to_backend(
    backend: LLMBackend,
    stats: StatsAccumulator,
) -> LLMBackend:
    """Wrap native backends or wire existing Python stats hooks in place."""
    from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend

    try:
        return StatsLlmBackend(backend, stats)
    except TypeError:
        # Python-only backends cannot be wrapped by the Rust stats binding.
        # Fail loudly if the backend does not expose one of the known
        # compatibility hooks; silent metrics loss is worse than a build error.
        if not _attach_stats_to_python_backend(backend, stats):
            backend_type = type(backend).__qualname__
            raise TypeError(
                f"{backend_type} cannot be wrapped for stats and exposes no "
                "Python stats compatibility hook"
            ) from None
        return backend


def _attach_stats_to_python_backend(
    backend: LLMBackend,
    stats: StatsAccumulator,
) -> bool:
    """Wire stats into Python-only backend shapes that already support it."""
    attached = False
    if hasattr(backend, "_stats"):
        cast(Any, backend)._stats = stats
        attached = True

    inner = getattr(backend, "_inner", None)
    if inner is not None:
        from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend

        try:
            cast(Any, backend)._inner = StatsLlmBackend(inner, stats)
            attached = True
        except TypeError:
            attached = _attach_stats_to_python_backend(inner, stats) or attached

    nested = getattr(backend, "_backends", None)
    if not isinstance(nested, dict):
        return attached

    from switchyard.lib.backends.stats_llm_backend import StatsLlmBackend

    for label, child in list(nested.items()):
        try:
            nested[label] = StatsLlmBackend(child, stats)
            attached = True
        except TypeError:
            attached = _attach_stats_to_python_backend(child, stats) or attached
    return attached


def _attach_stats_to_request_processors(
    processors: Sequence[Any],
    stats: StatsAccumulator,
) -> None:
    """Wire existing classifier/planner stats hooks without config fields."""
    for processor in processors:
        attach_stats = getattr(processor, "attach_stats_accumulator", None)
        if callable(attach_stats):
            attach_stats(stats)
        if hasattr(processor, "_stats_accumulator"):
            cast(Any, processor)._stats_accumulator = stats


__all__ = ["ChainProcessedRequest", "ComponentChainProfile"]

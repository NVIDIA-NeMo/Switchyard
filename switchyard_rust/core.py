# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard core bindings plus Python-only compatibility chain wrappers."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from os import PathLike
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, cast

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class _ChatRequestType(Protocol):
    """Rust-owned request format tag exposed through PyO3."""

    OPENAI_CHAT: ClassVar[_ChatRequestType]
    OPENAI_RESPONSES: ClassVar[_ChatRequestType]
    ANTHROPIC: ClassVar[_ChatRequestType]

    value: str


class _ChatRequest(Protocol):
    """Rust-owned chat request value exposed through PyO3."""

    request_type: _ChatRequestType
    body: Any
    model: str | None

    @classmethod
    def openai_chat(cls, body: Mapping[str, Any] | JsonValue) -> _ChatRequest: ...

    @classmethod
    def openai_responses(cls, body: Mapping[str, Any] | JsonValue) -> _ChatRequest: ...

    @classmethod
    def anthropic(cls, body: Mapping[str, Any] | JsonValue) -> _ChatRequest: ...

    def set_model(self, model: str) -> None: ...
    def replace_body(self, body: Mapping[str, Any] | JsonValue) -> None: ...
    def to_body(self) -> JsonValue: ...


class _ChatResponseType(Protocol):
    """Rust-owned response format/delivery tag exposed through PyO3."""

    OPENAI_COMPLETION: ClassVar[_ChatResponseType]
    OPENAI_STREAM: ClassVar[_ChatResponseType]
    OPENAI_RESPONSES_COMPLETION: ClassVar[_ChatResponseType]
    OPENAI_RESPONSES_STREAM: ClassVar[_ChatResponseType]
    ANTHROPIC_COMPLETION: ClassVar[_ChatResponseType]
    ANTHROPIC_STREAM: ClassVar[_ChatResponseType]

    value: str


class _ChatResponse(Protocol):
    """Rust-owned chat response value exposed through PyO3."""

    response_type: _ChatResponseType
    body: Any
    stream: Any

    @classmethod
    def openai_completion(cls, body: Mapping[str, Any] | JsonValue | object) -> _ChatResponse: ...

    @classmethod
    def openai_stream(cls, stream: object) -> _ChatResponse: ...

    @classmethod
    def openai_responses_completion(
        cls, body: Mapping[str, Any] | JsonValue | object,
    ) -> _ChatResponse: ...

    @classmethod
    def openai_responses_stream(cls, stream: object) -> _ChatResponse: ...

    @classmethod
    def anthropic_completion(cls, body: Mapping[str, Any] | JsonValue | object) -> _ChatResponse: ...

    @classmethod
    def anthropic_stream(cls, stream: object) -> _ChatResponse: ...

    def replace_body(self, body: Mapping[str, Any] | JsonValue | object) -> None: ...
    def to_body(self) -> JsonValue: ...


class _ChatResponseStream(Protocol):
    """Rust-owned async stream adapter exposed through PyO3."""

    def __init__(self, source: object) -> None: ...
    def tap(self, callback: object) -> _ChatResponseStream: ...
    def map(self, callback: object) -> _ChatResponseStream: ...
    def on_complete(self, callback: object) -> _ChatResponseStream: ...
    def __aiter__(self) -> _ChatResponseStream: ...
    async def __anext__(self) -> Any: ...


class _ProxyMetadata(Protocol):
    """Rust-owned metadata map exposed through PyO3."""

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def setdefault(self, key: str, default: Any = None) -> Any: ...
    def update(self, metadata: Mapping[str, Any] | Iterable[tuple[str, Any]], /) -> None: ...
    def copy(self) -> dict[str, Any]: ...
    def keys(self) -> KeysView[str]: ...
    def values(self) -> ValuesView[Any]: ...
    def items(self) -> ItemsView[str, Any]: ...


class _ProxyContext(Protocol):
    """Rust-owned per-request context exposed through PyO3."""

    metadata: _ProxyMetadata
    request_id: str | None
    inbound_format: _ChatRequestType | None
    selected_model: str | None
    selected_target: str | None
    backend_call_latency_ms: float | None
    evicted_targets: list[str] | None


class _LLMBackend(Protocol):
    """Rust-owned backend role exposed through PyO3."""

    supported_request_types: list[_ChatRequestType]

    async def call(self, ctx: _ProxyContext, request: _ChatRequest) -> _ChatResponse: ...
    def startup(self) -> Any: ...
    def shutdown(self) -> Any: ...


class _Switchyard(Protocol):
    """Python compatibility Switchyard chain."""

    state_key: ClassVar[str]
    _backend: _LLMBackend
    _translator: Any

    def __init__(
        self,
        *,
        request_processors: Iterable[Any] | None = None,
        backend: _LLMBackend,
        response_processors: Iterable[Any] | None = None,
        translator: Any,
    ) -> None: ...
    def iter_components(self) -> list[Any]: ...
    async def call(
        self,
        request: _ChatRequest,
        *,
        ctx: _ProxyContext | None = None,
    ) -> Any: ...


class _ProfileConfigDocument(Protocol):
    """Rust-owned profile config document exposed through PyO3."""

    def resolve(self) -> _ProfileConfigPlan: ...


class _ProfileConfigPlan(Protocol):
    """Rust-owned resolved profile config plan exposed through PyO3."""

    def profile_ids(self) -> list[str]: ...
    def target_ids(self) -> list[str]: ...
    def profile_type(self, profile_id: str) -> str | None: ...
    def target(self, target_id: str) -> object | None: ...
    def build_profile(self, profile_id: str) -> _Profile: ...
    def build_profiles(self) -> dict[str, _Profile]: ...


class _Profile(Protocol):
    """Rust-owned profile runtime exposed through PyO3."""

    profile_id: str

    async def run(self, request: _ChatRequest) -> _ChatResponse: ...


class _ParseProfileConfigStr(Protocol):
    def __call__(
        self,
        input: str,
        format: str = "yaml",
    ) -> _ProfileConfigDocument: ...


class _ParseProfileConfigPath(Protocol):
    def __call__(self, path: str | PathLike[str]) -> _ProfileConfigDocument: ...


class _LoadProfileConfig(Protocol):
    def __call__(self, path: str | PathLike[str]) -> _ProfileConfigPlan: ...


class _NativeModule(Protocol):
    ChatRequest: type[_ChatRequest]
    ChatRequestType: type[_ChatRequestType]
    ChatResponse: type[_ChatResponse]
    ChatResponseStream: type[_ChatResponseStream]
    ChatResponseType: type[_ChatResponseType]
    LLMBackend: type[_LLMBackend]
    ProxyMetadata: type[_ProxyMetadata]
    ProxyContext: type[_ProxyContext]
    SwitchyardRuntimeError: type[RuntimeError]
    SwitchyardConfigError: type[RuntimeError]
    SwitchyardInvalidIdError: type[RuntimeError]
    SwitchyardDuplicateRegistrationError: type[RuntimeError]
    SwitchyardModelNotFoundError: type[RuntimeError]
    SwitchyardUnsupportedRequestTypeError: type[RuntimeError]
    SwitchyardInvalidRequestError: type[RuntimeError]
    SwitchyardProcessorError: type[RuntimeError]
    SwitchyardBackendError: type[RuntimeError]
    SwitchyardUpstreamError: type[RuntimeError]
    SwitchyardContextWindowExceededError: type[RuntimeError]
    SwitchyardContextPoolExhaustedError: type[RuntimeError]
    ProfileConfigDocument: type[_ProfileConfigDocument]
    ProfileConfigPlan: type[_ProfileConfigPlan]
    Profile: type[_Profile]
    load_profile_config: _LoadProfileConfig
    parse_profile_config_path: _ParseProfileConfigPath
    parse_profile_config_str: _ParseProfileConfigStr
    # Shared profile input bindings re-exported through `switchyard_rust.profiles`.
    ProfileRequestMetadata: type[Any]
    ProfileInput: type[Any]
    # Session-affinity primitives.
    SessionCache: type[Any]
    session_key_from_body: Any

    def run_profile_server(
        self,
        config_path: str,
        host: str,
        port: int,
        backlog: int,
        dry_run: bool,
    ) -> None: ...


def _ensure_switchyard_version_env() -> None:
    if os.environ.get("SWITCHYARD_VERSION", "").strip():
        return
    for distribution in ("switchyard", "nemo-switchyard"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version.strip():
            os.environ["SWITCHYARD_VERSION"] = version
            return


def _load_native() -> _NativeModule:
    _ensure_switchyard_version_env()
    try:
        return cast(_NativeModule, importlib.import_module("switchyard_rust._switchyard_rust"))
    except ImportError as exc:  # pragma: no cover - broken install guard
        raise RuntimeError(
            "Rust core bindings are required. Run `uv run maturin develop` "
            "or install a built switchyard wheel."
        ) from exc


def _role_type_name(value: object) -> str:
    """Return a stable display name for role validation errors."""
    return type(value).__name__


def _processor_error(error: BaseException) -> RuntimeError:
    """Wrap processor failures in the public Switchyard processor error."""
    native = _load_native()
    if isinstance(error, native.SwitchyardRuntimeError):
        return error
    return native.SwitchyardProcessorError(str(error))


def _backend_error(error: BaseException) -> RuntimeError:
    """Wrap backend failures in the public Switchyard backend error."""
    native = _load_native()
    if isinstance(error, native.SwitchyardRuntimeError):
        return error
    return native.SwitchyardBackendError(str(error))


class Switchyard:
    """Python-only compatibility chain for the current FastAPI/recipe surface."""

    state_key: ClassVar[str] = "switchyard"

    def __init__(
        self,
        *,
        request_processors: Iterable[Any] | None = None,
        backend: Any,
        response_processors: Iterable[Any] | None = None,
        translator: Any,
        fallback_target_on_evict: str | None = None,
    ) -> None:
        if not isinstance(backend, _load_native().LLMBackend):
            actual = _role_type_name(backend)
            raise TypeError(f"Switchyard backend must be LLMBackend, got {actual}")
        self._request_components = tuple(request_processors or ())
        self._backend = backend
        self._response_components = tuple(response_processors or ())
        self._translator = translator
        self._fallback_target_on_evict = fallback_target_on_evict

    def iter_components(self) -> list[Any]:
        """Return lifecycle components in startup order."""
        return [
            *self._request_components,
            self._backend,
            *self._response_components,
            self._translator,
        ]

    async def call(
        self,
        request: Any,
        *,
        ctx: Any | None = None,
    ) -> Any:
        """Run the compatibility chain and translate the final response."""
        context = ctx if ctx is not None else _load_native().ProxyContext()
        processed_request = await self._process_request_components(context, request)
        try:
            response = await self._call_backend_stage(context, processed_request)
        except _load_native().SwitchyardContextWindowExceededError as error:
            if self._fallback_target_on_evict is None:
                raise
            response = await self._retry_after_context_overflow(
                context,
                processed_request,
                error,
            )
        return await self._translator.translate(context, processed_request, response)

    async def _call_backend_stage(
        self,
        ctx: Any,
        request: Any,
    ) -> Any:
        """Call the backend and then response processors."""
        native = _load_native()
        try:
            response = await self._backend.call(ctx, request)
        except native.SwitchyardContextWindowExceededError:
            raise
        except Exception as error:
            raise _backend_error(error) from error
        if not isinstance(response, native.ChatResponse):
            actual = _role_type_name(response)
            raise native.SwitchyardBackendError(
                f"Switchyard backend returned {actual}, expected ChatResponse",
            )
        return await self._process_response_components(ctx, response)

    async def _process_request_components(self, ctx: Any, request: Any) -> Any:
        """Run request-side compatibility components in order."""
        native = _load_native()
        current = request
        for component in self._request_components:
            process = getattr(component, "process", None)
            if not callable(process):
                actual = _role_type_name(component)
                raise native.SwitchyardProcessorError(
                    f"Request component {actual} must define process(ctx, request)",
                )
            try:
                current = await process(ctx, current)
            except Exception as error:
                raise _processor_error(error) from error
            if not isinstance(current, native.ChatRequest):
                actual = _role_type_name(current)
                raise native.SwitchyardProcessorError(
                    f"Request component returned {actual}, expected ChatRequest",
                )
        return current

    async def _process_response_components(self, ctx: Any, response: Any) -> Any:
        """Run response-side compatibility components in order."""
        native = _load_native()
        current = response
        for component in self._response_components:
            process = getattr(component, "process", None)
            if not callable(process):
                actual = _role_type_name(component)
                raise native.SwitchyardProcessorError(
                    f"Response component {actual} must define process(ctx, response)",
                )
            try:
                current = await process(ctx, current)
            except Exception as error:
                raise _processor_error(error) from error
            if not isinstance(current, native.ChatResponse):
                actual = _role_type_name(current)
                raise native.SwitchyardProcessorError(
                    f"Response component returned {actual}, expected ChatResponse",
                )
        return current

    async def _retry_after_context_overflow(
        self,
        ctx: Any,
        request: Any,
        error: BaseException,
    ) -> Any:
        """Record the evicted target, rewrite the selection, and retry once."""
        native = _load_native()
        target_id = self._overflow_target_id(ctx, error)
        if target_id is not None:
            evicted = set(ctx.evicted_targets or [])
            evicted.add(target_id)
            ctx.evicted_targets = sorted(evicted)
        self._rewrite_evicted_pick(ctx)
        try:
            return await self._call_backend_stage(ctx, request)
        except native.SwitchyardContextWindowExceededError as second:
            last_target = self._overflow_target_id(ctx, second) or "unknown"
            reason = "all attempted targets returned context-window overflow"
            pool_error = native.SwitchyardContextPoolExhaustedError(
                f"context pool exhausted after target {last_target}: {reason}",
            )
            cast(Any, pool_error).last_target_id = last_target
            cast(Any, pool_error).reason = reason
            raise pool_error from second

    def _overflow_target_id(
        self,
        ctx: Any,
        error: BaseException,
    ) -> str | None:
        """Return the target id carried by an overflow error or current context."""
        target_id = getattr(error, "target_id", None)
        if isinstance(target_id, str) and target_id:
            return target_id
        selected = ctx.selected_target
        return selected if isinstance(selected, str) and selected else None

    def _rewrite_evicted_pick(self, ctx: Any) -> None:
        """Rewrite an evicted or exception-only target to the configured fallback."""
        selected = ctx.selected_target
        evicted = set(ctx.evicted_targets or [])
        if (selected is not None and selected in evicted) or (not selected and evicted):
            ctx.selected_target = self._fallback_target_on_evict


if TYPE_CHECKING:
    class ChatRequestType:
        """Static view of the Rust-owned ChatRequestType class."""

        OPENAI_CHAT: ClassVar[ChatRequestType]
        OPENAI_RESPONSES: ClassVar[ChatRequestType]
        ANTHROPIC: ClassVar[ChatRequestType]
        value: str

    class ChatRequest:
        """Static view of the Rust-owned ChatRequest class."""

        request_type: ChatRequestType
        body: Any
        model: str | None

        @classmethod
        def openai_chat(cls, body: Mapping[str, Any] | JsonValue) -> ChatRequest: ...
        @classmethod
        def openai_responses(cls, body: Mapping[str, Any] | JsonValue) -> ChatRequest: ...
        @classmethod
        def anthropic(cls, body: Mapping[str, Any] | JsonValue) -> ChatRequest: ...
        def validate(self) -> None: ...
        def set_model(self, model: str) -> None: ...
        def replace_body(self, body: Mapping[str, Any] | JsonValue) -> None: ...
        def to_body(self) -> JsonValue: ...

    class ChatResponseType:
        """Static view of the Rust-owned ChatResponseType class."""

        OPENAI_COMPLETION: ClassVar[ChatResponseType]
        OPENAI_STREAM: ClassVar[ChatResponseType]
        OPENAI_RESPONSES_COMPLETION: ClassVar[ChatResponseType]
        OPENAI_RESPONSES_STREAM: ClassVar[ChatResponseType]
        ANTHROPIC_COMPLETION: ClassVar[ChatResponseType]
        ANTHROPIC_STREAM: ClassVar[ChatResponseType]
        value: str

    class ChatResponse:
        """Static view of the Rust-owned ChatResponse class."""

        response_type: ChatResponseType
        body: Any
        stream: Any

        @classmethod
        def openai_completion(
            cls,
            body: Mapping[str, Any] | JsonValue | object,
        ) -> ChatResponse: ...
        @classmethod
        def openai_stream(cls, stream: object) -> ChatResponse: ...
        @classmethod
        def openai_responses_completion(
            cls,
            body: Mapping[str, Any] | JsonValue | object,
        ) -> ChatResponse: ...
        @classmethod
        def openai_responses_stream(cls, stream: object) -> ChatResponse: ...
        @classmethod
        def anthropic_completion(
            cls,
            body: Mapping[str, Any] | JsonValue | object,
        ) -> ChatResponse: ...
        @classmethod
        def anthropic_stream(cls, stream: object) -> ChatResponse: ...
        def replace_body(self, body: Mapping[str, Any] | JsonValue | object) -> None: ...
        def to_body(self) -> JsonValue: ...

    class ChatResponseStream:
        """Static view of the Rust-owned ChatResponseStream class."""

        def __init__(self, source: object) -> None: ...
        def tap(self, callback: object) -> ChatResponseStream: ...
        def map(self, callback: object) -> ChatResponseStream: ...
        def on_complete(self, callback: object) -> ChatResponseStream: ...
        def __aiter__(self) -> ChatResponseStream: ...
        async def __anext__(self) -> Any: ...

    class ProxyMetadata(_ProxyMetadata, Protocol):
        """Static view of the Rust-owned ProxyMetadata class."""

    class ProxyContext:
        """Static view of the Rust-owned ProxyContext class."""

        metadata: _ProxyMetadata
        request_id: str | None
        inbound_format: _ChatRequestType | None
        selected_model: str | None
        selected_target: str | None
        backend_call_latency_ms: float | None
        evicted_targets: list[str] | None

        def __init__(
            self,
            metadata: Mapping[str, Any] | None = None,
            request_id: str | None = None,
        ) -> None: ...

    class LLMBackend:
        """Static view of the Rust-owned LLMBackend role class."""

        @property
        def supported_request_types(self) -> list[ChatRequestType]: ...
        async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse: ...
        def startup(self) -> Any: ...
        def shutdown(self) -> Any: ...

    class SwitchyardRuntimeError(RuntimeError):
        """Base class for Rust-owned Switchyard runtime errors."""

    class SwitchyardConfigError(SwitchyardRuntimeError):
        """Raised for invalid Rust Switchyard configuration."""

    class SwitchyardInvalidIdError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard identifier is invalid."""

    class SwitchyardDuplicateRegistrationError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard registry receives a duplicate ID."""

    class SwitchyardModelNotFoundError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard route table cannot find a model."""

    class SwitchyardUnsupportedRequestTypeError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard component rejects a request format."""

    class SwitchyardInvalidRequestError(SwitchyardRuntimeError):
        """Raised when a request body fails semantic validation (e.g. empty messages)."""

    class SwitchyardProcessorError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard processor fails."""

    class SwitchyardBackendError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard backend fails."""

    class SwitchyardUpstreamError(SwitchyardRuntimeError):
        """Raised when a Rust Switchyard upstream call fails."""
        status_code: int
        body: str

    class SwitchyardContextWindowExceededError(SwitchyardBackendError):
        """Raised by an LLMBackend on an upstream context-window overflow."""

    class SwitchyardContextPoolExhaustedError(SwitchyardBackendError):
        """Raised when every attempted routing target was evicted."""

    class SessionCache:
        """Static view of the Rust-owned bounded-LRU session cache."""

        def __init__(self, max_sessions: int) -> None: ...
        def get(self, key: str) -> Any | None: ...
        def put(self, key: str, value: Any) -> None: ...
        def values(self) -> list[Any]: ...
        @property
        def max_sessions(self) -> int: ...
        def __len__(self) -> int: ...

    def session_key_from_body(
        body: Mapping[str, Any] | JsonValue, depth: int = 0
    ) -> str | None: ...


def __getattr__(name: str) -> object:
    if name == "SessionCache":
        return _load_native().SessionCache
    if name == "session_key_from_body":
        return _load_native().session_key_from_body
    if name == "ChatRequest":
        return _load_native().ChatRequest
    if name == "ChatRequestType":
        return _load_native().ChatRequestType
    if name == "ChatResponse":
        return _load_native().ChatResponse
    if name == "ChatResponseStream":
        return _load_native().ChatResponseStream
    if name == "ChatResponseType":
        return _load_native().ChatResponseType
    if name == "LLMBackend":
        return _load_native().LLMBackend
    if name == "ProxyMetadata":
        return _load_native().ProxyMetadata
    if name == "ProxyContext":
        return _load_native().ProxyContext
    if name in {
        "SwitchyardRuntimeError",
        "SwitchyardConfigError",
        "SwitchyardInvalidIdError",
        "SwitchyardDuplicateRegistrationError",
        "SwitchyardModelNotFoundError",
        "SwitchyardUnsupportedRequestTypeError",
        "SwitchyardInvalidRequestError",
        "SwitchyardProcessorError",
        "SwitchyardBackendError",
        "SwitchyardUpstreamError",
        "SwitchyardContextWindowExceededError",
        "SwitchyardContextPoolExhaustedError",
    }:
        return getattr(_load_native(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def request_type_value(value: object) -> str:
    """Normalize Rust request-format tags and wire strings."""
    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, str):
        raise TypeError(f"Request type must be a string-like value, got {type(raw).__name__}")
    if raw == "anthropic_messages":
        return "anthropic"
    if raw in {"openai_chat", "openai_responses", "anthropic"}:
        return raw
    raise ValueError(f"Unknown request type: {value!r}")


def request_type_enum(value: object) -> _ChatRequestType:
    """Normalize a request-format tag to the Rust-bound enum object."""
    normalized = request_type_value(value)
    request_type = _load_native().ChatRequestType
    if normalized == "openai_chat":
        return request_type.OPENAI_CHAT
    if normalized == "openai_responses":
        return request_type.OPENAI_RESPONSES
    if normalized == "anthropic":
        return request_type.ANTHROPIC
    raise ValueError(f"Unknown request type: {value!r}")


def request_type_matches(request: object, request_type: object) -> bool:
    """Return whether a Rust-backed request has the requested wire format."""
    return request_type_value(cast(Any, request).request_type) == request_type_value(request_type)


def request_with_type(request_type: object, body: Mapping[str, Any] | JsonValue) -> ChatRequest:
    """Build a Rust-backed request with the given wire format."""
    normalized = request_type_value(request_type)
    chat_request = _load_native().ChatRequest
    if normalized == "openai_chat":
        return cast("ChatRequest", chat_request.openai_chat(body))
    if normalized == "openai_responses":
        return cast("ChatRequest", chat_request.openai_responses(body))
    if normalized == "anthropic":
        return cast("ChatRequest", chat_request.anthropic(body))
    raise ValueError(f"Unknown request type: {request_type!r}")


def response_type_value(value: object) -> str:
    """Normalize Rust response-format tags and wire strings."""
    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, str):
        raise TypeError(f"Response type must be a string-like value, got {type(raw).__name__}")
    legacy_aliases = {
        "completion": "openai_completion",
        "stream": "openai_stream",
        "responses_api_completion": "openai_responses_completion",
        "responses_api_stream": "openai_responses_stream",
    }
    raw = legacy_aliases.get(raw, raw)
    if raw in {
        "openai_completion",
        "openai_stream",
        "openai_responses_completion",
        "openai_responses_stream",
        "anthropic_completion",
        "anthropic_stream",
    }:
        return raw
    raise ValueError(f"Unknown response type: {value!r}")


def response_type_enum(value: object) -> _ChatResponseType:
    """Normalize a response-format tag to the Rust-bound enum object."""
    normalized = response_type_value(value)
    response_type = _load_native().ChatResponseType
    if normalized == "openai_completion":
        return response_type.OPENAI_COMPLETION
    if normalized == "openai_stream":
        return response_type.OPENAI_STREAM
    if normalized == "openai_responses_completion":
        return response_type.OPENAI_RESPONSES_COMPLETION
    if normalized == "openai_responses_stream":
        return response_type.OPENAI_RESPONSES_STREAM
    if normalized == "anthropic_completion":
        return response_type.ANTHROPIC_COMPLETION
    if normalized == "anthropic_stream":
        return response_type.ANTHROPIC_STREAM
    raise ValueError(f"Unknown response type: {value!r}")


def response_type_matches(response: object, response_type: object) -> bool:
    """Return whether a Rust-backed response has the requested wire shape."""
    return response_type_value(cast(Any, response).response_type) == response_type_value(response_type)


def response_with_type(
    response_type: object,
    body_or_stream: Mapping[str, Any] | JsonValue | object,
) -> ChatResponse:
    """Build a Rust-backed response with the given wire shape."""
    normalized = response_type_value(response_type)
    chat_response = _load_native().ChatResponse
    if normalized == "openai_completion":
        return cast("ChatResponse", chat_response.openai_completion(body_or_stream))
    if normalized == "openai_stream":
        return cast("ChatResponse", chat_response.openai_stream(body_or_stream))
    if normalized == "openai_responses_completion":
        return cast("ChatResponse", chat_response.openai_responses_completion(body_or_stream))
    if normalized == "openai_responses_stream":
        return cast("ChatResponse", chat_response.openai_responses_stream(body_or_stream))
    if normalized == "anthropic_completion":
        return cast("ChatResponse", chat_response.anthropic_completion(body_or_stream))
    if normalized == "anthropic_stream":
        return cast("ChatResponse", chat_response.anthropic_stream(body_or_stream))
    raise ValueError(f"Unknown response type: {response_type!r}")


def response_type_for_request_type(
    request_type: object,
    *,
    stream: bool,
) -> _ChatResponseType:
    """Return the response shape that corresponds to a request format."""
    normalized = request_type_value(request_type)
    if normalized == "openai_chat":
        return response_type_enum("openai_stream" if stream else "openai_completion")
    if normalized == "openai_responses":
        return response_type_enum(
            "openai_responses_stream" if stream else "openai_responses_completion"
        )
    if normalized == "anthropic":
        return response_type_enum("anthropic_stream" if stream else "anthropic_completion")
    raise ValueError(f"Unknown request type: {request_type!r}")


def response_matches_request_type(
    response: object,
    request_type: object,
) -> bool:
    """Return whether a response's provider format matches a request format."""
    response_type = response_type_value(cast(Any, response).response_type)
    request_type_normalized = request_type_value(request_type)
    if request_type_normalized == "openai_chat":
        return response_type in {"openai_completion", "openai_stream"}
    if request_type_normalized == "openai_responses":
        return response_type in {
            "openai_responses_completion",
            "openai_responses_stream",
        }
    if request_type_normalized == "anthropic":
        return response_type in {"anthropic_completion", "anthropic_stream"}
    raise ValueError(f"Unknown request type: {request_type!r}")


def response_is_streaming(response: object) -> bool:
    """Return whether a response carries a live stream."""
    return response_type_value(cast(Any, response).response_type).endswith("_stream")


__all__ = [
    "ChatRequest",
    "ChatRequestType",
    "ChatResponse",
    "ChatResponseStream",
    "ChatResponseType",
    "LLMBackend",
    "ProxyMetadata",
    "ProxyContext",
    "SessionCache",
    "Switchyard",
    "session_key_from_body",
    "SwitchyardBackendError",
    "SwitchyardConfigError",
    "SwitchyardContextPoolExhaustedError",
    "SwitchyardContextWindowExceededError",
    "SwitchyardDuplicateRegistrationError",
    "SwitchyardInvalidIdError",
    "SwitchyardInvalidRequestError",
    "SwitchyardModelNotFoundError",
    "SwitchyardProcessorError",
    "SwitchyardRuntimeError",
    "SwitchyardUnsupportedRequestTypeError",
    "SwitchyardUpstreamError",
    "request_type_enum",
    "request_type_matches",
    "request_type_value",
    "request_with_type",
    "response_is_streaming",
    "response_matches_request_type",
    "response_type_enum",
    "response_type_for_request_type",
    "response_type_matches",
    "response_type_value",
    "response_with_type",
]

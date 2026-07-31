# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python core values and compatibility chain wrappers."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from abc import ABCMeta
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, cast

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ChatRequestType(Enum):
    """Wire format carried by a :class:`ChatRequest`."""

    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"

    def __str__(self) -> str:
        return self.value


class ChatResponseType(Enum):
    """Wire format and delivery mode carried by a :class:`ChatResponse`."""

    OPENAI_COMPLETION = "openai_completion"
    OPENAI_STREAM = "openai_stream"
    OPENAI_RESPONSES_COMPLETION = "openai_responses_completion"
    OPENAI_RESPONSES_STREAM = "openai_responses_stream"
    ANTHROPIC_COMPLETION = "anthropic_completion"
    ANTHROPIC_STREAM = "anthropic_stream"

    def __str__(self) -> str:
        return self.value


def _json_copy(value: object) -> JsonValue:
    """Normalize an object to an owned JSON value."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
    else:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
    try:
        return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error


class ChatRequest:
    """Owned provider request body with an explicit wire format."""

    def __init__(self, request_type: ChatRequestType, body: object) -> None:
        self.request_type = request_type
        self._body = _json_copy(body)

    @classmethod
    def openai_chat(cls, body: object) -> ChatRequest:
        """Build an OpenAI Chat Completions request."""
        return cls(ChatRequestType.OPENAI_CHAT, body)

    @classmethod
    def openai_responses(cls, body: object) -> ChatRequest:
        """Build an OpenAI Responses request."""
        return cls(ChatRequestType.OPENAI_RESPONSES, body)

    @classmethod
    def anthropic(cls, body: object) -> ChatRequest:
        """Build an Anthropic Messages request."""
        return cls(ChatRequestType.ANTHROPIC, body)

    @property
    def body(self) -> Any:
        """Return an owned copy of the request body."""
        return _json_copy(self._body)

    @property
    def model(self) -> str | None:
        """Return the request model when it is a string."""
        if isinstance(self._body, dict):
            model = self._body.get("model")
            return model if isinstance(model, str) else None
        return None

    def validate(self) -> None:
        """Reject semantically invalid message-based requests."""
        if self.request_type in {ChatRequestType.OPENAI_CHAT, ChatRequestType.ANTHROPIC}:
            if isinstance(self._body, dict) and self._body.get("messages") == []:
                raise _load_native().SwitchyardInvalidRequestError(
                    "messages must be a non-empty array"
                )

    def set_model(self, model: str) -> None:
        """Set the request model, replacing a malformed non-object body."""
        if not isinstance(self._body, dict):
            self._body = {}
        self._body["model"] = model

    def replace_body(self, body: object) -> None:
        """Replace the body without changing its wire format."""
        self._body = _json_copy(body)

    def to_body(self) -> JsonValue:
        """Return an owned copy of the request body."""
        return _json_copy(self._body)

    def __repr__(self) -> str:
        model = f", model={self.model!r}" if self.model is not None else ""
        return f"ChatRequest(request_type={self.request_type.value!r}{model})"


class ChatResponseStream:
    """Single-consumer async response stream with callback transforms."""

    def __init__(self, source: object) -> None:
        self._native = _load_native()._NativeChatResponseStream(source)

    @classmethod
    def _from_native(cls, stream: Any) -> ChatResponseStream:
        instance = cls.__new__(cls)
        instance._native = stream
        return instance

    def tap(self, callback: object) -> ChatResponseStream:
        """Observe each event before mapping."""
        self._native.tap(callback)
        return self

    def map(self, callback: object) -> ChatResponseStream:
        """Transform each event in registration order."""
        self._native.map(callback)
        return self

    def on_complete(self, callback: object) -> ChatResponseStream:
        """Run a callback once after normal stream exhaustion."""
        self._native.on_complete(callback)
        return self

    def __aiter__(self) -> ChatResponseStream:
        self._native.__aiter__()
        return self

    async def __anext__(self) -> Any:
        return await self._native.__anext__()

    async def aclose(self) -> None:
        """Close the upstream source and release its connection."""
        await self._native.aclose()

    def __repr__(self) -> str:
        return "ChatResponseStream()"


class ChatResponse:
    """Owned buffered response or live response stream."""

    def __init__(
        self,
        response_type: ChatResponseType,
        *,
        body: object | None = None,
        stream: object | None = None,
    ) -> None:
        self.response_type = response_type
        self._body: JsonValue | None
        self._stream: ChatResponseStream | None
        if response_type.value.endswith("_stream"):
            if stream is None:
                raise TypeError("streaming ChatResponse requires a stream")
            self._stream = (
                stream if isinstance(stream, ChatResponseStream) else ChatResponseStream(stream)
            )
            self._body = None
        else:
            self._body = _json_copy(body)
            self._stream = None

    @classmethod
    def openai_completion(cls, body: object) -> ChatResponse:
        """Build a buffered OpenAI Chat Completions response."""
        return cls(ChatResponseType.OPENAI_COMPLETION, body=body)

    @classmethod
    def openai_stream(cls, stream: object) -> ChatResponse:
        """Build a streaming OpenAI Chat Completions response."""
        return cls(ChatResponseType.OPENAI_STREAM, stream=stream)

    @classmethod
    def openai_responses_completion(cls, body: object) -> ChatResponse:
        """Build a buffered OpenAI Responses response."""
        return cls(ChatResponseType.OPENAI_RESPONSES_COMPLETION, body=body)

    @classmethod
    def openai_responses_stream(cls, stream: object) -> ChatResponse:
        """Build a streaming OpenAI Responses response."""
        return cls(ChatResponseType.OPENAI_RESPONSES_STREAM, stream=stream)

    @classmethod
    def anthropic_completion(cls, body: object) -> ChatResponse:
        """Build a buffered Anthropic response."""
        return cls(ChatResponseType.ANTHROPIC_COMPLETION, body=body)

    @classmethod
    def anthropic_stream(cls, stream: object) -> ChatResponse:
        """Build a streaming Anthropic response."""
        return cls(ChatResponseType.ANTHROPIC_STREAM, stream=stream)

    @property
    def body(self) -> Any:
        """Return an owned buffered body."""
        if self._stream is not None:
            raise AttributeError("streaming ChatResponse values do not have a buffered body")
        return _json_copy(self._body)

    @property
    def stream(self) -> ChatResponseStream:
        """Return the live stream."""
        if self._stream is None:
            raise AttributeError("buffered ChatResponse values do not have a stream")
        return self._stream

    def replace_body(self, body: object) -> None:
        """Replace a buffered body without changing its wire shape."""
        if self._stream is not None:
            raise ValueError("streaming ChatResponse values do not have a replaceable body")
        self._body = _json_copy(body)

    def to_body(self) -> JsonValue:
        """Return an owned buffered body."""
        if self._stream is not None:
            raise AttributeError("streaming ChatResponse values do not have a buffered body")
        return _json_copy(self._body)

    def __repr__(self) -> str:
        return f"ChatResponse(response_type={self.response_type.value!r})"


class ProxyMetadata(dict[str, Any]):
    """Mutable per-request metadata owned by Python."""


class ProxyContext:
    """Per-request state shared by Python and native components."""

    def __init__(
        self,
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        self._native = _load_native()._NativeProxyContext(metadata, request_id)
        self.metadata = ProxyMetadata(metadata or {})

    @property
    def request_id(self) -> str | None:
        return cast(str | None, self._native.request_id)

    @request_id.setter
    def request_id(self, value: str | None) -> None:
        self._native.request_id = value

    @property
    def inbound_format(self) -> ChatRequestType | None:
        value = self._native.inbound_format
        return request_type_enum(value) if value is not None else None

    @inbound_format.setter
    def inbound_format(self, value: object | None) -> None:
        self._native.inbound_format = value

    @property
    def selected_model(self) -> str | None:
        return cast(str | None, self._native.selected_model)

    @selected_model.setter
    def selected_model(self, value: str | None) -> None:
        self._native.selected_model = value

    @property
    def selected_target(self) -> str | None:
        return cast(str | None, self._native.selected_target)

    @selected_target.setter
    def selected_target(self, value: str | None) -> None:
        self._native.selected_target = value

    @property
    def evicted_targets(self) -> list[str] | None:
        return cast(list[str] | None, self._native.evicted_targets)

    @evicted_targets.setter
    def evicted_targets(self, value: list[str] | None) -> None:
        self._native.evicted_targets = value

    @property
    def backend_call_latency_ms(self) -> float | None:
        return cast(float | None, self._native.backend_call_latency_ms)

    @backend_call_latency_ms.setter
    def backend_call_latency_ms(self, value: float | None) -> None:
        self._native.backend_call_latency_ms = value

    def __repr__(self) -> str:
        return repr(self._native)


class LLMBackend(metaclass=ABCMeta):  # noqa: B024
    """Base class for Python and registered native LLM backends."""

    def __new__(cls, *args: object, **kwargs: object) -> LLMBackend:
        if cls is LLMBackend:
            raise TypeError("can't instantiate abstract role LLMBackend")
        return super().__new__(cls)

    @property
    def supported_request_types(self) -> list[ChatRequestType]:
        """Return request formats accepted by this backend."""
        raise NotImplementedError("LLMBackend.supported_request_types must be implemented")

    async def startup(self) -> None:
        """Start backend resources."""
        return None

    async def shutdown(self) -> None:
        """Stop backend resources."""
        return None

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        """Call the upstream model."""
        raise NotImplementedError("LLMBackend.call must be implemented")


class _NativeModule(Protocol):
    _NativeChatResponseStream: Any
    _NativeProxyContext: Any
    is_subagent_request: Any
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
            "The Switchyard native extension is required. Run `uv run maturin develop` "
            "or install a built switchyard wheel."
        ) from exc


if TYPE_CHECKING:
    class SwitchyardRuntimeError(RuntimeError):
        """Base class for native Switchyard runtime errors."""

    class SwitchyardConfigError(SwitchyardRuntimeError):
        """Raised for invalid Switchyard configuration."""

    class SwitchyardInvalidIdError(SwitchyardRuntimeError):
        """Raised when a Switchyard identifier is invalid."""

    class SwitchyardDuplicateRegistrationError(SwitchyardRuntimeError):
        """Raised when a registry receives a duplicate ID."""

    class SwitchyardModelNotFoundError(SwitchyardRuntimeError):
        """Raised when a route table cannot find a model."""

    class SwitchyardUnsupportedRequestTypeError(SwitchyardRuntimeError):
        """Raised when a component rejects a request format."""

    class SwitchyardInvalidRequestError(SwitchyardRuntimeError):
        """Raised when a request body fails semantic validation."""

    class SwitchyardProcessorError(SwitchyardRuntimeError):
        """Raised when a processor fails."""

    class SwitchyardBackendError(SwitchyardRuntimeError):
        """Raised when a backend fails."""

    class SwitchyardUpstreamError(SwitchyardRuntimeError):
        """Raised when an upstream call fails."""

        status_code: int
        body: str

    class SwitchyardContextWindowExceededError(SwitchyardBackendError):
        """Raised when an upstream request exceeds the context window."""

    class SwitchyardContextPoolExhaustedError(SwitchyardBackendError):
        """Raised when every attempted routing target was evicted."""
else:
    SwitchyardRuntimeError = _load_native().SwitchyardRuntimeError
    SwitchyardConfigError = _load_native().SwitchyardConfigError
    SwitchyardInvalidIdError = _load_native().SwitchyardInvalidIdError
    SwitchyardDuplicateRegistrationError = _load_native().SwitchyardDuplicateRegistrationError
    SwitchyardModelNotFoundError = _load_native().SwitchyardModelNotFoundError
    SwitchyardUnsupportedRequestTypeError = _load_native().SwitchyardUnsupportedRequestTypeError
    SwitchyardInvalidRequestError = _load_native().SwitchyardInvalidRequestError
    SwitchyardProcessorError = _load_native().SwitchyardProcessorError
    SwitchyardBackendError = _load_native().SwitchyardBackendError
    SwitchyardUpstreamError = _load_native().SwitchyardUpstreamError
    SwitchyardContextWindowExceededError = _load_native().SwitchyardContextWindowExceededError
    SwitchyardContextPoolExhaustedError = _load_native().SwitchyardContextPoolExhaustedError


def is_subagent_request(headers: Mapping[str, str]) -> bool:
    """Return whether canonical protocol metadata marks delegated agent work."""
    return bool(_load_native().is_subagent_request(headers))


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
        if not isinstance(backend, LLMBackend):
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
        context = ctx if ctx is not None else ProxyContext()
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
            response: object = await cast(Any, self._backend).call(ctx, request)
        except native.SwitchyardContextWindowExceededError:
            raise
        except Exception as error:
            raise _backend_error(error) from error
        if not isinstance(response, ChatResponse):
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
            if not isinstance(current, ChatRequest):
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
            if not isinstance(current, ChatResponse):
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


def request_type_value(value: object) -> str:
    """Normalize request-format tags and wire strings."""
    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, str):
        raise TypeError(f"Request type must be a string-like value, got {type(raw).__name__}")
    if raw == "anthropic_messages":
        return "anthropic"
    if raw in {"openai_chat", "openai_responses", "anthropic"}:
        return raw
    raise ValueError(f"Unknown request type: {value!r}")


def request_type_enum(value: object) -> ChatRequestType:
    """Normalize a request-format tag to the Python enum."""
    return ChatRequestType(request_type_value(value))


def request_type_matches(request: object, request_type: object) -> bool:
    """Return whether a request has the requested wire format."""
    return request_type_value(cast(Any, request).request_type) == request_type_value(request_type)


def request_with_type(request_type: object, body: Mapping[str, Any] | JsonValue) -> ChatRequest:
    """Build a request with the given wire format."""
    normalized = request_type_value(request_type)
    if normalized == "openai_chat":
        return ChatRequest.openai_chat(body)
    if normalized == "openai_responses":
        return ChatRequest.openai_responses(body)
    if normalized == "anthropic":
        return ChatRequest.anthropic(body)
    raise ValueError(f"Unknown request type: {request_type!r}")


def response_type_value(value: object) -> str:
    """Normalize response-format tags and wire strings."""
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


def response_type_enum(value: object) -> ChatResponseType:
    """Normalize a response-format tag to the Python enum."""
    return ChatResponseType(response_type_value(value))


def response_type_matches(response: object, response_type: object) -> bool:
    """Return whether a Rust-backed response has the requested wire shape."""
    return response_type_value(cast(Any, response).response_type) == response_type_value(response_type)


def response_with_type(
    response_type: object,
    body_or_stream: Mapping[str, Any] | JsonValue | object,
) -> ChatResponse:
    """Build a response with the given wire shape."""
    normalized = response_type_value(response_type)
    if normalized == "openai_completion":
        return ChatResponse.openai_completion(body_or_stream)
    if normalized == "openai_stream":
        return ChatResponse.openai_stream(body_or_stream)
    if normalized == "openai_responses_completion":
        return ChatResponse.openai_responses_completion(body_or_stream)
    if normalized == "openai_responses_stream":
        return ChatResponse.openai_responses_stream(body_or_stream)
    if normalized == "anthropic_completion":
        return ChatResponse.anthropic_completion(body_or_stream)
    if normalized == "anthropic_stream":
        return ChatResponse.anthropic_stream(body_or_stream)
    raise ValueError(f"Unknown response type: {response_type!r}")


def response_type_for_request_type(
    request_type: object,
    *,
    stream: bool,
) -> ChatResponseType:
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
    "Switchyard",
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
    "is_subagent_request",
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

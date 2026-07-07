# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-side Switchyard wrappers over the Rust translation engine."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from openai.types.chat import ChatCompletionChunk

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.roles import TranslatedResponse, TranslatedStream
    from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse


class _NativeStreamTranslation(Protocol):
    def translate_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]: ...
    def finish(self) -> list[dict[str, Any]]: ...


class _NativeTranslationEngine(Protocol):
    def translate_request(
        self,
        source: str,
        target: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def translate_response(
        self,
        source: str,
        target: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def stream(
        self,
        source: str,
        target: str,
        model: str | None = None,
        message_id: str | None = None,
    ) -> _NativeStreamTranslation: ...

    def normalize_anthropic_tool_use_ids(self, messages: object) -> object: ...


class _NativeModule(Protocol):
    TranslationEngine: type[_NativeTranslationEngine]


_T = TypeVar("_T")
_NativeFormat = Literal[
    "openai_chat", "openai_responses", "anthropic_messages", "gemini_generate_content"
]
_StreamOutput = Literal["objects", "chat_chunks", "responses_sse"]
_native: _NativeModule | None = None
_log = logging.getLogger(__name__)


def _load_native() -> _NativeModule:
    global _native
    if _native is None:
        try:
            _native = cast(
                _NativeModule,
                importlib.import_module("switchyard_rust._switchyard_rust"),
            )
        except ImportError as exc:  # pragma: no cover - broken install guard
            raise RuntimeError(
                "Rust translation extension is required. Run `uv run maturin develop` "
                "or install a built switchyard wheel."
            ) from exc
    return _native


class TranslationEngine:
    """Single Python-facing engine for request, response, and stream translation."""

    def __init__(self) -> None:
        self._inner = _load_native().TranslationEngine()

    def translate_request(
        self,
        source: str | ChatRequestType,
        target: str | ChatRequestType,
        body: Mapping[str, Any] | object,
    ) -> dict[str, Any]:
        """Translate a JSON request body between provider wire formats."""
        return self._inner.translate_request(
            _format_name(source),
            _format_name(target),
            _jsonable_mapping(body),
        )

    def translate_response(
        self,
        source: str | ChatRequestType,
        target: str | ChatRequestType,
        body: Mapping[str, Any] | object,
    ) -> dict[str, Any]:
        """Translate a JSON response body between provider wire formats."""
        return self._inner.translate_response(
            _format_name(source),
            _format_name(target),
            _jsonable_mapping(body),
        )

    def request_to(
        self,
        target: str | ChatRequestType,
        request: ChatRequest,
    ) -> ChatRequest:
        """Return *request* in the target wire format."""
        source = _request_format(request)
        target_format = _format_name(target)
        if source == target_format:
            return request
        return _wrap_request(
            target_format,
            self.translate_request(source, target_format, request.body),
        )

    def request_to_any_of(
        self,
        request: ChatRequest,
        supported: Sequence[ChatRequestType],
    ) -> ChatRequest:
        """Passthrough when possible, otherwise translate to the first supported type."""
        if not supported:
            raise ValueError("supported must be non-empty")
        source = _request_format(request)
        supported_formats = [_format_name(item) for item in supported]
        if source in supported_formats:
            return request
        return self.request_to(supported_formats[0], request)

    def response_to(
        self,
        target: str | ChatRequestType,
        response: ChatResponse,
        *,
        original_body: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        """Return a ChatResponse wrapper in the target wire format."""
        source = _response_format(response)
        target_format = _format_name(target)
        if source == target_format:
            return response
        if _is_streaming_response(response):
            return _wrap_streaming_response(
                target_format,
                self._translate_response_stream(source, target_format, response, original_body),
            )
        return _wrap_response(
            target_format,
            self.translate_response(source, target_format, _response_body(response)),
        )

    def response_for_request(
        self,
        request: ChatRequest,
        response: ChatResponse,
    ) -> TranslatedResponse:
        """Translate a backend response to the original client's wire format."""
        if _is_streaming_response(response):
            return self.stream_for_request(request, response)
        source = _response_format(response)
        target = _request_format(request)
        if source == target:
            return cast("TranslatedResponse", _response_body(response))
        return cast(
            "TranslatedResponse",
            self.translate_response(source, target, _response_body(response)),
        )

    def stream_for_request(
        self,
        request: ChatRequest,
        response: ChatResponse,
    ) -> TranslatedStream:
        """Translate a backend stream to the original client's stream contract."""
        source = _response_format(response)
        target = _request_format(request)
        if source == target:
            return cast("TranslatedStream", _response_stream(response))
        return cast(
            "TranslatedStream",
            self._translate_response_stream(source, target, response, getattr(request, "body", None)),
        )

    async def translate(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        response: ChatResponse,
    ) -> TranslatedResponse:
        """Implement Switchyard's terminal TranslationEngine role."""
        _ = ctx
        return self.response_for_request(request, response)

    async def translate_stream(
        self,
        source: str | ChatRequestType,
        target: str | ChatRequestType,
        stream: AsyncIterable[Any],
        *,
        model: str | None = None,
        message_id: str | None = None,
        output: _StreamOutput = "objects",
    ) -> AsyncGenerator[Any, None]:
        """Translate an async stream between provider event formats.

        Closes *stream* on every exit path. When the client disconnects, the
        ASGI server ``aclose()``-es the SSE generator consuming this one, which
        raises ``GeneratorExit`` at the suspended ``yield``; without the
        ``finally`` the upstream input stream (and the pooled connection it
        holds) would never be released.
        """
        target_format = _format_name(target)
        translator = self._inner.stream(
            _format_name(source),
            target_format,
            model,
            message_id,
        )
        try:
            async for event in stream:
                for payload in _stream_event_payloads(event):
                    for translated in translator.translate_event(payload):
                        yield _coerce_stream_output(target_format, translated, output)
            for translated in translator.finish():
                yield _coerce_stream_output(target_format, translated, output)
        finally:
            await _aclose_input_stream(stream)

    def _translate_response_stream(
        self,
        source: _NativeFormat,
        target: _NativeFormat,
        response: ChatResponse,
        original_body: Mapping[str, Any] | object | None,
    ) -> AsyncGenerator[Any, None]:
        return self.translate_stream(
            source,
            target,
            _response_stream(response),
            model=_model_from_body(original_body),
            output=_stream_wire_output_for_target(target),
        )

    def normalize_anthropic_tool_use_ids(self, messages: object) -> object:
        """Normalize Anthropic tool IDs without breaking result references."""
        return self._inner.normalize_anthropic_tool_use_ids(_jsonable(messages, set()))


async def _aclose_input_stream(stream: object) -> None:
    """Best-effort close of a translated stream's upstream input.

    ``ChatResponseStream`` and async generators expose ``aclose``; SDK
    ``AsyncStream`` objects expose ``close``; either may be a coroutine.
    Closing must never mask the control flow that triggered it, so failures
    are logged and swallowed.
    """
    closer = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        _log.debug("Failed to close translated input stream: %s: %s", type(exc).__name__, exc)


def is_native_translation_available() -> bool:
    """Return whether the required native translation extension loaded."""
    _load_native()
    return True


def _format_name(value: str | ChatRequestType) -> _NativeFormat:
    raw = value.value if hasattr(value, "value") else str(value)
    if raw == "anthropic":
        raw = "anthropic_messages"
    if raw == "gemini":
        raw = "gemini_generate_content"
    if raw in {
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
    }:
        return cast(_NativeFormat, raw)
    raise ValueError(f"Unknown translation format: {value!r}")


def _request_format(request: ChatRequest) -> _NativeFormat:
    from switchyard_rust.core import request_type_value

    try:
        return _format_name(request_type_value(request.request_type))
    except (AttributeError, ValueError) as exc:
        raise NotImplementedError(
            f"Request translation not implemented for {type(request).__name__}"
        ) from exc


def _response_format(response: ChatResponse) -> _NativeFormat:
    from switchyard_rust.core import response_type_value

    response_type = response_type_value(response.response_type)
    if response_type in {"openai_completion", "openai_stream"}:
        return "openai_chat"
    if response_type in {"openai_responses_completion", "openai_responses_stream"}:
        return "openai_responses"
    if response_type in {"anthropic_completion", "anthropic_stream"}:
        return "anthropic_messages"
    if response_type in {"gemini_completion", "gemini_stream"}:
        return "gemini_generate_content"
    raise NotImplementedError(
        f"Response translation not implemented for {type(response).__name__}"
    )


def _wrap_request(format_name: _NativeFormat, body: dict[str, Any]) -> ChatRequest:
    from switchyard_rust.core import request_with_type

    if format_name == "openai_chat":
        return request_with_type("openai_chat", body)
    if format_name == "openai_responses":
        return request_with_type("openai_responses", body)
    if format_name == "anthropic_messages":
        return request_with_type("anthropic", body)
    if format_name == "gemini_generate_content":
        return request_with_type("gemini", body)
    raise ValueError(f"Unknown request format: {format_name!r}")


def _wrap_response(format_name: _NativeFormat, body: dict[str, Any]) -> ChatResponse:
    from switchyard_rust.core import ChatResponse

    if format_name == "openai_chat":
        return ChatResponse.openai_completion(body)
    if format_name == "openai_responses":
        return ChatResponse.openai_responses_completion(body)
    if format_name == "anthropic_messages":
        return ChatResponse.anthropic_completion(body)
    if format_name == "gemini_generate_content":
        return ChatResponse.gemini_completion(body)
    raise ValueError(f"Unknown response format: {format_name!r}")


def _wrap_streaming_response(
    format_name: _NativeFormat,
    stream: AsyncIterable[Any],
) -> ChatResponse:
    from switchyard.lib.chat_response.anthropic import AnthropicResponseStream
    from switchyard.lib.chat_response.gemini import GeminiResponseStream
    from switchyard.lib.chat_response.openai_chat import ResponseStream
    from switchyard.lib.chat_response.openai_responses import ResponsesApiStream
    from switchyard_rust.core import ChatResponse

    if format_name == "openai_chat":
        return ChatResponse.openai_stream(ResponseStream(cast(Any, stream)))
    if format_name == "openai_responses":
        return ChatResponse.openai_responses_stream(ResponsesApiStream(cast(Any, stream)))
    if format_name == "anthropic_messages":
        return ChatResponse.anthropic_stream(AnthropicResponseStream(cast(Any, stream)))
    if format_name == "gemini_generate_content":
        return ChatResponse.gemini_stream(GeminiResponseStream(cast(Any, stream)))
    raise ValueError(f"Unknown streaming response format: {format_name!r}")


def _stream_wire_output_for_target(target: _NativeFormat) -> _StreamOutput:
    if target == "openai_chat":
        return "chat_chunks"
    if target == "openai_responses":
        return "responses_sse"
    return "objects"


def _model_from_body(body: Mapping[str, Any] | object | None) -> str | None:
    if isinstance(body, Mapping):
        model = body.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def _response_body(response: ChatResponse) -> Any:
    return response.body


def _response_stream(response: ChatResponse) -> AsyncIterable[Any]:
    return cast(AsyncIterable[Any], response.stream)


def _is_streaming_response(response: ChatResponse) -> bool:
    from switchyard_rust.core import response_is_streaming

    return response_is_streaming(response)


def _coerce_stream_output(
    target: _NativeFormat,
    event: Mapping[str, Any],
    output: _StreamOutput,
) -> Any:
    if output == "responses_sse":
        return _sse_frame(event)
    if output == "chat_chunks":
        return _validate_or_construct(ChatCompletionChunk, dict(event))
    _ = target
    return dict(event)


def _validate_or_construct(model_cls: type[_T], body: dict[str, Any]) -> _T:
    validator = getattr(model_cls, "model_validate", None)
    if callable(validator):
        try:
            return cast(_T, validator(body))
        except Exception:
            pass
    constructor = getattr(model_cls, "model_construct", None)
    if callable(constructor):
        return cast(_T, constructor(**body))
    return model_cls(**body)


def _sse_frame(event: Mapping[str, Any]) -> str:
    event_type = event.get("type", "message")
    return f"event: {event_type}\ndata: {json.dumps(dict(event))}\n\n"


def _jsonable_mapping(value: Any) -> dict[str, Any]:
    value = _jsonable(value, set())
    return dict(value) if isinstance(value, Mapping) else {}


def _stream_event_payloads(event: Any) -> list[dict[str, Any]]:
    """JSON payload(s) of one stream item, for cross-format translation.

    Backends that preserve wire fidelity yield raw SSE frame *strings*
    (see ``RawSSEFrameStream``); those are parsed here so the translator
    still sees event dicts. Every other item shape keeps the existing
    jsonable-mapping coercion.
    """
    if isinstance(event, str):
        return sse_frame_payloads(event)
    return [_jsonable_mapping(event)]


def sse_frame_payloads(frame: str) -> list[dict[str, Any]]:
    """Parse an SSE frame string into its JSON ``data:`` payload(s).

    Follows the SSE contract: a frame's ``data:`` lines are joined with
    newlines to form one payload; an optional single leading space after the
    colon is stripped. Comment/keep-alive frames, the ``[DONE]`` sentinel,
    and non-JSON or non-object payloads yield nothing. Accepts a string
    containing multiple ``\\n\\n``-separated frames and returns payloads in
    order.
    """
    payloads: list[dict[str, Any]] = []
    for block in frame.split("\n\n"):
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _jsonable(value: Any, seen: set[int]) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(exclude_none=True), seen)
        except TypeError:
            return _jsonable(value.model_dump(), seen)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict(), seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), seen)
    if isinstance(value, Mapping):
        obj_id = id(value)
        if obj_id in seen:
            return str(value)
        seen.add(obj_id)
        try:
            if any(id(item) in seen for item in value.values()):
                return str(value)
            return {str(key): _jsonable(item, seen) for key, item in value.items()}
        finally:
            seen.remove(obj_id)
    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        if obj_id in seen:
            return str(value)
        seen.add(obj_id)
        try:
            if any(id(item) in seen for item in value):
                return str(value)
            return [_jsonable(item, seen) for item in value]
        finally:
            seen.remove(obj_id)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item, seen) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        obj_id = id(value)
        if obj_id in seen:
            return str(value)
        seen.add(obj_id)
        try:
            return {
                key: _jsonable(item, seen)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        finally:
            seen.remove(obj_id)
    return value


__all__ = [
    "TranslationEngine",
    "is_native_translation_available",
]

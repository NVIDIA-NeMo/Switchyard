# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar, Protocol, final

class WireFormat:
    OPENAI_CHAT: ClassVar[WireFormat]
    ANTHROPIC_MESSAGES: ClassVar[WireFormat]
    OPENAI_RESPONSES: ClassVar[WireFormat]

    value: str

class FormatId:
    value: str

    def __init__(self, value: str) -> None: ...
    @classmethod
    def known(cls, value: WireFormat) -> FormatId: ...

class Role:
    SYSTEM: ClassVar[Role]
    DEVELOPER: ClassVar[Role]
    USER: ClassVar[Role]
    ASSISTANT: ClassVar[Role]
    TOOL: ClassVar[Role]

    value: str

class StopReason:
    END_TURN: ClassVar[StopReason]
    MAX_TOKENS: ClassVar[StopReason]
    TOOL_USE: ClassVar[StopReason]
    CONTENT_FILTER: ClassVar[StopReason]
    ERROR: ClassVar[StopReason]
    UNKNOWN: ClassVar[StopReason]

    value: str

class ImageSource:
    class Url:
        url: str
        detail: str | None

        def __init__(self, url: str, detail: str | None) -> None: ...

    class Base64:
        media_type: str | None
        data: str

        def __init__(self, media_type: str | None, data: str) -> None: ...

    class Raw:
        value: Any

        def __init__(self, value: Any) -> None: ...

ImageSourceValue = ImageSource.Url | ImageSource.Base64 | ImageSource.Raw

class FileSource:
    class FileId:
        id: str

        def __init__(self, id: str) -> None: ...

    class FileData:
        data: str
        filename: str | None

        def __init__(self, data: str, filename: str | None) -> None: ...

    class Raw:
        value: Any

        def __init__(self, value: Any) -> None: ...

FileSourceValue = FileSource.FileId | FileSource.FileData | FileSource.Raw

class MediaSource:
    class Url:
        url: str
        media_type: str | None

        def __init__(self, url: str, media_type: str | None) -> None: ...

    class Base64:
        media_type: str | None
        data: str

        def __init__(self, media_type: str | None, data: str) -> None: ...

    class Raw:
        value: Any

        def __init__(self, value: Any) -> None: ...

MediaSourceValue = MediaSource.Url | MediaSource.Base64 | MediaSource.Raw

class ToolCall:
    id: str
    name: str
    arguments: Any

    def __init__(self, id: str, name: str, arguments: Any) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ContentBlock:
    class Text:
        text: str

        def __init__(self, text: str) -> None: ...

    class Reasoning:
        text: str
        signature: str | None

        def __init__(self, text: str, signature: str | None) -> None: ...

    class Image:
        source: ImageSourceValue

        def __init__(self, source: ImageSourceValue) -> None: ...

    class Audio:
        source: MediaSourceValue

        def __init__(self, source: MediaSourceValue) -> None: ...

    class Video:
        source: MediaSourceValue

        def __init__(self, source: MediaSourceValue) -> None: ...

    class File:
        source: FileSourceValue

        def __init__(self, source: FileSourceValue) -> None: ...

    class ToolCallBlock:
        tool_call: ToolCall

        def __init__(self, tool_call: ToolCall) -> None: ...

    class ToolResultBlock:
        tool_result: ToolResult

        def __init__(self, tool_result: ToolResult) -> None: ...

    class Refusal:
        text: str

        def __init__(self, text: str) -> None: ...

    class Unknown:
        provider: FormatId
        raw: Any

        def __init__(self, provider: FormatId, raw: Any) -> None: ...

ContentBlockValue = (
    ContentBlock.Text
    | ContentBlock.Reasoning
    | ContentBlock.Image
    | ContentBlock.Audio
    | ContentBlock.Video
    | ContentBlock.File
    | ContentBlock.ToolCallBlock
    | ContentBlock.ToolResultBlock
    | ContentBlock.Refusal
    | ContentBlock.Unknown
)

class ToolResult:
    tool_call_id: str
    content: list[ContentBlockValue]
    is_error: bool | None

    def __init__(
        self,
        tool_call_id: str,
        content: Sequence[ContentBlockValue],
        *,
        is_error: bool | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class InstructionBlock:
    role: Role
    content: list[ContentBlockValue]

    def __init__(self, role: Role, content: Sequence[ContentBlockValue]) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Message:
    role: Role
    content: list[ContentBlockValue]

    def __init__(self, role: Role, content: Sequence[ContentBlockValue]) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ToolDefinition:
    name: str
    description: str | None
    parameters: Any
    strict: bool | None

    def __init__(
        self,
        name: str,
        parameters: Any,
        *,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ToolChoice:
    kind: str
    name: str | None
    raw_value: Any | None

    @classmethod
    def auto(cls) -> ToolChoice: ...
    @classmethod
    def required(cls) -> ToolChoice: ...
    @classmethod
    def none(cls) -> ToolChoice: ...
    @classmethod
    def tool(cls, name: str) -> ToolChoice: ...
    @classmethod
    def raw(cls, value: Any) -> ToolChoice: ...
    def to_dict(self) -> Any: ...

class SamplingParams:
    temperature: float | None
    top_p: float | None
    top_k: int | None

    def __init__(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class OutputParams:
    max_output_tokens: int | None
    response_format: Any | None

    def __init__(
        self,
        *,
        max_output_tokens: int | None = None,
        response_format: Any | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ReasoningParams:
    effort: str | None
    raw: Any | None

    def __init__(self, *, effort: str | None = None, raw: Any | None = None) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ProviderExtensions:
    fields: dict[str, Any]

    def __init__(self, fields: Mapping[str, Any] | None = None) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class PreservationMetadata:
    requests: dict[str, Any]
    responses: dict[str, Any]

    def __init__(
        self,
        *,
        requests: Mapping[str, Any] | None = None,
        responses: Mapping[str, Any] | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class LlmRequest:
    model: str | None
    instructions: list[InstructionBlock]
    messages: list[Message]
    tools: list[ToolDefinition]
    tool_choice: ToolChoice | None
    sampling: SamplingParams
    output: OutputParams
    reasoning: ReasoningParams
    stream: bool
    extensions: ProviderExtensions
    preservation: PreservationMetadata

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: Sequence[InstructionBlock] | None = None,
        messages: Sequence[Message] | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        sampling: SamplingParams | None = None,
        output: OutputParams | None = None,
        reasoning: ReasoningParams | None = None,
        stream: bool = False,
        extensions: ProviderExtensions | None = None,
        preservation: PreservationMetadata | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None

    def __init__(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ResponseOutput:
    role: Role
    content: list[ContentBlockValue]
    stop_reason: StopReason | None

    def __init__(
        self,
        role: Role,
        content: Sequence[ContentBlockValue],
        *,
        stop_reason: StopReason | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class AggLlmResponse:
    id: str | None
    model: str | None
    outputs: list[ResponseOutput]
    usage: Usage
    extensions: ProviderExtensions
    preservation: PreservationMetadata

    def __init__(
        self,
        *,
        id: str | None = None,
        model: str | None = None,
        outputs: Sequence[ResponseOutput] | None = None,
        usage: Usage | None = None,
        extensions: ProviderExtensions | None = None,
        preservation: PreservationMetadata | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class LlmResponseChunk:
    class MessageStart:
        id: str | None
        model: str | None

        def __init__(self, id: str | None, model: str | None) -> None: ...

    class TextDelta:
        index: int
        text: str

        def __init__(self, index: int, text: str) -> None: ...

    class ReasoningDelta:
        index: int
        text: str

        def __init__(self, index: int, text: str) -> None: ...

    class ToolCallDelta:
        index: int
        id: str | None
        name: str | None
        arguments_delta: str | None

        def __init__(
            self,
            index: int,
            id: str | None,
            name: str | None,
            arguments_delta: str | None,
        ) -> None: ...

    class UsageUpdate:
        usage: Usage

        def __init__(self, usage: Usage) -> None: ...

    class MessageStop:
        reason: str | None

        def __init__(self, reason: str | None) -> None: ...

    class Error:
        message: str

        def __init__(self, message: str) -> None: ...

LlmResponseChunkValue = (
    LlmResponseChunk.MessageStart
    | LlmResponseChunk.TextDelta
    | LlmResponseChunk.ReasoningDelta
    | LlmResponseChunk.ToolCallDelta
    | LlmResponseChunk.UsageUpdate
    | LlmResponseChunk.MessageStop
    | LlmResponseChunk.Error
)

class Context:
    values: dict[str, str]

    def __init__(self, *, values: Mapping[str, str] | None = None) -> None: ...

@final
class Decision:
    selected_model: str
    reasoning: str | None

class RoutedLlmClient(Protocol):
    async def call(
        self,
        context: Context,
        request: Request,
        decision: Decision,
        /,
    ) -> Response: ...

class Metadata:
    session_id: str | None
    agent_id: str | None
    task_id: str | None
    correlation_id: str | None
    extra_metadata: dict[str, str] | None
    http_headers: dict[str, str] | None
    wire_format: WireFormat | None

    def __init__(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        correlation_id: str | None = None,
        extra_metadata: Mapping[str, str] | None = None,
        http_headers: Mapping[str, str] | None = None,
        wire_format: WireFormat | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Request:
    llm_request: LlmRequest
    raw_request: Any | None
    metadata: Metadata | None
    requested_model: str | None

    def __init__(
        self,
        llm_request: LlmRequest,
        *,
        raw_request: Any | None = None,
        metadata: Metadata | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class LlmResponseStream(AsyncIterator[LlmResponseChunkValue]):
    def __aiter__(self) -> LlmResponseStream: ...
    async def __anext__(self) -> LlmResponseChunkValue: ...
    async def aclose(self) -> None: ...

class Response:
    is_streaming: bool
    selected_model: str | None
    aggregate: AggLlmResponse
    stream: LlmResponseStream
    metadata: Metadata | None

    def __init__(
        self,
        llm_response: AggLlmResponse,
        *,
        metadata: Metadata | None = None,
    ) -> None: ...
    @classmethod
    def from_stream(
        cls,
        source: AsyncIterator[LlmResponseChunkValue],
        *,
        metadata: Metadata | None = None,
    ) -> Response: ...
    def to_dict(self) -> dict[str, Any]: ...

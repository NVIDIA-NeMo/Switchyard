# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Python bindings for the Rust-owned Switchyard protocol."""

from switchyard_rust.libsy_protocol import (
    AggLlmResponse,
    ContentBlock,
    Context,
    Decision,
    FileSource,
    FormatId,
    ImageSource,
    InstructionBlock,
    LlmRequest,
    LlmResponseChunk,
    LlmResponseStream,
    MediaSource,
    Message,
    Metadata,
    OutputParams,
    PreservationMetadata,
    ProviderExtensions,
    ReasoningParams,
    Request,
    Response,
    ResponseOutput,
    Role,
    RoutedLlmClient,
    SamplingParams,
    StopReason,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResult,
    Usage,
    WireFormat,
)

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
ImageSourceValue = ImageSource.Url | ImageSource.Base64 | ImageSource.Raw
FileSourceValue = FileSource.FileId | FileSource.FileData | FileSource.Raw
MediaSourceValue = MediaSource.Url | MediaSource.Base64 | MediaSource.Raw
LlmResponseChunkValue = (
    LlmResponseChunk.MessageStart
    | LlmResponseChunk.TextDelta
    | LlmResponseChunk.ReasoningDelta
    | LlmResponseChunk.ToolCallDelta
    | LlmResponseChunk.UsageUpdate
    | LlmResponseChunk.MessageStop
    | LlmResponseChunk.Error
)

__all__ = [
    "AggLlmResponse",
    "ContentBlock",
    "ContentBlockValue",
    "Context",
    "Decision",
    "FileSource",
    "FileSourceValue",
    "FormatId",
    "ImageSource",
    "ImageSourceValue",
    "InstructionBlock",
    "LlmRequest",
    "LlmResponseChunk",
    "LlmResponseChunkValue",
    "LlmResponseStream",
    "MediaSource",
    "MediaSourceValue",
    "Message",
    "Metadata",
    "OutputParams",
    "PreservationMetadata",
    "ProviderExtensions",
    "ReasoningParams",
    "Request",
    "Response",
    "ResponseOutput",
    "Role",
    "RoutedLlmClient",
    "SamplingParams",
    "StopReason",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ToolResult",
    "Usage",
    "WireFormat",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from switchyard_rust.core import (
    ChatRequestType,
    LLMBackend,
    ProxyContext,
)

class BackendFormat:
    AUTO: ClassVar[BackendFormat]
    OPENAI: ClassVar[BackendFormat]
    RESPONSES: ClassVar[BackendFormat]
    ANTHROPIC: ClassVar[BackendFormat]

    value: str

    def __init__(self, value: str = "auto") -> None: ...


class EndpointConfig:
    base_url: str | None
    api_key: str | None
    timeout_secs: float | None

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_secs: float | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class LlmTarget:
    id: str
    model: str
    format: BackendFormat
    backend_format: BackendFormat
    endpoint: EndpointConfig
    base_url: str | None
    api_key: str | None
    timeout: float | None
    extra_body: dict[str, Any] | None
    extra_headers: dict[str, str]

    def __init__(
        self,
        id: str | None = None,
        model: str | None = None,
        format: BackendFormat | str | None = None,
        backend_format: BackendFormat | str | None = None,
        endpoint: EndpointConfig | dict[str, Any] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_secs: float | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def model_dump(self) -> dict[str, Any]: ...


class RandomRoutingProcessorConfig:
    strong: LlmTarget
    weak: LlmTarget
    strong_probability: float
    rng_seed: int | None

    def __init__(
        self,
        strong: LlmTarget,
        weak: LlmTarget,
        strong_probability: float = 0.5,
        rng_seed: int | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class IntakeQueueFullPolicy:
    DROP: ClassVar[IntakeQueueFullPolicy]
    BLOCK: ClassVar[IntakeQueueFullPolicy]

    value: str

    def __init__(self, value: str = "drop") -> None: ...


class IntakeSinkConfig:
    intake_base_url: str | None
    workspace: str | None
    user_id: str
    api_key: str | None
    nvdataflow_project: str | None
    max_queue_size: int
    request_timeout_s: float
    max_retries: int
    on_queue_full: IntakeQueueFullPolicy
    capture_content: bool

    def __init__(
        self,
        intake_base_url: str | None = None,
        workspace: str | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
        nvdataflow_project: str | None = None,
        max_queue_size: int | None = None,
        request_timeout_s: float | None = None,
        max_retries: int | None = None,
        on_queue_full: IntakeQueueFullPolicy | str | None = None,
        capture_content: bool | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class IntakeRequestMetadata:
    enabled: bool | None
    app: str | None
    task: str | None

    def __init__(
        self,
        enabled: bool | None = None,
        app: str | None = None,
        task: str | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class RequestMetadata:
    session_id: str | None
    intake: IntakeRequestMetadata

    def __init__(
        self,
        session_id: str | None = None,
        intake: IntakeRequestMetadata | None = None,
    ) -> None: ...
    @classmethod
    def from_headers(cls, headers: Any) -> RequestMetadata: ...
    def to_dict(self) -> dict[str, Any]: ...
    def apply_to_context(self, ctx: Any) -> None: ...


def set_stats_route_label(ctx: Any, label: str) -> None: ...


class LlmTargetBackend:
    target: LlmTarget

    def __init__(self, target: LlmTarget, backend: LLMBackend) -> None: ...


class OpenAiNativeBackend(LLMBackend):
    target: LlmTarget

    def __init__(self, target: LlmTarget) -> None: ...


class OpenAiPassthroughBackend(LLMBackend):
    endpoint: EndpointConfig

    def __init__(
        self,
        endpoint: EndpointConfig | dict[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_secs: float | None = None,
        timeout: float | None = None,
    ) -> None: ...


class AnthropicNativeBackend(LLMBackend):
    target: LlmTarget

    def __init__(self, target: LlmTarget) -> None: ...


class MultiLlmBackend(LLMBackend):
    def __init__(
        self,
        targets: Iterable[LlmTargetBackend | tuple[LlmTarget, LLMBackend]],
        supported_request_types: Iterable[ChatRequestType | str] | None = None,
        default_target_id: str | None = None,
    ) -> None: ...
    def target_ids(self) -> list[str]: ...
    def default_target_id(self) -> str | None: ...


class IntakeRequestProcessor:
    def __init__(self) -> None: ...
    async def process(self, ctx: ProxyContext, request: Any) -> Any: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...


class IntakeResponseProcessor:
    config: IntakeSinkConfig

    def __init__(self, config: IntakeSinkConfig) -> None: ...
    async def process(self, ctx: ProxyContext, response: Any) -> Any: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...


class DimensionScore:
    name: str
    score: float
    signal: str | None


class ContextSignals:
    dimensions: list[DimensionScore]
    token_count_estimate: int


class ScoringConfig:
    def __init__(
        self,
        token_count_short: int = 50,
        token_count_long: int = 500,
        code_keywords: Iterable[str] = (),
        reasoning_keywords: Iterable[str] = (),
        simple_keywords: Iterable[str] = (),
        technical_keywords: Iterable[str] = (),
        creative_keywords: Iterable[str] = (),
        imperative_verbs: Iterable[str] = (),
        constraint_indicators: Iterable[str] = (),
        output_format_keywords: Iterable[str] = (),
        reference_keywords: Iterable[str] = (),
        negation_keywords: Iterable[str] = (),
        domain_specific_keywords: Iterable[str] = (),
    ) -> None: ...


class DimensionCollector:
    def __init__(
        self,
        config: ScoringConfig | None = None,
        *,
        recent_window: int | None = None,
    ) -> None: ...
    async def process(self, ctx: ProxyContext, request: Any) -> Any: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...


def get_context_signals(ctx: ProxyContext) -> ContextSignals | None: ...


class ToolResultSignal:
    severity: float
    patterns: tuple[str, ...]
    turn_depth: int
    write_count: int
    edit_count: int
    read_count: int
    todowrite_count: int
    recent_write_count: int
    recent_edit_count: int
    recent_read_count: int
    recent_todowrite_count: int
    pure_bash_streak: int
    no_error_streak: int
    tests_passed: bool
    prompt_char_count: int


def get_tool_result_signal(ctx: ProxyContext) -> ToolResultSignal | None: ...


class ResponseFlag:
    MALFORMED_TOOL_CALL_JSON: ClassVar[ResponseFlag]
    EMPTY_RESPONSE: ClassVar[ResponseFlag]
    TRUNCATED_COMPLETION: ClassVar[ResponseFlag]
    MISSING_REQUIRED_ARGS: ClassVar[ResponseFlag]

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class ResponseSignals:
    flags: list[ResponseFlag]

    def has_failures(self) -> bool: ...
    def contains(self, flag: ResponseFlag) -> bool: ...


class ResponseSignalCollector:
    def __init__(self) -> None: ...
    async def process(self, ctx: ProxyContext, response: Any) -> Any: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...


def get_response_signals(ctx: ProxyContext) -> ResponseSignals | None: ...


def extract_response_signals(body: dict[str, Any] | None) -> ResponseSignals: ...


__all__: list[str]

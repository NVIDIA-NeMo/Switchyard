# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from switchyard_rust.core import ChatRequest, ChatRequestType

class ProfileInput:
    request: ChatRequest
    metadata: ProfileRequestMetadata

    def __init__(
        self,
        request: ChatRequest,
        metadata: ProfileRequestMetadata | None = None,
    ) -> None: ...

class ProfileRequestMetadata:
    request_id: str | None
    inbound_format: ChatRequestType | None
    headers: dict[str, list[str]]

    def __init__(
        self,
        request_id: str | None = None,
        inbound_format: ChatRequestType | str | None = None,
        headers: Mapping[str, str | list[str]] | None = None,
    ) -> None: ...
    @classmethod
    def from_headers(
        cls,
        headers: Mapping[str, str | list[str]],
        inbound_format: ChatRequestType | str | None = None,
    ) -> ProfileRequestMetadata: ...
    def to_dict(self) -> dict[str, object]: ...

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-owned request values shared by profile implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from switchyard_rust.core import request_type_enum, request_type_value

if TYPE_CHECKING:
    from collections.abc import Mapping

    from switchyard_rust.core import ChatRequest, ChatRequestType


@dataclass(frozen=True, slots=True, init=False)
class ProfileRequestMetadata:
    """Request metadata available to Python profile runtimes."""

    request_id: str | None
    inbound_format: ChatRequestType | None
    headers: dict[str, list[str]]

    def __init__(
        self,
        request_id: str | None = None,
        inbound_format: ChatRequestType | str | None = None,
        headers: Mapping[str, str | list[str]] | None = None,
    ) -> None:
        if request_id is not None and not request_id.strip():
            raise ValueError("invalid request_id: RequestId must not be empty")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "inbound_format",
            request_type_enum(inbound_format) if inbound_format is not None else None,
        )
        object.__setattr__(self, "headers", _normalize_headers(headers))

    @classmethod
    def from_headers(
        cls,
        headers: Mapping[str, str | list[str]],
        inbound_format: ChatRequestType | str | None = None,
    ) -> ProfileRequestMetadata:
        """Build metadata from headers and infer the request ID when present."""
        normalized = _normalize_headers(headers)
        request_ids = normalized.get("x-request-id", [])
        return cls(
            request_id=request_ids[0] if request_ids else None,
            inbound_format=inbound_format,
            headers=normalized,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation."""
        return {
            "request_id": self.request_id,
            "inbound_format": (
                request_type_value(self.inbound_format)
                if self.inbound_format is not None
                else None
            ),
            "headers": {name: list(values) for name, values in self.headers.items()},
        }


@dataclass(frozen=True, slots=True, init=False)
class ProfileInput:
    """Provider request and metadata passed to a Python profile."""

    request: ChatRequest
    metadata: ProfileRequestMetadata

    def __init__(
        self,
        request: ChatRequest,
        metadata: ProfileRequestMetadata | None = None,
    ) -> None:
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "metadata", metadata or ProfileRequestMetadata())


def _normalize_headers(
    headers: Mapping[str, str | list[str]] | None,
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for name, value in (headers or {}).items():
        if not isinstance(name, str):
            raise TypeError("ProfileRequestMetadata header names must be strings")
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            values = list(value)
        else:
            raise TypeError(
                "ProfileRequestMetadata headers must map strings to strings or lists of strings"
            )
        normalized.setdefault(name.lower(), []).extend(values)
    return normalized


__all__ = ["ProfileInput", "ProfileRequestMetadata"]

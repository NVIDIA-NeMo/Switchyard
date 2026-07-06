# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from typing import Literal

from switchyard_rust.components import LlmTarget
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse

ProfileConfigFormat = Literal["yaml", "yml", "json", "toml"]

# --- Erased serving surface ---------------------------------------------------

class ProfileConfigDocument:
    def resolve(self) -> ProfileConfigPlan: ...

class ProfileConfigPlan:
    def profile_ids(self) -> list[str]: ...
    def target_ids(self) -> list[str]: ...
    def profile_type(self, profile_id: str) -> str | None: ...
    def target(self, target_id: str) -> LlmTarget | None: ...
    def build_profile(self, profile_id: str) -> Profile: ...
    def build_profiles(self) -> dict[str, Profile]: ...

class Profile:
    profile_id: str

    async def run(self, request: ChatRequest) -> ChatResponse: ...

def parse_profile_config_str(
    input: str,
    format: ProfileConfigFormat = "yaml",
) -> ProfileConfigDocument: ...
def parse_profile_config_path(path: str | PathLike[str]) -> ProfileConfigDocument: ...
def load_profile_config(path: str | PathLike[str]) -> ProfileConfigPlan: ...

# --- Metadata for direct typed profile calls ----------------------------------

class ProfileInput:
    request: ChatRequest
    metadata: ProfileRequestMetadata

    def __init__(
        self,
        request: ChatRequest,
        metadata: ProfileRequestMetadata | None = None,
    ) -> None: ...

class ProfileRequestMetadata:
    session_id: str | None
    request_id: str | None
    inbound_format: ChatRequestType | None
    headers: dict[str, list[str]]

    def __init__(
        self,
        request_id: str | None = None,
        inbound_format: ChatRequestType | str | None = None,
        headers: dict[str, str | list[str]] | None = None,
        session_id: str | None = None,
    ) -> None: ...
    @classmethod
    def from_headers(
        cls,
        headers: dict[str, str | list[str]],
        inbound_format: ChatRequestType | str | None = None,
    ) -> ProfileRequestMetadata: ...
    def to_dict(self) -> dict[str, object]: ...

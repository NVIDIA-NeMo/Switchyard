# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-backed stable per-conversation key.

Derived from the prefix an agent harness never rewrites — the system prompt
plus the first user message — so every turn of one conversation hashes alike.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from switchyard_rust.core import session_key_from_body as _native_session_key_from_body


@overload
def session_key_from_body(body: Any, depth: Literal[0] = 0) -> str: ...


@overload
def session_key_from_body(body: Any, depth: int) -> str | None: ...


def session_key_from_body(body: Any, depth: int = 0) -> str | None:
    """Derive the per-conversation key from a request body.

    ``depth == 0`` (default) hashes the stable anchors only and always returns
    a key. ``depth > 0`` extends the hashed prefix with the first ``depth``
    post-first-user messages — so repeated trials of an identical task diverge
    via early model responses — and returns ``None`` until the conversation is
    long enough for that prefix to exist.
    """
    return _native_session_key_from_body(body, depth)


__all__ = ["session_key_from_body"]

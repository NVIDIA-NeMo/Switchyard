# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixtures for executing the guide's Python snippets safely.

Points the guide's passthrough profile at a local in-process mock OpenAI
server (loopback, a fixed ``chat.completion``) instead of a live backend, and
stubs ``uvicorn.run`` to a no-op — the "host as HTTP server" snippet would
otherwise block the test session forever. Both are gated on the
``--markdown-docs`` flag so regular runs are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._mock_openai_server import (
    _MockOpenAIServer,
    markdown_docs_active,
    redirect_passthrough_to_local_mock,
)


@pytest.fixture
def local_mock_openai_server() -> Iterator[_MockOpenAIServer]:
    """A running local mock OpenAI upstream for hermetic serve tests."""
    with _MockOpenAIServer() as server:
        yield server


@pytest.fixture(autouse=True, scope="session")
def _markdown_docs_hermetic_upstream(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if not markdown_docs_active(request.config):
        yield
        return
    with redirect_passthrough_to_local_mock(stub_uvicorn=True):
        yield

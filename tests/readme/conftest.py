# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixture: run the README snippet against a local mock upstream.

Mirrors ``tests/getting_started/conftest.py``: the README's "Use as a Python
library" snippet builds a passthrough profile and calls ``switchyard.call``,
which would otherwise hit a real backend. The fixture points that passthrough
profile at a local in-process mock OpenAI server (loopback, a fixed
``chat.completion``) so it runs offline. Gated on the ``--markdown-docs`` flag
so regular runs are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._mock_openai_server import (
    markdown_docs_active,
    redirect_passthrough_to_local_mock,
)


@pytest.fixture(autouse=True, scope="session")
def _markdown_docs_hermetic_upstream(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if not markdown_docs_active(request.config):
        yield
        return
    with redirect_passthrough_to_local_mock():
        yield

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixtures for executing the guide's Python snippets safely.

Points the guide's passthrough profile at a local in-process mock OpenAI
server (loopback, canned ``chat.completion``) instead of a live backend, and
stubs ``uvicorn.run`` to a no-op — the "host as HTTP server" snippet would
otherwise block the test session forever. Both are gated on the
``--markdown-docs`` flag so regular runs are untouched.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from tests._mock_openai_server import _MockOpenAIServer


@pytest.fixture
def local_mock_openai_server() -> Iterator[_MockOpenAIServer]:
    """A running local mock OpenAI upstream for hermetic serve tests."""
    with _MockOpenAIServer() as server:
        yield server


def _markdown_docs_active(config: pytest.Config) -> bool:
    try:
        return bool(config.getoption("markdowndocs", default=False))
    except (KeyError, ValueError):
        return False


@pytest.fixture(autouse=True, scope="session")
def _markdown_docs_hermetic_upstream(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if not _markdown_docs_active(request.config):
        yield
        return

    import uvicorn

    from switchyard import PassthroughProfileConfig

    real_build = PassthroughProfileConfig.build
    monkeypatch = pytest.MonkeyPatch()
    with _MockOpenAIServer() as upstream:
        # Redirect the snippet's passthrough profile at the local mock upstream
        # so ``switchyard.call`` runs the real backend path fully offline, while
        # keeping any other fields the snippet set on its config.
        def _hermetic_build(_self: PassthroughProfileConfig) -> object:
            return real_build(
                dataclasses.replace(
                    _self, api_key="test-key", base_url=upstream.base_url
                )
            )

        monkeypatch.setattr(PassthroughProfileConfig, "build", _hermetic_build)
        # The "host as HTTP server" snippet ends in a blocking uvicorn.run(); stub
        # it so the snippet still exercises build_switchyard_app() without serving.
        monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
        try:
            yield
        finally:
            monkeypatch.undo()

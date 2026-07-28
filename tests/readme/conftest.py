# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixture: run the README snippet against a local mock upstream.

Mirrors ``tests/getting_started/conftest.py``: the README's "Use as a Python
library" snippet builds a passthrough profile and calls ``switchyard.call``,
which would otherwise hit a real backend. The fixture points that passthrough
profile at a local in-process mock OpenAI server (loopback, canned
``chat.completion``) so it runs offline. Gated on the ``--markdown-docs`` flag
so regular runs are untouched.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from tests._mock_openai_server import _MockOpenAIServer


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
        try:
            yield
        finally:
            monkeypatch.undo()

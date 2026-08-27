# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import litellm
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest.fixture(autouse=True)
async def manage_litellm_test_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> AsyncIterator[None]:
    """Use the mocked HTTP transport and drain LiteLLM callbacks between event loops."""
    if request.node.get_closest_marker("e2e") is None:
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    yield
    await GLOBAL_LOGGING_WORKER.flush()
    await GLOBAL_LOGGING_WORKER.stop()

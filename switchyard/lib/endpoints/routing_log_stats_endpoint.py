# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP access to session-scoped aggregates from the durable routing log."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from switchyard.lib.endpoints.base import Endpoint

if TYPE_CHECKING:
    from fastapi import FastAPI

    from switchyard.lib.processors.routing_log_response_processor import (
        RoutingLogResponseProcessor,
    )


class RoutingLogStatsEndpoint(Endpoint):
    """Expose one trial session's model and token aggregates."""

    register_once = True

    def __init__(self, processor: RoutingLogResponseProcessor) -> None:
        self._processor = processor

    def register(self, app: FastAPI) -> None:
        routes = APIRouter()
        processor = self._processor

        async def get_session_stats(session_id: str) -> dict[str, object]:
            snapshot = await asyncio.to_thread(processor.snapshot_session, session_id)
            if snapshot is None:
                raise HTTPException(status_code=404, detail="routing session not found")
            return snapshot

        routes.get("/v1/routing/session-stats")(get_session_stats)
        app.include_router(routes, tags=["Routing log"])


__all__ = ["RoutingLogStatsEndpoint"]

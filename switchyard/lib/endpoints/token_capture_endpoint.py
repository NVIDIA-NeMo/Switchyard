# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI endpoint exposing captured token-level completion records.

Serves the on-disk record store written by
:class:`~switchyard.lib.processors.token_capture_response_processor.TokenCaptureResponseProcessor`
via ``GET /v1/sessions/{session_id}/completions`` — one session's records in
capture order.

Reads go straight to disk (no in-memory registry), so retrieval works across
uvicorn workers and after a proxy restart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.processors.token_capture_response_processor import (
    SCHEMA_VERSION,
    session_dir_name,
    sessions_root,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


class TokenCaptureSessionsEndpoint(NemoSwitchyardEndpoint):
    """Exposes captured completion records for retrieval by session id.

    Contributed automatically by
    :meth:`TokenCaptureResponseProcessor.get_endpoint` — no manual wiring
    required.
    """

    register_once = True

    def __init__(self, capture_dir: Path | str) -> None:
        self._capture_dir = Path(capture_dir)

    def register(self, app: FastAPI) -> None:
        routes = APIRouter()
        capture_dir = self._capture_dir

        async def get_session_completions(session_id: str) -> dict[str, Any]:
            """All captured records for one session, in capture order."""
            dir_name = session_dir_name(session_id)
            # The sanitizer passes "." and ".." through; never treat them as
            # session directories.
            if dir_name in {".", ".."}:
                raise HTTPException(status_code=404, detail="unknown session")
            session_dir = sessions_root(capture_dir) / dir_name
            if not session_dir.is_dir():
                raise HTTPException(status_code=404, detail="unknown session")

            completions: list[dict[str, Any]] = []
            for record_path in sorted(session_dir.glob("*.json")):
                try:
                    completions.append(json.loads(record_path.read_text()))
                except (OSError, json.JSONDecodeError) as exc:
                    # A torn record must not hide the rest of the session.
                    logger.warning(
                        "token capture: skipping unreadable record %s: %s",
                        record_path,
                        exc,
                    )
            completions.sort(
                key=lambda record: (record.get("captured_at", ""), record.get("uuid", ""))
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "completions": completions,
            }

        routes.get("/v1/sessions/{session_id}/completions")(get_session_completions)
        app.include_router(routes, tags=["TokenCapture"])

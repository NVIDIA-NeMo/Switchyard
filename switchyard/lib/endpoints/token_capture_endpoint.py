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


def _session_head(session_dir: Path) -> dict[str, Any] | None:
    """``{session_id, parent_session_id}`` from the first readable record in *session_dir*.

    ``None`` when the directory holds no readable, session-tagged record.
    """
    for record_path in sorted(session_dir.glob("*.json")):
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        session_id = record.get("session_id")
        if isinstance(session_id, str) and session_id:
            return {
                "session_id": session_id,
                "parent_session_id": record.get("parent_session_id"),
            }
    return None


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
            # The directory name is a hash of the session id (path-safe by
            # construction, no traversal possible).
            session_dir = sessions_root(capture_dir) / session_dir_name(session_id)
            if not session_dir.is_dir():
                raise HTTPException(status_code=404, detail="unknown session")

            completions: list[dict[str, Any]] = []
            for record_path in sorted(session_dir.glob("*.json")):
                try:
                    record = json.loads(record_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    # A torn record must not hide the rest of the session.
                    logger.warning(
                        "token capture: skipping unreadable record %s: %s",
                        record_path,
                        exc,
                    )
                    continue
                # Defense in depth: only surface records that actually belong to
                # the requested session, never a neighbor sharing the directory.
                if record.get("session_id") == session_id:
                    completions.append(record)
            completions.sort(
                key=lambda record: (record.get("captured_at", ""), record.get("uuid", ""))
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "completions": completions,
            }

        async def list_sessions() -> dict[str, Any]:
            """Every session id captured under this log dir, with its parent link.

            Lets a caller enumerate a run's sessions when the harness mints its
            own ids (e.g. OpenCode's ``X-Session-Id``), without reading any
            harness logs. Directory names are session-id hashes, so the ids come
            from the records. Scope one run to one ``--rl-log-dir`` so the list is
            not mixed across rollouts.
            """
            root = sessions_root(capture_dir)
            sessions: list[dict[str, Any]] = []
            if root.is_dir():
                for session_dir in sorted(root.iterdir()):
                    entry = _session_head(session_dir) if session_dir.is_dir() else None
                    if entry is not None:
                        sessions.append(entry)
            return {"schema_version": SCHEMA_VERSION, "sessions": sessions}

        routes.get("/v1/sessions")(list_sessions)
        routes.get("/v1/sessions/{session_id}/completions")(get_session_completions)
        app.include_router(routes, tags=["TokenCapture"])

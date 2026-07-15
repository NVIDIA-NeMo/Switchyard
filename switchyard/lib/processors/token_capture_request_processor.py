# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-side processor that resolves the capture session and preps the upstream call."""

from __future__ import annotations

import logging

from switchyard.lib.proxy_context import CTX_CALLER_API_KEY, ProxyContext
from switchyard.lib.request_metadata import CTX_REQUEST_METADATA
from switchyard_rust.core import ChatRequest

logger = logging.getLogger(__name__)

#: Context metadata key holding the resolved capture session id. Written by
#: :class:`TokenCaptureRequestProcessor`; read by
#: :class:`~switchyard.lib.processors.token_capture_response_processor.TokenCaptureResponseProcessor`
#: — a call without it is forwarded uncaptured.
CTX_TOKEN_CAPTURE_SESSION = "_token_capture_session"

#: Context metadata key set when the inbound request asked for streaming and
#: the request was flipped to non-streaming for faithful token capture. Read
#: by the response processor, which synthesizes the client-facing stream.
CTX_TOKEN_CAPTURE_ORIGINAL_STREAM = "_token_capture_original_stream"

#: Header-less harnesses (OpenClaw) ride the session id on their API key. The
#: launcher prefixes it with this marker so the server can recognize a session
#: id explicitly, rather than guessing from the string shape — a real client
#: credential (no marker) is therefore never mistaken for a session id and
#: leaked into record files / directory names. Shared with the launcher, which
#: applies the prefix (see ``launch_intake_config``). The colon makes an
#: accidental collision with a real API key effectively impossible.
SESSION_API_KEY_MARKER = "sycap-session:"

#: Top-level request-body key carrying the capture session for harness clients
#: with no custom-header or API-key surface (e.g. clients whose only reachable
#: knob is extra JSON body fields). Always stripped before the request is
#: forwarded upstream, captured or not.
BODY_SESSION_KEY = "proxy_x_session_id"

#: Caller-supplied sampling params stripped so the target's derived
#: token-capture ``extra_body`` params (see ``llm_target_with_token_capture``)
#: always win.
_CALLER_PARAM_KEYS = ("logprobs", "top_logprobs", "return_token_ids")


class TokenCaptureRequestProcessor:
    """Resolve the capture session and force the upstream call non-streaming.

    Runs after :class:`~switchyard.lib.processors.rl_logging_request_processor.RlLoggingRequestProcessor`,
    whose translated snapshot supplies the record's ``messages``. Calls with no
    resolvable session are forwarded untouched and go uncaptured.

    For captured calls, caller-supplied ``logprobs`` / ``top_logprobs`` /
    ``return_token_ids`` are stripped so the target's derived params win, and
    streaming requests are flipped to ``stream: false`` upstream — vLLM only
    returns token IDs on buffered completions. The original intent is recorded
    so the response processor can synthesize an equivalent stream for the
    client (harnesses like Claude Code cannot disable streaming).
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        """Resolve the session; strip caller params and flip ``stream`` off."""
        body = dict(request.body)
        mutated = BODY_SESSION_KEY in body
        body_session = body.pop(BODY_SESSION_KEY, None)

        session_id = _resolve_session_id(ctx, body_session)
        if session_id is None:
            logger.debug("token capture: no session id on request; forwarding uncaptured")
            if mutated:
                request.replace_body(body)
            return request
        ctx.metadata[CTX_TOKEN_CAPTURE_SESSION] = session_id

        for key in _CALLER_PARAM_KEYS:
            if key in body:
                del body[key]
                mutated = True
        if body.get("stream"):
            ctx.metadata[CTX_TOKEN_CAPTURE_ORIGINAL_STREAM] = True
            body["stream"] = False
            # stream_options is only valid alongside stream=true.
            body.pop("stream_options", None)
            mutated = True
        if mutated:
            request.replace_body(body)
        return request


def _resolve_session_id(ctx: ProxyContext, body_session: object) -> str | None:
    metadata = ctx.metadata.get(CTX_REQUEST_METADATA)
    session_id = getattr(metadata, "session_id", None)
    if isinstance(session_id, str) and session_id:
        return session_id
    # Clients whose only reachable knob is extra JSON body fields ride the
    # session id on a top-level body key (stripped by the caller above).
    if isinstance(body_session, str) and body_session:
        return body_session
    # Harnesses with no custom-header surface (OpenClaw) ride the session id on
    # their API key, marker-prefixed by the launcher. Strip the marker to get
    # the session id; a caller key without the marker is a real credential and
    # is never treated as a session.
    caller_key = ctx.metadata.get(CTX_CALLER_API_KEY)
    if isinstance(caller_key, str) and caller_key.startswith(SESSION_API_KEY_MARKER):
        session = caller_key[len(SESSION_API_KEY_MARKER):]
        return session or None
    return None

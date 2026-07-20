# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional tier-transition handoff notes.

When the stage_router flips a session between the efficient (weak) and capable
(strong) tiers, this injector appends a short, ephemeral guidance note to the
outgoing request so the model taking over knows *why* it was handed the task.

**Ephemeral by construction.** The note is appended to the request the proxy
forwards for this turn only; it is never written back into the agent's
conversation store. On the next turn the agent re-sends its own history (which
never contained the note) and the injector decides afresh. So notes never
accumulate across turns — a task that escalates, de-escalates, then escalates
again gets at most one note per *transition*, each living in a single request.

**Transition-gated.** A note is injected only when the picked tier *differs*
from the tier this session used on its previous turn (tracked per session in
:attr:`_last_tier`). Staying on a tier — however many turns — injects nothing,
which keeps the note out of the prompt-cache prefix on the steady-state turns.

**Truthful escalation gate.** With ``only_on_wrong_signal_escalation`` (default),
the escalation note fires only when the escalation was driven by a real signal
(``override`` — critical severity or compaction — or ``dimensions`` — the scorer
crossing the threshold on WRONG signals), not by an ambiguous default
(``fall_open`` / ``llm-classifier``). This keeps "the previous model was
stalling" from being asserted when no such signal was seen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from switchyard.lib.processors.stage_router.picker import CAPABLE, EFFICIENT

if TYPE_CHECKING:
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

#: Default weak→strong note. Kept to two sentences: state the reason, then the
#: corrective (don't blindly repeat the weak model's last approach).
DEFAULT_ESCALATION_NOTE: str = (
    "[router-guidance] A weaker model was handling this task and showed signs of "
    "stalling, looping, or repeated errors on the preceding steps, so control was "
    "escalated to you, a stronger model. Re-examine the current state directly and "
    "do not simply repeat the previous approach."
)

#: Default strong→weak note (only used when a deescalation note is configured;
#: the direction is off unless the operator sets one).
DEFAULT_DEESCALATION_NOTE: str = (
    "[router-guidance] A stronger model just completed the hard reasoning for this "
    "task; the remaining work is expected to be mechanical. Execute the established "
    "plan and avoid re-architecting what is already in place."
)

#: Decision sources that represent a *real* signal-driven escalation (vs. an
#: ambiguous default). Used by the escalation gate.
_WRONG_ESCALATION_SOURCES: frozenset[str] = frozenset({"override", "dimensions"})

#: Cap on the per-session last-tier map so a long-lived proxy process doesn't
#: grow it without bound. When exceeded the whole map is cleared (cheap, and the
#: only cost is a possible missed note on the first turn of in-flight sessions).
_MAX_SESSIONS: int = 8192


class HandoffNoteInjector:
    """Appends a tier-transition guidance note to the outgoing request.

    One instance per stage_router processor; safe under the single-threaded
    asyncio loop (the per-session map is only touched between awaits).
    """

    def __init__(
        self,
        *,
        escalation_note: str = DEFAULT_ESCALATION_NOTE,
        deescalation_note: str | None = None,
        only_on_wrong_signal_escalation: bool = True,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        self._escalation_note = escalation_note
        self._deescalation_note = deescalation_note
        self._only_on_wrong_signal = only_on_wrong_signal_escalation
        self._max_sessions = max_sessions
        self._last_tier: dict[str, int] = {}

    def maybe_inject(
        self,
        request: ChatRequest,
        *,
        tier: int,
        source: str | None,
    ) -> bool:
        """Inject a note if this turn crosses a tier boundary for its session.

        Returns ``True`` iff a note was appended to ``request``. Never raises —
        any failure degrades to "no note" so it can't block routing.
        """
        try:
            key = _session_key(request)
            if key is None:
                return False
            prev = self._last_tier.get(key)
            self._remember(key, tier)
            if prev is None or prev == tier:
                return False  # first turn of the session, or no transition

            note = self._note_for_transition(prev, tier, source)
            if not note:
                return False
            return _append_note(request, note)
        except Exception:
            log.debug("handoff-note injection failed; continuing without note", exc_info=True)
            return False

    def _note_for_transition(self, prev: int, tier: int, source: str | None) -> str | None:
        if prev == EFFICIENT and tier == CAPABLE:
            if self._only_on_wrong_signal and source not in _WRONG_ESCALATION_SOURCES:
                return None
            return self._escalation_note
        if prev == CAPABLE and tier == EFFICIENT:
            return self._deescalation_note
        return None

    def _remember(self, key: str, tier: int) -> None:
        if len(self._last_tier) >= self._max_sessions and key not in self._last_tier:
            self._last_tier.clear()
        self._last_tier[key] = tier


def _session_key(request: ChatRequest) -> str | None:
    """Stable per-session fingerprint from the conversation's fixed prefix.

    Claude Code (and agentic callers generally) keep the system prompt and the
    first user turn — the task instruction — constant for the life of a session
    while appending turns, so a hash of that prefix identifies the session
    without relying on an optional session-id header.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return None
    parts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.append(_text_of(system))
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            parts.append(_text_of(first.get("content")))
    if not any(parts):
        return None
    return str(hash("\x1f".join(parts)))


def _text_of(content: object) -> str:
    """Flatten message/system content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out)
    return ""


def _append_note(request: ChatRequest, note: str) -> bool:
    """Append ``note`` to the request's final user turn.

    Anthropic: add a ``text`` block after the trailing user turn's content
    (``tool_result`` blocks must stay first, so the note goes last). OpenAI Chat:
    append a fresh trailing ``user`` message (the trailing turn is often a
    ``tool`` result, and OpenAI permits a following user message). Both preserve
    the cached prefix — only the suffix of the request changes.
    """
    body = getattr(request, "body", None)
    if not isinstance(body, dict):
        return False
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    fmt = getattr(getattr(request, "request_type", None), "value", None)
    block = {"type": "text", "text": note}

    if fmt == "anthropic":
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "user":
            return False
        content = last.get("content")
        if isinstance(content, list):
            content.append(block)
        elif isinstance(content, str):
            last["content"] = [{"type": "text", "text": content}, block]
        else:
            return False
    else:
        # OpenAI Chat / Responses: a trailing user message is the portable append.
        messages.append({"role": "user", "content": note})

    request.replace_body(body)
    return True


__all__ = [
    "DEFAULT_DEESCALATION_NOTE",
    "DEFAULT_ESCALATION_NOTE",
    "HandoffNoteInjector",
]

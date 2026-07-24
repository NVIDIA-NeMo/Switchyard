# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional signal-driven handoff notes for the capable (strong) tier.

When the stage_router picks the capable tier because a real signal fired, this
injector appends a short, ephemeral guidance note to the outgoing request so the
model knows *why* it was handed the task.

**Stateless.** No per-session state is tracked. The note fires on every turn
where the capable tier is picked *and* the decision source was a real signal
(``dimensions`` or ``override``). This avoids the stale-state and hash-collision
problems that came with tracking previous tier per session across a shared proxy.

**Ephemeral by construction.** The note is appended to the request the proxy
forwards for this turn only; it is never written back into the agent's
conversation store. On the next turn the agent re-sends its own history (which
never contained the note) and the injector decides afresh.

**Truthful escalation gate.** With ``only_on_wrong_signal_escalation`` (default),
the note fires only when the escalation was driven by a real signal
(``override`` — critical severity or compaction — or ``dimensions`` — the scorer
crossing threshold on wrong signals), not by an ambiguous default
(``fall_open`` / ``llm-classifier``). This keeps "the previous model was
stalling" from being asserted when no such signal was seen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from switchyard.lib.processors.stage_router.picker import CAPABLE

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

class HandoffNoteInjector:
    """Appends a guidance note to capable-tier requests driven by real signals.

    Stateless — no per-session tracking. Fires on every turn where the capable
    tier is picked and the decision source was a real wrong signal.
    """

    def __init__(
        self,
        *,
        escalation_note: str = DEFAULT_ESCALATION_NOTE,
        only_on_wrong_signal_escalation: bool = True,
    ) -> None:
        self._escalation_note = escalation_note
        self._only_on_wrong_signal = only_on_wrong_signal_escalation

    def maybe_inject(
        self,
        request: ChatRequest,
        *,
        tier: int,
        source: str | None,
    ) -> bool:
        """Inject a note when the capable tier is picked due to a real signal.

        Returns ``True`` iff a note was appended to ``request``. Never raises —
        any failure degrades to "no note" so it can't block routing.
        """
        try:
            if tier != CAPABLE:
                return False
            if self._only_on_wrong_signal and source not in _WRONG_ESCALATION_SOURCES:
                return False
            return _append_note(request, self._escalation_note)
        except Exception:
            log.debug("handoff-note injection failed; continuing without note", exc_info=True)
            return False



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

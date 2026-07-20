# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional tier-transition handoff-note injector."""

from __future__ import annotations

from switchyard.lib.processors.stage_router import CAPABLE, EFFICIENT
from switchyard.lib.processors.stage_router.handoff_notes import (
    DEFAULT_DEESCALATION_NOTE,
    DEFAULT_ESCALATION_NOTE,
    HandoffNoteInjector,
)
from switchyard_rust.core import ChatRequest


def _anthropic(task: str = "solve the task", turn: str = "next") -> ChatRequest:
    """Anthropic request whose trailing user turn carries a tool_result block."""
    return ChatRequest.anthropic({
        "system": "you are a coding agent",
        "messages": [
            {"role": "user", "content": task},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": turn}]},
        ],
    })


def _last_user_text(request: ChatRequest) -> str:
    content = request.body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def test_first_turn_never_injects():
    inj = HandoffNoteInjector()
    req = _anthropic()
    # No prior tier recorded → no transition → no note.
    assert inj.maybe_inject(req, tier=CAPABLE, source="dimensions") is False
    assert DEFAULT_ESCALATION_NOTE not in _last_user_text(req)


def test_escalation_transition_injects_note_as_trailing_block():
    inj = HandoffNoteInjector()
    # Establish the session on the weak tier first.
    inj.maybe_inject(_anthropic(turn="a"), tier=EFFICIENT, source="fall_open")
    # Now escalate on a real signal → note injected.
    req = _anthropic(turn="b")
    assert inj.maybe_inject(req, tier=CAPABLE, source="dimensions") is True
    content = req.body["messages"][-1]["content"]
    # tool_result stays first; the note is appended after it as a text block.
    assert content[0]["type"] == "tool_result"
    assert content[-1] == {"type": "text", "text": DEFAULT_ESCALATION_NOTE}


def test_staying_on_a_tier_injects_nothing():
    inj = HandoffNoteInjector()
    inj.maybe_inject(_anthropic(turn="a"), tier=CAPABLE, source="dimensions")
    req = _anthropic(turn="b")
    # Same tier again → no transition → no note (keeps the cache prefix clean).
    assert inj.maybe_inject(req, tier=CAPABLE, source="dimensions") is False
    assert DEFAULT_ESCALATION_NOTE not in _last_user_text(req)


def test_escalation_from_ambiguous_source_gated_off_by_default():
    inj = HandoffNoteInjector()
    inj.maybe_inject(_anthropic(turn="a"), tier=EFFICIENT, source="fall_open")
    req = _anthropic(turn="b")
    # capable_first can escalate on fall_open (ambiguous default) — the truthful
    # gate suppresses the "prior model stalled" note there.
    assert inj.maybe_inject(req, tier=CAPABLE, source="fall_open") is False


def test_wrong_signal_gate_can_be_disabled():
    inj = HandoffNoteInjector(only_on_wrong_signal_escalation=False)
    inj.maybe_inject(_anthropic(turn="a"), tier=EFFICIENT, source="fall_open")
    req = _anthropic(turn="b")
    assert inj.maybe_inject(req, tier=CAPABLE, source="fall_open") is True


def test_override_source_counts_as_wrong_signal():
    """Compaction / critical-severity escalations stamp ``override`` → note fires."""
    inj = HandoffNoteInjector()
    inj.maybe_inject(_anthropic(turn="a"), tier=EFFICIENT, source="fall_open")
    req = _anthropic(turn="b")
    assert inj.maybe_inject(req, tier=CAPABLE, source="override") is True


def test_deescalation_off_by_default():
    inj = HandoffNoteInjector()
    inj.maybe_inject(_anthropic(turn="a"), tier=CAPABLE, source="dimensions")
    req = _anthropic(turn="b")
    # No de-escalation note configured → hand-back injects nothing.
    assert inj.maybe_inject(req, tier=EFFICIENT, source="dimensions") is False


def test_deescalation_note_injected_when_configured():
    inj = HandoffNoteInjector(deescalation_note=DEFAULT_DEESCALATION_NOTE)
    inj.maybe_inject(_anthropic(turn="a"), tier=CAPABLE, source="dimensions")
    req = _anthropic(turn="b")
    assert inj.maybe_inject(req, tier=EFFICIENT, source="dimensions") is True
    assert DEFAULT_DEESCALATION_NOTE in _last_user_text(req)


def test_notes_do_not_accumulate_across_turns():
    """Each injected request holds at most one note — nothing carries over."""
    inj = HandoffNoteInjector(deescalation_note=DEFAULT_DEESCALATION_NOTE)
    inj.maybe_inject(_anthropic(turn="a"), tier=EFFICIENT, source="fall_open")
    up = _anthropic(turn="b")
    inj.maybe_inject(up, tier=CAPABLE, source="dimensions")  # escalate
    down = _anthropic(turn="c")
    inj.maybe_inject(down, tier=EFFICIENT, source="dimensions")  # de-escalate
    # The de-escalation request carries only the de-escalation note, not the
    # escalation one (notes live in a single request, never persisted).
    text = _last_user_text(down)
    assert DEFAULT_DEESCALATION_NOTE in text
    assert DEFAULT_ESCALATION_NOTE not in text


def test_openai_chat_appends_trailing_user_message():
    inj = HandoffNoteInjector()
    first = ChatRequest.openai_chat({"messages": [{"role": "user", "content": "task"}]})
    inj.maybe_inject(first, tier=EFFICIENT, source="fall_open")
    req = ChatRequest.openai_chat({
        "messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "Bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "err"},
        ],
    })
    assert inj.maybe_inject(req, tier=CAPABLE, source="dimensions") is True
    last = req.body["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == DEFAULT_ESCALATION_NOTE
